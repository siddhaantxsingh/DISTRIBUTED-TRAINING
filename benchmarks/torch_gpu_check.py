"""GPU checks for the owner (NOT run here: this machine has no GPU). Launch with torchrun.

    torchrun --nproc_per_node 1 benchmarks/torch_gpu_check.py --out gpu_1.json      # RTX 4060
    torchrun --nproc_per_node 4 benchmarks/torch_gpu_check.py --out gpu_4.json      # rented multi-GPU box

What it does, on every rank (one GPU per rank, NCCL backend):

1. NCCL sanity: all_reduce of a rank-dependent tensor; checks the exact sum.
2. Correctness: the same GPT as dtrain (`dtrain.torch_ref.TorchGPT`) trained
   with CUDA DDP in float64 for --steps steps. Rank 0 trains dtrain's NumPy
   GPT on CPU in one process over the same *global* batch (one micro-batch
   per virtual rank, averaged), which is mathematically identical to W-rank
   DDP. Reports the max abs parameter difference per step. Expect about
   1e-15 .. 1e-13 (float64 reassociation, as on CPU with gloo), not 0.
3. NCCL all-reduce latency / bus bandwidth, float32, 1 KiB .. 256 MiB.
4. DDP throughput: float32 GPT (dim 512, 8 layers, ctx 256), tokens/s per rank
   count = this job's world size. Run with 1, 2, 4, ... GPUs to get scaling.

Rank 0 writes everything to --out as JSON together with GPU name, driver,
CUDA, NCCL and torch versions.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel  # noqa: E402

from dtrain.data import CharData, load_text, rank_batch  # noqa: E402
from dtrain.nn import GPT, GPTConfig  # noqa: E402
from dtrain.optim import AdamW  # noqa: E402
from dtrain.torch_ref import TorchGPT  # noqa: E402

ADAM = dict(lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--bench-steps", type=int, default=40)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible to PyTorch")
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl")
    res = {"world": world, "torch": torch.__version__, "cuda": torch.version.cuda,
           "nccl": ".".join(map(str, torch.cuda.nccl.version())), "gpu": torch.cuda.get_device_name(dev)}

    # 1. NCCL sanity
    t = torch.full((1 << 20,), float(rank + 1), device=dev)
    dist.all_reduce(t)
    torch.cuda.synchronize()
    res["nccl_sanity_ok"] = bool(torch.all(t == world * (world + 1) / 2).item())

    # 2. CUDA DDP float64 vs dtrain NumPy single-process over the same global batch
    data = CharData(load_text(None))
    cfg = GPTConfig(vocab=data.vocab, ctx=32, dim=64, heads=4, layers=2, dtype="float64")
    ref = GPT(cfg, seed=0)
    tm = TorchGPT.from_dtrain(ref).to(dev)
    ddp = DistributedDataParallel(tm, device_ids=[local])
    topt = torch.optim.AdamW(tm.parameters(), **ADAM)
    rp = ref.parameters()
    ropt = AdamW([p.data for p in rp], **ADAM)
    diffs = []
    for step in range(a.steps):
        x, y = rank_batch(data.train, step, 0, 4, cfg.ctx, 0, rank, world)
        topt.zero_grad(set_to_none=True)
        _, loss = ddp(torch.as_tensor(x).to(dev), torch.as_tensor(y).to(dev))
        loss.backward()
        topt.step()
        if rank == 0:
            ref.zero_grad()
            for r in range(world):
                xr, yr = rank_batch(data.train, step, 0, 4, cfg.ctx, 0, r, world)
                _, l = ref(xr, yr)
                (l * (1.0 / world)).backward()
            ropt.step([p.grad for p in rp])
            pt = np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in tm.ps])
            pd = np.concatenate([p.data.reshape(-1) for p in rp])
            diffs.append(float(np.max(np.abs(pt - pd))))
    res["ddp_vs_dtrain_param_maxabs_diff_per_step"] = diffs

    # 3. NCCL all-reduce bandwidth
    res["allreduce"] = []
    for k in range(10, 29, 2):
        n = 1 << k
        x = torch.zeros(n // 4, device=dev)
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        iters = max(10, min(200, (256 << 20) // n))
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        s = (time.perf_counter() - t0) / iters
        res["allreduce"].append({"bytes": n, "latency_us": s * 1e6,
                                 "bus_GBps": 2 * (world - 1) / world * n / s / 1e9 if world > 1 else None})

    # 4. DDP throughput, float32
    bcfg = GPTConfig(vocab=data.vocab, ctx=256, dim=512, heads=8, layers=8, dtype="float32")
    bm = TorchGPT.from_dtrain(GPT(bcfg, seed=0)).to(dev)
    bddp = DistributedDataParallel(bm, device_ids=[local])
    bopt = torch.optim.AdamW(bm.parameters(), **ADAM)
    bs, times = 16, []
    for step in range(a.bench_steps):
        x, y = rank_batch(data.train, step, 0, bs, bcfg.ctx, 0, rank, world)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        bopt.zero_grad(set_to_none=True)
        _, loss = bddp(torch.as_tensor(x).to(dev), torch.as_tensor(y).to(dev))
        loss.backward()
        bopt.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    med = float(np.median(times[5:]))
    res["ddp_throughput"] = {"median_step_s": med, "tokens_per_s": bs * bcfg.ctx * world / med,
                             "params": int(sum(p.numel() for p in bm.parameters()))}
    if rank == 0:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps({k: v for k, v in res.items() if k != "allreduce"}, indent=1))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
