"""Cross-check dtrain DDP / ZeRO-1 / checkpoint-resume against torch.distributed (gloo, CPU).

Run one copy per rank through the dtrain launcher. Every rank holds both
implementations side by side: a dtrain GPT (DDP or ZeroAdamW over dtrain's
TCP collectives) and an identical `TorchGPT` (torch DDP, or torch DDP plus
`ZeroRedundancyOptimizer`, over a gloo process group). Both start from the
same weights, see the same per-rank batches (`dtrain.data.rank_batch`) and
use AdamW with the same hyper-parameters. After every step, rank 0 records
the largest absolute difference in averaged gradients (DDP mode) and in
parameters.

    python -m dtrain.launch --nproc 4 -- python benchmarks/torch_crosscheck.py \
        --mode ddp --steps 20 --dtype float64 --out out/ddp_w4_f64

Checkpoint/resume (`--phase first` then `--phase resume`, with the same --ckpt)
saves both implementations after --k steps, exits, and continues in new
processes; compare the final parameters with a `--phase full` run.

`--only dtrain|torch` runs one implementation (used for throughput); then no
differences are recorded, only step times.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.distributed.optim import ZeroRedundancyOptimizer  # noqa: E402
from torch.nn.parallel import DistributedDataParallel  # noqa: E402

from dtrain import checkpoint  # noqa: E402
from dtrain.comm import init_from_env  # noqa: E402
from dtrain.data import CharData, load_text, rank_batch  # noqa: E402
from dtrain.ddp import DDP  # noqa: E402
from dtrain.nn import GPT, GPTConfig  # noqa: E402
from dtrain.optim import AdamW  # noqa: E402
from dtrain.torch_ref import TorchGPT  # noqa: E402
from dtrain.zero import ZeroAdamW  # noqa: E402


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ddp", "zero"], default="ddp")
    ap.add_argument("--phase", choices=["full", "first", "resume"], default="full")
    ap.add_argument("--only", choices=["both", "dtrain", "torch"], default="both")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--k", type=int, default=10, help="checkpoint step for --phase first/resume")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=32)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--opt", choices=["adamw", "sgd"], default="adamw",
                    help="sgd (DDP mode only): plain SGD, so parameter differences are lr x gradient differences")
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-bucket-mb", type=float, default=25.0, help="torch DDP bucket_cap_mb (25 = torch default)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--torch-init", required=True, help="path for gloo's FileStore (must not exist yet)")
    ap.add_argument("--out", required=True, help="output prefix (rank 0 writes <out>.jsonl and <out>.npz)")
    return ap.parse_args(argv)


ADAM = dict(betas=(0.9, 0.95), eps=1e-8)


class _SGD:
    """Plain SGD on NumPy arrays, matching torch.optim.SGD with default arguments."""

    def __init__(self, params, lr):
        self.params, self.lr = params, lr

    def step(self, grads):
        for p, g in zip(self.params, grads):
            p -= self.lr * g

    def state_bytes(self):
        return 0


def main(argv=None):
    a = parse(argv)
    torch.set_num_threads(1)
    pg = init_from_env()
    rank, world = pg.rank, pg.world
    dist.init_process_group("gloo", init_method=f"file://{a.torch_init}", rank=rank, world_size=world)

    data = CharData(load_text(a.data))
    cfg = GPTConfig(vocab=data.vocab, ctx=a.ctx, dim=a.dim, heads=a.heads, layers=a.layers, dtype=a.dtype)
    use_d, use_t = a.only in ("both", "dtrain"), a.only in ("both", "torch")

    model = GPT(cfg, seed=a.seed)
    params = model.parameters()
    tmodel = TorchGPT.from_dtrain(model)  # identical initial weights
    start = 0
    dckpt = os.path.join(a.ckpt, "dtrain") if a.ckpt else None
    tckpt = os.path.join(a.ckpt, "torch.pt") if a.ckpt else None

    if a.phase == "resume":
        step, st = checkpoint.load(checkpoint.latest(dckpt), rank, world)
        start = step
        flat = st["params"]
        off = 0
        for p in params:
            p.data[...] = flat[off:off + p.data.size].reshape(p.shape)
            off += p.data.size
        tsd = torch.load(tckpt, weights_only=False)
        tmodel.load_state_dict(tsd["model"])
        assert tsd["step"] == step

    # ---- dtrain side
    ddp = zero = opt = None
    if use_d:
        if a.mode == "ddp":
            ddp = DDP(model, pg, bucket_mb=1.0, overlap=True)
            opt = AdamW([p.data for p in params], lr=a.lr, weight_decay=a.wd, **ADAM) if a.opt == "adamw" \
                else _SGD([p.data for p in params], a.lr)
        else:
            zero = ZeroAdamW(model, pg, max_grad_norm=None, lr=a.lr, weight_decay=a.wd, **ADAM)
        if a.phase == "resume":
            if a.mode == "ddp":
                n = len(params)
                opt.load_state_dict({"t": st["t"], "m": [st[f"m{i}"] for i in range(n)],
                                     "v": [st[f"v{i}"] for i in range(n)]})
            else:
                zero.load_state_dict({"t": st["t"], "m": [st["m"]], "v": [st["v"]]})

    # ---- torch side
    tddp = topt = None
    if use_t:
        tddp = DistributedDataParallel(tmodel, bucket_cap_mb=a.torch_bucket_mb)
        if a.mode == "ddp":
            topt = torch.optim.AdamW(tmodel.parameters(), lr=a.lr, weight_decay=a.wd, **ADAM) if a.opt == "adamw" \
                else torch.optim.SGD(tmodel.parameters(), lr=a.lr)
        else:
            topt = ZeroRedundancyOptimizer(tmodel.parameters(), optimizer_class=torch.optim.AdamW,
                                           lr=a.lr, weight_decay=a.wd, **ADAM)
        if a.phase == "resume":
            topt.load_state_dict(tsd["opt"])

    end = a.k if a.phase == "first" else a.steps
    log = open(a.out + ".jsonl", "w") if rank == 0 else None
    for step in range(start, end):
        x, y = rank_batch(data.train, step, 0, a.batch, a.ctx, a.seed, rank, world)
        rec = {"step": step + 1}
        if use_d:
            t0 = time.perf_counter()
            model.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            if ddp is not None:
                ddp.finish()
                gd = np.concatenate([p.grad.reshape(-1) for p in params])
                opt.step([p.grad for p in params])
            else:
                zero.step()
            l = np.array([float(loss.data)])
            pg.allreduce(l)
            rec["dtrain_s"] = time.perf_counter() - t0
            rec["dtrain_loss"] = float(l[0]) / world
        if use_t:
            t0 = time.perf_counter()
            topt.zero_grad(set_to_none=True)
            _, tloss = tddp(x, y)
            tloss.backward()
            if use_d and ddp is not None:
                gt = tmodel.flat_grads()
            topt.step()
            tl = tloss.detach().reshape(1).clone()
            dist.all_reduce(tl)
            rec["torch_s"] = time.perf_counter() - t0
            rec["torch_loss"] = float(tl[0]) / world
        if use_d and use_t:
            pd = np.concatenate([p.data.reshape(-1) for p in params])
            pt = tmodel.flat_params()
            rec["param_maxabs_diff"] = float(np.max(np.abs(pd - pt)))
            rec["param_maxabs"] = float(np.max(np.abs(pt)))
            rec["param_bit_identical_frac"] = float(np.mean(pd == pt))
            if ddp is not None:
                rec["grad_maxabs_diff"] = float(np.max(np.abs(gd - gt)))
                rec["grad_maxabs"] = float(np.max(np.abs(gt)))
                rec["grad_bit_identical_frac"] = float(np.mean(gd == gt))
            rec["loss_abs_diff"] = abs(rec["dtrain_loss"] - rec["torch_loss"])
        if log:
            log.write(json.dumps(rec) + "\n")
            log.flush()

    if use_d and use_t:  # replicas must agree across ranks, in both implementations
        pd = np.concatenate([p.data.reshape(-1) for p in params])
        pt = tmodel.flat_params()
        for arr, name in ((pd, "dtrain"), (pt, "torch")):
            r0 = arr.copy()
            pg.broadcast(r0, root=0)
            if not np.array_equal(r0, arr):
                raise SystemExit(f"rank {rank}: {name} replica differs from rank 0")

    # optimizer-state bytes per rank, for the ZeRO comparison
    sb = {}
    if zero is not None:
        sb["dtrain_state_bytes"] = zero.state_bytes()
    elif opt is not None:
        sb["dtrain_state_bytes"] = opt.state_bytes()
    if topt is not None:
        inner = topt.optim if isinstance(topt, ZeroRedundancyOptimizer) else topt
        sb["torch_state_bytes"] = sum(v.numel() * v.element_size() for s in inner.state.values()
                                      for k, v in s.items() if torch.is_tensor(v) and k != "step")
    allsb = [None] * world
    dist.all_gather_object(allsb, sb)

    if a.phase == "first":
        if use_d:
            if zero is not None:
                sd = zero.state_dict()
                st = {"params": zero.flat, "t": sd["t"], "m": sd["m"][0], "v": sd["v"][0]}
            else:
                sd = opt.state_dict()
                st = {"params": np.concatenate([p.data.reshape(-1) for p in params]), "t": sd["t"]}
                st.update({f"m{i}": m for i, m in enumerate(sd["m"])})
                st.update({f"v{i}": v for i, v in enumerate(sd["v"])})
            checkpoint.save(dckpt, end, rank, world, st, pg.barrier)
        if use_t:
            if isinstance(topt, ZeroRedundancyOptimizer):
                topt.consolidate_state_dict(to=0)
            if rank == 0:
                torch.save({"model": tmodel.state_dict(), "opt": topt.state_dict(), "step": end}, tckpt + ".tmp")
                os.replace(tckpt + ".tmp", tckpt)
            dist.barrier()

    if rank == 0:
        out = {}
        if use_d:
            out["dtrain"] = np.concatenate([p.data.reshape(-1) for p in params])
        if use_t:
            out["torch"] = tmodel.flat_params()
        np.savez(a.out + ".npz", **out)
        with open(a.out + ".meta.json", "w") as f:
            json.dump({"args": vars(a), "world": world, "torch": torch.__version__, "numpy": np.__version__,
                       "n_params": int(sum(p.data.size for p in params)), "state_bytes_per_rank": allsb}, f)
    dist.barrier()
    dist.destroy_process_group()
    pg.barrier()
    pg.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
