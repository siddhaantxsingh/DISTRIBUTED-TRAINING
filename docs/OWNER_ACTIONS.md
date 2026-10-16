# Owner actions (need hardware this environment does not have)

Status of everything here: **BLOCKED** in this environment, because it has no GPU. The scripts are written but have never run. Treat them as IMPLEMENTED BUT UNVERIFIED until you run them.

Important: dtrain itself is NumPy-only and runs on CPU. The GPU scripts check **PyTorch** on CUDA/NCCL against dtrain's CPU result on the same model and data. They do not run dtrain on a GPU.

## 1. RTX 4060 (single GPU): NCCL sanity and CUDA DDP correctness

```bash
bash owner/rtx4060_single_gpu.sh
```

What it does:

- installs CUDA PyTorch from download.pytorch.org (cu126);
- runs `torchrun --nproc_per_node 1 benchmarks/torch_gpu_check.py`;
- runs the CPU/gloo cross-check tests.

Expected output:

- `"nccl_sanity_ok": true`;
- `ddp_vs_dtrain_param_maxabs_diff_per_step`: values around 1e-15 to 1e-12 (float64; CUDA kernels reassociate differently), not exactly 0;
- `pytest`: 8 passed.

Save these files to `benchmarks/results/<date>/gpu/` and commit:

- `owner_rtx4060_gpu1.json`, `.log`;
- `owner_rtx4060_nvidia_smi.txt`;
- `owner_rtx4060_pytest_torch.log`.

Then update `docs/requirements.md` (row "GPU / NCCL").

## 2. Rented multi-GPU node: real NCCL scaling

```bash
bash owner/multi_gpu_nccl.sh        # runs 1, 2, 4, 8 GPUs (up to what is visible)
```

Expected output:

- `nccl_sanity_ok` true for every world size;
- DDP-vs-dtrain differences at float64 rounding level;
- NCCL bus bandwidth that rises with message size. NVLink should reach tens to hundreds of GB/s; PCIe only a few to ~20 GB/s;
- `ddp_throughput.tokens_per_s` that grows with GPU count.

Save `gpu_results/` (including `nvidia_smi.txt` and `topo.txt`) and record the GPU model and interconnect.

## 3. Optional: re-run the CPU throughput comparison on an idle machine

The 2026-10-15 run had load average 16–21 on 4 vCPUs. Run this on an idle 4+ core machine and replace the throughput section of `docs/torch_crosscheck.md`:

```bash
python benchmarks/run_torch_crosscheck.py --out benchmarks/results/<date>/torch
```
