# Requirements matrix — Project 9 (Distributed training, CPU)

| Requirement | Status | Evidence |
|---|---|---|
| Collective communication (all-reduce, reduce-scatter, all-gather, broadcast, barrier) | VERIFIED | from-scratch TCP implementation; `tests/test_comm.py` |
| Ring all-reduce + baseline algorithm | VERIFIED | ring vs direct; benchmarked against OpenMPI |
| Data-parallel training (DDP) | VERIFIED | equals single process to 1e-10 (float64); convergence run |
| Gradient bucketing + comm/compute overlap | VERIFIED (functionally) | results identical with/without overlap; overlap gives no speed-up here (documented) |
| Gradient accumulation | VERIFIED | equals larger batch |
| Optimizer state sharding (ZeRO-1) | VERIFIED | equals DDP; per-rank state = 1/W (exact byte count tested) |
| Sharded checkpointing (atomic, checksummed) | VERIFIED | bit-identical resume; corrupted shard rejected |
| Fault tolerance (worker failure → restart → resume) | VERIFIED | injected crash; final params bit-identical to failure-free run |
| Failure detection (no hang on dead peer) | VERIFIED | `test_peer_failure_raises_commerror_not_hang` |
| Throughput scaling measurement | VERIFIED (1–4 ranks, single host) | docs/benchmarks.md |
| Cross-check vs torch.distributed DDP (gloo, CPU) | VERIFIED (CPU, 1–4 ranks) | grads equal to ≤ 1.6e-15 (float64); `tests/test_torch_crosscheck.py`; [torch_crosscheck.md](torch_crosscheck.md); `benchmarks/results/2026-10-15/torch/` |
| Cross-check ZeRO-1 vs `ZeroRedundancyOptimizer` | VERIFIED (CPU, 2–4 ranks) | params equal to ≤ 3.9e-14 after 50 steps (float64); same total state, dtrain shards evenly |
| Checkpoint/resume equivalence vs PyTorch | VERIFIED (CPU) | dtrain resume bit-identical; matches torch after resume to 1.3e-14; torch+gloo itself not bit-identical at 4 ranks (~5e-16, cause not identified) |
| Throughput vs torch DDP/gloo (CPU) | PARTIAL | measured, but under load 16–21 on 4 vCPUs; scaling not measurable; see torch_crosscheck.md |
| GPU / NCCL (RTX 4060 single GPU, multi-GPU scaling) | BLOCKED | no GPU here; owner scripts `owner/rtx4060_single_gpu.sh`, `owner/multi_gpu_nccl.sh` and `benchmarks/torch_gpu_check.py` written, never run (IMPLEMENTED BUT UNVERIFIED) |
| PyTorch available locally | VERIFIED | PyPI wheel is CUDA-only (import fails without ~3 GB CUDA libs); used conda-forge `pytorch 2.14.1 cpu_generic` via micromamba; see `pypi_wheel_probe.txt` |
| Multi-host | IMPLEMENTED BUT UNVERIFIED | plain TCP; only loopback tested |
| Tensor / pipeline parallelism, ZeRO-2/3, mixed precision | NOT IMPLEMENTED | — |
| CI | VERIFIED | `.github/workflows/distributed-training.yml`: `test` job (36 passed, torch tests skip) and `torch-crosscheck` job (torch 2.14.1+cpu, 8 passed); both green in run 38861771722 |
