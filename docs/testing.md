# Testing

`python -m pytest` runs 36 tests in about 30 s, plus 8 PyTorch cross-check tests when `torch` is installed (44 in about 2.5 min); every distributed test starts real OS processes.

| File | What it checks |
|---|---|
| `tests/test_autograd.py` (10) | Finite-difference gradient checks in float64 for every op: broadcasting add/mul/sub, batched and broadcast matmul, reductions, reshape/transpose, GELU, ReLU, LayerNorm, causal attention, embedding with repeated indices, cross-entropy. Spot-checks of every parameter of a full 2-layer GPT. The grad-ready hook fires exactly once, with the final gradient, for a parameter used twice. |
| `tests/test_comm.py` (16) | Ring and direct all-reduce on 2, 3 and 4 processes for sizes 0, 1, 2, 3, 5, 7, 64, 1000 and 12345 (empty and uneven chunks), against a NumPy sum to 1e-12 and bit-identical on every rank. Reduce-scatter and all-gather against exact totals. Broadcast from a non-zero root. Async all-reduces complete in order. A peer that crashes mid-collective makes the survivor raise `CommError` instead of hanging. |
| `tests/test_training.py` (10) | End-to-end through the launcher, in float64. DDP (4 ranks) equals one process on the same global batch to ≤ 1e-10. ZeRO equals DDP, and its per-rank optimizer state is exactly 2 × 8 × ⌈n/4⌉ bytes. Bucket size, overlap on/off and ring/direct give identical results. Gradient accumulation equals the larger batch. Preempt-and-resume is **bit-identical** for DDP and ZeRO. A worker crash at step 5, followed by an automatic restart from the step-3 checkpoint, gives **bit-identical** final parameters to a run with no failure, for DDP and ZeRO. With no restarts left, the failing exit code propagates. A corrupted checkpoint shard is rejected by its checksum. |
| `tests/test_torch_crosscheck.py` (8, skipped without torch) | dtrain ring all-reduce vs gloo on identical inputs (bit-identical at 2 ranks; both within the summation error bound at 4). DDP (AdamW and SGD) vs `torch.nn.parallel.DistributedDataParallel`, and ZeRO-1 vs `ZeroRedundancyOptimizer`, at float64, 2 ranks, gradients/params ≤ 1e-13. Checkpoint, exit, resume is bit-identical for both at 2 ranks. dtrain AdamW vs `torch.optim.AdamW` on identical gradients. |

Benchmarks (`benchmarks/`) also cross-check results. The 300-step float32 convergence run of 4-rank DDP and its single-process emulation differ by at most 2.4e-7 in loss per step.

## Not covered

* Multi-host networking: everything runs on one machine over loopback.
* Network partitions or slow (rather than dead) peers, beyond the receive timeout.
* Restarting with a different world size: checkpoints are tied to W, and loading with another W is rejected.
