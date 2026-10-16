# Limitations

* **CPU / NumPy only.** No GPU, no CUDA/NCCL. dtrain itself never runs on a GPU. PyTorch (CPU, conda-forge build) is used only to cross-check dtrain against torch.distributed with gloo, and it is optional. The GPU scripts in `owner/` check PyTorch on CUDA against dtrain's CPU result; they have not been run. The autograd engine is written from scratch, so throughput is far below a production framework. The point of the project is the distributed mechanics and their verification.
* **Single host.** All ranks were run over TCP loopback on one 4-vCPU VM. The code is plain TCP, so multi-host should work, but it is untested. Scaling numbers stop at 4 ranks, the number of cores.
* **Small-message latency** is 10–50× worse than OpenMPI, because of Python and thread overhead per ring step. Large-message bandwidth is about 70–80 % of OpenMPI.
* **Overlap does not pay off.** Computation and communication share one Python process and its GIL. A C/C++ communication thread would be needed.
* **ZeRO stage 1 only.** Optimizer state is sharded; full parameters and gradients are still materialised on every rank. There is no stage 3 (parameter sharding), no tensor or pipeline parallelism, and no mixed precision.
* **Elasticity is restart-all with a fixed world size.** A failed rank causes a full-group restart from the last checkpoint. The group cannot shrink or grow, and checkpoints cannot be re-sharded to a different W.
* **The rendezvous store** lives in the launcher process, so it is a single point of failure. There is no authentication; the system is meant for a trusted cluster network.
* **Cross-check scope.** The PyTorch comparison covers one small GPT (110k parameters), 1–4 CPU processes on one host and 50 steps. Results agree to rounding level, not bit for bit: the summation order differs for 3+ ranks, the kernels differ (NumPy vs ATen), and the AdamW formulas differ. torch DDP+gloo did not resume bit-identically at 4 ranks (about 5e-16); the cause was not found.
* **Throughput comparison was taken under heavy load** (load average 16–21 on 4 vCPUs) and is indicative only.
