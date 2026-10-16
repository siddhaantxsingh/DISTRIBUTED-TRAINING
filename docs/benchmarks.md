# Benchmarks

Rendered by `python benchmarks/report.py 2026-10-13` from `benchmarks/results/2026-10-13/`.

**Machine.** One cloud VM with 4 vCPUs (Intel Xeon @ 2.80 GHz) and no GPU. Every process is pinned to one BLAS/OpenMP thread, so 4 ranks use all 4 vCPUs. All ranks run on one host and communicate over TCP loopback. Absolute numbers are small, because this is a NumPy CPU implementation; the comparisons that matter are:

- dtrain against OpenMPI on the same machine;
- 1 rank against 2 and 4 ranks;
- each variant against the others.

## All-reduce (float32, median of repeated calls)

Bus bandwidth = 2(W−1)/W × bytes / time (the NCCL convention). OpenMPI 4.1 `MPI_Allreduce` through mpi4py runs on the same machine as the baseline.

### 2 ranks

| size | dtrain ring: latency / bus BW | dtrain direct | OpenMPI | ring as % of MPI BW |
|---|---|---|---|---|
| 1 KiB | 117 µs / 0.01 GB/s | 42 µs / 0.02 GB/s | 4 µs / 0.27 GB/s | 3 % |
| 4 KiB | 118 µs / 0.03 GB/s | 49 µs / 0.08 GB/s | 8 µs / 0.53 GB/s | 6 % |
| 16 KiB | 95 µs / 0.17 GB/s | 59 µs / 0.28 GB/s | 22 µs / 0.75 GB/s | 23 % |
| 64 KiB | 126 µs / 0.52 GB/s | 94 µs / 0.70 GB/s | 57 µs / 1.15 GB/s | 45 % |
| 256 KiB | 448 µs / 0.58 GB/s | 517 µs / 0.51 GB/s | 151 µs / 1.73 GB/s | 34 % |
| 1 MiB | 763 µs / 1.37 GB/s | 1,505 µs / 0.70 GB/s | 620 µs / 1.69 GB/s | 81 % |
| 4 MiB | 3,748 µs / 1.12 GB/s | 10,386 µs / 0.40 GB/s | 2,654 µs / 1.58 GB/s | 71 % |
| 16 MiB | 17,704 µs / 0.95 GB/s | 52,511 µs / 0.32 GB/s | 13,536 µs / 1.24 GB/s | 76 % |
| 64 MiB | 92,703 µs / 0.72 GB/s | 253,892 µs / 0.26 GB/s | 62,495 µs / 1.07 GB/s | 67 % |

### 4 ranks

| size | dtrain ring: latency / bus BW | dtrain direct | OpenMPI | ring as % of MPI BW |
|---|---|---|---|---|
| 1 KiB | 410 µs / 0.00 GB/s | 106 µs / 0.01 GB/s | 8 µs / 0.20 GB/s | 2 % |
| 4 KiB | 481 µs / 0.01 GB/s | 166 µs / 0.04 GB/s | 18 µs / 0.35 GB/s | 4 % |
| 16 KiB | 456 µs / 0.05 GB/s | 169 µs / 0.15 GB/s | 46 µs / 0.54 GB/s | 10 % |
| 64 KiB | 509 µs / 0.19 GB/s | 324 µs / 0.30 GB/s | 80 µs / 1.22 GB/s | 16 % |
| 256 KiB | 711 µs / 0.55 GB/s | 1,441 µs / 0.27 GB/s | 250 µs / 1.57 GB/s | 35 % |
| 1 MiB | 2,879 µs / 0.55 GB/s | 4,975 µs / 0.32 GB/s | 897 µs / 1.75 GB/s | 31 % |
| 4 MiB | 8,531 µs / 0.74 GB/s | 23,870 µs / 0.26 GB/s | 5,456 µs / 1.15 GB/s | 64 % |
| 16 MiB | 32,478 µs / 0.77 GB/s | 248,925 µs / 0.10 GB/s | 24,997 µs / 1.01 GB/s | 77 % |
| 64 MiB | 132,731 µs / 0.76 GB/s | 665,733 µs / 0.15 GB/s | 93,797 µs / 1.07 GB/s | 71 % |

For large messages, the ring reaches about 70–80 % of OpenMPI's bandwidth. For small messages, latency is 10–50× worse: every ring step pays Python and thread hand-off overhead, and a ring has 2(W−1) steps. The direct algorithm has one step, so it beats the ring below about 64 KiB, but at 64 MiB it moves (W−1)× more data per rank.

## Training throughput (weak scaling)

Model: a GPT with 4 layers, width 128, 4 heads and a context of 64, about 0.8 M parameters, in float32. Each rank processes 8 sequences × 64 tokens per step. Throughput is the median over steps 6–30.

| configuration | ranks | tokens/s | speed-up vs 1 rank | median comm wait / step |
|---|---|---|---|---|
| single process | 1 | 2,577 | 1.00× | — |
| DDP, overlap | 2 | 4,591 | 1.78× | 6.5 ms |
| DDP, no overlap | 2 | 4,727 | 1.83× | 5.2 ms |
| ZeRO-1 | 2 | 4,509 | 1.75× | — |
| DDP, overlap | 4 | 8,192 | 3.18× | 24.2 ms |
| DDP, no overlap | 4 | 8,697 | 3.37× | 15.8 ms |
| ZeRO-1 | 4 | 9,266 | 3.60× | — |

Overlapping gradient all-reduce with backward does **not** help in this implementation; it is slightly slower. The likely reason, not measured: the communication thread and the autograd thread are in the same Python process and compete for the GIL, while NumPy kernels on these small tensors are too short to hide the communication. ZeRO-1 is fastest at 4 ranks because it makes one reduce-scatter and one all-gather over a single flat buffer, instead of one all-reduce per bucket.

## ZeRO-1 optimizer memory

With 4 ranks, each rank stores AdamW state for 1/4 of the parameters: 1.64 MB per rank, against 6.55 MB replicated on every rank under DDP. The parameters themselves and the full gradient are still replicated; this is stage 1, with a stage-2-style gradient reduce-scatter on the wire.

## Convergence: 4-rank DDP vs one process

The run uses 300 steps on tiny-shakespeare, with a global batch of 32 × 64 tokens, warm-up and then cosine decay. The single process runs the identical global batch (`--emulate-world 4`).

* Final training loss: DDP 2.2423, single process 2.2423.
* Largest per-step loss difference over the run: 2.38e-07 (float32).
* Loss at steps 1 / 100 / 200 / 300: 4.238 / 2.422 / 2.307 / 2.242

Exactness at float64 (≤ 1e-10) and bit-identical resume and crash recovery are covered by tests; see [testing.md](testing.md).

