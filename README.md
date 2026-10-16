<div align="center">

# 🌐 dtrain — Distributed Training from Scratch

### DDP, ZeRO-1, ring all-reduce and elastic fault tolerance — built from first principles and checked against PyTorch

**Own autograd engine · own TCP collectives · gradient bucketing · sharded optimizer · checksummed checkpoints · crash → restart → bit-identical resume**

<br/>

![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy_only-013243?style=for-the-badge&logo=numpy&logoColor=white)
![TCP](https://img.shields.io/badge/collectives-raw_TCP-1565C0?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/cross--checked_vs-torch.distributed-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-2E7D32?style=for-the-badge)

![Tests](https://img.shields.io/badge/tests-44%2F44_passing-2E7D32?style=flat-square)
![DDP](https://img.shields.io/badge/DDP_%3D_single_process-≤1e--10-2E7D32?style=flat-square)
![Resume](https://img.shields.io/badge/crash_recovery-bit--identical-6A1B9A?style=flat-square)
![ZeRO](https://img.shields.io/badge/ZeRO--1_optimizer_memory-4×_smaller-E65100?style=flat-square)
![Scaling](https://img.shields.io/badge/weak_scaling_1→4-3.60×-1565C0?style=flat-square)
![Ring](https://img.shields.io/badge/ring_all--reduce-71%25_of_OpenMPI-00838F?style=flat-square)

</div>

---

## ⚡ TL;DR

`dtrain` trains a small GPT across multiple processes **without any deep-learning framework**. Every layer of the distributed-training stack — the stuff hidden inside `torch.distributed`, DeepSpeed and `torchrun` — is written by hand and verified:

- 🧠 **Autograd engine** on NumPy, with grad-ready hooks that fire exactly once per parameter (what DDP needs to start communicating early).
- 🔗 **Collectives over raw TCP** — ring and direct all-reduce, reduce-scatter, all-gather, broadcast — with failure detection and a fixed summation order, so every rank gets **bit-identical** results.
- 📦 **DDP** with ~1 MiB gradient buckets, comm/compute overlap and `no_sync()` gradient accumulation.
- 🧩 **ZeRO-1** — optimizer state sharded across ranks: **4× less optimizer memory** at 4 ranks.
- 💾 **Sharded, SHA-256-checksummed, atomic checkpoints**.
- ♻️ **An elastic launcher** that restarts the whole group after a worker crash. Training then resumes **bit-identically** to a run that never failed.

> [!NOTE]
> **Verified against the real thing.** Gradients match `torch.nn.parallel.DistributedDataParallel` to **≤ 1.6e-15** (float64); ZeRO-1 matches `ZeroRedundancyOptimizer` to **≤ 3.9e-14**; the ring all-reduce is **bit-identical to gloo** at 2 ranks.

---

## 🏗️ Architecture

```mermaid
flowchart TB
    L["♻️ launch.py — elastic launcher<br/>TCP rendezvous store · spawns N workers · restart-all on failure"]:::launch

    subgraph W["👷 Each worker process (rank r of W)"]
        direction TB
        T["🔁 train.py<br/>single · ddp · zero · LR schedule · fault injection"]:::train
        subgraph M["Model"]
            direction LR
            NN["🧱 nn.py<br/>GPT · attention · LayerNorm"]:::model
            AG["🧠 autograd.py<br/>reverse-mode · grad-ready hooks"]:::model
        end
        subgraph P["Parallelism"]
            direction LR
            DDP["📦 ddp.py<br/>buckets · overlap · no_sync"]:::par
            Z["🧩 zero.py<br/>ZeRO-1 sharded AdamW"]:::par
        end
        C["🔗 comm.py<br/>ring / direct all-reduce · reduce-scatter<br/>all-gather · broadcast · async queue"]:::comm
        CK["💾 checkpoint.py<br/>sharded · SHA-256 manifest · atomic LATEST"]:::ckpt
        D["📚 data.py<br/>deterministic global batch per (step, micro-step)"]:::data
        T --> M --> P --> C
        T --> CK
        T --> D
    end

    NET["🌐 Full-mesh TCP<br/>one connection per rank pair · TCP_NODELAY · 4 MiB buffers"]:::net

    L -- "spawn / restart" --> W
    C <--> NET

    classDef launch fill:#6A1B9A,stroke:#4A148C,color:#fff
    classDef train fill:#37474F,stroke:#263238,color:#fff
    classDef model fill:#1565C0,stroke:#0D47A1,color:#fff
    classDef par fill:#E65100,stroke:#BF360C,color:#fff
    classDef comm fill:#00838F,stroke:#006064,color:#fff
    classDef ckpt fill:#2E7D32,stroke:#1B5E20,color:#fff
    classDef data fill:#F9A825,stroke:#F57F17,color:#000
    classDef net fill:#455A64,stroke:#263238,color:#fff
    style W fill:transparent,stroke:#90A4AE,stroke-dasharray:5 5
    style M fill:transparent,stroke:#1565C0,stroke-dasharray:3 3
    style P fill:transparent,stroke:#E65100,stroke-dasharray:3 3
```

| Module | Responsibility |
|---|---|
| `autograd.py` | Reverse-mode autograd on NumPy; hooks fire when a leaf's gradient is final |
| `nn.py` | Linear, LayerNorm, causal self-attention, GPT |
| `optim.py` | AdamW on flat arrays (shared by DDP and ZeRO), global-norm clipping |
| `comm.py` | Rendezvous store, full-mesh process group, ring/direct collectives, async work queue |
| `ddp.py` | Parameter broadcast, gradient buckets, overlap, `no_sync` |
| `zero.py` | Flat parameter buffer, reduce-scatter grads, sharded AdamW, all-gather params |
| `checkpoint.py` | Per-rank shards, SHA-256 manifest, atomic `LATEST` pointer |
| `data.py` | Char-level corpus; global batch is a pure function of (seed, step, micro-step) |
| `launch.py` | Elastic launcher with `--max-restarts` |

---

## 🔗 Ring all-reduce

Each rank's buffer is split into W chunks. Two phases of **W − 1 steps** each move only **2(W−1)/W** of the data per rank — bandwidth-optimal, independent of W.

```mermaid
flowchart TB
    subgraph RS["① Reduce-scatter · W−1 steps"]
        direction LR
        R0["Rank 0"]:::r0 -- "chunk" --> R1["Rank 1"]:::r1 -- "chunk" --> R2["Rank 2"]:::r2 -- "chunk" --> R3["Rank 3"]:::r3 -- "chunk" --> R0
    end
    subgraph AGX["② All-gather · W−1 steps"]
        direction LR
        G0["Rank 0"]:::r0 -- "summed chunk" --> G1["Rank 1"]:::r1 -- "summed chunk" --> G2["Rank 2"]:::r2 -- "summed chunk" --> G3["Rank 3"]:::r3 -- "summed chunk" --> G0
    end
    RS == "rank r now owns the<br/>total of chunk r+1" ==> AGX
    AGX == "every rank holds the full sum<br/>bit-identical everywhere" ==> OUT(["✅ Done"]):::done

    classDef r0 fill:#1565C0,stroke:#0D47A1,color:#fff
    classDef r1 fill:#6A1B9A,stroke:#4A148C,color:#fff
    classDef r2 fill:#E65100,stroke:#BF360C,color:#fff
    classDef r3 fill:#00838F,stroke:#006064,color:#fff
    classDef done fill:#2E7D32,stroke:#1B5E20,color:#fff
    style RS fill:transparent,stroke:#90A4AE,stroke-dasharray:5 5
    style AGX fill:transparent,stroke:#90A4AE,stroke-dasharray:5 5
```

- **Deterministic:** the summation order is fixed, so every rank — and every rerun — gets bit-identical results.
- **Full duplex:** each connection has a dedicated sender thread, so a rank sends to its successor while receiving from its predecessor. Messages ≤ 64 KiB are sent inline to skip the thread hand-off.
- **Safe framing:** every message carries `(sequence number, byte count)`; a mismatch raises `CommError("protocol desync")` instead of silently misreading data.
- **Failure detection:** every receive has a timeout (`DTRAIN_TIMEOUT`, default 60 s). A dead peer makes survivors raise `CommError` — never hang.
- **Direct all-reduce** is the baseline: one step, but (W−1)× the data. It wins below ~64 KiB; the ring wins above.

---

## 📦 DDP — gradient bucketing and overlap

```mermaid
sequenceDiagram
    autonumber
    participant BW as 🧠 Backward pass
    participant HK as 🪝 Grad-ready hook
    participant BK as 📦 Bucket (~1 MiB)
    participant CT as 🔗 Comm thread
    participant OP as ⚙️ Optimizer

    Note over BW: parameters packed in reverse definition order
    BW->>HK: last use of param p back-propagated
    HK->>BK: copy grad(p) into its bucket slot
    BK->>CT: bucket full → async ring all-reduce
    Note over BW,CT: backward continues while the bucket is reduced (overlap)
    BW->>HK: … next parameters …
    HK->>BK: next bucket fills
    BK->>CT: async all-reduce
    BW->>OP: backward finished
    OP->>CT: finish() — wait for all buckets
    CT-->>OP: summed grads ÷ W written back
    OP->>OP: AdamW step (identical on every rank)
```

`no_sync()` skips communication on gradient-accumulation micro-steps — and accumulated training equals training on the larger batch (tested).

---

## 🧩 ZeRO-1 — sharded optimizer state

```mermaid
flowchart LR
    G["🧮 Local grads<br/>(full, flattened)"]:::g --> RS["🔻 Reduce-scatter<br/>rank r gets summed shard r"]:::c
    RS --> N["📏 Shard ‖g‖² → 1 scalar all-reduce<br/>global-norm clipping"]:::c
    N --> A["⚙️ AdamW on own shard only<br/>holds 1/W of m and v"]:::opt
    A --> AG["🔺 All-gather updated shards<br/>into the flat parameter buffer"]:::c
    AG --> P["🧱 Model arrays are views<br/>into the flat buffer — no copy"]:::p

    classDef g fill:#1565C0,stroke:#0D47A1,color:#fff
    classDef c fill:#00838F,stroke:#006064,color:#fff
    classDef opt fill:#E65100,stroke:#BF360C,color:#fff
    classDef p fill:#2E7D32,stroke:#1B5E20,color:#fff
```

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#E65100"}}}}%%
xychart-beta
    title "AdamW optimizer state per rank, 4 ranks (MB)"
    x-axis ["DDP (replicated)", "ZeRO-1 (sharded)"]
    y-axis "MB per rank" 0 --> 7
    bar [6.55, 1.64]
```

dtrain splits the flat buffer **evenly** (every rank holds exactly 441,856 B in the 4-rank float64 cross-check), while PyTorch's `ZeroRedundancyOptimizer` assigns whole tensors (422,912 – 458,752 B per rank).

---

## ♻️ Fault tolerance — crash, restart, resume bit-identically

```mermaid
sequenceDiagram
    autonumber
    participant L as ♻️ Launcher
    participant R0 as 👷 Rank 0
    participant R1 as 👷 Rank 1…3
    participant FS as 💾 Checkpoint dir

    L->>R0: spawn (attempt a0)
    L->>R1: spawn (attempt a0)
    R0->>FS: step 3 · each rank writes its shard
    R1->>FS: shard
    R0->>FS: barrier → manifest (SHA-256 per shard) → atomic replace LATEST
    Note over R1: 💥 crash at step 5
    R0->>R0: recv timeout / closed socket → CommError → exit ≠ 0
    L->>L: stop whole group · restarts left?
    L->>R0: respawn (fresh rendezvous namespace a1)
    L->>R1: respawn
    R0->>FS: load LATEST · verify world size + checksums
    R1->>FS: load shard
    Note over R0,R1: deterministic data + deterministic reductions
    R0-->>L: final params bit-identical to a run with no failure ✅
```

- The global batch for each (step, micro-step) is a **pure function of the seed**, so a resumed run sees exactly the data it would have seen.
- A **corrupted shard is rejected** by its checksum (tested).
- With no restarts left, the failing exit code propagates.

---

## 📈 Results

Measured on one 4-vCPU VM (Xeon @ 2.80 GHz), one BLAS thread per rank, TCP loopback. Details: [`docs/benchmarks.md`](docs/benchmarks.md).

### Correctness

| Check | Result |
|---|---|
| DDP (4 ranks) vs one process, same global batch | equal to **≤ 1e-10** (float64); max loss diff **2.4e-7** over 300 float32 steps |
| ZeRO-1 vs DDP | equal to **≤ 1e-10**; optimizer state per rank cut **4×** |
| Crash at step 5 → auto-restart → resume | final parameters **bit-identical** to the uninterrupted run |
| vs PyTorch DDP (gloo, CPU), same model / data / seed | gradients **≤ 1.6e-15** (float64), **≤ 1.4e-6** (float32) |
| vs `ZeroRedundancyOptimizer` | parameters **≤ 3.9e-14** after 50 steps (float64) |
| Ring all-reduce vs gloo, 2 ranks | **bit-identical** |
| Checkpoint → exit → resume at 4 ranks | dtrain bit-identical; torch DDP + gloo drifts ~5e-16 (documented) |

### Weak scaling (GPT, ~0.8 M params, 8 × 64 tokens per rank)

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#E65100, #90A4AE, #1565C0"}}}}%%
xychart-beta
    title "Weak scaling — tokens/s"
    x-axis ["1 rank", "2 ranks", "4 ranks"]
    y-axis "tokens / s" 0 --> 11000
    bar [2577, 4509, 9266]
    line [2577, 5154, 10308]
    line [2577, 4727, 8697]
```

<sub>🟧 bars = ZeRO-1 · 🟦 blue line = DDP (no overlap) · grey line = ideal linear scaling</sub>

| Configuration | Ranks | Tokens/s | Speed-up | Comm wait / step |
|---|---|---|---|---|
| Single process | 1 | 2,577 | 1.00× | — |
| DDP, no overlap | 2 | 4,727 | 1.83× | 5.2 ms |
| ZeRO-1 | 2 | 4,509 | 1.75× | — |
| DDP, overlap | 4 | 8,192 | 3.18× | 24.2 ms |
| DDP, no overlap | 4 | 8,697 | 3.37× | 15.8 ms |
| **ZeRO-1** | 4 | **9,266** | **3.60×** | — |

ZeRO-1 wins at 4 ranks because it does one reduce-scatter and one all-gather over a single flat buffer, instead of one all-reduce per bucket.

> 🔬 **Negative result, reported honestly:** overlapping all-reduce with backward is *slightly slower* here. The communication and autograd threads share one Python process and compete for the GIL, and NumPy kernels on these small tensors are too short to hide communication. A C/C++ comm thread would be needed.

### All-reduce vs OpenMPI 4.1 (4 ranks, float32, bus bandwidth)

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#00838F, #90A4AE, #C62828"}}}}%%
xychart-beta
    title "All-reduce bus bandwidth, 4 ranks (GB/s)"
    x-axis ["1K", "4K", "16K", "64K", "256K", "1M", "4M", "16M", "64M"]
    y-axis "GB/s" 0 --> 2
    line [0.00, 0.01, 0.05, 0.19, 0.55, 0.55, 0.74, 0.77, 0.76]
    line [0.20, 0.35, 0.54, 1.22, 1.57, 1.75, 1.15, 1.01, 1.07]
    line [0.01, 0.04, 0.15, 0.30, 0.27, 0.32, 0.26, 0.10, 0.15]
```

<sub>🟦 teal = dtrain ring · ⬜ grey = OpenMPI 4.1 · 🟥 red = dtrain direct</sub>

- **Large messages:** the Python ring reaches **~70–80 % of OpenMPI's bandwidth** (0.76 GB/s at 64 MiB, 71 %).
- **Small messages:** 10–50× OpenMPI's latency — each of the 2(W−1) ring steps pays Python and thread hand-off overhead.
- **Ring vs direct:** direct wins below ~64 KiB; at 64 MiB the ring is **5×** faster because direct moves (W−1)× more data.

---

## 🧪 Testing

`python -m pytest` → **36 tests in ~30 s**, plus **8 PyTorch cross-check tests** when `torch` is installed (**44 / 44 passing** in the 2026-10-15 run). Every distributed test launches **real OS processes**.

```mermaid
flowchart LR
    A["🧠 test_autograd · 10<br/>finite-difference grad checks<br/>every op + full GPT"]:::a
    B["🔗 test_comm · 16<br/>2–4 processes · sizes 0…12345<br/>bit-identical · crash → CommError"]:::b
    C["🏋️ test_training · 10<br/>DDP = single · ZeRO = DDP<br/>crash/resume bit-identical"]:::c
    D["🔥 test_torch_crosscheck · 8<br/>vs DDP · ZeroRedundancyOptimizer<br/>gloo · torch AdamW"]:::d
    A --> B --> C --> D

    classDef a fill:#1565C0,stroke:#0D47A1,color:#fff
    classDef b fill:#00838F,stroke:#006064,color:#fff
    classDef c fill:#6A1B9A,stroke:#4A148C,color:#fff
    classDef d fill:#EE4C2C,stroke:#B71C1C,color:#fff
```

| Suite | Highlights |
|---|---|
| `test_autograd.py` | float64 finite-difference checks for broadcasting ops, batched matmul, reductions, GELU, LayerNorm, causal attention, embedding with repeated indices, cross-entropy; the grad-ready hook fires **exactly once** for a parameter used twice |
| `test_comm.py` | ring and direct all-reduce on 2/3/4 processes for sizes 0, 1, 2, 3, 5, 7, 64, 1000, 12345 (empty and uneven chunks), within 1e-12 of NumPy and bit-identical on every rank; a peer crashing mid-collective makes the survivor raise instead of hang |
| `test_training.py` | DDP = single process (≤ 1e-10); ZeRO = DDP with optimizer state exactly 2 × 8 × ⌈n/4⌉ bytes; bucket size, overlap and ring/direct give identical results; preempt-and-resume and crash-and-restart are **bit-identical**; a corrupted shard is rejected |
| `test_torch_crosscheck.py` | gloo all-reduce, `DistributedDataParallel` (AdamW and SGD), `ZeroRedundancyOptimizer`, `torch.optim.AdamW`, checkpoint/resume |

<details>
<summary><b>🔍 Why dtrain and PyTorch differ at all (rounding-level analysis)</b></summary>

- **All-reduce:** at 3+ ranks, 32–44 % of elements differ in the last bit — only the order of additions differs, and both are within 1 ulp of the exact sum.
- **AdamW:** with *identical* gradients the two optimizers already differ by 6.7e-6 (float32), because they compute the bias correction in a different order (`sqrt(v / b2t)` vs `sqrt(v) / sqrt(b2t)`), and Adam's update has derivative lr/eps = 1e5 near eps.
- **Plain SGD** removes the optimizer effect: the step-1 difference is 1e-17 (float64) and grows ~1e4× over 50 steps at the same rate in both precisions — training dynamics amplifying a rounding perturbation, not a bug.

Full write-up: [`docs/torch_crosscheck.md`](docs/torch_crosscheck.md).

</details>

---

## 🚀 Quick start

```bash
pip install -e '.[dev]'
python -m pytest                                     # 36 tests, ~30 s

# single process (downloads tiny-shakespeare, SHA-256 checked)
python -m dtrain.train --mode single --steps 200 --warmup 10

# 4-rank DDP
python -m dtrain.launch --nproc 4 -- python -m dtrain.train --mode ddp --steps 200 --warmup 10

# 4-rank ZeRO-1 that crashes at step 50, restarts itself and resumes from the checkpoint
python -m dtrain.launch --nproc 4 --max-restarts 2 -- python -m dtrain.train --mode zero \
    --steps 200 --ckpt-dir ckpt --ckpt-every 20 --resume --fail-at-step 50
```

<details>
<summary><b>🔥 PyTorch cross-check and benchmarks</b></summary>

```bash
pip install -e '.[torch]'                                       # a CPU build is enough
python -m pytest tests/test_torch_crosscheck.py                 # 8 tests
python benchmarks/run_torch_crosscheck.py --out benchmarks/results/<date>/torch
python benchmarks/torch_report.py benchmarks/results/<date>/torch > docs/torch_crosscheck_tables.md

python benchmarks/allreduce_bench.py                            # + mpirun baseline (pip install -e '.[bench]')
bash benchmarks/scaling.sh
python benchmarks/report.py <date>
```

</details>

---

## 📁 Repository layout

```text
DISTRIBUTED-TRAINING/
├── dtrain/          # autograd · nn · optim · comm · ddp · zero · checkpoint · data · train · launch · torch_ref
├── tests/           # autograd · comm · training · torch cross-check
├── benchmarks/      # all-reduce, scaling and PyTorch cross-check scripts + results/
├── owner/           # GPU scripts: RTX 4060 single-GPU, multi-GPU NCCL
└── docs/            # architecture · testing · benchmarks · limitations · torch_crosscheck · requirements
```

---

## 🧭 Limitations and roadmap

| Area | Today | Next |
|---|---|---|
| Hardware | CPU / NumPy; GPU + NCCL scripts written (`owner/`) but not yet run | Run on an RTX 4060 and multi-GPU NCCL |
| Hosts | Single host over TCP loopback, up to 4 ranks | Multi-host test (plain TCP, should work) |
| Small messages | 10–50× OpenMPI's latency (Python per-step overhead) | C/C++ transport thread |
| Overlap | No gain because of the GIL | Native comm thread |
| Sharding | ZeRO stage 1 only | ZeRO-2/3, tensor and pipeline parallelism, mixed precision |
| Elasticity | Restart-all at a fixed world size | Shrink/grow and re-sharding checkpoints to a new W |
| Rendezvous | Store lives in the launcher (single point of failure), no auth | Replicated store, authentication |

More in [`docs/limitations.md`](docs/limitations.md).

---

<div align="center">

**What `torchrun` + DDP + ZeRO do under the hood — rebuilt, measured, and proven equal to PyTorch.**

<sub>MIT © siddhaantxsingh</sub>

</div>
