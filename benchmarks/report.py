"""Render docs/benchmarks.md from benchmarks/results/<date>/."""
import json
import statistics
import sys

D = sys.argv[1]
R = f"benchmarks/results/{D}"


def jl(name):
    return [json.loads(l) for l in open(f"{R}/{name}.jsonl")]


L = []
w = L.append
w("# Benchmarks\n")
w(f"Rendered by `python benchmarks/report.py {D}` from `{R}/`.\n")
w("**Machine.** One cloud VM with 4 vCPUs (Intel Xeon @ 2.80 GHz) and no GPU. Every process is pinned to one "
  "BLAS/OpenMP thread, so 4 ranks use all 4 vCPUs. All ranks run on one host and communicate over TCP loopback. "
  "Absolute numbers are small, because this is a NumPy CPU implementation; the comparisons that matter are:\n\n"
  "- dtrain against OpenMPI on the same machine;\n"
  "- 1 rank against 2 and 4 ranks;\n"
  "- each variant against the others.\n")

w("## All-reduce (float32, median of repeated calls)\n")
w("Bus bandwidth = 2(W−1)/W × bytes / time (the NCCL convention). OpenMPI 4.1 `MPI_Allreduce` through mpi4py runs on the same machine as the baseline.\n")
for W in (2, 4):
    rows = json.load(open(f"{R}/allreduce_dtrain_w{W}.json")) + json.load(open(f"{R}/allreduce_mpi_w{W}.json"))
    by = {}
    for r in rows:
        by.setdefault(r["bytes"], {})[r["impl"]] = r
    w(f"### {W} ranks\n")
    w("| size | dtrain ring: latency / bus BW | dtrain direct | OpenMPI | ring as % of MPI BW |")
    w("|---|---|---|---|---|")
    for b in sorted(by):
        x = by[b]
        size = f"{b >> 20} MiB" if b >= 1 << 20 else f"{b >> 10} KiB"
        cell = lambda k: f"{x[k]['latency_us']:,.0f} µs / {x[k]['bus_GBps']:.2f} GB/s"
        w(f"| {size} | {cell('dtrain-ring')} | {cell('dtrain-direct')} | {cell('openmpi')} | "
          f"{100 * x['dtrain-ring']['bus_GBps'] / x['openmpi']['bus_GBps']:.0f} % |")
    w("")
w("For large messages, the ring reaches about 70–80 % of OpenMPI's bandwidth. For small messages, latency is 10–50× "
  "worse: every ring step pays Python and thread hand-off overhead, and a ring has 2(W−1) steps. The direct "
  "algorithm has one step, so it beats the ring below about 64 KiB, but at 64 MiB it moves (W−1)× more data per rank.\n")

w("## Training throughput (weak scaling)\n")
w("Model: a GPT with 4 layers, width 128, 4 heads and a context of 64, about 0.8 M parameters, in float32. Each rank processes "
  "8 sequences × 64 tokens per step. Throughput is the median over steps 6–30.\n")
w("| configuration | ranks | tokens/s | speed-up vs 1 rank | median comm wait / step |")
w("|---|---|---|---|---|")
base = None
for name, W, label in [("tp_single_w1", 1, "single process"), ("tp_ddp_w2", 2, "DDP, overlap"),
                       ("tp_ddp_nooverlap_w2", 2, "DDP, no overlap"), ("tp_zero_w2", 2, "ZeRO-1"),
                       ("tp_ddp_w4", 4, "DDP, overlap"), ("tp_ddp_nooverlap_w4", 4, "DDP, no overlap"),
                       ("tp_zero_w4", 4, "ZeRO-1")]:
    rows = [r for r in jl(name) if "step" in r][5:]
    tps = statistics.median(r["tokens_per_s"] for r in rows)
    cw = statistics.median(r["comm_wait_s"] for r in rows) * 1000
    base = base or tps
    w(f"| {label} | {W} | {tps:,.0f} | {tps / base:.2f}× | {cw:.1f} ms |" if "DDP" in label else
      f"| {label} | {W} | {tps:,.0f} | {tps / base:.2f}× | — |")
w("")
w("Overlapping gradient all-reduce with backward does **not** help in this implementation; it is slightly slower. The "
  "likely reason, not measured: the communication thread and the autograd thread are in the same Python process "
  "and compete for the GIL, while NumPy kernels on these small tensors are too short to hide the communication. "
  "ZeRO-1 is fastest at 4 ranks because it makes one reduce-scatter and one all-gather over a single flat buffer, "
  "instead of one all-reduce per bucket.\n")

z = [r for r in jl("tp_zero_w4") if "optimizer_state_bytes_per_rank" in r]
d = [r for r in jl("tp_ddp_w4") if "step" in r]
if z:
    zs = z[0]["optimizer_state_bytes_per_rank"]
    w("## ZeRO-1 optimizer memory\n")
    w(f"With 4 ranks, each rank stores AdamW state for 1/4 of the parameters: {zs / 1e6:.2f} MB per rank, against "
      f"{4 * zs / 1e6:.2f} MB replicated on every rank under DDP. The parameters themselves and the full gradient are "
      "still replicated; this is stage 1, with a stage-2-style gradient reduce-scatter on the wire.\n")

a, b = [r for r in jl("conv_ddp_w4") if "step" in r], [r for r in jl("conv_single_emul4") if "step" in r]
diff = max(abs(x["loss"] - y["loss"]) for x, y in zip(a, b))
w("## Convergence: 4-rank DDP vs one process\n")
w(f"The run uses {len(a)} steps on tiny-shakespeare, with a global batch of 32 × 64 tokens, warm-up and then cosine decay. The single process runs the "
  f"identical global batch (`--emulate-world 4`).\n")
w(f"* Final training loss: DDP {a[-1]['loss']:.4f}, single process {b[-1]['loss']:.4f}.")
w(f"* Largest per-step loss difference over the run: {diff:.2e} (float32).")
w(f"* Loss at steps 1 / 100 / 200 / {len(a)}: " + " / ".join(f"{a[i]['loss']:.3f}" for i in (0, 99, 199, len(a) - 1)) + "\n")
w("Exactness at float64 (≤ 1e-10) and bit-identical resume and crash recovery are covered by tests; see [testing.md](testing.md).\n")
open("docs/benchmarks.md", "w").write("\n".join(L) + "\n")
print("ok")
