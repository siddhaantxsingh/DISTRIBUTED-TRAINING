"""dtrain ring all-reduce vs torch.distributed all_reduce (gloo, CPU): exactness and speed.

    python -m dtrain.launch --nproc 4 -- python benchmarks/torch_allreduce.py --torch-init /tmp/x --out r.json

1. Exactness. Every rank draws the same-sized random float32 and float64
   arrays (rank-dependent seed). Both libraries all-reduce identical inputs.
   Rank 0 reports, per dtype, the largest absolute difference between the
   two results, the fraction of bit-identical elements, and each library's
   error against an exact reference (math.fsum of the W inputs).
2. Speed. Median latency of float32 all-reduces of 1 KiB .. 64 MiB, between
   barriers, as in allreduce_bench.py (bus bandwidth = 2 (W-1)/W * bytes / time).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from dtrain.comm import init_from_env  # noqa: E402

SIZES = [1 << k for k in range(10, 27, 2)]


def iters_for(n):
    return max(5, min(200, (64 << 20) // max(n, 1) // 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch-init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-check", type=int, default=200_000)
    ap.add_argument("--no-speed", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(1)
    pg = init_from_env()
    rank, world = pg.rank, pg.world
    dist.init_process_group("gloo", init_method=f"file://{a.torch_init}", rank=rank, world_size=world)
    out = {"world": world, "torch": torch.__version__, "check": {}, "speed": []}

    for dt in (np.float32, np.float64):
        xs = [np.random.default_rng([7, r]).standard_normal(a.n_check).astype(dt) for r in range(world)]
        d = xs[rank].copy()
        pg.allreduce(d)
        t = torch.from_numpy(xs[rank].copy())
        dist.all_reduce(t)
        t = t.numpy()
        if rank == 0:
            exact = np.array([math.fsum(v) for v in zip(*[x.astype(np.float64) for x in xs])])
            # for float64 inputs fsum is the correctly rounded sum; for float32 inputs it is exact in float64
            out["check"][np.dtype(dt).name] = {
                "max_abs_diff_dtrain_vs_gloo": float(np.max(np.abs(d.astype(np.float64) - t))),
                "bit_identical_frac": float(np.mean(d == t)),
                "max_abs_err_dtrain": float(np.max(np.abs(d.astype(np.float64) - exact))),
                "max_abs_err_gloo": float(np.max(np.abs(t.astype(np.float64) - exact))),
                "max_abs_value": float(np.max(np.abs(exact))),
            }

    if not a.no_speed:
        for impl in ("dtrain-ring", "gloo"):
            for nbytes in SIZES:
                x = np.zeros(nbytes // 4, np.float32)
                tx = torch.from_numpy(x)
                f = (lambda: pg.allreduce(x)) if impl == "dtrain-ring" else (lambda: dist.all_reduce(tx))
                for _ in range(3):
                    f()
                ts = []
                for _ in range(iters_for(nbytes)):
                    pg.barrier()
                    dist.barrier()
                    t0 = time.perf_counter()
                    f()
                    ts.append(time.perf_counter() - t0)
                s = float(np.median(ts))
                out["speed"].append({"impl": impl, "bytes": nbytes, "seconds": s, "latency_us": s * 1e6,
                                     "bus_GBps": 2 * (world - 1) / world * nbytes / s / 1e9})
    if rank == 0:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
    dist.barrier()
    dist.destroy_process_group()
    pg.close()


if __name__ == "__main__":
    main()
