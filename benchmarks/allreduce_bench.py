"""All-reduce latency / bus bandwidth: dtrain ring, dtrain direct, OpenMPI.

    python benchmarks/allreduce_bench.py --world 4 --out results.json
    (MPI part: mpirun -n 4 python benchmarks/allreduce_bench.py --mpi --out mpi.json)

Bus bandwidth follows the NCCL convention: 2 (W-1)/W * bytes / time, i.e.
the per-rank link bandwidth an optimal ring would need. float32 data; the
median of `iters` timed calls after warm-up, measured on rank 0 between
barriers.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SIZES = [1 << k for k in range(10, 27, 2)]  # 1 KiB .. 64 MiB


def iters_for(n):
    return max(5, min(200, (64 << 20) // max(n, 1) // 4))


def run_dtrain(rank, world, addr, q):
    from dtrain.comm import ProcessGroup
    pg = ProcessGroup(rank, world, addr, timeout=120)
    res = []
    for algo in ("ring", "direct"):
        for nbytes in SIZES:
            x = np.zeros(nbytes // 4, np.float32)  # zeros: repeated sums cannot overflow
            for _ in range(3):
                pg.allreduce(x, algo)
            ts = []
            for _ in range(iters_for(nbytes)):
                pg.barrier()
                t0 = time.perf_counter()
                pg.allreduce(x, algo)
                ts.append(time.perf_counter() - t0)
            res.append({"impl": f"dtrain-{algo}", "bytes": nbytes, "seconds": float(np.median(ts))})
    pg.close()
    if rank == 0:
        q.put(res)


def run_mpi():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    res = []
    for nbytes in SIZES:
        x = np.zeros(nbytes // 4, np.float32)
        y = np.empty_like(x)
        for _ in range(3):
            comm.Allreduce(x, y)
        ts = []
        for _ in range(iters_for(nbytes)):
            comm.Barrier()
            t0 = time.perf_counter()
            comm.Allreduce(x, y)
            ts.append(time.perf_counter() - t0)
        res.append({"impl": "openmpi", "bytes": nbytes, "seconds": float(np.median(ts))})
    return res if comm.rank == 0 else None


def annotate(res, world):
    for r in res:
        r["world"] = world
        r["bus_GBps"] = 2 * (world - 1) / world * r["bytes"] / r["seconds"] / 1e9
        r["latency_us"] = r["seconds"] * 1e6
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--mpi", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mpi:
        res = run_mpi()
        if res is not None:
            from mpi4py import MPI
            json.dump(annotate(res, MPI.COMM_WORLD.size), open(a.out, "w"), indent=1)
        return
    import multiprocessing as mp
    from dtrain.comm import StoreServer
    srv = StoreServer()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=run_dtrain, args=(r, a.world, srv.address, q)) for r in range(a.world)]
    for p in ps:
        p.start()
    res = q.get(timeout=1800)
    for p in ps:
        p.join()
    srv.close()
    json.dump(annotate(res, a.world), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
