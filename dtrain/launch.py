"""Elastic launcher.

    python -m dtrain.launch --nproc N [--max-restarts K] [--timeout S] -- CMD ARGS...

Starts a rendezvous store and N worker processes with RANK, WORLD_SIZE,
STORE_ADDR and DTRAIN_RESTART_COUNT set. If any worker exits non-zero (crash,
injected fault, peer-failure CommError), the launcher stops the remaining
workers and restarts the whole group, up to K times, with a fresh rendezvous
namespace; the training script resumes from its latest checkpoint. Exit code
0 when all workers of an attempt exit 0, otherwise the failing code.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

from .comm import StoreServer


def run(nproc: int, cmd: list[str], max_restarts: int = 0, timeout: float = 60.0, poll: float = 0.05) -> int:
    store = StoreServer()
    try:
        for attempt in range(max_restarts + 1):
            procs = []
            for r in range(nproc):
                env = dict(os.environ, RANK=str(r), WORLD_SIZE=str(nproc), STORE_ADDR=store.address,
                           DTRAIN_PREFIX=f"a{attempt}/", DTRAIN_RESTART_COUNT=str(attempt),
                           DTRAIN_TIMEOUT=str(timeout), OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"),
                           OPENBLAS_NUM_THREADS=os.environ.get("OPENBLAS_NUM_THREADS", "1"))
                procs.append(subprocess.Popen(cmd, env=env))
            failed = None
            while True:
                codes = [p.poll() for p in procs]
                bad = [(r, c) for r, c in enumerate(codes) if c not in (None, 0)]
                if bad:
                    failed = bad[0]
                    break
                if all(c == 0 for c in codes):
                    return 0
                time.sleep(poll)
            r, c = failed
            print(f"[launch] attempt {attempt}: rank {r} exited with {c}; stopping the group", file=sys.stderr, flush=True)
            for p in procs:
                if p.poll() is None:
                    p.send_signal(signal.SIGTERM)
            deadline = time.monotonic() + 10
            for p in procs:
                try:
                    p.wait(max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            if attempt == max_restarts:
                return c
            print(f"[launch] restarting (restart {attempt + 1}/{max_restarts})", file=sys.stderr, flush=True)
        return 1
    finally:
        store.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, required=True)
    ap.add_argument("--max-restarts", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    sys.exit(run(a.nproc, cmd, a.max_restarts, a.timeout))


if __name__ == "__main__":
    main()
