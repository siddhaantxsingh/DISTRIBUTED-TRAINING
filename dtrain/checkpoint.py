"""Sharded, atomic, checksummed checkpoints.

Layout under `root/`:

    step_000100/rank0.npz ... rankN.npz   per-rank state
    step_000100/manifest.json            step, world size, SHA-256 of every shard
    LATEST                               name of the newest complete step dir

Each rank writes its shard to `<name>.tmp` and renames it; rank 0 writes the
manifest only after a barrier (all shards complete), then atomically replaces
`LATEST`. A crash at any point leaves `LATEST` pointing at the previous
complete checkpoint. On load every shard's checksum is verified.
"""
from __future__ import annotations

import hashlib
import io
import json
import os

import numpy as np


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_dir(d):
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(root: str, step: int, rank: int, world: int, state: dict, barrier) -> str:
    name = f"step_{step:06d}"
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    buf = io.BytesIO()
    np.savez(buf, **{k: np.asarray(v) for k, v in state.items()})
    tmp = os.path.join(d, f"rank{rank}.npz.tmp")
    with open(tmp, "wb") as f:
        f.write(buf.getvalue())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, os.path.join(d, f"rank{rank}.npz"))
    barrier()
    if rank == 0:
        man = {"step": step, "world": world,
               "shards": {f"rank{r}.npz": _sha(os.path.join(d, f"rank{r}.npz")) for r in range(world)}}
        with open(os.path.join(d, "manifest.json.tmp"), "w") as f:
            json.dump(man, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(os.path.join(d, "manifest.json.tmp"), os.path.join(d, "manifest.json"))
        _fsync_dir(d)
        with open(os.path.join(root, "LATEST.tmp"), "w") as f:
            f.write(name)
            f.flush()
            os.fsync(f.fileno())
        os.replace(os.path.join(root, "LATEST.tmp"), os.path.join(root, "LATEST"))
        _fsync_dir(root)
    barrier()
    return d


def latest(root: str) -> str | None:
    p = os.path.join(root, "LATEST")
    if not os.path.exists(p):
        return None
    return os.path.join(root, open(p).read().strip())


def load(d: str, rank: int, world: int) -> tuple[int, dict]:
    with open(os.path.join(d, "manifest.json")) as f:
        man = json.load(f)
    if man["world"] != world:
        raise ValueError(f"checkpoint was written by {man['world']} ranks, this job has {world}")
    shard = os.path.join(d, f"rank{rank}.npz")
    if _sha(shard) != man["shards"][f"rank{rank}.npz"]:
        raise ValueError(f"checksum mismatch in {shard}")
    with np.load(shard) as z:
        return man["step"], {k: z[k] for k in z.files}
