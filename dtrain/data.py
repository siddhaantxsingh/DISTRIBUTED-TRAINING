"""Character-level text data and deterministic distributed batching.

The global batch for (step, micro-step) is a pure function of the seed, so
every rank can take its slice without communication, a run resumed from a
checkpoint sees exactly the batches it would have seen, and a single process
can reproduce the global batch of a W-rank job (used by equivalence tests).
"""
from __future__ import annotations

import hashlib
import os
import urllib.request

import numpy as np

TINY_SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TINY_SHAKESPEARE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def load_text(path: str | None = None, cache_dir: str = "data") -> str:
    if path:
        return open(path, encoding="utf-8").read()
    p = os.path.join(cache_dir, "tinyshakespeare.txt")
    if not os.path.exists(p):
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(TINY_SHAKESPEARE, p + ".tmp")
        got = hashlib.sha256(open(p + ".tmp", "rb").read()).hexdigest()
        if got != TINY_SHAKESPEARE_SHA256:
            os.remove(p + ".tmp")
            raise ValueError(f"tiny shakespeare checksum mismatch: {got}")
        os.replace(p + ".tmp", p)
    return open(p, encoding="utf-8").read()


class CharData:
    def __init__(self, text: str, val_fraction: float = 0.1):
        chars = sorted(set(text))
        self.vocab = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        ids = np.array([self.stoi[c] for c in text], dtype=np.int32)
        n = int(len(ids) * (1 - val_fraction))
        self.train, self.val = ids[:n], ids[n:]


def global_batch(ids: np.ndarray, step: int, micro: int, global_bs: int, ctx: int, seed: int):
    rng = np.random.default_rng([seed, step, micro])
    starts = rng.integers(0, len(ids) - ctx - 1, size=global_bs)
    x = np.stack([ids[s:s + ctx] for s in starts])
    y = np.stack([ids[s + 1:s + ctx + 1] for s in starts])
    return x, y


def rank_batch(ids, step, micro, bs_per_rank, ctx, seed, rank, world):
    x, y = global_batch(ids, step, micro, bs_per_rank * world, ctx, seed)
    sl = slice(rank * bs_per_rank, (rank + 1) * bs_per_rank)
    return x[sl], y[sl]
