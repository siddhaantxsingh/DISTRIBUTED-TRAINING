"""Optimisers operating on flat NumPy views (so ZeRO can run them on shards)."""
from __future__ import annotations

import numpy as np


class AdamW:
    """AdamW (decoupled weight decay) over a list of (param, grad) array pairs.

    State is kept per array; `state_dict` / `load_state_dict` round-trip it
    exactly (used for checkpoint/resume tests).
    """

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.params = list(params)  # list of np.ndarray (updated in place)
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        b1t = 1 - self.b1 ** self.t
        b2t = 1 - self.b2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            if g is None:
                continue
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * (g * g)
            p -= self.lr * (self.wd * p)
            p -= self.lr * (m / b1t) / (np.sqrt(v / b2t) + self.eps)

    def state_bytes(self):
        return sum(a.nbytes for a in self.m) + sum(a.nbytes for a in self.v)

    def state_dict(self):
        return {"t": self.t, "m": [a.copy() for a in self.m], "v": [a.copy() for a in self.v]}

    def load_state_dict(self, s):
        self.t = int(s["t"])
        for dst, src in zip(self.m, s["m"]):
            dst[...] = src
        for dst, src in zip(self.v, s["v"]):
            dst[...] = src


def clip_scale(total_sq_norm: float, max_norm: float) -> float:
    """Factor to multiply gradients by so that their global L2 norm is <= max_norm."""
    n = float(np.sqrt(total_sq_norm))
    return 1.0 if n <= max_norm or n == 0 else max_norm / (n + 1e-6)
