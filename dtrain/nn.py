"""Modules: Linear, Embedding, LayerNorm, a GPT-style transformer."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .autograd import Tensor, causal_attention, cross_entropy, embedding, layer_norm


class Module:
    def parameters(self):
        """All parameters in a deterministic order (attribute definition order, depth-first)."""
        out, seen = [], set()

        def walk(m):
            for v in m.__dict__.values():
                if isinstance(v, Tensor) and v.requires_grad and id(v) not in seen:
                    seen.add(id(v))
                    out.append(v)
                elif isinstance(v, Module):
                    walk(v)
                elif isinstance(v, (list, tuple)):
                    for x in v:
                        if isinstance(x, Module):
                            walk(x)
        walk(self)
        return out

    def named_parameters(self):
        names = {}

        def walk(m, prefix):
            for k, v in m.__dict__.items():
                if isinstance(v, Tensor) and v.requires_grad:
                    names.setdefault(id(v), prefix + k)
                elif isinstance(v, Module):
                    walk(v, prefix + k + ".")
                elif isinstance(v, (list, tuple)):
                    for i, x in enumerate(v):
                        if isinstance(x, Module):
                            walk(x, f"{prefix}{k}.{i}.")
        walk(self, "")
        return [(names[id(p)], p) for p in self.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


def param(arr, name=None):
    return Tensor(arr, requires_grad=True, name=name)


class Linear(Module):
    def __init__(self, nin, nout, rng, dtype=np.float32, bias=True, std=0.02):
        self.w = param((rng.standard_normal((nin, nout)) * std).astype(dtype))
        self.b = param(np.zeros(nout, dtype)) if bias else None

    def forward(self, x):
        y = x @ self.w
        return y + self.b if self.b is not None else y


class LayerNorm(Module):
    def __init__(self, n, dtype=np.float32):
        self.w = param(np.ones(n, dtype))
        self.b = param(np.zeros(n, dtype))

    def forward(self, x):
        return layer_norm(x, self.w, self.b)


@dataclass
class GPTConfig:
    vocab: int = 65
    ctx: int = 64
    dim: int = 128
    heads: int = 4
    layers: int = 4
    dtype: str = "float32"


class Block(Module):
    def __init__(self, c: GPTConfig, rng):
        dt = np.dtype(c.dtype)
        self.c = c
        self.ln1 = LayerNorm(c.dim, dt)
        self.qkv = Linear(c.dim, 3 * c.dim, rng, dt)
        self.proj = Linear(c.dim, c.dim, rng, dt, std=0.02 / np.sqrt(2 * c.layers))
        self.ln2 = LayerNorm(c.dim, dt)
        self.fc = Linear(c.dim, 4 * c.dim, rng, dt)
        self.out = Linear(4 * c.dim, c.dim, rng, dt, std=0.02 / np.sqrt(2 * c.layers))

    def forward(self, x):
        B, T, C = x.shape
        H = self.c.heads
        qkv = self.qkv(self.ln1(x)).reshape(B, T, 3, H, C // H).transpose(2, 0, 3, 1, 4)
        q, k, v = _split3(qkv)
        a = causal_attention(q, k, v).transpose(0, 2, 1, 3).reshape(B, T, C)
        x = x + self.proj(a)
        return x + self.out(self.fc(self.ln2(x)).gelu())


def _split3(t: Tensor):
    """Split the leading axis (size 3) into three tensors."""
    outs = []
    for i in range(3):
        def bw(g, i=i):
            full = np.zeros_like(t.data)
            full[i] = g
            return (full,)
        outs.append(Tensor._make(t.data[i], (t,), bw))
    return outs


class GPT(Module):
    def __init__(self, c: GPTConfig, seed=0):
        rng = np.random.default_rng(seed)
        dt = np.dtype(c.dtype)
        self.c = c
        self.tok = param((rng.standard_normal((c.vocab, c.dim)) * 0.02).astype(dt))
        self.pos = param((rng.standard_normal((c.ctx, c.dim)) * 0.01).astype(dt))
        self.blocks = [Block(c, rng) for _ in range(c.layers)]
        self.ln = LayerNorm(c.dim, dt)
        self.head = Linear(c.dim, c.vocab, rng, dt, bias=False)

    def forward(self, idx: np.ndarray, targets: np.ndarray | None = None):
        B, T = idx.shape
        x = embedding(self.tok, idx) + embedding(self.pos, np.arange(T))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln(x))
        if targets is None:
            return logits
        return logits, cross_entropy(logits, targets)
