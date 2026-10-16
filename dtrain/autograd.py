"""A small reverse-mode autograd engine over NumPy arrays.

Only what a GPT-style model needs, but each op has an analytic backward that
is checked against finite differences in tests/test_autograd.py.

Gradient-ready hooks: a leaf tensor (parameter) can register a hook that
fires as soon as its gradient is final during `backward()`, i.e. once every
use of the leaf in the graph has propagated into it. Distributed data
parallel uses this to start all-reducing gradient buckets while the rest of
the backward pass is still running.
"""
from __future__ import annotations

import numpy as np

_grad_enabled = True


class no_grad:
    def __enter__(self):
        global _grad_enabled
        self.prev, _grad_enabled = _grad_enabled, False

    def __exit__(self, *a):
        global _grad_enabled
        _grad_enabled = self.prev


def _unbroadcast(g: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum `g` over the axes that were broadcast to produce it from `shape`."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward", "_hooks", "name", "__weakref__")

    def __init__(self, data, requires_grad=False, _parents=(), _backward=None, name=None):
        self.data = data if isinstance(data, np.ndarray) else np.asarray(data)
        self.grad = None
        self.requires_grad = requires_grad
        self._parents = _parents
        self._backward = _backward
        self._hooks = []
        self.name = name

    # ------------------------------------------------------------ utilities
    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def numpy(self):
        return self.data

    def register_grad_ready_hook(self, fn):
        """fn(tensor) is called once this leaf's gradient is complete."""
        self._hooks.append(fn)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, requires_grad={self.requires_grad})"

    @staticmethod
    def _make(data, parents, backward):
        rg = _grad_enabled and any(p.requires_grad for p in parents)
        return Tensor(data, rg, parents if rg else (), backward if rg else None)

    # ---------------------------------------------------------------- ops
    def __add__(self, o):
        o = o if isinstance(o, Tensor) else Tensor(np.asarray(o, dtype=self.dtype))
        a, b = self, o

        def bw(g):
            return _unbroadcast(g, a.shape), _unbroadcast(g, b.shape)
        return Tensor._make(a.data + b.data, (a, b), bw)

    __radd__ = __add__

    def __neg__(self):
        return Tensor._make(-self.data, (self,), lambda g: (-g,))

    def __sub__(self, o):
        return self + (-(o if isinstance(o, Tensor) else Tensor(np.asarray(o, dtype=self.dtype))))

    def __mul__(self, o):
        o = o if isinstance(o, Tensor) else Tensor(np.asarray(o, dtype=self.dtype))
        a, b = self, o

        def bw(g):
            return _unbroadcast(g * b.data, a.shape), _unbroadcast(g * a.data, b.shape)
        return Tensor._make(a.data * b.data, (a, b), bw)

    __rmul__ = __mul__

    def __matmul__(self, o):
        a, b = self, o

        def bw(g):
            ga = g @ np.swapaxes(b.data, -1, -2)
            gb = np.swapaxes(a.data, -1, -2) @ g
            return _unbroadcast(ga, a.shape), _unbroadcast(gb, b.shape)
        return Tensor._make(a.data @ b.data, (a, b), bw)

    def sum(self, axis=None, keepdims=False):
        a = self

        def bw(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return (np.broadcast_to(g, a.shape).copy(),)
        return Tensor._make(a.data.sum(axis=axis, keepdims=keepdims), (a,), bw)

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis, keepdims) * (1.0 / n)

    def reshape(self, *shape):
        a = self
        return Tensor._make(a.data.reshape(*shape), (a,), lambda g: (g.reshape(a.shape),))

    def transpose(self, *axes):
        a = self
        inv = np.argsort(axes)
        return Tensor._make(a.data.transpose(*axes), (a,), lambda g: (g.transpose(*inv),))

    def gelu(self):
        # tanh approximation (as in GPT-2)
        x = self.data
        c = np.sqrt(2.0 / np.pi).astype(x.dtype)
        u = c * (x + 0.044715 * x ** 3)
        t = np.tanh(u)
        y = 0.5 * x * (1.0 + t)

        def bw(g):
            du = c * (1.0 + 3 * 0.044715 * x ** 2)
            return (g * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du),)
        return Tensor._make(y, (self,), bw)

    def relu(self):
        m = self.data > 0
        return Tensor._make(self.data * m, (self,), lambda g: (g * m,))

    # ------------------------------------------------------------ backward
    def backward(self, grad=None):
        if grad is None:
            if self.data.size != 1:
                raise ValueError("backward() without a gradient needs a scalar output")
            grad = np.ones_like(self.data)
        # topological order + number of consumers of each node in this graph
        order, seen, users = [], set(), {}
        stack = [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                users[id(p)] = users.get(id(p), 0) + 1
                if id(p) not in seen:
                    stack.append((p, False))
        grads = {id(self): grad}
        pending = dict(users)
        for node in reversed(order):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if node._backward is None:
                # leaf: accumulate into .grad
                if node.requires_grad:
                    node.grad = g if node.grad is None else node.grad + g
                    if node._hooks:
                        for h in node._hooks:
                            h(node)
                continue
            for p, pg in zip(node._parents, node._backward(g)):
                if not p.requires_grad or pg is None:
                    pending[id(p)] -= 1
                    continue
                if pg.shape != p.shape:
                    raise RuntimeError(f"gradient shape {pg.shape} != {p.shape}")
                k = id(p)
                grads[k] = pg if k not in grads else grads[k] + pg
                pending[k] -= 1
        return None


# ------------------------------------------------------------ functional ops
def embedding(weight: Tensor, idx: np.ndarray) -> Tensor:
    def bw(g):
        gw = np.zeros_like(weight.data)
        np.add.at(gw, idx.reshape(-1), g.reshape(-1, weight.shape[1]))
        return (gw,)
    return Tensor._make(weight.data[idx], (weight,), bw)


def layer_norm(x: Tensor, w: Tensor, b: Tensor, eps=1e-5) -> Tensor:
    xd = x.data
    mu = xd.mean(-1, keepdims=True)
    xc = xd - mu
    var = (xc * xc).mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xh = xc * rstd
    y = xh * w.data + b.data

    def bw(g):
        n = xd.shape[-1]
        gxh = g * w.data
        gx = rstd * (gxh - gxh.mean(-1, keepdims=True) - xh * (gxh * xh).mean(-1, keepdims=True))
        gw = (g * xh).reshape(-1, n).sum(0)
        gb = g.reshape(-1, n).sum(0)
        return gx, gw, gb
    return Tensor._make(y, (x, w, b), bw)


def causal_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """softmax(q k^T / sqrt(d) + causal mask) v for (B, H, T, D) inputs."""
    d = q.shape[-1]
    T = q.shape[-2]
    scale = 1.0 / np.sqrt(d)
    s = (q.data @ np.swapaxes(k.data, -1, -2)) * scale
    mask = np.triu(np.ones((T, T), dtype=bool), 1)
    s = np.where(mask, -np.inf, s)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    out = p @ v.data

    def bw(g):
        gv = np.swapaxes(p, -1, -2) @ g
        gp = g @ np.swapaxes(v.data, -1, -2)
        gs = p * (gp - (gp * p).sum(-1, keepdims=True))
        gs *= scale
        gq = gs @ k.data
        gk = np.swapaxes(gs, -1, -2) @ q.data
        return gq, gk, gv
    return Tensor._make(out.astype(q.dtype, copy=False), (q, k, v), bw)


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean cross-entropy over all positions; logits (..., V), targets (...)."""
    z = logits.data
    zmax = z.max(-1, keepdims=True)
    e = np.exp(z - zmax)
    s = e.sum(-1, keepdims=True)
    logp = (z - zmax) - np.log(s)
    flat = logp.reshape(-1, z.shape[-1])
    t = targets.reshape(-1)
    n = t.shape[0]
    loss = -flat[np.arange(n), t].mean()

    def bw(g):
        p = (e / s).reshape(-1, z.shape[-1]).copy()
        p[np.arange(n), t] -= 1.0
        return ((p * (g / n)).reshape(z.shape).astype(z.dtype, copy=False),)
    return Tensor._make(np.asarray(loss, dtype=z.dtype), (logits,), bw)
