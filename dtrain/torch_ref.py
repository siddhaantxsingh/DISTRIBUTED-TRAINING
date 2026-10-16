"""A PyTorch copy of `dtrain.nn.GPT`, used only to cross-check dtrain against torch.distributed.

`TorchGPT.from_dtrain(model)` copies every parameter of a dtrain GPT, in the
same order and with the same shapes (Linear weights are stored (in, out) and
applied as `x @ w`, as in dtrain). The forward pass uses the same operations:
LayerNorm (eps 1e-5), causal softmax attention, tanh-approximated GELU, mean
cross-entropy. Only the floating-point kernels differ (NumPy vs ATen), which
is the source of the small differences reported by the cross-check.

PyTorch is optional: importing this module needs `torch`; nothing else in
dtrain does.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


class TorchGPT(torch.nn.Module):
    def __init__(self, cfg, named_arrays):
        super().__init__()
        self.cfg = cfg
        self.names = [n for n, _ in named_arrays]
        self.ps = torch.nn.ParameterList([torch.nn.Parameter(torch.from_numpy(np.array(a, copy=True)))
                                          for _, a in named_arrays])
        self._i = {n: i for i, n in enumerate(self.names)}

    @classmethod
    def from_dtrain(cls, model):
        return cls(model.c, [(n, p.data) for n, p in model.named_parameters()])

    def p(self, name):
        return self.ps[self._i[name]]

    def flat_params(self) -> np.ndarray:
        return np.concatenate([p.detach().numpy().reshape(-1) for p in self.ps])

    def flat_grads(self) -> np.ndarray:
        return np.concatenate([p.grad.numpy().reshape(-1) for p in self.ps])

    def forward(self, idx, targets=None):
        c = self.cfg
        idx = _long(idx)
        B, T = idx.shape
        dev = idx.device
        C, H = c.dim, c.heads
        x = self.p("tok")[idx] + self.p("pos")[torch.arange(T, device=dev)]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=dev), 1)
        for i in range(c.layers):
            pre = f"blocks.{i}."
            h = F.layer_norm(x, (C,), self.p(pre + "ln1.w"), self.p(pre + "ln1.b"), 1e-5)
            qkv = (h @ self.p(pre + "qkv.w") + self.p(pre + "qkv.b")).reshape(B, T, 3, H, C // H).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            s = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(C // H))
            s = s.masked_fill(mask, float("-inf"))
            a = torch.softmax(s, -1) @ v
            a = a.transpose(1, 2).reshape(B, T, C)
            x = x + (a @ self.p(pre + "proj.w") + self.p(pre + "proj.b"))
            h = F.layer_norm(x, (C,), self.p(pre + "ln2.w"), self.p(pre + "ln2.b"), 1e-5)
            h = F.gelu(h @ self.p(pre + "fc.w") + self.p(pre + "fc.b"), approximate="tanh")
            x = x + (h @ self.p(pre + "out.w") + self.p(pre + "out.b"))
        logits = F.layer_norm(x, (C,), self.p("ln.w"), self.p("ln.b"), 1e-5) @ self.p("head.w")
        if targets is None:
            return logits
        t = _long(targets)
        return logits, F.cross_entropy(logits.reshape(-1, logits.shape[-1]), t.reshape(-1))


def _long(a):
    """int64 tensor from a NumPy array or a tensor (kept on its device)."""
    return a.long() if torch.is_tensor(a) else torch.as_tensor(np.asarray(a, dtype=np.int64))
