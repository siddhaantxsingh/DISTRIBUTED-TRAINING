"""Distributed data parallel with gradient bucketing and comm/compute overlap.

* Parameters are broadcast from rank 0 at construction, so all replicas
  start identical.
* Parameters are grouped into buckets of about `bucket_mb` MiB, in reverse
  definition order (roughly the order in which backward produces their
  gradients).
* A grad-ready hook on every parameter copies its gradient into the bucket
  buffer. When a bucket is complete it is all-reduced asynchronously on the
  communication thread (`overlap=True`) while backward continues; or all
  buckets are reduced after backward (`overlap=False`).
* `no_sync()` skips communication for gradient-accumulation micro-steps:
  gradients accumulate locally and are reduced once, on the last micro-step.
* `finish()` waits for all reductions, divides by the world size and writes
  the averaged gradients back to `param.grad`.
"""
from __future__ import annotations

import contextlib
import time

import numpy as np

from .comm import ProcessGroup


class _Bucket:
    def __init__(self, params, dtype):
        self.params = params
        self.offsets = []
        off = 0
        for p in params:
            self.offsets.append(off)
            off += p.data.size
        self.buf = np.zeros(off, dtype)
        self.pending = len(params)
        self.work = None


class DDP:
    def __init__(self, model, pg: ProcessGroup, bucket_mb: float = 1.0, overlap: bool = True, algo: str = "ring"):
        self.model, self.pg, self.overlap, self.algo = model, pg, overlap, algo
        self.params = model.parameters()
        for p in self.params:
            pg.broadcast(p.data, root=0)
        dtype = self.params[0].data.dtype
        limit = int(bucket_mb * (1 << 20))
        self.buckets, cur, size = [], [], 0
        for p in reversed(self.params):
            cur.append(p)
            size += p.data.nbytes
            if size >= limit:
                self.buckets.append(_Bucket(cur, dtype))
                cur, size = [], 0
        if cur:
            self.buckets.append(_Bucket(cur, dtype))
        self._where = {}
        for bi, b in enumerate(self.buckets):
            for pi, p in enumerate(b.params):
                self._where[id(p)] = (bi, pi)
                p.register_grad_ready_hook(self._on_grad)
        self._sync = True
        self.comm_wait_s = 0.0

    @contextlib.contextmanager
    def no_sync(self):
        prev, self._sync = self._sync, False
        try:
            yield
        finally:
            self._sync = prev

    def _on_grad(self, p):
        if not self._sync:
            return
        bi, pi = self._where[id(p)]
        b = self.buckets[bi]
        off = b.offsets[pi]
        b.buf[off:off + p.data.size] = p.grad.reshape(-1)
        b.pending -= 1
        if b.pending == 0 and self.overlap:
            b.work = self.pg.allreduce_async(b.buf, self.algo)

    def finish(self):
        """Complete gradient synchronisation; call after the last backward of a step."""
        t0 = time.perf_counter()
        for b in self.buckets:
            if b.pending:
                # parameters that received no gradient this step: contribute zeros
                for pi, p in enumerate(b.params):
                    if p.grad is None:
                        off = b.offsets[pi]
                        b.buf[off:off + p.data.size] = 0
                b.pending = 0
                if self.overlap:
                    b.work = self.pg.allreduce_async(b.buf, self.algo)
            if not self.overlap:
                self.pg.allreduce(b.buf, self.algo)
        for b in self.buckets:
            if b.work is not None:
                b.work.wait()
                b.work = None
        self.comm_wait_s += time.perf_counter() - t0
        inv = 1.0 / self.pg.world
        for b in self.buckets:
            b.buf *= inv
            for pi, p in enumerate(b.params):
                off = b.offsets[pi]
                p.grad = b.buf[off:off + p.data.size].reshape(p.data.shape).copy()
            b.pending = len(b.params)
