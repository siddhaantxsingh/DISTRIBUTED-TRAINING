"""ZeRO-style sharded optimizer (stage 1, with stage-2 style gradient reduce-scatter).

All parameters are flattened into one contiguous buffer and the model's
parameter arrays become views into it. The buffer is partitioned across
ranks (`comm.partition`). Each step:

1. the full local gradient is flattened and *reduce-scattered*, so each rank
   receives only the summed gradient of its own shard (half the traffic of
   an all-reduce);
2. optional global-norm clipping (sum of squared shard norms, all-reduced);
3. each rank runs AdamW on its shard only, so optimizer state (m, v) is
   1/W of the replicated size;
4. the updated shards are *all-gathered* back into the full parameter buffer.
"""
from __future__ import annotations

import numpy as np

from .comm import ProcessGroup, partition
from .optim import AdamW, clip_scale


class ZeroAdamW:
    def __init__(self, model, pg: ProcessGroup, max_grad_norm: float | None = None, **adam):
        self.pg = pg
        self.params = model.parameters()
        n = sum(p.data.size for p in self.params)
        dtype = self.params[0].data.dtype
        self.flat = np.empty(n, dtype)
        off = 0
        for p in self.params:
            k = p.data.size
            self.flat[off:off + k] = p.data.reshape(-1)
            p.data = self.flat[off:off + k].reshape(p.data.shape)  # view into the flat buffer
            off += k
        pg.broadcast(self.flat, root=0)
        self.bounds = partition(n, pg.world)
        lo, hi = self.bounds[pg.rank], self.bounds[pg.rank + 1]
        self.shard = self.flat[lo:hi]
        self.gflat = np.empty(n, dtype)
        self.opt = AdamW([self.shard], **adam)
        self.max_grad_norm = max_grad_norm
        self.last_grad_norm = 0.0

    def step(self):
        off = 0
        for p in self.params:
            k = p.data.size
            if p.grad is None:
                self.gflat[off:off + k] = 0
            else:
                self.gflat[off:off + k] = p.grad.reshape(-1)
            off += k
        g = self.pg.reduce_scatter(self.gflat)
        g *= 1.0 / self.pg.world
        sq = np.array([float(np.dot(g.astype(np.float64), g.astype(np.float64)))])
        self.pg.allreduce(sq)
        self.last_grad_norm = float(np.sqrt(sq[0]))
        if self.max_grad_norm:
            g *= clip_scale(sq[0], self.max_grad_norm)
        self.opt.step([g])
        self.pg.all_gather(self.flat)

    def state_bytes(self):
        return self.opt.state_bytes()

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, s):
        self.opt.load_state_dict(s)
