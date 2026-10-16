"""Train a character-level GPT, single-process or distributed.

    python -m dtrain.train --mode single ...                 # one process
    python -m dtrain.launch --nproc 4 -- python -m dtrain.train --mode ddp ...
    python -m dtrain.launch --nproc 4 -- python -m dtrain.train --mode zero ...

`--mode single --emulate-world W` runs the exact global batch of a W-rank job
in one process (micro-batches per virtual rank, averaged), which is the
reference for the equivalence tests.

Writes one JSON line per step to --log (step, loss, grad norm, step time,
tokens/s, communication wait).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from . import checkpoint
from .autograd import Tensor
from .data import CharData, load_text, rank_batch
from .nn import GPT, GPTConfig
from .optim import AdamW, clip_scale


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "ddp", "zero"], default="single")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8, help="sequences per rank per micro-step")
    ap.add_argument("--accum", type=int, default=1, help="gradient-accumulation micro-steps")
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=0, help="linear warm-up steps, then cosine decay to lr/10")
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=None, help="text file (default: tiny shakespeare)")
    ap.add_argument("--emulate-world", type=int, default=1)
    ap.add_argument("--bucket-mb", type=float, default=1.0)
    ap.add_argument("--no-overlap", action="store_true")
    ap.add_argument("--algo", choices=["ring", "direct"], default="ring")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="resume from the latest checkpoint if any")
    ap.add_argument("--stop-after", type=int, default=-1, help="stop cleanly after this many steps (preemption)")
    ap.add_argument("--fail-at-step", type=int, default=-1, help="fault injection: crash at this step")
    ap.add_argument("--fail-rank", type=int, default=1)
    ap.add_argument("--log", default=None)
    ap.add_argument("--save-params", default=None, help="write final parameters (.npz) from rank 0")
    return ap.parse_args(argv)


def lr_at(step: int, a) -> float:
    if a.warmup and step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    if not a.warmup:
        return a.lr
    t = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(1.0, t))))


def main(argv=None):
    a = parse(argv)
    text = load_text(a.data)
    data = CharData(text)
    cfg = GPTConfig(vocab=data.vocab, ctx=a.ctx, dim=a.dim, heads=a.heads, layers=a.layers, dtype=a.dtype)
    model = GPT(cfg, seed=a.seed)
    params = model.parameters()

    pg = None
    rank, world = 0, 1
    if a.mode != "single":
        from .comm import init_from_env
        pg = init_from_env()
        rank, world = pg.rank, pg.world

    ddp = zero = opt = None
    if a.mode == "ddp":
        from .ddp import DDP
        ddp = DDP(model, pg, bucket_mb=a.bucket_mb, overlap=not a.no_overlap, algo=a.algo)
        opt = AdamW([p.data for p in params], lr=a.lr, weight_decay=a.wd)
    elif a.mode == "zero":
        from .zero import ZeroAdamW
        zero = ZeroAdamW(model, pg, max_grad_norm=a.clip, lr=a.lr, weight_decay=a.wd)
    else:
        opt = AdamW([p.data for p in params], lr=a.lr, weight_decay=a.wd)

    start = 0
    if a.resume and a.ckpt_dir and checkpoint.latest(a.ckpt_dir):
        step, st = checkpoint.load(checkpoint.latest(a.ckpt_dir), rank, world)
        if a.mode == "zero":
            zero.flat[...] = st["params"]
            zero.load_state_dict({"t": st["t"], "m": [st["m"]], "v": [st["v"]]})
        else:
            off = 0
            for p in params:
                p.data[...] = st["params"][off:off + p.data.size].reshape(p.shape)
                off += p.data.size
            n = len(params)
            opt.load_state_dict({"t": st["t"], "m": [st[f"m{i}"] for i in range(n)], "v": [st[f"v{i}"] for i in range(n)]})
        start = step
        print(f"[rank {rank}] resumed from step {step}", file=sys.stderr, flush=True)

    log = open(a.log, "a") if (a.log and rank == 0) else None
    restarts = int(os.environ.get("DTRAIN_RESTART_COUNT", "0"))
    tokens_per_step = a.batch * a.ctx * a.accum * max(world, a.emulate_world)

    for step in range(start, a.steps):
        if step == a.fail_at_step and rank == a.fail_rank and restarts == 0:
            print(f"[rank {rank}] injected failure at step {step}", file=sys.stderr, flush=True)
            os._exit(17)
        t0 = time.perf_counter()
        lr = lr_at(step, a)
        if zero is not None:
            zero.opt.lr = lr
        else:
            opt.lr = lr
        model.zero_grad()
        loss_sum = 0.0
        vworlds = a.emulate_world if a.mode == "single" else 1
        n_micro = a.accum * vworlds
        for micro in range(a.accum):
            for vr in range(vworlds):
                r = vr if a.mode == "single" else rank
                w = vworlds if a.mode == "single" else world
                x, y = rank_batch(data.train, step, micro, a.batch, a.ctx, a.seed, r, w)
                _, loss = model(x, y)
                scaled = loss * (1.0 / n_micro)
                last = micro == a.accum - 1
                if ddp is not None and not last:
                    with ddp.no_sync():
                        scaled.backward()
                else:
                    scaled.backward()
                loss_sum += float(loss.data) / n_micro
        comm_wait = 0.0
        if ddp is not None:
            before = ddp.comm_wait_s
            ddp.finish()
            comm_wait = ddp.comm_wait_s - before
        if zero is not None:
            zero.step()
            gnorm = zero.last_grad_norm
        else:
            sq = sum(float(np.dot(p.grad.reshape(-1).astype(np.float64), p.grad.reshape(-1).astype(np.float64)))
                     for p in params if p.grad is not None)
            gnorm = float(np.sqrt(sq))
            s = clip_scale(sq, a.clip) if a.clip else 1.0
            opt.step([p.grad * s if p.grad is not None else None for p in params])
        if pg is not None:
            l = np.array([loss_sum])
            pg.allreduce(l)
            loss_sum = float(l[0]) / world
        dt = time.perf_counter() - t0
        if log:
            log.write(json.dumps({"step": step + 1, "lr": lr, "loss": loss_sum, "grad_norm": gnorm, "step_s": dt,
                                  "tokens_per_s": tokens_per_step / dt, "comm_wait_s": comm_wait}) + "\n")
            log.flush()

        if a.ckpt_dir and a.ckpt_every and (step + 1) % a.ckpt_every == 0:
            barrier = pg.barrier if pg else (lambda: None)
            if a.mode == "zero":
                sd = zero.state_dict()
                st = {"params": zero.flat, "t": sd["t"], "m": sd["m"][0], "v": sd["v"][0]}
            else:
                sd = opt.state_dict()
                st = {"params": np.concatenate([p.data.reshape(-1) for p in params]), "t": sd["t"]}
                st.update({f"m{i}": m for i, m in enumerate(sd["m"])})
                st.update({f"v{i}": v for i, v in enumerate(sd["v"])})
            checkpoint.save(a.ckpt_dir, step + 1, rank, world, st, barrier)
        if a.stop_after >= 0 and step + 1 >= a.stop_after:
            break

    if a.save_params and rank == 0:
        np.savez(a.save_params, **{n: p.data for n, p in model.named_parameters()})
    if zero is not None and rank == 0 and log:
        log.write(json.dumps({"optimizer_state_bytes_per_rank": zero.state_bytes()}) + "\n")
    if pg is not None:
        pg.barrier()
        pg.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
