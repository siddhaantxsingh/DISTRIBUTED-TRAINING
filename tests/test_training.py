"""End-to-end distributed training through the launcher (real processes).

All runs use float64 and a tiny model so that equivalence can be checked
tightly; the corpus is generated (no network).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

PY = sys.executable
TINY = ["--dim", "16", "--heads", "2", "--layers", "2", "--ctx", "16", "--batch", "2", "--dtype", "float64",
        "--lr", "3e-3", "--warmup", "2"]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    rng = np.random.default_rng(0)
    words = ["the", "king", "and", "queen", "of", "rome", "said", "to", "thee", "now"]
    text = " ".join(rng.choice(words, 4000)) + "\n"
    p = tmp_path_factory.mktemp("data") / "corpus.txt"
    p.write_text(text)
    return str(p)


def train(tmp, name, corpus, mode, nproc=1, extra=(), restarts=0, check=True):
    out = tmp / f"{name}.npz"
    log = tmp / f"{name}.jsonl"
    args = ["-m", "dtrain.train", "--mode", mode, "--data", corpus, "--save-params", str(out), "--log", str(log),
            *TINY, *extra]
    if mode == "single":
        cmd = [PY, *args]
    else:
        cmd = [PY, "-m", "dtrain.launch", "--nproc", str(nproc), "--max-restarts", str(restarts), "--timeout", "30",
               "--", PY, *args]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    if check:
        assert r.returncode == 0, r.stderr[-3000:]
    params = dict(np.load(out)) if out.exists() else None
    logs = [json.loads(l) for l in open(log)] if log.exists() else []
    return params, logs, r


def assert_close(a, b, tol):
    assert a.keys() == b.keys()
    worst = max(float(np.max(np.abs(a[k] - b[k]))) for k in a)
    assert worst <= tol, f"max abs param difference {worst:g} > {tol:g}"


STEPS = ["--steps", "6"]


def test_ddp_matches_single_process(tmp_path, corpus):
    ref, ref_log, _ = train(tmp_path, "single", corpus, "single", extra=[*STEPS, "--emulate-world", "4"])
    ddp, ddp_log, _ = train(tmp_path, "ddp", corpus, "ddp", 4, extra=STEPS)
    assert_close(ddp, ref, 1e-10)
    for a, b in zip(ref_log, ddp_log):
        assert abs(a["loss"] - b["loss"]) < 1e-10


def test_zero_matches_ddp_and_shards_state(tmp_path, corpus):
    ddp, _, _ = train(tmp_path, "ddp", corpus, "ddp", 4, extra=STEPS)
    zero, zlog, _ = train(tmp_path, "zero", corpus, "zero", 4, extra=STEPS)
    assert_close(zero, ddp, 1e-10)
    nparams = sum(v.size for v in ddp.values())
    state = [l for l in zlog if "optimizer_state_bytes_per_rank" in l][0]["optimizer_state_bytes_per_rank"]
    # rank 0 holds m and v (float64) for ceil(n / 4) elements
    assert state == 2 * 8 * -(-nparams // 4)


def test_overlap_bucket_size_and_algorithm_do_not_change_results(tmp_path, corpus):
    a, _, _ = train(tmp_path, "a", corpus, "ddp", 3, extra=[*STEPS, "--bucket-mb", "0.001"])
    b, _, _ = train(tmp_path, "b", corpus, "ddp", 3, extra=[*STEPS, "--no-overlap", "--bucket-mb", "100"])
    c, _, _ = train(tmp_path, "c", corpus, "ddp", 3, extra=[*STEPS, "--algo", "direct"])
    assert_close(a, b, 1e-12)
    assert_close(a, c, 1e-12)


def test_gradient_accumulation_matches_larger_batch(tmp_path, corpus):
    # 2 ranks x 2 micro-steps x batch 2 == single process emulating 2 ranks with accum 2
    ref, _, _ = train(tmp_path, "ref", corpus, "single", extra=[*STEPS, "--emulate-world", "2", "--accum", "2"])
    ddp, _, _ = train(tmp_path, "acc", corpus, "ddp", 2, extra=[*STEPS, "--accum", "2"])
    assert_close(ddp, ref, 1e-10)


@pytest.mark.parametrize("mode", ["ddp", "zero"])
def test_checkpoint_resume_is_exact(tmp_path, corpus, mode):
    full, _, _ = train(tmp_path, f"{mode}-full", corpus, mode, 2, extra=["--steps", "8"])
    ck = tmp_path / f"ck-{mode}"
    train(tmp_path, f"{mode}-part", corpus, mode, 2,
          extra=["--steps", "8", "--stop-after", "4", "--ckpt-dir", str(ck), "--ckpt-every", "2"])
    resumed, _, r = train(tmp_path, f"{mode}-res", corpus, mode, 2,
                          extra=["--steps", "8", "--ckpt-dir", str(ck), "--ckpt-every", "2", "--resume"])
    assert "resumed from step 4" in r.stderr
    assert_close(resumed, full, 0.0)  # bit-identical


@pytest.mark.parametrize("mode", ["ddp", "zero"])
def test_worker_crash_restart_resume_matches_failure_free_run(tmp_path, corpus, mode):
    clean, _, _ = train(tmp_path, f"{mode}-clean", corpus, mode, 3, extra=["--steps", "9"])
    ck = tmp_path / f"ck-fail-{mode}"
    crashed, _, r = train(tmp_path, f"{mode}-crash", corpus, mode, 3, restarts=2,
                          extra=["--steps", "9", "--ckpt-dir", str(ck), "--ckpt-every", "3", "--resume",
                                 "--fail-at-step", "5", "--fail-rank", "1"])
    assert "injected failure at step 5" in r.stderr
    assert "restarting" in r.stderr and "resumed from step 3" in r.stderr
    assert_close(crashed, clean, 0.0)


def test_no_restarts_left_propagates_failure(tmp_path, corpus):
    _, _, r = train(tmp_path, "fail", corpus, "ddp", 2, restarts=0, check=False,
                    extra=["--steps", "4", "--fail-at-step", "1", "--fail-rank", "0"])
    assert r.returncode == 17


def test_corrupt_checkpoint_is_rejected(tmp_path, corpus):
    ck = tmp_path / "ck-bad"
    train(tmp_path, "w", corpus, "ddp", 2, extra=["--steps", "2", "--ckpt-dir", str(ck), "--ckpt-every", "2"])
    shard = ck / "step_000002" / "rank1.npz"
    b = bytearray(shard.read_bytes())
    b[len(b) // 2] ^= 0xFF
    shard.write_bytes(bytes(b))
    _, _, r = train(tmp_path, "r", corpus, "ddp", 2, check=False,
                    extra=["--steps", "4", "--ckpt-dir", str(ck), "--resume"])
    assert r.returncode != 0 and "checksum mismatch" in r.stderr
