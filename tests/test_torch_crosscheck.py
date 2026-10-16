"""dtrain against torch.distributed (gloo, CPU). Skipped when PyTorch is not installed.

Each test starts real processes through the dtrain launcher; every rank runs
dtrain and PyTorch side by side on identical weights and batches (see
benchmarks/torch_crosscheck.py). float64 throughout, so differences come only
from floating-point reassociation and must stay at the 1e-15 .. 1e-13 level.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
    pytest.skip("torch.distributed with gloo is not available", allow_module_level=True)

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMALL = ["--dim", "32", "--heads", "2", "--layers", "2", "--ctx", "16", "--batch", "2", "--dtype", "float64"]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    rng = np.random.default_rng(0)
    words = ["the", "king", "and", "queen", "of", "rome", "said", "to", "thee", "now"]
    p = tmp_path_factory.mktemp("data") / "corpus.txt"
    p.write_text(" ".join(rng.choice(words, 4000)) + "\n")
    return str(p)


def launch(tmp, nproc, script, args, tag):
    cmd = [PY, "-m", "dtrain.launch", "--nproc", str(nproc), "--timeout", "60", "--",
           PY, os.path.join(ROOT, "benchmarks", script), "--torch-init", str(tmp / f"gloo_{tag}"), *args]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def xcheck(tmp, corpus, nproc, tag, *extra):
    out = tmp / tag
    launch(tmp, nproc, "torch_crosscheck.py", ["--data", corpus, *SMALL, "--out", str(out), *extra], tag)
    rows = [json.loads(l) for l in open(str(out) + ".jsonl")]
    meta = json.load(open(str(out) + ".meta.json"))
    return rows, meta, np.load(str(out) + ".npz")


@pytest.mark.parametrize("world", [2, 4])
def test_ring_allreduce_vs_gloo(tmp_path, world):
    f = tmp_path / "ar.json"
    launch(tmp_path, world, "torch_allreduce.py", ["--no-speed", "--n-check", "20000", "--out", str(f)], "ar")
    chk = json.load(open(f))["check"]
    for dt, ulp in (("float32", 2.0 ** -23), ("float64", 2.0 ** -52)):
        c = chk[dt]
        bound = 2 * (world - 1) * ulp * c["max_abs_value"]  # textbook bound for a (W-1)-term recursive sum
        assert c["max_abs_err_dtrain"] <= bound and c["max_abs_err_gloo"] <= bound
        if world == 2:  # a + b == b + a exactly: any order gives the same bits
            assert c["bit_identical_frac"] == 1.0


@pytest.mark.parametrize("opt", ["adamw", "sgd"])
def test_ddp_matches_torch_ddp(tmp_path, corpus, opt):
    rows, _, _ = xcheck(tmp_path, corpus, 2, f"ddp_{opt}", "--mode", "ddp", "--steps", "5", "--opt", opt)
    assert len(rows) == 5
    for r in rows:
        assert r["grad_maxabs_diff"] <= 1e-14 * max(1.0, r["grad_maxabs"])
        assert r["param_maxabs_diff"] <= 1e-13
        assert r["loss_abs_diff"] <= 1e-13


def test_zero1_matches_torch_zero_redundancy_optimizer(tmp_path, corpus):
    rows, meta, _ = xcheck(tmp_path, corpus, 2, "zero", "--mode", "zero", "--steps", "5")
    for r in rows:
        assert r["param_maxabs_diff"] <= 1e-13
    sb = meta["state_bytes_per_rank"]
    n = meta["n_params"]
    # both shard AdamW state (2 float64 buffers per element) and together hold exactly one copy
    assert sum(s["torch_state_bytes"] for s in sb) == 16 * n
    assert sum(s["dtrain_state_bytes"] for s in sb) == 16 * n


@pytest.mark.parametrize("mode", ["ddp", "zero"])
def test_checkpoint_resume_bit_identical_in_both(tmp_path, corpus, mode):
    common = ["--mode", mode, "--steps", "6", "--k", "3"]
    _, _, full = xcheck(tmp_path, corpus, 2, f"{mode}_full", *common, "--phase", "full")
    ck = str(tmp_path / f"ck_{mode}")
    xcheck(tmp_path, corpus, 2, f"{mode}_first", *common, "--phase", "first", "--ckpt", ck)
    _, _, res = xcheck(tmp_path, corpus, 2, f"{mode}_res", *common, "--phase", "resume", "--ckpt", ck)
    assert np.array_equal(full["dtrain"], res["dtrain"])
    assert np.array_equal(full["torch"], res["torch"])
    assert np.max(np.abs(res["dtrain"] - res["torch"])) <= 1e-13


def test_adamw_matches_torch_adamw_on_identical_gradients():
    from dtrain.optim import AdamW
    rng = np.random.default_rng(0)
    p0 = rng.standard_normal(5000)
    pd, pt = p0.copy(), torch.from_numpy(p0.copy()).requires_grad_()
    od = AdamW([pd], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    ot = torch.optim.AdamW([pt], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    for _ in range(10):
        g = rng.standard_normal(p0.size) * 10.0 ** rng.uniform(-6, 0, p0.size)
        od.step([g])
        pt.grad = torch.from_numpy(g.copy())
        ot.step()
    assert np.max(np.abs(pd - pt.detach().numpy())) <= 1e-14
