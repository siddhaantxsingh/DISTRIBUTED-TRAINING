"""Run the whole dtrain-vs-PyTorch cross-check and benchmark, and write a summary.

    python benchmarks/run_torch_crosscheck.py --out benchmarks/results/2026-10-15/torch [--quick] [--skip-bench]

Needs an interpreter with torch (CPU build is enough) and numpy; run it from
the project root. Everything runs through `python -m dtrain.launch` on this
machine (CPU, gloo over loopback). Writes raw per-run files plus
`summary.json` to --out; `torch_report.py` renders the summary as Markdown.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def launch(nproc, script, args, tmp, timeout=1800):
    init = tempfile.mktemp(prefix="gloo_", dir=tmp)
    cmd = [PY, "-m", "dtrain.launch", "--nproc", str(nproc), "--timeout", "120", "--",
           PY, os.path.join(HERE, script), "--torch-init", init, *args]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")


def rows(path):
    return [json.loads(l) for l in open(path)]


def load_avg():
    return open("/proc/loadavg").read().split()[:3]


def adamw_only(steps):
    import torch
    from dtrain.optim import AdamW
    out = {}
    for dt in ("float64", "float32"):
        rng = np.random.default_rng(1)
        p0 = rng.standard_normal(100_000).astype(dt)
        pd, pt = p0.copy(), torch.from_numpy(p0.copy()).requires_grad_()
        od = AdamW([pd], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
        ot = torch.optim.AdamW([pt], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
        diffs = []
        for _ in range(steps):
            # gradients spanning 1e-12 .. 1, so some are far below eps and some far above
            g = (rng.standard_normal(p0.size) * 10.0 ** rng.uniform(-12, 0, p0.size)).astype(dt)
            od.step([g])
            pt.grad = torch.from_numpy(g.copy())
            ot.step()
            diffs.append(float(np.max(np.abs(pd - pt.detach().numpy()))))
        out[dt] = {"param_maxabs_diff_per_step": diffs, "param_maxabs_diff_max": max(diffs)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="fewer steps (used by the opt-in test)")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="xc_")
    steps = 8 if a.quick else 50
    summ = {"steps": steps, "load_before": load_avg(), "equivalence": [], "resume": [], "collective": [],
            "throughput": [], "allreduce_speed": []}
    import torch
    summ["torch"] = torch.__version__
    summ["numpy"] = np.__version__

    # 0. optimizer alone: identical parameters and gradients into dtrain AdamW and torch.optim.AdamW
    summ["adamw_only"] = adamw_only(steps)

    # 1. collectives alone: identical inputs through dtrain ring and gloo
    for w in (2, 3, 4):
        f = os.path.join(a.out, f"allreduce_check_w{w}.json")
        launch(w, "torch_allreduce.py", ["--no-speed", "--out", f], tmp)
        summ["collective"].append({"world": w, **json.load(open(f))["check"]})

    # 2. DDP and ZeRO-1 step by step against torch DDP / ZeroRedundancyOptimizer
    for mode, worlds in (("ddp", (1, 2, 4)), ("ddp-sgd", (1, 2, 4)), ("zero", (2, 4))):
        for dt in ("float64", "float32"):
            for w in worlds:
                name = f"{mode}_w{w}_{dt}"
                p = os.path.join(a.out, name)
                extra = ["--mode", "ddp", "--opt", "sgd", "--lr", "0.1"] if mode == "ddp-sgd" else ["--mode", mode]
                launch(w, "torch_crosscheck.py", [*extra, "--steps", str(steps), "--dtype", dt, "--out", p], tmp)
                rs = rows(p + ".jsonl")
                meta = json.load(open(p + ".meta.json"))
                e = {"mode": mode, "world": w, "dtype": dt, "steps": len(rs), "n_params": meta["n_params"],
                     "param_maxabs_diff_per_step": [r["param_maxabs_diff"] for r in rs],
                     "param_maxabs_diff_max": max(r["param_maxabs_diff"] for r in rs),
                     "loss_abs_diff_max": max(r["loss_abs_diff"] for r in rs),
                     "final_loss_dtrain": rs[-1]["dtrain_loss"], "final_loss_torch": rs[-1]["torch_loss"],
                     "state_bytes_per_rank": meta["state_bytes_per_rank"]}
                if mode != "zero":
                    e["grad_maxabs_diff_per_step"] = [r["grad_maxabs_diff"] for r in rs]
                    e["grad_maxabs_diff_max"] = max(r["grad_maxabs_diff"] for r in rs)
                    e["grad_maxabs"] = max(r["grad_maxabs"] for r in rs)
                    e["grad_bit_identical_frac_step1"] = rs[0]["grad_bit_identical_frac"]
                summ["equivalence"].append(e)

    # 3. checkpoint / resume in new processes vs an uninterrupted run, both libraries
    rsteps, k = (6, 3) if a.quick else (20, 10)
    for mode in ("ddp", "zero"):
        for dt in ("float64", "float32"):
            w = 4 if not a.quick else 2
            base = os.path.join(a.out, f"resume_{mode}_w{w}_{dt}")
            ck = os.path.join(tmp, f"ck_{mode}_{dt}")
            common = ["--mode", mode, "--steps", str(rsteps), "--k", str(k), "--dtype", dt]
            launch(w, "torch_crosscheck.py", [*common, "--phase", "full", "--out", base + "_full"], tmp)
            launch(w, "torch_crosscheck.py", [*common, "--phase", "first", "--ckpt", ck, "--out", base + "_first"], tmp)
            launch(w, "torch_crosscheck.py", [*common, "--phase", "resume", "--ckpt", ck, "--out", base + "_resumed"], tmp)
            full, res = np.load(base + "_full.npz"), np.load(base + "_resumed.npz")
            summ["resume"].append({
                "mode": mode, "world": w, "dtype": dt, "steps": rsteps, "ckpt_step": k,
                "dtrain_resumed_bit_identical": bool(np.array_equal(full["dtrain"], res["dtrain"])),
                "torch_resumed_bit_identical": bool(np.array_equal(full["torch"], res["torch"])),
                "dtrain_vs_torch_full_maxabs": float(np.max(np.abs(full["dtrain"] - full["torch"]))),
                "dtrain_vs_torch_resumed_maxabs": float(np.max(np.abs(res["dtrain"] - res["torch"]))),
            })
            for suffix in ("_full.npz", "_first.npz", "_resumed.npz"):
                os.remove(base + suffix)  # keep the logs, not the parameter dumps

    for e in summ["equivalence"]:
        os.remove(os.path.join(a.out, f"{e['mode']}_w{e['world']}_{e['dtype']}.npz"))

    # 4. throughput: same float32 GPT as docs/benchmarks.md, one implementation per run
    if not a.skip_bench:
        tp_args = ["--mode", "ddp", "--steps", "30", "--dtype", "float32", "--dim", "128", "--layers", "4",
                   "--heads", "4", "--ctx", "64", "--batch", "8"]
        for rep in range(a.repeats):
            for w in (1, 2, 4):
                for impl in ("dtrain", "torch"):
                    p = os.path.join(a.out, f"tp_{impl}_w{w}_r{rep}")
                    la = load_avg()
                    launch(w, "torch_crosscheck.py", [*tp_args, "--only", impl, "--out", p], tmp)
                    rs = rows(p + ".jsonl")[5:]
                    s = float(np.median([r[f"{impl}_s"] for r in rs]))
                    summ["throughput"].append({"impl": impl, "world": w, "repeat": rep, "median_step_s": s,
                                               "tokens_per_s": 8 * 64 * w / s, "load_before": la,
                                               "load_after": load_avg()})
                    os.remove(p + ".npz")
            for w in (2, 4):
                f = os.path.join(a.out, f"allreduce_speed_w{w}_r{rep}.json")
                la = load_avg()
                launch(w, "torch_allreduce.py", ["--n-check", "1000", "--out", f], tmp)
                for r in json.load(open(f))["speed"]:
                    summ["allreduce_speed"].append({"world": w, "repeat": rep, "load_before": la, **r})

    summ["load_after"] = load_avg()
    shutil.rmtree(tmp, ignore_errors=True)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps({k: summ[k] for k in ("collective", "resume")}, indent=1))


if __name__ == "__main__":
    main()
