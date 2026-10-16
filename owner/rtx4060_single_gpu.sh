#!/usr/bin/env bash
# Owner-only (BLOCKED here: no GPU). Single RTX 4060: NCCL 1-rank sanity + CUDA DDP 1-rank correctness vs dtrain.
# Needs: NVIDIA driver, Python 3.10+, internet access to download.pytorch.org.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m venv .venv-gpu
. .venv-gpu/bin/activate
pip install -e '.[dev]'
pip install torch --index-url https://download.pytorch.org/whl/cu126
nvidia-smi | tee owner_rtx4060_nvidia_smi.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from dtrain.data import load_text; load_text(None)"   # tiny-shakespeare, SHA-256 checked
torchrun --standalone --nproc_per_node 1 benchmarks/torch_gpu_check.py --out owner_rtx4060_gpu1.json 2>&1 | tee owner_rtx4060_gpu1.log
# Also run the CPU cross-check suite on this machine (gloo):
python -m pytest -q tests/test_torch_crosscheck.py 2>&1 | tee owner_rtx4060_pytest_torch.log
echo "Save owner_rtx4060_* into benchmarks/results/<date>/gpu/ and commit."
