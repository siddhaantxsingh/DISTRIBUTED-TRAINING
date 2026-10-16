#!/usr/bin/env bash
# Owner-only (BLOCKED here: no GPUs). Rented multi-GPU host (e.g. 4x or 8x GPU, one node): real NCCL scaling.
# Usage: bash owner/multi_gpu_nccl.sh [max_gpus]   (default: all visible GPUs)
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-$(nvidia-smi -L | wc -l)}
python -m venv .venv-gpu && . .venv-gpu/bin/activate
pip install -e '.[dev]'
pip install torch --index-url https://download.pytorch.org/whl/cu126
mkdir -p gpu_results
nvidia-smi > gpu_results/nvidia_smi.txt
nvidia-smi topo -m > gpu_results/topo.txt || true
python -c "from dtrain.data import load_text; load_text(None)"
for n in 1 2 4 8; do
  [ "$n" -le "$N" ] || break
  torchrun --standalone --nproc_per_node "$n" benchmarks/torch_gpu_check.py --out gpu_results/gpu_$n.json 2>&1 | tee gpu_results/gpu_$n.log
done
# NCCL's own benchmark for reference (optional): https://github.com/NVIDIA/nccl-tests
echo "Copy gpu_results/ to benchmarks/results/<date>/gpu/ and commit; report tokens/s scaling and bus GB/s per size."
