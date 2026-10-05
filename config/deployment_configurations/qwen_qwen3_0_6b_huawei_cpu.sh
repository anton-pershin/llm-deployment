#!/usr/bin/env bash
# Environment setup for the (Qwen/Qwen3-0.6B, huawei-cpu) deployment.
# Installs the CPU build of vLLM into the active Python environment and
# prepares the runtime environment required by the vLLM CPU backend.
set -euo pipefail

PYTHON="${PYTHON:-python3}"

# vLLM CPU wheel: the plain PyPI 'vllm' package is a GPU build; the CPU build
# is distributed as a release wheel from the vLLM GitHub repository.
pip install "https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0+cpu-cp38-abi3-manylinux_2_39_x86_64.whl"
pip install pyyaml pytest ruff

# Intel OpenMP must be preloaded for the vLLM CPU backend; resolve the path
# inside the active environment instead of hardcoding it.
LIBIOMP5_PATH="$(find "$(python3 -c 'import sys; print(sys.prefix)')" -iname 'libiomp5.so' 2>/dev/null | head -n 1)"
if [ -z "$LIBIOMP5_PATH" ]; then
    echo "error: libiomp5.so not found under the active Python prefix" >&2
    exit 1
fi
export LD_PRELOAD="${LIBIOMP5_PATH}:${LD_PRELOAD:-}"

echo "environment ready: LD_PRELOAD=$LD_PRELOAD"
