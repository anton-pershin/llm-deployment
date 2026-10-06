#!/usr/bin/env bash
# Environment setup for the (Qwen/Qwen3-0.6B, rtx3090-oct-22) deployment.
# Installs the GPU build of vLLM into the active Python environment and
# prepares the runtime environment required by the GPU backend on this host.
set -euo pipefail

# vLLM GPU build: the default PyPI wheels carry the CUDA runtime dependencies,
# so no extra index is needed for this pair (unlike the CPU pair).
pip install "vllm==0.30.0"
pip install pyyaml

# The flashinfer sampler JIT-compiles its kernels and requires nvcc >= 12.4,
# while this host's system nvcc is 12.0, so the JIT build fails at server
# startup. Disabling it makes vLLM use its native sampler, which does not
# change the deployment options of this baseline entry.
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "environment ready: VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
