#!/usr/bin/env bash
# Environment setup for the (Qwen/Qwen3-8B-AWQ, rtx3090-oct-22) deployment.
# Installs the GPU build of vLLM into the active Python environment and
# prepares the runtime environment required by the GPU backend on this host.
set -euo pipefail

# vLLM GPU build: the default PyPI wheels carry the CUDA runtime dependencies,
# so no extra index is needed for this pair (unlike the CPU pair). The 4-bit
# AWQ weights are served by vLLM's own quantization kernels.
pip install "vllm==0.30.0"
pip install pyyaml

# This host's system nvcc is 12.0 while the flashinfer sampler JIT-compiles its
# kernels and requires nvcc >= 12.4, so that JIT build fails at server startup
# and the engine core never initialises. Disabling the flashinfer sampler makes
# vLLM use its native sampler; this does not change the vLLM options of this
# entry, and the server cannot start on this host without it.
#
# The setting must be an environment variable of the vLLM process, but this
# script runs as a process of its own (the deployment is started separately),
# so an export here would not reach it. Install the setting in the environment
# instead: every process started with this interpreter picks it up.
SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
cat > "${SITE_PACKAGES}/sitecustomize.py" <<'PYEOF'
"""Environment hook for the (Qwen/Qwen3-8B-AWQ, rtx3090-oct-22) deployment.

The pair's environment setup script runs as a process of its own, so the
settings it exports do not reach the deployment process. This hook applies the
one setting this pair's deployment needs on this host: the flashinfer sampler
JIT-compiles its kernels and requires nvcc >= 12.4, while this host's system
nvcc is 12.0. Disabling it makes vLLM use its native sampler, which does not
change the vLLM options of this entry.
"""

import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
PYEOF

# Also export it, so that sourcing this script has the same effect as running
# it before the deployment.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

echo "environment ready: VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER} (sitecustomize.py in ${SITE_PACKAGES})"
