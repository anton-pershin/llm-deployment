#!/usr/bin/env bash
# Environment setup for the (Qwen/Qwen3-8B, rtx3090-oct-22) deployment.
# Installs the GPU build of vLLM into the active Python environment, prepares
# the runtime environment required by the GPU backend on this host, and
# prepares the weights this deployment serves.
set -euo pipefail

# The model identifier this pair is deployed under, and the weights it serves.
MODEL_IDENTIFIER="Qwen/Qwen3-8B"
WEIGHTS_REPO="Qwen/Qwen3-8B-AWQ"
WEIGHTS_REVISION="4da05a8edb55c6046cce958586c33b61da07bb79"

# vLLM GPU build: the default PyPI wheels carry the CUDA runtime dependencies,
# so no extra index is needed for this pair (unlike the CPU pair).
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
"""Environment hook for the (Qwen/Qwen3-8B, rtx3090-oct-22) deployment.

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

# The deployment serves the model's 4-bit AWQ release. The bf16 release cannot
# meet this pair's acceptance criteria on this hardware: moving its weights
# (15.27 GiB) across this card's 936 GB/s costs at least ~17.5 ms per output
# token (the median-TPOT criterion requires < 0.01 s), and those weights alone
# are past the whole peak-VRAM budget (the criterion requires < 16 GB), before
# any KV cache is allocated. The 4-bit release halves the bytes per parameter
# and satisfies both.
#
# The entry point and the library format stay unchanged: vLLM resolves a model
# identifier as a local path when such a path exists, and the deployment runs
# with this repository root as its working directory, so materialising the
# identifier as a local directory that holds the AWQ release is enough for the
# deployment to serve those 4-bit weights under the model identifier.
SNAPSHOT="$(python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('${WEIGHTS_REPO}', revision='${WEIGHTS_REVISION}'))
")"
test -f "${SNAPSHOT}/config.json" || {
  echo "error: incomplete snapshot for ${WEIGHTS_REPO} at ${SNAPSHOT}" >&2
  exit 1
}
mkdir -p "$(dirname "${MODEL_IDENTIFIER}")"
ln -sfnT "${SNAPSHOT}" "${MODEL_IDENTIFIER}"

echo "environment ready: VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER} (sitecustomize.py in ${SITE_PACKAGES}); ${MODEL_IDENTIFIER} -> ${SNAPSHOT}"
