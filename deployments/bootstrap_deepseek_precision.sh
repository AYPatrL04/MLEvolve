#!/usr/bin/env bash
set -euo pipefail
export GIT_LFS_SKIP_SMUDGE=1
export PIP_NO_CACHE_DIR=1
export PYTHONUNBUFFERED=1
export HF_HOME=/experiment/cache/huggingface
export MPLCONFIGDIR=/runtime/matplotlib
mkdir -p /runtime /experiment/cache
phase=${1:?Expected prepare or run}
exec > >(tee -a "/experiment/${phase}.log") 2>&1
date -u

if [[ "$phase" == prepare ]]; then
  : "${SOURCE_COMMIT:?A pinned source commit is required}"
  source /launcher/git_retry.sh
  git init /runtime/repo
  cd /runtime/repo
  git remote add origin https://github.com/AYPatrL04/MLEvolve.git
  git_retry fetch --depth 1 origin "$SOURCE_COMMIT"
  git checkout --detach FETCH_HEAD
  git_retry submodule update --init --recursive --depth 1
  python -m venv --system-site-packages /runtime/venv
  export PATH=/runtime/venv/bin:$PATH
  python -m pip install -r requirements_base.txt pytest
  git rev-parse HEAD > /experiment/source_commit.txt
  git submodule status > /experiment/submodules.txt
  python -m pip freeze > /experiment/environment.txt
  python -m pytest -q tests/test_lightweight_agents.py tests/test_qwen_vllm_output_limit.py tests/context_cache/test_vllm.py tests/test_config_loading.py tests/test_hwdb_prompt_safety.py tests/test_hardware_feature_filter.py tests/test_hardware_knowledge_client.py tests/test_hardware_context.py tests/test_stage_hardware_prompt_preview.py tests/test_model_preflight_integration.py > /experiment/regression-tests.log 2>&1
  python deployments/run_hwdb_precision_matrix.py --agent-profile deepseek-flash --root /experiment/results --smoke-only
  # Both pods use this exact image and /runtime prefix, including editable installs.
  tar -C /runtime -cf /experiment/runtime.tar.partial repo venv
  mv /experiment/runtime.tar.partial /experiment/runtime.tar
  cd /experiment
  sha256sum runtime.tar > runtime.tar.sha256
  touch READY
  echo 'CPU preparation, regression tests and live agent smoke passed.'
elif [[ "$phase" == run ]]; then
  test -f /experiment/READY
  cd /experiment
  sha256sum -c runtime.tar.sha256
  tar -C /runtime -xf runtime.tar
  export PATH=/runtime/venv/bin:$PATH
  cd /runtime/repo
  nvidia-smi > /experiment/gpu.txt
  exec python deployments/run_hwdb_precision_matrix.py --agent-profile deepseek-flash --root /experiment/results --seconds 10800 --nodes 10
else
  echo 'Expected prepare or run' >&2
  exit 2
fi
