#!/usr/bin/env bash
set -euo pipefail
root=/experiment
export GIT_LFS_SKIP_SMUDGE=1
export PIP_NO_CACHE_DIR=1
export HF_HOME="$root/cache/huggingface"
export MPLCONFIGDIR="$root/cache/matplotlib"
mkdir -p "$root/cache"
exec > >(tee -a "$root/bootstrap.log") 2>&1
date -u
if [[ ! -d "$root/repo/.git" ]]; then
  git clone --depth 1 --branch hwdb-precision-guidance https://github.com/AYPatrL04/MLEvolve.git "$root/repo"
fi
cd "$root/repo"
git fetch --depth 1 origin 48feb8809057fa341793e610b4db072d84ee0cd9
git checkout 48feb8809057fa341793e610b4db072d84ee0cd9
git submodule update --init --recursive --depth 1
cp /launcher/run_hwdb_precision_matrix.py deployments/run_hwdb_precision_matrix.py
python -m venv --system-site-packages "$root/venv"
export PATH="$root/venv/bin:$PATH"
python -m pip install --upgrade pip
python -m pip install -r requirements_base.txt pytest
git rev-parse HEAD > "$root/source_commit.txt"
git submodule status > "$root/submodules.txt"
python -m pip freeze > "$root/environment.txt"
nvidia-smi > "$root/gpu.txt"
python -m pytest -q tests/test_hwdb_prompt_safety.py tests/test_hardware_feature_filter.py tests/test_hardware_knowledge_client.py tests/test_hardware_context.py tests/test_stage_hardware_prompt_preview.py tests/test_model_preflight_integration.py > "$root/regression-tests.log" 2>&1
python /launcher/run_hwdb_precision_matrix.py --help > /dev/null
exec python deployments/run_hwdb_precision_matrix.py --root "$root/results"
