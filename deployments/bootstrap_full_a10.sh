#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME=/experiment/cache/huggingface MPLCONFIGDIR=/runtime/matplotlib
phase=${1:?Expected prepare or run}
root=/experiment/mlevolve-full-a10-s42-20260914
: "${SOURCE_SHA256:?Local snapshot checksum required}"
mkdir -p "$root" /runtime
exec > >(tee -a "$root/${phase}.log") 2>&1
date -u
cd /experiment/disaster-compare-20260912
sha256sum -c runtime.tar.sha256
tar -C /runtime -xf runtime.tar venv
cd "$root"
printf '%s  source.tar.gz\n' "$SOURCE_SHA256" | sha256sum -c -
tar -C /runtime -xzf source.tar.gz
export PATH=/runtime/venv/bin:$PATH
export PYTHONPATH=/runtime/repo
cd /runtime/repo
python -c 'from deployments.prepare_kaggle_data import stage_public; stage_public({"nlp-getting-started"})'

if [[ "$phase" == prepare ]]; then
  python -m pytest -q tests/test_full_a10_deployment.py tests/test_config_loading.py tests/test_hardware_feature_filter.py tests/test_hwdb_prompt_safety.py tests/test_model_preflight_integration.py tests/test_pipeline_logging_metrics.py tests/test_training_contract_validation.py localml_scheduler/tests/test_ml_prediction.py -k 'not blackwell' > "$root/regression-tests.log" 2>&1
  python -m deployments.smoke_agent --agent-profile deepseek-flash > "$root/agent-smoke.log" 2>&1
  python -m deployments.run_full_a10 --prepare-only
  touch "$root/READY"
  echo 'CPU gate passed: local source, A10 predictor, Preflight, HWDB, agent API and config.'
elif [[ "$phase" == run ]]; then
  test -f "$root/READY"
  nvidia-smi > "$root/gpu.txt"
  cd /experiment
  sha256sum -c mlebench-source.tar.sha256
  mkdir -p /runtime/mlebench-source
  tar -C /runtime/mlebench-source -xf mlebench-source.tar
  export PYTHONPATH=/runtime/mlebench-source:/runtime/repo
  cd /runtime/repo
  exec python -m deployments.run_full_a10
else
  exit 2
fi
