#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1 GIT_LFS_SKIP_SMUDGE=1
export HF_HOME=/experiment/cache/huggingface MPLCONFIGDIR=/runtime/matplotlib
phase=${1:?Expected prepare or run}
root=/experiment/disaster-compare-20260912
: "${SOURCE_COMMIT:?A pinned source commit is required}"
mkdir -p "$root" /runtime
exec > >(tee -a "$root/${phase}.log") 2>&1
date -u

if [[ "$phase" == prepare ]]; then
  test ! -e "$root/READY"
  cd /experiment/milestone-20260912
  sha256sum -c runtime.tar.sha256
  tar -C /runtime -xf runtime.tar
  export PATH=/runtime/venv/bin:$PATH
  source /launcher/git_retry.sh
  cd /runtime/repo
  git_retry fetch --depth 1 https://github.com/AYPatrL04/MLEvolve.git "$SOURCE_COMMIT"
  git checkout --detach FETCH_HEAD
  test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
  git rev-parse HEAD > "$root/source_commit.txt"
  git submodule status > "$root/submodules.txt"
  python -m pytest -q tests/test_script_introspection.py tests/test_training_contract_validation.py tests/test_impl_guideline.py tests/test_stage_review_workflow.py tests/test_training_diagnostics.py tests/test_pipeline_logging_metrics.py tests/test_disaster_comparison.py tests/test_milestone.py tests/test_hwdb_prompt_safety.py tests/test_hardware_feature_filter.py > "$root/regression-tests.log" 2>&1
  python -c 'from deployments.prepare_kaggle_data import stage_public; stage_public({"nlp-getting-started"})'
  python -m deployments.compare_disaster_precision --cpu-smoke --root "$root" > "$root/candidate-cpu-smoke.log" 2>&1
  tar --blocking-factor=8192 -C /runtime -cf "$root/runtime.tar.partial" repo venv
  mv "$root/runtime.tar.partial" "$root/runtime.tar"
  cd "$root"
  sha256sum runtime.tar > runtime.tar.sha256
  touch READY
  echo 'Controlled comparison CPU regressions and frozen-model update passed.'
elif [[ "$phase" == run ]]; then
  test -f "$root/READY"
  cd "$root"
  sha256sum -c runtime.tar.sha256
  tar -C /runtime -xf runtime.tar
  export PATH=/runtime/venv/bin:$PATH
  cd /runtime/repo
  test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
  python -c 'from deployments.prepare_kaggle_data import stage_public; stage_public({"nlp-getting-started"})'
  nvidia-smi > "$root/gpu.txt"
  exec python -m deployments.compare_disaster_precision --root "$root/results"
else
  exit 2
fi
