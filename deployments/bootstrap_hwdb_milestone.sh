#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export GIT_LFS_SKIP_SMUDGE=1
export HF_HOME=/experiment/cache/huggingface
export MPLCONFIGDIR=/runtime/matplotlib
phase=${1:?Expected prepare or run}
root=/experiment/milestone-20260912
mkdir -p "$root" /runtime
exec > >(tee -a "$root/${phase}.log") 2>&1
date -u

if [[ "$phase" == prepare ]]; then
  : "${SOURCE_COMMIT:?A pinned source commit is required}"
  test ! -e "$root/READY"
  cd /experiment
  sha256sum -c runtime.tar.sha256
  tar -C /runtime -xf runtime.tar
  export PATH=/runtime/venv/bin:$PATH
  source /launcher/git_retry.sh
  cd /runtime/repo
  git_retry fetch --depth 1 origin "$SOURCE_COMMIT"
  git checkout --detach FETCH_HEAD
  git submodule status > "$root/submodules.txt"
  test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
  git rev-parse HEAD > "$root/source_commit.txt"
  python -m pytest -q tests/test_milestone.py tests/test_run_lifecycle.py tests/test_model_preflight_integration.py tests/test_stage_review_workflow.py tests/test_training_diagnostics.py tests/test_impl_guideline.py tests/test_draft_agent_preflight_prompt.py tests/test_pipeline_decision_contract.py tests/test_hwdb_precision_matrix.py tests/test_hwdb_prompt_safety.py tests/test_hardware_feature_filter.py tests/test_lightweight_agents.py tests/test_config_loading.py tests/test_kaggle_public_staging.py tests/test_milestone_launcher.py > "$root/regression-tests.log" 2>&1
  python -m deployments.smoke_agent --agent-profile deepseek-flash > "$root/agent-smoke.log" 2>&1
  tar --blocking-factor=8192 -C /runtime -cf "$root/runtime.tar.partial" repo venv
  mv "$root/runtime.tar.partial" "$root/runtime.tar"
  cd "$root"
  sha256sum runtime.tar > runtime.tar.sha256
  touch READY
  echo 'Milestone CPU regressions and live agent smoke passed.'
elif [[ "$phase" == run ]]; then
  test -f "$root/READY"
  cd "$root"
  sha256sum -c runtime.tar.sha256
  tar -C /runtime -xf runtime.tar
  export PATH=/runtime/venv/bin:$PATH
  cd /runtime/repo
  test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
  cd /experiment
  sha256sum -c mlebench-source.tar.sha256
  mkdir -p /runtime/mlebench-source
  tar -C /runtime/mlebench-source -xf mlebench-source.tar
  export PYTHONPATH=/runtime/mlebench-source:/runtime/repo
  cd /runtime/repo
  python -c 'from deployments.prepare_kaggle_data import stage_public; stage_public({"nlp-getting-started"})'
  nvidia-smi > "$root/gpu.txt"
  exec python deployments/run_hwdb_precision_matrix.py --agent-profile deepseek-flash --root "$root/results" --first-valid-node --seconds 3600 --nodes 1
else
  exit 2
fi
