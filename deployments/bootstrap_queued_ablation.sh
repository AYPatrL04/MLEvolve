#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1 GIT_LFS_SKIP_SMUDGE=1
export HF_HOME=/experiment/cache/huggingface MPLCONFIGDIR=/runtime/matplotlib
phase=${1:?phase required}
root=${ABLATION_ROOT:?root required}
base=/experiment/disaster-compare-20260912
mkdir -p "$root" /runtime
log="$root/$phase.log"
if [[ "$phase" == worker ]]; then log="$2/worker.log"; fi
exec > >(tee -a "$log") 2>&1
cd "$base"
sha256sum -c runtime.tar.sha256
if [[ "$phase" == prepare ]]; then
  test ! -e "$root/READY"
  tar -C /runtime -xf runtime.tar
  export PATH=/runtime/venv/bin:$PATH
  cd /runtime/repo
  source /launcher/git_retry.sh
  git_retry fetch https://github.com/AYPatrL04/MLEvolve.git "$SOURCE_COMMIT"
  git checkout --detach FETCH_HEAD
  git submodule update --init --recursive
  git_retry fetch https://github.com/JustinLinKK/MLEvolve.git hardware-awared
  test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
  git submodule status > "$root/submodules.txt"
  python -m pytest -q tests/test_queued_ablation.py tests/test_hwdb_ablation.py tests/test_design_knowledge.py tests/test_precision_policy.py tests/test_precision_quality.py tests/test_hwdb_prompt_safety.py tests/test_hardware_feature_filter.py tests/test_training_contract_validation.py tests/test_impl_guideline.py tests/test_model_preflight_integration.py > "$root/regression-tests.log" 2>&1
  python -m deployments.run_hwdb_ablation --root "$root" --prepare-only
  python -m deployments.smoke_agent --agent-profile deepseek-flash > "$root/agent-smoke.log" 2>&1
  python -c 'from deployments.prepare_kaggle_data import stage_public; stage_public({"nlp-getting-started"})'
  tar -C /runtime -cf "$root/repo.tar.partial" repo
  mv "$root/repo.tar.partial" "$root/repo.tar"
  cd "$root"
  sha256sum repo.tar > repo.tar.sha256
  touch READY
  exit 0
fi
test -f "$root/READY"
tar -C /runtime -xf runtime.tar venv
cd "$root"
sha256sum -c repo.tar.sha256
tar -C /runtime -xf repo.tar
export PATH=/runtime/venv/bin:$PATH
cd /experiment
sha256sum -c mlebench-source.tar.sha256
mkdir -p /runtime/mlebench-source
tar -C /runtime/mlebench-source -xf mlebench-source.tar
export PYTHONPATH=/runtime/mlebench-source:/runtime/repo
cd /runtime/repo
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
python -c 'from deployments.prepare_kaggle_data import stage_public; stage_public({"nlp-getting-started"})'
if [[ "$phase" == coordinator ]]; then
  exec python -m deployments.queued_ablation_coordinator --root "$root"
elif [[ "$phase" == worker ]]; then
  if [[ "$2" == "$root/probe" ]]; then
    python -c 'from pathlib import Path; import os; from localml_scheduler.hardware import detect_hardware_profile; from deployments.run_hwdb_precision_matrix import save; p=detect_hardware_profile(); assert p.gpu_name == "NVIDIA A10"; save(Path(os.environ["ABLATION_ROOT"])/"target-hardware.json", p.to_dict())'
  else
    exec python -m deployments.queued_execution "$2"
  fi
else
  exit 2
fi
