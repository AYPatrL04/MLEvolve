#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1 GIT_LFS_SKIP_SMUDGE=1
export HF_HOME=/experiment/cache/huggingface MPLCONFIGDIR=/runtime/matplotlib
phase=${1:?Expected prepare or run}
root=${ABLATION_ROOT:-/experiment/hwdb-content-ablation-20260913}
base=/experiment/disaster-compare-20260912
: "${SOURCE_COMMIT:?Pinned source commit required}"
mkdir -p "$root" /runtime
exec > >(tee -a "$root/${phase}.log") 2>&1
date -u
cd "$base"
sha256sum -c runtime.tar.sha256
if [[ "$phase" == prepare ]]; then
  test ! -e "$root/READY"
  tar -C /runtime -xf runtime.tar
  export PATH=/runtime/venv/bin:$PATH
  source /launcher/git_retry.sh
  cd /runtime/repo
  git_retry fetch --depth 1 https://github.com/AYPatrL04/MLEvolve.git "$SOURCE_COMMIT"
  git checkout --detach FETCH_HEAD
  test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
  git_retry fetch --depth 1 https://github.com/JustinLinKK/MLEvolve.git 93371dd64b8e2b888c1bde7cb9d90d7c03ac4e5d
  git rev-parse HEAD > "$root/source_commit.txt"
  git submodule status > "$root/submodules.txt"
  python -m pytest -q tests/test_hwdb_ablation.py tests/test_run_lifecycle.py tests/test_config_loading.py tests/test_hardware_feature_filter.py tests/test_hwdb_prompt_safety.py tests/test_milestone.py tests/test_pipeline_logging_metrics.py tests/test_training_contract_validation.py tests/test_script_introspection.py tests/test_model_preflight_integration.py > "$root/regression-tests.log" 2>&1
  python -m deployments.run_hwdb_ablation --root "$root" --prepare-only
  python -m deployments.smoke_agent --agent-profile deepseek-flash > "$root/agent-smoke.log" 2>&1
  python -c 'from deployments.prepare_kaggle_data import stage_public; stage_public({"nlp-getting-started"})'
  python -m deployments.compare_disaster_precision --cpu-smoke --root "$root" > "$root/candidate-cpu-smoke.log" 2>&1
  # Reuse immutable dependencies; publish only the updated repository, not another venv copy.
  tar --blocking-factor=8192 -C /runtime -cf "$root/repo.tar.partial" repo
  mv "$root/repo.tar.partial" "$root/repo.tar"
  cd "$root"
  sha256sum repo.tar > repo.tar.sha256
  touch READY
  echo 'HWDB ablation CPU gates and live agent smoke passed.'
elif [[ "$phase" == run ]]; then
  test -f "$root/READY"
  tar -C /runtime -xf runtime.tar venv
  cd "$root"
  sha256sum -c repo.tar.sha256
  tar -C /runtime -xf repo.tar
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
  {
    echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi -L
    df -h /experiment /runtime
    free -h
    while true; do
      printf 'timestamp=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory \
        --format=csv,noheader,nounits
      ps -eo pid,ppid,pcpu,pmem,rss,etime,args --sort=-pcpu | head -n 15
      sleep 10
    done
  } > "$root/resource-monitor.log" 2>&1 &
  exec python -m deployments.run_hwdb_ablation --root "$root"
else
  exit 2
fi
