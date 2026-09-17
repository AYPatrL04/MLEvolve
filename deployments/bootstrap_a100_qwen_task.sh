#!/usr/bin/env bash
set -euo pipefail

phase=${1:?Expected prepare or run}
task=${2:-full}
root=/experiment/mlevolve-a100-qwen-s42-20260914
base=/experiment/disaster-compare-20260912
qwen_model_dir=${QWEN_MODEL_DIR:-/root/downeyflyfan/qwen38-v100-int8/models/Qwen3.8-27B-INT8-W8A16-MTP}

: "${SOURCE_SHA256:?Local snapshot checksum required}"
mkdir -p "$root" /runtime
export PYTHONUNBUFFERED=1
exec > >(tee -a "$root/${phase}-${task}.log") 2>&1
date -u

cd "$base"
sha256sum -c runtime.tar.sha256
cd "$root"
printf '%s  source.tar.gz\n' "$SOURCE_SHA256" | sha256sum -c -

tar -C /runtime -xf "$base/runtime.tar" venv
tar -C /runtime -xzf source.tar.gz
export PATH=/runtime/venv/bin:$PATH
export PYTHONPATH=/runtime/repo
cd /runtime/repo

prepare_public() {
  python -c 'from deployments.prepare_kaggle_data import stage_public; stage_public({"petfinder-pawpularity-score", "nlp-getting-started"})'
}

if [[ "$phase" == data ]]; then
  : "${KAGGLE_CONFIG_DIR:?Kaggle credentials are required for data preparation}"
  python -u -m deployments.prepare_a100_qwen_data
  echo "PetFinder public dataset is archived and checksummed."
elif [[ "$phase" == prepare ]]; then
  test -d "$qwen_model_dir"
  test -n "$(find "$qwen_model_dir" -mindepth 1 -maxdepth 1 -print -quit)"
  prepare_public
  python -m pytest -q \
    tests/test_a100_qwen_deployment.py \
    tests/test_hwdb_prompt_safety.py \
    tests/test_precision_policy.py \
    tests/test_hardware_context.py \
    tests/test_pipeline_decision_contract.py \
    tests/test_design_knowledge.py \
    tests/test_training_contract_validation.py \
    tests/test_stage_review_workflow.py \
    tests/test_preflight_review_order.py \
    tests/test_component_modularity.py \
    > "$root/regression-tests.log" 2>&1
  python -u -m deployments.run_a100_qwen_task --task petfinder --prepare-only
  python -u -m deployments.run_a100_qwen_task --task full --prepare-only
  touch "$root/READY"
  echo "A100 Qwen CPU gate passed: model, public data, config and merged code."
elif [[ "$phase" == run ]]; then
  test -f "$root/READY"
  case "$task" in petfinder|full) ;; *) echo "Unknown task: $task" >&2; exit 2 ;; esac
  prepare_public
  nvidia-smi > "$root/gpu-${task}.txt"

  export MODEL_DIR="$qwen_model_dir"
  export SERVED_MODEL_NAME=qwen3.8-27b-int8-a100
  export PORT=8000
  export GPU_MEMORY_UTILIZATION=${QWEN_GPU_MEMORY_UTILIZATION:-0.50}
  export MAX_MODEL_LEN=${QWEN_MAX_MODEL_LEN:-65536}
  export MAX_NUM_SEQS=${QWEN_MAX_NUM_SEQS:-64}
  export DEPLOY_ROOT=/root/downeyflyfan/qwen38-v100-int8
  export LOG_DIR="$root/qwen-${task}"
  export STATE_DIR="$root/qwen-${task}-state"
  export RUNTIME_DIR=/tmp/mlevolve-vllm-a100
  export UV_CACHE_DIR=/tmp/uv-cache-a100
  export VLLM_CACHE_ROOT=/tmp/vllm-cache-a100
  bash deployments/bootstrap_qwen38_a100.sh

  for _ in $(seq 1 360); do
    if curl --fail --silent "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
      break
    fi
    sleep 5
  done
  curl --fail --silent "http://127.0.0.1:8000/health" >/dev/null
  python - <<'PY'
import json
import os
import urllib.request

payload = json.dumps(
    {
        "model": os.environ["SERVED_MODEL_NAME"],
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 8,
        "temperature": 0,
    }
).encode()
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=300) as response:
    result = json.load(response)
if not result.get("choices"):
    raise RuntimeError("vLLM smoke completion returned no choices")
print("vLLM smoke response OK", flush=True)
PY
  exec python -u -m deployments.run_a100_qwen_task --task "$task"
else
  echo "Expected data, prepare or run" >&2
  exit 2
fi
