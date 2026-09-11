# Lightweight Agent Comparison

Use the same agent model for design, draft, optimization and review within each
comparison arm. Candidate precision remains conservative IEEE FP32 or normal
FP32 plus selective FP16. Agent-model precision is a separate variable.

## Profiles

- `deepseek-flash`: the active replacement arm, DeepSeek V4.1 Flash at
  https://api.deepseek.com/v1. Uses DEEPSEEK_API_KEY from a Kubernetes Secret,
  never saved configuration. Generation uses low reasoning effort; structured
  review disables thinking and uses named function calls. The official model
  ID is documented at https://api-docs.deepseek.com/updates/.
- `openai-mini`: gpt-5.4-mini, direct OpenAI API. The official model page lists
  coding/subagent support, Chat Completions and structured output:
  https://developers.openai.com/api/docs/models/gpt-5.4-mini
  Requires OPENAI_API_KEY supplied securely to the agent controller. Never put
  credentials in YAML, result files, commits or chat. Existing Codex app access
  is not automatically an API credential for the remote pod.
- `qwen-small`: uses MLEVOLVE_QWEN_MODEL and MLEVOLVE_QWEN_BASE_URL, which must
  identify a verified small self-hosted Qwen model. No endpoint is invented.
  The discovered tree-qwen3-small-vllm service refused connections during setup.
- `legacy-qwen`: retained only to reproduce the existing 27B run. It is not a
  lightweight comparison arm.

## Compatibility Gate

Run from a prepared checkout with the environment installed on node-local disk
or baked into its container image. Do this on CPU before requesting a training
GPU. Preserve results on persistent storage. Do not invoke the historical
bootstrap pinned to 48feb88 for these new profiles; it predates these changes.

```bash
python deployments/run_hwdb_precision_matrix.py --agent-profile openai-mini --root /experiment/lightweight/openai-mini --data-root /datasets --smoke-only
python deployments/run_hwdb_precision_matrix.py --agent-profile qwen-small --root /experiment/lightweight/qwen-small --data-root /datasets --smoke-only
```

The gate exercises text generation and the actual CODE_REVIEW_SPEC schema,
validates the returned JSON, and has a 180-second process deadline. A failure
prevents benchmark launch. Missing OpenAI credentials fail explicitly. A small
probe passing does not prove full candidate generation or adapter compliance.

Once both arms pass and a GPU worker is ready, run the same commands without
`--smoke-only`. Each arm retains seed 42, 10 candidate-node budget and 10800
seconds per dataset/mode. Use separate result roots. Agent identity records
prevent accidental resumption under a different model or endpoint. The two
prepared datasets yield eight total mode/model runs; the other ten datasets
remain blocked until authorized public inputs are available.

New profiles disable fail-open code review. The Qwen profile uses JSON-schema
structured responses instead of forced tool calls, so the server does not need
a tool-call parser for review. Non-thinking is sent through vLLM's chat template
options. Native OpenAI GPT-5 requests use max_completion_tokens and server
sampling defaults; secrets are resolved by the SDK from the environment.

## Live Diagnosis

The old Qwen endpoint returned HTTP 400 with the message that a named
tool_choice requires --tool-call-parser. A schema-constrained JSON readiness
probe returned HTTP 200 with valid JSON on the same endpoint. This establishes
the request/server compatibility defect, not a precision/HWDB defect. No shared
model-serving deployment was restarted or modified.

The user subsequently requested resetting and redoing all failed tasks with
DeepSeek. Preserve the historical result directory as superseded evidence and
use a fresh experiment root, resetting all 24 entries. The ten unavailable
datasets remain blocked, not completed. OpenAI/small-Qwen arms are deferred.

`bootstrap_deepseek_precision.sh prepare` installs on CPU/node-local storage,
tests the runtime and real API, then packages an exact-prefix runtime archive.
Only after its READY marker and successful Job completion may the GPU job run
`bootstrap_deepseek_precision.sh run`. No pip installation runs on the GPU.
Both jobs must use the same container image and /runtime mount path.

The user subsequently supplied a local Kaggle credential file. The dedicated
CPU-only `data` phase now attempts the remaining ten datasets before the `run`
phase can request a GPU. It uses the pinned MLE-bench source (the wheel omits
competition metadata), official splits, and default checksum verification.
It never accepts rules automatically. Only verified public archives enter the
training runtime. Held-out labels and the Kaggle Secret are mounted only into
the dataset worker, not the agents. See `prepare_kaggle_data.py` and the
remote `datasets/status.json` for per-dataset completion or blockers.

Disaster Tweets is absent from that MLE-bench registry. Its explicitly custom
definition uses the labeled Kaggle training set with a stratified 80/20 split,
seed 42, binary F1, and identical inputs across precision modes. It is not an
official MLE-bench or Kaggle leaderboard result. Its verified `ready.json`
overrides the initial missing-definition row in the current preparation run.
