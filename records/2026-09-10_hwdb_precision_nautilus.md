# HWDB Precision Nautilus Comparison

## Scope

- Source: fork AYPatrL04/MLEvolve, branch hwdb-precision-guidance.
- Tested application commit: 48feb8809057fa341793e610b4db072d84ee0cd9.
- Submodules: parent-pinned recursive checkout; runtime records exact hashes.
- Twelve requested competitions, conservative versus normal: 24 planned runs.
- Budget: 10 budgeted candidate nodes and a hard 10800-second wall limit per run.
  Repair attempts are additional and retained; this is not a strict 10-attempt cap.
- Conservative: IEEE FP32 only. Normal: FP32 plus selective FP16 AMP.
- Agent: existing internal Qwen service, qwen3.8-27b-int8-a100.
  Its inference quantization is independent of generated candidate precision.
- Seed 42, branch-profile scheduler, no fixed parallel job cap.
- Static versioned HWDB JSON; Neo4j disabled to avoid stale shared graph content.
- Historical/global lesson memory disabled equally for both modes.
- Public training/test inputs only; no private labels mounted, no Kaggle submissions.
- Single seed and shortened budgets are diagnostic tests, not leaderboard results.

## Deployment

- Context: nautilus; namespace: ecepxie.
- Job/label: hwdb-precision-20260910 / app=hwdb-precision-20260910.
- ConfigMap: hwdb-precision-launcher-20260910.
- Initial A100 request remained Pending; replaced with A10 affinity and GPU resource.
- Manifest: deployments/hwdb-precision-a10.yaml.
- Remote PVC: yuze-li-vol, isolated subdirectory aypatrl04-hwdb-precision-20260910.
- Job mount: /experiment. CPU maintenance pod mount:
  /workspace/aypatrl04-hwdb-precision-20260910.
- Existing CPU pod mlevolve-agentic-knowledge-base-dev-cpu became Failed during
  setup; use the running benchmark pod for reads. Do not restart the unrelated pod.
- Hard job deadline 78 hours; no automatic retry of failed jobs.
- Existing workloads and shared repositories are not modified.

## Dataset Availability

Prepared public data verified for spooky-author-identification and
jigsaw-toxic-comment-classification-challenge. These supply four runnable mode
comparisons. The other ten competitions are planned but blocked on missing data.
No Kaggle credentials were found in the inspected local or remote standard
locations. Supply an authorized prepared-public remote dataset path or configure
Kaggle access and any required competition acceptance before extending the mounts.
Private evaluation labels must remain outside agent-accessible mounts.

## Monitoring

Hourly heartbeat: nautilus-hwdb-hourly-checks, ACTIVE, attached to this task.
Check current pods by label, not a stale pod suffix. Inspect:

- /experiment/bootstrap.log and regression-tests.log.
- /experiment/source_commit.txt, submodules.txt, environment.txt, gpu.txt.
- /experiment/results/matrix.json: queued/running/finished/failed/timeout/blocked_missing_data.
- /experiment/results/<competition>/<mode>/runtime.log.
- Per-run journals, hardware_prompt_audit, and working/preflight attempt evidence.
- /experiment/results/comparison.png: run timeline above per-node metrics.

A zero process exit is not proof of successful training or precision compliance.
Inspect journals, actual GPU executions, preflight outcomes, and artifacts before
calling a comparison successful. Attribute issues to HWDB/filtering only when
supported by joined prompt/code/runtime evidence. Do not restart failures blindly.

## Initial Verification

Launcher shell syntax and Python compilation passed. All 34 focused launcher/HWDB
tests passed locally. The A10 pod hwdb-precision-20260910-stzmj is Running on
gpu-06.nrp.mghpcc.org; bootstrap started at 2026-09-11 04:01:41 UTC. Source cloning
is in progress. This is container startup, not yet an agent-training result.
The first local cleanup test needed escalation because macOS sandboxing blocks
psutil process enumeration; the rerun with that access passed.
GPU query confirms NVIDIA A10, 23028 MiB. Remote matrix.json was initialized:
four queued mode runs and twenty blocked_missing_data entries.
The initially mounted bootstrap runs the five hardware regression modules;
the committed launcher also includes test_model_preflight_integration.py.
Rerun that module explicitly after dependency setup if the mounted bootstrap
has not picked up the updated ConfigMap; preserve its Linux worker results.
