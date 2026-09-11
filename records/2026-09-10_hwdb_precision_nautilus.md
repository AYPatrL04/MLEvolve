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

## Hourly Check: 2026-09-11 04:59 UTC

- Pod hwdb-precision-20260910-stzmj remains Running, zero restarts, on
  gpu-06.nrp.mghpcc.org. Scheduling, volume attachment and container startup
  events were successful.
- Parent checkout completed at the intended application commit 48feb88.
- Infrastructure issue: the first PerfSeer submodule clone failed with
  `curl 56 Recv failure: Connection reset by peer`, early EOF and invalid
  index-pack output. Git scheduled its built-in retry; the retry's Git/HTTP
  processes were still active at inspection. No manual restart was performed.
- Actual GPU: NVIDIA A10, 0 MiB allocated, 0% utilization during setup.
- Matrix unchanged: four queued runs, twenty blocked_missing_data entries.
- No regression-test log, runtime logs, journals or preflight feedback exist
  yet. Dependency setup and agent training have not started. There are no
  observed agent errors or precision-policy results; HWDB/filter attribution
  is unavailable, not a passing result.
- Hourly monitoring remains active. Preserve this retry and inspect the next
  bootstrap outcome before considering recovery. Missing datasets remain an
  independent blocker.

## Download Recovery: 2026-09-11 05:11 UTC

- User requested investigation and repair of the connection reset. The logs
  establish an interrupted Git HTTP receive, but do not identify whether
  GitHub, the network path, or an intermediary sent the reset.
- Git's existing retry completed before intervention was needed. Verified
  recursive submodule status now exactly matches PerfSeer b9b6d48 and
  preflight d908527, with no `+` mismatch markers. The parent remains pinned.
- The job progressed past Git to Python virtual-environment creation and
  ensurepip. No job restart, source replacement or other-workload changes
  were necessary. Agent training is not yet underway.
- Hardened the bootstrap for subsequent starts: use the existing pinned
  commit when present; fetch into an initialized checkout rather than
  restarting a full clone; three attempts with backoff; HTTP/1.1; a 60-second
  low-throughput timeout and 20-minute per-attempt process timeout; Git
  object validation. This is mitigation, not proof of an HTTP/2 defect.
- New deployments/git_retry.sh must be included in the launcher ConfigMap
  alongside bootstrap_hwdb_precision.sh and run_hwdb_precision_matrix.py.
  Updating that ConfigMap does not retrofit retry handling into an already
  executing Git process. The current download has independently recovered.
- Four new retry/syntax tests passed locally. Hourly monitoring remains active.

## Hourly Check: 2026-09-11 06:01 UTC

- Job remains active; pod Running on the same A10 node, zero restarts.
- Git recovery held. Dependency downloads and wheel builds completed;
  pip is installing packages into /experiment/venv. No new fatal error
  appeared in the inspected bootstrap tail. Messages about not uninstalling
  system-site packages are not themselves installation failures.
- Setup is slow: installer PID 461 had been running about 35 minutes and
  was in D state, with wait channel folio_wait_bit_common. This observation
  supports a filesystem-I/O wait at inspection, not a renewed Git failure.
  About 1.87 GB had been written by that process. Last bootstrap log update
  was 05:58:00 UTC, a few minutes before this check.
- Storage is not full: persistent mount has about 945 GB available;
  container overlay has about 4.2 TB available. No events were returned
  for this pod at the check; no new pod/volume warning was observed.
- GPU remains idle (0 MiB, 0%). Matrix is four queued runs and twenty
  blocked_missing_data entries. No regression-test log, per-mode runtime
  logs or journals exist yet. Agent errors and precision-policy outcomes
  cannot be assessed; HWDB/filter attribution remains unavailable.
- No restart or environment mutation performed. Hourly monitoring remains
  active; inspect installer completion and regression results next.

## Hourly Check: 2026-09-11 07:02 UTC

- Job active; same A10 pod Running with zero restarts. No new pod events
  were returned. GPU still idle, 0 MiB and 0% utilization.
- Pip installation continues after about 96 minutes. Bootstrap advanced
  to the typer system-site-package notice; no fatal installation error was
  observed. Latest log modification was 06:15:05 UTC.
- Installer PID 461 remains in D state at folio_wait_bit_common. Its
  write_bytes counter increased from 1,873,321,984 to 3,462,078,464 since
  the previous check, about 1.59 GB of additional writes. This establishes
  progress despite sparse log output; it is not a confirmed deadlock.
  Filesystem I/O remains the observed setup bottleneck, not Git transfer.
- Matrix unchanged: four queued runs, twenty blocked_missing_data.
  Regression tests have not started; no runtime logs, journals or preflight
  feedback were found. No agent or precision-policy result can be claimed,
  and HWDB/filter attribution remains unavailable.
- No restart or mutation of the active installation was performed. Hourly
  monitoring stays active. If recovery becomes necessary, prefer a fresh
  node-local runtime with persistent results rather than repeating heavy
  package installation on the shared volume; retain existing setup evidence.

## Hourly Check: Observed Remote Time 2026-09-11 21:58 UTC

The heartbeat carried 08:11 UTC, but the live pod clock reported 21:58 UTC
and its age was about 18 hours. Findings below use the observed live state,
not the stale trigger timestamp.

- Dependency installation completed. The mounted bootstrap's regression
  suite passed: 78 tests in 224.96 seconds. This does not establish success
  of the separate Linux live-worker integration module added later.
- Matrix: Jigsaw conservative, Jigsaw normal and Spooky conservative each
  reached the three-hour timeout (exit 124); Spooky normal remains running.
  Twenty entries remain blocked_missing_data. Job is active, pod Running,
  no restarts; no pod events returned. A10 currently 3 MiB, 0% utilization.
- New integration failure: repeated HTTP 400 responses from the agent
  service's chat/completions endpoint during code review, plus one observed
  server disconnect. Request validation details are not in these logs, so
  the exact rejected parameter/schema is not yet established. Persisted
  review history records review_unavailable after three attempts. Logs then
  say candidates passed review or were modified; these messages must not
  be treated as successful review evidence when the API was unavailable.
- Three completed journals contain only their root node, with null metrics.
  Runtime logs show candidates rejected before execution (two in each
  Jigsaw mode, one in Spooky conservative). There is no validated GPU
  training result or usable FP32/FP16 quality/speed comparison in this check.
- Three preserved preflight feedback files for Jigsaw conservative node
  ff88448398824dc8abc481b1820e224d report FAIL: missing CandidateAdapter and
  missing main guard (MLE_ADAPTER001, MLE_IMPORT001, also GPU003).
  Attempts 0/1/2 have the same code hash, so repair did not change that
  candidate. Diagnostic ownership is integration; root-cause attribution
  remains explicitly unassigned pending code/prompt comparison.
- hardware_prompt_audit is preserved in those feedback files, including
  static hardware_knowledge_graph.json stage evidence and conservative
  allowed policies fp32/disabled. Missing Neo4j/profile warnings alone do
  not prove missing static HWDB guidance. No runtime precision compliance
  or FP16 efficacy finding can be made from these rejected candidates.
- The active Spooky normal run is still spending time in generation/review,
  with repeated HTTP 400s; it has not produced a persisted valid metric.
  Preserve this evidence. API request compatibility and adapter-generation/
  repair must be diagnosed before allocating another comparison budget.
  No restart or active-run mutation performed; monitor remains active until
  the final mode terminates, then report and pause as configured.

## Follow-Up Check: 2026-09-11 22:00 UTC

- This trigger followed the prior inspection by only a few minutes.
  Live state unchanged: three timed-out mode runs, one active Spooky normal
  run, twenty blocked_missing_data entries. Active run elapsed 127.8 of
  180 minutes; job and pod remain active with no restarts.
- Latest active-run log remains at 21:30:21 UTC with repeated code-review
  HTTP 400 errors. No new journals or preflight feedback: counts remain
  three and three. The 78-test regression result is unchanged.
- A10 usage 3 MiB and 0%; no new pod events returned. There is still no
  validated quality, throughput or precision-policy outcome and no new
  evidence supporting HWDB/filter root-cause attribution.
- No intervention performed. Monitor remains active for the final bounded
  run's termination; do not infer completion merely from the earlier timeouts.

## Lightweight Agent Preparation

- User requested lightweight OpenAI agents, with a possible small local Qwen
  comparison. Added explicit openai-mini (gpt-5.4-mini) and qwen-small profiles,
  independent result roots, saved agent identity, fail-closed review and a
  bounded generation plus real-review-schema compatibility gate.
- OpenAI credentials are absent from the inspected controller/pod environment;
  the discovered cliproxy endpoint requires authentication. Asked for an
  authorized Secret/proxy location, not raw credentials. The discovered
  tree-qwen3-small-vllm and qwen38-a10-agent endpoints refused connections.
  No replacement GPU benchmark or model-serving workload was launched.
- Reproduced the old review failure with a tiny request: the server responds
  that named tool_choice requires --tool-call-parser. A JSON-schema alternative
  succeeded. A second live probe using the actual CODE_REVIEW_SPEC also
  succeeded (HTTP 200, valid approved/reasoning/issues payload, 1.68 seconds).
  Root cause is request/server compatibility, not model precision.
- Added opt-in vLLM JSON-schema review support with local schema validation,
  correct chat-template thinking option placement, and native OpenAI GPT-5
  completion-token/sampling parameter handling. Current running experiment
  was not hot-patched; its results remain the historical 27B arm.
- See docs/lightweight_agent_comparison.md. Stage runtime on CPU/node-local
  storage or in a built image before requesting another training GPU. Await
  OpenAI access and a verified small-Qwen endpoint; do not silently fall back
  to the original 27B service or claim a light-model comparison has started.
