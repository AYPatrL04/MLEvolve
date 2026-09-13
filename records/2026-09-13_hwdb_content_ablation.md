# Fresh-Agent HWDB Content Ablation

## Question and Scope

Does revised HWDB content improve agent choices and verified valid-node yield,
or reduce genuinely buggy GPU executions, under otherwise identical settings?
This first ablation isolates **HWDB content only**. Current prompt rendering,
feature filters, strict mode allowlists, validators, adapters, logging fixes and
execution machinery are held fixed. It is not an original-software-versus-fork
benchmark, nor does it isolate the effects of filter changes. A later separately
controlled filtering ablation is still needed for that part of the question.

Original content: captured JustinLinKK `hardware-awared` revision
`93371dd64b8e2b888c1bde7cb9d90d7c03ac4e5d`, only
`schema/hardware_knowledge_graph.json`. Revised content: the same file from the
pinned experiment source. Both are separate immutable snapshots, selected by a
checksum-guarded override. Active schema files and old experiment evidence are
never overwritten. Graph-backed stores are disabled; static retrieval uses the
chosen snapshot. Each candidate's prompt audit records its actual graph hash.

## Registered Protocol

- Dataset: Disaster Tweets public split, remote storage only. No held-out labels.
- Hardware: one A10 or A100, same GPU for all cells, A10 preferred.
- Agent: DeepSeek V4.1 Flash (`deepseek-flash`) for configured code/feedback roles.
- Factors: original/revised content x conservative/normal x search seeds 42/43.
  Eight fresh searches, counterordered within seed/mode pairs; no frozen model,
  inherited journal, cross-arm lesson memory or shared result root.
- Search seeds are paired, but hosted LLM output is not guaranteed deterministic.
- Conservative is strict IEEE FP32. Normal permits FP32 or selective FP16 and
  does not force AMP. BF16/TF32/quantized paths remain prohibited in both arms.
- Ten primary generation-attempt slots per cell: **80 primary attempts total**.
  A slot includes the existing bounded review/repair process. Rejected candidates,
  no-candidate outcomes and generation errors retain their slots. Repair rounds
  are recorded separately, not counted as independent generated-node samples.
- After ten primary attempts, a cell without a verified valid node continues
  in a separately labeled extension until at least one valid node is obtained.
  Extensions never enter the primary comparison denominator. No overall 3-hour
  cutoff; individual execution and API/preflight timeouts remain in force.
- Three consecutive operational errors stop a cell for investigation, without
  declaring success or blindly restarting. Repeated review rejections are not
  classified as operational errors and are inspected by the hourly monitor.
- A cell stops only after the primary budget and one valid-node goal are met.
  No live source/prompt hot patches across arms. Material fixes require a new
  source-pinned experiment; historical runs stay preserved.

## Evidence and Interpretation

Every attempt is saved before it begins and after it finishes. Candidate code,
stage plans, policy decisions, HWDB prompt audits, review history and runtime
observations are retained before and after dispatch. Existing per-repair
preflight snapshots remain intact. Submitted GPU packets are separate from
actual observed CUDA work. A valid node must pass the same independent milestone
verification and have matching HWDB-source audit evidence.

Primary outcomes include valid yield per attempt, generation errors, review and
preflight rejections, GPU submissions and unverified submitted results. Timing
records distinguish the serial generation-loop origin from process startup.
Normal-mode policy selections are compared with runtime autocast and TF32 flags,
completed/skipped updates, validation quality, and elapsed time.

**Do not equate a rejection with a genuine bug or false positive.** Confirmed
buggy-execution and validator-false-positive rates remain null until independent
review of the retained source and evidence. Do not execute rejected code by
bypassing admission just to obtain a denominator. CPU PASS does not establish
CUDA correctness, nor does INCONCLUSIVE mean PASS.

Input hashes are checked between cells and at completion. Hardware profile/cache
and scheduler state use separate result roots. A shared pretrained-download cache
can warm over time; counterordering helps but timing must still distinguish
downloads/setup from generation and training. This is a two-search-seed pilot,
not a definitive causal generalization across workloads or hardware.

## Deployment

- Jobs: `hwdb-content-ablation-20260913-prepare` then
  `hwdb-content-ablation-20260913-run` in Nautilus namespace `ecepxie`.
- Root: `/experiment/hwdb-content-ablation-20260913` on existing public/results
  PVC subPath `aypatrl04-hwdb-deepseek-20260911` of `yuze-li-vol`.
- Same pinned NVIDIA PyTorch image and immutable dependencies from the completed
  precision experiment. CPU preparation publishes only an updated repo archive,
  avoiding another full dependency archive write.
- Only existing DeepSeek Secret reference is supplied; no Kaggle credentials or
  held-out volume is mounted. CPU regression tests, filtered-content contrasts,
  live agent smoke and frozen-model CPU sanity check gate GPU allocation.
- Local targeted suite: **130 passed**. Remote results not yet claimed here.
- Historical 12-dataset/milestone searches remain suspended; completed controlled
  precision results are not reused as new agent-search observations.

## Inspect

`results/matrix.json` tracks cells. Each cell has `runtime.log` and
`runs/*/logs/ablation/{attempts,summary}.json`. Candidate evidence lives under
`runs/*/logs/ablation/candidates/<node-id>/{before_dispatch,after_dispatch,exception}/`.
Graph hashes and pre-filter contrast evidence live in `graph_manifest.json` and
`guidance_contrast.json`. The post-filter rendered prompt is retained in each
candidate's `hardware_prompt_audit`; actual injection must be checked before
attributing a result to the HWDB content difference.

## Launch Verification

- Experiment code pushed to fork at
  `7f3a36da257c0545db0196555051c416fdf54ad9`; both jobs pin this exact revision.
- CPU pod `hwdb-content-ablation-20260913-prepare-rbp7j` completed on
  `node-1-3.sdsc.optiputer.net`. **186 Linux tests passed in 116.68 seconds**;
  filtered-content contrasts, live DeepSeek smoke and model CPU sanity also
  passed. No GPU was requested for preparation.
- Original graph SHA256:
  `1922daa288aa8286dd59a5240ca3803a4e40e379efa24d423de4cfe47231c57f`.
  Revised graph SHA256:
  `a9a0b501879f609bd11f896de99091cac4f774584cbb3aecf3ca1c87e5734af5`.
- GPU pod `hwdb-content-ablation-20260913-run-54jfv` is Running on
  `gpu-15.nrp.mghpcc.org`; nvidia-smi confirms **NVIDIA A10**, 23028 MiB.
  Runtime, updated repository and MLE-bench source checksums passed.
- First cell `42-conservative-original` initialized run
  `20260913_072038_nlp-getting-started_conservative`. DeepSeek metric-direction
  query succeeded, correctly maximizing binary F1. Global memory is disabled;
  six per-search context packs were prepared. This is startup/agent evidence,
  not yet a completed candidate or a measured HWDB benefit.
- Hourly thread monitor `hwdb-agent-ablation-checks` is active. It is restricted
  to this experiment, cannot resume historical jobs, and must not hot-patch
  one arm or relabel unadjudicated rejections as genuine bugs. GPU utilization
  can legitimately be low during agent generation/review; report the real phase.

## Interim Inspection: First Cell Complete

Read-only inspection after approximately five hours of job runtime on September
13 UTC. The experiment remains pinned to `7f3a36d`; no arm was patched or restarted.

| Cell | Finished primary attempts | Review rejected | GPU submitted | Verified valid |
| --- | ---: | ---: | ---: | ---: |
| Seed 42, conservative, original | 10 | 4 | 6 | 4 |
| Seed 42, conservative, revised | 2 | 2 | 0 finished; attempt 3 submitted | 0 |

The other six cells remain queued. The original cell required no extension and
reached its first verified valid node after 4230 seconds (70.5 minutes) from the
generation-loop origin. Its four verified nodes reported FP32 parameters and
optimizer state, disabled autocast/TF32, completed CUDA updates, and validated
submission artifacts. Their F1 scores were 0.754967, 0.158730, 0.196621 and
0.757835: passing the execution contract does not establish useful model quality.

The two unverified original-cell GPU results are materially different:

- Attempt 2, node `fd84e0e443c84de2a95f98d4243c5db6`, crashed inside
  `utils/training_diagnostics.py:122` when reading `torch.backends.cudnn.allow_tf32`.
  Generated code selected newer `fp32_precision` controls; the installed PyTorch
  rejected the legacy aggregate getter because cuDNN conv/RNN precision flags
  differed. This is an observed precision-control/diagnostics integration failure,
  not evidence that FP32 training itself failed or that HWDB content caused it.
- Attempt 10, node `3fd780283faf4b048827e79f6b6c9975`, completed with F1 0.772313
  and a validated submission artifact, but lacked runtime diagnostics proving
  completed CUDA optimizer updates and actual precision. It is not a confirmed
  runtime crash and does not meet this study's GPU-verification objective.

Both initial revised-cell rejections name the same unproven checkpoint-receiver
contract. Source inspection found model/optimizer save and restore calls in both.
The checker in `engine/script_introspection.py:704` actually requires every
syntactic `state_dict` receiver to appear among `load_state_dict` receivers,
including diagnostics objects, not just model/optimizer as its message says:

- `d9c994f082384da69533dcb4cf19ede6`: an extra `diagnostics.state_dict()` save,
  while the restoration helper uses `diag.load_state_dict()`.
- `6e58a48bb00b4ed0a33aadf6a3b66c6b`: saves through
  `diagnostics_holder['obj']`, restores through `diagnostics`, and explicitly
  assigns the diagnostics object to that holder.

This identifies a literal-name/alias limitation in the rejection rationale.
It does not prove either entire rejected program is bug-free or establish an
overall false-positive rate; neither was executed by bypassing admission.
Three of four original-cell rejections also include this receiver check; one of
those additionally reports a TF32-policy violation. The remaining rejection
includes missing diagnostics and scheduler/RNG/data-position contracts.

First-candidate prompt text hashes in both arms were recomputed and matched the
saved audit. Original/revised graph identities matched their assigned arms;
revised audit entries contain nonempty rendered hardware prompts and filtered
contexts. Comprehensive prompt-content attribution remains part of final analysis.

The A10 utilization snapshot was 0% with 351 MiB allocated while revised attempt
3 was awaiting/completing scheduler execution; a snapshot alone does not diagnose
a hang. The job was Running and the third candidate had been submitted. Hourly
monitoring remains active. No comparative HWDB benefit or reduced bug rate can
yet be inferred from these incomplete, framework-confounded results.
