# Preflight and scheduler feedback fixes

Agent-facing feedback now retains scheduler limits, runtime-estimate availability and reasons, risks, structured preflight evidence, and significant execution events. Raw reports, tracebacks and logs remain source artifacts. No scheduler admission policy, job, branch profile, experiment configuration or database publication was changed.

## Replay evidence

| Archived CPU case | Repair issue count | Repair text characters |
| --- | ---: | ---: |
| broken-construction | 4 -> 1 | 4,568 -> 1,647 |
| broken-fixture | 4 -> 1 | 5,225 -> 1,724 |

Each repeated failure is grouped by cause, candidate location, evidence and classification. All distinct scenarios/checks and report references survive. Inconclusive checks and analytical memory risks remain separate advisories, not confirmed defects.

The synthetic execution log is 8,955 characters. Its previous 5,039-character excerpt lost a middle numerical warning; the new 430-character semantic summary retains the warning and final metric. Scheduler context increased from 418 to 1,371 characters because it now includes previously missing facts. Changing the budget, runtime estimate or risks changes the prompt. These are overhead and information-retention measurements, not model-quality results.

Evidence: [before](2026-09-13_preflight_scheduler_response_audit.json), [after](2026-09-14_preflight_scheduler_feedback_fixed.json), [reproducer](../benchmarks/audit_feedback_style.py).

## Local verification

300 focused tests passed in 22.55 seconds, covering precision, preflight, hardware context, review/repair, component modularity, training diagnostics, concise knowledge, CUDA documentation integration and scheduler client/executor surfaces. Syntax checks and git diff whitespace checks passed. No GPU jobs or full training entrypoints were executed.

The real-agent matrix uses an authenticated Sonnet text-only transport, four normal/conservative x hardware-knowledge on/off cells, fixture RTX 5090 evidence and real CPU preflight. Detailed prompts, responses, reports, counts and measured timings are recorded by [the harness](../benchmarks/petfinder_workflow_matrix.py). Hardware fixtures are not live database validation. The Luna-medium Codex subagent audits agent outputs independently.

## Final workflow matrix

All four final candidates pass the CPU construction, data contract, training-step and validation checks, with valid precision policies. Overall checker status remains INCONCLUSIVE/admitted because CUDA branches require GPU confirmation. The new draft target-fixture instruction and complete precision-location feedback address problems exposed by these runs.

| Mode | HWDB | Initial MLE draft calls | Feedback / repair calls | Final assertions | Active audit seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| normal | off | 1 | 1 / 1 | 14/14 | 221.9 |
| normal | on | 1 | 1 / 1 | 13/13 | 252.5 |
| conservative | off | 1 | 1 / 2 | 14/14 | 232.1 |
| conservative | on | 1 | 1 / 3 | 15/15 | 373.9 |

Initial drafts were not uniformly compliant. The normal/HWDB-on draft omitted fixture targets and was repaired from grouped DAT002 evidence. The conservative/HWDB-on draft used FP64 for evaluation and preprocessing; its first precision patch was incomplete. The precision validator now reports every forbidden site in one issue, and the next repair removed the remaining FP64 casts. Prior results and patches remain under each cell's continuation directories. The conservative/HWDB-off run also needed two stage-owned repair calls; later repair workflow behavior is retained.

HWDB-on prompts contain each architecture, precision and training record once, including applicability and fallbacks; HWDB-off cases made zero optional hardware retrieval calls. Fixture hardware records are advisory and separate from live databases. Each draft used one MLE generation transport call, with no MLE decision, selector, summarizer, staged coding or merge calls. The Claude CLI envelope also reports auxiliary Haiku usage, so one transport call is not a claim of one total internal provider request. CPU feedback consumes actual checker evidence through the execution-result adapter. No GPU inference, training run, scheduler admission experiment, or leaderboard metric was measured. Reported timings include active continuation phases and exclude pauses between them; they are overhead evidence only.

The CPU-only execution-result adapter also activates the parser's existing missing-metric and missing-submission guards. Those records reflect the unexecuted training entrypoint, not demonstrated candidate defects. The harness retains them in its audit output and uses the CPU gate's authoritative grouped issues for admission repairs. This matrix therefore validates generated feedback and CPU repair behavior, not the complete production execution-parser lifecycle.

[Combined Gantt/metric-node PNG](2026-09-14_petfinder_workflow_matrix_final/matrix.png), [matrix results](2026-09-14_petfinder_workflow_matrix_final/matrix.json), [independent Luna-medium audit](2026-09-14_petfinder_workflow_matrix_final/luna_audit.md). Per-call prompts, model responses, raw CLI usage and SHA-256-mapped CPU report copies are retained beside each result.

## Changed code

- `engine/preflight.py` groups repeated causes, retains diagnostic facts and references, and exports advisories; `engine/agent_search.py` clears those advisories when preflight is disabled.
- `utils/feedback.py` assembles semantic full-log feedback and mandatory scheduler configuration. Draft, feedback, debug, repair, improvement, evolution, fusion and leakage prompts use the relevant concise views.
- `agents/design_knowledge.py` retains scheduler budgets, risks and estimate status; `agents/hardware_context.py` keeps unavailable reasons and reads complete logs for resource evidence.
- `agents/precision_validation.py` reports all incompatible sites together. The draft prompt explains target creation when fixture shapes omit targets.
- Focused regression tests and replay/workflow harnesses cover these behaviors. No fixed knowledge-token or character cap was added.
