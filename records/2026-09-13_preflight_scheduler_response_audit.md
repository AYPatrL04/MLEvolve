# Preflight and scheduler response-style audit

Follow-up: [implemented fixes and final validation](2026-09-14_preflight_scheduler_feedback_report.md). This document preserves the original findings.

Verdict: **partly aligned, but not yet consistently short, comprehensive and
precise**. The status and error-origin contracts are useful. The largest gaps
are in the adapters that turn diagnostics, scheduler context and execution logs
into agent prompts. Some verbose evidence survives while some decision-relevant
facts disappear. This is an audit; runtime behavior was not changed.

Audited parent commit: `4da2d2a` (`refactor database and stuff`). Evidence:
[machine-readable audit](2026-09-13_preflight_scheduler_response_audit.json) and
[offline reproducer](../benchmarks/audit_feedback_style.py).

## Findings

### P1: The new concise view drops scheduler limits and operational feedback

The scheduler's aggregate response retains `runtime_estimate`, `risk_flags` and
the hardware context's `scheduler_limits`. However, the v2 prompt projection in
[hardware_records](../agents/design_knowledge.py) emits GPU identity, precision
policy, backend and selected knowledge records without projecting the safe VRAM
budget, runtime-estimate availability, or the scheduler's risk/diagnosis fields.
Additionally, `_compact_runtime_estimate` in
[hardware_context.py](../agents/hardware_context.py) drops the reason an estimate
is unavailable.

In the deterministic fixture, changing the safe budget from **31,744 MiB to
12,000 MiB**, replacing an unavailable runtime estimate with a numeric estimate,
and changing the risk flag produced an **identical 418-character prompt**. The
raw API still held those values. The scheduler continues to enforce its own
admission rules; the demonstrated defect is incomplete advice to the LLM. This
gap is in the newly introduced concise projection.

Recommended correction: include short operational records for authoritative
limits, estimate availability/reason, applicable risks and measured/estimated
resource facts. Retain units and provenance. Operational scheduler state should
remain authoritative in the scheduler, separate from hardware/lesson databases.

### P2: Preflight-to-agent conversion omits useful structured evidence and risk warnings

[diagnostic_to_review_issue](../engine/preflight.py) copies the diagnostic message,
scenario, traceback and reproduction command, but not `diagnostic.evidence`.
For the checker's unsupported-precision diagnostic, this loses the provided
supported-dtype list and target-profile reference. A useful fallback can thus be
absent when optional hardware knowledge is disabled.

Ordinary estimated-risk diagnostics such as `MEM001` are intentionally excluded
from repair issues. That correctly avoids labeling an estimate as a confirmed
candidate defect, but there is no separate concise advisory projection into the
result parser: its preflight section receives only `node.review_issues`.
Diagnostic codes, report paths and the GPU-check-required flag remain in the
journal; the full report retains the risk evidence.

Recommended correction: preserve selected structured facts and a separate
advisory list. A memory estimate must retain its uncertainty and must not become
an automatic OOM assertion or code-repair instruction.

### P2: A single preflight cause is repeated across repair issues

Two archived CPU reports were passed through the current conversion and existing
issue-deduplication functions:

| CPU case | Checker diagnostics | Repair issues after deduplication | Serialized repair issues | Terminal summary |
| --- | ---: | ---: | ---: | ---: |
| Construction raises the same ValueError | 5 | 4 | 4,568 characters | 960 characters |
| Fixture raises the same missing-column KeyError | 5 | 4 | 5,225 characters | 934 characters |

Different scenarios and stage-specific traceback text prevent exact-evidence
deduplication in [append_review_issue](../agents/review_contracts.py). The repair
prompt in [stage_repair.py](../agents/stage_repair.py) serializes every issue.
Tracebacks are clipped to their last 4,000 characters, scenarios to 1,500 and
reproduction strings to 500; those are excerpts, not semantic summaries.

Recommended correction: one cause record per exception/site with all affected
stages and distinct failing conditions grouped underneath it; one repair action
and an evidence reference. Preserve independent causes and exceptions rather
than merging solely by diagnostic code. Keep full tracebacks in artifacts.

### P2: Execution-log clipping does not reliably preserve important events

[SearchNode.term_out](../engine/search_node.py) uses
[trim_long_string](../utils/response.py): above 5,100 characters it keeps the first
and last 2,500 characters. [Result parsing](../agents/result_parse_agent.py) places
that text in the ordinary prompt.

An **8,955-character** synthetic training log became **5,039 characters**. The
final validation score survived, but a numerical-warning event in the middle
disappeared. Truncation is explicitly marked, and the raw log remains available;
this still fails the intended comprehensive-feedback criterion. The fixture
demonstrates information loss, not the frequency of this problem in real runs.

Recommended correction: extract status, failure origin, terminal exception,
metric/direction, relevant numerical warnings and resource outcomes into concise
records before prompt assembly. Keep a log reference and use targeted excerpts
only when a record cannot explain the failure.

## What already fits the intended style

- Preflight distinguishes `PASS`, `FAIL`, `INCONCLUSIVE` and `INTERNAL_ERROR`,
  with explicit admission and outstanding GPU-validation state.
- Fixture failures remain warnings rather than proven model defects.
- Scheduler/executor failures retain their origin and request unchanged-code
  retries instead of unsupported model, metric or submission repairs.
- The result parser explicitly requests a 2–3 sentence observation summary and
  separates weak model quality from executable-code bugs. This is an output
  instruction, not proof that every model response obeys it.
- Full checker reports, scenarios, tracebacks and logs remain diagnostic evidence.

## Proposed response shape

Use the same small set of fields across the prompt adapters: **outcome and
classification; cause and affected conditions; next action; relevant constraints
and measured/estimated facts; evidence reference**. Keep each distinct fact or
cause complete. Deduplication and semantic selection should replace fixed
character clipping; required task/code interfaces must remain intact.

Illustrative preflight response:

> FAIL — confirmed candidate defect. Model construction raises
> `ValueError: bad model configuration`; the same cause blocks the training and
> validation checks. Repair the real model constructor while preserving the
> adapter contract, then rerun CPU preflight. Evidence: report diagnostic refs.

Illustrative scheduler response from the fixture:

> Backend: `cuda_process`; safe VRAM budget: 31,744 MiB. Runtime estimate is
> unavailable because no valid profile matches this backend. Reported risk:
> high VRAM pressure. The scheduler retains control of admission and batch
> resolution; use current branch profiles and telemetry. Evidence: context refs.

## Verification and scope

**198 focused tests passed** in 26.03 seconds:

```bash
PATH="$PWD/.venv/bin:$PATH" \
PYTHONPATH="$PWD/nn-model-preflight-checker/src:$PWD" CUDA_VISIBLE_DEVICES="" \
.venv/bin/python -m pytest \
  tests/test_model_preflight_integration.py tests/test_stage_review_workflow.py \
  tests/test_hardware_context.py tests/test_component_modularity.py \
  tests/test_training_diagnostics.py localml_scheduler/tests/test_client_surface.py \
  localml_scheduler/tests/test_executor_scheduler_bridge.py -q
```

The offline audit also asserted the demonstrated omissions, repeated issues and
middle-log information loss. Reproduce with:

```bash
PYTHONPATH="$PWD/nn-model-preflight-checker/src:$PWD" CUDA_VISIBLE_DEVICES="" \
.venv/bin/python benchmarks/audit_feedback_style.py
```

The report uses archived CPU diagnostics projected through current code and
deterministic scheduler/log fixtures. No LLM request, GPU experiment, live
scheduler operation or live database mutation was performed. Passing existing
tests verifies their contracts; it does not establish concise or complete
responses. Only the audit script and report artifacts were added.
