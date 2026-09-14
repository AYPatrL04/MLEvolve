# Preflight before LLM review

Implemented and locally verified: the search controller now runs CPU preflight and its bounded targeted repairs before entering LLM code review. An unresolved admission rejection skips review. Admitted `INCONCLUSIVE` candidates can proceed under the existing policy; CPU admission is not GPU verification.

Review receives concise evidence for the exact checked source. Its internal repair/re-review rounds retain their previous behavior and mark earlier CPU evidence stale when code changes. The controller rechecks the final reviewed or scheduler-adjusted code before execution or deferral. Unchanged code reuses its existing preflight result.

## Changes

- `engine/agent_search.py`: changes ordering, skips review after preflight rejection, rechecks changed code, and clears only stale preflight-owned rejection state after fresh admission.
- `agents/code_review_agent.py`: includes current preflight evidence or an explicit stale marker and preserves preflight history.
- `engine/preflight.py`: retains distinct report copies across repeated checks of the same repair attempt; checker failure cannot attach an earlier report to a new candidate revision.
- `docs/model_preflight_admission.md`: documents the ordering, evidence, and unchanged-code behavior.
- `tests/test_preflight_review_order.py`: covers admission policy, targeted repair order, changed code, scheduler parameter edits, supplied/retry code, disabled components, rejection ownership, and report provenance.
- `benchmarks/preflight_review_matrix.py`: exercises the actual search controller with real draft, debug repair, review, and CPU checks. Shared transport in `benchmarks/petfinder_workflow_matrix.py` now permits a bounded timeout override; its existing default remains 240 seconds.

Scheduler admission policy, live jobs, branch profiles, database publication, and experiment configuration were not changed. Existing unrelated workspace changes were preserved.

## Verification

**318 tests passed in 21.52 seconds** on the final runtime code. Syntax compilation and `git diff --check` passed. The focused suite covers the new ordering plus precision, preflight, review/repair, hardware context, independent components, training diagnostics, concise knowledge, CUDA documentation integration, and scheduler client/executor surfaces.

The real-agent matrix uses the same small synthetic PetFinder-style image/tabular task and RTX 5090 fixture knowledge as the earlier audit. Each completed attempt generates a draft once. A controlled exception is inserted into the real `CandidateAdapter.training_step`; with preflight repair rounds set to zero for this rejection check, the controller must reject without calling review. The controller then runs its actual debug path, CPU admission, and LLM review on the repaired candidate. Preflight repair rounds return to the normal one-round setting for this path. No execution callback is permitted.

| Precision | HWDB | Draft attempts | Repair calls | Review calls | Final assertions | Active audit seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| normal | off | 1 | 1 | 1 | 17/17 | 284.6 |
| normal | on | 1 | 1 | 1 | 17/17 | 200.3 |
| conservative | off | 1 | 1 | 1 | 17/17 | 304.9 |
| conservative | on | 2 | 2 | 1 | 17/17 | 468.2 |

The initial conservative/HWDB-on transport timed out at 240 seconds before producing a candidate. Its failed result remains in the original cell directory. One retry, stored under `retry_1/conservative_hwdb_on`, completed with a 360-second per-call transport limit. The table and combined chart count the failed attempt; timings include active attempts and exclude idle gaps between attempts. Each completed attempt used one draft-generation transport call. Claude CLI envelopes also record auxiliary Haiku usage, so this is not a claim of one total internal provider inference per draft.

All four final candidates have approved LLM reviews and PASS results for construction, data contract, CPU training step, and validation checks. Overall preflight remains `INCONCLUSIVE`, admitted with `gpu_check_required=true`. HWDB-on draft prompts include each architecture, precision, and training record once with applicability/fallbacks; HWDB-off cases make zero optional hardware retrieval calls. Final conservative scripts pass the existing precision guard. Reviews preserve CUDA uncertainty and optional best-checkpoint/validation-tracking findings as warnings where applicable.

All completed real reviews approved without editing code. Focused regression tests separately force review patches and pre-submit parameter edits, proving the changed source is rechecked and a failed recheck prevents execution. The tests also confirm that admitted unchanged code does not incur another CPU check.

## Evidence and limits

[Combined Gantt/metric-node PNG](matrix.png), [all-attempt matrix](matrix.json), and [independent Luna-medium audit](luna_audit.md). Cell directories retain exact prompts, responses, usage, candidate revisions, review history, and CPU reports with SHA-256 mappings.

Raw prompt and response files preserve their original whitespace and SEARCH/REPLACE markers
byte for byte. Source-file whitespace checks passed; the raw text artifacts retain those
expected whitespace and patch-marker findings.

Observed LLM reviews took 30.5–42.1 seconds, while individual CPU gate calls took approximately 13–19 seconds. The call-order evidence confirms that blocked candidates receive no review. This is not a matched before/after latency benchmark or proof of model-quality improvement; provider variability and injected faults limit such conclusions.

No GPU jobs, full training entrypoints, scheduler admission experiments, or production execution-result-parser lifecycle ran. Hardware knowledge is fixture evidence, not live database verification. CPU checks do not establish CUDA compatibility or GPU performance.

To regenerate the combined artifact from saved attempts without new model calls:

```bash
PATH="$PWD/.venv/bin:$PATH" \
PYTHONPATH="$PWD/nn-model-preflight-checker/src:$PWD" \
CUDA_VISIBLE_DEVICES="" \
.venv/bin/python benchmarks/preflight_review_matrix.py --resume
```
