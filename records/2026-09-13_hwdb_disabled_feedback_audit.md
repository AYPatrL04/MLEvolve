The initial audit below records behavior before the fixes. The implementation follow-up at the end records the corrected behavior and validation.

The initial analysis confirmed that `hardware_knowledge.enabled=false` did not disable hardware feedback, and identified several ways feedback could direct the agent toward unnecessary or poorly informed repairs. These were reproduced response defects and plausible quality mechanisms; this audit does not establish their effect on validation scores.

Scope: the user's confirmed switch is `hardware_knowledge.enabled=false`. Source inspected: MLEvolve `4e0609dd8c2f138f56c4b9b6d5a2dd7b032ce053`; the existing checker checkout is `57f199c5446ba5c36f509fd8a808e67848853919`, while the parent gitlink records `d90852756346976c3f4e488ff8daffe172801ac6`. The existing submodule state was preserved. All reproductions used local CPU checks, fixed A100 hardware metadata, disabled external databases, and a 31 GiB scheduler memory setting. No scheduler service, training job, GPU experiment, or model API call was started.

1. **High priority: the requested disable flag is disconnected from the runtime.**

   `config/__init__.py:311` retains `hardware_knowledge` as a mapping, but `run.py:44` builds the scheduler from `scheduler.settings`, and `run.py:351` attaches `SchedulerClient` directly to the agent. Neither consumes `hardware_knowledge.enabled`. The prompt gates in `agents/hardware_context.py:1948` and `agents/coder/stepwise_coder.py:533` check experiment mode and `agent.hardware_context_enabled` instead. The independent `HardwareKnowledgeClient` is not instantiated by this run path.

   Reproduction: changing only `hardware_knowledge.enabled` between false and true produced byte-identical, nonempty 3,497-character improvement context blocks. The preflight repair prompt also contained hardware evidence with the flag false. Even additionally disabling `graph_db`, `hardware_knowledge_graph`, and `hardware_feature_db` left stage features available from the bundled JSON graph: `localml_scheduler/client.py:772` intentionally reads the static graph without consulting those database flags. Thus “database off” does not mean “hardware guidance off.”

   Consequence: a comparison that changes only the user's switch is not a hardware-feedback ablation. It also does not prevent scheduler-owned stores from using their independently configured settings.

2. **High priority: scheduler infrastructure failures become candidate repair instructions.**

   When no execution-result file exists, `engine/executor.py:1280` returns a generic `RuntimeError`, a status-reason string, and an empty stack. The single-job path duplicates this at `engine/executor.py:1716`. The result-parser prompt receives the script and terminal text (`agents/result_parse_agent.py:716`), without a structured indication of whether candidate execution ever started.

   I simulated a worker failing before launch. Even with a parser answer of `is_bug=false`, `_determine_buggy` deterministically added both of these critical issues because no metric or submission could exist (`agents/result_parse_agent.py:310`):

   ```json
   [
     {"source":"runtime_validator","severity":"critical","category":"missing_metric","owner":"training_evaluation","repair_instruction":"Compute the exact task metric and print it in the required final score format."},
     {"source":"runtime_validator","severity":"critical","category":"missing_submission","owner":"training_evaluation","repair_instruction":"Generate ./submission/submission.csv from real test inference using the required columns."}
   ]
   ```

   This excerpt omits each issue's evidence string. `agents/debug_agent.py:121` prioritizes existing critical issues for repair, so the next agent can be asked to change metric/submission code even though the candidate never ran. This defect is independent of HWDB, but remains present in the requested disabled configuration. The reproduction does not claim that a database outage actually triggered a worker failure in a particular run.

3. **High priority: preflight keeps useful failure evidence on disk but drops it from repair input.**

   A real CPU construction failure produced `CON001`, `FAIL`, and `admitted=false`, correctly owned by `model_design`. The checker diagnostic included `exception_type`, `stack_trace`, `scenario`, and `reproduction`. Its scenario identified batch size 2, train mode, FP32, and the fixture shape information. The worker did not populate separate file/line fields; the candidate location existed only in the traceback (`nn-model-preflight-checker/src/model_preflight/execution/worker_main.py:809`).

   `engine/preflight.py:711` converts the diagnostic to this actual six-field repair object:

   ```json
   {
     "source": "model_preflight",
     "severity": "critical",
     "category": "preflight_con001",
     "owner": "model_design",
     "evidence": "[CON001] construction raised ValueError: bad model configuration",
     "repair_instruction": "Repair the confirmed construction defect [CON001] and preserve the CandidateAdapter contract; do not suppress the check or replace real training behavior with mocks."
   }
   ```

   `_build_repair_prompt` in `agents/stage_repair.py:114` includes these issues, task context, pipeline decisions, stage notes, hardware evidence, ownership guidelines, and the complete script. It does not load the report or include its traceback, failing scenario, reproduction command, or report path. The agent must return raw SEARCH/REPLACE blocks. Some exceptions receive useful hand-written repair hints, but the general path loses the precise failure location and inputs.

   This is especially relevant to large generated scripts with several adapters or repeated model calls. It is not caused by disabling HWDB; the disabled switch also fails to remove unrelated hardware evidence from that same repair prompt.

4. **Medium priority: missing measurements are presented as precision diagnoses.**

   `localml_scheduler/graph_knowledge.py:1722` emits `precision_not_optimized` and `tensor_core_not_used` whenever `uses_amp` is false or absent. This does not require a runtime profile, measured kernel behavior, or even a known precision policy. Missing AMP is insufficient evidence for either conclusion.

   With no saved profiles and every external database disabled, the response was:

   ```json
   {
     "derived_diagnosis": {
       "profile_symptoms": ["precision_not_optimized", "tensor_core_not_used"],
       "optimization_targets": ["improve_precision_efficiency", "enable_tensor_core", "improve_throughput"]
     },
     "risk_flags": ["no_matching_profiles_for_current_hardware"],
     "confidence": 0.0
   }
   ```

   The datatype/precision prompt rendered these symptoms explicitly through `agents/hardware_context.py:1709`. In this reproduction they were omitted from some other stage prompts by filtering or the character budget, so exposure is stage-dependent. The repair/model-generation process can nevertheless receive a confident-sounding diagnosis alongside a zero-confidence score and be encouraged to change precision without measurement.

5. **Medium priority: generic backend confidence can look like model-selection confidence.**

   `localml_scheduler/client.py:1390` creates fallback model options even without model-specific evidence. Each reproduced option had `score=0.05`, `confidence=0.0`, and a rationale acknowledging missing evidence. The enclosing response still returned `found=true` and `confidence=0.9`, because it takes the maximum of model-option and generic backend-rule confidence (`localml_scheduler/client.py:1414`). The backend rules come from local seed records even when vector storage is disabled (`localml_scheduler/code_knowledge/store.py:514`).

   `compact_model_design_context` at `agents/hardware_context.py:1330` retains the combined confidence but drops `backend_guidance`, `effective_backend`, and `runner_contract` from that top-level response. The rendered design brief showed all three displayed model options at confidence 0.0, followed by an unqualified `Confidence: 0.9`. It also recommends a conservative architecture when evidence is weak. This is valid backend knowledge presented with ambiguous scope, rather than evidence that a particular model should be preferred.

6. **Medium priority: an adapter data-construction failure can produce no repair feedback.**

   A second real CPU check used an adapter whose batch builder raised `KeyError("missing_feature_column")`. The checker reported `FIX001`/`CHK001`, `INCONCLUSIVE`, and `gpu_check_required=true`. Under the configured balanced policy, MLEvolve admitted it with `issues=[]`.

   Fixture exceptions are classified as inconclusive in `nn-model-preflight-checker/src/model_preflight/execution/worker_main.py:793`; `diagnostic_to_review_issue` drops inconclusive diagnostics, except the specific offline-weight dependency case. `_run_node_preflight` repairs only `FAIL` outcomes with issues (`engine/agent_search.py:663`). This policy can reasonably avoid blaming the model for checker-fixture limitations, but it also suppresses the actionable adapter exception from repair input. An admitted candidate therefore must not be interpreted as having passed all CPU checks.

   `gpu_check_required` is propagated into scheduler metadata; it is not itself an independently enforced canary switch. Normal batch-probe configuration controls the probe. The existing tests cover metadata propagation, not unconditional canary execution.

The current response flow, including behavior that works correctly, is:

| Situation | Response and routing |
| --- | --- |
| Confirmed CPU candidate defect | Full checker report on disk; compact critical `ReviewIssue` sent to the owning repair stage; repaired source is rechecked. |
| Unresolved preflight rejection | GPU submission avoided; node gets `analysis`, terminal summary, worst metric, and `is_buggy=true`, then is removed from the active tree and excluded from the node count (`engine/agent_search.py:360`). Failure traces remain in pipeline logging; it is not retained as an ordinary journal node for later debug selection. |
| Checker infrastructure error | Normally `INTERNAL_ERROR`, admitted with GPU checking required, and no candidate issue; deterministic contract issues can still force rejection. |
| Inconclusive CPU result | Balanced policy admits; strict policy rejects; only confirmed failures normally generate repair issues. |
| Scheduler result file exists | `ExecutionResult` carries `term_out`, `exec_time`, `exc_type`, `exc_info`, and `exc_stack`, independent of HWDB. |
| Scheduler result file absent | Generic runtime error and status reason; candidate-vs-infrastructure origin is not a dedicated discriminator. |
| Agent interprets execution | Structured response requires `is_bug`, `summary`, `metric`, `lower_is_better`, and `issues`; deterministic post-processing can add missing-output issues. |

The scheduler's optimization-context dictionary contains `hardware_context`, `graph_evidence`, `derived_diagnosis`, `stage_hardware_features`, `vector_evidence`, `recommendations`, `risk_flags`, `evidence_refs`, and `confidence`. It has no explicit top-level HWDB availability field. The aggregate wrapper also drops `runtime_estimate`, `effective_backend`, and `runner_contract` returned by `get_profile_evidence` (`localml_scheduler/client.py:1181`); the reproducer observed a lower-level runtime estimate of `{"found":false,"reason":"insufficient evidence"}` disappearing entirely. Some backend identity remains nested under hardware context. A truly disabled store, an empty evidence store, and a caught lookup exception can therefore look similar downstream.

The smallest corrective work, in priority order, would be to wire the existing disable flag to every agent-facing hardware-context and feature-selection entry point; preserve failure origin before result parsing so infrastructure errors do not create candidate code-repair tasks; append bounded candidate traceback and scenario information to the existing `ReviewIssue.evidence` string; and keep measurement availability and model-specific confidence distinct from generic rules. Fixture uncertainty should retain actionable adapter feedback without being falsely labeled a proven model defect. These are recommendations only; runtime source was not changed.

For an immediate comparison, `agent.hardware_context_enabled=false` suppresses the dynamic hardware-context lookup in this reproduction while leaving CPU preflight enabled in hardware-aware mode. It is not a complete substitute for wiring the user's switch: stage repair/review use experiment mode for some ownership and precision guidance. Switching to `experiment.mode=baseline` would additionally skip preflight by default and alter the stage layout, confounding this particular comparison. Keep those factors fixed when evaluating the response corrections.

Verification: the existing focused suite passed **165 tests in 19.69 seconds**:

```bash
PYTHONPATH="$PWD/nn-model-preflight-checker/src" .venv/bin/python -m pytest -q \
  tests/test_model_preflight_integration.py tests/test_stage_review_workflow.py \
  tests/test_hardware_context.py localml_scheduler/tests/test_client_surface.py \
  localml_scheduler/tests/test_executor_scheduler_bridge.py
```

The additional deterministic audit passed assertions for flag equivalence, static feature injection with disabled databases, construction failure/rejection, dropped traceback/scenario, missing aggregate fields, and incorrect repair issues after a simulated infrastructure failure. Its script, exact prompt text, CPU checker reports, and JSON evidence are retained under [runs/hwdb-disabled-feedback-audit-20260913](../runs/hwdb-disabled-feedback-audit-20260913/). Reproduce with:

```bash
PYTHONPATH="$PWD:$PWD/nn-model-preflight-checker/src" .venv/bin/python \
  runs/hwdb-disabled-feedback-audit-20260913/audit.py
```

The run artifacts are ignored by Git and remain local. The existing passing tests do not validate the missing HWDB-disable behavior or prove model quality. No affected run directory was supplied, so this report makes no claim about the frequency of these failures or the size of any score regression.

Implementation follow-up:

- The existing HWDB flag now gates agent context, hardware feature selection, hardware-aware repair ownership, context prewarming, and evidence-driven code tuning. The existing independent hardware client is attached separately from scheduler execution, allowing hardware guidance with direct subprocess execution.
- Preflight and scheduler prompt/contracts follow their own switches. Disabled preflight clears stale admission state; enabled preflight also guards direct execution. Scheduler hooks, batch elasticity metadata, and structured scheduler epoch markers are conditional on scheduling.
- Preflight repair issues retain bounded traceback, scenario, and reproduction evidence. Inconclusive fixture exceptions remain warnings but receive bounded targeted repair/recheck; unresolved uncertainty remains visible.
- Scheduler failures and missing results retain `exc_info.failure_origin=scheduler`; direct launch failures use `executor`. The parser records an execution warning, and debug retries unchanged code within existing search limits. Candidate exceptions continue through ordinary debugging.
- Missing AMP no longer invents tensor-core/precision diagnoses. Aggregate responses retain runtime-estimate availability and backend identity. Model-option confidence is independent of generic backend-rule confidence, whose provenance is retained in compact model-design context.

All eight combinations are covered by `tests/test_component_modularity.py`. The full root test suite, excluding opt-in live provider integration, plus focused scheduler bridge/context/runner/batching/backend/config suites passed **582 tests in 68.94 seconds**. Python compilation and `git diff --check` passed. The first broad run found only a shell-wrapper environment issue: the system Python lacked `humanize`; using the project virtual environment on `PATH` resolved it without changing the wrapper.

```bash
PATH="$PWD/.venv/bin:$PATH" \
PYTHONPATH="$PWD/nn-model-preflight-checker/src" CUDA_VISIBLE_DEVICES="" \
MLEVOLVE_RUN_PROVIDER_CACHE_INTEGRATION=0 .venv/bin/python -m pytest -q \
  tests --ignore=tests/integration \
  localml_scheduler/tests/test_executor_scheduler_bridge.py \
  localml_scheduler/tests/test_client_surface.py \
  localml_scheduler/tests/test_backend_guidance.py \
  localml_scheduler/tests/test_graph_db_validation.py \
  localml_scheduler/tests/test_mlevolve_runner.py \
  localml_scheduler/tests/test_quality_safe_batching.py \
  localml_scheduler/tests/test_backend_mode_contract.py \
  localml_scheduler/tests/test_config_models.py
```

The original broken construction and fixture candidates were also rechecked through the real CPU checker with HWDB and scheduler disabled. Construction returned `FAIL` with traceback/scenario/reproduction in the repair prompt. The fixture returned `INCONCLUSIVE` with an actionable warning. The updated verification script is `runs/hwdb-disabled-feedback-audit-20260913/verify_fixed_feedback.py`; reports and the exact repair prompt are under its `fixed-feedback/` directory. The earlier `audit.py` intentionally retains its pre-fix assertions and is a historical reproducer.

`gpu_check_required` still records outstanding target validation rather than force-enabling a scheduler probe. General training-quality, precision, metric, and submission checks remain active. No GPU/LLM quality experiment was run, and no score improvement is claimed. Legacy mode presets and `preflight.enabled_modes` remain applicable; use a fixed `hardware_aware` mode for the eight component combinations, as documented in `docs/model_preflight_admission.md`. Existing submodule state and concurrent generation/configuration changes were preserved.
