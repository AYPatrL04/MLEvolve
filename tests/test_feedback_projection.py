import copy
import json
from types import SimpleNamespace

import pytest

from agents.design_knowledge import hardware_records
from agents.hardware_context import HardwarePromptContext, compact_optimization_context
from engine.preflight import PreflightOutcome, apply_outcome_to_node, project_preflight_diagnostics
from engine.search_node import SearchNode
from knowledge.records import render_records
from utils.feedback import execution_feedback, preflight_feedback, render_execution_feedback, scheduler_feedback


def diagnostic(**overrides):
    return {"code": "CON001", "classification": "confirmed_candidate_failure", "stage": "construction",
            "message": "construction raised ValueError: bad model configuration", "exception_type": "ValueError",
            "stack_trace": 'File "candidate.py", line 6, in build_model\nValueError: bad model configuration',
            "scenario": {"batch_size": 2, "precision": "fp32"}, **overrides}


def test_preflight_groups_causes_preserving_all_scenarios_and_source_locations():
    diagnostics = [diagnostic(), diagnostic(stage="cpu_training", scenario={"batch_size": 1, "precision": "fp32"}),
                   diagnostic(stage="validation"), diagnostic(classification="inconclusive", code="CHK001")]
    issues, advisories = project_preflight_diagnostics(diagnostics, report_ref="report_attempt_1.json")
    assert len(issues) == len(advisories) == 1
    evidence = issues[0].evidence
    assert "line 6" in evidence and "build_model" in evidence
    assert '"batch_size": 1' in evidence and '"batch_size": 2' in evidence
    assert "construction, cpu_training, validation" in evidence
    for i in range(3):
        assert f"report_attempt_1.json#/diagnostics/{i}" in evidence
    assert advisories[0]["classification"] == "inconclusive"


@pytest.mark.parametrize("change", [
    {"stack_trace": 'File "candidate.py", line 8, in build_model'},
    {"message": "different error"}, {"evidence": {"requested_precision": "bf16"}},
])
def test_distinct_preflight_causes_are_not_merged(change):
    issues, _ = project_preflight_diagnostics([diagnostic(), diagnostic(**change)])
    assert len(issues) == 2


def test_identical_structured_data_contract_failure_groups_across_checks():
    diagnostics = [{"code": "DAT002", "classification": "confirmed_candidate_failure",
                    "message": "No target tensor found", "evidence": {"expected_target_dtype": "float32"},
                    "stage": stage} for stage in ("data_contract", "cpu_training", "validation")]
    issues, _ = project_preflight_diagnostics(diagnostics)
    assert len(issues) == 1
    assert "data_contract, cpu_training, validation" in issues[0].evidence
    diagnostics[-1]["evidence"]["expected_target_dtype"] = "int64"
    assert len(project_preflight_diagnostics(diagnostics)[0]) == 2


def test_precision_feedback_lists_all_sites_in_one_repair_issue():
    from agents.precision_validation import validate_training_precision

    agent = SimpleNamespace(acfg=SimpleNamespace(precision_optimization_mode="conservative"))
    code = "import numpy as np\nsums = np.zeros(3, dtype=np.float64)\npreds = preds.astype(np.float64)\n"
    issues = validate_training_precision(agent, code)
    assert len(issues) == 1
    assert "line 2" in issues[0].evidence and "line 3" in issues[0].evidence
    assert "every listed occurrence" in issues[0].repair_instruction
    assert validate_training_precision(agent, code.replace("float64", "float32")) == ()


def test_structured_dtype_facts_and_memory_uncertainty_survive_without_raw_streams():
    facts = {"native_training_dtypes": ["float32", "float16"], "target_profile": "nvidia/v100_32gb",
             "captured_stdout": "unrelated progress\n" * 10000}
    issues, advisories = project_preflight_diagnostics([
        diagnostic(code="GPU001", stage="hardware", evidence=facts),
        {"code": "MEM001", "classification": "risk", "stage": "memory", "message": "Analytical memory risk.",
         "evidence": {"target_vram_fraction": 0.92, "is_gpu_oom_guarantee": False}},
    ])
    assert "native_training_dtypes" in issues[0].evidence
    assert "float16" in issues[0].evidence and "v100_32gb" in issues[0].evidence
    assert "unrelated progress" not in issues[0].evidence
    assert advisories[0]["facts"]["is_gpu_oom_guarantee"] is False
    assert advisories[0]["facts"]["target_vram_fraction"] == 0.92
    node = SearchNode(code="", plan="test", stage="draft")
    outcome = PreflightOutcome("PASS_WITH_WARNINGS", "full", "hash", True, True, advisories=advisories)
    apply_outcome_to_node(node, outcome, repair_count=0)
    assert preflight_feedback(node)["advisories"] == advisories
    assert outcome.to_dict()["advisories"] == advisories
    assert not node.review_issues


def test_repair_prompt_separates_advice_from_assigned_defects():
    from agents.stage_repair import _build_repair_prompt

    agent = SimpleNamespace(cfg=SimpleNamespace(experiment=SimpleNamespace(mode="baseline")),
                            acfg=SimpleNamespace(hardware_context_enabled=False), scheduler_client=None)
    node = SearchNode(code="model = None", plan="test", stage="draft")
    issues, advisories = project_preflight_diagnostics([
        diagnostic(), {"code": "MEM001", "classification": "risk", "stage": "memory",
                       "message": "High analytical memory use", "evidence": {"is_gpu_oom_guarantee": False}},
    ])
    node.diagnostics["preflight_advisories"] = advisories
    prompt = _build_repair_prompt(agent, node, node.code, issues[0].owner, issues)
    payload = json.loads(prompt.split("\n\n", 1)[1])
    assert len(payload["issues"]) == 1
    assert payload["preflight_advisories"][0]["facts"]["is_gpu_oom_guarantee"] is False
    assert "not additional repair requests" in payload["advisory_rule"]


def context(raw):
    return HardwarePromptContext(raw_context=raw, compact_context=compact_optimization_context(raw))


def test_scheduler_limits_risks_and_unavailable_reasons_are_prompt_visible():
    raw = {"hardware_context": {"scheduler_limits": {"safe_vram_budget_mb": 31744}},
           "runtime_estimate": {"found": False, "reason": "No matching runtime profile for cuda_process"},
           "risk_flags": ["high_vram_pressure"], "derived_diagnosis": {"profile_symptoms": ["high_vram_pressure"]}}
    prompt = render_records(hardware_records(context(raw)))
    assert "31744" in prompt and raw["runtime_estimate"]["reason"] in prompt
    assert prompt.count("high_vram_pressure") == 1
    changed = copy.deepcopy(raw)
    changed["hardware_context"]["scheduler_limits"]["safe_vram_budget_mb"] = 12000
    changed["runtime_estimate"] = {"found": True, "seconds_per_epoch": 12, "estimated_total_runtime_seconds": 60}
    changed_prompt = render_records(hardware_records(context(changed)))
    assert changed_prompt != prompt
    assert "12000" in changed_prompt and "60" in changed_prompt
    assert "estimate, not a measured runtime" in changed_prompt
    assert context(raw).compact_context["runtime_estimate"]["reason"] == raw["runtime_estimate"]["reason"]


def test_nested_profile_retains_measurements_and_labels_estimated_totals():
    raw = {"graph_evidence": {"exact_profiles": [{"ref": "profile:1", "kind": "runtime_profile",
            "summary_text": "Misleading prose is not proof of measurements.",
            "data": {"resolved_batch_size": 4, "seconds_per_epoch": 12, "estimated_total_runtime_seconds": 60}}]}}
    prompt = render_records(hardware_records(context(raw)))
    assert "resolved_batch_size" in prompt and "12" in prompt
    assert "Estimated total runtime (not measured): 60 seconds" in prompt
    assert "Measured matching" not in prompt
    raw["graph_evidence"]["exact_profiles"][0]["data"]["hardware_key"] = "other_gpu"
    assert "profile:1" not in render_records(hardware_records(context(raw)))


@pytest.mark.parametrize("hardware", [False, True])
def test_configured_scheduler_constraints_do_not_depend_on_knowledge(hardware):
    memory = SimpleNamespace(gpu_vram_gib=32, predicted_budget_fraction=0.96875,
                             live_admission_stop_fraction=0.98, live_admission_resume_fraction=0.95)
    client = SimpleNamespace(settings=SimpleNamespace(gpu_scheduler=SimpleNamespace(memory=memory, packing_backend="cuda_process")))
    agent = SimpleNamespace(scheduler_client=client, cfg=SimpleNamespace(hardware_knowledge={"enabled": hardware}))
    assert scheduler_feedback(agent)["safe_vram_budget_mb"] == 31744
    assert scheduler_feedback(SimpleNamespace(scheduler_client=None)) == {}


def test_full_log_events_survive_progress_deduplication_and_keep_raw_evidence():
    node = SearchNode(code="", plan="test", stage="draft")
    marker = "NUMERICAL_WARNING: nonfinite gradients at epoch 12; verify optimizer updates."
    raw = "Epoch completed normally.\n" * 170 + marker + "\n" + "Epoch completed normally.\n" * 170
    raw += marker + "\nFinal Validation Score: 0.75\n"
    node._term_out = [raw]
    feedback = execution_feedback(node)
    assert feedback["events"][0]["message"] == marker
    assert feedback["events"][0]["occurrences"] == 2
    assert feedback["reported_final_validation_score"] == 0.75
    assert len(render_execution_feedback(node)) < len(raw)
    assert node._term_out == [raw]


def test_optimizer_skips_and_precision_observations_survive_later_clean_reports():
    node = SearchNode(code="", plan="test", stage="draft")
    sample = {"schema_version": 1, "attempted_updates": 1, "completed_updates": 0,
              "skipped_updates": 1, "autocast_dtypes_observed": ["torch.float16"]}
    latest = {**sample, "attempted_updates": 2, "completed_updates": 2, "skipped_updates": 0,
              "autocast_dtypes_observed": [], "settings": {"physical_batch_size": 4, "raw_log": "raw" * 10000}}
    node._term_out = ["MLEVOLVE_TRAINING_DIAGNOSTICS " + json.dumps(x) + "\n" for x in (sample, latest)]
    feedback = execution_feedback(node)["runtime_measurements"]
    assert feedback["optimizer_skips_observed"]
    assert feedback["autocast_dtypes_observed"] == ["torch.float16"]
    assert feedback["completed_updates"] == 2
    assert feedback["settings"] == {"physical_batch_size": 4}


def test_backend_failure_and_unstructured_output_are_not_candidate_success_claims():
    node = SearchNode(code="", plan="test", stage="draft")
    node.exc_type = "ConnectionError"
    node.exc_info = {"failure_origin": "scheduler", "message": "service unavailable"}
    assert execution_feedback(node)["outcome"] == "backend_unavailable"
    node.exc_type, node.exc_info = None, None
    node._term_out = ["unrecognized terminal message"]
    assert "inspect" in execution_feedback(node)["uncertainty"]
    node._term_out = ['MLEVOLVE_EPOCH_METRIC {"epoch": 1, "metric": "not measured"}\n']
    assert execution_feedback(node)["invalid_diagnostic_records"] == 1
