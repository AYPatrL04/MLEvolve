"""Offline audit of current preflight/scheduler feedback projections; no model or GPU calls."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from agents.design_knowledge import hardware_records
from agents.hardware_context import HardwarePromptContext, compact_optimization_context
from agents.review_contracts import append_review_issue
from engine.preflight import diagnostic_to_review_issue, project_preflight_diagnostics
from utils.feedback import render_execution_feedback
from engine.search_node import SearchNode
from knowledge.records import render_records
from localml_scheduler.client import SchedulerClient
from model_preflight.core.results import PreflightReport
from model_preflight.reporting.text_report import render_text


def json_chars(value):
    return len(json.dumps(value, sort_keys=True, default=str))


def audit(archive: Path) -> dict:
    cases = []
    for path in sorted(archive.glob("*/report.json")):
        raw = json.loads(path.read_text())
        diagnostics = raw["diagnostics"]
        issues, advisories = project_preflight_diagnostics(diagnostics, report_ref=str(path))
        node = SearchNode(code="", plan="offline feedback audit", stage="draft")
        for issue in issues:
            append_review_issue(node, issue)
        cases.append({
            "source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "archived_report_chars": len(path.read_text()),
            "current_terminal_summary_chars": len(render_text(PreflightReport.from_dict(raw))),
            "diagnostic_count": len(diagnostics), "repair_issue_count": len(issues),
            "journal_issues_after_existing_dedup": len(node.review_issues),
            "repair_issues_chars": json_chars([issue.to_dict() for issue in issues]),
            "cause_occurrences": dict(Counter(item["message"] for item in diagnostics)),
            "per_issue_chars": [json_chars(issue.to_dict()) for issue in issues],
            "repair_issues": [issue.to_dict() for issue in issues], "advisories": advisories,
        })
    if not cases:
        raise ValueError("No archived CPU reports found")

    hardware_failure = {
        "code": "GPU001", "stage": "hardware", "classification": "confirmed_candidate_failure",
        "severity": "error", "message": "requested precision 'bf16' is unsupported by Tesla V100",
        "evidence": {"requested_precision": "bf16", "native_training_dtypes": ["float32", "float16"],
                     "target_profile": "nvidia/v100_32gb"},
        "reproduction": "model-preflight check preflight.yaml --only hardware",
    }
    precision_issue = diagnostic_to_review_issue(hardware_failure)
    memory_warning = {
        "code": "MEM001", "stage": "memory", "severity": "warning", "classification": "risk",
        "message": "analytical memory estimate is 92.0% of target VRAM; this is not a CUDA OOM guarantee",
        "evidence": {"target_vram_fraction": 0.92, "estimated_risk": "high", "is_gpu_oom_guarantee": False},
    }
    diagnostic_projection = {
        "hardware_source": hardware_failure, "hardware_issue": precision_issue.to_dict(),
        "supported_dtype_evidence_retained": "native_training_dtypes" in json.dumps(precision_issue.to_dict()),
        "memory_advisories": project_preflight_diagnostics([memory_warning])[1],
        "memory_source": memory_warning, "memory_warning_forwarded_as_issue": diagnostic_to_review_issue(memory_warning) is not None,
    }

    profile_context = {
        "effective_backend": "cuda_process", "runner_contract": "subprocess_job_v1",
        "hardware_context": {"hardware": {"gpu_name": "NVIDIA GeForce RTX 5090", "architecture": "blackwell", "total_vram_mb": 32768},
                             "scheduler_limits": {"safe_vram_budget_mb": 31744}},
        "runtime_estimate": {"found": False, "reason": "No valid runtime profile for the requested backend"},
        "risk_flags": ["high_vram_pressure"],
        "derived_diagnosis": {"profile_symptoms": ["high_vram_pressure"], "optimization_targets": ["reduce_vram"]},
        "confidence": 0.0,
    }
    scheduler = SchedulerClient.__new__(SchedulerClient)
    scheduler.get_profile_evidence = lambda **kwargs: copy.deepcopy(profile_context)
    scheduler.get_code_optimization_context = lambda **kwargs: {"vector_evidence": {}, "evidence_refs": []}
    scheduler.get_stage_hardware_features = lambda *args, **kwargs: {}
    candidate = {"agent_stage": "improve", "effective_backend": "cuda_process", "runner_contract": "subprocess_job_v1"}
    raw = scheduler.get_optimization_context(candidate=candidate)

    def project(value):
        compact = compact_optimization_context(value)
        context = HardwarePromptContext(candidate=candidate, raw_context=value, compact_context=compact)
        return compact, render_records(hardware_records(context, role="improve"))

    compact, prompt = project(raw)
    altered = copy.deepcopy(raw)
    altered["hardware_context"]["scheduler_limits"]["safe_vram_budget_mb"] = 12000
    altered["runtime_estimate"] = {"found": True, "seconds_per_epoch": 120, "estimated_total_runtime_seconds": 600}
    altered["risk_flags"] = ["fixture_alternative_risk"]
    _, alternate_prompt = project(altered)
    scheduler_projection = {
        "fixture_raw": raw, "compact_runtime_estimate": compact["runtime_estimate"],
        "current_v2_prompt": prompt, "raw_chars": json_chars(raw), "prompt_chars": len(prompt),
        "safe_vram_budget_retained": "31744" in prompt,
        "runtime_unavailable_reason_retained": raw["runtime_estimate"]["reason"] in prompt,
        "risk_flag_retained": "high_vram_pressure" in prompt,
        "prompt_unchanged_after_budget_runtime_risk_changes": prompt == alternate_prompt,
    }

    marker = "NUMERICAL_WARNING: nonfinite gradients detected at epoch 12; check optimizer updates."
    output = "Epoch completed normally.\n" * 170 + marker + "\n" + "Epoch completed normally.\n" * 170 + "Final Validation Score: 0.75\n"
    node = SearchNode(code="", plan="offline log projection audit", stage="draft")
    node._term_out = [output]
    log_projection = {
        "raw_chars": len(output), "agent_visible_chars": len(render_execution_feedback(node)),
        "middle_warning_retained": marker in render_execution_feedback(node),
        "final_metric_retained": "reported_final_validation_score" in render_execution_feedback(node),
        "truncation_reported": "characters truncated" in render_execution_feedback(node),
    }

    assert all(case["repair_issue_count"] == 1 for case in cases)
    assert diagnostic_projection["supported_dtype_evidence_retained"]
    assert diagnostic_projection["memory_advisories"][0]["facts"]["is_gpu_oom_guarantee"] is False
    assert not diagnostic_projection["memory_warning_forwarded_as_issue"]
    assert not scheduler_projection["prompt_unchanged_after_budget_runtime_risk_changes"]
    assert all(scheduler_projection[key] for key in ("safe_vram_budget_retained", "runtime_unavailable_reason_retained", "risk_flag_retained"))
    assert log_projection["middle_warning_retained"] and log_projection["final_metric_retained"]
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "evidence_tier": "Archived CPU reports projected through current functions plus deterministic fixtures; no LLM, GPU or live scheduler/database calls",
        "preflight_cases": cases, "diagnostic_projection": diagnostic_projection,
        "scheduler_projection": scheduler_projection, "log_projection": log_projection,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("runs/hwdb-disabled-feedback-audit-20260913/fixed-feedback/working/preflight"))
    parser.add_argument("--output", type=Path, default=Path("records/2026-09-14_preflight_scheduler_feedback_fixed.json"))
    args = parser.parse_args()
    result = audit(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"preflight": [{key: case[key] for key in ("source", "diagnostic_count", "repair_issue_count", "journal_issues_after_existing_dedup", "repair_issues_chars", "current_terminal_summary_chars")} for case in result["preflight_cases"]],
                      "scheduler_prompt_chars": result["scheduler_projection"]["prompt_chars"], "log_projection": result["log_projection"]}, indent=2))


if __name__ == "__main__":
    main()
