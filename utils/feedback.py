"""Deterministic prompt feedback from complete diagnostic and execution evidence."""

from __future__ import annotations

import json
import math
import re
from typing import Any


def traceback_site(text: str, *, file=None, line=None) -> dict[str, Any]:
    frames = re.findall(r'File "([^"\n]+)", line (\d+), in ([^\n]+)', str(text or ""))
    candidate_frames = [frame for frame in frames if "candidate" in frame[0].lower() or "preflight_adapter" in frame[0]]
    if file:
        return {"file": str(file), "line": line}
    if candidate_frames or frames:
        path, number, function = (candidate_frames or frames)[-1]
        return {"file": path, "line": int(number), "function": function.strip()}
    return {}


def raw_execution_output(node: Any) -> str:
    output = getattr(node, "_term_out", None)
    if output is None:
        return str(getattr(node, "term_out", "") or "")
    return output if isinstance(output, str) else "".join(str(part) for part in output)


def execution_feedback(node: Any) -> dict[str, Any]:
    """Keep outcomes and distinct significant events, never a clipped log window.

    Raw output stays on the node. Repeated progress and metadata are represented
    by their latest measurements; warning/error occurrences retain source lines.
    """
    from utils.training_diagnostics import TRAINING_DIAGNOSTICS_MARKER

    output = raw_execution_output(node)
    lines = output.splitlines()
    origin = (getattr(node, "exc_info", None) or {}).get("failure_origin")
    exception_type = getattr(node, "exc_type", None)
    result: dict[str, Any] = {
        "outcome": "backend_unavailable" if origin in {"scheduler", "executor"} else "execution_failed" if exception_type else "execution_returned",
        "failure_origin": origin or ("candidate" if exception_type else None),
        "evidence_ref": f"node:{getattr(node, 'id', 'unknown')}:execution_log",
    }
    if getattr(node, "exec_time", None) is not None:
        result["elapsed_seconds"] = node.exec_time
    if exception_type:
        info = getattr(node, "exc_info", None) or {}
        message = info.get("message") or info.get("error")
        if not message:
            message = next((text.strip() for text in reversed(lines) if text.strip().startswith(str(exception_type) + ":")), None)
        result["exception"] = {"type": exception_type, "message": message, **traceback_site(output)}
    scores = re.findall(r"Final\s+Validation\s+Score\s*:\s*([^\s]+)", output, re.I)
    if scores:
        try:
            score = float(scores[-1])
            result["reported_final_validation_score"] = score if math.isfinite(score) else "nonfinite"
        except ValueError:
            result["reported_final_validation_score"] = "unparseable"
    if getattr(getattr(node, "metric", None), "maximize", None) is not None:
        result["metric_direction"] = "maximize" if node.metric.maximize else "minimize"

    events = {}
    epoch_metrics = []
    runtime = None
    observed_dtypes = set()
    skipped_observed = False
    observed_statuses = set()
    invalid_diagnostics = 0
    latest_progress = None
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith(TRAINING_DIAGNOSTICS_MARKER + " "):
            try:
                sample = json.loads(stripped.split(" ", 1)[1])
                if not isinstance(sample, dict) or sample.get("schema_version") != 1:
                    raise ValueError("unsupported diagnostics")
                counters = [sample.get(key) for key in ("attempted_updates", "completed_updates", "skipped_updates")]
                if any(value is not None and (type(value) is not int or value < 0) for value in counters):
                    raise ValueError("invalid optimizer counter")
                dtypes = sample.get("autocast_dtypes_observed") or []
                if not isinstance(dtypes, list) or not all(isinstance(value, str) for value in dtypes):
                    raise ValueError("invalid dtypes")
                settings = sample.get("settings") or {}
                if not isinstance(settings, dict):
                    raise ValueError("invalid settings")
                observed_dtypes.update(dtypes)
                skipped_observed |= bool(sample.get("skipped_updates"))
                if isinstance(sample.get("status"), str):
                    observed_statuses.add(sample["status"])
                runtime = {key: sample[key] for key in (
                    "status", "epoch", "exception_type", "attempted_updates", "completed_updates", "skipped_updates",
                    "unaccounted_optimizer_calls", "parameter_dtypes", "optimizer_state_dtypes", "grad_scaler_enabled",
                    "tf32_matmul_allowed", "tf32_cudnn_allowed", "learning_rates", "gpu", "device", "torch_version", "grad_scale",
                ) if sample.get(key) is not None}
                runtime["settings"] = {key: settings[key] for key in (
                    "physical_batch_size", "effective_batch_size", "planned_epochs", "steps_per_epoch",
                    "model_family", "split_seed", "precision", "precision_policy", "precision_reason",
                ) if key in settings}
            except (TypeError, ValueError):
                invalid_diagnostics += 1
            continue
        if "MLEVOLVE_EPOCH_METRIC" in stripped:
            try:
                sample = json.loads(stripped.split("MLEVOLVE_EPOCH_METRIC", 1)[1].lstrip(" :"))
                if not isinstance(sample, dict) or type(sample.get("metric")) not in (int, float) or not math.isfinite(sample["metric"]) or type(sample.get("epoch")) not in (int, float) or not math.isfinite(sample["epoch"]) or not isinstance(sample.get("metric_name", ""), str):
                    raise ValueError("invalid epoch metric")
                epoch_metrics.append({key: sample[key] for key in ("epoch", "metric", "metric_name") if key in sample})
            except (TypeError, ValueError):
                invalid_diagnostics += 1
            continue
        significant = re.search(r"\b(?:\w*(?:warning|error|exception)|non[-_ ]?finite|nan|overflow|out of memory|killed|timed out|timeout)\b|skipped.{0,30}(?:update|step)", stripped, re.I)
        resource = re.search(r"(?:resolved[_ ]?batch|peak[_ ]?(?:vram|memory)|safe[_ ]?vram|runtime estimate).{0,20}[:=]", stripped, re.I)
        if stripped.startswith(('File "', "raise ", "warnings.warn(")):
            continue
        if significant or resource:
            item = events.setdefault(stripped, {"message": stripped, "occurrences": 0, "first_line": number, "last_line": number})
            item["occurrences"] += 1
            item["last_line"] = number
        elif re.search(r"\bepoch\b", stripped, re.I):
            latest_progress = stripped
    if events:
        result["events"] = list(events.values())
    if runtime is not None:
        runtime["autocast_dtypes_observed"] = sorted(observed_dtypes)
        runtime["optimizer_skips_observed"] = skipped_observed
        runtime["statuses_observed"] = sorted(observed_statuses)
        runtime["counter_scope"] = "latest reported snapshot, not the sum of log records"
        result["runtime_measurements"] = runtime
    else:
        result["runtime_measurements"] = {"available": False}
    if epoch_metrics:
        result["epoch_metrics"] = {"reports": len(epoch_metrics), "first": epoch_metrics[0], "last": epoch_metrics[-1]}
    elif latest_progress:
        result["last_progress"] = latest_progress
    if invalid_diagnostics:
        result["invalid_diagnostic_records"] = invalid_diagnostics
    if not events and not scores and runtime is None:
        nonempty = [line.strip() for line in lines if line.strip()]
        result["unclassified_terminal_line"] = nonempty[-1] if nonempty else None
        result["uncertainty"] = "No structured outcome or recognized event was reported; inspect the referenced raw log before inferring success or a repair."
    return result


def render_execution_feedback(node: Any) -> str:
    return json.dumps(execution_feedback(node), ensure_ascii=False, sort_keys=True)


def preflight_feedback(node: Any) -> dict[str, Any]:
    result = {
        "status": getattr(node, "preflight_status", None),
        "admitted": getattr(node, "preflight_admitted", None),
        "gpu_check_required": getattr(node, "preflight_gpu_check_required", None),
        "evidence_ref": getattr(node, "preflight_report_path", None),
        "issues": [issue for issue in (getattr(node, "review_issues", None) or []) if issue.get("source") == "model_preflight"],
        "advisories": (getattr(node, "diagnostics", None) or {}).get("preflight_advisories", []),
    }
    return {key: value for key, value in result.items() if value is not None and value != []}


def scheduler_feedback(agent: Any) -> dict[str, Any]:
    """Mandatory execution constraints, independent of optional knowledge lookup."""
    client = getattr(agent, "scheduler_client", None)
    if client is None:
        return {}
    scheduler = getattr(getattr(client, "settings", None), "gpu_scheduler", None)
    memory = getattr(scheduler, "memory", None)
    total_gib = getattr(memory, "gpu_vram_gib", None)
    fraction = getattr(memory, "predicted_budget_fraction", None)
    budget = int(float(total_gib) * 1024 * float(fraction)) if total_gib is not None and fraction is not None else None
    return {
        "backend": getattr(scheduler, "packing_backend", None),
        "safe_vram_budget_mb": budget,
        "budget_status": "configured admission constraint" if budget is not None else "unavailable from configuration; scheduler resolves hardware capacity at admission",
        "live_admission_stop_fraction": getattr(memory, "live_admission_stop_fraction", None),
        "live_admission_resume_fraction": getattr(memory, "live_admission_resume_fraction", None),
        "rule": "Preserve scheduler checkpoint hooks. Admission and concurrency are scheduler-owned; VRAM is a safety constraint, not the optimization objective.",
        "evidence_ref": "runtime:scheduler_configuration",
    }
