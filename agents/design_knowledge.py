"""Concise knowledge projection, shared across draft and later agent prompts."""

from __future__ import annotations

import re
import json
from typing import Any, Mapping

from knowledge.records import STAGE_TOPICS, from_source, select_records, strings


def scheduler_records(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project operational facts without promoting estimates to measurements."""
    records = []
    limits = (raw.get("hardware_context") or {}).get("scheduler_limits") or {}
    facts = {key: limits[key] for key in ("safe_vram_budget_mb", "vram_role", "placement_objective", "packing_backend", "runner_contract", "exclusive_fallback_enabled") if limits.get(key) is not None}
    if facts:
        records += from_source({"summary": "Scheduler constraints: " + json.dumps(facts, sort_keys=True) + ". The scheduler owns admission and concurrency; a memory budget is not a throughput target.",
                                "strength": "hard", "verified": True, "confidence": 1.0,
                                "evidence_refs": ["runtime:scheduler_limits"]}, domain="hardware", source_id="scheduler_limits", topics=["runtime"])
    estimate = raw.get("runtime_estimate") or {}
    if estimate:
        profile = estimate.get("profile") or {}
        values = {key: estimate.get(key, profile.get(key)) for key in ("found", "reason", "source", "match_reason", "matched_exact_batch_size", "seconds_per_epoch", "estimated_total_runtime_seconds")}
        values = {key: value for key, value in values.items() if value is not None}
        records += from_source({"summary": "Runtime estimate: " + json.dumps(values, sort_keys=True) + ". This is an estimate, not a measured runtime for the new candidate; missing evidence does not establish a fit.",
                                "evidence_refs": strings(raw.get("evidence_refs")) or ["runtime:runtime_estimate"]},
                               domain="hardware", source_id="runtime_estimate", topics=["runtime"])
    diagnosis = raw.get("derived_diagnosis") or {}
    seen = set()
    for field, label, values in (
        ("risk", "Reported risk", raw.get("risk_flags")),
        ("symptom", "Profile symptom", diagnosis.get("profile_symptoms")),
        ("target", "Suggested optimization target", diagnosis.get("optimization_targets")),
        ("recommendation", "Conditional recommendation", raw.get("recommendations")),
    ):
        for value in strings(values):
            if value in seen:
                continue
            seen.add(value)
            records += from_source({"summary": f"{label}: {value}", "verification_status": "advisory",
                                    "warnings": ["Verify this risk before assuming safe execution; it is not a confirmed candidate defect."] if field in {"risk", "symptom"} else [],
                                    "when_to_use": "Revalidate applicability to this candidate; preserve task and precision constraints.",
                                    "evidence_refs": strings(raw.get("evidence_refs")) or [f"runtime:{field}"]},
                                   domain="hardware", source_id=f"scheduler_{field}", topics=["training", "runtime"])
    return records


def runtime_scope(context: Any) -> dict[str, Any]:
    compact = getattr(context, "compact_context", {}) or {}
    candidate = getattr(context, "candidate", {}) or {}
    raw_hardware = ((getattr(context, "raw_context", {}) or {}).get("hardware_context") or {}).get("hardware") or {}
    hardware = {**raw_hardware, **((compact.get("hardware_context") or {}).get("hardware") or {})}
    policy = compact.get("precision_policy") or {}
    accelerator = re.sub(r"[^a-z0-9]+", "_", str(hardware.get("gpu_name") or hardware.get("name") or "").lower()).strip("_")
    def major_minor(value):
        matched = re.match(r"\d+\.\d+", str(value or ""))
        return matched.group() if matched else value
    return {
        "hardware_key": hardware.get("hardware_key") or hardware.get("hardware_id"),
        "hardware_id": hardware.get("hardware_id"),
        "accelerator_names": [accelerator, accelerator.removeprefix("nvidia_"), accelerator.removeprefix("nvidia_geforce_")] if accelerator else [],
        "gpu_architecture": hardware.get("architecture"),
        "compute_capability": hardware.get("compute_capability") or policy.get("compute_capability"),
        "framework": "pytorch", "framework_major_minor": major_minor(hardware.get("torch_version")),
        "cuda_major_minor": major_minor(hardware.get("cuda_runtime") or hardware.get("toolkit_version")),
        "driver_major_minor": major_minor(hardware.get("driver_version")),
        "backend_mode": compact.get("effective_backend") or candidate.get("effective_backend"),
        "runner_contract": compact.get("runner_contract") or candidate.get("runner_contract"),
        "model_family": candidate.get("model_family"),
        "workload_type": candidate.get("workload_type") or candidate.get("task_type"),
        "precision_mode": policy.get("mode"),
        "allowed_precision_policies": policy.get("allowed_policies", []),
    }


def hardware_records(context: Any, *, role: str = "draft") -> list[dict[str, Any]]:
    from agents.hardware_context import _precision_evidence_allowed
    from utils.precision_policy import resolve_precision_policy

    raw = getattr(context, "raw_context", {}) or {}
    compact = getattr(context, "compact_context", {}) or {}
    if not raw and not compact:
        return []
    policy_data = compact.get("precision_policy") or {}
    policy = resolve_precision_policy(policy_data, mode=policy_data.get("mode", "normal"))
    scope = runtime_scope(context)
    records: list[dict[str, Any]] = []
    records += scheduler_records(raw or compact)
    local = (compact.get("hardware_context") or {}).get("hardware") or {}
    if local:
        summary = "Hardware: " + "; ".join(f"{k}={local[k]}" for k in ("name", "gpu_name", "architecture", "compute_capability", "total_vram_mb") if local.get(k) is not None) + "."
        records += from_source({"summary": summary, "strength": "hard", "verified": True,
                                "evidence_refs": ["runtime:hardware"], "confidence": 1.0}, domain="hardware", source_id="runtime", topics=["runtime"])
    if policy_data:
        records += from_source({"summary": "Allowed precision policies: " + ", ".join(policy_data.get("allowed_policies") or ["disabled"]) + ". Use FP32 if a supported training path is unavailable.",
                                "strength": "hard", "verified": True, "confidence": 1.0, "evidence_refs": ["runtime:precision_policy"]},
                               domain="hardware", source_id="precision_policy", topics=["precision"])
    if scope.get("backend_mode"):
        records += from_source({"summary": f"Execution backend: {scope['backend_mode']}; runner contract: {scope.get('runner_contract') or 'unspecified'}. Scheduler-owned controls remain owned by the scheduler.",
                                "strength": "hard", "evidence_refs": ["runtime:backend"], "verified": True, "confidence": 1.0},
                               domain="hardware", source_id="backend", topics=["runtime"])

    def add(source: Mapping[str, Any], topics=None):
        source = dict(source)
        stored = source.get("design_records_v2")
        records.extend(stored if isinstance(stored, list) else from_source(source, domain="hardware", topics=topics))

    stage_context = raw.get("stage_hardware_features") or compact.get("stage_hardware_features") or {}
    stages = stage_context.get("stages") or []
    for stage in stages:
        topics = STAGE_TOPICS.get(stage.get("stage"))
        node = stage.get("node") or {}
        if node:
            add({**node, "evidence_refs": strings(stage_context.get("evidence_refs")) or [f"hardware:stage:{stage.get('stage')}"]}, topics)
        for feature in stage.get("features") or []:
            add(feature, topics)
    if not stages:
        for feature in stage_context.get("features") or []:
            add(feature)
    for key in ("code_knowledge", "code_knowledge_results", "selected_hardware_features"):
        value = raw.get(key) or compact.get(key) or []
        if isinstance(value, dict):
            value = value.get("results") or value.get("records") or []
        for item in value:
            if isinstance(item, Mapping):
                add(item)
    if not raw.get("design_records_v2"):
        for group in (raw.get("vector_evidence") or {}).values():
            if isinstance(group, list):
                for source in group:
                    if isinstance(source, Mapping):
                        add(source)
    for profile in (raw.get("graph_evidence") or {}).get("exact_profiles") or []:
        ref = profile.get("ref") or profile.get("evidence_ref")
        if not ref:
            continue
        data = profile.get("data") or profile
        values = {key: data[key] for key in ("resolved_batch_size", "epoch_seconds", "epoch_1_seconds", "seconds_per_epoch", "startup_seconds", "avg_step_time_ms", "steps_per_epoch", "runtime_seconds", "peak_vram_mb", "peak_vram_mib", "avg_sm_utilization_pct", "backend_name", "precision", "hardware_key", "model_family", "strategy", "observations") if data.get(key) is not None}
        summary = "Matching profile observation: " + json.dumps(values, sort_keys=True) + "."
        if data.get("estimated_total_runtime_seconds") is not None:
            summary += f" Estimated total runtime (not measured): {data['estimated_total_runtime_seconds']} seconds."
        if not values:
            continue
        add({"summary": summary, "record_id": ref, "evidence_refs": [ref],
             "verified": data.get("verified") is True or (type(data.get("observations")) is int and data["observations"] > 0),
             "confidence": data.get("confidence", 0.0),
             "when_to_use": "Historical evidence, not a guaranteed numeric default; preserve the current task budget and remeasure after changes.",
             "warnings": strings(profile.get("transferability")),
             "backend_modes": [data["backend_name"]] if data.get("backend_name") else [],
             "hardware_keys": [data["hardware_key"]] if data.get("hardware_key") else [],
             "model_families": [data["model_family"]] if data.get("model_family") else []}, ["training", "runtime"])
    for option in raw.get("model_options") or []:
        if not option.get("evidence_refs") or not option.get("confidence"):
            continue
        add({**option, "summary": option.get("rationale"), "model_families": [option.get("model_family")], "record_id": option.get("model_family")}, ["architecture"])
    # Some backends already return the canonical records directly.
    records.extend(raw.get("design_records_v2") or [])
    allowed_records = []
    for record in select_records(records, scope, role=role):
        text = record["summary"]
        if not _precision_evidence_allowed({"summary_text": text}, policy):
            continue
        if any(required not in policy.allowed_policies and re.search(pattern, text, re.I)
               for pattern, required in ((r"\b(?:bf16|bfloat16)\b", "bf16_amp"), (r"\b(?:fp16|float16)\b", "fp16_amp"), (r"\btf32\b", "tf32"))):
            continue
        allowed_records.append(record)
    return select_records(allowed_records, scope, role=role)


def history_records(agent: Any) -> list[dict[str, Any]]:
    records = []
    root = getattr(agent, "virtual_root", None)
    for node in getattr(root, "children", []) or []:
        code = str(getattr(node, "code", "") or "")
        from engine.script_introspection import introspect_training_script

        facts = introspect_training_script(code)
        family = facts.get("model_family") or facts.get("model_key") or "unspecified model"
        metric = getattr(getattr(node, "metric", None), "value", None)
        summary = f"Previous attempt {node.id}: {family}; validation metric={metric}; buggy={bool(getattr(node, 'is_buggy', False))}."
        plan = str(getattr(node, "plan", "") or "").strip()
        if plan:
            summary += " Proposed approach: " + re.split(r"(?<=[.!?])\s+", plan, maxsplit=1)[0]
        records += from_source({"summary": summary, "evidence_refs": [f"node:{node.id}"], "verified": True, "confidence": 1.0},
                               domain="history", source_id=str(node.id), topics=["architecture", "evaluation"])
    memory = getattr(agent, "global_memory", None)
    if memory is not None:
        seen_refs = {ref for item in records for ref in item["evidence_refs"]}
        for record, _score in memory.retrieve_similar_records(str(agent.task_desc), top_k=2):
            for item in record.design_records_v2:
                if not seen_refs.intersection(item.get("evidence_refs") or []):
                    records.append(item)
                    seen_refs.update(item.get("evidence_refs") or [])
    return records


def lesson_records(context: Any) -> list[dict[str, Any]]:
    compact = context.compact_context
    if "design_records_v2" in compact:
        return compact["design_records_v2"]
    # Resumed v1 runs keep their original lookup, but the single-call draft
    # still needs a concise projection of that lookup's evidence.
    view = compact.get("family_hardware_profile") or {}
    if view.get("match_level", "none") == "none":
        return []
    common = {"evidence_refs": view.get("evidence_refs", []), "warnings": view.get("warnings", []),
              "confidence": view.get("confidence", 0.0), "verification_status": "advisory",
              "when_to_use": "Revalidate this historical profile's model, hardware and runtime assumptions before reuse."}
    records = from_source({**common, "baseline": view.get("baseline", {})}, domain="lesson",
                          source_id=f"baseline:{view.get('profile_key')}", revision=int(view.get("revision") or 1), topics=["architecture", "training"])
    for lesson in view.get("relevant_lessons") or []:
        records += from_source({**lesson, **common, "evidence_refs": lesson.get("evidence_refs") or common["evidence_refs"]},
                               domain="lesson", revision=int(view.get("revision") or 1))
    return records


def cuda_records(context: Any) -> list[dict[str, Any]]:
    if not getattr(context, "applicable", False):
        return []
    records = []
    for chunk in getattr(context, "evidence_chunks", []) or []:
        if getattr(chunk, "support_status", "") == "unsupported":
            continue
        records += from_source({"record_id": getattr(chunk, "chunk_id", None), "text": chunk.text,
                                "source_url": chunk.source_url, "support_status": chunk.support_status,
                                "applicability": getattr(chunk, "applicability", {})}, domain="cuda")
    return records
