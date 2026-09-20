"""Source-backed design knowledge shared by the independent knowledge domains.

Raw evidence stays in its source store. This contract contains complete, short
claims, never excerpts cut to a character or token budget.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
from typing import Any, Mapping

SCHEMA_VERSION = "design-knowledge-v2"
TOPICS = {"architecture", "precision", "training", "evaluation", "runtime"}
STAGE_TOPICS = {
    "model_design": ["architecture"],
    "datatype_precision": ["precision"],
    "training_evaluation": ["training", "evaluation", "runtime"],
}
ROLE_TOPICS = {
    "model_design": {"architecture"}, "datatype_precision": {"precision"},
    "training_evaluation": {"training", "evaluation", "runtime"},
    "debug": {"precision", "training", "evaluation", "runtime"},
    "code_review": TOPICS, "review": TOPICS,
}


def strings(value: Any) -> list[str]:
    if value is not None and not isinstance(value, (list, tuple, set)):
        value = [value]
    return list(dict.fromkeys(" ".join(str(item).split()) for item in value or [] if str(item).strip()))


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def source_identity(source: Mapping[str, Any]) -> str:
    return str(next((source[key] for key in ("record_id", "lesson_id", "feature_id", "chunk_id", "recipe_id", "api_symbol_id", "rule_id", "id") if source.get(key)), None) or digest(source))


def validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(record))
    required = {"record_id", "revision", "domain", "category", "summary", "topics", "applicability", "applies_when", "restrictions", "fallbacks", "evidence_refs", "confidence", "verification_status", "strength", "source_id", "source_hash"}
    if required.difference(result):
        raise ValueError("Incomplete design knowledge record")
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported design knowledge schema")
    if not result.get("record_id") or not result.get("summary"):
        raise ValueError("Design knowledge requires identity and a complete summary")
    for field in ("topics", "applies_when", "restrictions", "fallbacks", "evidence_refs"):
        if not isinstance(result[field], list) or any(not isinstance(value, str) or not value.strip() for value in result[field]):
            raise ValueError(f"Knowledge {field} must be an array of non-empty strings")
    if not set(result.get("topics") or []).issubset(TOPICS) or not result.get("topics"):
        raise ValueError("Design knowledge requires semantic topics")
    if not result.get("evidence_refs"):
        raise ValueError("Design knowledge requires source evidence")
    confidence = float(result.get("confidence", 0.0))
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Knowledge confidence must be finite and between zero and one")
    if result.get("verification_status") not in {"verified", "advisory", "conflicted", "stale", "unsupported"}:
        raise ValueError("Invalid knowledge verification status")
    if not isinstance(result.get("applicability"), dict):
        raise ValueError("Knowledge applicability must be an object")
    if int(result.get("revision", 0)) < 1:
        raise ValueError("Knowledge revision must be positive")
    return result


def _summary_ready(text: str, *, explicit: bool) -> bool:
    # These are structural checks for unprocessed documents/logs, not size caps.
    if not text.strip() or "```" in text or re.search(r"(?m)^\s*(?:Traceback|File \"|def |class |import |#{1,6} )", text):
        return False
    return explicit or (len(text.split("\n\n")) == 1 and len(re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)) <= 3)


def from_source(source: Mapping[str, Any], *, domain: str, source_id: str | None = None,
                revision: int = 1, topics: list[str] | None = None) -> list[dict[str, Any]]:
    """Normalize already summarized fields; leave unprocessed prose for backfill.

    Independent recommended patterns become atomic records. Their applicability,
    prohibitions and provenance travel with them, even when retrieved separately.
    """
    source = dict(source)
    if source.get("schema_version") == SCHEMA_VERSION:
        return [validate_record(source)]
    content = source.get("content") or source.get("baseline") or {}
    identity = source_id or source_identity(source)
    refs = strings(source.get("evidence_refs"))
    refs += strings(source.get("evidence_ref"))
    refs += strings(source.get("ref"))
    refs += strings(source.get("source_url"))
    for ref in source.get("source_refs") or []:
        if isinstance(ref, Mapping):
            refs += strings(ref.get("url") or ref.get("path"))
    refs = strings(refs) or [f"{domain}:{identity}"]
    explicit = source.get("design_summary") or source.get("summary") or content.get("lesson") or content.get("model_summary")
    text = str(explicit or source.get("solution_summary") or source.get("usage_summary") or source.get("summary_text") or source.get("description") or source.get("text") or "").strip()
    patterns = [] if source.get("design_summary") else strings(source.get("recommended_patterns"))
    summaries = ([text] if _summary_ready(text, explicit=bool(explicit)) else []) + patterns
    checks = strings(content.get("checks"))
    if not summaries:
        summaries = checks
    record_topics = strings(topics or source.get("topics"))
    if not record_topics:
        for stage in source.get("pipeline_stages") or []:
            record_topics.extend(STAGE_TOPICS.get(stage, []))
    if not record_topics:
        label = str(source.get("category") or source.get("lesson_type") or "")
        record_topics = ["precision"] if re.search(r"precision|dtype|amp", label) else ["architecture"] if re.search(r"model|structure|family", label) else ["training", "runtime"]
    applicability = {k: v for k, v in dict(source.get("applicability") or {}).items() if v not in (None, "", [], {})}
    for key in ("hardware_keys", "hardware_id", "accelerator_names", "gpu_architectures", "compute_capabilities", "min_compute_capability", "frameworks", "framework_versions", "toolkit_versions", "driver_versions", "backend_modes", "runner_contracts", "model_families", "workload_types", "precision_modes", "agent_audiences"):
        if source.get(key):
            applicability[key] = source[key]
    if source.get("optimization_modes"):
        applicability["optimization_modes"] = source["optimization_modes"]
    if source.get("architectures"):
        applicability["gpu_architectures"] = source["architectures"]
    if source.get("framework"):
        applicability["frameworks"] = [source["framework"]]
    restrictions = strings(source.get("avoid_patterns")) + strings(source.get("warnings")) + strings(source.get("when_not_to_use")) + strings(source.get("limitations")) + strings(source.get("model_shape_limitations"))
    status = str(source.get("verification_status") or "")
    if not status:
        status = "unsupported" if source.get("support_status") == "unsupported" else "stale" if source.get("deprecated") else "conflicted" if source.get("maturity") == "conflicted" else "verified" if source.get("verified") is True or source.get("verified_source") is True else "advisory"
    records = []
    for summary in dict.fromkeys(summaries):
        if not _summary_ready(summary, explicit=bool(explicit) or summary in patterns or summary in checks):
            continue
        normalized = " ".join(summary.split())
        record = {
            "schema_version": SCHEMA_VERSION, "record_id": f"{domain}:{identity}:{digest(normalized)[:16]}",
            "revision": revision, "domain": domain, "category": str(source.get("category") or source.get("lesson_type") or source.get("record_type") or "guidance"),
            "topics": sorted(set(record_topics)), "summary": normalized,
            "applicability": applicability, "applies_when": strings(source.get("when_to_use")),
            "restrictions": strings(restrictions), "fallbacks": strings(source.get("fallbacks") or source.get("fallback_policy")),
            "evidence_refs": refs, "confidence": float(source.get("confidence", 0.0) or 0.0),
            "verification_status": status, "strength": source.get("strength", "informational"),
            "source_id": identity, "source_hash": digest(source),
        }
        if source.get("change_scope"):
            record["change_scope"] = source["change_scope"]
        records.append(validate_record(record))
    return records


def applicable(record: Mapping[str, Any], context: Mapping[str, Any], *, role: str = "draft") -> bool:
    if record.get("verification_status") in {"stale", "unsupported"}:
        return False
    if not set(record["topics"]) & ROLE_TOPICS.get(role, TOPICS):
        return False
    aliases = {
        "hardware_keys": "hardware_key", "hardware_id": "hardware_id", "accelerator_names": "accelerator_names",
        "gpu_architectures": "gpu_architecture", "compute_capabilities": "compute_capability",
        "frameworks": "framework", "framework_versions": "framework_major_minor",
        "toolkit_versions": "cuda_major_minor", "driver_versions": "driver_major_minor",
        "backend_modes": "backend_mode", "runner_contracts": "runner_contract",
        "model_families": "model_family", "workload_types": "workload_type",
        "agent_audiences": "role", "precision_modes": "precision_mode", "optimization_modes": "precision_mode",
    }
    for key, expected in record.get("applicability", {}).items():
        if key == "min_compute_capability":
            try:
                if float(context.get("compute_capability", 0)) < float(expected):
                    return False
            except (ValueError, TypeError):
                return False
            continue
        actual_key = aliases.get(key, key)
        actual = role if actual_key == "role" else context.get(actual_key)
        values = {item.lower() for item in strings(expected)}
        if key == "precision_modes" and not values.issubset({"normal", "conservative", "aggressive"}):
            policies = strings(context.get("allowed_precision_policies"))
            actual = policies + [value.removesuffix("_amp").removesuffix("_te") for value in policies]
        if values & {"*", "any", "all", "backend_neutral"}:
            continue
        # Before architecture selection, family facts remain explicitly conditional.
        if actual_key == "model_family" and not actual and role == "draft":
            continue
        if not actual or not values.intersection(item.lower() for item in strings(actual)):
            return False
    return True


def select_records(records: list[dict[str, Any]], context: Mapping[str, Any], *, role: str = "draft", already_filtered: bool = False) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in records:
        try:
            record = validate_record(item)
        except (TypeError, ValueError):
            logging.getLogger("MLEvolve").warning("Ignoring an invalid design knowledge record during retrieval")
            continue
        if not already_filtered and not applicable(record, context, role=role):
            continue
        key = digest({k: record[k] for k in ("summary", "applicability", "applies_when", "restrictions", "fallbacks", "verification_status", "strength")})
        if key in merged:
            prior = merged[key]
            prior["evidence_refs"] = strings(prior["evidence_refs"] + record["evidence_refs"])
            prior["topics"] = sorted(set(prior["topics"] + record["topics"]))
        else:
            merged[key] = record
    return sorted(merged.values(), key=lambda r: (r["strength"] != "hard", not bool(r["restrictions"]), r["verification_status"] != "verified", -r["confidence"], r["record_id"]))


def render_record_line(record: Mapping[str, Any]) -> str:
    scope = dict(record.get("applicability") or {})
    bits = [str(record["summary"])]
    if scope:
        bits.append("Applies to " + "; ".join(f"{k}={','.join(strings(v))}" for k, v in sorted(scope.items())) + ".")
    for key, prefix in (("applies_when", "Only when: "), ("restrictions", "Restrictions: "), ("fallbacks", "Fallback: ")):
        values = record.get(key) or []
        if values:
            bits.append(prefix + " ".join(values))
    bits.append(f"[{record['record_id']}; {record['verification_status']}]")
    return "- " + " ".join(bits)


def render_records(records: list[Mapping[str, Any]], *, max_chars: int | None = None,
                   dropped: list[str] | None = None) -> str:
    """Render the design knowledge prompt section.

    Claims stay complete. When ``max_chars`` is given, whole records are omitted
    instead of cutting a claim mid-sentence, and budget is reserved up front for
    mandatory (``hard``) records so a long optional tail can never evict a
    precision or scheduler constraint. Omitted record ids are appended to
    ``dropped`` when a list is supplied.
    """
    if not records:
        return ""
    header = ["# Design knowledge", "Source-backed reference. Current task constraints and fresh measurements take precedence."]
    rendered = [(record, render_record_line(record)) for record in records]
    if max_chars is None:
        kept = [line for _, line in rendered]
    else:
        mandatory = sum(1 + len(line) for record, line in rendered if str(record.get("strength")) == "hard")
        used = len("\n".join(header)) + mandatory
        kept = []
        for record, line in rendered:
            if str(record.get("strength")) == "hard":
                kept.append(line)
                continue
            cost = 1 + len(line)
            if used + cost > max_chars:
                if dropped is not None:
                    dropped.append(record["record_id"])
                continue
            used += cost
            kept.append(line)
    return "\n".join(header + kept)
