"""Build concise records from frozen, validated lesson publications."""

from __future__ import annotations

from typing import Any, Mapping

from knowledge.records import from_source


def publication_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    identity = dict(payload["identity"])
    revision = int(payload["revision_number"])
    trust = dict(payload.get("trust") or {})
    scope = {k: identity[k] for k in ("hardware_key", "accelerator_key", "resource_slice_key", "runtime_class", "framework_major", "cuda_major", "backend_class", "workload_bucket") if identity.get(k)}
    scope["model_families"] = [identity["model_family"]]
    status = "conflicted" if payload.get("maturity") == "conflicted" else "verified" if payload.get("maturity") == "stable" else "advisory"
    baseline = dict(payload.get("baseline") or {})
    common = {"applicability": scope, "verification_status": status}
    records = from_source({**common, "baseline": baseline, "confidence": trust.get("confidence", 0.0),
                           "warnings": baseline.get("warnings", []), "evidence_refs": trust.get("evidence_refs", []),
                           "agent_audiences": ["draft", "improve", "evolution", "fusion", "aggregation", "review"]},
                          domain="lesson", source_id=f"baseline:{payload['profile_key']}", revision=revision, topics=["architecture", "training"])
    for lesson in payload.get("lessons") or []:
        records += from_source({**lesson, **common}, domain="lesson", source_id=lesson["lesson_id"], revision=revision)
    return records
