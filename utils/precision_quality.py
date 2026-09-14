"""Fail-closed selection of a hardware optimization from measured comparisons."""

from __future__ import annotations

import math
from typing import Any, Mapping

from utils.precision_policy import normalize_precision_policy_name


COMPARISON_FIELDS = (
    "model", "data", "split", "initialization", "preprocessing", "training_budget",
    "effective_batch", "metric", "hardware", "software", "backend",
)


def select_validated_precision(
    requested: str, *, protocol: Mapping[str, Any] | None = None,
    comparison: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Use FP32 unless supplied measurements prove a faster, quality-safe choice.

    The caller supplies a protocol fingerprint for each comparison field and a
    measured FP32 reference/candidate, each with metric, epoch_seconds, precision,
    protocol and evidence_refs. Tolerance and metric direction must be explicit.
    This validates the comparison contract; it does not manufacture measurements
    or authenticate the contents of external evidence documents.
    """
    selected = normalize_precision_policy_name(requested)
    if selected is None:
        raise ValueError(f"Unknown precision policy: {requested}")
    result = {"requested_precision": selected, "precision": "fp32", "status": "unverified",
              "reason": "Matching measured quality/speed comparison unavailable; use FP32.", "evidence_refs": []}
    if selected in {"fp32", "disabled"}:
        return {**result, "status": "reference", "reason": "FP32 reference execution."}
    if not isinstance(comparison, Mapping) or not isinstance(protocol, Mapping):
        return result
    reference, candidate = comparison.get("reference"), comparison.get("candidate")
    if not isinstance(reference, Mapping) or not isinstance(candidate, Mapping):
        return result
    if any(protocol.get(key) in (None, "", [], {}) for key in COMPARISON_FIELDS):
        return {**result, "reason": "Incomplete comparison protocol; use FP32."}
    expected = {key: protocol[key] for key in COMPARISON_FIELDS}
    for measurement in (reference, candidate):
        measured_protocol = measurement.get("protocol")
        if not isinstance(measured_protocol, Mapping) or any(measured_protocol.get(key) != value for key, value in expected.items()):
            return {**result, "reason": "Quality comparison protocol does not match this execution; use FP32."}
        refs = measurement.get("evidence_refs")
        if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or not ref.strip() for ref in refs):
            return {**result, "reason": "Measurement evidence references unavailable; use FP32."}
    if normalize_precision_policy_name(reference.get("precision")) != "fp32" or normalize_precision_policy_name(candidate.get("precision")) != selected:
        return {**result, "reason": "Comparison precision does not match the FP32 reference and requested candidate; use FP32."}
    direction = comparison.get("direction")
    if not isinstance(direction, str) or direction not in {"minimize", "maximize"}:
        return {**result, "reason": "Metric direction unavailable; use FP32."}
    values = [comparison.get("tolerance"), reference.get("metric"), candidate.get("metric"),
              reference.get("epoch_seconds"), candidate.get("epoch_seconds")]
    try:
        if any(isinstance(value, bool) for value in values):
            raise ValueError("Boolean measurement")
        tolerance, before, after, baseline_time, candidate_time = map(float, values)
        if not all(math.isfinite(value) for value in (tolerance, before, after, baseline_time, candidate_time)) or tolerance < 0 or min(baseline_time, candidate_time) <= 0:
            raise ValueError("Invalid measurement")
    except (TypeError, ValueError, OverflowError):
        return {**result, "reason": "Finite measurements and an explicit nonnegative quality tolerance are required; use FP32."}
    result["evidence_refs"] = list(dict.fromkeys(reference["evidence_refs"] + candidate["evidence_refs"]))
    degradation = after - before if direction == "minimize" else before - after
    if degradation > tolerance:
        return {**result, "status": "rejected", "reason": "Task quality exceeds the supplied degradation tolerance; use FP32."}
    if candidate_time >= baseline_time:
        return {**result, "status": "rejected", "reason": "No measured epoch-time improvement; use FP32."}
    return {**result, "precision": selected, "status": "accepted",
            "reason": "Matching measured comparison satisfies the supplied quality tolerance and improves epoch time."}
