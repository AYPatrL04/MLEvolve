"""Evidence-backed success criteria for the first executable GPU candidate."""

import hashlib
import math
from pathlib import Path

import pandas as pd

from engine.preflight import candidate_code_hash
from utils.node_diagnostics import build_node_diagnostics


def verify_node(cfg, node, packet):
    reasons = []
    metric = getattr(getattr(node, "metric", None), "value", None)
    if node.is_buggy is not False or node.is_valid is not True:
        reasons.append("runtime result or submission is not valid")
    if not isinstance(metric, (int, float)) or isinstance(metric, bool) or not math.isfinite(metric):
        reasons.append("missing finite validation metric")
    if getattr(node, "review_status", None) not in {"approved", "repaired"}:
        reasons.append("stage review did not approve the candidate")
    if (getattr(node, "preflight_admitted", None) is not True
            or getattr(node, "preflight_mode", None) != "full_cpu"
            or getattr(node, "preflight_code_hash", None) != candidate_code_hash(node.code)):
        reasons.append("full CPU preflight evidence is missing or belongs to different code")
    if (not packet or packet.get("status") != "parsed_valid"
            or packet.get("node_id") != node.id
            or not packet.get("requires_gpu")
            or not packet.get("duration_seconds") or packet.get("duration_seconds", 0) <= 0
            or packet.get("metric") != metric):
        reasons.append("missing matching successful GPU scheduler job packet")
    diagnostics = build_node_diagnostics(cfg, node)
    runtime = diagnostics.get("runtime") or {}
    if (runtime.get("status") != "completed"
            or not str(runtime.get("device", "")).startswith("cuda")
            or not runtime.get("completed_updates", 0) > 0):
        reasons.append("runtime diagnostics must show completed CUDA optimizer updates")
    allowed = {"disabled"}
    if cfg.agent.precision_optimization_mode == "normal":
        allowed.add("torch.float16")
    observed = runtime.get("autocast_dtypes_observed")
    if (runtime.get("tf32_matmul_allowed") is not False
            or runtime.get("tf32_cudnn_allowed") is not False
            or runtime.get("parameter_dtypes") != ["torch.float32"]
            or not observed or not set(observed).issubset(allowed)):
        reasons.append("runtime precision evidence does not confirm the configured policy")
    if "torch.float16" in (observed or []) and runtime.get("grad_scaler_enabled") is not True:
        reasons.append("FP16 training requires enabled gradient scaling")
    submission = Path(cfg.workspace_dir) / "submission" / f"submission_{node.id}.csv"
    sample = Path(cfg.data_dir) / "sample_submission.csv"
    submission_hash = None
    try:
        expected, actual = pd.read_csv(sample), pd.read_csv(submission)
        if list(actual.columns) != list(expected.columns) or len(actual) != len(expected) or not len(actual):
            reasons.append("submission columns or row count do not match the public sample")
        elif actual.isna().any().any():
            reasons.append("submission contains missing predictions")
        else:
            # This first milestone uses Disaster Tweets; no held-out labels are read.
            if "id" in expected and not actual["id"].equals(expected["id"]):
                reasons.append("submission IDs or row order differ from the public sample")
            if cfg.exp_id == "nlp-getting-started" and (
                "target" not in actual or not actual["target"].isin([0, 1]).all()
            ):
                reasons.append("Disaster Tweets predictions must be binary class labels")
            submission_hash = hashlib.sha256(submission.read_bytes()).hexdigest()
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        reasons.append(f"cannot independently validate submission: {type(exc).__name__}")
    return {
        "met": not reasons, "node_id": node.id, "reasons": reasons, "metric": metric,
        "job_id": (packet or {}).get("job_id"), "code_sha256": candidate_code_hash(node.code),
        "submission": str(submission), "submission_sha256": submission_hash,
        "runtime": runtime, "preflight_report": getattr(node, "preflight_report_path", None),
    }
