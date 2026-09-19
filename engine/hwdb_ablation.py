"""Serial agent-search accounting with rejected candidates retained in the denominator."""

import hashlib
import json
import os
import time
from pathlib import Path

from deployments.run_hwdb_precision_matrix import save
from engine.milestone import verify_node
from utils.node_diagnostics import build_node_diagnostics


def summarize_attempts(rows, primary_attempts):
    primary = rows[:primary_attempts]
    terminal = [r for r in primary if r["status"] != "running"]
    submitted = [r for r in terminal if r.get("gpu_job_submitted")]
    return {
        "primary_attempt_budget": primary_attempts,
        "primary_attempts_finished": len(terminal),
        "primary_candidates_returned": sum(bool(r.get("node_id")) for r in terminal),
        "primary_generation_errors": sum(r["status"] == "generation_error" for r in terminal),
        "primary_no_candidate": sum(r["status"] == "no_candidate" for r in terminal),
        "primary_review_rejections": sum(r["status"] == "review_rejected" for r in terminal),
        "primary_preflight_rejections": sum(r["status"] == "preflight_rejected" for r in terminal),
        "primary_gpu_jobs_submitted": len(submitted),
        "primary_verified_valid_nodes": sum(r.get("verified_valid") is True for r in terminal),
        "primary_unverified_submitted_results": sum(r.get("verified_valid") is not True for r in submitted),
        "primary_valid_yield_per_attempt": sum(r.get("verified_valid") is True for r in terminal) / len(terminal) if terminal else None,
        "extension_attempts": max(0, len(rows) - primary_attempts),
        "all_verified_valid_nodes": sum(r.get("verified_valid") is True for r in rows),
        "first_valid_seconds": next((r["elapsed_seconds"] for r in rows if r.get("verified_valid")), None),
        "first_valid_finished_at": next((r.get("ended_at") for r in rows if r.get("verified_valid")), None),
        "time_origin": "serial generation loop; cell process startup is recorded separately in the matrix",
        "confirmed_buggy_execution_rate": None,
        "validator_false_positive_rate": None,
        "adjudication_note": "Rejections and failed results are not automatically genuine code defects. Independent review of retained evidence is required.",
    }


def guidance_provenance(candidate):
    expected = os.environ.get("MLEVOLVE_HWDB_GRAPH_SHA256")
    audits = getattr(candidate, "hardware_prompt_audit", None) or []
    verified = bool(expected and audits and all(
        a.get("experiment_graph", {}).get("sha256") == expected for a in audits))
    return {"verified": verified, "expected_graph_sha256": expected, "audit_entries": len(audits)}


def snapshot_candidate(folder, cfg, candidate, phase):
    path = folder / "candidates" / str(candidate.id) / phase
    path.mkdir(parents=True, exist_ok=True)
    (path / "candidate.py").write_text(candidate.code or "")
    record = build_node_diagnostics(cfg, candidate)
    record.update(
        node_id=candidate.id,
        candidate_sha256=hashlib.sha256((candidate.code or "").encode()).hexdigest(),
        pipeline_decision=getattr(candidate, "pipeline_decision", None),
        hardware_decision=getattr(candidate, "hardware_decision", None),
        stage_note_board=getattr(candidate, "stage_note_board", []),
        hardware_prompt_audit=getattr(candidate, "hardware_prompt_audit", []),
        review_history=getattr(candidate, "review_history", []),
        review_issues=getattr(candidate, "review_issues", []),
        independent_adjudication=None,
    )
    save(path / "evidence.json", json.loads(json.dumps(record, default=str)))
    return str(path)


def run_ablation_rounds(*, agent, interpreter, cfg, journal, logger, save_callback, ensure_capacity):
    primary = int(cfg.agent.ablation_primary_attempts)
    if primary <= 0:
        raise ValueError("A positive primary generation-attempt budget is required")
    folder = Path(cfg.log_dir) / "ablation"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "attempts.json").exists():
        raise ValueError("Ablation results already exist; refusing to overwrite or silently restart")
    rows = []
    started = time.time()
    operational_errors = 0
    exact_budget = os.environ.get("MLEVOLVE_ABLATION_EXACT_BUDGET") == "1"
    while len(rows) < primary or (not exact_budget and not any(r.get("verified_valid") for r in rows)):
        if not ensure_capacity(agent=agent, cfg=cfg, total_steps=int(cfg.agent.search.num_drafts) + 1, logger=logger):
            raise RuntimeError("Ablation search has no selectable work")
        row = {"attempt": len(rows) + 1, "phase": "primary" if len(rows) < primary else "extension",
               "status": "running", "started_at": time.time(), "verified_valid": False}
        rows.append(row)
        save(folder / "attempts.json", rows)
        candidate = None
        try:
            candidate = agent.step(exec_callback=interpreter.run, node=None, execute_immediately=False)
            if candidate is None or getattr(candidate, "stage", None) == "root":
                row["status"] = "no_candidate"
                continue
            row["node_id"] = candidate.id
            row["before_dispatch_evidence"] = snapshot_candidate(folder, cfg, candidate, "before_dispatch")
            rejected = getattr(candidate, "review_status", None) == "rejected"
            preflight_rejected = getattr(candidate, "preflight_admitted", None) is False
            agent.execute_deferred_nodes([candidate], interpreter.run_many)
            packet = agent.pipeline_logger.latest_job_packet(candidate.id)
            row["gpu_job_submitted"] = bool(packet and packet.get("requires_gpu"))
            row["job_packet"] = packet
            if rejected or getattr(candidate, "review_status", None) == "rejected":
                row["status"] = "preflight_rejected" if preflight_rejected or getattr(candidate, "preflight_admitted", None) is False else "review_rejected"
            else:
                evidence = verify_node(cfg, candidate, packet)
                guidance = guidance_provenance(candidate)
                met = evidence["met"] and guidance["verified"]
                row.update(status="verified_valid" if met else "unverified_execution",
                           verified_valid=met, runtime_verified_valid=evidence["met"], verification=evidence,
                           guidance_provenance=guidance)
            row["evidence_path"] = snapshot_candidate(folder, cfg, candidate, "after_dispatch")
        except Exception as exc:
            row.update(status="generation_error" if candidate is None else "execution_or_integration_error",
                       exception_type=type(exc).__name__)
            # The existing runtime log holds the traceback; do not duplicate potential API credentials.
            logger.exception("Ablation attempt %s failed; preserving its slot", row["attempt"])
            if candidate is not None and getattr(candidate, "stage", None) != "root":
                row["evidence_path"] = snapshot_candidate(folder, cfg, candidate, "exception")
                packet = agent.pipeline_logger.latest_job_packet(candidate.id)
                row["job_packet"] = packet
                row["gpu_job_submitted"] = bool(packet and packet.get("requires_gpu"))
        finally:
            row.update(ended_at=time.time(), elapsed_seconds=time.time() - started)
            save(folder / "attempts.json", rows)
            save(folder / "summary.json", summarize_attempts(rows, primary))
            save_callback(cfg, journal)
            print("MLEVOLVE_ABLATION_ATTEMPT " + json.dumps({k: v for k, v in row.items() if k not in {"verification", "job_packet"}}), flush=True)
            operational_errors = operational_errors + 1 if row["status"] in {"generation_error", "execution_or_integration_error", "no_candidate"} else 0
            if operational_errors >= 3:
                raise RuntimeError("Three consecutive operational failures; inspect evidence before spending more resources")
    return len(journal.nodes)
