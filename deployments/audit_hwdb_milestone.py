"""Reconcile historical milestone evidence without modifying its journal or DB."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

sys.path.insert(0, "/runtime/repo")
from engine.milestone import verify_node
from engine.preflight import candidate_code_hash


def audit(logs, data_dir):
    journal = json.loads((logs / "journal.json").read_text())
    cfg = SimpleNamespace(
        log_dir=logs, workspace_dir=logs.parent / "workspace", data_dir=data_dir,
        exp_id="nlp-getting-started", exp_name=logs.parent.name,
        experiment=SimpleNamespace(mode="hardware_aware"),
        agent=SimpleNamespace(precision_optimization_mode="conservative"),
    )
    results = []
    connection = sqlite3.connect("file:" + str(logs / "pipeline.sqlite3") + "?mode=ro", uri=True, timeout=3)
    connection.row_factory = sqlite3.Row
    try:
        for original in journal.get("nodes", []):
            node_id = original.get("id")
            packets = connection.execute("SELECT * FROM job_packets WHERE node_id=?", (node_id,)).fetchall()
            if len(packets) != 1:
                continue
            packet = dict(packets[0])
            events = connection.execute(
                "SELECT event_id,event_type,job_id,payload_json FROM pipeline_events WHERE node_id=? ORDER BY event_id", (node_id,),
            ).fetchall()
            submissions = [e for e in events if e["event_type"] == "scheduler_submission_created" and e["job_id"] == packet["job_id"]]
            finishes = [e for e in events if e["event_type"] == "job_finished" and e["job_id"] == packet["job_id"]]
            parsed = [e for e in events if e["event_type"] == "execution_result_parsed"]
            if len(submissions) != 1 or len(finishes) != 1 or len(parsed) != 1:
                continue
            submission = json.loads(submissions[0]["payload_json"])
            finish = json.loads(finishes[0]["payload_json"])
            result = json.loads(parsed[0]["payload_json"])
            if (submission.get("requires_gpu") is not True or finish.get("status") != "COMPLETED"
                    or submission.get("preflight_code_hash") != candidate_code_hash(original["code"])
                    or packet["status"] != "parsed_valid" or result.get("metric") != packet["metric"]
                    or result.get("is_buggy") is not False or result.get("is_valid") is not True):
                continue
            # Restore only facts explicitly recorded before the bookkeeping failure.
            # Neither the original journal nor the materialized job packet is changed.
            node = SimpleNamespace(**original)
            node.metric = SimpleNamespace(**original["metric"])
            node.parent = None
            node.is_buggy, node.is_valid = result["is_buggy"], result["is_valid"]
            original_gpu_flag = packet.get("requires_gpu")
            packet["requires_gpu"] = submission["requires_gpu"]
            evidence = verify_node(cfg, node, packet)
            report = json.loads(Path(node.preflight_report_path).read_text())
            statuses = {s["name"]: s["status"] for s in report["stages"]}
            required = ("hardware", "construction", "data_contract", "cpu_training", "validation", "memory")
            if not all(statuses.get(stage) == "PASS" for stage in required):
                evidence["met"] = False
                evidence["reasons"].append("required real CPU stages did not all pass")
            evidence.update(
                verification="event_reconciled_read_only",
                original_materialized_gpu_flag=original_gpu_flag,
                source_event_ids=[submissions[0]["event_id"], finishes[0]["event_id"], parsed[0]["event_id"]],
                preflight_stages=statuses,
            )
            results.append(evidence)
    finally:
        connection.close()
    return {"verification": "event_reconciled_read_only", "milestone_met": any(r["met"] for r in results), "nodes": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.logs, args.data_dir)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text)
    print(text)
    return 0 if result["milestone_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
