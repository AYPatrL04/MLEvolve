"""Offline demo: what the merged v2 design knowledge costs a prompt, before and after the budget fix.

This renders the real hardware knowledge graph for the target GPUs. It never
executes generated candidate code and never contacts a model provider: every
number below is a static property of the prompt text that would be sent.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from agents.design_knowledge import hardware_records
from agents.hardware_context import HardwarePromptContext
from knowledge.records import render_records
from localml_scheduler.hardware_knowledge.feature_filter import query_hardware_features, query_hardware_node
from utils.node_diagnostics import build_node_diagnostics
from utils.precision_policy import resolve_precision_policy

STAGES = ("model_design", "datatype_precision", "training_evaluation")


def build_records(repo: Path, gpu: str, mode: str):
    graph = repo / "schema" / "hardware_knowledge_graph.json"
    stages = [
        {
            "stage": stage,
            "node": query_hardware_node(gpu, stage, graph_path=graph, precision_mode=mode),
            "features": query_hardware_features(gpu, stage, graph_path=graph, precision_mode=mode).get("features", []),
        }
        for stage in STAGES
    ]
    architecture = "blackwell" if "5090" in gpu else "ampere"
    policy = resolve_precision_policy({"architecture": architecture}, mode=mode).to_dict()
    context = HardwarePromptContext(
        candidate={"workload_type": "nlp"},
        raw_context={"stage_hardware_features": {"stages": stages}},
        compact_context={"precision_policy": policy, "hardware_context": {"hardware": {"gpu_name": gpu, "architecture": architecture}}},
    )
    return policy, hardware_records(context)


def topic_cost(records, text):
    """Attribute rendered characters back to the record that produced them."""
    cost = {}
    for record in records:
        line = "- " + str(record["summary"]).split(" [")[0]
        if line.split(" [")[0] in text:
            cost[record["record_id"]] = len(record["summary"])
    return cost


def check(heading_prompt, expect_present):
    from engine.search_node import SearchNode

    cfg = SimpleNamespace(experiment=SimpleNamespace(mode="hardware_aware"), log_dir=Path("."))
    node = SearchNode(code="", stage="draft")
    node.prompt_input = heading_prompt
    report = build_node_diagnostics(cfg, node)["hardware_knowledge"]
    ok = (report["injection_status"] == "present") is expect_present
    return ok, report["injection_status"]


def sweep(repo: Path, budget: int):
    """Show what each budget keeps, including a deliberately impossible budget."""
    table = []
    for gpu, mode in (("NVIDIA A10", "conservative"), ("NVIDIA A10", "normal")):
        _, records = build_records(repo, gpu, mode)
        hard = [record for record in records if str(record.get("strength")) == "hard"]
        for candidate in (budget, max(budget, 6000), max(budget, 9000), 250):
            dropped: list[str] = []
            text = render_records(records, max_chars=candidate, dropped=dropped)
            table.append({
                "gpu": gpu,
                "mode": mode,
                "budget": candidate,
                "chars": len(text),
                "records_kept": len(records) - len(dropped),
                "records_total": len(records),
                "hard_kept": sum(1 for record in hard if str(record["summary"]) in text),
                "hard_total": len(hard),
                "overflowed_budget": len(text) > candidate,
            })
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=3500)
    parser.add_argument("--no-sweep", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    verdicts = []
    for gpu in ("NVIDIA A10", "NVIDIA A100 SXM4 80GB", "NVIDIA GeForce RTX 5090"):
        for mode in ("conservative", "normal"):
            policy, records = build_records(repo, gpu, mode)
            unbounded = render_records(records)
            dropped: list[str] = []
            bounded = render_records(records, max_chars=args.budget, dropped=dropped)
            (args.output / f"{gpu.replace(' ', '_')}-{mode}-unbounded.txt").write_text(unbounded)
            (args.output / f"{gpu.replace(' ', '_')}-{mode}-bounded.txt").write_text(bounded)

            hard = [record for record in records if str(record.get("strength")) == "hard"]
            allowed = ", ".join(policy.get("allowed_policies") or ["disabled"])
            row = {
                "gpu": gpu,
                "mode": mode,
                "allowed_policies": allowed,
                "records_total": len(records),
                "records_rendered": len(records) - len(dropped),
                "records_dropped": len(dropped),
                "chars_unbounded": len(unbounded),
                "chars_bounded": len(bounded),
                "hard_records": len(hard),
                "hard_records_kept": sum(1 for record in hard if str(record["summary"]) in bounded),
                "precision_policy_kept": f"Allowed precision policies: {allowed}" in bounded,
                "over_budget": len(bounded) > args.budget,
                "half_written_claims": sum(1 for line in bounded.splitlines() if line.startswith("- ") and not line.endswith("]")),
            }
            rows.append(row)
            verdicts.append(
                row["hard_records_kept"] == row["hard_records"]
                and row["precision_policy_kept"]
                and not row["over_budget"]
                and row["half_written_claims"] == 0
            )

    heading_ok_v2, status_v2 = check(render_records(build_records(repo, "NVIDIA A10", "conservative")[1]), True)
    heading_ok_absent, status_absent = check("# Task description\nNothing hardware specific.", False)

    report = {
        "budget_chars": args.budget,
        "rows": rows,
        "totals": {
            "mean_chars_unbounded": round(sum(r["chars_unbounded"] for r in rows) / len(rows)),
            "mean_chars_bounded": round(sum(r["chars_bounded"] for r in rows) / len(rows)),
            "max_chars_unbounded": max(r["chars_unbounded"] for r in rows),
            "max_chars_bounded": max(r["chars_bounded"] for r in rows),
            "records_dropped": sum(r["records_dropped"] for r in rows),
        },
        "budget_verdict_usable": all(verdicts),
        "heading_detection": {
            "v2_section_detected": f"{heading_ok_v2} ({status_v2})",
            "empty_section_still_absent": f"{heading_ok_absent} ({status_absent})",
        },
    }
    if not args.no_sweep:
        report["budget_sweep"] = sweep(repo, args.budget)
    (args.output / "demo.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
