"""Offline comparison of actual v2 prompt projections on the target GPUs."""

import argparse
import json
from pathlib import Path
import subprocess

from agents.design_knowledge import hardware_records
from agents.hardware_context import HardwarePromptContext
from knowledge.records import render_records
from localml_scheduler.hardware_knowledge.feature_filter import query_hardware_features, query_hardware_node
from utils.precision_policy import resolve_precision_policy
from deployments.run_hwdb_precision_matrix import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    root = parser.parse_args().output
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    rows = []
    for arm, revision in (("previous", "b4352ea"), ("upstream", "1bf8d2f"), ("revised", None)):
        graph = root / (arm + ".json")
        graph.write_bytes(subprocess.check_output(["git", "show", revision + ":schema/hardware_knowledge_graph.json"], cwd=repo)
                          if revision else (repo / "schema/hardware_knowledge_graph.json").read_bytes())
        data = json.loads(graph.read_text())
        for gpu in ("NVIDIA A10", "NVIDIA A100 SXM4 80GB", "NVIDIA GeForce RTX 5090"):
            for mode in ("conservative", "normal"):
                stages = [{"stage": stage, "node": query_hardware_node(gpu, stage, graph_path=graph, precision_mode=mode),
                           "features": query_hardware_features(gpu, stage, graph_path=graph, precision_mode=mode).get("features", [])}
                          for stage in ("model_design", "datatype_precision", "training_evaluation")]
                architecture = "blackwell" if "5090" in gpu else "ampere"
                policy = resolve_precision_policy({"architecture": architecture}, mode=mode).to_dict()
                context = HardwarePromptContext(candidate={"workload_type": "nlp"}, raw_context={"stage_hardware_features": {"stages": stages}},
                                                compact_context={"precision_policy": policy, "hardware_context": {"hardware": {"gpu_name": gpu, "architecture": architecture}}})
                records = hardware_records(context)
                prompt = render_records(records)
                filename = f"{arm}-{gpu.replace(' ', '_')}-{mode}"
                (root / (filename + ".txt")).write_text(prompt)
                save(root / (filename + ".records.json"), records)
                rows.append({"arm": arm, "gpu": gpu, "mode": mode, "nodes": len(data["nodes"]), "edges": len(data["edges"]),
                             "prompt_records": len(records), "prompt_chars": len(prompt), "hardware_found": all(s["node"].get("found") for s in stages)})
    save(root / "audit.json", rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
