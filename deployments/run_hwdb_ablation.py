"""Matched HWDB-content ablation with current filters and fresh agent searches."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from deployments.run_hwdb_precision_matrix import make_config, save, stop_group, wait_bounded


BASELINE = os.environ.get("MLEVOLVE_ABLATION_BASELINE", "93371dd64b8e2b888c1bde7cb9d90d7c03ac4e5d")
DEFAULT_SEEDS = (42, 43)
MODES = ("conservative", "normal")
ARMS = ("original", "revised")
PRIMARY_ATTEMPTS = int(os.environ.get("MLEVOLVE_ABLATION_PRIMARY_ATTEMPTS") or 10)
TARGET_VALID_NODES = int(os.environ.get("MLEVOLVE_ABLATION_TARGET_VALID") or 0)
PUBLIC = Path("/datasets/nlp-getting-started/prepared/public")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory_public(path):
    return {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob("*")) if p.is_file()}


def selected_seeds() -> tuple[int, ...]:
    raw = os.environ.get("MLEVOLVE_ABLATION_SEEDS", "").strip()
    if not raw:
        return DEFAULT_SEEDS
    seeds = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not seeds:
        raise ValueError("MLEVOLVE_ABLATION_SEEDS must contain at least one integer seed")
    return seeds


def matrix_rows(seeds: tuple[int, ...] | None = None):
    modes = tuple(item.strip() for item in (os.environ.get("MLEVOLVE_ABLATION_MODES") or "").split(",") if item.strip()) or MODES
    arms = tuple(item.strip() for item in (os.environ.get("MLEVOLVE_ABLATION_ARMS") or "").split(",") if item.strip()) or ARMS
    unknown = set(modes) - set(MODES) | set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"Unsupported ablation cell filter: {sorted(unknown)}")
    rows = []
    for i, seed in enumerate(seeds or selected_seeds()):
        for j, mode in enumerate(modes if i == 0 else tuple(reversed(modes))):
            for arm in (arms if (i + j) % 2 == 0 else tuple(reversed(arms))):
                rows.append({"seed": seed, "mode": mode, "arm": arm, "status": "queued"})
    return rows


def config_for_cell(repo, folder, seed, mode):
    cfg = make_config(repo, folder, PUBLIC, "nlp-getting-started", mode, 3600, PRIMARY_ATTEMPTS,
                      "deepseek-flash", milestone=True)
    # The ablation loop owns the stop condition: a fixed attempt budget, an
    # optional verified-valid target, and a wall-clock cap in the engine.
    cfg["agent"].update(seed=seed, ablation_primary_attempts=PRIMARY_ATTEMPTS)
    if TARGET_VALID_NODES > 0:
        cfg["agent"]["stop_after_valid_nodes"] = TARGET_VALID_NODES
    cfg["cpu_number"] = 8
    return cfg


def prepare_graphs(repo, root):
    from localml_scheduler.hardware_knowledge.feature_filter import query_hardware_features, query_hardware_node

    folder = root / "graphs"
    folder.mkdir(parents=True, exist_ok=True)
    baseline = subprocess.check_output(["git", "show", f"{BASELINE}:schema/hardware_knowledge_graph.json"], cwd=repo)
    original = folder / "original.json"
    revised = folder / "revised.json"
    original.write_bytes(baseline)
    revised.write_bytes((repo / "schema/hardware_knowledge_graph.json").read_bytes())
    paths = {"original": original, "revised": revised}
    manifest = {arm: {"path": str(path), "sha256": digest(path)} for arm, path in paths.items()}
    if manifest["original"]["sha256"] == manifest["revised"]["sha256"]:
        raise RuntimeError("The ablation requires genuinely different HWDB content")
    contrasts = []
    for gpu in ("NVIDIA A10", "NVIDIA A100 PCIe 40GB", "NVIDIA A100 PCIe 80GB", "NVIDIA A100 SXM4 40GB", "NVIDIA A100 SXM4 80GB"):
        for mode in MODES:
            visible = {}
            for arm, path in paths.items():
                visible[arm] = [{"stage": stage,
                                 "node": query_hardware_node(gpu, stage, graph_path=path, precision_mode=mode),
                                 "features": query_hardware_features(gpu, stage, graph_path=path, precision_mode=mode)}
                                for stage in ("model_design", "datatype_precision", "training_evaluation")]
                if not all(v["node"].get("found") for v in visible[arm]):
                    raise RuntimeError("Missing hardware in an ablation arm: " + gpu)
            if visible["original"] == visible["revised"]:
                raise RuntimeError("No visible content contrast after shared filtering: " + gpu + " " + mode)
            contrasts.append({"gpu": gpu, "mode": mode, "contexts": visible})
    save(root / "graph_manifest.json", manifest)
    save(root / "guidance_contrast.json", contrasts)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.root.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        prepare_graphs(repo, args.root)
        return
    import torch
    queued = bool(os.environ.get("MLEVOLVE_EXECUTION_QUEUE"))
    if not queued and not torch.cuda.is_available():
        raise RuntimeError("A10/A100 CUDA required")
    if queued:
        from localml_scheduler.hardware import detect_hardware_profile
        gpu = detect_hardware_profile().gpu_name
    else:
        gpu = torch.cuda.get_device_name(0)
    if gpu != "NVIDIA A10" and not gpu.startswith("NVIDIA A100"):
        raise RuntimeError("Only A10/A100 are authorized")
    graphs = json.loads((args.root / "graph_manifest.json").read_text())
    for entry in graphs.values():
        if digest(Path(entry["path"])) != entry["sha256"]:
            raise RuntimeError("HWDB snapshot changed")
    results = args.root / "results"
    results.mkdir(exist_ok=False)
    public_hashes = inventory_public(PUBLIC)
    if not public_hashes or "train.csv" not in public_hashes:
        raise RuntimeError("Prepared Disaster Tweets public data missing")
    identity = {"scope": "HWDB content only; current filtering, strict precision rules, validators and agent prompts held fixed",
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                "baseline_hwdb_commit": BASELINE, "gpu": gpu, "agent": "deepseek-flash",
                "primary_attempts_per_cell": PRIMARY_ATTEMPTS, "search_seeds": list(selected_seeds()),
                "llm_seed_note": "Search seeds are paired; hosted model responses are not guaranteed deterministic.",
                "public_hashes": public_hashes, "graphs": graphs}
    save(results / "experiment.json", identity)
    rows = matrix_rows()
    save(results / "matrix.json", rows)
    for row in rows:
        if inventory_public(PUBLIC) != public_hashes:
            raise RuntimeError("Public inputs changed between cells")
        cell = results / f'{row["seed"]}-{row["mode"]}-{row["arm"]}'
        cell.mkdir()
        cfg = config_for_cell(repo, cell, row["seed"], row["mode"])
        config_path = cell / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        graph = graphs[row["arm"]]
        env = dict(os.environ, MLEVOLVE_CONFIG=str(config_path), MLEVOLVE_HWDB_GRAPH_PATH=graph["path"],
                   MLEVOLVE_HWDB_GRAPH_SHA256=graph["sha256"], MLEVOLVE_HWDB_ARM=row["arm"],
                   MLEVOLVE_VLLM_CACHE_SALT="hwdb-content-ablation-20260913-" + cell.name,
                   PYTHONUNBUFFERED="1")
        row.update(status="running", started_at=time.time(), path=str(cell))
        save(results / "matrix.json", rows)
        with (cell / "runtime.log").open("x") as log:
            proc = subprocess.Popen([sys.executable, "run.py"], cwd=repo, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            descendants = set()
            try:
                code = wait_bounded(proc, None, descendants)
            finally:
                stop_group(proc, descendants)
        summaries = list((cell / "runs").glob("*/logs/ablation/summary.json"))
        summary = json.loads(summaries[0].read_text()) if len(summaries) == 1 else None
        met = bool(summary and summary["primary_attempts_finished"] == PRIMARY_ATTEMPTS and
                   (os.environ.get("MLEVOLVE_ABLATION_EXACT_BUDGET") == "1" or summary["all_verified_valid_nodes"] >= 1))
        row.update(status="complete" if code == 0 and met else "failed", exit_code=code, ended_at=time.time(), summary=summary)
        save(results / "matrix.json", rows)
        print("MLEVOLVE_ABLATION_CELL " + json.dumps(row), flush=True)
        if code != 0 or not met:
            raise RuntimeError("Cell failed or evidence incomplete; inspect logs before proceeding")
    if inventory_public(PUBLIC) != public_hashes:
        raise RuntimeError("Public inputs changed during final cell")


if __name__ == "__main__":
    main()
