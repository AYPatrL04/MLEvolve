"""One complete seed-42 search using the local HWDB and real A10 predictor."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import time

import yaml

from deployments.run_hwdb_precision_matrix import make_config, save
from localml_scheduler.config import PredictionSettings
from localml_scheduler.hardware import detect_hardware_profile
from localml_scheduler.prediction import MLVramPredictor


ROOT = Path("/experiment/mlevolve-full-a10-s42-20260914")


def configuration(repo, root):
    cfg = make_config(repo, root, Path("/datasets/nlp-getting-started/prepared/public"),
                      "nlp-getting-started", "normal", 172800, 10, "deepseek-flash")
    cfg["exp_name"] = "full_a10_seed42"
    cfg["agent"].update(seed=42, time_limit=None, initial_drafts=3)
    cfg["hardware_knowledge"]["enabled"] = True
    cfg["preflight"]["enabled"] = True
    cfg["scheduler"]["enabled"] = True
    cfg["scheduler"]["settings"]["prediction"]["mode"] = "ml_predictor"
    cfg["scheduler"]["settings"]["prediction"]["ml"]["registry_path"] = str(repo / "PerfSeer-predictor/models/registry.json")
    cfg["scheduler"]["settings"]["gpu_scheduler"]["packing_backend"] = "cuda_process"
    cfg["scheduler"]["settings"]["gpu_scheduler"]["parallel_job_cap"] = None
    return cfg


def plot(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    journals = sorted((root / "runs").glob("*/logs/journal.json"))
    if not journals:
        return
    journal = json.loads(journals[-1].read_text())
    nodes = journal.get("nodes", []) if isinstance(journal, dict) else journal
    fig, (gantt, metric) = plt.subplots(2, 1, figsize=(14, 9))
    jobs = []
    for database in (root / "runs").rglob("pipeline.sqlite3"):
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as conn:
            for name, start, end in conn.execute("SELECT job_id, started_at, finished_at FROM job_packets"):
                if start and end:
                    jobs.append((name, datetime.fromisoformat(start).timestamp(), datetime.fromisoformat(end).timestamp()))
    origin = min((job[1] for job in jobs), default=None)
    for i, (name, start, end) in enumerate(jobs):
        gantt.barh(i + 1, max(0, end - start), left=start - origin)
    for i, node in enumerate(nodes):
        value = node.get("metric")
        value = value.get("value") if isinstance(value, dict) else value
        if isinstance(value, (int, float)) and not node.get("is_buggy"):
            metric.plot(i + 1, value, "o", color="#16856b")
    if origin is None:
        gantt.text(0.5, 0.5, "No completed scheduler executions recorded", ha="center", transform=gantt.transAxes)
    gantt.set(xlabel="Execution time (seconds)", ylabel="Scheduler job", title="A10 seed 42 execution Gantt")
    metric.set(xlabel="Node", ylabel="Validation metric")
    fig.tight_layout()
    fig.savefig(root / "gantt_metrics.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    cfg = configuration(repo, ROOT)
    config_path = ROOT / ("config.cpu.yaml" if args.prepare_only else "config.yaml")
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    if args.prepare_only:
        from config import _load_cfg, prep_cfg
        prep_cfg(_load_cfg(config_path, use_cli_args=False))
        return
    hardware = detect_hardware_profile()
    if hardware.gpu_name != "NVIDIA A10":
        raise RuntimeError(f"Expected NVIDIA A10, got {hardware.gpu_name}")
    predictor = MLVramPredictor(PredictionSettings(mode="ml_predictor"), hardware)
    save(ROOT / "predictor-health.json", {"hardware": hardware.to_dict(), "available": predictor.available,
         "model_id": predictor.model_id, "artifact_sha256": predictor.artifact_hash,
         "error": predictor.unavailable_reason})
    if not predictor.available:
        raise RuntimeError("A10 ML Predictor must load successfully before the full task starts")
    state = {"status": "running", "seed": 42, "node_budget": 10, "started_at": time.time(),
             "source_sha256": os.environ["SOURCE_SHA256"], "precision_mode": "normal",
             "hwdb_backend": "local_schema", "predictor": predictor.model_id}
    save(ROOT / "status.json", state)
    env = dict(os.environ, MLEVOLVE_CONFIG=str(config_path), MLEVOLVE_VLLM_CACHE_SALT="full-a10-s42-20260914")
    code = 1
    try:
        with (ROOT / "runtime.log").open("a") as log:
            code = subprocess.call([sys.executable, "run.py"], cwd=repo, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
    finally:
        state.update(status="finished" if code == 0 else "failed", exit_code=code, ended_at=time.time())
        save(ROOT / "status.json", state)
        plot(ROOT)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
