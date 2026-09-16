"""Run one merged MLEvolve task with a colocated Qwen3.8 INT8 agent on A100."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import yaml


ROOT = Path("/experiment/mlevolve-a100-qwen-s42-20260914")
MODEL_DIR = Path(
    os.environ.get(
        "QWEN_MODEL_DIR",
        "/root/downeyflyfan/qwen38-v100-int8/models/Qwen3.8-27B-INT8-W8A16-MTP",
    )
)
MODEL_NAME = os.environ.get("QWEN_SERVED_MODEL_NAME", "qwen3.8-27b-int8-a100")
CONTEXT_WINDOW_TOKENS = int(os.environ.get("QWEN_MAX_MODEL_LEN", "65536"))
TASKS = {
    "petfinder": {
        "exp_id": "petfinder-pawpularity-score",
        "exp_name": "petfinder_pawpularity_score_a100_qwen_s42",
        "public": Path("/datasets/petfinder-pawpularity-score/prepared/public"),
    },
    "full": {
        "exp_id": "nlp-getting-started",
        "exp_name": "full_a100_qwen_s42",
        "public": Path("/datasets/nlp-getting-started/prepared/public"),
    },
}


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def configuration(repo: Path, root: Path, task: str) -> dict:
    task_config = TASKS[task]
    public = task_config["public"]
    config = yaml.safe_load((repo / "config.example.yaml").read_text())
    config.update(
        data_dir=str(public),
        dataset_dir="/datasets",
        desc_file=str(public / "description.md"),
        exp_name=task_config["exp_name"],
        exp_id=task_config["exp_id"],
        log_dir=str(root / "runs"),
        workspace_dir=str(root / "runs"),
        cpu_number=8,
        copy_data=False,
    )
    config["agent"].update(
        steps=10,
        time_limit=None,
        initial_drafts=3,
        seed=42,
        precision_optimization_mode="normal",
        design_knowledge_version="v2",
        use_global_memory=False,
    )
    endpoint = "http://127.0.0.1:8000/v1"
    for role in ("code", "feedback"):
        config["agent"][role].update(
            model=MODEL_NAME,
            provider="vllm",
            base_url=endpoint,
            api_key="EMPTY",
            context_window_tokens=CONTEXT_WINDOW_TOKENS,
            completion_tokens=8192,
            tokenizer_path=str(MODEL_DIR),
        )
    config["agent"]["review"]["fail_open_on_unavailable"] = False
    config["agent"]["search"]["parallel_search_num"] = None
    config["agent"]["search"]["num_gpus"] = 1
    config["vllm_client"].update(
        structured_output_mode="json_schema",
        default_completion_tokens=8192,
    )
    config["hardware_knowledge"].update(enabled=True, include_profile_evidence=True)
    config["hardware_knowledge"]["settings"]["runtime_root"] = str(root / "hwdb")
    config["hardware_knowledge"]["settings"]["graph"]["enabled"] = False
    config["preflight"].update(
        enabled=True,
        target_profile="auto",
        fail_open_on_internal_error=False,
    )
    config["scheduler"].update(
        enabled=True,
        runtime_root=str(root / "scheduler"),
        start_service=True,
    )
    scheduler = config["scheduler"]["settings"]
    scheduler["prediction"]["mode"] = "branch_profile"
    scheduler["gpu_scheduler"]["packing_backend"] = "cuda_process"
    scheduler["gpu_scheduler"]["parallel_job_cap"] = None
    scheduler["gpu_scheduler"]["memory"]["gpu_vram_gib"] = 32
    scheduler["gpu_scheduler"]["cuda_process"]["enabled"] = True
    scheduler["gpu_scheduler"]["mps"]["enabled"] = False
    config["context_cache"]["cache_dir"] = str(root / "context_cache")
    config["context_cache"]["knowledge_version"] = "k1"
    config["exec"]["timeout"] = 3600
    return config


def validate_colocated_memory(config: dict, total_vram_mb: int) -> dict:
    """Require the agent reservation and scheduler budget to leave VRAM headroom."""
    fraction = float(os.environ.get("QWEN_GPU_MEMORY_UTILIZATION", "0.52"))
    if not 0.0 < fraction < 1.0:
        raise RuntimeError("QWEN_GPU_MEMORY_UTILIZATION must be between 0 and 1")
    scheduler_gib = float(
        config["scheduler"]["settings"]["gpu_scheduler"]["memory"]["gpu_vram_gib"]
    )
    scheduler_mb = round(scheduler_gib * 1024)
    agent_reserved_mb = round(total_vram_mb * fraction)
    safety_mb = 2048
    headroom_mb = total_vram_mb - agent_reserved_mb - scheduler_mb
    if headroom_mb < safety_mb:
        raise RuntimeError(
            "Colocated A100 budget is too tight: "
            f"total={total_vram_mb}MiB agent={agent_reserved_mb}MiB "
            f"scheduler={scheduler_mb}MiB headroom={headroom_mb}MiB"
        )
    return {
        "agent_gpu_memory_fraction": fraction,
        "agent_reserved_vram_mb": agent_reserved_mb,
        "scheduler_vram_budget_mb": scheduler_mb,
        "unreserved_vram_headroom_mb": headroom_mb,
    }


def plot(root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    journals = sorted((root / "runs").glob("*/logs/journal.json"))
    if not journals:
        return
    journal = json.loads(journals[-1].read_text())
    nodes = journal.get("nodes", []) if isinstance(journal, dict) else journal
    figure, (gantt, metric) = plt.subplots(2, 1, figsize=(14, 9))
    jobs = []
    for database in (root / "runs").rglob("pipeline.sqlite3"):
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            for name, started, finished in connection.execute(
                "SELECT job_id, started_at, finished_at FROM job_packets"
            ):
                if started and finished:
                    jobs.append(
                        (
                            name,
                            datetime.fromisoformat(started).timestamp(),
                            datetime.fromisoformat(finished).timestamp(),
                        )
                    )
    origin = min((job[1] for job in jobs), default=None)
    for index, (_, started, finished) in enumerate(jobs):
        gantt.barh(index + 1, max(0, finished - started), left=started - origin)
    for index, node in enumerate(nodes):
        value = node.get("metric")
        value = value.get("value") if isinstance(value, dict) else value
        if isinstance(value, (int, float)) and not node.get("is_buggy"):
            metric.plot(index + 1, value, "o", color="#16856b")
    if origin is None:
        gantt.text(
            0.5,
            0.5,
            "No completed scheduler executions recorded",
            ha="center",
            transform=gantt.transAxes,
        )
    gantt.set(
        xlabel="Execution time (seconds)",
        ylabel="Scheduler job",
        title="A100 Qwen seed 42 execution Gantt",
    )
    metric.set(xlabel="Node", ylabel="Validation metric")
    figure.tight_layout()
    figure.savefig(root / "gantt_metrics.png", dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = ROOT / args.task
    config = configuration(repo, root, args.task)
    config_path = root / ("config.cpu.yaml" if args.prepare_only else "config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    public = TASKS[args.task]["public"]
    if not (public / "description.md").is_file() or not any(public.glob("*train*")):
        raise RuntimeError(f"Prepared public data missing for {args.task}: {public}")
    if not MODEL_DIR.is_dir() or not any(MODEL_DIR.iterdir()):
        raise RuntimeError(f"Qwen model directory is missing or empty: {MODEL_DIR}")
    if args.prepare_only:
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A100 CUDA is required")
    gpu = torch.cuda.get_device_name(0)
    if gpu not in {"NVIDIA A100-SXM4-80GB", "NVIDIA A100-80GB-PCIe"}:
        raise RuntimeError(f"Expected an 80GB NVIDIA A100, got {gpu}")
    hardware = {
        "gpu_name": gpu,
        "total_vram_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024**2)),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "agent_model": MODEL_NAME,
        "agent_mode": "colocated_local_vllm",
        "scheduler_prediction_mode": "branch_profile",
        **validate_colocated_memory(
            config,
            round(torch.cuda.get_device_properties(0).total_memory / (1024**2)),
        ),
    }
    save(root / "hardware.json", hardware)
    save(
        root / "status.json",
        {
            "status": "running",
            "task": args.task,
            "seed": 42,
            "node_budget": 10,
            "source_sha256": os.environ["SOURCE_SHA256"],
            "started_at": time.time(),
            **hardware,
        },
    )
    environment = dict(
        os.environ,
        MLEVOLVE_CONFIG=str(config_path),
        MLEVOLVE_VLLM_CACHE_SALT=f"mlevolve-a100-qwen-s42-{args.task}-20260914",
    )
    code = 1
    try:
        with (root / "runtime.log").open("a") as log:
            code = subprocess.call(
                [sys.executable, "run.py"],
                cwd=repo,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    finally:
        save(
            root / "status.json",
            {
                "status": "finished" if code == 0 else "failed",
                "task": args.task,
                "seed": 42,
                "node_budget": 10,
                "source_sha256": os.environ["SOURCE_SHA256"],
                "started_at": json.loads((root / "status.json").read_text())["started_at"],
                "ended_at": time.time(),
                "exit_code": code,
                **hardware,
            },
        )
        plot(root)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
