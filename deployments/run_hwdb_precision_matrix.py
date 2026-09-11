"""Bounded, resumable remote precision comparison; never downloads datasets."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml
import psutil


COMPETITIONS = [
    "nlp-getting-started",
    "tensorflow-speech-recognition-challenge",
    "jigsaw-toxic-comment-classification-challenge",
    "spooky-author-identification",
    "random-acts-of-pizza",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
]


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def make_config(repo, root, public, competition, mode, seconds, nodes):
    cfg = yaml.safe_load((repo / "config.example.yaml").read_text())
    cfg.update(data_dir=str(public), dataset_dir=str(public.parent.parent.parent),
               desc_file=str(public / "description.md"), exp_name=f"{competition}_{mode}",
               log_dir=str(root / "runs"), workspace_dir=str(root / "runs"),
               cpu_number=8, copy_data=False)
    cfg["agent"].update(steps=nodes, time_limit=seconds, seed=42,
                        precision_optimization_mode=mode, use_global_memory=False)
    for role in ("code", "feedback"):
        cfg["agent"][role].update(
            model="qwen3.8-27b-int8-a100", provider="vllm", api_key="EMPTY",
            base_url="http://mlevolve-qwen-a100.ecepxie.svc.cluster.local:8000/v1")
    cfg["hardware_knowledge"]["settings"]["graph"]["enabled"] = False
    cfg["hardware_knowledge"]["settings"]["runtime_root"] = str(root / "hwdb")
    cfg["scheduler"]["runtime_root"] = str(root / "scheduler")
    import torch
    actual_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if torch.cuda.is_available() else 24
    cfg["scheduler"]["settings"]["gpu_scheduler"]["memory"]["gpu_vram_gib"] = min(31, actual_gib)
    cfg["context_cache"]["cache_dir"] = str(root / "context_cache")
    cfg["preflight"].update(target_profile="auto", fail_open_on_internal_error=False)
    # No historical lessons or cross-mode memory may influence this comparison.
    cfg["lesson_profiles"] = {"enabled": False, "read_enabled": False, "write_enabled": False}
    cfg["exec"]["timeout"] = min(seconds, 3600)
    return cfg


def stop_group(proc, descendants=()):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    # Includes any descendants left behind after their parent exited.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()
    for child in descendants:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(descendants, timeout=10)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=10)


def wait_bounded(proc, seconds, descendants):
    deadline = time.monotonic() + seconds
    parent = psutil.Process(proc.pid)
    while proc.poll() is None:
        try:
            descendants.update(parent.children(recursive=True))
        except psutil.NoSuchProcess:
            pass
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired(proc.args, seconds)
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    return proc.returncode


def plot(root, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    completed = [r for r in rows if r.get("started_at")]
    if not completed:
        return
    fig, (timeline, metric) = plt.subplots(2, 1, figsize=(15, 10))
    origin = min(r["started_at"] for r in completed)
    for i, row in enumerate(completed):
        label = f'{row["competition"]} / {row["mode"]}'
        color = "#247ba0" if row["mode"] == "conservative" else "#c44e52"
        timeline.barh(i, (row.get("ended_at", time.time()) - row["started_at"]) / 3600,
                      left=(row["started_at"] - origin) / 3600, color=color)
        journals = list((root / row["competition"] / row["mode"] / "runs").glob("*/logs/journal.json"))
        if journals:
            data = json.loads(journals[-1].read_text())
            nodes = data.get("nodes", []) if isinstance(data, dict) else data
            points = []
            for index, node in enumerate(nodes):
                value = node.get("metric")
                if isinstance(value, dict):
                    value = value.get("value")
                if isinstance(value, (float, int)) and not node.get("is_buggy"):
                    points.append((index + 1, value))
            if points:
                metric.plot(*zip(*points), marker="o", label=label)
    timeline.set_yticks(range(len(completed)), [f'{r["competition"]} / {r["mode"]}' for r in completed])
    timeline.set_xlabel("Run timeline (hours)")
    metric.set(xlabel="Candidate node", ylabel="Validation metric (task-specific)")
    if metric.lines:
        metric.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/datasets"))
    parser.add_argument("--seconds", type=int, default=10800)
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.root.mkdir(parents=True, exist_ok=True)
    state_file = args.root / "matrix.json"
    previous = json.loads(state_file.read_text()) if state_file.exists() else []
    prior = {(r["competition"], r["mode"]): r for r in previous}
    rows = []
    for competition in COMPETITIONS:
        public = args.data_root / competition / "prepared/public"
        for mode in ("conservative", "normal"):
            old = prior.get((competition, mode), {})
            if old.get("status") in {"finished", "failed", "timeout"}:
                rows.append(old)
                continue
            ready = (public / "description.md").is_file() and any(public.glob("*train*"))
            rows.append(dict(competition=competition, mode=mode,
                             status="queued" if ready else "blocked_missing_data",
                             node_budget=args.nodes, wall_seconds=args.seconds))
    save(state_file, rows)
    if args.inventory_only:
        return
    for row in rows:
        if row["status"] != "queued":
            continue
        root = args.root / row["competition"] / row["mode"]
        root.mkdir(parents=True, exist_ok=True)
        public = args.data_root / row["competition"] / "prepared/public"
        config = make_config(repo, root, public, row["competition"], row["mode"], args.seconds, args.nodes)
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        env = dict(os.environ, MLEVOLVE_CONFIG=str(config_path),
                   MLEVOLVE_VLLM_CACHE_SALT=f'hwdb-20260910-{row["competition"]}-{row["mode"]}',
                   PYTHONUNBUFFERED="1")
        row.update(status="running", started_at=time.time(), stdout=str(root / "runtime.log"))
        save(state_file, rows)
        print(json.dumps(row), flush=True)
        with (root / "runtime.log").open("a") as log:
            proc = subprocess.Popen([sys.executable, "run.py"], cwd=repo, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            descendants = set()
            try:
                code = wait_bounded(proc, args.seconds, descendants)
                row.update(status="finished" if code == 0 else "failed", exit_code=code)
            except subprocess.TimeoutExpired:
                row.update(status="timeout", exit_code=124)
            finally:
                stop_group(proc, descendants)
        row["ended_at"] = time.time()
        save(state_file, rows)
        try:
            plot(args.root, rows)
        except Exception as exc:
            print(f"Plot error: {exc}", flush=True)
        print(json.dumps(row), flush=True)
        # Failures at process startup indicate an environment problem, not a task result.
        if row["status"] == "failed" and row["ended_at"] - row["started_at"] < 120:
            raise RuntimeError("Startup failure; inspect runtime.log before launching more runs")


if __name__ == "__main__":
    main()
