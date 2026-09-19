"""Own short-lived GPU jobs while the agent search remains on CPU."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError

from deployments.launch_queued_ablation import PREFIX, manifest
from deployments.run_hwdb_precision_matrix import save, stop_group


def run_worker(cluster, folder, commit, request_path=None):
    folder.mkdir(parents=True, exist_ok=True)
    job = manifest("worker", commit, request_path or str(folder))
    name = job["metadata"]["name"]
    endpoint = "/apis/batch/v1/namespaces/ecepxie/jobs"
    save(folder / "job-spec.json", job)
    # A create conflict is not permission to adopt or delete an existing job.
    cluster.request(endpoint, "POST", job)
    started = time.monotonic()
    ok = False
    try:
        while True:
            current = cluster.request(endpoint + "/" + name)
            save(folder / "job-status.json", current)
            conditions = current.get("status", {}).get("conditions", [])
            complete = any(c["type"] == "Complete" and c["status"] == "True" for c in conditions)
            failed = any(c["type"] == "Failed" and c["status"] == "True" for c in conditions)
            if complete or failed:
                ok = complete and not (folder / "guard-stop.json").exists()
                break
            if (folder / "cancel.json").exists() or time.monotonic()-started > 86400:
                break
            time.sleep(5)
    finally:
        # Persistent logs survive even if the container was killed by the guard.
        try:
            pods = cluster.pods(name)
            save(folder / "pods.json", pods)
            for pod in pods:
                pod_name = pod["metadata"]["name"]
                try:
                    output = cluster.request(f"/api/v1/namespaces/ecepxie/pods/{pod_name}/log", raw=True)
                    (folder / "container.log").write_text(output)
                except (HTTPError, subprocess.CalledProcessError):
                    pass  # Pending pods may have no container log.
        finally:
            cluster.request(endpoint + "/" + name, "DELETE", {"propagationPolicy": "Foreground"})
        deadline = time.monotonic() + 180
        while cluster.pods(name):
            if time.monotonic() > deadline:
                raise RuntimeError("GPU pod deletion not confirmed; stopping coordinator")
            time.sleep(3)
    save(folder / "released.json", {"ok": ok, "time": time.time(), "job": name})
    if not ok:
        raise RuntimeError("GPU worker failed; see " + str(folder))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root
    commit = os.environ["SOURCE_COMMIT"]
    queue = root / "queue"
    queue.mkdir(exist_ok=False)
    proc = None
    try:
        probe = root / "probe"
        probe.mkdir()
        save(probe / "request.json", {"kind": "hardware_probe"})
        deadline = time.monotonic() + 86400
        while not (probe / "released.json").exists():
            if (root / "STOP.json").exists() or time.monotonic() > deadline:
                raise RuntimeError("GPU hardware probe unavailable; coordinator stopped")
            time.sleep(5)
        if not json.loads((probe / "released.json").read_text())["ok"]:
            raise RuntimeError("GPU hardware probe failed")
        env = dict(os.environ, MLEVOLVE_EXECUTION_QUEUE=str(queue),
                   MLEVOLVE_TARGET_HARDWARE_PROFILE=str(root / "target-hardware.json"))
        proc = subprocess.Popen([sys.executable, "-m", "deployments.run_hwdb_ablation", "--root", str(root)],
                                env=env, start_new_session=True)
        while proc.poll() is None:
            if (root / "STOP.json").exists():
                raise RuntimeError("GPU dispatcher stopped; inspect preserved evidence")
            time.sleep(2)
        if proc.returncode:
            raise RuntimeError("Ablation search failed; inspect per-cell logs")
        from deployments.plot_queued_ablation import plot
        plot(root)
        save(root / "COMPLETE.json", {"time": time.time(), "source_commit": commit})
        deadline = time.monotonic() + 1800
        while not (root / "ARCHIVED.json").exists() and time.monotonic() < deadline:
            time.sleep(5)
    except BaseException as exc:
        save(root / "STOP.json", {"time": time.time(), "error_type": type(exc).__name__, "reason": str(exc)})
        raise
    finally:
        if proc is not None:
            stop_group(proc)


if __name__ == "__main__":
    main()
