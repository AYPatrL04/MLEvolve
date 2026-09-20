"""Bound how long one worker may hold a GPU and record utilization telemetry.

Utilization is recorded but no longer gates the run: a short or bursty
workload can legitimately read 0% between samples. The only hard bound is the
wall-clock lease below, which is the account-level commitment.
"""

from collections import deque
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


LEASE_SECONDS = float(os.environ.get("MLEVOLVE_GPU_LEASE_SECONDS") or 10800)
TELEMETRY_FAILURE_LIMIT = 3


def main():
    folder = Path(sys.argv[1])
    folder.mkdir(parents=True, exist_ok=True)
    samples = deque(maxlen=1000)
    proc = subprocess.Popen(sys.argv[2:], start_new_session=True)
    started = time.monotonic()
    reason = None
    telemetry_failures = 0
    try:
        with (folder / "resource-monitor.jsonl").open("a", buffering=1) as stream:
            while proc.poll() is None:
                now = time.monotonic()
                try:
                    probe = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                                           capture_output=True, text=True, timeout=8, check=True)
                    util, used, total = map(float, probe.stdout.strip().splitlines()[0].split(","))
                    telemetry_failures = 0
                except Exception:
                    telemetry_failures += 1
                    if telemetry_failures >= TELEMETRY_FAILURE_LIMIT:
                        reason = "GPU telemetry unavailable"
                        break
                    time.sleep(10)
                    continue
                samples.append((now, util))
                stream.write(json.dumps({"time": time.time(), "elapsed": now-started, "gpu_util": util, "memory_used_mb": used, "memory_total_mb": total}) + "\n")
                if now-started > LEASE_SECONDS:
                    reason = f"{int(LEASE_SECONDS)}-second GPU lease expired"
                    break
                time.sleep(10)
    finally:
        if reason or proc.poll() is None:
            (folder / "guard-stop.json").write_text(json.dumps({"reason": reason or "worker interrupted", "time": time.time()}) + "\n")
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    raise SystemExit(1 if reason else proc.returncode)


if __name__ == "__main__":
    main()
