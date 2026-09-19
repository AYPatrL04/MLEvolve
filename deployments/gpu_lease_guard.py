"""Monitor actual GPU utilization from container startup; release on stalled work."""

from collections import deque
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def low_utilization(samples, now, window=600, threshold=40):
    recent = [(stamp, value) for stamp, value in samples if stamp >= now - window]
    return bool(recent and now - recent[0][0] >= window - 15 and
                sum(value for _, value in recent) / len(recent) < threshold)


def main():
    folder = Path(sys.argv[1])
    folder.mkdir(parents=True, exist_ok=True)
    samples = deque(maxlen=1000)
    proc = subprocess.Popen(sys.argv[2:], start_new_session=True)
    started = time.monotonic()
    reason = None
    try:
        with (folder / "resource-monitor.jsonl").open("a", buffering=1) as stream:
            while proc.poll() is None:
                now = time.monotonic()
                try:
                    probe = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                                           capture_output=True, text=True, timeout=8, check=True)
                    util, used, total = map(float, probe.stdout.strip().splitlines()[0].split(","))
                except Exception:
                    reason = "GPU telemetry unavailable"
                    break
                samples.append((now, util))
                stream.write(json.dumps({"time": time.time(), "elapsed": now-started, "gpu_util": util, "memory_used_mb": used, "memory_total_mb": total}) + "\n")
                if low_utilization(samples, now):
                    reason = "10-minute average GPU utilization below 40%; stop and inspect"
                    break
                if now-started > 4200:
                    reason = "70-minute worker safety deadline"
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
