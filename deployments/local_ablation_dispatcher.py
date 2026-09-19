"""Local credential holder for a CPU-only Nautilus search; no credentials copied."""

import argparse
import json
from pathlib import Path
import subprocess
import time

from deployments.launch_queued_ablation import PREFIX, ROOT
from deployments.queued_ablation_coordinator import run_worker
from deployments.run_hwdb_precision_matrix import save

KUBE = ["kubectl", "--context", "nautilus", "--request-timeout=30s", "-n", "ecepxie"]


def kube(*args, payload=None):
    return subprocess.check_output(KUBE + list(args), input=payload, text=True, timeout=60)


class Cluster:
    def request(self, path, method="GET", payload=None, raw=False):
        if method == "POST":
            output = kube("create", "-f", "-", "-o", "json", payload=json.dumps(payload))
        elif method == "DELETE":
            kube("delete", "job", path.rsplit("/", 1)[1], "--cascade=foreground", "--wait=false")
            return {}
        elif raw:
            return kube("logs", path.split("/")[-2])
        else:
            output = kube("get", "job", path.rsplit("/", 1)[1], "-o", "json")
        return json.loads(output)

    def pods(self, job):
        return json.loads(kube("get", "pods", "-l", "job-name=" + job, "-o", "json"))["items"]


def remote_json(pod, path, payload):
    kube("exec", "-i", pod, "--", "python", "-c",
         "import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); t=p.with_suffix('.tmp'); t.write_text(json.dumps(json.load(sys.stdin))); t.replace(p)",
         path, payload=json.dumps(payload))


def archive(pod, remote, local):
    local.mkdir(parents=True, exist_ok=True)
    # Includes prompts, attempts, scheduler evidence and worker resource telemetry.
    source = subprocess.Popen(KUBE + ["exec", pod, "--", "tar", "--exclude=repo.tar", "--exclude=*.partial", "--exclude=*.pyc", "-C", remote, "-cf", "-", "."], stdout=subprocess.PIPE)
    try:
        subprocess.run(["tar", "-C", str(local), "-xf", "-"], stdin=source.stdout, check=True, timeout=180)
        source.stdout.close()
        if source.wait(timeout=30):
            raise RuntimeError("Remote archive stream failed")
    finally:
        if source.poll() is None:
            source.kill()
            source.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--records", type=Path, required=True)
    args = parser.parse_args()
    args.records.mkdir(parents=True, exist_ok=True)
    cluster = Cluster()
    failures = 0
    while True:
        try:
            pods = cluster.pods(PREFIX + "-coordinator")
            if len(pods) != 1 or pods[0]["status"]["phase"] == "Pending":
                time.sleep(15)
                continue
            pod = pods[0]["metadata"]["name"]
            if pods[0]["status"]["phase"] != "Running":
                save(args.records / "dispatcher-stopped.json", {"reason": "CPU coordinator is terminal", "pod": pods[0]})
                return
            state = json.loads(kube("exec", pod, "--", "python", "-c",
                "import json,pathlib,sys; r=pathlib.Path(sys.argv[1]); print(json.dumps({'stop':(r/'STOP.json').exists(),'complete':(r/'COMPLETE.json').exists(),'requests':[str(p.parent) for p in [r/'probe/request.json', *sorted((r/'queue').glob('*/request.json'))] if p.exists() and not (p.parent/'released.json').exists()]}))", ROOT))
            if state["stop"] or state["complete"]:
                archive(pod, ROOT, args.records / "remote")
                remote_json(pod, ROOT + "/ARCHIVED.json", {"time": time.time(), "local": str(args.records)})
                return
            for remote in state["requests"]:
                folder = args.records / "workers" / Path(remote).name
                try:
                    run_worker(cluster, folder, args.source_commit, remote)
                    archive(pod, remote, folder)
                    released = json.loads((folder / "released.json").read_text())
                    remote_json(pod, remote + "/released.json", released)
                except BaseException as exc:
                    try:
                        archive(pod, ROOT, args.records / "remote")
                    finally:
                        remote_json(pod, ROOT + "/STOP.json", {"reason": "GPU dispatch failed", "error_type": type(exc).__name__, "time": time.time()})
                    raise
            failures = 0
            save(args.records / "dispatcher-heartbeat.json", {"time": time.time(), "pod": pod})
        except subprocess.CalledProcessError:
            failures += 1
            if failures >= 3:
                raise
        time.sleep(15)


if __name__ == "__main__":
    main()
