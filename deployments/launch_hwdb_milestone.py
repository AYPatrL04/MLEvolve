"""CPU-gated first-valid-node test, preserving the suspended comparison."""

import argparse
import json
from pathlib import Path
import re

from deployments.launch_deepseek_precision import kubectl, manifest as base_manifest, PREFIX as BASE

PREFIX = "hwdb-milestone-20260912"
CONFIG = PREFIX + "-launcher-v1"


def manifest(phase, source_commit, image):
    job = base_manifest(phase, image)
    job["metadata"]["name"] = PREFIX + "-" + phase
    job["spec"]["template"]["metadata"]["labels"] = {"app": PREFIX, "phase": phase}
    if phase == "run":
        job["spec"].pop("activeDeadlineSeconds")
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    container["command"] = ["bash", "/launcher/bootstrap_hwdb_milestone.sh", phase]
    container["env"][0] = {"name": "SOURCE_COMMIT", "value": source_commit}
    container["volumeMounts"] = [m for m in container["volumeMounts"] if not m["mountPath"].startswith("/datasets/")]
    for volume in pod["volumes"]:
        if volume["name"] == "launcher":
            volume["configMap"]["name"] = CONFIG
    return job


def complete(job):
    return any(c["type"] == "Complete" and c["status"] == "True" for c in job.get("status", {}).get("conditions", []))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.source_commit):
        raise SystemExit("A full immutable source commit is required")
    existing = kubectl("get", "job", PREFIX + "-" + args.phase, "--ignore-not-found", "-o", "json")
    if existing.strip():
        print("Already exists; not restarting " + PREFIX + "-" + args.phase)
        return
    for name in (BASE + "-prepare", BASE + "-data"):
        if not complete(json.loads(kubectl("get", "job", name, "-o", "json"))):
            raise SystemExit("Original runtime/data preparation must be complete")
    old = json.loads(kubectl("get", "job", BASE + "-run", "-o", "json"))
    if not old["spec"].get("suspend"):
        raise SystemExit("Suspend the old comparison before starting the milestone")
    if args.phase == "run":
        prep = json.loads(kubectl("get", "job", PREFIX + "-prepare", "-o", "json"))
        if not complete(prep):
            raise SystemExit("Milestone CPU tests have not completed; no GPU requested")
        prepared_source = prep["spec"]["template"]["spec"]["containers"][0]["env"][0]["value"]
        if prepared_source != args.source_commit:
            raise SystemExit("Source differs from the CPU-tested runtime")
    prep_app = PREFIX if args.phase == "run" else BASE
    pods = json.loads(kubectl("get", "pods", "-l", f"app={prep_app},phase=prepare", "-o", "json"))["items"]
    successful = [p for p in pods if p["status"]["phase"] == "Succeeded"]
    image = successful[-1]["status"]["containerStatuses"][0]["imageID"].removeprefix("docker-pullable://")
    if "@sha256:" not in image:
        raise SystemExit("No exact prepared image digest")
    folder = Path(__file__).parent
    config = {"apiVersion": "v1", "kind": "ConfigMap", "immutable": True,
              "metadata": {"name": CONFIG, "namespace": "ecepxie"},
              "data": {n: (folder / n).read_text() for n in ("bootstrap_hwdb_milestone.sh", "git_retry.sh")}}
    print(kubectl("apply", "-f", "-", payload=json.dumps(config)))
    print(kubectl("apply", "-f", "-", payload=json.dumps(manifest(args.phase, args.source_commit, image))))


if __name__ == "__main__":
    main()
