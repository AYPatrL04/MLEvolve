"""CPU-gated, credential-free paired Disaster Tweets precision study."""

import argparse
import json
from pathlib import Path
import re

from deployments.launch_deepseek_precision import kubectl, manifest as base_manifest
from deployments.launch_hwdb_milestone import complete


PREFIX = "hwdb-disaster-compare-20260912"
CONFIG = PREFIX + "-launcher-v1"
IMAGE = "nvcr.io/nvidia/pytorch@sha256:192d749b4d773610ec9e01c0443a9df545d196c412b7b8fd33bfa3da362a49e7"


def manifest(phase, source_commit):
    job = base_manifest(phase, IMAGE)
    job["metadata"]["name"] = PREFIX + "-" + phase
    job["spec"]["template"]["metadata"]["labels"] = {"app": PREFIX, "phase": phase}
    if phase == "run":
        job["spec"].pop("activeDeadlineSeconds")
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    container["command"] = ["bash", "/launcher/bootstrap_disaster_comparison.sh", phase]
    container["env"] = [{"name": "SOURCE_COMMIT", "value": source_commit}]
    container["volumeMounts"] = [m for m in container["volumeMounts"] if not m["mountPath"].startswith("/datasets/")]
    container["resources"] = {"requests": {"cpu": "4", "memory": "8Gi", "ephemeral-storage": "24Gi"},
                              "limits": {"cpu": "4", "memory": "8Gi", "ephemeral-storage": "32Gi"}}
    if phase == "run":
        for limits in container["resources"].values():
            limits["nvidia.com/gpu"] = "1"
        pod["affinity"]["nodeAffinity"]["preferredDuringSchedulingIgnoredDuringExecution"] = [
            {"weight": 80, "preference": {"matchExpressions": [
                {"key": "nvidia.com/gpu.product", "operator": "In", "values": ["NVIDIA-A10"]}]}}]
    for volume in pod["volumes"]:
        if volume["name"] == "launcher":
            volume["configMap"]["name"] = CONFIG
        elif volume["name"] == "runtime":
            volume["emptyDir"]["sizeLimit"] = "24Gi"
        elif volume["name"] == "shm":
            volume["emptyDir"]["sizeLimit"] = "1Gi"
    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.source_commit):
        raise SystemExit("A full immutable source commit is required")
    if kubectl("get", "job", PREFIX + "-" + args.phase, "--ignore-not-found", "-o", "json").strip():
        print("Already exists; not restarting " + PREFIX + "-" + args.phase)
        return
    for old in ("hwdb-deepseek-20260911-run", "hwdb-milestone-20260912-run"):
        if not json.loads(kubectl("get", "job", old, "-o", "json"))["spec"].get("suspend"):
            raise SystemExit("Previous GPU jobs must remain suspended")
    predecessor = "hwdb-milestone-20260912-prepare" if args.phase == "prepare" else PREFIX + "-prepare"
    prepared = json.loads(kubectl("get", "job", predecessor, "-o", "json"))
    if not complete(prepared):
        raise SystemExit("CPU prerequisite has not completed; no GPU requested")
    if args.phase == "run":
        tested = prepared["spec"]["template"]["spec"]["containers"][0]
        env = {v["name"]: v.get("value") for v in tested["env"]}
        if env.get("SOURCE_COMMIT") != args.source_commit or tested["image"] != IMAGE:
            raise SystemExit("Source/image differs from the CPU-tested runtime")
    folder = Path(__file__).parent
    config = {"apiVersion": "v1", "kind": "ConfigMap", "immutable": True,
              "metadata": {"name": CONFIG, "namespace": "ecepxie"},
              "data": {n: (folder / n).read_text() for n in ("bootstrap_disaster_comparison.sh", "git_retry.sh")}}
    print(kubectl("apply", "-f", "-", payload=json.dumps(config)))
    print(kubectl("apply", "-f", "-", payload=json.dumps(manifest(args.phase, args.source_commit))))


if __name__ == "__main__":
    main()
