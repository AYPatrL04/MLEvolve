"""CPU coordinator and bounded, credential-free A10 execution jobs."""

import argparse
import json
from pathlib import Path
import re

from deployments.launch_hwdb_ablation import manifest as base_manifest
from deployments.launch_deepseek_precision import kubectl

PREFIX = "hwkg-merge-s42-20260919"
ROOT = "/experiment/" + PREFIX
CONFIG = PREFIX + "-launcher-v2"


def manifest(phase, commit, request=None):
    gpu = phase == "worker"
    job = base_manifest("run" if gpu else "prepare", commit)
    name = PREFIX + "-" + (Path(request).name[:12] if request else phase)
    job["metadata"]["name"] = name
    job["spec"].update(activeDeadlineSeconds=4500 if gpu else 259200, backoffLimit=0)
    pod = job["spec"]["template"]["spec"]
    job["spec"]["template"]["metadata"]["labels"] = {"app": PREFIX, "phase": phase}
    container = pod["containers"][0]
    container["command"] = (["python", "/launcher/gpu_lease_guard.py", request, "bash", "/launcher/bootstrap_queued_ablation.sh", phase, request]
                            if gpu else ["bash", "/launcher/bootstrap_queued_ablation.sh", phase])
    container["env"] += [
        {"name": "ABLATION_ROOT", "value": ROOT},
        {"name": "MLEVOLVE_ABLATION_SEEDS", "value": "42"},
        {"name": "MLEVOLVE_ABLATION_EXACT_BUDGET", "value": "1"},
        {"name": "MLEVOLVE_ABLATION_BASELINE", "value": "1bf8d2f27bce5b289cb29caec736722b2efd3d8a"},
    ]
    if gpu:
        container["env"] = [item for item in container["env"] if "valueFrom" not in item]
        pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"][0]["values"] = ["NVIDIA-A10"]
        for resources in container["resources"].values():
            resources["memory"] = "24Gi"
    for volume in pod["volumes"]:
        if volume["name"] == "launcher":
            volume["configMap"]["name"] = CONFIG
    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "coordinator"))
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.source_commit):
        raise ValueError("Full source commit required")
    if kubectl("get", "job", PREFIX + "-" + args.phase, "--ignore-not-found", "-o", "name").strip():
        raise RuntimeError("Job already exists; refusing to restart")
    if args.phase == "coordinator":
        prepared = json.loads(kubectl("get", "job", PREFIX + "-prepare", "-o", "json"))
        if not any(c["type"] == "Complete" and c["status"] == "True" for c in prepared.get("status", {}).get("conditions", [])):
            raise RuntimeError("CPU preparation must complete first")
        env = prepared["spec"]["template"]["spec"]["containers"][0]["env"]
        if next(e["value"] for e in env if e["name"] == "SOURCE_COMMIT") != args.source_commit:
            raise RuntimeError("Prepared source differs")
    folder = Path(__file__).parent
    config = {"apiVersion": "v1", "kind": "ConfigMap", "immutable": True,
              "metadata": {"name": CONFIG, "namespace": "ecepxie"},
              "data": {name: (folder / name).read_text() for name in
                       ("bootstrap_queued_ablation.sh", "gpu_lease_guard.py", "git_retry.sh")}}
    print(kubectl("apply", "-f", "-", payload=json.dumps(config)))
    print(kubectl("create", "-f", "-", payload=json.dumps(manifest(args.phase, args.source_commit))))


if __name__ == "__main__":
    main()
