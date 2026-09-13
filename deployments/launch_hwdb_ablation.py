"""CPU-gated launch of the user-authorized fresh-agent HWDB-content ablation."""

import argparse
import json
from pathlib import Path
import re

from deployments.launch_deepseek_precision import kubectl
from deployments.launch_disaster_comparison import IMAGE, manifest as base_manifest
from deployments.launch_hwdb_milestone import complete


PREFIX = "hwdb-content-ablation-20260913"
CONFIG = PREFIX + "-launcher-v1"


def manifest(phase, source_commit):
    job = base_manifest(phase, source_commit)
    job["metadata"]["name"] = PREFIX + "-" + phase
    job["spec"]["template"]["metadata"]["labels"] = {"app": PREFIX, "phase": phase}
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    container["command"] = ["bash", "/launcher/bootstrap_hwdb_ablation.sh", phase]
    container["env"].append({"name": "DEEPSEEK_API_KEY", "valueFrom": {"secretKeyRef": {"name": "hwdb-deepseek-20260911", "key": "api-key"}}})
    for limits in container["resources"].values():
        limits.update(cpu="8", memory="16Gi")
    for volume in pod["volumes"]:
        if volume["name"] == "launcher":
            volume["configMap"]["name"] = CONFIG
    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.source_commit):
        raise SystemExit("Full immutable source commit required")
    existing = kubectl("get", "job", PREFIX + "-" + args.phase, "--ignore-not-found", "-o", "json")
    if existing.strip():
        print("Already exists; refusing to restart " + PREFIX + "-" + args.phase)
        return
    for name in ("hwdb-deepseek-20260911-run", "hwdb-milestone-20260912-run"):
        if not json.loads(kubectl("get", "job", name, "-o", "json"))["spec"].get("suspend"):
            raise SystemExit("Historical searches must remain suspended")
    if not complete(json.loads(kubectl("get", "job", "hwdb-disaster-compare-20260912-run", "-o", "json"))):
        raise SystemExit("Previous precision experiment must be complete")
    prerequisite = "hwdb-disaster-compare-20260912-prepare" if args.phase == "prepare" else PREFIX + "-prepare"
    prep = json.loads(kubectl("get", "job", prerequisite, "-o", "json"))
    if not complete(prep):
        raise SystemExit("CPU gate has not passed; no GPU requested")
    if args.phase == "run":
        tested = prep["spec"]["template"]["spec"]["containers"][0]
        env = {v["name"]: v.get("value") for v in tested["env"]}
        if env.get("SOURCE_COMMIT") != args.source_commit or tested["image"] != IMAGE:
            raise SystemExit("Source/image does not match CPU-tested runtime")
    folder = Path(__file__).parent
    config = {"apiVersion": "v1", "kind": "ConfigMap", "immutable": True,
              "metadata": {"name": CONFIG, "namespace": "ecepxie"},
              "data": {n: (folder / n).read_text() for n in ("bootstrap_hwdb_ablation.sh", "git_retry.sh")}}
    print(kubectl("apply", "-f", "-", payload=json.dumps(config)))
    print(kubectl("apply", "-f", "-", payload=json.dumps(manifest(args.phase, args.source_commit))))


if __name__ == "__main__":
    main()
