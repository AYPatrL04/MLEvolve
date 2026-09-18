"""Launch a fresh HWDB ablation on the current code revision and A10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from deployments.launch_deepseek_precision import kubectl
from deployments.launch_hwdb_milestone import complete
from deployments.launch_hwdb_ablation import manifest as legacy_manifest


PREFIX = "hwdb-content-ablation-current-s42-20260918"
CONFIG = PREFIX + "-launcher-v1"
ROOT = "/experiment/" + PREFIX


def manifest(phase: str, source_commit: str) -> dict:
    job = legacy_manifest(phase, source_commit)
    job["metadata"]["name"] = PREFIX + "-" + phase
    job["spec"]["template"]["metadata"]["labels"] = {"app": PREFIX, "phase": phase}
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    container["command"] = ["bash", "/launcher/bootstrap_hwdb_ablation.sh", phase]
    container["env"].append({"name": "ABLATION_ROOT", "value": ROOT})
    container["env"].append({"name": "MLEVOLVE_ABLATION_SEEDS", "value": "42"})
    if phase == "run":
        expressions = pod["affinity"]["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ]["nodeSelectorTerms"][0]["matchExpressions"]
        expressions[0]["values"] = ["NVIDIA-A10"]
    for volume in pod["volumes"]:
        if volume["name"] == "launcher":
            volume["configMap"]["name"] = CONFIG
    return job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.source_commit):
        raise SystemExit("Full immutable source commit required")
    if kubectl(
        "get",
        "job",
        PREFIX + "-" + args.phase,
        "--ignore-not-found",
        "-o",
        "json",
    ).strip():
        raise SystemExit("Already exists; refusing to restart " + PREFIX + "-" + args.phase)
    for old in ("hwdb-deepseek-20260911-run", "hwdb-milestone-20260912-run"):
        if not json.loads(kubectl("get", "job", old, "-o", "json"))["spec"].get("suspend"):
            raise SystemExit("Historical searches must remain suspended")
    if args.phase == "run":
        prepared = json.loads(
            kubectl("get", "job", PREFIX + "-prepare", "-o", "json")
        )
        if not complete(prepared):
            raise SystemExit("CPU gate has not passed; no GPU requested")
        tested = prepared["spec"]["template"]["spec"]["containers"][0]
        env = {item["name"]: item.get("value") for item in tested["env"]}
        if env.get("SOURCE_COMMIT") != args.source_commit:
            raise SystemExit("Run source does not match the CPU-tested source")
    folder = Path(__file__).parent
    config = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "immutable": True,
        "metadata": {"name": CONFIG, "namespace": "ecepxie"},
        "data": {
            name: (folder / name).read_text()
            for name in ("bootstrap_hwdb_ablation.sh", "git_retry.sh")
        },
    }
    print(kubectl("apply", "-f", "-", payload=json.dumps(config)))
    print(
        kubectl(
            "apply",
            "-f",
            "-",
            payload=json.dumps(manifest(args.phase, args.source_commit)),
        )
    )


if __name__ == "__main__":
    main()
