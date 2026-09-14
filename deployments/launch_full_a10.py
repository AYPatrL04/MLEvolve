"""Stage the local checkout and launch an isolated, CPU-gated A10 task."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from deployments.launch_disaster_comparison import IMAGE, manifest as base_manifest


PREFIX = "mlevolve-full-a10-s42-20260914"
ROOT = "/experiment/" + PREFIX
PVC_ROOT = "/workspace/aypatrl04-hwdb-deepseek-20260911/" + PREFIX
HELPER = "mlevolve-agentic-knowledge-base-dev-cpu"
KUBECTL = ["kubectl", "--context", "nautilus", "--request-timeout=120s", "-n", "ecepxie"]
FILES = ("bootstrap_full_a10.sh", "run_full_a10.py")


def kubectl(*args, payload=None):
    return subprocess.check_output(KUBECTL + list(args), input=payload, text=True)


def manifest(phase, digest):
    job = base_manifest(phase, "local-snapshot")
    job["metadata"]["name"] = PREFIX + "-" + phase
    spec = job["spec"]
    spec["activeDeadlineSeconds"] = 280800 if phase == "run" else 10800
    spec["template"]["metadata"]["labels"] = {"app": PREFIX, "phase": phase}
    pod = spec["template"]["spec"]
    container = pod["containers"][0]
    container["command"] = ["bash", "/launcher/bootstrap_full_a10.sh", phase]
    container["env"] = [
        {"name": "SOURCE_SHA256", "value": digest},
        {"name": "DEEPSEEK_API_KEY", "valueFrom": {"secretKeyRef": {
            "name": "hwdb-deepseek-20260911", "key": "api-key"}}},
    ]
    for resources in container["resources"].values():
        resources.update(cpu="8", memory="24Gi")
    for volume in pod["volumes"]:
        if volume["name"] == "launcher":
            volume["configMap"]["name"] = PREFIX + "-launcher"
    if phase == "run":
        pod["affinity"]["nodeAffinity"] = {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [{"matchExpressions": [{
                    "key": "nvidia.com/gpu.product", "operator": "In", "values": ["NVIDIA-A10"]}]}]}}
    return job


def stage(repo, records):
    if (records / "source.json").exists():
        raise SystemExit("Snapshot already staged locally; reuse it, do not overwrite this run's source.")
    paths = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=repo).decode().split("\0"))
    submodules = ("PerfSeer-predictor", "nn-model-preflight-checker")
    for submodule in submodules:
        paths.discard(submodule)
        names = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo / submodule).decode().split("\0")
        paths.update(str(Path(submodule) / name) for name in names if name)
    paths.update("deployments/" + name for name in (*FILES, "launch_full_a10.py"))
    paths.add("tests/test_full_a10_deployment.py")
    paths.discard("")
    records.mkdir(parents=True, exist_ok=True)
    bundle = records / "source.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for name in sorted(paths):
            archive.add(repo / name, arcname="repo/" + name, recursive=False)
    with bundle.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "submodules": subprocess.check_output(["git", "submodule", "status", "--recursive"], cwd=repo, text=True),
        "working_tree": subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True),
        "sha256": digest, "files": len(paths), "root": ROOT,
    }
    kubectl("exec", HELPER, "--", "mkdir", "-p", PVC_ROOT)
    receiver = ("import pathlib,shutil,sys; p=pathlib.Path(sys.argv[1]); "
                "assert not p.exists(); q=p.with_suffix('.partial'); "
                "f=q.open('wb'); shutil.copyfileobj(sys.stdin.buffer,f); f.close(); q.rename(p)")
    with bundle.open("rb") as stream:
        subprocess.run(KUBECTL + ["exec", "-i", HELPER, "--", "python", "-c", receiver,
                                  PVC_ROOT + "/source.tar.gz"], stdin=stream, check=True)
    (records / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("stage", "prepare", "run"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    records = repo / "records" / PREFIX
    if args.phase == "stage":
        stage(repo, records)
        return
    metadata = json.loads((records / "source.json").read_text())
    name = PREFIX + "-" + args.phase
    if kubectl("get", "job", name, "--ignore-not-found", "-o", "name").strip():
        print("Already submitted; not restarting: " + name)
        return
    if args.phase == "run":
        prep = json.loads(kubectl("get", "job", PREFIX + "-prepare", "-o", "json"))
        if not any(c["type"] == "Complete" and c["status"] == "True"
                   for c in prep.get("status", {}).get("conditions", [])):
            raise SystemExit("CPU gate not complete; no GPU job submitted.")
        prepared = prep["spec"]["template"]["spec"]["containers"][0]
        assert prepared["image"] == IMAGE
        assert {e["name"]: e.get("value") for e in prepared["env"]}["SOURCE_SHA256"] == metadata["sha256"]
    config = {"apiVersion": "v1", "kind": "ConfigMap", "immutable": True,
              "metadata": {"name": PREFIX + "-launcher", "namespace": "ecepxie"},
              "data": {name: (repo / "deployments" / name).read_text() for name in FILES}}
    job = manifest(args.phase, metadata["sha256"])
    records.mkdir(parents=True, exist_ok=True)
    (records / (args.phase + "-job.json")).write_text(json.dumps(job, indent=2) + "\n")
    print(kubectl("apply", "-f", "-", payload=json.dumps(config)))
    print(kubectl("create", "-f", "-", payload=json.dumps(job)))


if __name__ == "__main__":
    main()
