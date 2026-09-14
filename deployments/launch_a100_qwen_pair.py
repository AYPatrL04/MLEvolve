"""Stage and launch two CPU-gated A100 runs with local Qwen agents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


PREFIX = "mlevolve-a100-qwen-s42-20260914"
ROOT = "/experiment/" + PREFIX
PVC_ROOT = "/workspace/aypatrl04-hwdb-deepseek-20260911/" + PREFIX
HELPER = "mlevolve-agentic-knowledge-base-dev-cpu"
IMAGE = "nvcr.io/nvidia/pytorch@sha256:192d749b4d773610ec9e01c0443a9df545d196c412b7b8fd33bfa3da362a49e7"
KUBECTL = [
    "kubectl",
    "--context",
    "nautilus",
    "--request-timeout=120s",
    "-n",
    "ecepxie",
]
LAUNCHER_FILES = (
    "bootstrap_a100_qwen_task.sh",
    "bootstrap_qwen38_a100.sh",
    "run_a100_qwen_task.py",
)
TASKS = ("petfinder", "full")


def kubectl(*args: str, payload: str | None = None) -> str:
    return subprocess.check_output(
        KUBECTL + list(args),
        input=payload,
        text=True,
    )


def complete(job: dict) -> bool:
    return any(
        condition.get("type") == "Complete" and condition.get("status") == "True"
        for condition in job.get("status", {}).get("conditions", [])
    )


def manifest(phase: str, digest: str, task: str | None = None) -> dict:
    name = PREFIX + "-" + phase if task is None else PREFIX + "-" + task
    app = PREFIX
    labels = {"app": app, "phase": phase}
    if task:
        labels["task"] = task
    resources = {
        "requests": {
            "cpu": "8",
            "memory": "48Gi" if phase == "run" else "24Gi",
            "ephemeral-storage": "32Gi",
        },
        "limits": {
            "cpu": "8",
            "memory": "64Gi" if phase == "run" else "32Gi",
            "ephemeral-storage": "48Gi",
        },
    }
    mounts = [
        {"name": "workspace", "mountPath": "/experiment", "subPath": "aypatrl04-hwdb-deepseek-20260911"},
        {"name": "runtime", "mountPath": "/runtime"},
        {"name": "launcher", "mountPath": "/launcher", "readOnly": True},
        {"name": "shm", "mountPath": "/dev/shm"},
    ]
    for competition in ("petfinder-pawpularity-score", "nlp-getting-started"):
        mounts.append(
            {
                "name": "workspace",
                "mountPath": f"/datasets/{competition}/prepared/public",
                "subPath": f"data/mlebench/{competition}/prepared/public",
                "readOnly": True,
            }
        )
    mounts.append(
        {"name": "home", "mountPath": "/root", "readOnly": phase == "prepare"}
    )
    if phase == "run":
        for value in resources.values():
            value["nvidia.com/a100"] = "1"
    command = ["bash", "/launcher/bootstrap_a100_qwen_task.sh", phase]
    if task:
        command.append(task)
    environment = [{"name": "SOURCE_SHA256", "value": digest}]
    if phase == "run":
        environment.extend(
            [
                {"name": "QWEN_MODEL_DIR", "value": "/root/downeyflyfan/qwen38-v100-int8/models/Qwen3.8-27B-INT8-W8A16-MTP"},
                {"name": "QWEN_SERVED_MODEL_NAME", "value": "qwen3.8-27b-int8-a100"},
                {"name": "QWEN_GPU_MEMORY_UTILIZATION", "value": "0.50"},
                {"name": "QWEN_MAX_MODEL_LEN", "value": "65536"},
            ]
        )
    pod = {
        "restartPolicy": "Never",
        "automountServiceAccountToken": False,
        "containers": [
            {
                "name": "matrix",
                "image": IMAGE,
                "command": command,
                "env": environment,
                "resources": resources,
                "volumeMounts": mounts,
            }
        ],
        "volumes": [
            {"name": "workspace", "persistentVolumeClaim": {"claimName": "yuze-li-vol"}},
            {"name": "home", "persistentVolumeClaim": {"claimName": "yuw-home"}},
            {"name": "runtime", "emptyDir": {"sizeLimit": "64Gi"}},
            {"name": "launcher", "configMap": {"name": PREFIX + "-launcher-v1"}},
            {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}},
        ],
    }
    if phase == "run":
        pod["tolerations"] = [{"key": "nvidia.com/gpu", "operator": "Exists"}]
        pod["affinity"] = {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": "nvidia.com/gpu.product",
                                    "operator": "In",
                                    "values": ["NVIDIA-A100-SXM4-80GB", "NVIDIA-A100-80GB-PCIe"],
                                }
                            ]
                        }
                    ]
                }
            }
        }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "namespace": "ecepxie", "labels": labels},
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 604800 if phase == "run" else 21600,
            "template": {"metadata": {"labels": labels}, "spec": pod},
        },
    }


def stage(repo: Path, records: Path) -> None:
    if (records / "source.json").exists():
        raise SystemExit("Snapshot already staged; reuse it instead of overwriting it")
    paths = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=repo).decode().split("\0"))
    for submodule in ("PerfSeer-predictor", "nn-model-preflight-checker"):
        paths.discard(submodule)
        names = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo / submodule).decode().split("\0")
        paths.update(str(Path(submodule) / name) for name in names if name)
    paths.update("deployments/" + name for name in (*LAUNCHER_FILES, "launch_a100_qwen_pair.py"))
    paths.add("tests/test_a100_qwen_deployment.py")
    paths.discard("")
    records.mkdir(parents=True, exist_ok=True)
    bundle = records / "source.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for name in sorted(paths):
            archive.add(repo / name, arcname="repo/" + name, recursive=False)
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    metadata = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "submodules": subprocess.check_output(["git", "submodule", "status", "--recursive"], cwd=repo, text=True),
        "working_tree": subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True),
        "sha256": digest,
        "files": len(paths),
        "root": ROOT,
    }
    kubectl("exec", HELPER, "--", "mkdir", "-p", PVC_ROOT)
    receiver = (
        "import pathlib,shutil,sys; p=pathlib.Path(sys.argv[1]); "
        "assert not p.exists(); q=p.with_suffix('.partial'); "
        "f=q.open('wb'); shutil.copyfileobj(sys.stdin.buffer,f); f.close(); q.rename(p)"
    )
    with bundle.open("rb") as stream:
        subprocess.run(
            KUBECTL
            + ["exec", "-i", HELPER, "--", "python", "-c", receiver, PVC_ROOT + "/source.tar.gz"],
            stdin=stream,
            check=True,
        )
    (records / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def apply_launcher(repo: Path) -> None:
    config = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "immutable": True,
        "metadata": {"name": PREFIX + "-launcher-v1", "namespace": "ecepxie"},
        "data": {name: (repo / "deployments" / name).read_text() for name in LAUNCHER_FILES},
    }
    print(kubectl("apply", "-f", "-", payload=json.dumps(config)))


def ensure_absent(name: str) -> None:
    if kubectl("get", "job", name, "--ignore-not-found", "-o", "name").strip():
        raise SystemExit("Already submitted; not restarting: " + name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("stage", "prepare", "run"))
    parser.add_argument("--task", choices=TASKS + ("both",), default="both")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    records = repo / "records" / PREFIX
    if args.phase == "stage":
        stage(repo, records)
        return
    metadata = json.loads((records / "source.json").read_text())
    if args.phase == "prepare":
        ensure_absent(PREFIX + "-prepare")
        apply_launcher(repo)
        print(kubectl("create", "-f", "-", payload=json.dumps(manifest("prepare", metadata["sha256"]))))
        return
    prepared = json.loads(kubectl("get", "job", PREFIX + "-prepare", "-o", "json"))
    if not complete(prepared):
        raise SystemExit("CPU gate is not complete; no A100 jobs submitted")
    prepared_container = prepared["spec"]["template"]["spec"]["containers"][0]
    if prepared_container["image"] != IMAGE:
        raise SystemExit("Prepared image differs from the requested A100 image")
    tasks = TASKS if args.task == "both" else (args.task,)
    apply_launcher(repo)
    for task in tasks:
        ensure_absent(PREFIX + "-" + task)
        print(
            kubectl(
                "create",
                "-f",
                "-",
                payload=json.dumps(manifest("run", metadata["sha256"], task)),
            )
        )


if __name__ == "__main__":
    main()
