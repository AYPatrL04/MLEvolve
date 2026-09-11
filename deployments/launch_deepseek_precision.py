"""Idempotent, CPU-gated launcher for the user-authorized DeepSeek rerun."""

import argparse
import json
from pathlib import Path
import subprocess


PREFIX = "hwdb-deepseek-20260911"
LAUNCHER_CONFIG = PREFIX + "-launcher-v4"
SOURCE_COMMIT = "d57609bdeda55d00b6f3734b38f84b80bdf4ca9e"
IMAGE = "nvcr.io/nvidia/pytorch:26.04-py3"
KUBECTL = ["kubectl", "--context", "nautilus", "-n", "ecepxie"]


def kubectl(*args, payload=None):
    return subprocess.check_output(KUBECTL + list(args), input=payload, text=True)


def manifest(phase, image=IMAGE):
    gpu = phase == "run"
    resources = {
        "requests": {"cpu": "14" if gpu else "7", "memory": "56Gi" if gpu else "40Gi", "ephemeral-storage": "60Gi"},
        "limits": {"cpu": "16" if gpu else "8", "memory": "64Gi" if gpu else "48Gi", "ephemeral-storage": "120Gi"},
    }
    if gpu:
        for value in resources.values():
            value["nvidia.com/gpu"] = "1"
    mounts = [
        {"name": "workspace", "mountPath": "/experiment", "subPath": "aypatrl04-hwdb-deepseek-20260911"},
        {"name": "runtime", "mountPath": "/runtime"},
        {"name": "launcher", "mountPath": "/launcher", "readOnly": True},
        {"name": "shm", "mountPath": "/dev/shm"},
    ]
    for competition in ("spooky-author-identification", "jigsaw-toxic-comment-classification-challenge"):
        mounts.append({"name": "workspace", "mountPath": f"/datasets/{competition}/prepared/public",
                       "subPath": f"data/mlebench/{competition}/prepared/public", "readOnly": True})
    pod = {
        "restartPolicy": "Never", "automountServiceAccountToken": False,
        "containers": [{"name": "matrix", "image": image,
                        "command": ["bash", "/launcher/bootstrap_deepseek_precision.sh", phase],
                        "env": [{"name": "SOURCE_COMMIT", "value": SOURCE_COMMIT},
                                {"name": "DEEPSEEK_API_KEY", "valueFrom": {"secretKeyRef": {"name": PREFIX, "key": "api-key"}}}],
                        "resources": resources, "volumeMounts": mounts}],
        "volumes": [{"name": "workspace", "persistentVolumeClaim": {"claimName": "yuze-li-vol"}},
                    {"name": "runtime", "emptyDir": {"sizeLimit": "100Gi"}},
                    {"name": "launcher", "configMap": {"name": LAUNCHER_CONFIG}},
                    {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}}],
    }
    if phase == "data":
        pod["containers"][0]["env"] = [{"name": "KAGGLE_CONFIG_DIR", "value": "/credentials"}]
        pod["containers"][0]["volumeMounts"].extend([
            {"name": "kaggle", "mountPath": "/credentials", "readOnly": True},
            {"name": "workspace", "mountPath": "/heldout", "subPath": "aypatrl04-hwdb-heldout-20260911"},
        ])
        pod["volumes"].append({"name": "kaggle", "secret": {"secretName": "hwdb-kaggle-20260911", "defaultMode": 256}})
    if gpu:
        pod["tolerations"] = [{"key": "nvidia.com/gpu", "operator": "Exists"}]
        pod["affinity"] = {"nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [{"matchExpressions": [
                {"key": "nvidia.com/gpu.product", "operator": "In", "values": [
                    "NVIDIA-A10", "NVIDIA-A100-SXM4-80GB", "NVIDIA-A100-80GB-PCIe", "NVIDIA-A100-PCIE-40GB", "NVIDIA-A100-SXM4-40GB"]}]}]},
            "preferredDuringSchedulingIgnoredDuringExecution": [{"weight": 50, "preference": {"matchExpressions": [
                {"key": "nvidia.com/gpu.product", "operator": "In", "values": ["NVIDIA-A100-SXM4-80GB", "NVIDIA-A100-80GB-PCIe"]}]}}],
        }}
    return {"apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": PREFIX + "-" + phase, "namespace": "ecepxie"},
            "spec": {"backoffLimit": 0, "activeDeadlineSeconds": 280800 if gpu else (43200 if phase == "data" else 10800),
                     "template": {"metadata": {"labels": {"app": PREFIX, "phase": phase}}, "spec": pod}}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "data", "run"))
    args = parser.parse_args()
    existing = kubectl("get", "job", PREFIX + "-" + args.phase, "--ignore-not-found", "-o", "json")
    if existing.strip():
        print("Job already exists; not restarting: " + PREFIX + "-" + args.phase)
        return
    image = IMAGE
    if args.phase in {"prepare", "data"}:
        folder = Path(__file__).parent
        config = {"apiVersion": "v1", "kind": "ConfigMap", "immutable": True,
                  "metadata": {"name": LAUNCHER_CONFIG, "namespace": "ecepxie"},
                  "data": {name: (folder / name).read_text() for name in ("bootstrap_deepseek_precision.sh", "git_retry.sh", "prepare_kaggle_data.py")}}
        print(kubectl("apply", "-f", "-", payload=json.dumps(config)))
    if args.phase != "prepare":
        prep = json.loads(kubectl("get", "job", PREFIX + "-prepare", "-o", "json"))
        if not any(c["type"] == "Complete" and c["status"] == "True" for c in prep.get("status", {}).get("conditions", [])):
            raise SystemExit("CPU preparation has not completed successfully; no GPU requested.")
        pods = json.loads(kubectl("get", "pods", "-l", f"app={PREFIX},phase=prepare", "-o", "json"))["items"]
        successful = [p for p in pods if p["status"]["phase"] == "Succeeded"]
        image = successful[-1]["status"]["containerStatuses"][0]["imageID"].removeprefix("docker-pullable://")
        if "@sha256:" not in image:
            raise SystemExit("Could not resolve the exact prepared container image; no GPU requested.")
        if args.phase == "run":
            data = json.loads(kubectl("get", "job", PREFIX + "-data", "-o", "json"))
            if not any(c["type"] == "Complete" and c["status"] == "True" for c in data.get("status", {}).get("conditions", [])):
                raise SystemExit("Dataset preparation has not finished; no GPU requested.")
    print(kubectl("apply", "-f", "-", payload=json.dumps(manifest(args.phase, image))))


if __name__ == "__main__":
    main()
