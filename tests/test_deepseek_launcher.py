import json

import pytest

from deployments import launch_deepseek_precision as launcher


def test_prepare_has_no_gpu_and_only_secret_reference():
    pod = launcher.manifest("prepare")["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert not pod["automountServiceAccountToken"]
    assert all("nvidia.com/gpu" not in r for r in container["resources"].values())
    key = next(e for e in container["env"] if e["name"] == "DEEPSEEK_API_KEY")
    assert "value" not in key and key["valueFrom"]["secretKeyRef"]["name"] == launcher.PREFIX
    datasets = [m for m in container["volumeMounts"] if m["mountPath"].startswith("/datasets")]
    assert all(m["readOnly"] and m["subPath"].endswith("/public") for m in datasets)


def test_training_uses_same_runtime_prefix_and_supported_gpu():
    pod = launcher.manifest("run", "test@sha256:abc")["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["image"] == "test@sha256:abc"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    terms = pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
    assert all(v.startswith(("NVIDIA-A10", "NVIDIA-A100")) for v in terms[0]["matchExpressions"][0]["values"])
    assert any(m["mountPath"] == "/runtime" for m in container["volumeMounts"])


def test_failed_prep_never_launches_gpu(monkeypatch):
    calls = []
    def kubectl(*args, **kwargs):
        calls.append(args)
        return "" if "--ignore-not-found" in args else json.dumps({"status": {"conditions": [{"type": "Failed", "status": "True"}]}})
    monkeypatch.setattr(launcher, "kubectl", kubectl)
    monkeypatch.setattr("sys.argv", ["launcher", "run"])
    with pytest.raises(SystemExit, match="no GPU requested"):
        launcher.main()
    assert not any("apply" in call for call in calls)


def test_existing_run_is_not_restarted(monkeypatch):
    calls = []
    def kubectl(*args, **kwargs):
        calls.append(args)
        return '{"metadata": {"name": "existing"}}'
    monkeypatch.setattr(launcher, "kubectl", kubectl)
    monkeypatch.setattr("sys.argv", ["launcher", "run"])
    launcher.main()
    assert len(calls) == 1


def test_data_credentials_and_heldout_labels_are_not_mounted_into_agents():
    data = launcher.manifest("data")["spec"]["template"]["spec"]
    agent = launcher.manifest("run")["spec"]["template"]["spec"]
    assert any(v.get("secret", {}).get("secretName") == "hwdb-kaggle-20260911" for v in data["volumes"])
    assert all(v["name"] != "kaggle" for v in agent["volumes"])
    assert any(m["mountPath"] == "/heldout" for m in data["containers"][0]["volumeMounts"])
    assert all(m["mountPath"] != "/heldout" for m in agent["containers"][0]["volumeMounts"])
    assert all(e["name"] != "DEEPSEEK_API_KEY" for e in data["containers"][0]["env"])
