from pathlib import Path

from deployments.launch_full_a10 import IMAGE, manifest
from deployments.run_full_a10 import configuration


def test_cpu_gate_does_not_reserve_gpu():
    pod = manifest("prepare", "a" * 64)["spec"]["template"]["spec"]
    assert "affinity" not in pod
    assert "nvidia.com/gpu" not in pod["containers"][0]["resources"]["limits"]


def test_run_is_a10_only_and_pins_source():
    job = manifest("run", "b" * 64)
    assert job["spec"]["backoffLimit"] == 0
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["image"] == IMAGE
    assert container["env"][0]["value"] == "b" * 64
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    expressions = pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
    assert expressions[0]["values"] == ["NVIDIA-A10"]


def test_full_configuration_keeps_all_components_and_selective_precision(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    cfg = configuration(repo, tmp_path)
    assert cfg["agent"]["seed"] == 42
    assert cfg["agent"]["steps"] == 10
    assert cfg["agent"]["initial_drafts"] == 3
    assert cfg["agent"]["time_limit"] is None
    assert cfg["agent"]["precision_optimization_mode"] == "normal"
    assert cfg["hardware_knowledge"]["enabled"]
    assert cfg["hardware_knowledge"]["include_profile_evidence"]
    assert cfg["agent"]["hardware_context_enabled"]
    assert cfg["preflight"]["enabled"]
    assert not cfg["preflight"]["fail_open_on_internal_error"]
    assert cfg["scheduler"]["enabled"]
    settings = cfg["scheduler"]["settings"]
    assert settings["prediction"]["mode"] == "ml_predictor"
    assert Path(settings["prediction"]["ml"]["registry_path"]).is_file()
    assert settings["gpu_scheduler"]["parallel_job_cap"] is None
    assert settings["gpu_scheduler"]["memory"]["gpu_vram_gib"] <= 31
