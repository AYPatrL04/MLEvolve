from pathlib import Path

from deployments.launch_a100_qwen_pair import IMAGE, manifest
from deployments.run_a100_qwen_task import configuration


def test_prepare_job_has_no_gpu():
    pod = manifest("prepare", "a" * 64)["spec"]["template"]["spec"]
    assert "affinity" not in pod
    assert all("nvidia.com/a100" not in resources for resources in pod["containers"][0]["resources"].values())


def test_petfinder_run_is_a100_only_and_pins_source():
    job = manifest("run", "b" * 64, "petfinder")
    assert job["spec"]["backoffLimit"] == 0
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["image"] == IMAGE
    assert container["env"][0]["value"] == "b" * 64
    assert container["resources"]["limits"]["nvidia.com/a100"] == "1"
    expressions = pod["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"][0]["matchExpressions"]
    assert expressions[0]["values"] == ["NVIDIA-A100-SXM4-80GB", "NVIDIA-A100-80GB-PCIe"]


def test_jobs_use_node_prepared_datasets_without_masking_them():
    for phase in ("prepare", "run"):
        pod = manifest(phase, "c" * 64, "petfinder")["spec"]["template"]["spec"]
        assert all(
            mount["mountPath"] != "/datasets"
            and not mount["mountPath"].startswith("/datasets/")
            for mount in pod["containers"][0]["volumeMounts"]
        )
        assert all(volume["name"] != "datasets" for volume in pod["volumes"])


def test_petfinder_config_uses_local_qwen_and_merged_knowledge(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    config = configuration(repo, tmp_path, "petfinder")
    assert config["exp_id"] == "petfinder-pawpularity-score"
    assert config["agent"]["seed"] == 42
    assert config["agent"]["code"]["provider"] == "vllm"
    assert config["agent"]["code"]["base_url"] == "http://127.0.0.1:8000/v1"
    assert config["agent"]["design_knowledge_version"] == "v2"
    assert config["hardware_knowledge"]["enabled"]
    assert not config["hardware_knowledge"]["settings"]["graph"]["enabled"]
    assert config["preflight"]["enabled"]
    assert not config["preflight"]["fail_open_on_internal_error"]
    assert config["scheduler"]["enabled"]
    assert config["scheduler"]["settings"]["prediction"]["mode"] == "branch_profile"
    assert config["scheduler"]["settings"]["gpu_scheduler"]["parallel_job_cap"] is None
    assert config["scheduler"]["settings"]["gpu_scheduler"]["memory"]["gpu_vram_gib"] == 32


def test_full_config_matches_merged_full_pipeline_contract(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    config = configuration(repo, tmp_path, "full")
    assert config["exp_id"] == "nlp-getting-started"
    assert config["agent"]["steps"] == 10
    assert config["agent"]["initial_drafts"] == 3
    assert config["agent"]["time_limit"] is None
    assert config["agent"]["precision_optimization_mode"] == "normal"
    assert config["agent"]["code"]["model"] == "qwen3.8-27b-int8-a100"
