from deployments.launch_hwdb_current_ablation import PREFIX, ROOT, manifest


def test_current_ablation_uses_new_root_and_requires_a10():
    job = manifest("run", "a" * 40)
    assert job["metadata"]["name"] == PREFIX + "-run"
    pod = job["spec"]["template"]["spec"]
    assert any(
        item.get("name") == "ABLATION_ROOT" and item.get("value") == ROOT
        for item in pod["containers"][0]["env"]
    )
    values = pod["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"][0]["matchExpressions"][0]["values"]
    assert values == ["NVIDIA-A10"]


def test_current_ablation_keeps_secret_and_gpu_contract():
    prepare = manifest("prepare", "b" * 40)
    prepare_pod = prepare["spec"]["template"]["spec"]
    assert "nvidia.com/gpu" not in prepare_pod["containers"][0]["resources"]["requests"]
    run = manifest("run", "b" * 40)
    run_pod = run["spec"]["template"]["spec"]
    assert run_pod["containers"][0]["resources"]["requests"]["nvidia.com/gpu"] == "1"
    assert any(
        item.get("name") == "DEEPSEEK_API_KEY"
        for item in run_pod["containers"][0]["env"]
    )
