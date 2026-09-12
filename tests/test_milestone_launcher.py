from deployments.launch_hwdb_milestone import manifest


def test_milestone_job_has_no_overall_deadline_or_private_data():
    job = manifest("run", "a" * 40, "test@sha256:abc")
    assert "activeDeadlineSeconds" not in job["spec"]
    assert job["spec"]["backoffLimit"] == 0
    pod = job["spec"]["template"]["spec"]
    assert not pod["automountServiceAccountToken"]
    container = pod["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert all(m["mountPath"] != "/heldout" for m in container["volumeMounts"])
    assert all(v["name"] != "kaggle" for v in pod["volumes"])
    assert container["env"][0]["value"] == "a" * 40


def test_cpu_gate_retains_deadline_and_requests_no_gpu():
    job = manifest("prepare", "a" * 40, "test@sha256:abc")
    assert job["spec"]["activeDeadlineSeconds"] == 10800
    container = job["spec"]["template"]["spec"]["containers"][0]
    assert "nvidia.com/gpu" not in container["resources"]["limits"]
