import copy
from types import SimpleNamespace

import pytest
import torch

from deployments import compare_disaster_precision as comparison
from deployments.launch_disaster_comparison import IMAGE, manifest
from utils.training_diagnostics import TrainingDiagnostics


def rows(delta=-0.01):
    return [dict(seed=seed, mode=mode, status="complete", gpu="NVIDIA A10", source_sha256=comparison.SOURCE_HASH,
                 initial_weights_sha256=str(seed), split_sha256="fixed", best_validation_f1=0.75 + (delta if mode == "normal" else 0),
                 fixed_work_benchmark={"seconds": 1 if mode == "normal" else 2}, runtime={"skipped_updates": 0})
            for seed in comparison.SEEDS for mode in comparison.MODES]


def test_quality_first_keeps_slower_fp32():
    result = comparison.summarize(rows())
    assert result["status"] == "complete"
    assert result["median_fixed_work_speedup"] == 2
    assert result["provisional_quality_first_choice"] == "fp32"


def test_normal_can_win_quality_comparison():
    assert comparison.summarize(rows(0.01))["provisional_quality_first_choice"] == "normal_fp16"


def test_incomplete_or_unstable_results_do_not_recommend():
    assert "provisional_quality_first_choice" not in comparison.summarize(rows()[:2])
    values = rows()
    values[1]["runtime"]["skipped_updates"] = 1
    result = comparison.summarize(values)
    assert result["stability_review_required"]
    assert "provisional_quality_first_choice" not in result


@pytest.mark.parametrize("key", ["initial_weights_sha256", "split_sha256", "source_sha256", "gpu"])
def test_rejects_unmatched_pair(key):
    values = rows()
    values[1][key] = "mismatch"
    with pytest.raises(RuntimeError):
        comparison.summarize(values)


def test_checksum_checked_before_import(tmp_path):
    source = tmp_path / "candidate.py"
    source.write_text("raise AssertionError('must not execute')")
    with pytest.raises(ValueError, match="checksum"):
        comparison.load_candidate(source)


def test_cpu_update_is_real_fp32():
    model = torch.nn.Linear(3, 1)
    original = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    with TrainingDiagnostics(model, optimizer, scaler=scaler) as diagnostics:
        assert comparison.update(model, optimizer, scaler, diagnostics, torch.ones(2, 3), torch.ones(2, 1),
                                 "conservative", torch.device("cpu"), 1.0)
        assert diagnostics.completed_updates == 1
        assert diagnostics.skipped_updates == 0
        assert diagnostics.report()["parameter_dtypes"] == ["torch.float32"]
    assert not torch.equal(original["weight"], model.weight)


def test_nonfinite_loss_rejected():
    with pytest.raises(FloatingPointError):
        comparison.forward_loss(torch.nn.Linear(2, 1), torch.full((2, 2), float("nan")),
                                torch.ones(2, 1), "conservative", torch.device("cpu"))


def test_seed_reproduces_initial_weights():
    candidate = SimpleNamespace(set_seed=torch.manual_seed, SmallTransformerClassifier=lambda vocab_size: torch.nn.Linear(vocab_size, 1),
                                LR=0.001, WEIGHT_DECAY=0.01, BETAS=(0.9, 0.999), EPS=1e-8)
    a = comparison.make_training_objects(candidate, 3, 42, "conservative", torch.device("cpu"))
    b = comparison.make_training_objects(candidate, 3, 42, "normal", torch.device("cpu"))
    assert a[3] == b[3]


@pytest.mark.parametrize("phase", ["prepare", "run"])
def test_launcher_is_credential_free_and_gpu_gated(phase):
    job = manifest(phase, "a" * 40)
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["image"] == IMAGE
    assert container["env"] == [{"name": "SOURCE_COMMIT", "value": "a" * 40}]
    assert all("secret" not in v for v in pod["volumes"])
    assert all("heldout" not in str(m) and not m["mountPath"].startswith("/datasets/") for m in container["volumeMounts"])
    assert ("nvidia.com/gpu" in container["resources"]["requests"]) == (phase == "run")
    assert ("activeDeadlineSeconds" in job["spec"]) == (phase == "prepare")
