from copy import deepcopy

import pytest

from utils.precision_quality import COMPARISON_FIELDS, select_validated_precision


def comparison():
    protocol = {key: f"same-{key}" for key in COMPARISON_FIELDS}
    reference = {"protocol": protocol, "precision": "fp32", "metric": 20.0,
                 "epoch_seconds": 10.0, "evidence_refs": ["test-measurement:reference"]}
    candidate = {"protocol": dict(protocol), "precision": "fp16_amp", "metric": 20.0,
                 "epoch_seconds": 8.0, "evidence_refs": ["test-measurement:candidate"]}
    return protocol, {"reference": reference, "candidate": candidate, "direction": "minimize", "tolerance": 0.0}


def test_missing_quality_evidence_uses_fp32_without_inventing_tolerance():
    assert select_validated_precision("fp16_amp")["precision"] == "fp32"
    assert select_validated_precision("bf16_amp")["status"] == "unverified"
    protocol, evidence = comparison()
    evidence.pop("tolerance")
    assert select_validated_precision("fp16_amp", protocol=protocol, comparison=evidence)["status"] == "unverified"


@pytest.mark.parametrize("direction,before,after,accepted", [
    ("minimize", 20.0, 20.0, True), ("minimize", 20.0, 19.0, True),
    ("minimize", 20.0, 20.01, False), ("maximize", 0.9, 0.91, True),
    ("maximize", 0.9, 0.89, False),
])
def test_quality_is_required_before_speed(direction, before, after, accepted):
    protocol, evidence = comparison()
    evidence.update(direction=direction)
    evidence["reference"]["metric"] = before
    evidence["candidate"]["metric"] = after
    original = deepcopy(evidence)
    decision = select_validated_precision("fp16_amp", protocol=protocol, comparison=evidence)
    assert (decision["status"] == "accepted") == accepted
    assert decision["precision"] == ("fp16_amp" if accepted else "fp32")
    assert evidence == original


@pytest.mark.parametrize("field", COMPARISON_FIELDS)
def test_unmatched_quality_reference_is_not_transferable(field):
    protocol, evidence = comparison()
    evidence["candidate"]["protocol"][field] = "different"
    assert select_validated_precision("fp16_amp", protocol=protocol, comparison=evidence)["status"] == "unverified"


@pytest.mark.parametrize("change", [
    {"metric": float("nan")}, {"metric": True}, {"epoch_seconds": 0},
    {"epoch_seconds": 10}, {"epoch_seconds": float("inf")},
    {"evidence_refs": []}, {"precision": "bf16_amp"},
])
def test_invalid_or_non_improving_measurements_do_not_enable_amp(change):
    protocol, evidence = comparison()
    evidence["candidate"].update(change)
    assert select_validated_precision("fp16_amp", protocol=protocol, comparison=evidence)["precision"] == "fp32"


def test_explicit_quality_tolerance_is_respected():
    protocol, evidence = comparison()
    evidence["candidate"]["metric"] = 20.05
    evidence["tolerance"] = .1
    assert select_validated_precision("fp16_amp", protocol=protocol, comparison=evidence)["status"] == "accepted"
    evidence["tolerance"] = -1
    assert select_validated_precision("fp16_amp", protocol=protocol, comparison=evidence)["precision"] == "fp32"
    evidence["direction"] = {}
    assert select_validated_precision("fp16_amp", protocol=protocol, comparison=evidence)["precision"] == "fp32"


def test_runtime_diagnostics_reject_ignored_fp32_selection(capsys):
    import torch
    from utils.training_diagnostics import TrainingDiagnostics

    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    decision = select_validated_precision("bf16_amp")
    before = model.weight.detach().clone()
    with pytest.raises(RuntimeError, match="quality comparison"):
        with TrainingDiagnostics(model, optimizer, settings={"precision_quality": decision}):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                model(torch.ones(2, 4))
    assert torch.equal(before, model.weight)
    assert not model._forward_pre_hooks


def test_runtime_diagnostics_accept_validated_amp_and_observe_fp32_fallback(capsys):
    import torch
    from utils.training_diagnostics import TrainingDiagnostics

    protocol, evidence = comparison()
    evidence["candidate"]["precision"] = "bf16_amp"
    decision = select_validated_precision("bf16_amp", protocol=protocol, comparison=evidence)
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    with TrainingDiagnostics(model, optimizer, settings={"precision_quality": decision}) as diagnostics:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model(torch.ones(2, 4)).float().square().mean()
        loss.backward()
        optimizer.step()
        diagnostics.after_update()
    assert diagnostics.completed_updates == 1
    with TrainingDiagnostics(model, optimizer, settings={"precision_quality": select_validated_precision("bf16_amp")}):
        assert model(torch.ones(2, 4)).dtype == torch.float32


def test_runtime_fp16_cannot_omit_loss_scaling(capsys):
    import torch
    from utils.training_diagnostics import TrainingDiagnostics

    protocol, evidence = comparison()
    decision = select_validated_precision("fp16_amp", protocol=protocol, comparison=evidence)
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    with pytest.raises(RuntimeError, match="enabled GradScaler"):
        with TrainingDiagnostics(model, optimizer, settings={"precision_quality": decision}):
            with torch.autocast("cpu", dtype=torch.float16):
                model(torch.ones(2, 4))
