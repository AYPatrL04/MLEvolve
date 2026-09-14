from __future__ import annotations

from types import SimpleNamespace

from agents.prompts import impl_guideline
from config import PreflightConfig


def _agent(*, exec_timeout: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        acfg=SimpleNamespace(
            time_limit=60,
            steps=5,
        ),
        cfg=SimpleNamespace(
            exec=SimpleNamespace(timeout=exec_timeout),
            preflight=PreflightConfig(),
            pretrain_model_dir="",
        ),
        current_step=2,
        start_time=100.0,
    )


def test_impl_guideline_uses_remaining_time_when_exec_timeout_is_none(monkeypatch) -> None:
    monkeypatch.setattr(impl_guideline.time, "time", lambda: 110.0)

    guideline = impl_guideline.get_impl_guideline_from_agent(_agent(exec_timeout=None))

    text = "\n".join(guideline["Implementation guideline"])
    assert "Time left" in text
    assert "Steps left = 3" in text
    assert "Max execution time per run = 50 seconds" in text


def test_impl_guideline_caps_configured_exec_timeout_by_remaining_time(monkeypatch) -> None:
    monkeypatch.setattr(impl_guideline.time, "time", lambda: 110.0)

    guideline = impl_guideline.get_impl_guideline_from_agent(_agent(exec_timeout=5))

    text = "\n".join(guideline["Implementation guideline"])
    assert "Max execution time per run = 5 seconds" in text


def test_impl_guideline_requires_partial_preflight_context_support(monkeypatch) -> None:
    monkeypatch.setattr(impl_guideline.time, "time", lambda: 110.0)

    guideline = impl_guideline.get_impl_guideline_from_agent(_agent(exec_timeout=5))

    text = "\n".join(guideline["Implementation guideline"])
    assert "caller-supplied `context` as a partial mapping" in text
    assert "context mutations do not persist" in text
    assert "real criterion" in text
    assert "merge it over adapter defaults" in text


def test_impl_guideline_requires_adapter_build_model_to_return_module(monkeypatch) -> None:
    monkeypatch.setattr(impl_guideline.time, "time", lambda: 110.0)

    guideline = impl_guideline.get_impl_guideline_from_agent(_agent(exec_timeout=5))

    text = "\n".join(guideline["Implementation guideline"])
    assert "build_model(context) MUST return the real torch.nn.Module" in text


def test_petfinder_guideline_names_its_multimodal_preflight_batch(monkeypatch) -> None:
    monkeypatch.setattr(impl_guideline.time, "time", lambda: 110.0)
    agent = _agent(exec_timeout=5)
    agent.cfg.exp_id = "petfinder-pawpularity-score"

    guideline = impl_guideline.get_impl_guideline_from_agent(agent)

    text = "\n".join(guideline["Implementation guideline"])
    assert "image [B, 3, 256, 256]" in text
    assert "tabular [B, 12]" in text


def test_precision_quality_and_rmse_rules_do_not_depend_on_coldstart_or_hwdb():
    agent = _agent(exec_timeout=5)
    agent.use_coldstart = False
    agent.hardware_knowledge_client = None
    agent.acfg.hardware_context_enabled = False
    guideline = impl_guideline.get_impl_guideline_from_agent(agent)
    safety = " ".join(guideline["Mixed precision safety"])
    assert "model.half()" in safety and "model parameters and optimizer state in FP32" in safety
    assert "select_validated_precision" in safety and "do not invent measurements" in safety
    implementation = " ".join(guideline["Implementation guideline"])
    assert "np.sqrt(mean_squared_error(y_true, y_pred))" in implementation
    assert "restore it before final validation" in implementation
    agent.acfg.precision_optimization_mode = "conservative"
    guideline = impl_guideline.get_impl_guideline_from_agent(agent)
    assert "Conservative precision" in guideline and "Mixed precision safety" not in guideline
