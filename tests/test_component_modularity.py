from __future__ import annotations

import itertools
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents import debug_agent, result_parse_agent
from agents.coder.stepwise_coder import create_default_step_agents
from agents.hardware_context import get_hardware_context_for_stage, get_hardware_design_brief
from agents.prompts.impl_guideline import get_impl_guideline_from_agent
from agents.stage_repair import _build_repair_prompt, is_hardware_aware
from config import PreflightConfig
from engine.agent_search import AgentSearch
from engine.executor import ExecutionResult, Interpreter
from engine.preflight import ModelPreflightGate, PreflightOutcome, candidate_code_hash, diagnostic_to_review_issue
from engine.search_node import SearchNode
from hardware_knowledge_graph.client import HardwareKnowledgeClient
from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerConfig
from localml_scheduler.hardware import HardwareProfile


def _agent(tmp_path, *, hardware=True, preflight=True, scheduler=True):
    settings = SchedulerConfig(
        runtime_root=tmp_path / "scheduler",
        gpu_scheduler={"memory": {"gpu_vram_gib": 31}},
        graph_db={"enabled": False}, hardware_knowledge_graph={"enabled": False},
        hardware_feature_db={"enabled": False}, redis_cache={"enabled": False},
    )
    acfg = SimpleNamespace(
        hardware_context_enabled=True, hardware_context_mode="compact",
        hardware_context_limit=4, hardware_context_max_prompt_chars=3500,
        precision_optimization_mode="normal", time_limit=600, steps=10,
        search=SimpleNamespace(parallel_search_num=1),
    )
    agent = AgentSearch.__new__(AgentSearch)
    agent.cfg = SimpleNamespace(
        workspace_dir=tmp_path, experiment=SimpleNamespace(mode="hardware_aware"),
        hardware_knowledge={"enabled": hardware, "settings": {
            "graph": {"enabled": False}, "redis_cache": {"enabled": False},
        }},
        preflight=PreflightConfig(enabled=preflight, target_profile="nvidia/a10_24gb"),
        scheduler=SimpleNamespace(enabled=scheduler), agent=acfg,
        exec=SimpleNamespace(timeout=10), start_cpu_id="0", cpu_number="1",
        exp_id="modularity-audit", exp_name="modularity-audit", pretrain_model_dir="",
    )
    agent.acfg = acfg
    agent.scheduler_client = SchedulerClient(settings) if scheduler else None
    if scheduler:
        agent.scheduler_client.store.backend._hardware_profile = HardwareProfile(
            "fixture", "linux", "NVIDIA A100 80GB PCIe", 81920, "8.0", "12.4", "fixture"
        )
    agent.hardware_knowledge_client = None
    agent.pipeline_logger = None
    agent.task_desc = "Image regression using RMSE."
    agent.data_preview = "image input, continuous target"
    agent.current_step = 0
    agent.start_time = time.time()
    return agent


@pytest.mark.parametrize("hardware,preflight,scheduler", itertools.product([False, True], repeat=3))
def test_components_operate_independently(tmp_path, monkeypatch, hardware, preflight, scheduler):
    agent = _agent(tmp_path, hardware=hardware, preflight=preflight, scheduler=scheduler)
    probe = Mock(return_value={"ok": True, "source": "fixture", "hardware_profile": {
        "hardware_key": "fixture", "os_name": "linux", "gpu_name": "NVIDIA A100 80GB PCIe",
        "total_vram_mb": 81920, "compute_capability": "8.0", "cuda_runtime": "12.4", "torch_version": "fixture",
    }})
    monkeypatch.setattr(HardwareKnowledgeClient, "probe_current_hardware", probe)
    agent.attach_hardware_knowledge()
    assert (agent.hardware_knowledge_client is not None) == hardware
    assert is_hardware_aware(agent) == hardware
    code = "MODEL_FAMILY = 'linear'\nBATCH_SIZE = 2\n"
    context = get_hardware_context_for_stage(agent, "datatype_precision", code=code)
    assert bool(context.prompt_section) == hardware
    brief = get_hardware_design_brief(agent)
    assert bool(brief.prompt_section) == hardware
    if hardware:
        assert brief.raw_context["confidence"] == 0.0
        assert "A100" in context.prompt_section
        assert context.raw_context["stage_hardware_features"]["found"]
        if not scheduler:
            assert not context.raw_context["hardware_context"].get("scheduler_limits")
            assert "scheduler owns" not in context.prompt_section
    else:
        probe.assert_not_called()
    guideline = "\n".join(get_impl_guideline_from_agent(agent)["Implementation guideline"])
    assert ("CandidateAdapter" in guideline) == preflight
    assert ("script_scheduler_context" in guideline) == scheduler
    stages = create_default_step_agents(hardware_aware=hardware, scheduler_enabled=scheduler)
    assert ("datatype_precision" in [stage.name for stage in stages]) == hardware
    training = "\n".join(next(stage for stage in stages if stage.name == "training_evaluation").guidelines)
    assert ("script_scheduler_context" in training) == scheduler
    node = SearchNode(code=code, plan="test", stage="draft")
    assert agent._run_node_preflight(node, generated=False)
    assert node.preflight_status == ("PASS" if preflight else "SKIPPED")
    interpreter = Interpreter(tmp_path, cfg=agent.cfg)
    interpreter.scheduler_client = agent.scheduler_client
    direct = Mock(return_value=ExecutionResult(["direct"], 0, None))
    scheduled = Mock(return_value=ExecutionResult(["scheduled"], 0, None))
    monkeypatch.setattr(interpreter, "_run_subprocess", direct)
    monkeypatch.setattr(interpreter, "_run_scheduler_job", scheduled)
    result = interpreter.run(code, node.id, node_context=node)
    assert result.term_out == (["scheduled"] if scheduler else ["direct"])
    assert direct.call_count == int(not scheduler)
    assert scheduled.call_count == int(scheduler)


@pytest.mark.parametrize("scheduler", [False, True])
def test_disabled_preflight_does_not_enforce_stale_admission(tmp_path, monkeypatch, scheduler):
    agent = _agent(tmp_path, hardware=False, preflight=False, scheduler=scheduler)
    node = SearchNode(code="print('valid')", plan="test", stage="draft")
    node.preflight_admitted = False
    node.preflight_code_hash = "stale"
    node.preflight_gpu_check_required = True
    node.review_status = "rejected"
    node.review_issues = [{"source": "model_preflight", "severity": "critical"}]
    interpreter = Interpreter(tmp_path, cfg=agent.cfg)
    interpreter.scheduler_client = agent.scheduler_client
    monkeypatch.setattr(interpreter, "_run_subprocess", lambda **kwargs: ExecutionResult(["ran"], 0, None))
    monkeypatch.setattr(interpreter, "_run_scheduler_job", lambda **kwargs: ExecutionResult(["ran"], 0, None))
    assert interpreter.run(node.code, node.id, node_context=node).term_out == ["ran"]
    assert agent._ensure_node_preflight_before_execution(node)
    assert node.preflight_status == "SKIPPED"
    assert node.preflight_code_hash is None
    assert not node.preflight_gpu_check_required
    assert not node.review_issues
    assert node.review_status != "rejected"


def test_preflight_rejection_also_blocks_direct_execution(tmp_path, monkeypatch):
    agent = _agent(tmp_path, hardware=False, scheduler=False)
    interpreter = Interpreter(tmp_path, cfg=agent.cfg)
    direct = Mock(side_effect=AssertionError("rejected code executed"))
    monkeypatch.setattr(interpreter, "_run_subprocess", direct)
    node = SearchNode(code="print('must not run')", plan="test", stage="draft")
    node.preflight_admitted = False
    assert interpreter.run(node.code, node.id, node_context=node).exc_type == "PreflightRejected"
    assert interpreter.run_many([{"node": node, "code": node.code, "id": node.id}])[node.id].exc_type == "PreflightRejected"
    direct.assert_not_called()


def test_preflight_repair_preserves_failure_inputs_and_location(tmp_path):
    agent = _agent(tmp_path, hardware=False, scheduler=False)
    diagnostic = {"classification": "confirmed_candidate_failure", "code": "CON001",
        "stage": "construction", "exception_type": "ValueError", "message": "bad shape",
        "scenario": {"batch_size": 2, "precision": "fp32"},
        "stack_trace": 'File "candidate.py", line 17, in build_model\nValueError: bad shape',
        "reproduction": "model-preflight check preflight.yaml --only construction"}
    issue = diagnostic_to_review_issue(diagnostic)
    node = SearchNode(code="model = None", plan="test", stage="draft")
    prompt = _build_repair_prompt(agent, node, node.code, issue.owner, [issue])
    assert "line 17" in prompt and "batch_size" in prompt and "fp32" in prompt
    assert diagnostic["reproduction"] in prompt
    assert "# Hardware" not in prompt


def test_fixture_failure_is_actionable_without_claiming_model_defect():
    from agents.stage_repair import group_repair_issues
    issue = diagnostic_to_review_issue({"classification": "inconclusive", "code": "FIX001",
        "stage": "data_contract", "exception_type": "KeyError", "message": "missing feature"})
    assert issue.severity == "warning"
    assert "does not establish a model defect" in issue.repair_instruction
    assert "model_design" in group_repair_issues(SimpleNamespace(), [issue])


def test_empty_profile_evidence_is_not_a_precision_diagnosis(tmp_path):
    agent = _agent(tmp_path)
    context = agent.scheduler_client.get_optimization_context(candidate={"model_key": "unknown", "stage": "datatype_precision", "uses_amp": False})
    assert "precision_not_optimized" not in context["derived_diagnosis"]["profile_symptoms"]
    assert "tensor_core_not_used" not in context["derived_diagnosis"]["profile_symptoms"]
    assert context["runtime_estimate"]["found"] is False
    assert context["effective_backend"] and context["runner_contract"]


@pytest.mark.parametrize("origin", ["scheduler", "executor"])
def test_scheduler_failure_never_requests_candidate_code_repair(tmp_path, monkeypatch, origin):
    agent = _agent(tmp_path, hardware=False)
    node = SearchNode(code="print('valid code')", plan="test", stage="draft")
    execution = ExecutionResult(["worker failed before launch"], 0, "RuntimeError",
        {"message": "worker failed before launch", "failure_origin": origin}, [])
    monkeypatch.setattr(result_parse_agent, "query", Mock(side_effect=AssertionError("LLM should not classify infrastructure")))
    result_parse_agent.run(agent, node, execution)
    assert {issue["severity"] for issue in node.review_issues} == {"warning"}
    assert not {"missing_metric", "missing_submission"} & {issue["category"] for issue in node.review_issues}
    agent.scfg = SimpleNamespace()
    monkeypatch.setattr(SearchNode, "add_expected_child_count", lambda *args, **kwargs: True)
    monkeypatch.setattr(debug_agent, "register_node", lambda *args, **kwargs: None)
    monkeypatch.setattr(debug_agent, "repair_selected_stages", Mock(side_effect=AssertionError("code must remain unchanged")))
    retry = debug_agent.run(agent, node)
    assert retry.code == node.code
    assert retry.diagnostics["execution_retry"] is True


def test_direct_training_does_not_require_scheduler_contracts():
    from agents.training_contract_validation import validate_training_contract
    code = "import torch\nBATCH_SIZE = 2\nEPOCHS = 3\nloss.backward()\noptimizer.step()\n"
    categories = {issue.category for issue in validate_training_contract(
        code, require_scheduler_hooks=True, scheduler_enabled=False,
    )}
    assert categories == {"validation_early_stopping", "training_runtime_diagnostics"}


@pytest.mark.parametrize("repaired", [False, True])
def test_inconclusive_fixture_gets_bounded_repair_and_recheck(tmp_path, monkeypatch, repaired):
    agent = _agent(tmp_path, hardware=False, scheduler=False)
    agent.refresh_hardware_context = lambda node: None
    agent.cfg.preflight.max_repair_rounds = 1
    issue = diagnostic_to_review_issue({"classification": "inconclusive", "code": "FIX001",
        "stage": "data_contract", "exception_type": "KeyError", "message": "missing feature"})
    gate_calls = []

    def check(_gate, node, *, generated, attempt):
        gate_calls.append((node.code, attempt))
        if repaired and attempt:
            return PreflightOutcome("PASS", "full_cpu", candidate_code_hash(node.code), True, False)
        return PreflightOutcome("INCONCLUSIVE", "full_cpu", candidate_code_hash(node.code), True, True,
            ["FIX001"], issues=[issue])

    repair = Mock(return_value=("fixed fixture" if repaired else "original", [], {}))
    monkeypatch.setattr(ModelPreflightGate, "run", check)
    monkeypatch.setattr("agents.stage_repair.repair_selected_stages", repair)
    node = SearchNode(code="original", plan="test", stage="draft")
    assert agent._run_node_preflight(node, generated=True)
    assert repair.call_count == 1
    assert len(gate_calls) == 2
    assert node.preflight_repair_count == 1
    assert node.preflight_status == ("PASS" if repaired else "INCONCLUSIVE")
    assert node.preflight_gpu_check_required == (not repaired)
    assert bool(node.review_issues) == (not repaired)


@pytest.mark.parametrize("batch", [False, True])
def test_missing_scheduler_result_keeps_origin_without_hwdb(tmp_path, monkeypatch, batch):
    from localml_scheduler.tests.test_executor_scheduler_bridge import _FakeSchedulerClient
    from localml_scheduler.domain import JobStatus
    agent = _agent(tmp_path, hardware=False, preflight=False)
    scheduler = _FakeSchedulerClient(agent.scheduler_client.settings)
    scheduler.plan_job_packet = Mock(side_effect=AssertionError("HWDB must not be queried"))
    original_submit = scheduler.submit

    def submit(job):
        original_submit(job)
        Path(job.config.runner_kwargs["result_path"]).unlink()
        job.mark_status(JobStatus.FAILED, reason="worker unavailable before candidate launch")
        return job

    monkeypatch.setattr(scheduler, "submit", submit)
    interpreter = Interpreter(tmp_path, cfg=agent.cfg)
    interpreter.attach_scheduler(scheduler, SimpleNamespace(wait_timeout_seconds=5, wait_poll_interval_seconds=0.01))
    node = SearchNode(code="print('valid code')", plan="test", stage="draft")
    if batch:
        result = interpreter.run_many([{"id": node.id, "code": node.code, "node": node}])[node.id]
    else:
        result = interpreter.run(node.code, node.id, node_context=node)
    assert result.exc_info["failure_origin"] == "scheduler"
    assert "worker unavailable" in result.exc_info["message"]
    scheduler.plan_job_packet.assert_not_called()


@pytest.mark.parametrize("failure", [False, True])
def test_direct_subprocess_preserves_candidate_result(tmp_path, failure):
    agent = _agent(tmp_path, hardware=False, preflight=False, scheduler=False)
    interpreter = Interpreter(tmp_path, cfg=agent.cfg, timeout=5)
    result = interpreter.run("raise ValueError('bad candidate')" if failure else "print('completed')", "cpu-smoke")
    assert result.exc_type == ("ValueError" if failure else None)
    assert "failure_origin" not in result.exc_info
    assert ("bad candidate" if failure else "completed") in "".join(result.term_out)


def test_direct_launch_error_is_execution_failure(tmp_path, monkeypatch):
    agent = _agent(tmp_path, hardware=False, preflight=False, scheduler=False)
    interpreter = Interpreter(tmp_path, cfg=agent.cfg, timeout=5)
    monkeypatch.setattr("engine.executor.subprocess.Popen", Mock(side_effect=OSError("launch unavailable")))
    result = interpreter.run("print('valid')", "launch-smoke")
    assert result.exc_info["failure_origin"] == "executor"
