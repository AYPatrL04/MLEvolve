import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from agents import code_review_agent, draft_agent
from agents.review_contracts import ReviewDecision, ReviewIssue, StageRepairResult
from config import PreflightConfig
from engine.agent_search import AgentSearch
from engine.preflight import ModelPreflightGate, PreflightOutcome, candidate_code_hash
from engine.search_node import Journal, SearchNode


def search_fixture(tmp_path, monkeypatch, *, enabled=True):
    agent = AgentSearch.__new__(AgentSearch)
    agent.cfg = SimpleNamespace(
        workspace_dir=tmp_path, pretrain_model_dir="",
        experiment=SimpleNamespace(mode="hardware_aware"),
        preflight=PreflightConfig(enabled=enabled, max_repair_rounds=1),
    )
    agent.acfg = SimpleNamespace(
        code=SimpleNamespace(model="fake", temp=0), hardware_context_enabled=False,
        precision_optimization_mode="normal",
        review=SimpleNamespace(enabled=True, max_repair_rounds=1, repair_retries=1),
    )
    agent.scfg = SimpleNamespace(num_drafts=10)
    agent.task_desc = "Train a regression model."
    agent.scheduler_client = None
    agent.pipeline_logger = None
    agent.journal = Journal()
    agent.virtual_root = SearchNode(code="", plan="root", stage="root")
    agent.refresh_hardware_context = lambda node: None
    agent._review_training_parameters_before_submission = lambda node: None
    agent._validate_node_precision_before_execution = lambda node: True
    agent._validate_node_dependencies_before_execution = lambda node: True
    node = SearchNode(code="candidate = 1\n", plan="test", stage="draft", parent=agent.virtual_root)
    monkeypatch.setattr(draft_agent, "run", lambda *args, **kwargs: node)
    return agent, node


def outcome(node, *, admitted=True, status="PASS"):
    return PreflightOutcome(
        status=status, mode="balanced", code_hash=candidate_code_hash(node.code),
        admitted=admitted, gpu_check_required=status == "INCONCLUSIVE", diagnostic_codes=[],
    )


def deferred_step(agent):
    return agent._run_single_step(
        agent.virtual_root, lambda *args, **kwargs: pytest.fail("Unexpected execution"),
        execute_immediately=False,
    )[1]


@pytest.mark.parametrize("status,admitted", [("PASS", True), ("INCONCLUSIVE", True), ("FAIL", False), ("INCONCLUSIVE", False)])
def test_preflight_admission_controls_review_and_unchanged_code_reuses_check(tmp_path, monkeypatch, status, admitted):
    agent, node = search_fixture(tmp_path, monkeypatch)
    events = []

    def gate(_gate, candidate, **kwargs):
        events.append("preflight")
        return outcome(candidate, status=status, admitted=admitted)

    def classify(_agent, candidate, code):
        events.append("review")
        assert candidate.preflight_admitted is True
        assert candidate.preflight_code_hash == candidate_code_hash(code)
        return ReviewDecision(approved=True, reasoning="Approved", issues=()), {}

    monkeypatch.setattr(ModelPreflightGate, "run", gate)
    monkeypatch.setattr(code_review_agent, "classify_code", classify)
    assert deferred_step(agent) is node
    assert events == (["preflight", "review"] if admitted else ["preflight"])
    assert (node.review_status == "rejected") is not admitted
    if admitted:
        assert [entry["event"] for entry in node.review_history] == ["model_preflight_completed", "review_decision"]
        assert agent._ensure_node_preflight_before_execution(node)
        assert events == ["preflight", "review"]


@pytest.mark.parametrize("rechecked_admitted", [True, False])
def test_review_edits_are_rechecked_before_deferral(tmp_path, monkeypatch, rechecked_admitted):
    agent, node = search_fixture(tmp_path, monkeypatch)
    events = []
    issue = ReviewIssue(source="static_review", severity="critical", category="task_contract",
                        owner="training_evaluation", evidence="Wrong target", repair_instruction="Correct the target")
    decisions = iter([
        ReviewDecision(approved=False, reasoning="Repair", issues=(issue,)),
        ReviewDecision(approved=True, reasoning="Fixed", issues=()),
    ])

    def gate(_gate, candidate, **kwargs):
        events.append(("preflight", candidate.code))
        allowed = candidate.code == "candidate = 1\n" or rechecked_admitted
        return outcome(candidate, admitted=allowed, status="PASS" if allowed else "FAIL")

    def classify(_agent, candidate, code):
        events.append(("review", code))
        return next(decisions), {}

    monkeypatch.setattr(ModelPreflightGate, "run", gate)
    monkeypatch.setattr(code_review_agent, "classify_code", classify)
    monkeypatch.setattr(code_review_agent, "repair_selected_stages", lambda *args, **kwargs: (
        "candidate = 2\n", [StageRepairResult(stage="training_evaluation", applied=True)], {},
    ))
    assert deferred_step(agent) is node
    assert events == [
        ("preflight", "candidate = 1\n"), ("review", "candidate = 1\n"),
        ("review", "candidate = 2\n"), ("preflight", "candidate = 2\n"),
    ]
    assert node.preflight_code_hash == candidate_code_hash(node.code)
    assert (node.review_status == "rejected") is not rechecked_admitted


def test_preflight_targeted_repair_precedes_review(tmp_path, monkeypatch):
    agent, node = search_fixture(tmp_path, monkeypatch)
    events = []
    issue = ReviewIssue(source="model_preflight", severity="critical", category="preflight_aut002",
                        owner="training_evaluation", evidence="Broken loss", repair_instruction="Repair the real loss")

    def gate(_gate, candidate, **kwargs):
        events.append("preflight")
        result = outcome(candidate, admitted=candidate.code == "fixed = 1\n", status="PASS" if candidate.code == "fixed = 1\n" else "FAIL")
        result.issues = [] if result.admitted else [issue]
        return result

    def repair(*args, **kwargs):
        events.append("repair")
        return "fixed = 1\n", [StageRepairResult(stage="training_evaluation", applied=True)], {}

    def classify(_agent, candidate, code):
        events.append("review")
        assert code == "fixed = 1\n"
        assert candidate.preflight_admitted
        return ReviewDecision(approved=True, reasoning="Fixed", issues=()), {}

    monkeypatch.setattr(ModelPreflightGate, "run", gate)
    monkeypatch.setattr("agents.stage_repair.repair_selected_stages", repair)
    monkeypatch.setattr(code_review_agent, "classify_code", classify)
    deferred_step(agent)
    assert events == ["preflight", "repair", "preflight", "review"]
    assert node.preflight_repair_count == 1
    assert node.review_history[0]["event"] == "model_preflight_completed"
    assert node.review_history[1]["event"] == "model_preflight_rechecked"


def test_disabled_preflight_keeps_review(tmp_path, monkeypatch):
    agent, node = search_fixture(tmp_path, monkeypatch, enabled=False)
    calls = []
    monkeypatch.setattr(ModelPreflightGate, "run", lambda *args, **kwargs: pytest.fail("Disabled preflight ran"))
    monkeypatch.setattr(code_review_agent, "classify_code", lambda *args: (
        calls.append("review") or ReviewDecision(approved=True, reasoning="Approved", issues=()), {},
    ))
    deferred_step(agent)
    assert calls == ["review"]
    assert node.preflight_status == "SKIPPED"
    assert node.review_status == "approved"


@pytest.mark.parametrize("reuse", ["supplied", "execution_retry"])
def test_reused_code_keeps_review_bypass_but_runs_preflight(tmp_path, monkeypatch, reuse):
    agent, node = search_fixture(tmp_path, monkeypatch)
    calls = []
    if reuse == "execution_retry":
        node.diagnostics["execution_retry"] = True

    def gate(_gate, candidate, **kwargs):
        calls.append(kwargs["generated"])
        return outcome(candidate)

    monkeypatch.setattr(ModelPreflightGate, "run", gate)
    monkeypatch.setattr(code_review_agent, "classify_code", lambda *args: pytest.fail("Reused code was reviewed"))
    agent._run_single_step(agent.virtual_root, lambda *args, **kwargs: pytest.fail("Unexpected execution"),
                           execute_immediately=False, init_solution_path="supplied.py" if reuse == "supplied" else None)
    assert calls == [reuse != "supplied"]
    assert node.preflight_admitted is True


def test_disabled_review_keeps_preflight_and_deterministic_guards(tmp_path, monkeypatch):
    agent, node = search_fixture(tmp_path, monkeypatch)
    agent.acfg.review.enabled = False
    calls = []

    def gate(_gate, candidate, **kwargs):
        calls.append("preflight")
        return outcome(candidate)

    monkeypatch.setattr(ModelPreflightGate, "run", gate)
    monkeypatch.setattr(code_review_agent, "classify_code", lambda *args: pytest.fail("Disabled review ran"))
    monkeypatch.setattr(code_review_agent, "validate_training_precision", lambda *args, **kwargs: calls.append("precision") or [])
    monkeypatch.setattr(code_review_agent, "validate_training_contract", lambda *args, **kwargs: calls.append("training_contract") or [])
    monkeypatch.setattr(code_review_agent, "validate_runtime_dependencies", lambda *args, **kwargs: calls.append("dependencies") or [])
    deferred_step(agent)
    assert calls == ["preflight", "precision", "training_contract", "dependencies"]
    assert node.review_history[0]["event"] == "model_preflight_completed"
    assert node.review_history[-1]["event"] == "review_disabled"


def test_submission_parameter_edit_is_rechecked_before_immediate_execution(tmp_path, monkeypatch):
    from agents import result_parse_agent
    from engine import evaluation, execution
    from engine.executor import ExecutionResult
    from utils.metric import MetricValue

    agent, node = search_fixture(tmp_path, monkeypatch)
    agent.journal_lock = threading.Lock()
    agent.best_node = None
    events = []

    def gate(_gate, candidate, **kwargs):
        events.append(("preflight", candidate.code))
        return outcome(candidate)

    def classify(*args):
        events.append(("review", node.code))
        return ReviewDecision(approved=True, reasoning="Approved", issues=()), {}

    def parse(_agent, *, node, exec_result):
        node.metric = MetricValue(0.5, maximize=False)
        node.is_buggy = False
        return node

    def execute(code, *args, node_context, **kwargs):
        assert node_context.preflight_code_hash == candidate_code_hash(code)
        events.append(("execute", code))
        return ExecutionResult(["Final Validation Score: 0.5"], 0.01, None, {})

    monkeypatch.setattr(ModelPreflightGate, "run", gate)
    monkeypatch.setattr(code_review_agent, "classify_code", classify)
    monkeypatch.setattr(result_parse_agent, "run", parse)
    monkeypatch.setattr(execution, "validate_executed_node", lambda *args: None)
    monkeypatch.setattr(evaluation, "check_improvement", lambda *args: False)
    agent._review_training_parameters_before_submission = lambda candidate: setattr(candidate, "code", "candidate = 2\n")
    agent._run_single_step(agent.virtual_root, execute)
    assert events == [("preflight", "candidate = 1\n"), ("review", "candidate = 1\n"),
                      ("preflight", "candidate = 2\n"), ("execute", "candidate = 2\n")]


@pytest.mark.parametrize("independent_rejection", [False, True])
def test_fresh_admission_clears_only_previous_preflight_rejection(tmp_path, monkeypatch, independent_rejection):
    agent, node = search_fixture(tmp_path, monkeypatch)
    node.review_status = "rejected"
    node.preflight_admitted = False
    node.preflight_code_hash = candidate_code_hash("previous code")
    node.review_issues = [{"source": "model_preflight", "severity": "critical"}]
    if independent_rejection:
        node.review_issues.append({"source": "static_review", "severity": "critical"})
    monkeypatch.setattr(ModelPreflightGate, "run", lambda _gate, candidate, **kwargs: outcome(candidate))
    assert agent._ensure_node_preflight_before_execution(node)
    assert (node.review_status == "rejected") is independent_rejection


@pytest.mark.parametrize("changed,enabled", [(False, True), (True, True), (False, False)])
def test_review_receives_only_current_enabled_preflight_evidence(tmp_path, monkeypatch, changed, enabled):
    agent, node = search_fixture(tmp_path, monkeypatch, enabled=enabled)
    node.preflight_code_hash = candidate_code_hash(node.code)
    node.preflight_status = "INCONCLUSIVE"
    node.preflight_admitted = True
    node.preflight_gpu_check_required = True
    node.preflight_report_path = "report_attempt_0.json"
    node.diagnostics["preflight_advisories"] = [{"message": "CUDA branch unverified"}]
    empty = SimpleNamespace(prompt_section="")
    for name in ("get_hardware_context_for_stage", "get_cuda_docs_context", "get_lesson_context_for_stage"):
        monkeypatch.setattr(code_review_agent, name, lambda *args, **kwargs: empty)
    monkeypatch.setattr(code_review_agent, "format_cuda_docs_prompt_section", lambda *args, **kwargs: "")
    prompt, _ = code_review_agent._build_review_prompt(agent, node, "changed = 1\n" if changed else node.code)
    if not enabled:
        assert "CPU preflight evidence" not in prompt
    elif changed:
        assert prompt["CPU preflight evidence"]["status"] == "STALE"
        assert "evidence_ref" not in prompt["CPU preflight evidence"]
    else:
        evidence = prompt["CPU preflight evidence"]
        assert evidence["admitted"] is True
        assert evidence["gpu_check_required"] is True
        assert evidence["advisories"] == [{"message": "CUDA branch unverified"}]
        assert evidence["evidence_ref"] == "report_attempt_0.json"


def test_rechecks_preserve_report_references_for_previous_review(tmp_path, monkeypatch):
    import model_preflight
    from model_preflight.reporting import json_report

    agent, node = search_fixture(tmp_path, monkeypatch)
    payload = {"overall_status": "PASS", "gpu_check_required": False, "diagnostics": []}
    monkeypatch.setattr(model_preflight, "check", lambda *args, **kwargs: SimpleNamespace(to_dict=lambda: dict(payload)))
    monkeypatch.setattr(json_report, "write_json", lambda report, path: path.write_text(json.dumps(report.to_dict())))
    gate = ModelPreflightGate(agent.cfg)
    first = gate.run(node, generated=False, attempt=0)
    original = Path(first.report_path).read_text()
    node.code = "candidate = 2\n"
    payload["overall_status"] = "INCONCLUSIVE"
    second = gate.run(node, generated=False, attempt=0)
    assert first.report_path != second.report_path
    assert Path(first.report_path).read_text() == original
    assert json.loads(Path(second.report_path).read_text())["overall_status"] == "INCONCLUSIVE"
    monkeypatch.setattr(model_preflight, "check", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("checker unavailable")))
    node.code = "candidate = 3\n"
    failed = gate.run(node, generated=False, attempt=0)
    assert failed.status == "INTERNAL_ERROR"
    assert failed.admitted
    assert failed.report_path is None
    assert Path(first.report_path).read_text() == original
