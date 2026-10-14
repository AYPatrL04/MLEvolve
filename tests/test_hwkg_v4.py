import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import pytest

from hwkg.injection import hardware_guidance
from hwkg.routing import hardware_context, with_hardware_prompt

ROOT = Path(__file__).resolve().parents[1]
BASE = "9c5c8a3b23f0361708b59a401452dddc00f97189"
GPU = NS(name="NVIDIA A10", major=8, minor=6, total_memory=23 * 1024**3)
CONFIG = NS(agent=NS(code=NS(model="deepseek-flash")))


@pytest.fixture
def enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("MLEVOLVE_HWKG_GRAPH_PATH", str(ROOT / "hwkg/graph.json"))
    monkeypatch.setenv("MLEVOLVE_HWKG_GRAPH_SHA256", "ef5850d7ab5af271413141ade5777878fbf8dcee0750e7eba406c924e4c6bec1")
    monkeypatch.setenv("MLEVOLVE_HWKG_HARDWARE_ID", "nvidia.ampere.a10.spec")
    monkeypatch.setenv("MLEVOLVE_HWKG_EVIDENCE_POLICY", "v4")
    monkeypatch.setenv("MLEVOLVE_HWKG_AUDIT_DIR", str(tmp_path))
    monkeypatch.delenv("MLEVOLVE_HWKG_EXPLORATORY", raising=False)
    with patch("hwkg.injection._visible_cuda_device", return_value=GPU):
        yield tmp_path


def test_v4_preserves_evaluation_and_budget(enabled):
    text = hardware_guidance("draft", "Disaster Tweets text classification")
    audit = json.loads((enabled / "prompt-audit.jsonl").read_text())
    assert len(text) <= 1600
    assert audit["advice_ids"] == ["precision", "padding"]
    assert "15%" not in text and "0.5" not in text and "160" not in text
    assert "threshold" in text and "Keep the prescribed" in text
    assert "post-step parameters" in text and "BOTH" in text and "IN that window" in text
    assert audit["event"] == "context_built"
    assert audit["verified_hardware_feature_ids"] == ["feat:amp", "feat:bf16"]


@pytest.mark.parametrize("code", ["model = LogisticRegression()", "# torch.Module autocast()"])
def test_cpu_and_comments_do_not_trigger_cuda_optimization(enabled, code):
    text = hardware_guidance("improve", "text classification", code)
    assert "[precision]" not in text and "[padding]" not in text


def test_debug_only_gets_safety_not_performance_changes(enabled):
    text = hardware_guidance("debug", "text classification", "import torch\nmodel = torch.nn.Linear(2,2)")
    assert "[precision]" in text and "failure only" in text
    assert "[padding]" not in text and "[transfer]" not in text


def test_budget_keeps_precision_even_with_padding_and_transfer(enabled):
    text = hardware_guidance("evolution", "text classification", "import torch\nmodel = AutoModel.from_pretrained('deberta')")
    assert len(text) <= 1600
    assert "[precision]" in text and "post-step parameters" in text


def test_bf16_requires_verified_graph_edge(enabled):
    from hwkg.policy import guidance_v4
    text, audit = guidance_v4("draft", "images", "", GPU, {})
    assert "No verified BF16" in text
    assert "is_bf16_supported" not in text
    assert audit["verified_hardware_feature_ids"] == []


def test_unknown_policy_and_wrong_device_fail_closed(enabled, monkeypatch):
    monkeypatch.setenv("MLEVOLVE_HWKG_EVIDENCE_POLICY", "typo")
    with pytest.raises(ValueError, match="Unknown"):
        hardware_guidance("draft")
    monkeypatch.setenv("MLEVOLVE_HWKG_EVIDENCE_POLICY", "v4")
    monkeypatch.setenv("MLEVOLVE_HWKG_HARDWARE_ID", "nvidia.blackwell.geforce_rtx_5080.spec")
    with pytest.raises(RuntimeError, match="does not uniquely match"):
        hardware_guidance("draft")


def test_dispatch_keeps_all_calls_without_mutating_and_records_failure(enabled):
    seen = []
    @with_hardware_prompt
    def send(prompt, cfg):
        seen.append(prompt)
        if len(seen) == 3:
            raise RuntimeError("provider unavailable")
        return "response"
    @hardware_context("improve")
    def run(agent, parent_node):
        prompt = {"system": "system", "user": "planner", "assistant": "source"}
        send(prompt, CONFIG)
        send("diff code", CONFIG)
        assert prompt["user"] == "planner"
        send("retry", CONFIG)
    with pytest.raises(RuntimeError):
        run(NS(task_desc="text classification"), NS(code="import torch", id="p1"))
    assert all("HWKG v4" in str(p) for p in seen)
    send("outside scope", CONFIG)
    assert seen[-1] == "outside scope"
    audit = [json.loads(x) for x in (enabled / "dispatch-audit.jsonl").read_text().splitlines()]
    assert [r["event"] for r in audit] == ["provider_dispatch", "provider_return"] * 2 + ["provider_dispatch", "provider_failure"]
    assert len({r["trace_id"] for r in audit}) == 1
    assert all(r["parent_id"] == "p1" for r in audit)


def test_disabled_is_exact_noop(monkeypatch):
    monkeypatch.delenv("MLEVOLVE_HWKG_GRAPH_PATH", raising=False)
    prompt = {"user": "original"}
    @with_hardware_prompt
    def send(p, cfg):
        assert p is prompt
        return p
    @hardware_context("draft")
    def run(agent):
        return send(prompt, CONFIG)
    with patch("hwkg.routing.hardware_guidance", side_effect=AssertionError("must not probe")):
        assert run(object()) is prompt


def test_initial_solution_without_generation_does_not_probe(enabled):
    @hardware_context("draft")
    def run(agent):
        return "initial solution"
    with patch("hwkg.routing.hardware_guidance", side_effect=AssertionError("must not probe")):
        assert run(NS(task_desc="text")) == "initial solution"


def test_concurrent_and_nested_scopes_do_not_leak(enabled):
    @with_hardware_prompt
    def send(prompt, cfg):
        return prompt
    @hardware_context("debug")
    def inner(agent, parent_node):
        return send("debug", CONFIG)
    @hardware_context("improve")
    def outer(agent, parent_node):
        inner(agent, parent_node)
        return send("improve", CONFIG)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(lambda i: outer(NS(task_desc="images"), NS(code="import torch", id=str(i))), range(2)))
    assert all("HWKG v4" in text for text in outputs)
    assert send("outside", CONFIG) == "outside"
    rows = [json.loads(x) for x in (enabled / "dispatch-audit.jsonl").read_text().splitlines()]
    assert {r["stage"] for r in rows} == {"debug", "improve"}
    assert {r["parent_id"] for r in rows} == {"0", "1"}


@pytest.mark.parametrize("path", [f"agents/{s}_agent.py" for s in ("draft", "improve", "evolution", "debug", "fusion", "aggregation")] + ["llm/__init__.py"])
def test_only_hwkg_hooks_differ_from_pinned_upstream(path):
    """Catch accidental edits to search/prompt/metric/backend behavior."""
    original = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT, text=True)
    tree = ast.parse((ROOT / path).read_text())
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and (n.module or "").startswith("hwkg"))]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = [d for d in node.decorator_list if not (
                isinstance(d, ast.Name) and d.id == "with_hardware_prompt" or
                isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "hardware_context")]
    assert ast.dump(tree, include_attributes=False) == ast.dump(ast.parse(original), include_attributes=False)


@pytest.mark.parametrize("stage,mode", [("draft", "full"), ("draft", "stepwise"), ("improve", "full"),
    ("improve", "diff"), ("improve", "diff_fallback"), ("improve", "diff_retry"),
    ("evolution", "full"), ("evolution", "diff"), ("debug", "full"), ("debug", "diff"), ("fusion", "diff"), ("aggregation", "full")])
@pytest.mark.parametrize("policy", ["v4", "v5", "v6"])
def test_real_agent_routes_deliver_to_provider(enabled, stage, mode, policy, monkeypatch):
    """Exercise actual stage entry, planner, coder, merge and diff generation."""
    from engine.search_node import SearchNode
    monkeypatch.setenv("MLEVOLVE_HWKG_EVIDENCE_POLICY", policy)
    module = importlib.import_module(f"agents.{stage}_agent")
    parent = SearchNode(code="x = 1", stage="draft", branch_id=1, _term_out=["output"])
    if policy == "v6":parent.code = "import torch\nx = 1"
    parent.local_best_node = parent
    agent = MagicMock()
    agent.cfg.agent.code.model = "deepseek-flash"
    agent.acfg = agent.cfg.agent
    agent.acfg.use_diff_mode = mode.startswith("diff")
    agent.acfg.code.temp = 1
    agent.cfg.pretrain_model_dir = ""
    agent.cfg.agent.use_global_memory = False
    agent.global_memory = None
    agent.use_coldstart = False
    agent.use_stepwise_generation = mode == "stepwise"
    agent.virtual_root = SearchNode(code="", stage="root")
    agent.virtual_root.fetch_child_memory = lambda: ""
    agent.task_desc = "text classification"
    agent.data_preview = "id,text,target"
    agent.fusion_draft_count = 0
    agent.max_fusion_drafts = 1
    received = []
    def backend(prompt, cfg, **kwargs):
        received.append(prompt)
        assert str(prompt).count("HWKG " + policy) == 1
        if kwargs.get("json_schema"):
            if mode == "diff_fallback":
                raise RuntimeError("planner fails, full-code fallback must retain context")
            return json.dumps({"reason": "fix x", "module": ["model_design"], "plan": {"model_design": "set x to 2"}})
        if "SEARCH/REPLACE" in str(prompt) or "<<<<<<< SEARCH" in str(prompt):
            if mode == "diff_retry" and len(received) == 2:
                return "<<<<<<< SEARCH\nabsent = 1\n=======\nabsent = 2\n>>>>>>> REPLACE"
            return "Fix x.\n```python\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n```"
        return "A complete plan.\n```python\nx = 2\n```"
    with ExitStack() as stack:
        if policy == "v6":
            stack.enter_context(patch("hwkg.host_policy.runtime_facts", return_value={
                'bf16_supported':True,'sdpa_api':True,'free_mib_snapshot':20000,
                'transformer_engine_package':False,'torchao_package':False}))
        stack.enter_context(patch("llm._openai.generate", side_effect=backend))
        stack.enter_context(patch.object(module, "register_node"))
        stack.enter_context(patch.object(module, "get_impl_guideline_from_agent", return_value={"Implementation guideline": []}))
        if hasattr(module, "get_patience_counter"):
            stack.enter_context(patch.object(module, "get_patience_counter", return_value=(0, 0, None)))
        if stage == "draft":
            stack.enter_context(patch.object(module, "get_prompt_environment", return_value={}))
        if stage == "evolution":
            stack.enter_context(patch.object(module, "_get_branch_trajectory_for_evolution", return_value="Step 1: learn\nStep 2: improve"))
        if stage == "fusion":
            reference = SearchNode(code="y = 1", stage="draft", _term_out=["output"], branch_id=2)
            parent.metric = reference.metric = NS(value=0.5)
            stack.enter_context(patch.object(module, "_get_fusion_candidates", return_value=[reference]))
        if stage == "aggregation":
            parent.metric = NS(value=0.5)
            stack.enter_context(patch.object(module, "_collect_branch_representatives", return_value=[parent, parent]))
        node = module.run(agent) if stage in {"draft", "aggregation"} else module.run(agent, parent)
    assert node.code and received
    if (mode.startswith("diff") or mode == "stepwise") and stage != "debug":
        assert len(received) >= 2
    if mode == "diff_retry":
        assert len(received) == 3
    assert "HWKG " + policy not in node.code
    if policy in {"v5", "v6"}:
        rows = [json.loads(s) for s in (enabled / "dispatch-audit.jsonl").read_text().splitlines()]
        phases = [r["phase"] for r in rows if r["event"] == "provider_dispatch"]
        if mode == "stepwise":
            assert "coder" in phases and phases[-1] == "merge"
        elif mode.startswith("diff") and stage != "debug":
            assert phases[0] == "planner"
            assert phases[-1] == ("plan_and_code" if mode == "diff_fallback" else "coder")
        else:
            assert all(p in {"coder", "plan_and_code"} for p in phases)
