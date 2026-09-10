from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from agents.hardware_context import (
    HardwarePromptContext,
    _compact_stage_hardware_feature,
    _filter_precision_stage_features,
    _format_stage_hardware_features,
    _render_prompt_lines,
    apply_hardware_context_to_node,
    format_compact_hardware_prompt_section,
)
from agents.precision_validation import validate_training_precision
from config import PreflightConfig
from engine.preflight import ModelPreflightGate, candidate_code_hash
from engine.search_node import SearchNode
from localml_scheduler.hardware_knowledge.feature_filter import query_hardware_features, query_hardware_node
from utils.precision_policy import NORMAL_PRECISION_INSTRUCTION, resolve_precision_policy


GPUS = ["GeForce RTX 5090", "NVIDIA A10", "NVIDIA A100 PCIe 80GB", "NVIDIA A100 SXM4 40GB"]


def test_database_loader_retains_gpu_specific_guidance():
    from pathlib import Path
    from localml_scheduler.hardware_knowledge.records import load_hardware_knowledge_from_schema
    from localml_scheduler.hardware_knowledge.store import HardwareKnowledgeGraphStore

    bundle = load_hardware_knowledge_from_schema(Path(__file__).parents[1] / "schema")
    hardware = next(h for h in bundle["hardware"] if h["name"] == "NVIDIA A10")
    feature = next(f for f in bundle["features"] if f["feature_id"] == "amp")
    edge = next(r for r in bundle["relationships"] if r["hardware_id"] == hardware["hardware_id"] and r["feature_id"] == "amp")
    store = object.__new__(HardwareKnowledgeGraphStore)
    result = store._public_result({"hardware": hardware, "feature": feature, "relationship": edge})
    assert "FP32 parameters" in result["summary_text"]
    assert len(result["recommended_patterns"]) == 4
    assert "dtype=torch.float16" in result["sample_code"]
    assert "Do not cast the whole model" in result["avoid_patterns"][0]


@pytest.mark.parametrize("gpu", GPUS)
@pytest.mark.parametrize("mode", ["conservative", "normal"])
def test_target_gpu_modes_and_guidance(gpu, mode):
    for stage in (None, "model_design", "datatype_precision", "training_evaluation"):
        node = query_hardware_node(gpu, stage, precision_mode=mode)
        assert node["gpu_name"] == gpu
        features = query_hardware_features(gpu, stage, precision_mode=mode)["features"]
        precision = {f["feature_id"] for f in features if f["category"] == "precision" and f.get("recommended")}
        assert precision <= ({"amp", "fp16"} if mode == "normal" else set())
        patterns = node.get("recommended_patterns", []) + [p for f in features for p in f.get("recommended_patterns", [])]
        assert not any(token in " ".join(patterns).lower() for token in ("bf16", "tf32", "fp64", "quantized"))
    if mode == "normal":
        amp = next(f for f in query_hardware_features(gpu, "datatype_precision")["features"] if f["feature_id"] == "amp")
        assert "FP32 parameters" in amp["description"]
        assert "backward and optimizer updates outside autocast" in " ".join(amp["recommended_patterns"])
        compile(amp["example_code"], "hwdb-example", "exec")


def test_a100_lookup_does_not_fall_back_to_a10():
    assert query_hardware_node("NVIDIA A100-SXM4-80GB")["gpu_name"] == "NVIDIA A100 SXM4 80GB"
    assert query_hardware_node("A100")["found"] is False  # Capacity and form factor ambiguous.
    assert query_hardware_node("A10")["gpu_name"] == "NVIDIA A10"


@pytest.mark.parametrize("mode", ["conservative", "normal"])
@pytest.mark.parametrize("code", [
    "torch.backends.cuda.matmul.allow_tf32 = True",
    "torch.backends.cudnn.allow_tf32 = True",
    "torch.backends.cuda.matmul.fp32_precision = 'tf32'",
    "torch.set_float32_matmul_precision('high')",
    "PRECISION = 'tf32'",
    "AMP_DTYPE = torch.bfloat16",
    "with torch.autocast('cuda', dtype=torch.float16):\n    y = model(x)\nx = x.bfloat16()",
    "model = model.half()",
])
def test_strict_modes_reject_explicit_unsupported_paths(mode, code):
    agent = SimpleNamespace(acfg=SimpleNamespace(precision_optimization_mode=mode))
    context = HardwarePromptContext(compact_context={"hardware_context": {"hardware": {"architecture": "ampere"}}})
    assert validate_training_precision(agent, code, context=context)


def test_normal_allows_fp32_islands_and_scales_backward_outside_autocast():
    feature = next(f for f in query_hardware_features("NVIDIA A10", "datatype_precision")["features"] if f["feature_id"] == "fp16")
    agent = SimpleNamespace(acfg=SimpleNamespace(precision_optimization_mode="normal"))
    assert not validate_training_precision(agent, feature["example_code"])
    policy = resolve_precision_policy({"architecture": "blackwell"}, mode="normal")
    prompt = format_compact_hardware_prompt_section({"precision_policy": policy.to_dict()}, stage="datatype_precision")
    assert "FP16 AMP is optional and local" in prompt
    assert "FP32 parameters/inputs" in prompt
    assert "CPU preflight uses FP32" in NORMAL_PRECISION_INSTRUCTION


def test_filter_does_not_promote_explicit_hwdb_rejection(tmp_path):
    graph = {"nodes": [
        {"id": "hw:a10", "label": "Hardware", "properties": {"name": "NVIDIA A10", "architecture": "ampere"}},
        {"id": "feat:fp16", "label": "Feature", "properties": {"feature_id": "fp16", "category": "precision"}},
    ], "edges": [{"from": "hw:a10", "to": "feat:fp16", "type": "HAS_FEATURE", "properties": {"recommended": False}}]}
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(graph))
    feature = query_hardware_features("NVIDIA A10", "datatype_precision", graph_path=path)["features"][0]
    assert feature["recommended"] is False
    node = query_hardware_node("NVIDIA A10", "datatype_precision", graph_path=path)
    assert not node.get("recommended_feature_keys")
    context = {"stages": [{"stage": "datatype_precision", "features": [feature]}]}
    filtered = _filter_precision_stage_features(context, resolve_precision_policy({"architecture": "ampere"}))
    assert filtered["stages"][0]["features"][0]["recommended"] is False


def test_prompt_budget_cannot_separate_advice_from_its_condition():
    condition = "Only after " + "measured validation " * 35 + "on the target GPU."
    feature = _compact_stage_hardware_feature({"feature_id": "fp16", "category": "precision", "recommended": True,
        "limitations": condition, "recommended_patterns": ["CANDIDATE_SENTINEL: consider AMP."]})
    assert feature["limitations"] == condition
    lines = _format_stage_hardware_features({"found": True, "stages": [{"stage": "datatype_precision", "features": [feature]}]})
    for budget in (250, 500, 1500):
        prompt = _render_prompt_lines(lines, max_chars=budget)
        assert len(prompt) <= budget
        if "CANDIDATE_SENTINEL" in prompt:
            assert condition in prompt


def test_feedback_keeps_each_attempt_and_does_not_invent_attribution(tmp_path, monkeypatch):
    import model_preflight

    def unavailable(*args, **kwargs):
        raise RuntimeError("checker infrastructure unavailable")

    monkeypatch.setattr(model_preflight, "check", unavailable)
    cfg = SimpleNamespace(workspace_dir=tmp_path, preflight=PreflightConfig(target_profile="nvidia/a10_24gb"),
                          scheduler=SimpleNamespace(settings=None), exp_id="test")
    node = SearchNode(code="print('first')", stage="draft", id="audit")
    context = HardwarePromptContext(compact_context={"precision_policy": {"mode": "normal"}},
        raw_context={"recommended_patterns": ["raw advice"]}, filtered_context={"features": ["fp16"]}, prompt_section="filtered guidance")
    apply_hardware_context_to_node(node, context)
    gate = ModelPreflightGate(cfg)
    first = gate.run(node, generated=False, attempt=0)
    node.code = "print('second')"
    gate.run(node, generated=False, attempt=1)
    folder = tmp_path / "working" / "preflight" / "audit"
    old = json.loads((folder / "feedback_attempt_0.json").read_text())
    new = json.loads((folder / "feedback_attempt_1.json").read_text())
    assert old["outcome"]["code_hash"] == first.code_hash == candidate_code_hash("print('first')")
    assert new["outcome"]["code_hash"] != first.code_hash
    assert (folder / "candidate_attempt_0.py").read_text() == "print('first')"
    assert old["outcome"]["status"] == "INTERNAL_ERROR"
    assert old["attribution"] == "unassigned"
    audit = old["hardware_prompt_audit"][0]
    assert audit["raw_context"] == context.raw_context
    assert audit["filtered_context"] == context.filtered_context
    assert audit["prompt_sha256"] == hashlib.sha256(b"filtered guidance").hexdigest()
