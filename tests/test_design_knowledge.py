from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from knowledge.migration import migrate
from knowledge.records import from_source, render_records, select_records
from knowledge.runtime import fit_prompt, pin_version


def record(summary="Use cached features for repeated evaluation.", **values):
    return from_source({"record_id": "source", "summary": summary, "evidence_refs": ["doc:source"], **values}, domain="hardware")[0]


def test_complete_conditions_deduplicated_across_topics_without_cap():
    first = record(avoid_patterns=["Do not reuse cached features after changing augmentation."], fallbacks=["Recompute features when inputs change."])
    second = {**first, "record_id": "second", "topics": ["evaluation"], "evidence_refs": ["doc:second"]}
    records = select_records([first, second], {})
    prompt = render_records(records)
    assert len(records) == 1
    assert records[0]["evidence_refs"] == ["doc:source", "doc:second"]
    assert prompt.count(first["summary"]) == 1
    assert "after changing augmentation" in prompt
    assert "Recompute features" in prompt
    long_condition = "Only when " + "all source-specific conditions remain satisfied; " * 150 + "otherwise use FP32."
    prompt = render_records([record(long_condition)])
    assert long_condition in prompt
    assert len(prompt) > 3500


@pytest.mark.parametrize("field,expected,actual", [
    ("hardware_keys", ["v100"], "a100"), ("framework_versions", ["2.8"], "2.7"),
    ("backend_modes", ["mps_process"], "cuda_process"), ("min_compute_capability", "8.0", "7.0"),
])
def test_wrong_runtime_filtered(field, expected, actual):
    key = {"hardware_keys": "hardware_key", "framework_versions": "framework_major_minor", "backend_modes": "backend_mode", "min_compute_capability": "compute_capability"}[field]
    item = record(**{field: expected})
    assert select_records([item], {key: actual}) == []
    assert select_records([item], {}) == []


def test_distinct_conditions_and_malformed_records():
    first = record(when_to_use="Only for fixed shapes.", avoid_patterns=["Keep an eager fallback."])
    second = record(when_to_use="Only for variable shapes.", avoid_patterns=["Keep an eager fallback."])
    selected = select_records([first, second, {"summary": "Invalid record."}], {})
    assert len(selected) == 2
    assert render_records(selected).count("Keep an eager fallback.") == 2
    assert "min_compute_capability=8.0" in render_records([record(min_compute_capability=8.0)])


def test_legacy_lesson_view_is_preserved_as_concise_advice():
    from agents.design_knowledge import lesson_records

    context = SimpleNamespace(compact_context={"family_hardware_profile": {
        "match_level": "exact", "profile_key": "prior", "revision": 2,
        "baseline": {"model_summary": "Keep validation preprocessing deterministic."},
        "evidence_refs": ["node:prior"], "relevant_lessons": [],
    }})
    found = lesson_records(context)
    assert found[0]["verification_status"] == "advisory"
    assert found[0]["revision"] == 2
    assert found[0]["evidence_refs"] == ["node:prior"]


def test_local_and_retrieved_history_do_not_repeat_the_same_attempt():
    from agents.design_knowledge import history_records

    node = SimpleNamespace(id="prior", code="import torch", plan="Use a small classifier.", metric=SimpleNamespace(value=0.9), is_buggy=False)
    memory = SimpleNamespace(design_records_v2=from_source({"summary": "Stored classifier attempt.", "evidence_refs": ["node:prior"]}, domain="history"))
    agent = SimpleNamespace(task_desc="Classify images", virtual_root=SimpleNamespace(children=[node]),
                            global_memory=SimpleNamespace(retrieve_similar_records=lambda *_a, **_k: [(memory, 1.0)]))
    assert len(history_records(agent)) == 1


def test_unknown_family_is_conditional_and_conflicts_are_not_merged():
    first = record(model_families=["vit"], verification_status="verified")
    second = record("Checkpointing caused a slowdown in the measured case.", model_families=["vit"], verification_status="conflicted")
    selected = select_records([first, second], {}, role="draft")
    assert len(selected) == 2
    assert "model_families=vit" in render_records(selected)
    assert "conflicted" in render_records(selected)
    assert select_records(selected, {"model_family": "cnn"}) == []


def test_unprocessed_document_is_withheld_without_cutting_it():
    source = {"record_id": "long", "text": "First section.\n\n" + "Raw document. " * 5000, "evidence_refs": ["doc:long"]}
    assert from_source(source, domain="hardware") == []
    assert "Raw document" in source["text"]


def test_migration_dry_run_resume_provenance_and_source_revisions(tmp_path):
    sources = [{"record_id": str(i), "summary": f"Fact {i}.", "evidence_refs": [f"doc:{i}"]} for i in range(3)]
    output = tmp_path / "migration"
    report = migrate(sources, domain="hardware", output=output)
    assert report["complete"] and not output.exists()
    partial = migrate(sources, domain="hardware", output=output, apply=True, max_sources=1)
    assert not partial["complete"] and not (output / "hardware-v2.json").exists()
    complete = migrate(sources, domain="hardware", output=output, apply=True)
    again = migrate(sources, domain="hardware", output=output, apply=True)
    assert complete["records_hash"] == again["records_hash"]
    assert all(row["resumed"] for row in again["mapping"])
    export = json.loads((output / "hardware-v2.json").read_text())
    assert {ref for item in export["records"] for ref in item["evidence_refs"]} == {"doc:0", "doc:1", "doc:2"}
    sources[0]["summary"] = "Corrected fact."
    changed = migrate(sources, domain="hardware", output=output, apply=True)
    assert changed["mapping"][0]["revision"] == 2
    import sqlite3

    with sqlite3.connect(output / "migration.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 4


def test_migration_accounts_for_invalid_sources_with_stable_native_ids(tmp_path):
    sources = [{"rule_id": "rule", "summary": "Use an eager fallback."},
               {"chunk_id": "bad", "summary": "Invalid confidence.", "confidence": 2.0}]
    report = migrate(sources, domain="hardware", output=tmp_path)
    assert report["complete"] and report["awaiting_correction"] == ["bad"]
    assert report["mapping"][0]["source_id"] == "rule"


def test_cuda_source_retains_complete_conditions_without_legacy_caps():
    from localml_scheduler.cuda_docs.normalizer import normalize_mcp_result

    source = "Short fact.\n\n" + "Additional evidence. " * 3000 + "Only use with compatible shapes."
    result = normalize_mcp_result({"text": source, "url": "https://docs.nvidia.com/cuda/"},
                                  retrieved_date="2026-09-13", preserve_complete_text=True)
    assert result.chunks[0].text == source
    assert from_source({"text": result.chunks[0].text}, domain="cuda") == []


def test_version_pin_preserves_resumed_runs_and_rejects_invalid_versions(tmp_path):
    assert pin_version(tmp_path / "new", "v2", resuming=False) == "v2"
    assert pin_version(tmp_path / "new", "v1", resuming=True) == "v2"
    assert pin_version(tmp_path / "old", "v2", resuming=True) == "v1"
    with pytest.raises(ValueError):
        pin_version(tmp_path / "bad", "v3", resuming=False)


def test_context_fit_reserves_completion_and_does_not_truncate_task():
    config = SimpleNamespace(context_window_tokens=100, completion_tokens=40)
    agent = SimpleNamespace(acfg=SimpleNamespace(code=config), design_prompt_token_counter=lambda prompt: len(prompt))
    optional = record("Optional reference.")
    prompt, records, diagnostic = fit_prompt(agent, lambda knowledge: "Task" + knowledge, [optional])
    assert prompt == "Task" and records == []
    assert diagnostic["dropped_record_ids"] == [optional["record_id"]]
    with pytest.raises(ValueError, match="not truncated"):
        fit_prompt(agent, lambda knowledge: "Task" * 100 + knowledge, [])
    config.context_window_tokens = None
    prompt, records, diagnostic = fit_prompt(agent, lambda knowledge: "Task" + knowledge, [optional])
    assert records == [optional] and diagnostic["sizing"] == "unavailable"


def test_draft_is_one_complete_call_with_all_subjects(monkeypatch, tmp_path):
    from agents import draft_agent, hardware_context
    from agents.coder import base_coder, stepwise_coder
    from agents.prompts import pipeline_decision
    from engine.search_node import SearchNode
    from agents.lesson_context import LessonPromptContext
    from config import PreflightConfig

    forbidden = Mock(side_effect=AssertionError("extra LLM call"))
    monkeypatch.setattr(hardware_context, "generate", forbidden)
    monkeypatch.setattr(pipeline_decision, "generate", forbidden)
    monkeypatch.setattr(stepwise_coder, "generate", forbidden)
    root = SearchNode(plan="root", code="", stage="root")
    root.add_expected_child_count = Mock(return_value=True)
    agent = SimpleNamespace(
        cfg=SimpleNamespace(preflight=PreflightConfig(enabled=False)),
        acfg=SimpleNamespace(code=SimpleNamespace(model="qwen", temp=0.0), design_knowledge_version="v2"),
        scfg=SimpleNamespace(), virtual_root=root, task_desc="Train an image classifier.",
        data_preview="Images with class labels.", use_coldstart=False, scheduler_client=None,
    )
    stages = [{"stage": stage, "features": [{"record_id": stage, "summary_text": text, "evidence_refs": [f"doc:{stage}"]}]} for stage, text in (
        ("model_design", "Use convolution for image structure."),
        ("datatype_precision", "Keep evaluation reductions in float32."),
        ("training_evaluation", "Log the validation metric after each epoch."),
    )]
    stages[0]["features"][0].update(avoid_patterns=["Do not change the validation split."], fallbacks=["Use eager execution if compilation fails."])
    context = hardware_context.HardwarePromptContext(raw_context={"stage_hardware_features": {"stages": stages + [stages[0]]}})
    seen_selection = []
    def brief(_agent, *, select_features=True):
        seen_selection.append(select_features)
        return hardware_context.HardwarePromptContext()
    monkeypatch.setattr(draft_agent, "get_hardware_design_brief", brief)
    monkeypatch.setattr(draft_agent, "get_hardware_context_for_stage", lambda *_a, **_k: context)
    monkeypatch.setattr(draft_agent, "get_cuda_docs_context", lambda *_a, **_k: None)
    lessons = from_source({"lesson_id": "prior", "summary": "Keep metric reductions deterministic.", "evidence_refs": ["node:prior"]}, domain="lesson")
    monkeypatch.setattr(draft_agent, "get_lesson_context_for_stage", lambda *_a, **_k: LessonPromptContext("draft", {"design_records_v2": lessons}, ""))
    monkeypatch.setattr(draft_agent, "get_impl_guideline_from_agent", lambda *_a: {"Implementation guideline": ["Keep required local model paths."]})
    register = Mock()
    monkeypatch.setattr(draft_agent, "register_node", register)
    generation = Mock(return_value="Use a complete classifier.\n```python\nprint('Final Validation Score: 0.8')\n```")
    monkeypatch.setattr(base_coder, "generate", generation)
    node = draft_agent.run(agent)
    assert generation.call_count == 1 and forbidden.call_count == 0
    assert seen_selection == [False]
    assert node.generation_strategy == "single_pass"
    assert node.pipeline_decision is None and node.stage_note_board == []
    actual = generation.call_args.kwargs["prompt"]
    assert register.call_args.args[2] == actual
    text = json.dumps(actual)
    for stage in stages:
        assert text.count(stage["features"][0]["summary_text"]) == 1
    assert "Do not change the validation split." in text
    assert "Use eager execution if compilation fails." in text
    assert "Keep metric reductions deterministic." in text
    assert len(node.diagnostics["design_knowledge"]["evidence_refs"]) == 4


def test_code_index_filters_before_ranking_and_replaces_stale_summaries():
    import numpy as np
    from qdrant_client import QdrantClient
    from localml_scheduler.code_knowledge.store import CodeKnowledgeStore
    from localml_scheduler.config.models import HardwareFeatureDBSettings

    embedding = SimpleNamespace(dimension=2, encode=lambda texts, **kwargs: np.asarray([[1.0, 0.5] for _ in texts]))
    client = QdrantClient(":memory:")
    store = CodeKnowledgeStore(SimpleNamespace(hardware_feature_db=HardwareFeatureDBSettings()), qdrant_client=client, embedding_model=embedding)
    sources = [{"schema_version": "code_doc_chunk_v1", "chunk_id": name, "title": name,
                "text": summary, "hardware_keys": [hardware]} for name, hardware, summary in (
        ("wrong", "a100", "Only applies on A100."), ("right", "v100", "Only applies on V100."),
    )]
    store.ingest_records(sources)
    scope = {"hardware_key": "v100", "framework": "pytorch", "runner_contract": "subprocess_job_v1"}
    found = store.search_design_knowledge(query="training", context=scope, limit=1)
    assert [item["summary"] for item in found] == ["Only applies on V100."]
    sources[1]["text"] = "Updated V100 guidance."
    store.ingest_records([sources[1]])
    found = store.search_design_knowledge(query="training", context=scope)
    assert [item["summary"] for item in found] == ["Updated V100 guidance."]
    sources[1]["text"] = "Unprocessed source.\n\n" + "A document paragraph. " * 1000
    store.ingest_records([sources[1]])
    assert store.search_design_knowledge(query="training", context=scope) == []
    assert store.search(query="training", filters={"hardware_keys": "v100"})
    for source in sources:
        source["hardware_keys"] = ["v100"]
        source["text"] = "Installed-version-specific guidance."
        source["applicability"] = {"framework_major_minor": "2.8" if source["chunk_id"] == "right" else "2.7"}
    store.ingest_records(sources)
    scope["framework_major_minor"] = "2.8"
    found = store.search_design_knowledge(query="training", context=scope, limit=1)
    assert len(found) == 1 and found[0]["applicability"]["framework_major_minor"] == "2.8"
    client.close()


def test_graph_ingestion_persists_conditions_and_original_evidence(monkeypatch, tmp_path):
    from hardware_knowledge_graph.config import HardwareKnowledgeSettings
    from localml_scheduler.hardware_knowledge.store import HardwareKnowledgeGraphStore

    store = HardwareKnowledgeGraphStore(HardwareKnowledgeSettings(runtime_root=tmp_path))
    writes = []
    monkeypatch.setattr(store, "_run_write", lambda query, params=None: writes.append((query, params)) or [])
    hardware = {"hardware_id": "h", "name": "V100"}
    feature = {"feature_id": "f", "name": "feature", "description": "Full source description.",
               "design_summary": "Use only on a supported runtime.", "avoid_patterns": ["Do not use dynamic shapes."], "sample_code": "original code"}
    relationship = {"hardware_id": "h", "feature_id": "f", "verified": True, "limitations": "Keep an eager fallback."}
    store.ingest_bundle({"hardware": [hardware], "features": [feature], "relationships": [relationship]})
    props = next(params["props"] for query, params in writes if "SET r +=" in query)
    records = json.loads(props["design_records_v2_json"])
    assert records[0]["summary"] == feature["design_summary"]
    assert "Do not use dynamic shapes." in records[0]["restrictions"]
    assert "Keep an eager fallback." in records[0]["restrictions"]
    assert any(params and params.get("props", {}).get("sample_code") == "original code" for _, params in writes)

    from hardware_knowledge_graph.client import HardwareKnowledgeClient, _sanitize_agent_response
    from knowledge.records import validate_record

    # Exercise the exact graph projection that previously erased required empty fields.
    records[0]["evidence_refs"].append("https://example.org/precision#conditions")
    props["design_records_v2_json"] = json.dumps(records)
    monkeypatch.setattr(store, "_query_neighborhood_rows", lambda **kwargs: [
        {"hardware": hardware, "feature": feature, "relationship": props},
    ])
    public = _sanitize_agent_response(store.get_feature_neighborhood(hardware_terms=["V100"]))
    projected = public["features"][0]["design_records_v2"][0]
    assert validate_record(projected) == records[0]
    assert projected["applies_when"] == [] and projected["fallbacks"] == []
    client = HardwareKnowledgeClient(store.settings, include_profile_evidence=False)
    client._hardware_knowledge_store = store
    monkeypatch.setattr(client, "hardware_profile", lambda: SimpleNamespace(gpu_name="V100", hardware_key="v100"))
    found = client.get_design_knowledge(candidate={}, context={"hardware_key": "v100"})
    assert len(found) == 1
    assert found[0]["applicability"] == {"hardware_keys": ["v100"]}
    assert found[0]["evidence_refs"] == records[0]["evidence_refs"]
