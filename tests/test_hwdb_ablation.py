import copy
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from deployments.run_hwdb_ablation import config_for_cell, matrix_rows, prepare_graphs
from deployments.launch_hwdb_ablation import manifest
from engine import hwdb_ablation as ablation
from localml_scheduler.hardware_knowledge import feature_filter


def test_eight_cells_are_balanced_and_counterordered():
    rows = matrix_rows()
    assert len(rows) == 8
    assert len({(r["seed"], r["mode"], r["arm"]) for r in rows}) == 8
    for index in range(0, 8, 2):
        a, b = rows[index:index+2]
        assert a["seed"] == b["seed"] and a["mode"] == b["mode"]
        assert {a["arm"], b["arm"]} == {"original", "revised"}
    assert rows[0]["arm"] != rows[2]["arm"]


def test_seed42_only_matrix_has_four_cells(monkeypatch):
    monkeypatch.setenv("MLEVOLVE_ABLATION_SEEDS", "42")
    rows = matrix_rows()
    assert len(rows) == 4
    assert {row["seed"] for row in rows} == {42}
    assert {(row["mode"], row["arm"]) for row in rows} == {
        (mode, arm) for mode in ("conservative", "normal") for arm in ("original", "revised")
    }


def test_cell_configs_share_settings_and_allow_normal_fp32(tmp_path):
    repo = Path(__file__).parents[1]
    cfg = config_for_cell(repo, tmp_path, 42, "normal")
    assert cfg["agent"]["ablation_primary_attempts"] == 10
    assert cfg["agent"]["stop_after_valid_nodes"] == 1
    assert cfg["agent"]["time_limit"] is None
    assert cfg["agent"]["precision_optimization_mode"] == "normal"
    assert not cfg["hardware_knowledge"]["settings"]["graph"]["enabled"]
    assert not cfg["agent"]["use_global_memory"]
    assert not cfg["lesson_profiles"]["read_enabled"]
    assert cfg["agent"]["code"]["model"] == cfg["agent"]["feedback"]["model"] == "deepseek-flash"
    other = config_for_cell(repo, tmp_path, 43, "normal")
    other["agent"]["seed"] = 42
    assert other == cfg


def test_graph_override_is_checksum_guarded_and_explicit_path_wins(tmp_path, monkeypatch):
    path = tmp_path / "graph.json"
    path.write_text('{"nodes": [], "edges": []}')
    monkeypatch.setenv("MLEVOLVE_HWDB_GRAPH_PATH", str(path))
    monkeypatch.delenv("MLEVOLVE_HWDB_GRAPH_SHA256", raising=False)
    with pytest.raises(ValueError, match="SHA256"):
        feature_filter._load_graph()
    monkeypatch.setenv("MLEVOLVE_HWDB_GRAPH_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    assert feature_filter._load_graph() == {"nodes": [], "edges": []}
    path.write_text("{}")
    with pytest.raises(ValueError):
        feature_filter._load_graph()
    assert feature_filter._load_graph(path) == {}


def test_real_graphs_have_visible_contrast_under_shared_filters(tmp_path):
    result = prepare_graphs(Path(__file__).parents[1], tmp_path)
    assert result["original"]["sha256"] != result["revised"]["sha256"]
    contrasts = json.loads((tmp_path / "guidance_contrast.json").read_text())
    assert len(contrasts) == 10
    assert all(r["contexts"]["original"] != r["contexts"]["revised"] for r in contrasts)


def test_summary_does_not_call_rejections_bugs_or_count_extensions_as_primary():
    rows = [dict(status="review_rejected", node_id="a", verified_valid=False, elapsed_seconds=10),
            dict(status="generation_error", verified_valid=False, elapsed_seconds=20),
            dict(status="verified_valid", node_id="c", gpu_job_submitted=True, verified_valid=True, elapsed_seconds=30)]
    summary = ablation.summarize_attempts(rows, 2)
    assert summary["primary_attempts_finished"] == 2
    assert summary["primary_review_rejections"] == 1
    assert summary["primary_generation_errors"] == 1
    assert summary["primary_valid_yield_per_attempt"] == 0
    assert summary["all_verified_valid_nodes"] == 1
    assert summary["first_valid_seconds"] == 30
    assert summary["extension_attempts"] == 1
    assert summary["confirmed_buggy_execution_rate"] is None
    assert summary["validator_false_positive_rate"] is None


@pytest.mark.parametrize("outcomes,expected", [([False, False, True], 3), ([True, False], 2)])
def test_serial_rounds_count_rejections_and_continue_to_one_valid(tmp_path, monkeypatch, outcomes, expected):
    cfg = SimpleNamespace(log_dir=tmp_path, agent=SimpleNamespace(ablation_primary_attempts=2, search=SimpleNamespace(num_drafts=1)))
    journal = SimpleNamespace(nodes=[])
    calls = []
    class Agent:
        pipeline_logger = SimpleNamespace(latest_job_packet=lambda node_id: {"requires_gpu": node_id == "valid"})
        def step(self, **kwargs):
            ok = outcomes[len(calls)]
            calls.append(ok)
            return SimpleNamespace(id="valid" if ok else f"rejected-{len(calls)}", stage="draft", review_status="approved" if ok else "rejected", preflight_admitted=ok)
        def execute_deferred_nodes(self, nodes, callback):
            journal.nodes.extend(nodes)
    monkeypatch.setattr(ablation, "snapshot_candidate", lambda *args: "evidence")
    monkeypatch.setattr(ablation, "guidance_provenance", lambda *args: {"verified": True})
    monkeypatch.setattr(ablation, "verify_node", lambda cfg, candidate, packet: {"met": candidate.id == "valid"})
    ablation.run_ablation_rounds(agent=Agent(), interpreter=SimpleNamespace(run=lambda: None, run_many=lambda: None),
                                cfg=cfg, journal=journal, logger=logging.getLogger("test"),
                                save_callback=lambda *args: None, ensure_capacity=lambda **kwargs: True)
    rows = json.loads((tmp_path / "ablation/attempts.json").read_text())
    assert len(rows) == expected
    assert rows[0]["phase"] == "primary"
    if expected == 3:
        assert rows[-1]["phase"] == "extension"
    assert json.loads((tmp_path / "ablation/summary.json").read_text())["all_verified_valid_nodes"] == 1


def test_missing_or_wrong_graph_audit_cannot_verify_guidance(monkeypatch):
    monkeypatch.setenv("MLEVOLVE_HWDB_GRAPH_SHA256", "expected")
    node = SimpleNamespace(hardware_prompt_audit=[])
    assert not ablation.guidance_provenance(node)["verified"]
    node.hardware_prompt_audit = [{"experiment_graph": {"sha256": "wrong"}}]
    assert not ablation.guidance_provenance(node)["verified"]
    node.hardware_prompt_audit = [{"experiment_graph": {"sha256": "expected"}}]
    assert ablation.guidance_provenance(node)["verified"]


def test_three_operational_failures_stop_with_saved_accounting(tmp_path):
    cfg = SimpleNamespace(log_dir=tmp_path, agent=SimpleNamespace(ablation_primary_attempts=10, search=SimpleNamespace(num_drafts=1)))
    class Agent:
        def step(self, **kwargs):
            raise RuntimeError("provider unavailable")
    with pytest.raises(RuntimeError, match="Three consecutive"):
        ablation.run_ablation_rounds(agent=Agent(), interpreter=SimpleNamespace(run=None), cfg=cfg,
                                    journal=SimpleNamespace(nodes=[]), logger=logging.getLogger("test"),
                                    save_callback=lambda *args: None, ensure_capacity=lambda **kwargs: True)
    summary = json.loads((tmp_path / "ablation/summary.json").read_text())
    assert summary["primary_generation_errors"] == 3
    assert summary["primary_attempts_finished"] == 3


@pytest.mark.parametrize("phase", ["prepare", "run"])
def test_launch_uses_only_deepseek_secret_no_private_labels(phase):
    job = manifest(phase, "a"*40)
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["env"][1]["valueFrom"]["secretKeyRef"]["name"] == "hwdb-deepseek-20260911"
    assert all("heldout" not in str(m) and "credentials" not in str(m) for m in container["volumeMounts"])
    assert ("nvidia.com/gpu" in container["resources"]["requests"]) == (phase == "run")
    assert ("activeDeadlineSeconds" in job["spec"]) == (phase == "prepare")
