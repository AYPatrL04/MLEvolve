import json
import logging
from types import SimpleNamespace

import pytest

from engine.milestone import verify_node
from engine.preflight import candidate_code_hash, diagnostic_to_review_issue
from engine.search_node import SearchNode
from utils.metric import MetricValue
from utils.training_diagnostics import TRAINING_DIAGNOSTICS_MARKER


def fixture(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    (public / "sample_submission.csv").write_text("id,target\n10,0\n11,0\n")
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "submission_good.csv").write_text("id,target\n10,0\n11,1\n")
    cfg = SimpleNamespace(
        workspace_dir=tmp_path, log_dir=tmp_path, data_dir=public, exp_id="nlp-getting-started",
        agent=SimpleNamespace(precision_optimization_mode="conservative"),
    )
    runtime = dict(schema_version=1, status="completed", device="cuda:0", completed_updates=2,
                   tf32_matmul_allowed=False, tf32_cudnn_allowed=False,
                   parameter_dtypes=["torch.float32"], autocast_dtypes_observed=["disabled"])
    node = SearchNode(id="good", code="x = 1\n", plan="test", stage="draft",
                      is_buggy=False, is_valid=True, exec_time=12,
                      metric=MetricValue(0.7, maximize=True), review_status="approved",
                      preflight_admitted=True, preflight_mode="full_cpu",
                      preflight_code_hash=candidate_code_hash("x = 1\n"))
    node._term_out = [TRAINING_DIAGNOSTICS_MARKER + " " + json.dumps(runtime)]
    packet = dict(job_id="gpu-job", node_id="good", status="parsed_valid", metric=0.7,
                  requires_gpu=True, duration_seconds=12)
    return cfg, node, packet, runtime


def test_verified_node_can_be_fast_without_fake_thirty_second_minimum(tmp_path):
    cfg, node, packet, _ = fixture(tmp_path)
    assert verify_node(cfg, node, packet)["met"]


@pytest.mark.parametrize("defect", ["packet", "admission", "stale_code", "nan", "submission", "ids", "diagnostics", "cpu", "tf32", "fp16", "no_updates"])
def test_success_requires_all_runtime_evidence(tmp_path, defect):
    cfg, node, packet, runtime = fixture(tmp_path)
    if defect == "packet":
        packet = None
    elif defect == "admission":
        node.preflight_admitted = False
    elif defect == "stale_code":
        node.code += "y = 2\n"
    elif defect == "nan":
        node.metric = MetricValue(float("nan"), maximize=True)
    elif defect == "submission":
        (tmp_path / "submission/submission_good.csv").unlink()
    elif defect == "ids":
        (tmp_path / "submission/submission_good.csv").write_text("id,target\n11,1\n10,0\n")
    elif defect == "diagnostics":
        runtime = {}
    elif defect == "cpu":
        runtime["device"] = "cpu"
    elif defect == "tf32":
        runtime["tf32_matmul_allowed"] = True
    elif defect == "fp16":
        runtime["autocast_dtypes_observed"] = ["torch.float16"]
    elif defect == "no_updates":
        runtime["completed_updates"] = 0
    node._term_out = [TRAINING_DIAGNOSTICS_MARKER + " " + json.dumps(runtime)]
    evidence = verify_node(cfg, node, packet)
    assert not evidence["met"] and evidence["reasons"]


def test_normal_accepts_fp32_or_scaled_selective_fp16(tmp_path):
    cfg, node, packet, runtime = fixture(tmp_path)
    cfg.agent.precision_optimization_mode = "normal"
    assert verify_node(cfg, node, packet)["met"]
    runtime.update(autocast_dtypes_observed=["disabled", "torch.float16"], grad_scaler_enabled=True)
    node._term_out = [TRAINING_DIAGNOSTICS_MARKER + " " + json.dumps(runtime)]
    assert verify_node(cfg, node, packet)["met"]


def test_dat001_repair_explains_multiinput_keys_and_integer_indices():
    issue = diagnostic_to_review_issue(dict(
        code="DAT001", stage="data_contract", classification="confirmed_candidate_failure",
        message="could not identify model inputs in the representative batch",
    ))
    assert "'inputs': (word_ids, char_ids, keyword_ids)" in issue.repair_instruction
    assert "torch.long" in issue.repair_instruction


def test_milestone_does_not_stop_at_failed_budget_or_overlap_generation(tmp_path):
    import run
    cfg, good, packet, _ = fixture(tmp_path)
    cfg.agent.stop_after_valid_nodes = 1
    cfg.agent.steps = 1
    cfg.agent.search = SimpleNamespace(num_drafts=1)
    journal = SimpleNamespace(nodes=[])

    class Agent:
        scfg = cfg.agent.search
        pipeline_logger = SimpleNamespace(latest_job_packet=lambda node_id: packet if node_id == "good" else None)
        generated = 0

        def has_selectable_work(self):
            return True

        def step(self, **kwargs):
            assert len(journal.nodes) == self.generated
            self.generated += 1
            return good if self.generated == 3 else SearchNode(
                id=str(self.generated), stage="draft", code="bad", plan="test",
                is_buggy=True, is_valid=False, exec_time=4000,
            )

        def execute_deferred_nodes(self, nodes, callback):
            journal.nodes.extend(nodes)

    agent = Agent()
    run._run_milestone_rounds(
        agent=agent, cfg=cfg, journal=journal, logger=logging.getLogger("milestone"),
        interpreter=SimpleNamespace(run=None, run_many=None), save_callback=lambda *_: None,
    )
    assert agent.generated == 3
    assert json.loads((tmp_path / "milestone_success.json").read_text())["met"]


def test_rejected_candidate_evidence_survives_discard(tmp_path):
    from engine.agent_search import AgentSearch
    cfg, node, _, _ = fixture(tmp_path)
    node.review_issues = [{"severity": "critical", "owner": "integration", "evidence": "bad batch keys"}]
    node.review_history = [{"round": 2, "issues": node.review_issues}]
    node.hardware_prompt_audit = [{"rendered_sha256": "audit"}]
    agent = AgentSearch.__new__(AgentSearch)
    agent.cfg, agent.pipeline_logger = cfg, None
    agent._discard_unfinished_node = lambda n: None
    agent._finalize_review_rejected_node(node)
    evidence = json.loads((tmp_path / "rejected_candidates/good/diagnostics.json").read_text())
    assert evidence["review_history"] == node.review_history
    assert evidence["hardware_prompt_audit"] == node.hardware_prompt_audit
    assert evidence["review_issues"] == node.review_issues
    assert (tmp_path / "rejected_candidates/good/candidate.py").read_text() == node.code
