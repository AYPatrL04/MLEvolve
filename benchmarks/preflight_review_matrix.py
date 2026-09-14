"""Audit preflight-before-review through the real search controller, without training.

Generate each draft once, inject a reproducible loss failure, prove review is
skipped, then run the controller's real debug/preflight/review path. Only model
transport and evidence-recording spies are adapted; CPU checks run normally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from unittest.mock import patch

from agents import code_review_agent, draft_agent, stage_repair
from agents.coder import base_coder
from benchmarks.petfinder_workflow_matrix import (
    MARKER, ROOT, SYSTEM, Transport, inject_failure, make_agent, prepare_data, write_json,
)
from engine.agent_search import AgentSearch
from engine.preflight import ModelPreflightGate, candidate_code_hash, is_fresh_preflight
from engine.search_node import Journal

DEFAULT = ROOT / "records/2026-09-14_preflight_review_matrix"


class AuditTransport(Transport):
    timeout_seconds = 360

    def __init__(self, output, result, started):
        super().__init__(output)
        self.result = result
        self.started = started

    def call(self, role, prompt, schema=None):
        begin = time.monotonic()
        try:
            return super().call(role, prompt, schema)
        except subprocess.TimeoutExpired as exc:
            message = f"{role} transport timed out after {self.timeout_seconds} seconds; prompt retained in its artifact file"
            self.calls[-1]["error"] = message
            write_json(self.output / "calls.json", self.calls)
            raise RuntimeError(message) from exc
        finally:
            self.result["stages"].append({
                "name": role, "start_seconds": begin - self.started,
                "duration_seconds": time.monotonic() - begin,
            })

    def review(self, system_message, user_message, func_spec, **kwargs):
        return self.call("review", system_message, func_spec.json_schema)


def run_cell(mode, enabled, root):
    name = f"{mode}_hwdb_{'on' if enabled else 'off'}"
    output = root / name
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    workspace.mkdir()
    (output / "checks").mkdir()
    prepare_data(workspace)
    agent = AgentSearch.__new__(AgentSearch)
    agent.__dict__.update(vars(make_agent(workspace, mode, enabled)))
    agent.journal = Journal()
    agent.journal_lock = threading.Lock()
    agent.best_node = None
    agent.acfg.review.classifier_retries = 1
    agent.acfg.review.max_repair_rounds = 1
    agent.acfg.review.fail_open_on_unavailable = False
    agent.cfg.preflight.max_repair_rounds = 0
    started = time.monotonic()
    result = {"cell": name, "precision": mode, "hardware_enabled": enabled,
              "assertions": {}, "stages": [], "checks": [], "review_entries": []}
    transport = AuditTransport(output, result, started)
    original_draft = draft_agent.run
    original_gate = ModelPreflightGate.run
    original_review = code_review_agent.review_and_repair

    def draft(*args, **kwargs):
        node = original_draft(*args, **kwargs)
        (output / "draft.py").write_text(node.code)
        result["generation_strategy"] = node.generation_strategy
        node.code = inject_failure(node.code)
        (output / "injected.py").write_text(node.code)
        return node

    def gate(gate_self, node, **kwargs):
        begin = time.monotonic()
        index = len(result["checks"]) + 1
        (output / "checks" / f"{index:02d}_candidate.py").write_text(node.code)
        outcome = original_gate(gate_self, node, **kwargs)
        entry = {"index": index, "node_id": node.id, "start_seconds": begin - started,
                 "duration_seconds": time.monotonic() - begin, **outcome.to_dict()}
        if outcome.report_path:
            source = Path(outcome.report_path)
            target = output / "checks" / f"{index:02d}_report.json"
            target.write_bytes(source.read_bytes())
            entry.update(report_copy=str(target), report_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        result["checks"].append(entry)
        result["stages"].append({"name": "preflight", "start_seconds": entry["start_seconds"],
                                 "duration_seconds": entry["duration_seconds"]})
        write_json(output / "checks.json", result["checks"])
        return outcome

    def review(review_agent, node):
        result["review_entries"].append({
            "node_id": node.id, "start_seconds": time.monotonic() - started,
            "admitted": node.preflight_admitted, "status": node.preflight_status,
            "fresh": is_fresh_preflight(node), "code_hash": candidate_code_hash(node.code),
            "evidence_ref": node.preflight_report_path,
        })
        assert node.preflight_admitted is True and is_fresh_preflight(node)
        return original_review(review_agent, node)

    def no_execution(*args, **kwargs):
        raise AssertionError("Training/execution callback must not be invoked")

    try:
        with patch.object(base_coder, "generate", transport.draft), \
             patch.object(stage_repair, "generate", transport.repair), \
             patch.object(code_review_agent, "query", transport.review), \
             patch.object(draft_agent, "run", draft), \
             patch.object(ModelPreflightGate, "run", gate), \
             patch.object(code_review_agent, "review_and_repair", review):
            _, rejected = agent._run_single_step(agent.virtual_root, no_execution, execute_immediately=False)
            result["initial_generation_calls"] = sum(call["role"] == "draft" for call in transport.calls)
            result["assertions"].update({
                "one_initial_generation": result["initial_generation_calls"] == 1,
                "single_pass": result.get("generation_strategy") == "single_pass",
                "blocked_candidate_skips_review": rejected.review_status == "rejected" and not result["review_entries"],
                "injected_failure_reproduced": any(MARKER in issue["evidence"] for issue in rejected.review_issues),
            })
            assert result["assertions"]["blocked_candidate_skips_review"]
            write_json(output / "rejected_issues.json", rejected.review_issues)
            agent._finalize_review_rejected_node(rejected)
            agent.cfg.preflight.max_repair_rounds = 1
            _, final = agent._run_single_step(rejected, no_execution, execute_immediately=False)
            if final is None:
                raise AssertionError("Debug produced no candidate")
            (output / "repaired.py").write_text(final.code)
            write_json(output / "review_history.json", final.review_history)
            write_json(output / "final_issues.json", final.review_issues)
            result.update(final_review_status=final.review_status, final_preflight_status=final.preflight_status,
                          gpu_check_required=final.preflight_gpu_check_required)
            check_count = len(result["checks"])
            fresh_admitted = agent._ensure_node_preflight_before_execution(final)
            final_report = json.loads(Path(final.preflight_report_path).read_text())
            statuses = {stage["name"]: stage["status"] for stage in final_report["stages"]}
            result["cpu_stage_statuses"] = statuses
            prompt = (output / "01_draft.prompt.txt").read_text()
            counts = {topic: prompt.count(f"hardware:audit-{topic}:") for topic in ("architecture", "precision", "training")}
            result["knowledge_record_occurrences"] = counts
            review_prompts = [(output / (call["prefix"] + ".prompt.txt")).read_text()
                              for call in transport.calls if call["role"] == "review"]
            result["assertions"].update({
                "debug_removes_injected_failure": MARKER not in final.code,
                "review_follows_fresh_admission": bool(result["review_entries"]) and all(entry["admitted"] and entry["fresh"] for entry in result["review_entries"]),
                "review_receives_preflight_evidence": bool(review_prompts) and "CPU preflight evidence" in review_prompts[0] and "gpu_check_required" in review_prompts[0],
                "final_review_approved": final.review_status in {"approved", "repaired"},
                "final_code_admitted": fresh_admitted and is_fresh_preflight(final),
                "unchanged_code_reuses_preflight": len(result["checks"]) == check_count,
                "cpu_adapter_checks_passed": all(statuses.get(stage) == "PASS" for stage in ("construction", "data_contract", "cpu_training", "validation")),
                "preflight_history_preserved": any(entry["event"] == "model_preflight_completed" for entry in final.review_history),
                "knowledge_not_duplicated": all(count == int(enabled) for count in counts.values()),
                "hardware_toggle_respected": bool(agent.hardware_knowledge_client.calls) == enabled,
                "no_unresolved_critical_issues": not any(issue["severity"] == "critical" for issue in final.review_issues),
                "cpu_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not agent.journal.nodes,
            })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result.update(duration_seconds=time.monotonic() - started, calls=transport.calls,
                  hardware_calls=agent.hardware_knowledge_client.calls,
                  evidence_tier="Real controller, Sonnet responses, CPU checks; fixture RTX 5090 knowledge; no training entrypoint or GPU execution")
    result["assertions"]["completed_without_error"] = "error" not in result
    write_json(output / "result.json", result)
    return result


def plot(results, path):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1400, 780), "white")
    draw = ImageDraw.Draw(image)
    draw.text((25, 15), "Preflight before review: real search controller; CPU checks only", fill="black")
    colors = {"draft": "#609ed4", "draft_timeout": "#db9291", "preflight": "#63b596", "repair": "#f2b56a", "review": "#9a84c9"}
    maximum = max([result["duration_seconds"] for result in results] or [1])
    for index, result in enumerate(results):
        y = 60 + index * 70
        draw.text((20, y + 6), result["cell"], fill="black")
        for stage in result["stages"]:
            x = 240 + stage["start_seconds"] / maximum * 1100
            width = max(1, stage["duration_seconds"] / maximum * 1100)
            draw.rectangle((x, y, x + width, y + 25), fill=colors[stage["name"]])
        draw.text((240, y + 30), f"{result['duration_seconds']:.1f}s active audit time", fill="black")
    draw.text((25, 360), "Blue=draft; red=timed-out draft; green=CPU preflight; orange=targeted repair; purple=LLM review", fill="black")
    draw.text((25, 395), "Metric-node audit checks (no model-quality or GPU performance measurements)", fill="black")
    for index, result in enumerate(results):
        x = 250 + index * 300
        passed, total = sum(result["assertions"].values()), len(result["assertions"])
        draw.ellipse((x, 450, x + 100, 550), fill="#63b596" if total and passed == total else "#db9291")
        draw.text((x + 20, 490), f"{passed}/{total}", fill="black")
        draw.text((x - 50, 575), result["cell"], fill="black")
        counts = {role: sum(call["role"] == role for call in result["calls"]) for role in ("draft", "review", "repair")}
        draw.text((x - 50, 605), f"Calls: draft {counts['draft']}, review {counts['review']}, repair {counts['repair']}", fill="black")
        draw.text((x - 50, 630), f"Review: {result.get('final_review_status', 'incomplete')}", fill="black")
        draw.text((x - 50, 655), f"Preflight: {result.get('final_preflight_status', 'incomplete')}", fill="black")
        draw.text((x - 50, 680), "CPU adapter: PASS; GPU unverified" if result.get("assertions", {}).get("cpu_adapter_checks_passed") else "CPU adapter: see report", fill="black")
    image.save(path)


def load_attempts(paths):
    """Aggregate saved attempts without hiding transport failures or idle gaps."""
    attempts = [(path, json.loads(path.read_text())) for path in paths]
    result = dict(attempts[-1][1])
    result["completed_attempt_generation_calls"] = result.pop("initial_generation_calls", 0)
    result["assertions"] = dict(result["assertions"])
    if "one_initial_generation" in result["assertions"]:
        result["assertions"]["one_generation_in_completed_attempt"] = result["assertions"].pop("one_initial_generation")
    result["stages"], result["calls"], result["attempts"] = [], [], []
    offset = 0.0
    for path, attempt in attempts:
        result["attempts"].append({"result_ref": str(path), "completed_without_error": "error" not in attempt,
                                   "duration_seconds": attempt["duration_seconds"]})
        for stage in attempt["stages"]:
            name = "draft_timeout" if stage["name"] == "draft" and "timed out" in attempt.get("error", "") else stage["name"]
            result["stages"].append({**stage, "name": name, "start_seconds": offset + stage["start_seconds"]})
        result["calls"].extend({**call, "artifact_dir": str(path.parent)} for call in attempt["calls"])
        offset += attempt["duration_seconds"]
    result["duration_seconds"] = offset
    result["initial_generation_calls"] = sum(call["role"] == "draft" for call in result["calls"])
    result["timing_scope"] = "All recorded active attempts; excludes idle gaps between attempts"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--cell", choices=[f"{mode}_hwdb_{toggle}" for mode in ("normal", "conservative") for toggle in ("off", "on")])
    parser.add_argument("--resume", action="store_true", help="Load existing results without fresh model calls")
    args = parser.parse_args()
    args.output = args.output.resolve()
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "transport_system.txt").write_text(SYSTEM)
    results = []
    for mode in ("normal", "conservative"):
        for enabled in (False, True):
            name = f"{mode}_hwdb_{'on' if enabled else 'off'}"
            if args.cell and name != args.cell:
                continue
            existing = args.output / name / "result.json"
            paths = [existing, *sorted(args.output.glob(f"retry_*/{name}/result.json"))]
            paths = [path for path in paths if path.exists()]
            result = load_attempts(paths) if args.resume and paths else run_cell(mode, enabled, args.output)
            results.append(result)
            print(json.dumps({"cell": name, "assertions": result["assertions"], "error": result.get("error")}), flush=True)
    target = args.output / args.cell if args.cell else args.output
    write_json(target / "matrix.json", {"provider": "Claude CLI Sonnet, text-only transport", "results": results})
    plot(results, target / "matrix.png")


if __name__ == "__main__":
    main()
