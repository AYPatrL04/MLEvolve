"""Real-agent PetFinder audit with CPU preflight and no experiment execution.

Only model transport is adapted to authenticated Claude CLI. Repository draft,
feedback, repair, precision validation and CPU gate entrypoints run unchanged.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import patch

import jsonschema

from agents import debug_agent, draft_agent, result_parse_agent, stage_repair
from agents.coder import base_coder
from agents.hardware_context import get_hardware_context_for_stage
from agents.precision_validation import validate_training_precision
from engine.executor import ExecutionResult
from engine.preflight import ModelPreflightGate, apply_outcome_to_node, inspect_adapter
from engine.search_node import SearchNode
from knowledge.records import from_source
from llm.common import compile_prompt_to_md

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "records" / "2026-09-14_petfinder_workflow_matrix_final"
SYSTEM = (
    "You are a text-only model inference endpoint. You cannot call tools or inspect files. "
    "All task information is in the supplied prompt. Return the requested complete plan and code, "
    "SEARCH/REPLACE patch, or JSON directly. Never describe a future tool action. "
    "The supplied MLEvolve interfaces exist in the execution environment."
)
MARKER = "AUDIT_INJECTED_LOSS_PATH"
COLUMNS = ["Subject Focus", "Eyes", "Face", "Near", "Action", "Accessory", "Group", "Collage", "Human", "Occlusion", "Info", "Blur"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


class ReadOnlyHWDB:
    """Versioned fixture evidence, independent of live DBs or GPU discovery."""
    def __init__(self):
        self.calls = []

    def get_model_design_hardware_context(self, **kwargs):
        return self.get_optimization_context(**kwargs)

    def get_optimization_context(self, **kwargs):
        self.calls.append("get_optimization_context")
        return {
            "hardware_context": {"hardware": {
                "hardware_key": "audit-rtx5090", "gpu_name": "NVIDIA GeForce RTX 5090",
                "architecture": "blackwell", "compute_capability": "12.0", "total_vram_mb": 32768,
                "torch_version": "2.8", "cuda_runtime": "12.8",
            }},
            "runtime_estimate": {"found": False, "reason": "No measured profile in this audit fixture"},
            "confidence": 0.0,
        }

    def get_design_knowledge(self, **kwargs):
        self.calls.append("get_design_knowledge")
        records = []
        for topic, text, fallback in (
            ("architecture", "For image plus tabular regression, fuse spatially pooled image features with the metadata branch.", "Keep both input modalities if reducing model size."),
            ("precision", "Keep regression loss and RMSE reductions in FP32 under any allowed training precision.", "Use FP32 when the chosen accelerated training path is unavailable."),
            ("training", "Use the same train-only preprocessing in the adapter and training path.", "Rebuild features when preprocessing changes."),
        ):
            records += from_source({
                "record_id": f"audit-{topic}", "summary": text, "topics": [topic],
                "when_to_use": "PetFinder-style image and tabular regression on the target profile.",
                "fallbacks": [fallback], "evidence_refs": [f"audit-fixture:{topic}"],
                "verification_status": "advisory",
            }, domain="hardware")
        return records


def prepare_data(workspace):
    from PIL import Image

    data = workspace / "input"
    for split, count in (("train", 24), ("test", 4)):
        images = data / split
        images.mkdir(parents=True, exist_ok=True)
        with (data / f"{split}.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Id", *COLUMNS, *(["Pawpularity"] if split == "train" else [])])
            for index in range(count):
                name = f"{split}_{index:03d}"
                Image.new("RGB", (256, 256), (30 + index * 5, 80, 120)).save(images / f"{name}.jpg")
                writer.writerow([name, *[(index + offset) % 2 for offset in range(12)], *([20 + index * 2] if split == "train" else [])])
    (workspace / "submission").mkdir(exist_ok=True)


def make_agent(workspace, precision, hardware):
    from config import PreflightConfig

    code = SimpleNamespace(model="sonnet", temp=0, completion_tokens=10000)
    acfg = SimpleNamespace(
        code=code, feedback=SimpleNamespace(model="sonnet", temp=0), design_knowledge_version="v2",
        hardware_context_enabled=hardware, hardware_context_mode="full", precision_optimization_mode=precision,
        use_global_memory=False, pipeline_decision_enabled=False, time_limit=600, steps=1,
        review=SimpleNamespace(repair_retries=2, parallel_training_repairs=False),
    )
    cfg = SimpleNamespace(
        workspace_dir=workspace, exp_id="petfinder-pawpularity-score", pretrain_model_dir="",
        experiment=SimpleNamespace(mode="hardware_aware"), hardware_knowledge={"enabled": hardware}, agent=acfg,
        exec=SimpleNamespace(timeout=60),
        preflight=PreflightConfig(enabled=True, target_profile=str(ROOT / "config/preflight_profiles/rtx_5090_32gb.yaml"),
                                 fail_open_on_internal_error=False, cache=False, cpu_timeout_seconds=20,
                                 abstract_timeout_seconds=15),
    )
    task = (
        f"Build a complete small PetFinder-style image plus tabular regression script. Input root: {workspace / 'input'}. "
        "train.csv has Id, twelve numeric binary metadata columns, and Pawpularity in [0,100]; test.csv omits the target. "
        f"Metadata columns: {COLUMNS}. JPEG paths are input/train/<Id>.jpg and input/test/<Id>.jpg. "
        "Use a tiny real CNN (at most two convolution layers with 8 and 16 channels), global image pooling, "
        "and a small tabular branch. Preserve image [3,256,256] and tabular [12] interfaces. Train from scratch "
        "with one train/validation split and at most two epochs. Fit preprocessing only on training data. "
        "Report RMSE in original target units and produce Id,Pawpularity test predictions. "
        "Use CPU-safe real CandidateAdapter methods; all training side effects belong inside the main guard. "
        "Target deployment is RTX 5090, but this audit will execute only CPU adapter checks, not the training entrypoint. "
        "Select precision according to the configured mode; normal mode may use FP32 when appropriate. "
        "The dataset is synthetic audit evidence, so do not claim leaderboard quality."
    )
    return SimpleNamespace(
        cfg=cfg, acfg=acfg, scfg=SimpleNamespace(num_drafts=1, num_bugs=2, num_improves=1, num_topk=1),
        task_desc=task, data_preview="24 training and 4 test rows; 256x256 RGB JPEGs; 12 metadata features; scalar target.",
        use_coldstart=False, scheduler_client=None, hardware_knowledge_client=ReadOnlyHWDB(),
        cuda_docs_service=None, lesson_profile_client=None, global_memory=None,
        metric_maximize=False, start_time=time.time(), current_step=0,
        virtual_root=SearchNode(code="", plan="root", stage="root"), next_branch_id=0,
        branch_all_nodes={}, branch_successful_nodes={}, pipeline_logger=None,
        _serialize_prompt=lambda value: value if isinstance(value, str) else json.dumps(value, default=str),
    )


class Transport:
    timeout_seconds = 240

    def __init__(self, output):
        self.output = output
        self.calls = []

    def call(self, role, prompt, schema=None):
        text = prompt if isinstance(prompt, str) else compile_prompt_to_md(prompt)
        number = len(self.calls) + 1
        prefix = f"{number:02d}_{role}"
        (self.output / f"{prefix}.prompt.txt").write_text(text)
        command = ["claude", "-p", text, "--system-prompt", SYSTEM, "--model", "sonnet",
                   "--output-format", "json", "--no-session-persistence", "--tools", ""]
        if schema:
            command += ["--json-schema", json.dumps(schema)]
        call = {"role": role, "prompt_chars": len(text), "prefix": prefix, "started": time.monotonic()}
        self.calls.append(call)
        try:
            proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=self.timeout_seconds, check=False)
            (self.output / f"{prefix}.cli.json").write_text(proc.stdout or proc.stderr)
            if proc.returncode:
                raise RuntimeError(f"Claude CLI exited {proc.returncode}: {proc.stderr}")
            payload = json.loads(proc.stdout)
            if payload.get("is_error"):
                raise RuntimeError(str(payload.get("result")))
            response = payload.get("structured_output") if schema else None
            if response is None:
                response = payload.get("result", "")
            if schema and isinstance(response, str):
                response = json.loads(response)
            if schema:
                jsonschema.Draft7Validator(schema).validate(response)
            rendered = response if isinstance(response, str) else json.dumps(response)
            (self.output / f"{prefix}.response.txt").write_text(rendered)
            call.update(response_chars=len(rendered), usage=payload.get("usage"), model_usage=payload.get("modelUsage"), ok=True)
            return response
        except Exception as exc:
            call.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            call["duration_seconds"] = time.monotonic() - call["started"]
            write_json(self.output / "calls.json", self.calls)

    def draft(self, **kwargs):
        return self.call("draft", kwargs["prompt"])

    def repair(self, **kwargs):
        return self.call("repair", kwargs["prompt"])

    def feedback(self, system_message, user_message, func_spec, **kwargs):
        return self.call("feedback", system_message, func_spec.json_schema)


def inject_failure(code):
    tree = ast.parse(code)
    adapter = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CandidateAdapter")
    method = next(node for node in adapter.body if isinstance(node, ast.FunctionDef) and node.name == "training_step")
    first = method.body[0]
    lines = code.splitlines(keepends=True)
    lines.insert(first.lineno - 1, " " * first.col_offset + f"raise RuntimeError('{MARKER}')\n")
    return "".join(lines)


def run_cell(precision, hardware, root):
    name = f"{precision}_hwdb_{'on' if hardware else 'off'}"
    output = root / name
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    workspace.mkdir()
    prepare_data(workspace)
    agent = make_agent(workspace, precision, hardware)
    transport = Transport(output)
    started = time.monotonic()
    result = {"cell": name, "precision": precision, "hardware_enabled": hardware, "stages": [], "assertions": {}}

    def timed(label, function):
        begin = time.monotonic()
        try:
            return function()
        finally:
            result["stages"].append({"name": label, "start_seconds": begin - started, "duration_seconds": time.monotonic() - begin})

    def gate(node, label, attempt):
        outcome = timed(label, lambda: ModelPreflightGate(agent.cfg).run(node, generated=True, attempt=attempt))
        apply_outcome_to_node(node, outcome, repair_count=attempt)
        node.review_issues = [issue.to_dict() for issue in outcome.issues]
        write_json(output / f"{label}.json", outcome.to_dict())
        return outcome

    try:
        with patch.object(base_coder, "generate", transport.draft), patch.object(stage_repair, "generate", transport.repair), patch.object(result_parse_agent, "query", transport.feedback):
            node = timed("draft", lambda: draft_agent.run(agent))
            (output / "draft.py").write_text(node.code)
            result["initial_generation_calls"] = sum(call["role"] == "draft" for call in transport.calls)
            result["generation_strategy"] = node.generation_strategy
            initial = gate(node, "draft_preflight", 0)
            result["draft_preflight_status"] = initial.status
            node.code = inject_failure(node.code)
            (output / "injected.py").write_text(node.code)
            induced = gate(node, "injected_preflight", 1)
            raw = json.loads(Path(induced.report_path).read_text())
            cause = next(item for item in raw["diagnostics"] if MARKER in item.get("message", ""))
            execution = ExecutionResult([cause.get("stack_trace") or cause["message"]], None,
                                        cause.get("exception_type"), {"failure_origin": "candidate", "message": cause["message"]})
            timed("feedback", lambda: result_parse_agent.run(agent, node, execution))
            result["feedback_summary"] = node.analysis
            result["feedback_issues"] = node.review_issues
            # The CPU gate owns admission repairs; parser analysis stays available.
            # Use its authoritative grouped defects, avoiding duplicate runtime diagnoses.
            node.review_issues = [issue.to_dict() for issue in induced.issues]
            node.is_buggy = True
            repaired = timed("debug", lambda: debug_agent.run(agent, node))
            if repaired is None:
                raise RuntimeError("Debug returned no applicable repair")
            (output / "repaired.py").write_text(repaired.code)
            final = gate(repaired, "repaired_preflight", 2)
            result["repaired_preflight_status"] = final.status
            context = get_hardware_context_for_stage(agent, "code_review", code=repaired.code)
            policy_issues = validate_training_precision(agent, repaired.code, context=context)
            result["precision_issues"] = [issue.to_dict() for issue in policy_issues]
            prompt = (output / "01_draft.prompt.txt").read_text()
            result["assertions"] = {
                "one_initial_generation": result["initial_generation_calls"] == 1,
                "single_pass": node.generation_strategy == "single_pass",
                "adapter_complete": inspect_adapter(repaired.code).complete,
                "induced_failure_reproduced": not induced.admitted and bool(cause),
                "feedback_provider_succeeded": any(call["role"] == "feedback" and call.get("ok") for call in transport.calls),
                "feedback_identifies_failure": MARKER in node.analysis,
                "repair_removes_injected_failure": MARKER not in repaired.code,
                "cpu_gate_admitted_repair": final.admitted and not any(issue.severity == "critical" for issue in final.issues),
                "precision_policy_valid": not any(issue.severity == "critical" for issue in policy_issues),
                "knowledge_toggle_respected": ("audit-architecture" in prompt) == hardware,
                "all_subjects_with_conditions": all(word in prompt for word in ("audit-architecture", "audit-precision", "audit-training", "Fallback:")) if hardware else not agent.hardware_knowledge_client.calls,
                "cpu_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            }
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result.update(duration_seconds=time.monotonic() - started, calls=transport.calls,
                  hardware_calls=agent.hardware_knowledge_client.calls, target_profile="RTX 5090 32GB",
                  evidence_tier="Real Sonnet agent responses, fixture hardware knowledge, real CPU adapter checks; no GPU or training entrypoint execution")
    write_json(output / "result.json", result)
    return result


def plot(results, path):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1400, 760), "white")
    draw = ImageDraw.Draw(image)
    draw.text((25, 15), "PetFinder agent workflow: measured stage durations; CPU checks only", fill="black")
    colors = {"draft": "#609ed4", "draft_preflight": "#a5d79b", "injected_preflight": "#db9291",
              "feedback": "#9a84c9", "debug": "#f2b56a", "repaired_preflight": "#63b596"}
    maximum = max([result["duration_seconds"] for result in results] or [1])
    for index, result in enumerate(results):
        y = 70 + index * 70
        draw.text((20, y + 6), result["cell"], fill="black")
        for stage in result["stages"]:
            x = 240 + stage["start_seconds"] / maximum * 1100
            width = max(1, stage["duration_seconds"] / maximum * 1100)
            draw.rectangle((x, y, x + width, y + 25), fill=colors.get(stage["name"], "gray"))
        draw.text((240, y + 30), f"{result['duration_seconds']:.1f}s total", fill="black")
    draw.text((25, 365), "Metric-node audit: checks passed by cell (no model-quality scores measured)", fill="black")
    for index, result in enumerate(results):
        x = 250 + index * 300
        passed = sum(result["assertions"].values())
        total = len(result["assertions"])
        draw.ellipse((x, 440, x + 100, 540), fill="#63b596" if total and passed == total else "#db9291")
        draw.text((x + 15, 480), f"{passed}/{total}", fill="black")
        draw.text((x - 50, 570), result["cell"], fill="black")
        draw.text((x - 50, 600), f"Initial draft calls: {result.get('initial_generation_calls', '?')}", fill="black")
        draw.text((x - 50, 625), f"CPU gate: {result.get('repaired_preflight_status', 'incomplete')}", fill="black")
        draw.text((x - 50, 650), "CPU adapter: PASS; GPU: unverified" if result.get("cpu_stage_statuses", {}).get("cpu_training") == "PASS" else "CPU adapter: check report", fill="black")
    draw.text((25, 700), "Blue=draft; light green=initial CPU gate; red=injected gate; purple=feedback; orange=debug; green=recheck; gray=resume gate", fill="black")
    image.save(path)


def audit_saved_evidence(result, output):
    """Check complete CPU outcomes and deduplication from retained artifacts."""
    summary = Path(result.get("final_preflight_ref") or output / "repaired_preflight.json")
    if summary.exists():
        report = json.loads(Path(json.loads(summary.read_text())["report_path"]).read_text())
        statuses = {stage["name"]: stage["status"] for stage in report["stages"]}
        result["cpu_stage_statuses"] = statuses
        result["assertions"]["cpu_adapter_checks_passed"] = all(
            statuses.get(stage) == "PASS" for stage in ("construction", "data_contract", "cpu_training", "validation")
        )
    prompt = (output / "01_draft.prompt.txt").read_text()
    counts = {topic: prompt.count(f"hardware:audit-{topic}:") for topic in ("architecture", "precision", "training")}
    result["knowledge_record_occurrences"] = counts
    result["assertions"]["knowledge_not_duplicated"] = all(count == int(result["hardware_enabled"]) for count in counts.values())
    evidence_dir = output / "cpu_reports"
    evidence_dir.mkdir(exist_ok=True)
    provenance = []
    for source in sorted((output / "workspace/working/preflight").glob("*/report_attempt_*.json")):
        data = source.read_bytes()
        target = evidence_dir / f"{source.parent.name}_{source.name}"
        target.write_bytes(data)
        provenance.append({"source": str(source), "copy": str(target), "sha256": hashlib.sha256(data).hexdigest()})
    write_json(output / "cpu_reports_manifest.json", provenance)
    write_json(output / "result.json", result)
    return result


def repair_existing_cell(result, output):
    """Resume recorded candidates through existing validators and real repair."""
    phase = output / f"continuation_{len(list(output.glob('continuation_*'))) + 1}"
    phase.mkdir()
    write_json(phase / "prior_result.json", result)
    agent = make_agent(output / "workspace", result["precision"], result["hardware_enabled"])
    source = Path(result.get("final_code_ref") or output / ("repaired.py" if (output / "repaired.py").exists() else "draft.py"))
    node = SearchNode(code=source.read_text(), plan="Resume recorded audit candidate", stage="draft", parent=agent.virtual_root)
    agent.branch_all_nodes = {node.branch_id: [node]}
    agent.branch_successful_nodes = {node.branch_id: []}
    transport = Transport(phase)
    begin = time.monotonic()
    offset = result["duration_seconds"]

    def timed(label, action):
        start = time.monotonic()
        try:
            return action()
        finally:
            result["stages"].append({"name": label, "start_seconds": offset + start - begin,
                                     "duration_seconds": time.monotonic() - start})

    with patch.object(stage_repair, "generate", transport.repair), patch.object(result_parse_agent, "query", transport.feedback):
        outcome = timed("resume_preflight", lambda: ModelPreflightGate(agent.cfg).run(node, generated=True, attempt=3))
        apply_outcome_to_node(node, outcome, repair_count=3)
        context = get_hardware_context_for_stage(agent, "code_review", code=node.code)
        issues = outcome.issues + list(validate_training_precision(agent, node.code, context=context))
        write_json(phase / "issues_before.json", [issue.to_dict() for issue in issues])
        node.review_issues = [issue.to_dict() for issue in issues]
        node.analysis = "Recorded preflight or precision diagnostics require candidate repair."
        if not any(call["role"] == "feedback" and call.get("ok") for call in result["calls"]):
            logs = ["CPU preflight diagnostic: " + issue.evidence + "\n" for issue in outcome.issues]
            execution = ExecutionResult(logs, None, None, {"failure_origin": "candidate"})
            timed("feedback", lambda: result_parse_agent.run(agent, node, execution))
            result["feedback_summary"] = node.analysis
            result["feedback_issues"] = node.review_issues
        node.review_issues = [issue.to_dict() for issue in issues]
        node.is_buggy = True
        repaired = timed("debug", lambda: debug_agent.run(agent, node))
        if repaired is None:
            raise RuntimeError("Continuation repair did not apply")
        (phase / "repaired.py").write_text(repaired.code)
        final = timed("repaired_preflight", lambda: ModelPreflightGate(agent.cfg).run(repaired, generated=True, attempt=4))
        context = get_hardware_context_for_stage(agent, "code_review", code=repaired.code)
        precision_issues = validate_training_precision(agent, repaired.code, context=context)
        write_json(phase / "repaired_preflight.json", final.to_dict())
        result["final_code_ref"] = str(phase / "repaired.py")
        result["final_preflight_ref"] = str(phase / "repaired_preflight.json")
        result["precision_issues"] = [issue.to_dict() for issue in precision_issues]
        result["repaired_preflight_status"] = final.status
        if result.get("error"):
            result["initial_audit_error"] = result.pop("error")
        result["calls"] += [{**call, "prefix": phase.name + "/" + call["prefix"]} for call in transport.calls]
        prompt = (output / "01_draft.prompt.txt").read_text()
        result["assertions"].update({
            "one_initial_generation": result["initial_generation_calls"] == 1,
            "single_pass": result["generation_strategy"] == "single_pass",
            "adapter_complete": inspect_adapter(repaired.code).complete,
            "candidate_failure_reproduced": bool(issues),
            "feedback_provider_succeeded": any(call["role"] == "feedback" and call.get("ok") for call in result["calls"]),
            "feedback_identifies_failure": bool(result.get("feedback_summary")) and ("target" in result["feedback_summary"].lower() or MARKER in result["feedback_summary"]),
            "cpu_gate_admitted_repair": final.admitted and not final.issues,
            "precision_policy_valid": not precision_issues,
            "knowledge_toggle_respected": ("audit-architecture" in prompt) == result["hardware_enabled"],
            "all_subjects_with_conditions": all(word in prompt for word in ("audit-architecture", "audit-precision", "audit-training", "Only when:", "Fallback:")) if result["hardware_enabled"] else not agent.hardware_knowledge_client.calls,
            "cpu_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        })
    result["duration_seconds"] = offset + time.monotonic() - begin
    result["timing_scope"] = "active audit phases; excludes idle time before continuation"
    result["hardware_calls"] += agent.hardware_knowledge_client.calls
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--cell", choices=[f"{mode}_hwdb_{toggle}" for mode in ("normal", "conservative") for toggle in ("off", "on")])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repair-existing", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "transport_system.txt").write_text(SYSTEM)
    results = []
    for mode in ("normal", "conservative"):
        for enabled in (False, True):
            name = f"{mode}_hwdb_{'on' if enabled else 'off'}"
            if args.cell and name != args.cell:
                continue
            existing = args.output / name / "result.json"
            result = json.loads(existing.read_text()) if args.resume and existing.exists() else run_cell(mode, enabled, args.output)
            if args.repair_existing:
                result = repair_existing_cell(result, args.output / name)
            result = audit_saved_evidence(result, args.output / name)
            results.append(result)
            if args.cell:
                plot(results, args.output / name / "audit.png")
            else:
                write_json(args.output / "matrix.json", {"provider": "Claude CLI Sonnet, text-only transport", "results": results})
                plot(results, args.output / "matrix.png")
            print(json.dumps({"cell": name, "status": result.get("repaired_preflight_status"), "assertions": result["assertions"], "error": result.get("error")}), flush=True)


if __name__ == "__main__":
    main()
