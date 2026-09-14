"""Compare an archived staged draft with a rebuilt single-call prompt, offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents import draft_agent
from agents.hardware_context import HardwarePromptContext, compact_optimization_context
from agents.lesson_context import LessonPromptContext
from config import PreflightConfig
from engine.search_node import SearchNode
from localml_scheduler.client import SchedulerClient
from utils.precision_policy import resolve_precision_policy


def audit(database: Path, *, hardware_name: str) -> tuple[dict, dict]:
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        node_id, archived = connection.execute("SELECT node_id,prompt_text FROM prompt_snapshots WHERE stage='draft' ORDER BY prompt_id LIMIT 1").fetchone()
    parts = re.split(r"(?m)^# Used Prompt \d+: ", archived)[1:]
    if len(parts) < 2:
        raise ValueError("Archive must contain the captured staged draft prompts")
    task_match = re.search(r"# Task description\s*\n(.*?)\n# (?:Memory|Hardware)", parts[0], re.S)
    if task_match is None:
        raise ValueError("Cannot extract the archived task without altering its requirements")
    stage_evidence = SchedulerClient._stage_feature_context_from_static_graph(
        hardware_name=hardware_name, stages=["model_design", "datatype_precision", "training_evaluation"], limit=8, precision_mode="normal")
    if not stage_evidence.get("found"):
        raise ValueError("Static hardware evidence was not found")
    raw = {"hardware_context": {"found": True, "hardware": stage_evidence["hardware"]}, "stage_hardware_features": stage_evidence}
    compact = compact_optimization_context(raw)
    compact["precision_policy"] = resolve_precision_policy(stage_evidence["hardware"], mode="normal").to_dict()
    context = HardwarePromptContext(raw_context=raw, compact_context=compact)
    root = SearchNode(plan="root", code="", stage="root")
    root.add_expected_child_count = Mock(return_value=True)
    agent = SimpleNamespace(
        cfg=SimpleNamespace(preflight=PreflightConfig(enabled=False), exec=SimpleNamespace(timeout=90), pretrain_model_dir="", exp_id="petfinder-pawpularity-score"),
        acfg=SimpleNamespace(code=SimpleNamespace(model="qwen-replay", temp=0), time_limit=3600, steps=1, precision_optimization_mode="normal"),
        design_knowledge_version="v2", virtual_root=root, scfg=SimpleNamespace(), task_desc=task_match.group(1).strip(),
        data_preview="PetFinder images and tabular metadata; target Pawpularity.", use_coldstart=False,
        scheduler_client=None, current_step=0, start_time=time.time(),
    )
    completion = Mock(return_value="Prompt-construction audit only.\n```python\npass\n```")
    with patch.object(draft_agent, "get_hardware_design_brief", return_value=HardwarePromptContext()), \
         patch.object(draft_agent, "get_hardware_context_for_stage", return_value=context), \
         patch.object(draft_agent, "get_cuda_docs_context", return_value=None), \
         patch.object(draft_agent, "get_lesson_context_for_stage", return_value=LessonPromptContext("draft", {}, "")), \
         patch.object(draft_agent, "register_node"), \
         patch("agents.coder.base_coder.generate", completion), \
         patch("agents.hardware_context.generate", side_effect=AssertionError("Unexpected LLM call")):
        node = draft_agent.run(agent)
    prompt = completion.call_args.kwargs["prompt"]
    prompt_chars = sum(len(value) for value in prompt.values())
    old_sizes = {part.split("\n", 1)[0]: len(part.split("\n", 1)[1]) for part in parts}
    report = {
        "evidence_tier": "offline prompt construction; no model or training execution",
        "archive": str(database), "archive_node_id": node_id,
        "archive_sha256": hashlib.sha256(archived.encode()).hexdigest(),
        "hardware_fixture": hardware_name, "archived_captured_calls": len(parts), "rebuilt_generation_calls": completion.call_count,
        "archived_prompt_chars": old_sizes, "archived_total_chars": sum(old_sizes.values()),
        "rebuilt_prompt_chars": prompt_chars, "rebuilt_knowledge": node.diagnostics["design_knowledge"],
        "limitations": ["Captured archive sections exclude any separately logged decision/feature-selection requests.",
                        "The rebuilt prompt uses the unchanged archived task with a fixed data-preview fixture and current curated hardware records.",
                        "No claims about latency, output correctness or model quality; token counts are unavailable."],
    }
    return report, prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-db", type=Path, required=True)
    parser.add_argument("--hardware", default="NVIDIA GeForce RTX 5090")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    report, prompt = audit(args.pipeline_db, hardware_name=args.hardware)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    args.output_prefix.with_suffix(".prompt.json").write_text(json.dumps(prompt, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(report["archived_prompt_chars"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1, 1.6]})
    for index, name in enumerate(names):
        axes[0].barh("Archived", 0.85, left=index, color="#6681a5")
        axes[0].text(index + .42, "Archived", name.replace("_", "\n"), ha="center", va="center", fontsize=8)
    axes[0].barh("Single call", .85, color="#4a9977")
    axes[0].set_xlabel("Call sequence (equal slots; not measured time)")
    axes[0].set_title("Offline draft prompt comparison")
    axes[1].plot(range(1, len(names) + 1), list(report["archived_prompt_chars"].values()), "o-", label="Archived characters per call", color="#6681a5")
    axes[1].scatter([1], [report["rebuilt_prompt_chars"]], label="Single-call characters", color="#4a9977", s=90)
    axes[1].set_xlabel("Call index")
    axes[1].set_ylabel("Prompt characters")
    axes[1].legend()
    axes[1].grid(alpha=.2)
    fig.text(.5, .02, "Prompt construction only. Fixed preview fixture; no LLM latency, training, or quality measurement.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(json.dumps({key: report[key] for key in ("archived_captured_calls", "rebuilt_generation_calls", "archived_total_chars", "rebuilt_prompt_chars")}, indent=2))


if __name__ == "__main__":
    main()
