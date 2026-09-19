"""Draft Agent: initial plan and code draft."""

import logging
import time
from pathlib import Path
from typing import Any, Optional

from llm import compile_prompt_to_md
from engine.search_node import SearchNode
from agents.coder import plan_and_code_query
from agents.triggers import register_node
from agents.hardware_context import (
    apply_hardware_design_brief_to_node,
    apply_hardware_context_to_node,
    get_hardware_design_brief,
    get_hardware_context_for_stage,
    hardware_context_instructions,
)
from agents.cuda_docs_context import get_cuda_docs_context
from agents.design_knowledge import cuda_records, hardware_records, history_records, lesson_records, runtime_scope
from knowledge.records import select_records
from knowledge.runtime import fit_prompt
from agents.lesson_context import (
    apply_lesson_context_to_node,
    get_lesson_context_for_stage,
    lesson_context_instructions,
)
from agents.prompts import (
    ROBUSTNESS_GENERALIZATION_STRATEGY,
    prompt_leakage_prevention,
    prompt_resp_fmt,
    get_prompt_environment,
    get_impl_guideline_from_agent,
)
from agents.planner import build_chat_prompt_for_model

logger = logging.getLogger("MLEvolve")


def model_preflight_generation_instructions() -> list[str]:
    """Return the non-negotiable CPU-admission interface for generated code."""
    return [
        "- **CPU PREFLIGHT CONTRACT (MANDATORY)**: Define a no-argument "
        "`class CandidateAdapter` in the generated file. It must implement "
        "`build_model(context)`, `build_optimizer(model, context)`, "
        "`build_train_batch(scenario, device)`, "
        "`build_validation_batch(scenario, device)`, "
        "`training_step(model, batch, context)`, and "
        "`validation_step(model, batch, context)`.",
        "- The adapter must exercise the same real model, loss, inputs, and optimizer "
        "as the training pipeline; do not use mock tensors or a different toy model. "
        "Its batch builders must honor `scenario['batch_size']` and return all inputs "
        "required by the model plus a target.",
        "- `scenario['fixture']` describes input shapes and may omit the target. "
        "Construct the target separately according to the task's target contract "
        "in every fixture and fallback batch path.",
        "- Keep imports, constants, class/function definitions, and read-only device "
        "configuration import-safe. Put all training, validation, prediction, and "
        "submission side effects under `if __name__ == '__main__':`.",
    ]


def run(agent, init_solution_path: Optional[str] = None) -> SearchNode | None:
    """Generate initial draft. If init_solution_path is provided and readable, use file content directly."""
    if init_solution_path:
        try:
            code = Path(init_solution_path).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to read init_solution from {init_solution_path}: {e}, falling back to LLM generation")
            init_solution_path = None
        else:
            plan = "User-provided init solution."
            hardware_ctx = get_hardware_context_for_stage(agent, "draft", code=code)
            if not agent.virtual_root.add_expected_child_count(agent.scfg):
                logger.info("Draft limit reached before init solution could reserve a child slot.")
                return None
            new_node = SearchNode(
                plan=plan,
                code=code,
                parent=agent.virtual_root,
                stage="draft",
                local_best_node=agent.virtual_root,
            )
            apply_hardware_context_to_node(new_node, hardware_ctx)
            lesson_ctx = get_lesson_context_for_stage(agent, "draft", parent_node=new_node, code=code)
            apply_lesson_context_to_node(new_node, lesson_ctx)
            register_node(agent, new_node, "User-provided init solution (no LLM).", new_branch=True)
            logger.info(f"[draft] → node {new_node.id} (branch={new_node.branch_id}) [init_solution]")
            return new_node

    professional_identity = (
        "🏆 You are a Kaggle Grandmaster - a top-tier ML expert competing to WIN.\n\n"
        "**Your Standards**:\n"
        "✓ Design complete ML pipelines (data → model → training → inference)\n"
        "✓ Implement real models that LEARN from data (not baseline scripts with constants)\n"
        "✓ Generate predictions through ACTUAL MODEL INFERENCE on each sample\n"
        "✓ Compete for TOP performance, not trivial baselines\n\n"
        "Your solution will be evaluated on a real leaderboard. Treat this with professionalism.\n\n"
    )

    introduction = (
        professional_identity +
        "Now, let's begin the competition. "
        "You need to come up with an excellent and creative plan for a competitive solution "
        "and then implement this solution in Python with the quality expected of a Kaggle Grandmaster. "
        "We will now provide a description of the task."
    )
    if getattr(agent.acfg, "stop_after_valid_nodes", 0):
        introduction = (
            "Build one compact, real trainable model and a complete executable pipeline "
            "for this feasibility test. Prioritize correct data handling, learned GPU "
            "training, validation, test inference and the required integration contracts. "
            "Avoid unnecessary ensembles or architectural complexity. A modest measured "
            "score is acceptable; fabricated metrics and dummy predictions are not. "
            "This is a local validation experiment, not a leaderboard submission."
        )
    prompt: Any = {
        "Introduction": introduction,
        "Task description": agent.task_desc,
        "Instructions": {},
    }
    hardware_design_brief = get_hardware_design_brief(agent, select_features=False)
    hardware_ctx = get_hardware_context_for_stage(agent, "draft")
    cuda_docs_ctx = get_cuda_docs_context(agent, "draft", hardware_context=hardware_ctx)
    lesson_ctx = get_lesson_context_for_stage(agent, "draft", parent_node=agent.virtual_root)
    knowledge_records = hardware_records(hardware_ctx) + hardware_records(hardware_design_brief)
    knowledge_records += select_records(cuda_records(cuda_docs_ctx), runtime_scope(hardware_ctx)) + history_records(agent)
    knowledge_records += lesson_records(lesson_ctx)
    knowledge_records = select_records(knowledge_records, {}, already_filtered=True)
    from utils.feedback import scheduler_feedback
    if operational := scheduler_feedback(agent):
        prompt["Instructions"]["Scheduler execution constraints"] = operational
    prompt["Instructions"] |= prompt_resp_fmt()
    prompt["Instructions"] |= hardware_context_instructions(hardware_ctx)
    if "Hardware/Profile reasoning rule" in prompt["Instructions"]:
        prompt["Instructions"]["Hardware/Profile reasoning rule"] = [
            rule for rule in prompt["Instructions"]["Hardware/Profile reasoning rule"]
            if "Cross-Stage Note Board" not in rule and "stage-specific hardware node" not in rule
        ]
    prompt["Instructions"] |= lesson_context_instructions(lesson_ctx)
    prompt["Instructions"]["Joint design"] = [
        "Choose data preparation, model, precision, optimizer, training, evaluation and inference together.",
        "Return a brief design followed by one complete runnable Python script; there are no later coding or integration stages.",
        "Keep every applicable knowledge restriction and fallback; family-specific experience is conditional until you choose that family.",
    ]

    prompt["Instructions"] |= {
        "🔬 Critical: Scientific Approach to Design": [
            "",
            "Before designing your solution, you must answer three fundamental questions:",
            "",
            "1. **WHAT makes this task unique?**",
            "   - Not generic observations like 'it's a classification task'",
            "   - What SPECIFIC patterns, challenges, or domain characteristics?",

            "",
            "2. **WHY is your approach suitable for this task?**",
            "   - Not just 'this model is good' - explain the MATCH between approach and task",
            "   - What properties of your method address the task characteristics?",

            "",
            "3. **HOW will you validate your hypothesis?**",
            "   - What outcome would confirm your approach is right?",
            "   - What outcome would suggest you need to reconsider?",

            "",
            "---",
            "",
            "⚠️ This is not a template to fill - this is how scientists think.",
            "Blindly applying standard methods without understanding WHY is not acceptable.",
            "",
            "Your plan should naturally reflect this reasoning process.",
        ],
    }

    prompt["Instructions"] |= {
        "Solution sketch guideline": [
            "- This first solution design should be relatively simple — avoid complex ensemble strategies or extensive hyperparameter searches at this stage.\n",
            "- 🎯 **CRITICAL: NOVELTY & DIVERSITY REQUIREMENT**:\n",
            "  • **Mandatory**: Your solution MUST be NOVEL compared to ALL existing attempts in Memory.\n",
            "  • **Step 1**: Carefully analyze the core idea of EACH previous attempt in Memory.\n",
            "  • **Step 2 - Choose Strategy**:\n",
            "    → **Option A (Preferred)**: Propose a COMPLETELY DIFFERENT approach exploring an untried direction.\n",
            "    → **Option B**: Build upon an existing approach BUT add significant novel insights that fundamentally change the solution.\n",
            "  • **Forbidden**: Minor variations (changing hyperparameters, swapping similar models, tweaking preprocessing).\n",
            "  • **Think**: 'Does my approach explore a fundamentally different hypothesis?' If NO → redesign.\n",
            "- Don't propose the same modelling solution but keep the evaluation the same.\n",
            "- Your plan should be concise but comprehensive: Must address WHAT/WHY/HOW (2-4 sentences each). Avoid verbosity - every sentence should add new insight. Natural length: around 8-12 sentences for a complete reasoning process.\n",
            "- Propose an evaluation metric that is reasonable for this task.\n",
            "- Don't suggest to do EDA.\n",
            "- The data is already prepared in `./input` directory. No need to unzip files.\n",
        ],
        "Coding & Execution Guidelines (CRITICAL)": [
            "- **NO PROGRESS BARS**: You MUST NOT use `tqdm`. Assume `tqdm` is not installed. Use standard Python loops only. Do not use `verbose=1`.",
            "- **MINIMAL LOGGING**: Print ONLY 1 line per epoch (e.g. loss/accuracy). Do NOT print batch-level logs.",
            "- **FINAL OUTPUT**: The VERY LAST line of execution MUST be `print(f'Final Validation Score: {score}')`. This is required for the score parser."
        ],
        "Backend design contract": [
            "- In the plan, explicitly state the selected memory strategy and physical/effective batch elasticity strategy.",
            "- State operator and normalization considerations, the model dimensions that remain configurable, and the fallback behavior.",
            "- List scheduler-owned backend controls that the generated subprocess must not implement.",
            "- Do not assume code-level integration, shared framework state, or synchronized start time with another job.",
        ],
    }
    from engine.preflight import preflight_enabled

    if preflight_enabled(agent.cfg):
        prompt["Instructions"]["CPU preflight adapter contract"] = model_preflight_generation_instructions()
    if getattr(agent, "scheduler_client", None) is None:
        prompt["Instructions"].pop("Backend design contract", None)
    prompt["Instructions"] |= get_impl_guideline_from_agent(agent)
    prompt["Instructions"] |= prompt_leakage_prevention()

    if agent.use_coldstart and (agent.coldstart_description != "None model"):
        coldstart_guideline = [
            f"""
            **Pretrained Model Strategy**:

            • **Option A [RECOMMENDED]**: {agent.coldstart_description}
              → SOTA models with proven performance. Use for end-to-end fine-tuning OR as frozen feature extractors.

            • **Option B**: Alternative pretrained models if better suited to task characteristics.

            • **Option C**: Train from scratch / non-DL methods (only when pretraining provides no advantage).

            **CRITICAL: When using any recommended pretrained model (Option A), you MUST copy the Code template EXACTLY as provided — including model variant names, file paths, and checkpoint filenames. Only the listed weights are available locally; other variants will fail to load. Do NOT invent Kaggle/input paths, dummy checkpoints, or placeholder model files. If a local path is not explicitly shown in the template, choose Option B or C instead.**

            **Key Techniques**:
            1. **Feature Extractor Pattern**: If dataset is small or domain mismatch exists → Freeze backbone + train only final layers (or feed to XGBoost/SVM).

            2. **Avoid Timeouts**: #1 cause is slow data loading, NOT GPU model.
               • Use DataLoader with num_workers>=2, pin_memory=True (NOT raw for loops)
               • For large datasets + heavy backbones: Extract & cache features to disk (.npy/.h5)
            """
        ]
    else:
        coldstart_guideline = [""]

    prompt["Instructions"]["Implementation guideline"].extend(coldstart_guideline)
    prompt["Instructions"] |= get_prompt_environment()
    prompt["Instructions"] |= ROBUSTNESS_GENERALIZATION_STRATEGY

    instructions = f"\n# Instructions\n\n"
    instructions += compile_prompt_to_md(prompt["Instructions"], 2)

    assistant_prefix = f"Let me approach this systematically.\nFirst, I'll examine the dataset:\n{agent.data_preview}"

    def build_prompt(knowledge_section):
        user_prompt = f"\n# Task description\n{prompt['Task description']}\n{knowledge_section}\n{instructions}"
        return build_chat_prompt_for_model(agent.acfg.code.model, introduction, user_prompt, assistant_prefix)

    prompt_complete, knowledge_records, knowledge_diagnostics = fit_prompt(agent, build_prompt, knowledge_records)
    if knowledge_diagnostics["sizing"] == "unavailable":
        logger.info("Draft context sizing unavailable; injecting complete concise records without a fixed cap.")
    if not agent.virtual_root.add_expected_child_count(agent.scfg):
        logger.info("Draft limit reached before draft generation could reserve a child slot.")
        return None

    plan, code = plan_and_code_query(agent, prompt_complete)
    new_node = SearchNode(plan=plan, code=code, parent=agent.virtual_root, stage="draft",
                        local_best_node=agent.virtual_root)
    apply_hardware_context_to_node(new_node, hardware_ctx)
    apply_hardware_design_brief_to_node(new_node, hardware_design_brief)
    apply_lesson_context_to_node(new_node, lesson_ctx)
    new_node.generation_strategy = "single_pass"
    new_node.pipeline_decision = None
    new_node.stage_note_board = []
    new_node.diagnostics["design_knowledge"] = {
        **knowledge_diagnostics,
        "record_ids": [record["record_id"] for record in knowledge_records],
        "evidence_refs": sorted({ref for record in knowledge_records for ref in record["evidence_refs"]}),
    }
    register_node(agent, new_node, prompt_complete, new_branch=True)

    logger.info(f"[draft] → node {new_node.id} (branch={new_node.branch_id})")
    return new_node
