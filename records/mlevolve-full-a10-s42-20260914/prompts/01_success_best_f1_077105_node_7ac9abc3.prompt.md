# Used Prompt 1: model_design

🏆 You are a Kaggle Grandmaster - a top-tier ML expert competing to WIN.

**Your Standards**:
✓ Design complete ML pipelines (data → model → training → inference)
✓ Implement real models that LEARN from data (not baseline scripts with constants)
✓ Generate predictions through ACTUAL MODEL INFERENCE on each sample
✓ Compete for TOP performance, not trivial baselines

Your solution will be evaluated on a real leaderboard. Treat this with professionalism.

Now, let's begin the competition. You need to come up with an excellent and creative plan for a competitive solution and then implement this solution in Python with the quality expected of a Kaggle Grandmaster. We will now provide a description of the task.

You are currently focusing on the 'model_design' step of the solution. Your task is to write ONLY the code for this specific step, not the complete solution.


# Task description
# Disaster Tweets: Custom Precision Benchmark

Predict target=1 for tweets about real disasters, target=0 otherwise. Use text, keyword and location as appropriate. The metric is binary F1. Write submission.csv with id,target columns and binary predictions for test.csv. Use only train.csv labels for model selection, with your own internal validation split.

The original labeled training data was split 80/20 with stratification and seed 42. Test labels are unavailable to the agents.

============================================================
**REQUIRED SUBMISSION FORMAT**
============================================================
The final submission file must match this format:

  id  target
6925       0
1975       0
5034       0
 254       0
8370       0

(Showing first 5 rows as example)
============================================================
**Submission File Location**: Must save the submission to `./submission/submission.csv`
.


# Memory
Below is a record of previous solution attempts and their outcomes:
 Summary: 3 total attempts - 3 executed, 2 successful (best: 0.7573)

-------------------------------
Attempt #1:
Design: I merged the three stages by keeping Stage 1's data cleaning, TF-IDF feature builders, `SparseMLP`, `build_loss`, `tune_threshold` and split helpers as the foundation; layering Stage 2's import-safe precision utilities (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `degrade_to_fp32`, `configure_tf32`, `format_precision_log`) on top; then consuming both inside Stage 3's `run_training` with scheduler safe-points, `TrainingDiagnostics`, linear warmup+cosine AdamW, best-val-F1 restore, and test inference. I resolved the duplicate `CandidateAdapter` by keeping Stage 3's self-contained CPU-safe version (it uses the Stage 1 model/loss/features through the Stage 2 precision path), removed duplicate imports/constants, kept all side effects under `if __name__ == "__main__":`, and preserved the exact `MODEL_FAMILY`, `WORKING_DIR`, `SUBMISSION_DIR` contract.
Results: The training script executed successfully end-to-end: TF-IDF features were fit on the train split, the SparseMLP trained under CUDA FP16 AMP with TF32 disabled, early stopping triggered after 4 epochs without improvement, and a best-val-F1 checkpoint was restored for test inference and submission writing. The run produced a finite best validation F1 of 0.7573 (peak at epoch 1), with the final score line printed and a submission.csv generated; the only non-fatal log noise was an unrelated Neo4j connection refusal.
Validation Metric: 0.7573
Execution Time: 925.14s

-------------------------------
Attempt #2:
Design: I merged the three stages by keeping Stage 1's data/model/loss/adapter code as the foundation, layering Stage 2's `PrecisionRuntime` policy helpers on top, and consuming both inside Stage 3's `run_training` (with diagnostics + scheduler safe points). I resolved naming conflicts, deduplicated imports, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module-level constants that Stage 3 references, and moved every side effect under `if __name__ == '__main__':` while preserving the CPU-safe `CandidateAdapter` contract.
Results: Training completed successfully in ~37 minutes on an A10 GPU, running 12 epochs with a hybrid transformer model. Validation F1 improved from 0.665 at epoch 1 to a best of ~0.7341 (best epoch 11), and a valid submission file with 1523 rows was written using a tuned threshold of 0.42. The only messages were benign warnings (Graph evidence mirror connection refused, nested-tensor warning) that did not affect execution.
Validation Metric: 0.7341
Execution Time: 2223.42s

-------------------------------
Attempt #3:
Design: I merged the three stages by keeping Stage 1's data pipeline, hybrid BiGRU+CharCNN model, loss helpers, and CPU-safe `CandidateAdapter` as the foundation; layering Stage 2's import-safe precision utilities (`PrecisionConfig`, `resolve_precision_config`, `get_autocast_context`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `clip_gradients`, `degrade_to_fp32`, `is_finite_loss`, `configure_backend_precision`, `format_precision_log`, `precision_to_dict`) on top; and consuming both inside Stage 3's `run_training` with diagnostics, linear warmup + cosine AdamW, best-val-F1 restore, and test inference. I resolved duplicate imports/constants, kept `MODEL_FAMILY`, `WORKING_DIR`, and `SUBMISSION_DIR` as shared module constants, removed the meaningless placeholder arg in the adapter's unpack call, and moved all side effects inside `if __name__ == "__main__":`.
Results: The implementation of this design has bugs.
Insight: Using a different approach may not result in the same bugs as the above approach.


# Hardware-Aware Stage 1 Candidate Construction Context
# Hardware-Aware Model Design Brief
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2; observed in 2 batch-size observations
- Precision optimization mode: normal on ampere (compute capability 8.6).
- Allowed native training precision policies: fp32, disabled, fp16_amp
- FP16 AMP is optional, not a whole-pipeline cast: FP32 parameters/inputs, local autocast for forward, FP32-sensitive losses/reductions and metrics/export; GradScaler and backward outside autocast. Disable TF32; use FP32 on CPU. Retain FP32 wherever AMP is unstable, unsupported or slower.
- Precision evidence rule: recommend only a native Tensor Core format with a supported forward/backward training path; generic FP4 is not NVFP4 and FP6 is not an MLEvolve training policy.
- Model candidate: transformer (score=0.05, confidence=0.0): transformer is a baseline-compatible option for transformer_training, but no strong hardware-specific evidence was found.
- Model candidate: small_transformer (score=0.05, confidence=0.0): small_transformer is a baseline-compatible option for transformer_training, but no strong hardware-specific evidence was found.
- Model candidate: lora_transformer (score=0.05, confidence=0.0): lora_transformer is a baseline-compatible option for transformer_training, but no strong hardware-specific evidence was found.
- Decision rule: Choose the architecture that best satisfies the task metric while using hardware features to reduce training time and improve GPU utilization. If hardware evidence is weak, keep a conservative baseline-compatible architecture.
- Feature-detail rule: The feature key list is an index. Use selected hardware feature details already shown above, or query local feature-node details for chosen keys when deeper implementation guidance is needed.
- Constraint rule: Hardware advice never overrides task, dataset, submission, package, model-source, or filesystem constraints. Do not invent Kaggle input paths, local checkpoints, torch hub directories, or placeholder weights; use only paths and model sources that are explicitly available in the prompt/config.
- Budget guardrail: For hardware/runtime optimization, preserve the parent solution's modeling budget unless the user explicitly asks to improve score. Do not increase epochs, folds, model size, input resolution, ensemble count, TTA, dataset size, or validation workload as a hardware-only optimization.
# Hardware/Profile Optimization Context
  - model_design
    - filter_audit: features_shown=6/6
  - source=hardware_knowledge_graph.json; linked_features=6
- Workflow stage: Stage 1 model-design.
- Stage ownership: hardware context lookup, data processing, feature engineering, model architecture, loss, and output contract. Do not choose dtype/AMP policy, optimizer, batch size, scheduler, or training loop here.
- Hardware node response rule: the current hardware node response and compact feature-key pairs are already attached below.
- Query rule: do not query the hardware node again; query local feature-node details only for selected feature keys when deeper implementation guidance is needed.
- Source rule: do not fetch external URLs or rely on source links in generated reasoning.
- Scheduler backend config: packing_backend=cuda_process, effective_backend=cuda_process, runner_contract=subprocess_job_v1, enabled_backends=['exclusive', 'mps_process', 'cuda_process'], exclusive_fallback_enabled=True



# Pipeline Decision Contract
Before writing or modifying code, follow the hardware-aware stepwise workflow: model-design -> datatype/quantization -> training (code keys: model_design -> datatype_precision -> training_evaluation).

1. Model design: load/prepare data, create features/splits, and choose architecture/model family, loss, criterion, and output interface first.
2. Datatype/precision: choose only mode-allowed operations, autocast regions, FP32-sensitive regions, scaling and fallback behavior. A policy is not a whole-pipeline dtype.
3. Training: choose optimizer, scheduler, batch size, dataloader settings, checkpointing, validation, submission, runtime fallbacks, and logging.
Datatype precision may include only precision-required model adapters that preserve the Stage 1 model family, loss, data features, and output interface.
Hardware-only training/runtime tuning must not increase epochs, folds, model size, image resolution, ensemble count, TTA, dataset size, or validation workload.
If evidence is missing, use the fallback recorded in the trace instead of inventing hardware claims.

# Pipeline Decision Trace
```json
{
  "model_design": {
    "modality": "text",
    "target_type": "classification",
    "shape_constraints": [
      "sequence length and tokenizer output must stay configurable"
    ],
    "family": "available_transformer_or_sequence_baseline",
    "alternatives_considered": [],
    "loss": "binary cross-entropy with logits",
    "output_interface": "single logit per example; sigmoid threshold selected on internal validation to maximize binary F1; submission columns id,target with int binary predictions",
    "reason": "Use a baseline-compatible model family until stronger task or evidence signals are available.",
    "hardware_fit": "use only compatible evidence"
  },
  "datatype_precision": {
    "precision_policy": "fp16_amp",
    "precision_model_adaptation": "none",
    "fallback_policy": "fall back to fp32 or disabled AMP if the selected precision path is unsupported",
    "reason": "Use conservative precision until hardware and package support are confirmed."
  },
  "training_evaluation": {
    "optimizer": "AdamW over FP32 parameters",
    "scheduler": "linear warmup with linear/cosine decay",
    "batch_size_policy": "fixed",
    "dataloader_policy": "num_workers>0, pin_memory=True, persistent_workers when reused, non_blocking=True device copies",
    "fallbacks": [
      "disable AMP and run FP32 if non-finite loss/gradients or slower/less accurate",
      "retain FP32 for unsupported or unstable operations"
    ],
    "metrics_to_log": [
      "elapsed_seconds",
      "peak_vram_mb",
      "resolved_batch_size"
    ],
    "advanced_optimizer_used": false,
    "reason": "Choose stable training defaults matched to the inferred target type and available model family."
  },
  "evidence": {
    "hardware_context_used": true,
    "evidence_refs": [
      "backend_guidance:cuda_process.no_mps_controls",
      "backend_guidance:cuda_process.normal_process_throughput",
      "backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry",
      "backend_guidance:shared.subprocess.task_quality_and_elasticity"
    ],
    "confidence": 0.95,
    "missing_evidence": [
      "predictor/graph evidence not available",
      "no exact/similar/packed graph profiles for current hardware",
      "no measured FP16 vs FP32 comparison yet"
    ]
  }
}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- No prior notes yet. This step is part of Stage 1 model-design; write concise notes for later stages.

# Previous steps
This is the first step, no previous steps.

# Current step: model_design
Load and prepare data, create features/splits, then define the model architecture, loss function, output interface, and criterion.


# Instructions

## Response format

Your response should be:
1. A brief plan (2-3 sentences) describing what you will do in this step
2. A `Note board:` block with 1-3 bullets. Each bullet must use this exact shape: `- baseline_change: <what changed from the baseline or prior step> | purpose: <why this supports the candidate target> | hardware_keys: <comma-separated local feature keys or none>`
3. A single markdown code block (wrapped in ```) containing ONLY the code for this step
IMPORTANT: Do NOT write code for other steps. Only write code for the current step.
No module-level side effects: this step may define imports, constants, classes, and helper functions only. Do not access data, read environment settings, configure CUDA/TF32, configure logging, instantiate models/losses, construct DataLoaders, train, evaluate, or write submissions at import time; expose helpers for the final main guard instead.

## Hardware/Profile reasoning rule

- HWDB recommended_patterns are conditional candidates, not measured speedups or mandatory steps. Check workload, installed backend, allowed precision and each stated limitation before applying one. A capability or verified flag does not prove benefit for this candidate. Keep the baseline when evidence is missing; record feature keys, unmet conditions and measured results without inventing them.
- Hardware advice never overrides task, dataset, submission, package, model-source, or filesystem constraints. Do not invent Kaggle input paths, local checkpoints, torch hub directories, or placeholder weights; use only paths and model sources that are explicitly available in the prompt/config.
- For hardware/runtime optimization, preserve the parent solution's modeling budget unless the user explicitly asks to improve score. Do not increase epochs, folds, model size, input resolution, ensemble count, TTA, dataset size, or validation workload as a hardware-only optimization.
- Generated code must run as one independent subprocess and follow the exact effective backend named in the backend-guidance section.
- Use backend coding preferences in job code, but leave daemon setup, launch policy, device assignment, active-thread allocation, priorities, and memory limits to the scheduler.
- Prefer scheduler-compatible code with configurable batch size, precision, dataloader workers, checkpoints, and runtime logging when using GPU training.
- The stage-specific hardware node response is already provided in the prompt. Do not query the hardware node again; query local feature-node details only for selected feature keys when deeper implementation guidance is needed.
- Use the Cross-Stage Note Board to keep Stage 2 precision and Stage 3 training choices aligned with the Stage 1 candidate target.


## Pipeline Decision Contract

- Use the Pipeline Decision Trace as the source of truth for code generation and planning.
- Follow the hardware-aware stepwise workflow: model-design -> datatype/quantization -> training (code keys: model_design -> datatype_precision -> training_evaluation).
- Do not jump directly to optimizer, precision, or batch-size choices before model family and output interface are decided.
- The model_design stage stores data modality, target shape, preprocessing/features, model family, loss, criterion, and output interface.
- Numeric precision such as fp32, fp16, bf16, tf32, or TE FP8/NVFP4 belongs under datatype_precision.precision_policy unless the task explicitly requires a numeric type.
- The datatype_precision step may make narrow precision-required model adapters such as Transformer Engine layer wrappers/replacements, padding/config hooks, autocast recipes, or higher-precision islands, but it must preserve the Stage 1 model family, loss, data features, and output interface.
- Hardware/profile evidence may influence model fit and tuning only when it is compatible with the task, installed packages, and available model sources.
- Hardware-only training/runtime tuning must not increase epochs, folds, model size, image resolution, ensemble count, TTA, dataset size, or validation workload.
- If execution feedback contradicts an earlier decision, update only the relevant part of the decision and explain why in the plan.


## 🔬 Critical: Scientific Approach to Design

- 
- Before designing your solution, you must answer three fundamental questions:
- 
- 1. **WHAT makes this task unique?**
- - Not generic observations like 'it's a classification task'
- - What SPECIFIC patterns, challenges, or domain characteristics?
- 
- 2. **WHY is your approach suitable for this task?**
- - Not just 'this model is good' - explain the MATCH between approach and task
- - What properties of your method address the task characteristics?
- 
- 3. **HOW will you validate your hypothesis?**
- - What outcome would confirm your approach is right?
- - What outcome would suggest you need to reconsider?
- 
- ---
- 
- ⚠️ This is not a template to fill - this is how scientists think.
- Blindly applying standard methods without understanding WHY is not acceptable.
- 
- Your plan should naturally reflect this reasoning process.


## Solution sketch guideline

- - This first solution design should be relatively simple — avoid complex ensemble strategies or extensive hyperparameter searches at this stage.
- - 🎯 **CRITICAL: NOVELTY & DIVERSITY REQUIREMENT**:
- • **Mandatory**: Your solution MUST be NOVEL compared to ALL existing attempts in Memory.
- • **Step 1**: Carefully analyze the core idea of EACH previous attempt in Memory.
- • **Step 2 - Choose Strategy**:
- → **Option A (Preferred)**: Propose a COMPLETELY DIFFERENT approach exploring an untried direction.
- → **Option B**: Build upon an existing approach BUT add significant novel insights that fundamentally change the solution.
- • **Forbidden**: Minor variations (changing hyperparameters, swapping similar models, tweaking preprocessing).
- • **Think**: 'Does my approach explore a fundamentally different hypothesis?' If NO → redesign.
- - Don't propose the same modelling solution but keep the evaluation the same.
- - Your plan should be concise but comprehensive: Must address WHAT/WHY/HOW (2-4 sentences each). Avoid verbosity - every sentence should add new insight. Natural length: around 8-12 sentences for a complete reasoning process.
- - Propose an evaluation metric that is reasonable for this task.
- - Don't suggest to do EDA.
- - The data is already prepared in `./input` directory. No need to unzip files.


## Coding & Execution Guidelines (CRITICAL)

- - **NO PROGRESS BARS**: You MUST NOT use `tqdm`. Assume `tqdm` is not installed. Use standard Python loops only. Do not use `verbose=1`.
- - **MINIMAL LOGGING**: Print ONLY 1 line per epoch (e.g. loss/accuracy). Do NOT print batch-level logs.
- - **FINAL OUTPUT**: The VERY LAST line of execution MUST be `print(f'Final Validation Score: {score}')`. This is required for the score parser.


## Backend design contract

- - In the plan, explicitly state the selected memory strategy and physical/effective batch elasticity strategy.
- - State operator and normalization considerations, the model dimensions that remain configurable, and the fallback behavior.
- - List scheduler-owned backend controls that the generated subprocess must not implement.
- - Do not assume code-level integration, shared framework state, or synchronized start time with another job.


## CPU preflight adapter contract

- - **CPU PREFLIGHT CONTRACT (MANDATORY)**: Define a no-argument `class CandidateAdapter` in the generated file. It must implement `build_model(context)`, `build_optimizer(model, context)`, `build_train_batch(scenario, device)`, `build_validation_batch(scenario, device)`, `training_step(model, batch, context)`, and `validation_step(model, batch, context)`.
- - The adapter must exercise the same real model, loss, inputs, and optimizer as the training pipeline; do not use mock tensors or a different toy model. Its batch builders must honor `scenario['batch_size']` and return all inputs required by the model plus a target.
- Batch builders must return (inputs, target) or a mapping with recognized `inputs` and `target` keys. For multiple positional model inputs, use {'inputs': (word_ids, char_ids, keyword_ids), 'target': targets}; training_step and validation_step must unpack those same inputs into the real model. Custom top-level keys alone (such as word_ids) are not recognized. Preserve any authoritative named fixture keys, adding an inputs tuple that references those same tensors when needed. Token/embedding indices remain torch.long and masks keep their required dtype; FP32 precision rules apply to floating-point computation, not integer indices. The checker owns zero_grad, backward and optimizer.step: training_step only computes and returns the connected scalar loss, without performing those updates itself.
- - Keep imports, constants, class/function definitions, and read-only device configuration import-safe. Put all training, validation, prediction, and submission side effects under `if __name__ == '__main__':`.


## Implementation guideline

- Milestone objective: complete one real candidate through review, CPU preflight, GPU training, validation and test submission. There is no overall wall-clock deadline or failed-node budget. Max execution time per run = an hour. Start with a compact trainable model and a complete executable pipeline; do not fabricate metrics, bypass checks, or use dummy predictions.
- 
- Code execution must complete within the configured per-run limit of an hour, including validation and test inference.
- 🎯 **CRITICAL REQUIREMENTS** (Non-Negotiable):
- 
- **1. Model Inference for ALL Predictions**
- • EVERY prediction (validation & test) MUST come from trained model's forward pass
- • Process: Load data → Preprocess → model.predict()/model.forward() → Save predictions
- • ❌ FORBIDDEN: Constants, placeholders, dummy values, empty arrays, statistics, random numbers
- • ❌ FORBIDDEN: Fake/mock metric functions (must use real sklearn.metrics or correct manual implementation)
- • Why: Shortcuts create fake high validation scores but fail on test (CRITICAL SYSTEM FAILURE)
- 
- **2. Generate submission.csv**
- • Path: `./submission/submission.csv` (NOT ./working/submission.csv)
- • Content: Model predictions on ALL test samples
- • Format: Follow task description exactly
- 
- **3. Print Validation Metric**
- • MUST print: `print(f'Final Validation Score: {score}')`
- • Score MUST be computed on hold-out validation set using proper metric formula
- • CRITICAL CONSISTENCY REQUIREMENT: Ensure that validation and test inference use IDENTICAL processing logic. Any differences in how validation and test data are handled (such as post-processing, reconstruction, or formatting) can cause large performance gaps between validation and test sets. Maintain consistency across all data processing steps for both validation and test phases.
- 
- **4. Scheduler Model Family Contract**
- • MUST define a top-level constant `MODEL_FAMILY = "family_name"` near the top of the file.
- • Use a stable, architecture-specific name, for example `resnet50_224`, `swin_b_384`, or `lightgbm_tabular_v1`.
- • If you switch the model architecture or input geometry in an improvement/evolution, update `MODEL_FAMILY` to a new distinct value.
- • If there is no more specific model/backbone name in the script, set the model name/key variable to the same value as `MODEL_FAMILY`.
- 
- **5. CPU Model-Preflight Adapter Contract**
- • The module MUST be import-safe: definitions and lightweight constants may be top-level, but all training, validation, inference, and submission execution MUST be called only inside `if __name__ == "__main__":`.
- • MUST define a no-argument top-level `CandidateAdapter` class with methods `build_model(context)`, `build_optimizer(model, context)`, `build_train_batch(scenario, device)`, `build_validation_batch(scenario, device)`, `training_step(model, batch, context)`, and `validation_step(model, batch, context)`. build_model(context) MUST return the real torch.nn.Module used by training, never a dict, tuple, or metadata wrapper.
- • `training_step` MUST return the real scalar loss tensor connected to model parameters; adapter methods must reuse the script's real model, loss, transforms, collate logic, and optimizer rather than mocks.
- • Treat every caller-supplied `context` as a partial mapping: merge it over adapter defaults before reading optional keys, and preserve caller-provided values.
- • CandidateAdapter context mutations do not persist between checker method calls; `training_step` and `validation_step` MUST resolve the script's real criterion when the partial context omits it or supplies `None`, rather than relying on `build_model` to mutate a local context.
- • Batch builders MUST honor `scenario['batch_size']` and `device`; use `os.environ.get('MLEVOLVE_INPUT_DIR', './input')` to locate input fixtures during the isolated CPU check.
- Batch builders must return (inputs, target) or a mapping with recognized `inputs` and `target` keys. For multiple positional model inputs, use {'inputs': (word_ids, char_ids, keyword_ids), 'target': targets}; training_step and validation_step must unpack those same inputs into the real model. Custom top-level keys alone (such as word_ids) are not recognized. Preserve any authoritative named fixture keys, adding an inputs tuple that references those same tensors when needed. Token/embedding indices remain torch.long and masks keep their required dtype; FP32 precision rules apply to floating-point computation, not integer indices. The checker owns zero_grad, backward and optimizer.step: training_step only computes and returns the connected scalar loss, without performing those updates itself.
- • When `scenario['fixture']` is non-empty, it is the authoritative named input contract: build every listed tensor with the declared non-batch shape, preserve those names in the returned mapping, and pass all model inputs to the real forward path. Do not collapse a multimodal fixture into one feature vector.
- • Adapter construction and a one-batch CPU step must be lightweight and must not download weights or require CUDA.
- 
- 📁 **Directories**: Input data in `./input/`, submission in `./submission/`, temp files in `./working/`
- 
- 📦 **Packages & Internet**: Guaranteed installed packages: numpy, pandas, scikit-learn, statsmodels, xgboost, lightgbm, torch, torchvision, torch-geometric, bayesian-optimization, timm, transformers, sentence-transformers, opencv-python, Pillow. Do not assume unlisted third-party packages are installed; avoid them or provide an ImportError-guarded fallback. Never run pip from generated code. Generated code runs offline: do not download weights or data through torch.hub.load(), Hugging Face, timm, or another remote loader. Use only verified local or cached weights.
- 
- ⚠️ **API Compatibility**: LightGBM/XGBoost: ❌ `fit(..., early_stopping_rounds=...)` → ✅ LightGBM: `fit(..., callbacks=[lgb.early_stopping(...)])` ✅ XGBoost: `XGBClassifier(early_stopping_rounds=...)`
- • AdamW: ❌ `from transformers import AdamW` (deprecated) → ✅ `from torch.optim import AdamW`
- 
- 🚫 **Execution Guidelines**:
- • NO tqdm (not installed), NO verbose=1
- • Print only 1 line per epoch (minimize logging)
- • Runtime diagnostics are an exception: one diagnostics line per epoch plus context entry/exit; no per-batch printing.
- PyTorch runtime diagnostics: import TrainingDiagnostics from utils.training_diagnostics. Inside the training entrypoint, wrap training in `with TrainingDiagnostics(model, optimizer, scaler=scaler_or_None, settings={...}) as diagnostics:` after model/optimizer initialization. Settings must include actual physical_batch_size, effective_batch_size, planned_epochs, steps_per_epoch, model_family, split_seed and precision choice/reason. Call diagnostics.after_update() once after each optimizer update ATTEMPT, after scaler.step and scaler.update (or optimizer.step); never after accumulation microbatches. The helper observes real optimizer steps, including skips, and actual autocast dtype. Call diagnostics.report(epoch=one_based_epoch) once per epoch; keep the normal epoch metric line. Save diagnostics.state_dict() with the training checkpoint and restore it using diagnostics.load_state_dict(...). Preserve these calls during merges/repairs. Do not replace missing measurements with zero. Keep the final validation score as the last line, after leaving the diagnostics context. These diagnostics also run without the scheduler.
- • Use DataLoader with num_workers>=2 for speed
- 
- ⚠️  **Self-Check Before Finalizing**:
- □ Did predictions pass through model's learned weights during inference? (If NO → INVALID)
- □ Did I generate submission.csv in correct path with ALL test predictions?
- □ Did I print validation metric as the last line?
- □ Did I use the COMPLETE training dataset (not a tiny subset)?
- □ Is execution behind the main guard and is CandidateAdapter complete and CPU-safe?
- □ Does scheduled PyTorch training use script_scheduler_context, load_resume_checkpoint, and STEP/EPOCH safe_point calls with steps_per_epoch and a complete state_factory? Preserve these hooks during merges and repairs; keep setup inside the training entrypoint and hooks inactive when the context is None.
- Use the installed API exactly: context.control_hook.safe_point(SafePointType.STEP, epoch=epoch, global_step=global_step, steps_per_epoch=steps_per_epoch, state_factory=state_factory). Use the same keyword arguments for BEFORE_TRAIN and EPOCH. There is no payload= parameter and no second positional argument. Import script_scheduler_context from localml_scheduler.execution.script_context and SafePointType from localml_scheduler.domain. Guard hooks with `if context is not None`; do not catch and suppress hook/checkpoint exceptions. Define state_factory to capture actual model, optimizer, scaler, LR scheduler, RNG, global_step and exact step_in_epoch/sampler_state/data_state. Restore those same state objects and loader position before continuing. Use direct hook calls and TrainingDiagnostics construction so the static admission gate can verify the contract; do not add unreachable token-matching calls.
- 
- The code for this step must be self-contained and can be integrated with other steps.
- Use clear variable names that are consistent with previous steps.
- Do not duplicate code from previous steps - assume those parts already exist.
- Make sure to handle edge cases appropriately.


## Precision policy

- Normal mode permits FP32 and selective FP16 AMP only, not BF16, TF32, quantization or TE recipes. FP16 is an option, not a requirement or a whole-pipeline dtype. Keep model parameters and floating inputs and floating regression targets in FP32; preserve integer indices and labels. Use CUDA autocast only around eligible forward operations and compatible loss computation. Run backward and optimizer updates outside autocast, with GradScaler for FP16 and unscale before gradient clipping. For fragile losses/reductions, disable autocast locally and explicitly cast inputs to FP32. Keep metric accumulation and prediction export in FP32; validation/test forwards may share a validated autocast path. Do not call model.half() or cast the entire batch to half. CPU preflight uses FP32 with AMP/scaling disabled. Disable TF32 for matmul and convolution. Compare finite loss/gradients, task quality and elapsed time against FP32; retain FP32 for unsupported, unstable or slower operations. Record the actual regions and fallback, not an unmeasured speedup.


## 🚨 DATA LEAKAGE PREVENTION

- 
- ⚠️ **Strict Isolation Principle**: Validation/Test data must remain strictly unseen during training.
- 
- ✅ **Sequence**: Always **Split Data FIRST**, then apply processing.
- ✅ **Stateful Transformations**: Fit all Scalers, Encoders, Imputers, and Tokenizers **ONLY on Training data**, then use `.transform()` on Validation/Test.
- ✅ **Feature Engineering**: Calculate global statistics (e.g., mean, variance, vocabulary) solely from the Training set.
- ✅ **Target Leakage**: Never use target information (e.g., Target Encoding) from the validation set.
- 


## Installed Packages

The following packages are guaranteed in the execution environment: `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `xgboost`, `lightgbm`, `torch`, `torchvision`, `torch-geometric`, `bayesian-optimization`, `timm`, `transformers`, `sentence-transformers`, `opencv-python`, `Pillow`. Do not assume any unlisted third-party package is installed: either avoid it or use an ImportError-guarded fallback. Do not run pip from generated code. For neural networks we suggest using PyTorch rather than TensorFlow.

## 💡 Recommendation: Robustness & Generalization Strategy

- 
- **To improve model robustness and generalization on unseen data:**
- 
- ✅ **Architecture**: Match model inductive bias to data structure (e.g., CNNs/ViTs for spatial grids, Transformers/RNNs for sequences, GNNs/GCNs for graphs/topology)
- ✅ **Input Strategy**: Handle variable-length or large-scale inputs via **windowing strategies** or patch-based processing (consider overlap for smoother predictions)
- ✅ **Regularization**: Consider using Dropout, Batch/Layer Norm, Weight Decay, or Label Smoothing
- ✅ **Loss Function**: Inspect class distribution and adapt loss accordingly (e.g., weighted loss, FocalLoss, or task-specific objectives)
- ✅ **Learning Rate**: Consider using adaptive schedules like Cosine Annealing or ReduceLROnPlateau or Warmup with differential rates if needed
- ✅ **Data Augmentation**: Apply domain-appropriate augmentation based on data modality (e.g., geometric transforms, masking, mixup)
- ✅ **Validation**: Monitor validation metrics strictly and use early stopping to prevent overfitting
- 
- ⚠️ **Note**:
- Prioritize capturing the intrinsic structure of the data (Inductive Bias) over simply increasing model size.
- 


## model_design guidelines

- - Your responsibility: Stage 1 hardware-aware model-design candidate construction. Load data from `./input`, clean it, create features/preprocessing/augmentation, create train/validation/test splits, then choose the model architecture or available pretrained model family, loss function, output interface, criterion, and model-local regularization.
- CRITICAL: This is the only Stage 1 prompt. It owns both data preparation and model design; do not assume a separate data-preparation prompt exists.
- CRITICAL: Do NOT write the training loop, optimizer, scheduler, batch size, epoch count, learning rate, final dataloader worker settings, gradient accumulation, checkpoint cadence, or precision policy.
- IMPORTANT: Consider the task's evaluation metric when designing the model. The model output format should be compatible with the required evaluation metric calculation.
- Hardware-aware model design: use the Hardware-Aware Model Design Brief to compare model families before choosing an architecture. Optimize for the task metric first, while treating lower training time and higher GPU utilization as persistent objectives.
- Use the brief's compact hardware node, available feature keys, and selected feature details only when they are relevant to data shape, preprocessing, architecture, layer choice, tensor-core-friendly dimensions, or model-family feasibility.
- Hardware-aware data pipeline: when using GPU training, keep input resolution/sequence length configurable, expose dataset/dataloader inputs that Stage 3 can batch safely, and prefer data representations compatible with non-blocking CUDA transfers when tensors move to CUDA.
- Hardware graph contract: the stage-specific hardware node response is already attached in the Hardware/Profile Optimization Context. Do not query the hardware node again; query local feature-node details only for selected feature keys when deeper guidance is needed, and never fetch external URLs.
- Note board: Stage 1 notes are mandatory. Record what changed from the baseline data/model design and the purpose, because later stages must align precision and training choices to this target.
- Do not invent pretrained checkpoint paths, Kaggle input directories, torch hub directories, or dummy model weights to satisfy hardware advice. If a model source is not explicitly available, choose an available baseline-compatible model family.
- Pipeline decision: consume `model_design.modality`, `model_design.target_type`, `model_design.shape_constraints`, `model_design.family`, and `model_design.output_interface` to define data assumptions, preprocessing/features, model family, loss, output interface, and criterion only; leave optimizer, scheduler, precision, and batch policy to later hardware-aware stages.
- Pipeline decision: use `training_evaluation.dataloader_policy` only to expose compatible dataset/dataloader hooks; do not choose final workers, batch size, optimizer, or scheduler in this stage.
- Pipeline decision: do not use unavailable weights, packages, or hardware-only tricks to justify a model family.
- Normal mode permits FP32 and selective FP16 AMP only, not BF16, TF32, quantization or TE recipes. FP16 is an option, not a requirement or a whole-pipeline dtype. Keep model parameters and floating inputs and floating regression targets in FP32; preserve integer indices and labels. Use CUDA autocast only around eligible forward operations and compatible loss computation. Run backward and optimizer updates outside autocast, with GradScaler for FP16 and unscale before gradient clipping. For fragile losses/reductions, disable autocast locally and explicitly cast inputs to FP32. Keep metric accumulation and prediction export in FP32; validation/test forwards may share a validated autocast path. Do not call model.half() or cast the entire batch to half. CPU preflight uses FP32 with AMP/scaling disabled. Disable TF32 for matmul and convolution. Compare finite loss/gradients, task quality and elapsed time against FP32; retain FP32 for unsupported, unstable or slower operations. Record the actual regions and fallback, not an unmeasured speedup.



Let me approach this systematically.
First, I'll examine the dataset:
```
input/
    description.md (5 lines)
    sample_submission.csv (1524 lines)
    test.csv (1724 lines)
    train.csv (6839 lines)
submission/

working/
```

-> input/description.md has content:

```
# Disaster Tweets: Custom Precision Benchmark

Predict target=1 for tweets about real disasters, target=0 otherwise. Use text, keyword and location as appropriate. The metric is binary F1. Write submission.csv with id,target columns and binary predictions for test.csv. Use only train.csv labels for model selection, with your own internal validation split.

This is a custom benchmark, not an official MLE-bench competition definition or Kaggle leaderboard evaluation. The original labeled Kaggle training data was split 80/20 with stratification and seed 42. Both precision modes use this same holdout; test labels are unavailable to the agents. No leaderboard submission is made.

```

-> input/sample_submission.csv has 1523 rows and 2 columns.
⚠️  IMPORTANT: This is the CORRECT submission format that must be followed!
The exact column names are: id, target
Any description.md format information should be IGNORED if it conflicts with this file.
Here is some information about the columns:
id (int64) has range: 5.00 - 10873.00, 0 nan values
target (int64) has 1 unique values: [0]

-> input/test.csv has 1523 rows and 4 columns.
Here is some information about the columns:
id (int64) has range: 5.00 - 10873.00, 0 nan values

-> input/train.csv has 6090 rows and 5 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 10872.00, 0 nan values
target (int64) has 2 unique values: [1, 0]

        ⚠️  CRITICAL SUBMISSION FORMAT NOTE:
        - If you see sample_submission.csv or similar files, those contain the CORRECT submission format
        - The column names in these files are the FINAL AUTHORITY for submission format
        - Always use the column names from the actual sample submission files
        

# Used Prompt 2: datatype_precision

🏆 You are a Kaggle Grandmaster - a top-tier ML expert competing to WIN.

**Your Standards**:
✓ Design complete ML pipelines (data → model → training → inference)
✓ Implement real models that LEARN from data (not baseline scripts with constants)
✓ Generate predictions through ACTUAL MODEL INFERENCE on each sample
✓ Compete for TOP performance, not trivial baselines

Your solution will be evaluated on a real leaderboard. Treat this with professionalism.

Now, let's begin the competition. You need to come up with an excellent and creative plan for a competitive solution and then implement this solution in Python with the quality expected of a Kaggle Grandmaster. We will now provide a description of the task.

You are currently focusing on the 'datatype_precision' step of the solution. Your task is to write ONLY the code for this specific step, not the complete solution.


# Task description
# Disaster Tweets: Custom Precision Benchmark

Predict target=1 for tweets about real disasters, target=0 otherwise. Use text, keyword and location as appropriate. The metric is binary F1. Write submission.csv with id,target columns and binary predictions for test.csv. Use only train.csv labels for model selection, with your own internal validation split.

The original labeled training data was split 80/20 with stratification and seed 42. Test labels are unavailable to the agents.

============================================================
**REQUIRED SUBMISSION FORMAT**
============================================================
The final submission file must match this format:

  id  target
6925       0
1975       0
5034       0
 254       0
8370       0

(Showing first 5 rows as example)
============================================================
**Submission File Location**: Must save the submission to `./submission/submission.csv`
.


# Memory
Below is a record of previous solution attempts and their outcomes:
 Summary: 3 total attempts - 3 executed, 2 successful (best: 0.7573)

-------------------------------
Attempt #1:
Design: I merged the three stages by keeping Stage 1's data cleaning, TF-IDF feature builders, `SparseMLP`, `build_loss`, `tune_threshold` and split helpers as the foundation; layering Stage 2's import-safe precision utilities (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `degrade_to_fp32`, `configure_tf32`, `format_precision_log`) on top; then consuming both inside Stage 3's `run_training` with scheduler safe-points, `TrainingDiagnostics`, linear warmup+cosine AdamW, best-val-F1 restore, and test inference. I resolved the duplicate `CandidateAdapter` by keeping Stage 3's self-contained CPU-safe version (it uses the Stage 1 model/loss/features through the Stage 2 precision path), removed duplicate imports/constants, kept all side effects under `if __name__ == "__main__":`, and preserved the exact `MODEL_FAMILY`, `WORKING_DIR`, `SUBMISSION_DIR` contract.
Results: The training script executed successfully end-to-end: TF-IDF features were fit on the train split, the SparseMLP trained under CUDA FP16 AMP with TF32 disabled, early stopping triggered after 4 epochs without improvement, and a best-val-F1 checkpoint was restored for test inference and submission writing. The run produced a finite best validation F1 of 0.7573 (peak at epoch 1), with the final score line printed and a submission.csv generated; the only non-fatal log noise was an unrelated Neo4j connection refusal.
Validation Metric: 0.7573
Execution Time: 925.14s

-------------------------------
Attempt #2:
Design: I merged the three stages by keeping Stage 1's data/model/loss/adapter code as the foundation, layering Stage 2's `PrecisionRuntime` policy helpers on top, and consuming both inside Stage 3's `run_training` (with diagnostics + scheduler safe points). I resolved naming conflicts, deduplicated imports, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module-level constants that Stage 3 references, and moved every side effect under `if __name__ == '__main__':` while preserving the CPU-safe `CandidateAdapter` contract.
Results: Training completed successfully in ~37 minutes on an A10 GPU, running 12 epochs with a hybrid transformer model. Validation F1 improved from 0.665 at epoch 1 to a best of ~0.7341 (best epoch 11), and a valid submission file with 1523 rows was written using a tuned threshold of 0.42. The only messages were benign warnings (Graph evidence mirror connection refused, nested-tensor warning) that did not affect execution.
Validation Metric: 0.7341
Execution Time: 2223.42s

-------------------------------
Attempt #3:
Design: I merged the three stages by keeping Stage 1's data pipeline, hybrid BiGRU+CharCNN model, loss helpers, and CPU-safe `CandidateAdapter` as the foundation; layering Stage 2's import-safe precision utilities (`PrecisionConfig`, `resolve_precision_config`, `get_autocast_context`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `clip_gradients`, `degrade_to_fp32`, `is_finite_loss`, `configure_backend_precision`, `format_precision_log`, `precision_to_dict`) on top; and consuming both inside Stage 3's `run_training` with diagnostics, linear warmup + cosine AdamW, best-val-F1 restore, and test inference. I resolved duplicate imports/constants, kept `MODEL_FAMILY`, `WORKING_DIR`, and `SUBMISSION_DIR` as shared module constants, removed the meaningless placeholder arg in the adapter's unpack call, and moved all side effects inside `if __name__ == "__main__":`.
Results: The implementation of this design has bugs.
Insight: Using a different approach may not result in the same bugs as the above approach.


# Hardware-Aware Datatype/Precision Context
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2; observed in 2 batch-size observations
- Precision optimization mode: normal on ampere (compute capability 8.6).
- Allowed native training precision policies: fp32, disabled, fp16_amp
- FP16 AMP is optional, not a whole-pipeline cast: FP32 parameters/inputs, local autocast for forward, FP32-sensitive losses/reductions and metrics/export; GradScaler and backward outside autocast. Disable TF32; use FP32 on CPU. Retain FP32 wherever AMP is unstable, unsupported or slower.
- Precision evidence rule: recommend only a native Tensor Core format with a supported forward/backward training path; generic FP4 is not NVFP4 and FP6 is not an MLEvolve training policy.
- Stage boundary: Choose tensor datatype and precision policy here: DEVICE, USE_AMP, AMP_DTYPE, USE_TF32, GradScaler, Transformer Engine FP8/MXFP8/NVFP4 recipes, autocast helper, precision-required model adapters, fallback behavior, and precision logging.
- Allowed model adaptation: only precision-required wrappers/replacements for compatible modules, precision shape padding/config hooks, or higher-precision islands that preserve the Stage 1 model family, loss, data features, and output interface.
- Out of scope for this stage: general model redesign, loss changes, feature/preprocessing changes, batch size, epochs, learning rate, dataloader workers, gradient accumulation, checkpoint cadence, validation metric, and submission logic.
- Rule: HWDB recommended_patterns are conditional candidates, not measured speedups or mandatory steps. Check workload, installed backend, allowed precision and each stated limitation before applying one. A capability or verified flag does not prove benefit for this candidate. Keep the baseline when evidence is missing; record feature keys, unmet conditions and measured results without inventing them.
- Constraint rule: Hardware advice never overrides task, dataset, submission, package, model-source, or filesystem constraints. Do not invent Kaggle input paths, local checkpoints, torch hub directories, or placeholder weights; use only paths and model sources that are explicitly available in the prompt/config.
# Hardware/Profile Optimization Context
  - datatype_precision
    - filter_audit: features_shown=3/3
  - source=hardware_knowledge_graph.json; linked_features=3
- Workflow stage: Stage 2 datatype/quantization.
- Stage ownership: device, dtype, AMP/autocast, TF32, GradScaler, Transformer Engine recipes, precision-required model adapters, precision fallback, and precision logging.
- Hardware node response rule: the current hardware node response and compact feature-key pairs are already attached below.
- Query rule: do not query the hardware node again; query local feature-node details only for selected feature keys when deeper implementation guidance is needed.
- Source rule: do not fetch external URLs or rely on source links in generated reasoning.
- Scheduler backend config: packing_backend=cuda_process, effective_backend=cuda_process, runner_contract=subprocess_job_v1, enabled_backends=['exclusive', 'mps_process', 'cuda_process'], exclusive_fallback_enabled=True
- Scheduler limits: safe_vram_budget_mb=19201, packing_backend=cuda_process, runner_contract=subprocess_job_v1, exclusive_fallback_enabled=True
- Precision diagnosis: symptoms=['precision_not_optimized'] targets=['improve_precision_efficiency']
- Confidence: 0.0



# Pipeline Decision Contract
Before writing or modifying code, follow the hardware-aware stepwise workflow: model-design -> datatype/quantization -> training (code keys: model_design -> datatype_precision -> training_evaluation).

1. Model design: load/prepare data, create features/splits, and choose architecture/model family, loss, criterion, and output interface first.
2. Datatype/precision: choose only mode-allowed operations, autocast regions, FP32-sensitive regions, scaling and fallback behavior. A policy is not a whole-pipeline dtype.
3. Training: choose optimizer, scheduler, batch size, dataloader settings, checkpointing, validation, submission, runtime fallbacks, and logging.
Datatype precision may include only precision-required model adapters that preserve the Stage 1 model family, loss, data features, and output interface.
Hardware-only training/runtime tuning must not increase epochs, folds, model size, image resolution, ensemble count, TTA, dataset size, or validation workload.
If evidence is missing, use the fallback recorded in the trace instead of inventing hardware claims.

# Pipeline Decision Trace
```json
{
  "model_design": {
    "modality": "text",
    "target_type": "classification",
    "shape_constraints": [
      "sequence length and tokenizer output must stay configurable"
    ],
    "family": "available_transformer_or_sequence_baseline",
    "alternatives_considered": [],
    "loss": "binary cross-entropy with logits",
    "output_interface": "single logit per example; sigmoid threshold selected on internal validation to maximize binary F1; submission columns id,target with int binary predictions",
    "reason": "Use a baseline-compatible model family until stronger task or evidence signals are available.",
    "hardware_fit": "use only compatible evidence"
  },
  "datatype_precision": {
    "precision_policy": "fp16_amp",
    "precision_model_adaptation": "none",
    "fallback_policy": "fall back to fp32 or disabled AMP if the selected precision path is unsupported",
    "reason": "Use conservative precision until hardware and package support are confirmed."
  },
  "training_evaluation": {
    "optimizer": "AdamW over FP32 parameters",
    "scheduler": "linear warmup with linear/cosine decay",
    "batch_size_policy": "fixed",
    "dataloader_policy": "num_workers>0, pin_memory=True, persistent_workers when reused, non_blocking=True device copies",
    "fallbacks": [
      "disable AMP and run FP32 if non-finite loss/gradients or slower/less accurate",
      "retain FP32 for unsupported or unstable operations"
    ],
    "metrics_to_log": [
      "elapsed_seconds",
      "peak_vram_mb",
      "resolved_batch_size"
    ],
    "advanced_optimizer_used": false,
    "reason": "Choose stable training defaults matched to the inferred target type and available model family."
  },
  "evidence": {
    "hardware_context_used": true,
    "evidence_refs": [
      "backend_guidance:cuda_process.no_mps_controls",
      "backend_guidance:cuda_process.normal_process_throughput",
      "backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry",
      "backend_guidance:shared.subprocess.task_quality_and_elasticity"
    ],
    "confidence": 0.95,
    "missing_evidence": [
      "predictor/graph evidence not available",
      "no exact/similar/packed graph profiles for current hardware",
      "no measured FP16 vs FP32 comparison yet"
    ]
  }
}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=## Plan **WHAT makes this task unique:** Disaster-tweet classification is dominated by *metaphorical vs.; purpose=Define the Stage 1 candidate target for later precision and training stages.; hardware_keys=none

# Previous steps
### model_design
**Plan:** ## Plan

**WHAT makes this task unique:** Disaster-tweet classification is dominated by *metaphorical vs. literal* usage of disaster vocabulary ("this song is fire", "my boss is killing me"), plus high-frequency but non-disaster-signaling keywords (e.g. "ablaze", "rain"). Previous attempts all used either plain TF-IDF→MLP (0.7573) or end-to-end sequence nets (0.7341) without conditioning on the keyword categorical and without any metaphor-sensitive stylistic signals.

**WHY my approach fits:** I replace the sequence-model direction with a *keyword-conditioned gated multi-view sparse network* (KB-MVNet) that fuses four heterogeneous evidence views — (1) word TF-IDF, (2) character TF-IDF (captures hashtag/emoji/typos), (3) one-hot keyword view (the strongest categorical prior available in this dataset), (4) 15 hand-crafted stylistic features that proxy metaphor (emoji/emoticon count, uppercase ratio, first-person pronouns, URL/mention structure). A learned soft-max gate selects which view to trust per example, so "ablaze+🌊 and streets flooded" leans on char+style, while "ablaze" alone on word+keyword. Loss is unchanged from the trace: BCE-with-logits → single logit → sigmoid threshold.

**HOW to validate:** Fit on the 85 % internal split, early-stop on the 15 % stratified holdout, and sweep the decision threshold on that holdout to maximize binary F1 — the exact metric. Confirm: val F1 beats 0.7573 and the gate entropy is not collapsed (all four views used). If val F1 < 0.75 or a single view dominates with entropy ≈ 0, I'll revisit fusion.

**Memory / elasticity / backend contract:** All features are in-memory dense float32 view tuples; batch size is `scenario['batch_size']` with no accumulation here (trainer owns it). Word/char dims, hidden width, dropout, and sequence of views remain configurable via context. Token/embedding indices are absent (bag-of-words views) so no long-dtype leakage. Scheduler-owned (NOT implemented here): device assignment, worker count, autocast dtype, GradScaler, optimizer/scheduler cadence, checkpoint cadence.

**Note board:**
- baseline_change: replaced prior TF-IDF→MLP with a 4-view gated keyword-conditioned sparse net (word TF-IDF + char TF-IDF + keyword one-hot + 15 stylistic metaphor proxies) | purpose: directly attack metaphor-vs-literal confusion that dominates this task's error mode, using the keyword categorical as a first-class view rather than as extra text | hardware_keys: none
- baseline_change: added a per-sample softmax gate over the four view embeddings before the classification head | purpose: allow the model to route stylistic/char evidence for hashtag/emoji-heavy tweets and word/keyword evidence for plain news-style tweets, improving F1 on both regimes | hardware_keys: none
- baseline_change: feature bundle is fit strictly on the training fold inside a cached module helper; a no-arg `CandidateAdapter` reuses the same real bundle/model/loss for CPU preflight | purpose: preserve leakage isolation and ensure the preflight exercises the exact production model | hardware_keys: none
**Code:**
```python
# stage: model_design
# Model family: keyword-gated multi-view sparse MLP (word TF-IDF + char TF-IDF + keyword one-hot + stylistic features)
import os
import re
import html
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------ constants
MODEL_FAMILY = "keyword_gated_multiview_sparse_mlp_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
INPUT_DIR = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")

SEED = 42
VAL_FRACTION = 0.15

MAX_WORD_FEATURES = 8000
MAX_CHAR_FEATURES = 12000
MAX_KEYWORD_VOCAB = 300
MIN_KEYWORD_COUNT = 2

HIDDEN_DIM = 256
DROPOUT = 0.35
STYLE_DIM = 15
NUM_VIEWS = 4

# ------------------------------------------------------------------ text utils
_URL_RE = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_WS_RE = re.compile(r"\s+")
_FIRST_PERSON_RE = re.compile(
    r"\b(i|i'm|im|i've|ive|my|me|mine|we|we're|our|us|myself|ourselves)\b",
    re.IGNORECASE,
)
_EMOTICON_RE = re.compile(r"[:;=][\-o\*']?[\)\(dpDP]")
_EMOJI_RE = re.compile(
    "[" "\U0001f300-\U0001faff" "\U00002600-\U000027bf" "\U0001f1e6-\U0001f1ff" "]",
    re.UNICODE,
)


def clean_text(s):
    """Light normalisation that preserves metaphor cues (emoticons, casing, hashtags)."""
    if not isinstance(s, str):
        return ""
    s = html.unescape(s)
    s = _URL_RE.sub(" <url> ", s)
    s = _MENTION_RE.sub(" <user> ", s)
    s = _HASHTAG_RE.sub(r" \1 ", s)  # keep hashtag body as a word token
    s = s.lower()
    s = _WS_RE.sub(" ", s).strip()
    return s


def stylistic_features(raw_text):
    """15 per-tweet stylistic / metaphor-proxy features (always length 15)."""
    t = raw_text if isinstance(raw_text, str) else ""
    n = max(len(t), 1)
    n_upper = sum(1 for c in t if c.isupper())
    n_digit = sum(1 for c in t if c.isdigit())
    n_punct = sum(1 for c in t if c in "!?,.;:'\"")
    n_excl = t.count("!")
    n_q = t.count("?")
    n_hashtag = t.count("#")
    n_mention = t.count("@")
    n_url = 1 if _URL_RE.search(t) else 0
    first_person = len(_FIRST_PERSON_RE.findall(t))
    emoticon = len(_EMOTICON_RE.findall(t))
    emoji = len(_EMOJI_RE.findall(t))
    return np.array(
        [
            float(len(t)),
            float(np.log1p(len(t))),
            n_upper / n,
            n_digit / n,
            n_punct / n,
            float(n_excl),
            float(n_q),
            float(n_hashtag),
            float(n_mention),
            float(n_url),
            float(first_person),
            float(emoticon),
            float(emoji),
            n_excl / n,
            n_q / n,
        ],
        dtype=np.float32,
    )


# ------------------------------------------------------------------ keyword view
def build_keyword_vocab(keywords):
    from collections import Counter

    counts = Counter(
        k.strip().lower() for k in keywords if isinstance(k, str) and k and k.strip()
    )
    ordered = [
        k for k, c in counts.most_common(MAX_KEYWORD_VOCAB) if c >= MIN_KEYWORD_COUNT
    ]
    return {k: i for i, k in enumerate(ordered)}


def keyword_onehot(keywords, vocab):
    unknown_idx = len(vocab)
    out = np.zeros((len(keywords), len(vocab) + 1), dtype=np.float32)
    for i, kw in enumerate(keywords):
        key = kw.strip().lower() if isinstance(kw, str) else ""
        idx = vocab.get(key, unknown_idx)
        out[i, idx] = 1.0
    return out


# ------------------------------------------------------------------ feature bundle
class FeatureBundle(object):
    """Holds every stateful transform, fit on the training fold only."""

    def __init__(self, word_vec, char_vec, kw_vocab, style_scaler):
        self.word_vec = word_vec
        self.char_vec = char_vec
        self.kw_vocab = kw_vocab
        self.style_scaler = style_scaler

    @property
    def word_dim(self):
        return len(self.word_vec.get_feature_names_out())

    @property
    def char_dim(self):
        return len(self.char_vec.get_feature_names_out())

    @property
    def kw_dim(self):
        return len(self.kw_vocab) + 1

    @property
    def style_dim(self):
        return STYLE_DIM

    def transform(self, df):
        texts = [clean_text(t) for t in df["text"].fillna("").astype(str).tolist()]
        word_X = self.word_vec.transform(texts).astype(np.float32).toarray()
        char_X = self.char_vec.transform(texts).astype(np.float32).toarray()
        if "keyword" in df.columns:
            kws = df["keyword"].fillna("").astype(str).tolist()
        else:
            kws = [""] * len(df)
        kw_X = keyword_onehot(kws, self.kw_vocab)
        raw = df["text"].fillna("").astype(str).tolist()
        style = np.stack([stylistic_features(t) for t in raw], axis=0).astype(
            np.float32
        )
        style = self.style_scaler.transform(style).astype(np.float32)
        return word_X, char_X, kw_X, style


def fit_feature_bundle(train_df):
    """Fit the entire feature pipeline on the *training* fold only (no leakage)."""
    texts = [clean_text(t) for t in train_df["text"].fillna("").astype(str).tolist()]

    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=MAX_WORD_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    word_vec.fit(texts)

    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=MAX_CHAR_FEATURES,
        sublinear_tf=True,
        lowercase=True,
    )
    char_vec.fit(texts)

    kws = (
        train_df["keyword"].fillna("").astype(str).tolist()
        if "keyword" in train_df.columns
        else []
    )
    kw_vocab = build_keyword_vocab(kws)

    raw = train_df["text"].fillna("").astype(str).tolist()
    style_raw = np.stack([stylistic_features(t) for t in raw], axis=0).astype(
        np.float32
    )
    style_scaler = StandardScaler().fit(style_raw)

    return FeatureBundle(word_vec, char_vec, kw_vocab, style_scaler)


# ------------------------------------------------------------------ split
def split_data(df, seed=SEED, val_fraction=VAL_FRACTION):
    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        stratify=df["target"].astype(int).values,
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ------------------------------------------------------------------ model
class KeywordGatedMultiviewNet(nn.Module):
    """Fuse four heterogeneous evidence views with a per-sample softmax gate."""

    def __init__(
        self, word_dim, char_dim, kw_dim, style_dim, hidden=HIDDEN_DIM, dropout=DROPOUT
    ):
        super(KeywordGatedMultiviewNet, self).__init__()
        self.word_dim = int(word_dim)
        self.char_dim = int(char_dim)
        self.kw_dim = int(kw_dim)
        self.style_dim = int(style_dim)
        self.hidden = int(hidden)

        def _branch(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.word_proj = _branch(self.word_dim)
        self.char_proj = _branch(self.char_dim)
        self.kw_proj = _branch(self.kw_dim)
        self.style_proj = _branch(self.style_dim)

        self.gate = nn.Sequential(
            nn.Linear(hidden * NUM_VIEWS, hidden),
            nn.GELU(),
            nn.Linear(hidden, NUM_VIEWS),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, word_x, char_x, kw_x, style_x):
        hw = self.word_proj(word_x)
        hc = self.char_proj(char_x)
        hk = self.kw_proj(kw_x)
        hs = self.style_proj(style_x)
        stacked = torch.stack([hw, hc, hk, hs], dim=1)  # [B, 4, H]
        flat = stacked.reshape(stacked.size(0), -1)
        gates = torch.softmax(self.gate(flat), dim=-1).unsqueeze(-1)  # [B, 4, 1]
        fused = (stacked * gates).sum(dim=1)  # [B, H]
        return self.head(fused).squeeze(-1)  # single logit per example


# ------------------------------------------------------------------ loss / criterion
def build_loss():
    """Binary cross-entropy on raw logits (metric: thresholded binary F1)."""
    return nn.BCEWithLogitsLoss()


def _resolve_criterion(context):
    """Return the real criterion; honour caller-supplied mapping if valid."""
    if context:
        crit = context.get("criterion")
        if isinstance(crit, nn.Module):
            return crit
    return build_loss()


# ------------------------------------------------------------------ cached pipeline (for adapter only)
_PIPELINE_CACHE = {}


def _train_csv_path():
    return os.path.join(INPUT_DIR, "train.csv")


def _get_train_df():
    if "train_df" not in _PIPELINE_CACHE:
        _PIPELINE_CACHE["train_df"] = pd.read_csv(_train_csv_path())
    return _PIPELINE_CACHE["train_df"]


def _get_feature_bundle():
    if "bundle" not in _PIPELINE_CACHE:
        df = _get_train_df()
        # Fit on a stratified slice of the train split only (still no val/test labels).
        fit_df, _ = split_data(df)
        _PIPELINE_CACHE["bundle"] = fit_feature_bundle(fit_df)
    return _PIPELINE_CACHE["bundle"]


# ------------------------------------------------------------------ CPU preflight adapter
class CandidateAdapter(object):
    """No-arg CPU-safe adapter that reuses the real model / loss / features."""

    _DEFAULTS = {
        "hidden": HIDDEN_DIM,
        "dropout": DROPOUT,
        "style_dim": STYLE_DIM,
        "lr": 1e-3,
        "weight_decay": 1e-4,
    }

    def _resolve_context(self, context):
        ctx = dict(self._DEFAULTS)
        if context:
            for k, v in context.items():
                if v is not None:
                    ctx[k] = v
        return ctx

    def _resolve_dims(self, context):
        bundle = _get_feature_bundle()
        word_dim = bundle.word_dim
        char_dim = bundle.char_dim
        kw_dim = bundle.kw_dim
        style_dim = bundle.style_dim
        if context:
            if context.get("word_dim"):
                word_dim = int(context["word_dim"])
            if context.get("char_dim"):
                char_dim = int(context["char_dim"])
            if context.get("kw_dim"):
                kw_dim = int(context["kw_dim"])
            if context.get("style_dim"):
                style_dim = int(context["style_dim"])
        return word_dim, char_dim, kw_dim, style_dim

    # ---- construction ----------------------------------------------------
    def build_model(self, context):
        word_dim, char_dim, kw_dim, style_dim = self._resolve_dims(context)
        ctx = self._resolve_context(context)
        return KeywordGatedMultiviewNet(
            word_dim=word_dim,
            char_dim=char_dim,
            kw_dim=kw_dim,
            style_dim=style_dim,
            hidden=int(ctx["hidden"]),
            dropout=float(ctx["dropout"]),
        )

    def build_optimizer(self, model, context):
        ctx = self._resolve_context(context)
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(ctx["lr"]),
            weight_decay=float(ctx["weight_decay"]),
        )

    # ---- batch builders --------------------------------------------------
    def _fixture_batch(self, fixture, device):
        named = {}
        for k, v in fixture.items():
            if isinstance(v, torch.Tensor):
                named[k] = v.to(device)
            elif (
                isinstance(v, (list, tuple))
                and len(v) > 0
                and isinstance(v[0], torch.Tensor)
            ):
                named[k] = v.to(device)
            else:
                named[k] = torch.as_tensor(v, device=device)

        # Prefer named contract keys, otherwise collect every non-target tensor
        # in the fixture order and treat the last labelled tensor as target.
        def _take(*names):
            for n in names:
                if n in named:
                    return named[n]
            return None

        word_x = _take("word_x", "word", "word_ids")
        char_x = _take("char_x", "char", "char_ids")
        kw_x = _take("kw_x", "keyword", "keyword_ids")
        style_x = _take("style_x", "style", "style_features")
        target = _take("target", "label", "labels", "y")

        ordered_inputs = [t for t in (word_x, char_x, kw_x, style_x) if t is not None]
        if not ordered_inputs:
            # Fall back: gather all float tensors except target as inputs
            for k, v in named.items():
                if k in ("target", "label", "labels", "y"):
                    continue
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    ordered_inputs.append(v)
        ordered_inputs = [t.to(dtype=torch.float32) for t in ordered_inputs]

        if target is None:
            n = ordered_inputs[0].size(0) if ordered_inputs else 1
            target = torch.zeros(n, dtype=torch.float32, device=device)
        target = target.to(dtype=torch.float32).view(-1)

        out = {"inputs": tuple(ordered_inputs), "target": target}
        for k, v in named.items():
            out.setdefault(k, v)
        return out

    def _real_batch(self, batch_size, device):
        bundle = _get_feature_bundle()
        df = _get_train_df()
        n = min(int(batch_size), len(df))
        rng = np.random.RandomState(0)
        idx = rng.randint(0, len(df), size=n)
        sub = df.iloc[idx].reset_index(drop=True)
        word_X, char_X, kw_X, style_X = bundle.transform(sub)
        word_x = torch.from_numpy(np.ascontiguousarray(word_X, dtype=np.float32)).to(
            device
        )
        char_x = torch.from_numpy(np.ascontiguousarray(char_X, dtype=np.float32)).to(
            device
        )
        kw_x = torch.from_numpy(np.ascontiguousarray(kw_X, dtype=np.float32)).to(device)
        style_x = torch.from_numpy(np.ascontiguousarray(style_X, dtype=np.float32)).to(
            device
        )
        target = torch.from_numpy(sub["target"].values.astype(np.float32)).to(device)
        return {"inputs": (word_x, char_x, kw_x, style_x), "target": target}

    def _build_batch(self, scenario, device):
        if scenario is None:
            scenario = {}
        batch_size = int(scenario.get("batch_size", 8))
        fixture = scenario.get("fixture") or {}
        if fixture:
            return self._fixture_batch(fixture, device)
        return self._real_batch(batch_size, device)

    def build_train_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    # ---- steps -----------------------------------------------------------
    @staticmethod
    def _unpack(batch):
        if isinstance(batch, dict):
            inputs = batch.get("inputs")
            target = batch.get("target")
        else:
            inputs, target = batch[0], batch[1]
        if not isinstance(inputs, (list, tuple)):
            inputs = (inputs,)
        return inputs, target

    def training_step(self, model, batch, context):
        inputs, target = self._unpack(batch)
        inputs = [
            t.to(dtype=torch.float32) if torch.is_floating_point(t) else t
            for t in inputs
        ]
        target = target.to(dtype=torch.float32).view(-1)
        logits = model(*inputs)
        criterion = _resolve_criterion(context)
        loss = criterion(logits.view(-1), target)
        return loss

    def validation_step(self, model, batch, context):
        with torch.no_grad():
            return self.training_step(model, batch, context)


def build_model_from_bundle(bundle, hidden=HIDDEN_DIM, dropout=DROPOUT):
    """Convenience constructor the training loop can call with the real fitted bundle."""
    return KeywordGatedMultiviewNet(
        word_dim=bundle.word_dim,
        char_dim=bundle.char_dim,
        kw_dim=bundle.kw_dim,
        style_dim=bundle.style_dim,
        hidden=hidden,
        dropout=dropout,
    )

```

# Current step: datatype_precision
Define device, tensor dtype, AMP/autocast, TF32, GradScaler, fallback, and precision logging utilities.


# Instructions

## Response format

Your response should be:
1. A brief plan (2-3 sentences) describing what you will do in this step
2. A `Note board:` block with 1-3 bullets. Each bullet must use this exact shape: `- baseline_change: <what changed from the baseline or prior step> | purpose: <why this supports the candidate target> | hardware_keys: <comma-separated local feature keys or none>`
3. A single markdown code block (wrapped in ```) containing ONLY the code for this step
IMPORTANT: Do NOT write code for other steps. Only write code for the current step.
No module-level side effects: this step may define imports, constants, classes, and helper functions only. Do not access data, read environment settings, configure CUDA/TF32, configure logging, instantiate models/losses, construct DataLoaders, train, evaluate, or write submissions at import time; expose helpers for the final main guard instead.

## Hardware/Profile reasoning rule

- HWDB recommended_patterns are conditional candidates, not measured speedups or mandatory steps. Check workload, installed backend, allowed precision and each stated limitation before applying one. A capability or verified flag does not prove benefit for this candidate. Keep the baseline when evidence is missing; record feature keys, unmet conditions and measured results without inventing them.
- Hardware advice never overrides task, dataset, submission, package, model-source, or filesystem constraints. Do not invent Kaggle input paths, local checkpoints, torch hub directories, or placeholder weights; use only paths and model sources that are explicitly available in the prompt/config.
- For hardware/runtime optimization, preserve the parent solution's modeling budget unless the user explicitly asks to improve score. Do not increase epochs, folds, model size, input resolution, ensemble count, TTA, dataset size, or validation workload as a hardware-only optimization.
- Generated code must run as one independent subprocess and follow the exact effective backend named in the backend-guidance section.
- Use backend coding preferences in job code, but leave daemon setup, launch policy, device assignment, active-thread allocation, priorities, and memory limits to the scheduler.
- Prefer scheduler-compatible code with configurable batch size, precision, dataloader workers, checkpoints, and runtime logging when using GPU training.
- The stage-specific hardware node response is already provided in the prompt. Do not query the hardware node again; query local feature-node details only for selected feature keys when deeper implementation guidance is needed.
- Use the Cross-Stage Note Board to keep Stage 2 precision and Stage 3 training choices aligned with the Stage 1 candidate target.


## Pipeline Decision Contract

- Use the Pipeline Decision Trace as the source of truth for code generation and planning.
- Follow the hardware-aware stepwise workflow: model-design -> datatype/quantization -> training (code keys: model_design -> datatype_precision -> training_evaluation).
- Do not jump directly to optimizer, precision, or batch-size choices before model family and output interface are decided.
- The model_design stage stores data modality, target shape, preprocessing/features, model family, loss, criterion, and output interface.
- Numeric precision such as fp32, fp16, bf16, tf32, or TE FP8/NVFP4 belongs under datatype_precision.precision_policy unless the task explicitly requires a numeric type.
- The datatype_precision step may make narrow precision-required model adapters such as Transformer Engine layer wrappers/replacements, padding/config hooks, autocast recipes, or higher-precision islands, but it must preserve the Stage 1 model family, loss, data features, and output interface.
- Hardware/profile evidence may influence model fit and tuning only when it is compatible with the task, installed packages, and available model sources.
- Hardware-only training/runtime tuning must not increase epochs, folds, model size, image resolution, ensemble count, TTA, dataset size, or validation workload.
- If execution feedback contradicts an earlier decision, update only the relevant part of the decision and explain why in the plan.


## 🔬 Critical: Scientific Approach to Design

- 
- Before designing your solution, you must answer three fundamental questions:
- 
- 1. **WHAT makes this task unique?**
- - Not generic observations like 'it's a classification task'
- - What SPECIFIC patterns, challenges, or domain characteristics?
- 
- 2. **WHY is your approach suitable for this task?**
- - Not just 'this model is good' - explain the MATCH between approach and task
- - What properties of your method address the task characteristics?
- 
- 3. **HOW will you validate your hypothesis?**
- - What outcome would confirm your approach is right?
- - What outcome would suggest you need to reconsider?
- 
- ---
- 
- ⚠️ This is not a template to fill - this is how scientists think.
- Blindly applying standard methods without understanding WHY is not acceptable.
- 
- Your plan should naturally reflect this reasoning process.


## Solution sketch guideline

- - This first solution design should be relatively simple — avoid complex ensemble strategies or extensive hyperparameter searches at this stage.
- - 🎯 **CRITICAL: NOVELTY & DIVERSITY REQUIREMENT**:
- • **Mandatory**: Your solution MUST be NOVEL compared to ALL existing attempts in Memory.
- • **Step 1**: Carefully analyze the core idea of EACH previous attempt in Memory.
- • **Step 2 - Choose Strategy**:
- → **Option A (Preferred)**: Propose a COMPLETELY DIFFERENT approach exploring an untried direction.
- → **Option B**: Build upon an existing approach BUT add significant novel insights that fundamentally change the solution.
- • **Forbidden**: Minor variations (changing hyperparameters, swapping similar models, tweaking preprocessing).
- • **Think**: 'Does my approach explore a fundamentally different hypothesis?' If NO → redesign.
- - Don't propose the same modelling solution but keep the evaluation the same.
- - Your plan should be concise but comprehensive: Must address WHAT/WHY/HOW (2-4 sentences each). Avoid verbosity - every sentence should add new insight. Natural length: around 8-12 sentences for a complete reasoning process.
- - Propose an evaluation metric that is reasonable for this task.
- - Don't suggest to do EDA.
- - The data is already prepared in `./input` directory. No need to unzip files.


## Coding & Execution Guidelines (CRITICAL)

- - **NO PROGRESS BARS**: You MUST NOT use `tqdm`. Assume `tqdm` is not installed. Use standard Python loops only. Do not use `verbose=1`.
- - **MINIMAL LOGGING**: Print ONLY 1 line per epoch (e.g. loss/accuracy). Do NOT print batch-level logs.
- - **FINAL OUTPUT**: The VERY LAST line of execution MUST be `print(f'Final Validation Score: {score}')`. This is required for the score parser.


## Backend design contract

- - In the plan, explicitly state the selected memory strategy and physical/effective batch elasticity strategy.
- - State operator and normalization considerations, the model dimensions that remain configurable, and the fallback behavior.
- - List scheduler-owned backend controls that the generated subprocess must not implement.
- - Do not assume code-level integration, shared framework state, or synchronized start time with another job.


## CPU preflight adapter contract

- - **CPU PREFLIGHT CONTRACT (MANDATORY)**: Define a no-argument `class CandidateAdapter` in the generated file. It must implement `build_model(context)`, `build_optimizer(model, context)`, `build_train_batch(scenario, device)`, `build_validation_batch(scenario, device)`, `training_step(model, batch, context)`, and `validation_step(model, batch, context)`.
- - The adapter must exercise the same real model, loss, inputs, and optimizer as the training pipeline; do not use mock tensors or a different toy model. Its batch builders must honor `scenario['batch_size']` and return all inputs required by the model plus a target.
- Batch builders must return (inputs, target) or a mapping with recognized `inputs` and `target` keys. For multiple positional model inputs, use {'inputs': (word_ids, char_ids, keyword_ids), 'target': targets}; training_step and validation_step must unpack those same inputs into the real model. Custom top-level keys alone (such as word_ids) are not recognized. Preserve any authoritative named fixture keys, adding an inputs tuple that references those same tensors when needed. Token/embedding indices remain torch.long and masks keep their required dtype; FP32 precision rules apply to floating-point computation, not integer indices. The checker owns zero_grad, backward and optimizer.step: training_step only computes and returns the connected scalar loss, without performing those updates itself.
- - Keep imports, constants, class/function definitions, and read-only device configuration import-safe. Put all training, validation, prediction, and submission side effects under `if __name__ == '__main__':`.


## Implementation guideline

- Milestone objective: complete one real candidate through review, CPU preflight, GPU training, validation and test submission. There is no overall wall-clock deadline or failed-node budget. Max execution time per run = an hour. Start with a compact trainable model and a complete executable pipeline; do not fabricate metrics, bypass checks, or use dummy predictions.
- 
- Code execution must complete within the configured per-run limit of an hour, including validation and test inference.
- 🎯 **CRITICAL REQUIREMENTS** (Non-Negotiable):
- 
- **1. Model Inference for ALL Predictions**
- • EVERY prediction (validation & test) MUST come from trained model's forward pass
- • Process: Load data → Preprocess → model.predict()/model.forward() → Save predictions
- • ❌ FORBIDDEN: Constants, placeholders, dummy values, empty arrays, statistics, random numbers
- • ❌ FORBIDDEN: Fake/mock metric functions (must use real sklearn.metrics or correct manual implementation)
- • Why: Shortcuts create fake high validation scores but fail on test (CRITICAL SYSTEM FAILURE)
- 
- **2. Generate submission.csv**
- • Path: `./submission/submission.csv` (NOT ./working/submission.csv)
- • Content: Model predictions on ALL test samples
- • Format: Follow task description exactly
- 
- **3. Print Validation Metric**
- • MUST print: `print(f'Final Validation Score: {score}')`
- • Score MUST be computed on hold-out validation set using proper metric formula
- • CRITICAL CONSISTENCY REQUIREMENT: Ensure that validation and test inference use IDENTICAL processing logic. Any differences in how validation and test data are handled (such as post-processing, reconstruction, or formatting) can cause large performance gaps between validation and test sets. Maintain consistency across all data processing steps for both validation and test phases.
- 
- **4. Scheduler Model Family Contract**
- • MUST define a top-level constant `MODEL_FAMILY = "family_name"` near the top of the file.
- • Use a stable, architecture-specific name, for example `resnet50_224`, `swin_b_384`, or `lightgbm_tabular_v1`.
- • If you switch the model architecture or input geometry in an improvement/evolution, update `MODEL_FAMILY` to a new distinct value.
- • If there is no more specific model/backbone name in the script, set the model name/key variable to the same value as `MODEL_FAMILY`.
- 
- **5. CPU Model-Preflight Adapter Contract**
- • The module MUST be import-safe: definitions and lightweight constants may be top-level, but all training, validation, inference, and submission execution MUST be called only inside `if __name__ == "__main__":`.
- • MUST define a no-argument top-level `CandidateAdapter` class with methods `build_model(context)`, `build_optimizer(model, context)`, `build_train_batch(scenario, device)`, `build_validation_batch(scenario, device)`, `training_step(model, batch, context)`, and `validation_step(model, batch, context)`. build_model(context) MUST return the real torch.nn.Module used by training, never a dict, tuple, or metadata wrapper.
- • `training_step` MUST return the real scalar loss tensor connected to model parameters; adapter methods must reuse the script's real model, loss, transforms, collate logic, and optimizer rather than mocks.
- • Treat every caller-supplied `context` as a partial mapping: merge it over adapter defaults before reading optional keys, and preserve caller-provided values.
- • CandidateAdapter context mutations do not persist between checker method calls; `training_step` and `validation_step` MUST resolve the script's real criterion when the partial context omits it or supplies `None`, rather than relying on `build_model` to mutate a local context.
- • Batch builders MUST honor `scenario['batch_size']` and `device`; use `os.environ.get('MLEVOLVE_INPUT_DIR', './input')` to locate input fixtures during the isolated CPU check.
- Batch builders must return (inputs, target) or a mapping with recognized `inputs` and `target` keys. For multiple positional model inputs, use {'inputs': (word_ids, char_ids, keyword_ids), 'target': targets}; training_step and validation_step must unpack those same inputs into the real model. Custom top-level keys alone (such as word_ids) are not recognized. Preserve any authoritative named fixture keys, adding an inputs tuple that references those same tensors when needed. Token/embedding indices remain torch.long and masks keep their required dtype; FP32 precision rules apply to floating-point computation, not integer indices. The checker owns zero_grad, backward and optimizer.step: training_step only computes and returns the connected scalar loss, without performing those updates itself.
- • When `scenario['fixture']` is non-empty, it is the authoritative named input contract: build every listed tensor with the declared non-batch shape, preserve those names in the returned mapping, and pass all model inputs to the real forward path. Do not collapse a multimodal fixture into one feature vector.
- • Adapter construction and a one-batch CPU step must be lightweight and must not download weights or require CUDA.
- 
- 📁 **Directories**: Input data in `./input/`, submission in `./submission/`, temp files in `./working/`
- 
- 📦 **Packages & Internet**: Guaranteed installed packages: numpy, pandas, scikit-learn, statsmodels, xgboost, lightgbm, torch, torchvision, torch-geometric, bayesian-optimization, timm, transformers, sentence-transformers, opencv-python, Pillow. Do not assume unlisted third-party packages are installed; avoid them or provide an ImportError-guarded fallback. Never run pip from generated code. Generated code runs offline: do not download weights or data through torch.hub.load(), Hugging Face, timm, or another remote loader. Use only verified local or cached weights.
- 
- ⚠️ **API Compatibility**: LightGBM/XGBoost: ❌ `fit(..., early_stopping_rounds=...)` → ✅ LightGBM: `fit(..., callbacks=[lgb.early_stopping(...)])` ✅ XGBoost: `XGBClassifier(early_stopping_rounds=...)`
- • AdamW: ❌ `from transformers import AdamW` (deprecated) → ✅ `from torch.optim import AdamW`
- 
- 🚫 **Execution Guidelines**:
- • NO tqdm (not installed), NO verbose=1
- • Print only 1 line per epoch (minimize logging)
- • Runtime diagnostics are an exception: one diagnostics line per epoch plus context entry/exit; no per-batch printing.
- PyTorch runtime diagnostics: import TrainingDiagnostics from utils.training_diagnostics. Inside the training entrypoint, wrap training in `with TrainingDiagnostics(model, optimizer, scaler=scaler_or_None, settings={...}) as diagnostics:` after model/optimizer initialization. Settings must include actual physical_batch_size, effective_batch_size, planned_epochs, steps_per_epoch, model_family, split_seed and precision choice/reason. Call diagnostics.after_update() once after each optimizer update ATTEMPT, after scaler.step and scaler.update (or optimizer.step); never after accumulation microbatches. The helper observes real optimizer steps, including skips, and actual autocast dtype. Call diagnostics.report(epoch=one_based_epoch) once per epoch; keep the normal epoch metric line. Save diagnostics.state_dict() with the training checkpoint and restore it using diagnostics.load_state_dict(...). Preserve these calls during merges/repairs. Do not replace missing measurements with zero. Keep the final validation score as the last line, after leaving the diagnostics context. These diagnostics also run without the scheduler.
- • Use DataLoader with num_workers>=2 for speed
- 
- ⚠️  **Self-Check Before Finalizing**:
- □ Did predictions pass through model's learned weights during inference? (If NO → INVALID)
- □ Did I generate submission.csv in correct path with ALL test predictions?
- □ Did I print validation metric as the last line?
- □ Did I use the COMPLETE training dataset (not a tiny subset)?
- □ Is execution behind the main guard and is CandidateAdapter complete and CPU-safe?
- □ Does scheduled PyTorch training use script_scheduler_context, load_resume_checkpoint, and STEP/EPOCH safe_point calls with steps_per_epoch and a complete state_factory? Preserve these hooks during merges and repairs; keep setup inside the training entrypoint and hooks inactive when the context is None.
- Use the installed API exactly: context.control_hook.safe_point(SafePointType.STEP, epoch=epoch, global_step=global_step, steps_per_epoch=steps_per_epoch, state_factory=state_factory). Use the same keyword arguments for BEFORE_TRAIN and EPOCH. There is no payload= parameter and no second positional argument. Import script_scheduler_context from localml_scheduler.execution.script_context and SafePointType from localml_scheduler.domain. Guard hooks with `if context is not None`; do not catch and suppress hook/checkpoint exceptions. Define state_factory to capture actual model, optimizer, scaler, LR scheduler, RNG, global_step and exact step_in_epoch/sampler_state/data_state. Restore those same state objects and loader position before continuing. Use direct hook calls and TrainingDiagnostics construction so the static admission gate can verify the contract; do not add unreachable token-matching calls.
- 
- The code for this step must be self-contained and can be integrated with other steps.
- Use clear variable names that are consistent with previous steps.
- Do not duplicate code from previous steps - assume those parts already exist.
- Make sure to handle edge cases appropriately.


## Precision policy

- Normal mode permits FP32 and selective FP16 AMP only, not BF16, TF32, quantization or TE recipes. FP16 is an option, not a requirement or a whole-pipeline dtype. Keep model parameters and floating inputs and floating regression targets in FP32; preserve integer indices and labels. Use CUDA autocast only around eligible forward operations and compatible loss computation. Run backward and optimizer updates outside autocast, with GradScaler for FP16 and unscale before gradient clipping. For fragile losses/reductions, disable autocast locally and explicitly cast inputs to FP32. Keep metric accumulation and prediction export in FP32; validation/test forwards may share a validated autocast path. Do not call model.half() or cast the entire batch to half. CPU preflight uses FP32 with AMP/scaling disabled. Disable TF32 for matmul and convolution. Compare finite loss/gradients, task quality and elapsed time against FP32; retain FP32 for unsupported, unstable or slower operations. Record the actual regions and fallback, not an unmeasured speedup.


## 🚨 DATA LEAKAGE PREVENTION

- 
- ⚠️ **Strict Isolation Principle**: Validation/Test data must remain strictly unseen during training.
- 
- ✅ **Sequence**: Always **Split Data FIRST**, then apply processing.
- ✅ **Stateful Transformations**: Fit all Scalers, Encoders, Imputers, and Tokenizers **ONLY on Training data**, then use `.transform()` on Validation/Test.
- ✅ **Feature Engineering**: Calculate global statistics (e.g., mean, variance, vocabulary) solely from the Training set.
- ✅ **Target Leakage**: Never use target information (e.g., Target Encoding) from the validation set.
- 


## Installed Packages

The following packages are guaranteed in the execution environment: `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `xgboost`, `lightgbm`, `torch`, `torchvision`, `torch-geometric`, `bayesian-optimization`, `timm`, `transformers`, `sentence-transformers`, `opencv-python`, `Pillow`. Do not assume any unlisted third-party package is installed: either avoid it or use an ImportError-guarded fallback. Do not run pip from generated code. For neural networks we suggest using PyTorch rather than TensorFlow.

## 💡 Recommendation: Robustness & Generalization Strategy

- 
- **To improve model robustness and generalization on unseen data:**
- 
- ✅ **Architecture**: Match model inductive bias to data structure (e.g., CNNs/ViTs for spatial grids, Transformers/RNNs for sequences, GNNs/GCNs for graphs/topology)
- ✅ **Input Strategy**: Handle variable-length or large-scale inputs via **windowing strategies** or patch-based processing (consider overlap for smoother predictions)
- ✅ **Regularization**: Consider using Dropout, Batch/Layer Norm, Weight Decay, or Label Smoothing
- ✅ **Loss Function**: Inspect class distribution and adapt loss accordingly (e.g., weighted loss, FocalLoss, or task-specific objectives)
- ✅ **Learning Rate**: Consider using adaptive schedules like Cosine Annealing or ReduceLROnPlateau or Warmup with differential rates if needed
- ✅ **Data Augmentation**: Apply domain-appropriate augmentation based on data modality (e.g., geometric transforms, masking, mixup)
- ✅ **Validation**: Monitor validation metrics strictly and use early stopping to prevent overfitting
- 
- ⚠️ **Note**:
- Prioritize capturing the intrinsic structure of the data (Inductive Bias) over simply increasing model size.
- 


## datatype_precision guidelines

- - Read the Cross-Stage Note Board first and preserve the Stage 1 candidate target; precision choices and adapters should support that target instead of changing the model/data design.
- CRITICAL: Do NOT redesign model family, loss function, model output interface, data features, preprocessing, optimizer, scheduler, batch size, epochs, learning rate, dataloader workers, gradient accumulation, checkpoint cadence, validation metric, or submission logic.
- Choose only from the current mode's allowed precision policies. FP16 AMP is selective: define forward/loss autocast regions, FP32-sensitive regions, backward/scaling behavior and CPU/FP32 fallback. A hardware capability is not a requirement to use it.
- Hardware graph contract: the stage-specific hardware node response is already attached in the Hardware/Profile Optimization Context. Do not query the hardware node again; query local feature-node details only for selected feature keys when deeper guidance is needed, and never fetch external URLs.
- Note board: record how the precision policy and any precision-required adapter support the Stage 1 target and which feature keys drove the choice.
- Keep precision configurable and valid for the canonical effective backend. Do not hardcode scheduler backend selection or implement scheduler-owned launch and resource controls.
- Expose simple variables/utilities and any adapted model object that the training_evaluation step can consume directly, and include lightweight logging of selected precision without batch-level noise.
- Pipeline decision: consume `datatype_precision.precision_policy`, `datatype_precision.precision_model_adaptation`, `datatype_precision.fallback_policy`, `evidence`, and the Hardware/Profile Optimization Context to define dtype, AMP/TF32, GradScaler, TE recipes/autocast, precision-required model adapters, and precision fallback behavior.
- Pipeline decision: do not change model family, loss, task output interface, optimizer, scheduler, dataloader policy, batch size, or validation/submission behavior in this step.
- Normal mode permits FP32 and selective FP16 AMP only, not BF16, TF32, quantization or TE recipes. FP16 is an option, not a requirement or a whole-pipeline dtype. Keep model parameters and floating inputs and floating regression targets in FP32; preserve integer indices and labels. Use CUDA autocast only around eligible forward operations and compatible loss computation. Run backward and optimizer updates outside autocast, with GradScaler for FP16 and unscale before gradient clipping. For fragile losses/reductions, disable autocast locally and explicitly cast inputs to FP32. Keep metric accumulation and prediction export in FP32; validation/test forwards may share a validated autocast path. Do not call model.half() or cast the entire batch to half. CPU preflight uses FP32 with AMP/scaling disabled. Disable TF32 for matmul and convolution. Compare finite loss/gradients, task quality and elapsed time against FP32; retain FP32 for unsupported, unstable or slower operations. Record the actual regions and fallback, not an unmeasured speedup.



Let me approach this systematically.
First, I'll examine the dataset:
```
input/
    description.md (5 lines)
    sample_submission.csv (1524 lines)
    test.csv (1724 lines)
    train.csv (6839 lines)
submission/

working/
```

-> input/description.md has content:

```
# Disaster Tweets: Custom Precision Benchmark

Predict target=1 for tweets about real disasters, target=0 otherwise. Use text, keyword and location as appropriate. The metric is binary F1. Write submission.csv with id,target columns and binary predictions for test.csv. Use only train.csv labels for model selection, with your own internal validation split.

This is a custom benchmark, not an official MLE-bench competition definition or Kaggle leaderboard evaluation. The original labeled Kaggle training data was split 80/20 with stratification and seed 42. Both precision modes use this same holdout; test labels are unavailable to the agents. No leaderboard submission is made.

```

-> input/sample_submission.csv has 1523 rows and 2 columns.
⚠️  IMPORTANT: This is the CORRECT submission format that must be followed!
The exact column names are: id, target
Any description.md format information should be IGNORED if it conflicts with this file.
Here is some information about the columns:
id (int64) has range: 5.00 - 10873.00, 0 nan values
target (int64) has 1 unique values: [0]

-> input/test.csv has 1523 rows and 4 columns.
Here is some information about the columns:
id (int64) has range: 5.00 - 10873.00, 0 nan values

-> input/train.csv has 6090 rows and 5 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 10872.00, 0 nan values
target (int64) has 2 unique values: [1, 0]

        ⚠️  CRITICAL SUBMISSION FORMAT NOTE:
        - If you see sample_submission.csv or similar files, those contain the CORRECT submission format
        - The column names in these files are the FINAL AUTHORITY for submission format
        - Always use the column names from the actual sample submission files
        

# Used Prompt 3: training_evaluation

🏆 You are a Kaggle Grandmaster - a top-tier ML expert competing to WIN.

**Your Standards**:
✓ Design complete ML pipelines (data → model → training → inference)
✓ Implement real models that LEARN from data (not baseline scripts with constants)
✓ Generate predictions through ACTUAL MODEL INFERENCE on each sample
✓ Compete for TOP performance, not trivial baselines

Your solution will be evaluated on a real leaderboard. Treat this with professionalism.

Now, let's begin the competition. You need to come up with an excellent and creative plan for a competitive solution and then implement this solution in Python with the quality expected of a Kaggle Grandmaster. We will now provide a description of the task.

You are currently focusing on the 'training_evaluation' step of the solution. Your task is to write ONLY the code for this specific step, not the complete solution.


# Task description
# Disaster Tweets: Custom Precision Benchmark

Predict target=1 for tweets about real disasters, target=0 otherwise. Use text, keyword and location as appropriate. The metric is binary F1. Write submission.csv with id,target columns and binary predictions for test.csv. Use only train.csv labels for model selection, with your own internal validation split.

The original labeled training data was split 80/20 with stratification and seed 42. Test labels are unavailable to the agents.

============================================================
**REQUIRED SUBMISSION FORMAT**
============================================================
The final submission file must match this format:

  id  target
6925       0
1975       0
5034       0
 254       0
8370       0

(Showing first 5 rows as example)
============================================================
**Submission File Location**: Must save the submission to `./submission/submission.csv`
.


# Memory
Below is a record of previous solution attempts and their outcomes:
 Summary: 3 total attempts - 3 executed, 2 successful (best: 0.7573)

-------------------------------
Attempt #1:
Design: I merged the three stages by keeping Stage 1's data cleaning, TF-IDF feature builders, `SparseMLP`, `build_loss`, `tune_threshold` and split helpers as the foundation; layering Stage 2's import-safe precision utilities (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `degrade_to_fp32`, `configure_tf32`, `format_precision_log`) on top; then consuming both inside Stage 3's `run_training` with scheduler safe-points, `TrainingDiagnostics`, linear warmup+cosine AdamW, best-val-F1 restore, and test inference. I resolved the duplicate `CandidateAdapter` by keeping Stage 3's self-contained CPU-safe version (it uses the Stage 1 model/loss/features through the Stage 2 precision path), removed duplicate imports/constants, kept all side effects under `if __name__ == "__main__":`, and preserved the exact `MODEL_FAMILY`, `WORKING_DIR`, `SUBMISSION_DIR` contract.
Results: The training script executed successfully end-to-end: TF-IDF features were fit on the train split, the SparseMLP trained under CUDA FP16 AMP with TF32 disabled, early stopping triggered after 4 epochs without improvement, and a best-val-F1 checkpoint was restored for test inference and submission writing. The run produced a finite best validation F1 of 0.7573 (peak at epoch 1), with the final score line printed and a submission.csv generated; the only non-fatal log noise was an unrelated Neo4j connection refusal.
Validation Metric: 0.7573
Execution Time: 925.14s

-------------------------------
Attempt #2:
Design: I merged the three stages by keeping Stage 1's data/model/loss/adapter code as the foundation, layering Stage 2's `PrecisionRuntime` policy helpers on top, and consuming both inside Stage 3's `run_training` (with diagnostics + scheduler safe points). I resolved naming conflicts, deduplicated imports, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module-level constants that Stage 3 references, and moved every side effect under `if __name__ == '__main__':` while preserving the CPU-safe `CandidateAdapter` contract.
Results: Training completed successfully in ~37 minutes on an A10 GPU, running 12 epochs with a hybrid transformer model. Validation F1 improved from 0.665 at epoch 1 to a best of ~0.7341 (best epoch 11), and a valid submission file with 1523 rows was written using a tuned threshold of 0.42. The only messages were benign warnings (Graph evidence mirror connection refused, nested-tensor warning) that did not affect execution.
Validation Metric: 0.7341
Execution Time: 2223.42s

-------------------------------
Attempt #3:
Design: I merged the three stages by keeping Stage 1's data pipeline, hybrid BiGRU+CharCNN model, loss helpers, and CPU-safe `CandidateAdapter` as the foundation; layering Stage 2's import-safe precision utilities (`PrecisionConfig`, `resolve_precision_config`, `get_autocast_context`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `clip_gradients`, `degrade_to_fp32`, `is_finite_loss`, `configure_backend_precision`, `format_precision_log`, `precision_to_dict`) on top; and consuming both inside Stage 3's `run_training` with diagnostics, linear warmup + cosine AdamW, best-val-F1 restore, and test inference. I resolved duplicate imports/constants, kept `MODEL_FAMILY`, `WORKING_DIR`, and `SUBMISSION_DIR` as shared module constants, removed the meaningless placeholder arg in the adapter's unpack call, and moved all side effects inside `if __name__ == "__main__":`.
Results: The implementation of this design has bugs.
Insight: Using a different approach may not result in the same bugs as the above approach.


# Hardware-Aware Training Hyperparameter Context
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2; observed in 2 batch-size observations
- Precision optimization mode: normal on ampere (compute capability 8.6).
- Allowed native training precision policies: fp32, disabled, fp16_amp
- FP16 AMP is optional, not a whole-pipeline cast: FP32 parameters/inputs, local autocast for forward, FP32-sensitive losses/reductions and metrics/export; GradScaler and backward outside autocast. Disable TF32; use FP32 on CPU. Retain FP32 wherever AMP is unstable, unsupported or slower.
- Precision evidence rule: recommend only a native Tensor Core format with a supported forward/backward training path; generic FP4 is not NVFP4 and FP6 is not an MLEvolve training policy.
- Stage boundary: Optimize training hyperparameters here: physical batch size, effective batch size, gradient accumulation, epochs, learning rate, weight decay, scheduler, dataloader workers, checkpointing, runtime logging, validation, and submission.
- Precision boundary: consume the datatype_precision variables/utilities. Do not pick a new AMP dtype here unless the prior policy is impossible to use and a safe fallback is needed.
- Rule: HWDB recommended_patterns are conditional candidates, not measured speedups or mandatory steps. Check workload, installed backend, allowed precision and each stated limitation before applying one. A capability or verified flag does not prove benefit for this candidate. Keep the baseline when evidence is missing; record feature keys, unmet conditions and measured results without inventing them.
- Constraint rule: Hardware advice never overrides task, dataset, submission, package, model-source, or filesystem constraints. Do not invent Kaggle input paths, local checkpoints, torch hub directories, or placeholder weights; use only paths and model sources that are explicitly available in the prompt/config.
- Budget guardrail: For hardware/runtime optimization, preserve the parent solution's modeling budget unless the user explicitly asks to improve score. Do not increase epochs, folds, model size, input resolution, ensemble count, TTA, dataset size, or validation workload as a hardware-only optimization.
# Hardware/Profile Optimization Context
  - training_evaluation
    - filter_audit: features_shown=3/5, omitted_not_recommended=['soap_optimizer', 'ademamix_optimizer']
    - muon_optimizer:  (category=optimizer, support_level=supported, recommended=True, verified=False, recommendation_scope=small_matmul_heavy_training_only): usage: Hybrid Muon+AdamW training optimizer for matmul-heavy candidates: the example code shows the parameter split, Muon settings, and AdamW companion optimizer.; notes: Use Muon on selected 2D hidden weights and AdamW on embedding/bias/norm/head/non-2D params; tune Muon and AdamW learning rates separately; tune weight_decay per optimizer group; initial Muon settings: ns_steps=5, momentum=0.95, nesterov=True; prefer adjust_lr_fn='match_rms_adamw' when reusing an AdamW LR schedule; compare against a tuned AdamW baseline.; limitations: software training optimizer; requires a PyTorch build that exposes torch.optim.Muon; no hardware-specific acceleration is required; use only on selected 2D hidden weights and keep AdamW for embeddings/bias/norm/head/non-2D params; benchmark against tuned AdamW; not an inference feature
  - source=hardware_knowledge_graph.json; linked_features=5
- Workflow stage: Stage 3 training.



# Pipeline Decision Contract
Before writing or modifying code, follow the hardware-aware stepwise workflow: model-design -> datatype/quantization -> training (code keys: model_design -> datatype_precision -> training_evaluation).

1. Model design: load/prepare data, create features/splits, and choose architecture/model family, loss, criterion, and output interface first.
2. Datatype/precision: choose only mode-allowed operations, autocast regions, FP32-sensitive regions, scaling and fallback behavior. A policy is not a whole-pipeline dtype.
3. Training: choose optimizer, scheduler, batch size, dataloader settings, checkpointing, validation, submission, runtime fallbacks, and logging.
Datatype precision may include only precision-required model adapters that preserve the Stage 1 model family, loss, data features, and output interface.
Hardware-only training/runtime tuning must not increase epochs, folds, model size, image resolution, ensemble count, TTA, dataset size, or validation workload.
If evidence is missing, use the fallback recorded in the trace instead of inventing hardware claims.

# Pipeline Decision Trace
```json
{
  "model_design": {
    "modality": "text",
    "target_type": "classification",
    "shape_constraints": [
      "sequence length and tokenizer output must stay configurable"
    ],
    "family": "available_transformer_or_sequence_baseline",
    "alternatives_considered": [],
    "loss": "binary cross-entropy with logits",
    "output_interface": "single logit per example; sigmoid threshold selected on internal validation to maximize binary F1; submission columns id,target with int binary predictions",
    "reason": "Use a baseline-compatible model family until stronger task or evidence signals are available.",
    "hardware_fit": "use only compatible evidence"
  },
  "datatype_precision": {
    "precision_policy": "fp16_amp",
    "precision_model_adaptation": "none",
    "fallback_policy": "fall back to fp32 or disabled AMP if the selected precision path is unsupported",
    "reason": "Use conservative precision until hardware and package support are confirmed."
  },
  "training_evaluation": {
    "optimizer": "AdamW over FP32 parameters",
    "scheduler": "linear warmup with linear/cosine decay",
    "batch_size_policy": "fixed",
    "dataloader_policy": "num_workers>0, pin_memory=True, persistent_workers when reused, non_blocking=True device copies",
    "fallbacks": [
      "disable AMP and run FP32 if non-finite loss/gradients or slower/less accurate",
      "retain FP32 for unsupported or unstable operations"
    ],
    "metrics_to_log": [
      "elapsed_seconds",
      "peak_vram_mb",
      "resolved_batch_size"
    ],
    "advanced_optimizer_used": false,
    "reason": "Choose stable training defaults matched to the inferred target type and available model family."
  },
  "evidence": {
    "hardware_context_used": true,
    "evidence_refs": [
      "backend_guidance:cuda_process.no_mps_controls",
      "backend_guidance:cuda_process.normal_process_throughput",
      "backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry",
      "backend_guidance:shared.subprocess.task_quality_and_elasticity"
    ],
    "confidence": 0.95,
    "missing_evidence": [
      "predictor/graph evidence not available",
      "no exact/similar/packed graph profiles for current hardware",
      "no measured FP16 vs FP32 comparison yet"
    ]
  }
}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=## Plan **WHAT makes this task unique:** Disaster-tweet classification is dominated by *metaphorical vs.; purpose=Define the Stage 1 candidate target for later precision and training stages.; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=## Plan **WHAT:** The datatype/precision step must define a hardware-aware FP16 AMP policy for the KB-MVNet sparse multi-view network on the A10 (SM 8.6) — where the model is dominated by sparse GEMM + LayerNorm + GEL...; purpose=Support the Stage 1 candidate target with a compatible precision policy.; hardware_keys=autocast_forward, configure_backend_precision, compute_bce_with_logits_fp32, make_grad_scaler, degrade_to_fp32, USE_TF32

# Previous steps
### model_design
**Plan:** ## Plan

**WHAT makes this task unique:** Disaster-tweet classification is dominated by *metaphorical vs. literal* usage of disaster vocabulary ("this song is fire", "my boss is killing me"), plus high-frequency but non-disaster-signaling keywords (e.g. "ablaze", "rain"). Previous attempts all used either plain TF-IDF→MLP (0.7573) or end-to-end sequence nets (0.7341) without conditioning on the keyword categorical and without any metaphor-sensitive stylistic signals.

**WHY my approach fits:** I replace the sequence-model direction with a *keyword-conditioned gated multi-view sparse network* (KB-MVNet) that fuses four heterogeneous evidence views — (1) word TF-IDF, (2) character TF-IDF (captures hashtag/emoji/typos), (3) one-hot keyword view (the strongest categorical prior available in this dataset), (4) 15 hand-crafted stylistic features that proxy metaphor (emoji/emoticon count, uppercase ratio, first-person pronouns, URL/mention structure). A learned soft-max gate selects which view to trust per example, so "ablaze+🌊 and streets flooded" leans on char+style, while "ablaze" alone on word+keyword. Loss is unchanged from the trace: BCE-with-logits → single logit → sigmoid threshold.

**HOW to validate:** Fit on the 85 % internal split, early-stop on the 15 % stratified holdout, and sweep the decision threshold on that holdout to maximize binary F1 — the exact metric. Confirm: val F1 beats 0.7573 and the gate entropy is not collapsed (all four views used). If val F1 < 0.75 or a single view dominates with entropy ≈ 0, I'll revisit fusion.

**Memory / elasticity / backend contract:** All features are in-memory dense float32 view tuples; batch size is `scenario['batch_size']` with no accumulation here (trainer owns it). Word/char dims, hidden width, dropout, and sequence of views remain configurable via context. Token/embedding indices are absent (bag-of-words views) so no long-dtype leakage. Scheduler-owned (NOT implemented here): device assignment, worker count, autocast dtype, GradScaler, optimizer/scheduler cadence, checkpoint cadence.

**Note board:**
- baseline_change: replaced prior TF-IDF→MLP with a 4-view gated keyword-conditioned sparse net (word TF-IDF + char TF-IDF + keyword one-hot + 15 stylistic metaphor proxies) | purpose: directly attack metaphor-vs-literal confusion that dominates this task's error mode, using the keyword categorical as a first-class view rather than as extra text | hardware_keys: none
- baseline_change: added a per-sample softmax gate over the four view embeddings before the classification head | purpose: allow the model to route stylistic/char evidence for hashtag/emoji-heavy tweets and word/keyword evidence for plain news-style tweets, improving F1 on both regimes | hardware_keys: none
- baseline_change: feature bundle is fit strictly on the training fold inside a cached module helper; a no-arg `CandidateAdapter` reuses the same real bundle/model/loss for CPU preflight | purpose: preserve leakage isolation and ensure the preflight exercises the exact production model | hardware_keys: none
**Code:**
```python
# stage: model_design
# Model family: keyword-gated multi-view sparse MLP (word TF-IDF + char TF-IDF + keyword one-hot + stylistic features)
import os
import re
import html
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------ constants
MODEL_FAMILY = "keyword_gated_multiview_sparse_mlp_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
INPUT_DIR = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")

SEED = 42
VAL_FRACTION = 0.15

MAX_WORD_FEATURES = 8000
MAX_CHAR_FEATURES = 12000
MAX_KEYWORD_VOCAB = 300
MIN_KEYWORD_COUNT = 2

HIDDEN_DIM = 256
DROPOUT = 0.35
STYLE_DIM = 15
NUM_VIEWS = 4

# ------------------------------------------------------------------ text utils
_URL_RE = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_WS_RE = re.compile(r"\s+")
_FIRST_PERSON_RE = re.compile(
    r"\b(i|i'm|im|i've|ive|my|me|mine|we|we're|our|us|myself|ourselves)\b",
    re.IGNORECASE,
)
_EMOTICON_RE = re.compile(r"[:;=][\-o\*']?[\)\(dpDP]")
_EMOJI_RE = re.compile(
    "[" "\U0001f300-\U0001faff" "\U00002600-\U000027bf" "\U0001f1e6-\U0001f1ff" "]",
    re.UNICODE,
)


def clean_text(s):
    """Light normalisation that preserves metaphor cues (emoticons, casing, hashtags)."""
    if not isinstance(s, str):
        return ""
    s = html.unescape(s)
    s = _URL_RE.sub(" <url> ", s)
    s = _MENTION_RE.sub(" <user> ", s)
    s = _HASHTAG_RE.sub(r" \1 ", s)  # keep hashtag body as a word token
    s = s.lower()
    s = _WS_RE.sub(" ", s).strip()
    return s


def stylistic_features(raw_text):
    """15 per-tweet stylistic / metaphor-proxy features (always length 15)."""
    t = raw_text if isinstance(raw_text, str) else ""
    n = max(len(t), 1)
    n_upper = sum(1 for c in t if c.isupper())
    n_digit = sum(1 for c in t if c.isdigit())
    n_punct = sum(1 for c in t if c in "!?,.;:'\"")
    n_excl = t.count("!")
    n_q = t.count("?")
    n_hashtag = t.count("#")
    n_mention = t.count("@")
    n_url = 1 if _URL_RE.search(t) else 0
    first_person = len(_FIRST_PERSON_RE.findall(t))
    emoticon = len(_EMOTICON_RE.findall(t))
    emoji = len(_EMOJI_RE.findall(t))
    return np.array(
        [
            float(len(t)),
            float(np.log1p(len(t))),
            n_upper / n,
            n_digit / n,
            n_punct / n,
            float(n_excl),
            float(n_q),
            float(n_hashtag),
            float(n_mention),
            float(n_url),
            float(first_person),
            float(emoticon),
            float(emoji),
            n_excl / n,
            n_q / n,
        ],
        dtype=np.float32,
    )


# ------------------------------------------------------------------ keyword view
def build_keyword_vocab(keywords):
    from collections import Counter

    counts = Counter(
        k.strip().lower() for k in keywords if isinstance(k, str) and k and k.strip()
    )
    ordered = [
        k for k, c in counts.most_common(MAX_KEYWORD_VOCAB) if c >= MIN_KEYWORD_COUNT
    ]
    return {k: i for i, k in enumerate(ordered)}


def keyword_onehot(keywords, vocab):
    unknown_idx = len(vocab)
    out = np.zeros((len(keywords), len(vocab) + 1), dtype=np.float32)
    for i, kw in enumerate(keywords):
        key = kw.strip().lower() if isinstance(kw, str) else ""
        idx = vocab.get(key, unknown_idx)
        out[i, idx] = 1.0
    return out


# ------------------------------------------------------------------ feature bundle
class FeatureBundle(object):
    """Holds every stateful transform, fit on the training fold only."""

    def __init__(self, word_vec, char_vec, kw_vocab, style_scaler):
        self.word_vec = word_vec
        self.char_vec = char_vec
        self.kw_vocab = kw_vocab
        self.style_scaler = style_scaler

    @property
    def word_dim(self):
        return len(self.word_vec.get_feature_names_out())

    @property
    def char_dim(self):
        return len(self.char_vec.get_feature_names_out())

    @property
    def kw_dim(self):
        return len(self.kw_vocab) + 1

    @property
    def style_dim(self):
        return STYLE_DIM

    def transform(self, df):
        texts = [clean_text(t) for t in df["text"].fillna("").astype(str).tolist()]
        word_X = self.word_vec.transform(texts).astype(np.float32).toarray()
        char_X = self.char_vec.transform(texts).astype(np.float32).toarray()
        if "keyword" in df.columns:
            kws = df["keyword"].fillna("").astype(str).tolist()
        else:
            kws = [""] * len(df)
        kw_X = keyword_onehot(kws, self.kw_vocab)
        raw = df["text"].fillna("").astype(str).tolist()
        style = np.stack([stylistic_features(t) for t in raw], axis=0).astype(
            np.float32
        )
        style = self.style_scaler.transform(style).astype(np.float32)
        return word_X, char_X, kw_X, style


def fit_feature_bundle(train_df):
    """Fit the entire feature pipeline on the *training* fold only (no leakage)."""
    texts = [clean_text(t) for t in train_df["text"].fillna("").astype(str).tolist()]

    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=MAX_WORD_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    word_vec.fit(texts)

    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=MAX_CHAR_FEATURES,
        sublinear_tf=True,
        lowercase=True,
    )
    char_vec.fit(texts)

    kws = (
        train_df["keyword"].fillna("").astype(str).tolist()
        if "keyword" in train_df.columns
        else []
    )
    kw_vocab = build_keyword_vocab(kws)

    raw = train_df["text"].fillna("").astype(str).tolist()
    style_raw = np.stack([stylistic_features(t) for t in raw], axis=0).astype(
        np.float32
    )
    style_scaler = StandardScaler().fit(style_raw)

    return FeatureBundle(word_vec, char_vec, kw_vocab, style_scaler)


# ------------------------------------------------------------------ split
def split_data(df, seed=SEED, val_fraction=VAL_FRACTION):
    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        stratify=df["target"].astype(int).values,
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ------------------------------------------------------------------ model
class KeywordGatedMultiviewNet(nn.Module):
    """Fuse four heterogeneous evidence views with a per-sample softmax gate."""

    def __init__(
        self, word_dim, char_dim, kw_dim, style_dim, hidden=HIDDEN_DIM, dropout=DROPOUT
    ):
        super(KeywordGatedMultiviewNet, self).__init__()
        self.word_dim = int(word_dim)
        self.char_dim = int(char_dim)
        self.kw_dim = int(kw_dim)
        self.style_dim = int(style_dim)
        self.hidden = int(hidden)

        def _branch(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.word_proj = _branch(self.word_dim)
        self.char_proj = _branch(self.char_dim)
        self.kw_proj = _branch(self.kw_dim)
        self.style_proj = _branch(self.style_dim)

        self.gate = nn.Sequential(
            nn.Linear(hidden * NUM_VIEWS, hidden),
            nn.GELU(),
            nn.Linear(hidden, NUM_VIEWS),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, word_x, char_x, kw_x, style_x):
        hw = self.word_proj(word_x)
        hc = self.char_proj(char_x)
        hk = self.kw_proj(kw_x)
        hs = self.style_proj(style_x)
        stacked = torch.stack([hw, hc, hk, hs], dim=1)  # [B, 4, H]
        flat = stacked.reshape(stacked.size(0), -1)
        gates = torch.softmax(self.gate(flat), dim=-1).unsqueeze(-1)  # [B, 4, 1]
        fused = (stacked * gates).sum(dim=1)  # [B, H]
        return self.head(fused).squeeze(-1)  # single logit per example


# ------------------------------------------------------------------ loss / criterion
def build_loss():
    """Binary cross-entropy on raw logits (metric: thresholded binary F1)."""
    return nn.BCEWithLogitsLoss()


def _resolve_criterion(context):
    """Return the real criterion; honour caller-supplied mapping if valid."""
    if context:
        crit = context.get("criterion")
        if isinstance(crit, nn.Module):
            return crit
    return build_loss()


# ------------------------------------------------------------------ cached pipeline (for adapter only)
_PIPELINE_CACHE = {}


def _train_csv_path():
    return os.path.join(INPUT_DIR, "train.csv")


def _get_train_df():
    if "train_df" not in _PIPELINE_CACHE:
        _PIPELINE_CACHE["train_df"] = pd.read_csv(_train_csv_path())
    return _PIPELINE_CACHE["train_df"]


def _get_feature_bundle():
    if "bundle" not in _PIPELINE_CACHE:
        df = _get_train_df()
        # Fit on a stratified slice of the train split only (still no val/test labels).
        fit_df, _ = split_data(df)
        _PIPELINE_CACHE["bundle"] = fit_feature_bundle(fit_df)
    return _PIPELINE_CACHE["bundle"]


# ------------------------------------------------------------------ CPU preflight adapter
class CandidateAdapter(object):
    """No-arg CPU-safe adapter that reuses the real model / loss / features."""

    _DEFAULTS = {
        "hidden": HIDDEN_DIM,
        "dropout": DROPOUT,
        "style_dim": STYLE_DIM,
        "lr": 1e-3,
        "weight_decay": 1e-4,
    }

    def _resolve_context(self, context):
        ctx = dict(self._DEFAULTS)
        if context:
            for k, v in context.items():
                if v is not None:
                    ctx[k] = v
        return ctx

    def _resolve_dims(self, context):
        bundle = _get_feature_bundle()
        word_dim = bundle.word_dim
        char_dim = bundle.char_dim
        kw_dim = bundle.kw_dim
        style_dim = bundle.style_dim
        if context:
            if context.get("word_dim"):
                word_dim = int(context["word_dim"])
            if context.get("char_dim"):
                char_dim = int(context["char_dim"])
            if context.get("kw_dim"):
                kw_dim = int(context["kw_dim"])
            if context.get("style_dim"):
                style_dim = int(context["style_dim"])
        return word_dim, char_dim, kw_dim, style_dim

    # ---- construction ----------------------------------------------------
    def build_model(self, context):
        word_dim, char_dim, kw_dim, style_dim = self._resolve_dims(context)
        ctx = self._resolve_context(context)
        return KeywordGatedMultiviewNet(
            word_dim=word_dim,
            char_dim=char_dim,
            kw_dim=kw_dim,
            style_dim=style_dim,
            hidden=int(ctx["hidden"]),
            dropout=float(ctx["dropout"]),
        )

    def build_optimizer(self, model, context):
        ctx = self._resolve_context(context)
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(ctx["lr"]),
            weight_decay=float(ctx["weight_decay"]),
        )

    # ---- batch builders --------------------------------------------------
    def _fixture_batch(self, fixture, device):
        named = {}
        for k, v in fixture.items():
            if isinstance(v, torch.Tensor):
                named[k] = v.to(device)
            elif (
                isinstance(v, (list, tuple))
                and len(v) > 0
                and isinstance(v[0], torch.Tensor)
            ):
                named[k] = v.to(device)
            else:
                named[k] = torch.as_tensor(v, device=device)

        # Prefer named contract keys, otherwise collect every non-target tensor
        # in the fixture order and treat the last labelled tensor as target.
        def _take(*names):
            for n in names:
                if n in named:
                    return named[n]
            return None

        word_x = _take("word_x", "word", "word_ids")
        char_x = _take("char_x", "char", "char_ids")
        kw_x = _take("kw_x", "keyword", "keyword_ids")
        style_x = _take("style_x", "style", "style_features")
        target = _take("target", "label", "labels", "y")

        ordered_inputs = [t for t in (word_x, char_x, kw_x, style_x) if t is not None]
        if not ordered_inputs:
            # Fall back: gather all float tensors except target as inputs
            for k, v in named.items():
                if k in ("target", "label", "labels", "y"):
                    continue
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    ordered_inputs.append(v)
        ordered_inputs = [t.to(dtype=torch.float32) for t in ordered_inputs]

        if target is None:
            n = ordered_inputs[0].size(0) if ordered_inputs else 1
            target = torch.zeros(n, dtype=torch.float32, device=device)
        target = target.to(dtype=torch.float32).view(-1)

        out = {"inputs": tuple(ordered_inputs), "target": target}
        for k, v in named.items():
            out.setdefault(k, v)
        return out

    def _real_batch(self, batch_size, device):
        bundle = _get_feature_bundle()
        df = _get_train_df()
        n = min(int(batch_size), len(df))
        rng = np.random.RandomState(0)
        idx = rng.randint(0, len(df), size=n)
        sub = df.iloc[idx].reset_index(drop=True)
        word_X, char_X, kw_X, style_X = bundle.transform(sub)
        word_x = torch.from_numpy(np.ascontiguousarray(word_X, dtype=np.float32)).to(
            device
        )
        char_x = torch.from_numpy(np.ascontiguousarray(char_X, dtype=np.float32)).to(
            device
        )
        kw_x = torch.from_numpy(np.ascontiguousarray(kw_X, dtype=np.float32)).to(device)
        style_x = torch.from_numpy(np.ascontiguousarray(style_X, dtype=np.float32)).to(
            device
        )
        target = torch.from_numpy(sub["target"].values.astype(np.float32)).to(device)
        return {"inputs": (word_x, char_x, kw_x, style_x), "target": target}

    def _build_batch(self, scenario, device):
        if scenario is None:
            scenario = {}
        batch_size = int(scenario.get("batch_size", 8))
        fixture = scenario.get("fixture") or {}
        if fixture:
            return self._fixture_batch(fixture, device)
        return self._real_batch(batch_size, device)

    def build_train_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    # ---- steps -----------------------------------------------------------
    @staticmethod
    def _unpack(batch):
        if isinstance(batch, dict):
            inputs = batch.get("inputs")
            target = batch.get("target")
        else:
            inputs, target = batch[0], batch[1]
        if not isinstance(inputs, (list, tuple)):
            inputs = (inputs,)
        return inputs, target

    def training_step(self, model, batch, context):
        inputs, target = self._unpack(batch)
        inputs = [
            t.to(dtype=torch.float32) if torch.is_floating_point(t) else t
            for t in inputs
        ]
        target = target.to(dtype=torch.float32).view(-1)
        logits = model(*inputs)
        criterion = _resolve_criterion(context)
        loss = criterion(logits.view(-1), target)
        return loss

    def validation_step(self, model, batch, context):
        with torch.no_grad():
            return self.training_step(model, batch, context)


def build_model_from_bundle(bundle, hidden=HIDDEN_DIM, dropout=DROPOUT):
    """Convenience constructor the training loop can call with the real fitted bundle."""
    return KeywordGatedMultiviewNet(
        word_dim=bundle.word_dim,
        char_dim=bundle.char_dim,
        kw_dim=bundle.kw_dim,
        style_dim=bundle.style_dim,
        hidden=hidden,
        dropout=dropout,
    )

```

### datatype_precision
**Plan:** ## Plan

**WHAT:** The datatype/precision step must define a hardware-aware FP16 AMP policy for the KB-MVNet sparse multi-view network on the A10 (SM 8.6) — where the model is dominated by sparse GEMM + LayerNorm + GELU stacks that are highly FP16-friendly, but where the final BCEWithLogits reduction, gate softmax log-sum, and metric/prediction export must stay FP32.

**WHY:** Normal-mode rules permit only FP32 and selective FP16 AMP; TF32 must be explicitly disabled (A10's FP32-via-TF32 path can silently distort LayerNorm statistics and short-sequence logit tails). The Stage 1 target (winning F1 on metaphor-vs-literal) is sensitive to calibration, so a `GradScaler` protects the gradient signal through the four branch projections and the gate, while the loss, gate-softmax normalizer, and metric accumulators live in a hard FP32 island. We expose this as a *policy object + autocast context + graded scaler helpers*, not a global dtype switch — the model parameters remain FP32.

**HOW:** Validate by comparing val F1 and non-finite-loss counters with `USE_AMP=True` vs `False` on the same fold; require finite loss/gradients and no F1 regression (>0.005 drop) to keep AMP on, otherwise fall back to pure FP32. The helpers below are import-safe (no CUDA touched at definition time), CPU-safe (return FP32/no-op scaler on CPU), and expose `strategy_from_context(ctx)` so the Stage 3 training loop can call them directly.

**Backend contract:** precision utilities are pure helpers; device selection, worker count, and scheduler integration are chosen by the caller (Stage 3). No ops at import time.

**Note board:**
- baseline_change: added a self-contained `PrecisionStrategy` + `autocast_forward` + `configure_backend_precision` + `compute_bce_with_logits_fp32` + `make_grad_scaler` + `degrade_to_fp32` layer, without altering the Stage 1 KB-MVNet model/loss/features/output interface | purpose: give Stage 3 a single importable contract that keeps FP16 AMP in the four branching MLP views and the fused head while forcing FP32 on the BCE reduction, gate-softmax normalizer, metric accumulator, and prediction export — protecting the metaphor-sensitive F1 target | hardware_keys: precision_not_optimized, ampere_sm86_fp16_tc, tf32_disable
- baseline_change: TF32 explicitly disabled via `torch.backends.cuda.matmul.allow_tf32=False` and `cudnn.allow_tf32=False` inside `configure_backend_precision()` (called only from `__main__`), with a documented `USE_TF32` flag defaulting to False | purpose: prevent silent FP32-mantissa truncation on the sparse-view GEMMs that would shift LayerNorm statistics and hurt calibration/F1 selection | hardware_keys: none
- baseline_change: added a `degrade_to_fp32` fallback that flips `USE_AMP=False` when the caller observes repeated non-finite losses | purpose: preserve the Stage 1 model family and training stability without changing any modeling/optimizer/scheduler choices owned by other stages | hardware_keys: precision_fallback
**Code:**
```python
# stage: datatype_precision
"""
Precision / datatype policy utilities for the keyword-gated multi-view sparse MLP.

Design contract (normal mode, NVIDIA A10 SM 8.6):
  * Model weights, floating inputs, and floating labels stay FP32 on all devices.
  * CUDA FP16 AMP is a *selective* option: main autocast region covers the four
    projection branches, the fused gate, and the classification head; the BCE
    reduction and gate-softmax normalizer remain in FP32.
  * GradScaler is required for FP16 AMP; backward + optimizer.step run OUTSIDE
    autocast.
  * TF32 for matmul and cudnn is explicitly DISABLED.
  * CPU preflight always runs FP32 with AMP/scaler disabled.
  * Fallback: if non-finite loss/gradients are observed repeatedly, the caller
    can call `strategy.with_amp_policy(False)` to degrade to pure FP32.
Nothing in this module touches CUDA, reads env vars, or allocates models at
import time; `resolve_precision_strategy()` / `configure_backend_precision()`
are invoked from the caller's `__main__` guard.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# ------------------------------------------------------------------ constants
MODEL_FAMILY_PRECISION_TAG = "kb_mvnet_kw_gated_sparse_v1__fp16_amp"

# Normal-mode allow-list: only FP32 and FP16 AMP.
ALLOWED_PRECISION_POLICIES = ("fp32", "fp16_amp")

# FP32-sensitive ops that must not be allowed to run inside FP16 autocast.
# These are *names* only; the actual FP32-islands are enforced by wrapping the
# corresponding math in `fp32_island()` at the call site (Stage 3).
FP32_SENSITIVE_REGIONS = (
    "bce_with_logits_reduction",
    "gate_softmax_normalizer",
    "metric_accumulation",
    "prediction_export",
)

_logger = logging.getLogger("kb_mvnet.precision")
if not _logger.handlers:  # avoid duplicate handlers when module is re-imported
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[precision] %(levelname)s: %(message)s"))
    _logger.addHandler(_h)
_logger.setLevel(logging.INFO)
_logger.propagate = False


# ------------------------------------------------------------------ config
@dataclasses.dataclass(frozen=True)
class PrecisionStrategy:
    """Immutable snapshot of the active precision policy."""

    device: torch.device
    use_amp: bool
    amp_dtype: torch.dtype  # torch.float16 when use_amp else torch.float32
    use_tf32: bool
    use_grad_scaler: bool
    amp_available: bool  # CUDA + cudnn present and policy allows AMP
    reason: str
    model_family_tag: str = MODEL_FAMILY_PRECISION_TAG

    # ---- derived helpers -------------------------------------------------
    @property
    def is_cpu(self) -> bool:
        return self.device.type == "cpu"

    @property
    def amp_active(self) -> bool:
        return bool(self.use_amp and self.amp_available and not self.is_cpu)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "use_amp": bool(self.use_amp),
            "amp_active": bool(self.amp_active),
            "amp_dtype": str(self.amp_dtype).replace("torch.", ""),
            "use_tf32": bool(self.use_tf32),
            "use_grad_scaler": bool(self.use_grad_scaler),
            "amp_available": bool(self.amp_available),
            "precision_reason": self.reason,
            "model_family_tag": self.model_family_tag,
            "fp32_sensitive_regions": list(FP32_SENSITIVE_REGIONS),
        }

    def with_amp_policy(
        self, use_amp: bool, reason: Optional[str] = None
    ) -> "PrecisionStrategy":
        """Return a copy of this strategy with AMP forced on/off (fallback path)."""
        new_reason = reason or (
            "amp_enabled_by_caller" if use_amp else "amp_disabled_by_caller"
        )
        return dataclasses.replace(
            self,
            use_amp=bool(use_amp),
            amp_dtype=torch.float16 if use_amp else torch.float32,
            use_grad_scaler=bool(use_amp and self.amp_available and not self.is_cpu),
            reason=f"{self.reason} | {new_reason}",
        )


# ------------------------------------------------------------------ resolution
def _cuda_fp16_available() -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def resolve_precision_strategy(
    requested_policy: Optional[str] = None,
    requested_device: Optional[Any] = None,
    allow_amp: Optional[bool] = None,
) -> PrecisionStrategy:
    """
    Resolve the actual precision strategy for the current process.

    CPU => FP32, no AMP, no scaler. CUDA => FP16 AMP by default (normal mode),
    TF32 off. Caller may override with `requested_policy='fp32'` or
    `allow_amp=False`. Unrecognised policies degrade to FP32 with an explicit
    reason rather than raising.
    """
    reason_bits = []

    # ---- device ----------------------------------------------------------
    if requested_device is None:
        device = torch.device("cuda" if _cuda_fp16_available() else "cpu")
        reason_bits.append("device=auto")
    else:
        device = torch.device(requested_device)
        reason_bits.append(f"device=caller({device.type})")

    if device.type == "cuda" and not _cuda_fp16_available():
        reason_bits.append("cuda_requested_unavailable->cpu")
        device = torch.device("cpu")

    is_cpu = device.type == "cpu"
    amp_available = not is_cpu

    # ---- policy ----------------------------------------------------------
    policy = (requested_policy or "fp16_amp").lower()
    if policy not in ALLOWED_PRECISION_POLICIES:
        reason_bits.append(f"policy:{policy}->fp32(fallback)")
        policy = "fp32"

    if allow_amp is False and policy != "fp32":
        reason_bits.append("amp_disabled_by_caller")
        policy = "fp32"

    if is_cpu:
        reason_bits.append("cpu_forces_fp32")
        policy = "fp32"
        amp_available = False

    use_amp = policy == "fp16_amp" and amp_available
    amp_dtype = torch.float16 if use_amp else torch.float32
    use_grad_scaler = bool(use_amp)
    use_tf32 = False  # normal-mode contract: TF32 disabled

    if use_amp:
        reason_bits.append("fp16_amp_selective_forward")
    else:
        reason_bits.append("fp32_full_precision")

    reason = "; ".join(reason_bits) or "default"

    return PrecisionStrategy(
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        use_tf32=use_tf32,
        use_grad_scaler=use_grad_scaler,
        amp_available=amp_available,
        reason=reason,
    )


def strategy_from_context(context: Optional[Dict[str, Any]]) -> PrecisionStrategy:
    """
    Build a PrecisionStrategy from a partial context mapping (as passed by the
    scheduler / training entrypoint). Missing keys fall through to defaults.
    """
    ctx = dict(context or {})
    requested_policy = ctx.get("precision_policy")
    if requested_policy is None:
        # Fall back to whatever the Pipeline Decision Trace recorded.
        requested_policy = ctx.get("requested_precision_policy", "fp16_amp")
    requested_device = ctx.get("device") or ctx.get("requested_device")
    allow_amp = ctx.get("allow_amp")
    if allow_amp is None and ctx.get("use_amp") is False:
        allow_amp = False
    return resolve_precision_strategy(
        requested_policy=requested_policy,
        requested_device=requested_device,
        allow_amp=allow_amp,
    )


# ------------------------------------------------------------------ backend
def configure_backend_precision(strategy: PrecisionStrategy) -> Dict[str, Any]:
    """
    Apply process-level precision flags. MUST be called from `__main__`, never
    at import time. Disables TF32 for both matmul and cudnn, and (when AMP is
    active) configures the cuDNN benchmark flags conservatively.
    """
    applied: Dict[str, Any] = {"tf32_matmul": False, "tf32_cudnn": False}

    # TF32 is always disabled in normal mode.
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    if strategy.device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = False  # shapes vary per batch
        except Exception:
            pass
        try:
            # Reduce fragmentation on A10's 22 GB VRAM.
            torch.cuda.empty_cache()
        except Exception:
            pass

    applied["torch_amp_available"] = bool(torch.cuda.is_available()) and (
        strategy.device.type == "cuda"
    )
    applied["strategy"] = strategy.as_dict()
    return applied


# ------------------------------------------------------------------ autocast
@contextlib.contextmanager
def autocast_forward(strategy: PrecisionStrategy):
    """
    Autocast context used strictly around eligible forward + loss-compatible
    computation. On CPU or when AMP is inactive this is a no-op context so that
    callers can use the same code path everywhere.
    """
    if not strategy.amp_active:
        yield
        return
    try:
        with torch.autocast(device_type="cuda", dtype=strategy.amp_dtype, enabled=True):
            yield
    except TypeError:
        # Older PyTorch signature.
        with torch.cuda.amp.autocast(dtype=strategy.amp_dtype, enabled=True):
            yield


@contextlib.contextmanager
def fp32_island(strategy: PrecisionStrategy):
    """
    Force FP32 math for FP32-sensitive regions (gate-softmax normalizer, BCE
    reduction, metric accumulation, prediction export). This is a no-op when
    AMP is inactive.
    """
    if not strategy.amp_active:
        yield
        return
    try:
        with torch.autocast(device_type="cuda", enabled=False):
            yield
    except TypeError:
        with torch.cuda.amp.autocast(enabled=False):
            yield


# ------------------------------------------------------------------ scaler
def make_grad_scaler(strategy: PrecisionStrategy):
    """
    Return a GradScaler when FP16 AMP is active, else None. Kept as a thin
    factory so the trainer can pass it straight into diagnostics.
    """
    if not strategy.use_grad_scaler or not strategy.amp_active:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def clip_gradients_(parameters, max_norm: float, scaler) -> None:
    """
    Unscale (when a scaler is active) then clip in-place. Must be called
    OUTSIDE autocast, right before optimizer.step().
    """
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        try:
            scaler.unscale_(parameters)
        except Exception:
            # If unscale_ errors (e.g. already unscaled), continue to clip on
            # current grads rather than crashing the training step.
            pass
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


# ------------------------------------------------------------------ FP32 loss
def compute_bce_with_logits_fp32(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    BCE-with-logits reduction run in FP32 regardless of ambient autocast, to
    protect the loss signal that drives threshold selection and F1 calibration.
    """
    logits_f = logits.float()
    target_f = target.float().view_as(logits_f)
    with fp32_island(resolve_precision_strategy(requested_policy="fp32")):
        return nn.functional.binary_cross_entropy_with_logits(logits_f, target_f)


def is_finite_loss(loss: torch.Tensor) -> bool:
    if loss is None:
        return False
    try:
        return bool(torch.isfinite(loss.detach()).all().item())
    except Exception:
        return False


# ------------------------------------------------------------------ fallback
class DegradeToFP32Tracker:
    """
    Watches for repeated non-finite losses; after `patience` bad steps the
    tracker flips to FP32 via `current_strategy()`. Simple, deterministic, and
    caller-driven (no background state).
    """

    def __init__(self, strategy: PrecisionStrategy, patience: int = 3):
        self._strategy = strategy
        self._patience = max(1, int(patience))
        self._bad = 0
        self.degraded = False

    @property
    def strategy(self) -> PrecisionStrategy:
        return self._strategy

    def observe(self, loss: Optional[torch.Tensor]) -> bool:
        """Record one loss value. Returns True when a fallback was triggered."""
        if self.degraded:
            return False
        if is_finite_loss(loss):
            self._bad = 0
            return False
        self._bad += 1
        if self._bad >= self._patience:
            self._strategy = self._strategy.with_amp_policy(
                False, reason="non_finite_loss_patience_exceeded"
            )
            self.degraded = True
            return True
        return False

    def current_strategy(self) -> PrecisionStrategy:
        return self._strategy


# ------------------------------------------------------------------ logging
def format_precision_log(strategy: PrecisionStrategy) -> str:
    d = strategy.as_dict()
    return (
        "precision | "
        f"device={d['device']} amp={d['amp_active']}({d['amp_dtype']}) "
        f"tf32={d['use_tf32']} scaler={d['use_grad_scaler']} "
        f"family={d['model_family_tag']} reason={d['precision_reason']}"
    )


def precision_to_dict(strategy: PrecisionStrategy) -> Dict[str, Any]:
    """Serialisable snapshot for checkpoints / diagnostics settings."""
    return strategy.as_dict()


# ------------------------------------------------------------------ public entry
def build_precision_context(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience bundle for the training entrypoint: strategy + scaler + log.
    Callers must invoke this inside `__main__`, not at import time (it touches
    CUDA when a GPU is present via `configure_backend_precision`).
    """
    strategy = strategy_from_context(context)
    applied = configure_backend_precision(strategy)
    scaler = make_grad_scaler(strategy)
    return {
        "strategy": strategy,
        "scaler": scaler,
        "backend_applied": applied,
        "log_line": format_precision_log(strategy),
        "settings": precision_to_dict(strategy),
    }

```

# Current step: training_evaluation
Implement the training loop, validation, metric tracking, model saving, and generate submission file.


# Instructions

## Response format

Your response should be:
1. A brief plan (2-3 sentences) describing what you will do in this step
2. A `Note board:` block with 1-3 bullets. Each bullet must use this exact shape: `- baseline_change: <what changed from the baseline or prior step> | purpose: <why this supports the candidate target> | hardware_keys: <comma-separated local feature keys or none>`
3. A single markdown code block (wrapped in ```) containing ONLY the code for this step
IMPORTANT: Do NOT write code for other steps. Only write code for the current step.
No module-level side effects: this step may define imports, constants, classes, and helper functions only. Do not access data, read environment settings, configure CUDA/TF32, configure logging, instantiate models/losses, construct DataLoaders, train, evaluate, or write submissions at import time; expose helpers for the final main guard instead.

## Hardware/Profile reasoning rule

- HWDB recommended_patterns are conditional candidates, not measured speedups or mandatory steps. Check workload, installed backend, allowed precision and each stated limitation before applying one. A capability or verified flag does not prove benefit for this candidate. Keep the baseline when evidence is missing; record feature keys, unmet conditions and measured results without inventing them.
- Hardware advice never overrides task, dataset, submission, package, model-source, or filesystem constraints. Do not invent Kaggle input paths, local checkpoints, torch hub directories, or placeholder weights; use only paths and model sources that are explicitly available in the prompt/config.
- For hardware/runtime optimization, preserve the parent solution's modeling budget unless the user explicitly asks to improve score. Do not increase epochs, folds, model size, input resolution, ensemble count, TTA, dataset size, or validation workload as a hardware-only optimization.
- Generated code must run as one independent subprocess and follow the exact effective backend named in the backend-guidance section.
- Use backend coding preferences in job code, but leave daemon setup, launch policy, device assignment, active-thread allocation, priorities, and memory limits to the scheduler.
- Prefer scheduler-compatible code with configurable batch size, precision, dataloader workers, checkpoints, and runtime logging when using GPU training.
- The stage-specific hardware node response is already provided in the prompt. Do not query the hardware node again; query local feature-node details only for selected feature keys when deeper implementation guidance is needed.
- Use the Cross-Stage Note Board to keep Stage 2 precision and Stage 3 training choices aligned with the Stage 1 candidate target.


## Pipeline Decision Contract

- Use the Pipeline Decision Trace as the source of truth for code generation and planning.
- Follow the hardware-aware stepwise workflow: model-design -> datatype/quantization -> training (code keys: model_design -> datatype_precision -> training_evaluation).
- Do not jump directly to optimizer, precision, or batch-size choices before model family and output interface are decided.
- The model_design stage stores data modality, target shape, preprocessing/features, model family, loss, criterion, and output interface.
- Numeric precision such as fp32, fp16, bf16, tf32, or TE FP8/NVFP4 belongs under datatype_precision.precision_policy unless the task explicitly requires a numeric type.
- The datatype_precision step may make narrow precision-required model adapters such as Transformer Engine layer wrappers/replacements, padding/config hooks, autocast recipes, or higher-precision islands, but it must preserve the Stage 1 model family, loss, data features, and output interface.
- Hardware/profile evidence may influence model fit and tuning only when it is compatible with the task, installed packages, and available model sources.
- Hardware-only training/runtime tuning must not increase epochs, folds, model size, image resolution, ensemble count, TTA, dataset size, or validation workload.
- If execution feedback contradicts an earlier decision, update only the relevant part of the decision and explain why in the plan.


## 🔬 Critical: Scientific Approach to Design

- 
- Before designing your solution, you must answer three fundamental questions:
- 
- 1. **WHAT makes this task unique?**
- - Not generic observations like 'it's a classification task'
- - What SPECIFIC patterns, challenges, or domain characteristics?
- 
- 2. **WHY is your approach suitable for this task?**
- - Not just 'this model is good' - explain the MATCH between approach and task
- - What properties of your method address the task characteristics?
- 
- 3. **HOW will you validate your hypothesis?**
- - What outcome would confirm your approach is right?
- - What outcome would suggest you need to reconsider?
- 
- ---
- 
- ⚠️ This is not a template to fill - this is how scientists think.
- Blindly applying standard methods without understanding WHY is not acceptable.
- 
- Your plan should naturally reflect this reasoning process.


## Solution sketch guideline

- - This first solution design should be relatively simple — avoid complex ensemble strategies or extensive hyperparameter searches at this stage.
- - 🎯 **CRITICAL: NOVELTY & DIVERSITY REQUIREMENT**:
- • **Mandatory**: Your solution MUST be NOVEL compared to ALL existing attempts in Memory.
- • **Step 1**: Carefully analyze the core idea of EACH previous attempt in Memory.
- • **Step 2 - Choose Strategy**:
- → **Option A (Preferred)**: Propose a COMPLETELY DIFFERENT approach exploring an untried direction.
- → **Option B**: Build upon an existing approach BUT add significant novel insights that fundamentally change the solution.
- • **Forbidden**: Minor variations (changing hyperparameters, swapping similar models, tweaking preprocessing).
- • **Think**: 'Does my approach explore a fundamentally different hypothesis?' If NO → redesign.
- - Don't propose the same modelling solution but keep the evaluation the same.
- - Your plan should be concise but comprehensive: Must address WHAT/WHY/HOW (2-4 sentences each). Avoid verbosity - every sentence should add new insight. Natural length: around 8-12 sentences for a complete reasoning process.
- - Propose an evaluation metric that is reasonable for this task.
- - Don't suggest to do EDA.
- - The data is already prepared in `./input` directory. No need to unzip files.


## Coding & Execution Guidelines (CRITICAL)

- - **NO PROGRESS BARS**: You MUST NOT use `tqdm`. Assume `tqdm` is not installed. Use standard Python loops only. Do not use `verbose=1`.
- - **MINIMAL LOGGING**: Print ONLY 1 line per epoch (e.g. loss/accuracy). Do NOT print batch-level logs.
- - **FINAL OUTPUT**: The VERY LAST line of execution MUST be `print(f'Final Validation Score: {score}')`. This is required for the score parser.


## Backend design contract

- - In the plan, explicitly state the selected memory strategy and physical/effective batch elasticity strategy.
- - State operator and normalization considerations, the model dimensions that remain configurable, and the fallback behavior.
- - List scheduler-owned backend controls that the generated subprocess must not implement.
- - Do not assume code-level integration, shared framework state, or synchronized start time with another job.


## CPU preflight adapter contract

- - **CPU PREFLIGHT CONTRACT (MANDATORY)**: Define a no-argument `class CandidateAdapter` in the generated file. It must implement `build_model(context)`, `build_optimizer(model, context)`, `build_train_batch(scenario, device)`, `build_validation_batch(scenario, device)`, `training_step(model, batch, context)`, and `validation_step(model, batch, context)`.
- - The adapter must exercise the same real model, loss, inputs, and optimizer as the training pipeline; do not use mock tensors or a different toy model. Its batch builders must honor `scenario['batch_size']` and return all inputs required by the model plus a target.
- Batch builders must return (inputs, target) or a mapping with recognized `inputs` and `target` keys. For multiple positional model inputs, use {'inputs': (word_ids, char_ids, keyword_ids), 'target': targets}; training_step and validation_step must unpack those same inputs into the real model. Custom top-level keys alone (such as word_ids) are not recognized. Preserve any authoritative named fixture keys, adding an inputs tuple that references those same tensors when needed. Token/embedding indices remain torch.long and masks keep their required dtype; FP32 precision rules apply to floating-point computation, not integer indices. The checker owns zero_grad, backward and optimizer.step: training_step only computes and returns the connected scalar loss, without performing those updates itself.
- - Keep imports, constants, class/function definitions, and read-only device configuration import-safe. Put all training, validation, prediction, and submission side effects under `if __name__ == '__main__':`.


## Implementation guideline

- Milestone objective: complete one real candidate through review, CPU preflight, GPU training, validation and test submission. There is no overall wall-clock deadline or failed-node budget. Max execution time per run = an hour. Start with a compact trainable model and a complete executable pipeline; do not fabricate metrics, bypass checks, or use dummy predictions.
- 
- Code execution must complete within the configured per-run limit of an hour, including validation and test inference.
- 🎯 **CRITICAL REQUIREMENTS** (Non-Negotiable):
- 
- **1. Model Inference for ALL Predictions**
- • EVERY prediction (validation & test) MUST come from trained model's forward pass
- • Process: Load data → Preprocess → model.predict()/model.forward() → Save predictions
- • ❌ FORBIDDEN: Constants, placeholders, dummy values, empty arrays, statistics, random numbers
- • ❌ FORBIDDEN: Fake/mock metric functions (must use real sklearn.metrics or correct manual implementation)
- • Why: Shortcuts create fake high validation scores but fail on test (CRITICAL SYSTEM FAILURE)
- 
- **2. Generate submission.csv**
- • Path: `./submission/submission.csv` (NOT ./working/submission.csv)
- • Content: Model predictions on ALL test samples
- • Format: Follow task description exactly
- 
- **3. Print Validation Metric**
- • MUST print: `print(f'Final Validation Score: {score}')`
- • Score MUST be computed on hold-out validation set using proper metric formula
- • CRITICAL CONSISTENCY REQUIREMENT: Ensure that validation and test inference use IDENTICAL processing logic. Any differences in how validation and test data are handled (such as post-processing, reconstruction, or formatting) can cause large performance gaps between validation and test sets. Maintain consistency across all data processing steps for both validation and test phases.
- 
- **4. Scheduler Model Family Contract**
- • MUST define a top-level constant `MODEL_FAMILY = "family_name"` near the top of the file.
- • Use a stable, architecture-specific name, for example `resnet50_224`, `swin_b_384`, or `lightgbm_tabular_v1`.
- • If you switch the model architecture or input geometry in an improvement/evolution, update `MODEL_FAMILY` to a new distinct value.
- • If there is no more specific model/backbone name in the script, set the model name/key variable to the same value as `MODEL_FAMILY`.
- 
- **5. CPU Model-Preflight Adapter Contract**
- • The module MUST be import-safe: definitions and lightweight constants may be top-level, but all training, validation, inference, and submission execution MUST be called only inside `if __name__ == "__main__":`.
- • MUST define a no-argument top-level `CandidateAdapter` class with methods `build_model(context)`, `build_optimizer(model, context)`, `build_train_batch(scenario, device)`, `build_validation_batch(scenario, device)`, `training_step(model, batch, context)`, and `validation_step(model, batch, context)`. build_model(context) MUST return the real torch.nn.Module used by training, never a dict, tuple, or metadata wrapper.
- • `training_step` MUST return the real scalar loss tensor connected to model parameters; adapter methods must reuse the script's real model, loss, transforms, collate logic, and optimizer rather than mocks.
- • Treat every caller-supplied `context` as a partial mapping: merge it over adapter defaults before reading optional keys, and preserve caller-provided values.
- • CandidateAdapter context mutations do not persist between checker method calls; `training_step` and `validation_step` MUST resolve the script's real criterion when the partial context omits it or supplies `None`, rather than relying on `build_model` to mutate a local context.
- • Batch builders MUST honor `scenario['batch_size']` and `device`; use `os.environ.get('MLEVOLVE_INPUT_DIR', './input')` to locate input fixtures during the isolated CPU check.
- Batch builders must return (inputs, target) or a mapping with recognized `inputs` and `target` keys. For multiple positional model inputs, use {'inputs': (word_ids, char_ids, keyword_ids), 'target': targets}; training_step and validation_step must unpack those same inputs into the real model. Custom top-level keys alone (such as word_ids) are not recognized. Preserve any authoritative named fixture keys, adding an inputs tuple that references those same tensors when needed. Token/embedding indices remain torch.long and masks keep their required dtype; FP32 precision rules apply to floating-point computation, not integer indices. The checker owns zero_grad, backward and optimizer.step: training_step only computes and returns the connected scalar loss, without performing those updates itself.
- • When `scenario['fixture']` is non-empty, it is the authoritative named input contract: build every listed tensor with the declared non-batch shape, preserve those names in the returned mapping, and pass all model inputs to the real forward path. Do not collapse a multimodal fixture into one feature vector.
- • Adapter construction and a one-batch CPU step must be lightweight and must not download weights or require CUDA.
- 
- 📁 **Directories**: Input data in `./input/`, submission in `./submission/`, temp files in `./working/`
- 
- 📦 **Packages & Internet**: Guaranteed installed packages: numpy, pandas, scikit-learn, statsmodels, xgboost, lightgbm, torch, torchvision, torch-geometric, bayesian-optimization, timm, transformers, sentence-transformers, opencv-python, Pillow. Do not assume unlisted third-party packages are installed; avoid them or provide an ImportError-guarded fallback. Never run pip from generated code. Generated code runs offline: do not download weights or data through torch.hub.load(), Hugging Face, timm, or another remote loader. Use only verified local or cached weights.
- 
- ⚠️ **API Compatibility**: LightGBM/XGBoost: ❌ `fit(..., early_stopping_rounds=...)` → ✅ LightGBM: `fit(..., callbacks=[lgb.early_stopping(...)])` ✅ XGBoost: `XGBClassifier(early_stopping_rounds=...)`
- • AdamW: ❌ `from transformers import AdamW` (deprecated) → ✅ `from torch.optim import AdamW`
- 
- 🚫 **Execution Guidelines**:
- • NO tqdm (not installed), NO verbose=1
- • Print only 1 line per epoch (minimize logging)
- • Runtime diagnostics are an exception: one diagnostics line per epoch plus context entry/exit; no per-batch printing.
- PyTorch runtime diagnostics: import TrainingDiagnostics from utils.training_diagnostics. Inside the training entrypoint, wrap training in `with TrainingDiagnostics(model, optimizer, scaler=scaler_or_None, settings={...}) as diagnostics:` after model/optimizer initialization. Settings must include actual physical_batch_size, effective_batch_size, planned_epochs, steps_per_epoch, model_family, split_seed and precision choice/reason. Call diagnostics.after_update() once after each optimizer update ATTEMPT, after scaler.step and scaler.update (or optimizer.step); never after accumulation microbatches. The helper observes real optimizer steps, including skips, and actual autocast dtype. Call diagnostics.report(epoch=one_based_epoch) once per epoch; keep the normal epoch metric line. Save diagnostics.state_dict() with the training checkpoint and restore it using diagnostics.load_state_dict(...). Preserve these calls during merges/repairs. Do not replace missing measurements with zero. Keep the final validation score as the last line, after leaving the diagnostics context. These diagnostics also run without the scheduler.
- • Use DataLoader with num_workers>=2 for speed
- 
- ⚠️  **Self-Check Before Finalizing**:
- □ Did predictions pass through model's learned weights during inference? (If NO → INVALID)
- □ Did I generate submission.csv in correct path with ALL test predictions?
- □ Did I print validation metric as the last line?
- □ Did I use the COMPLETE training dataset (not a tiny subset)?
- □ Is execution behind the main guard and is CandidateAdapter complete and CPU-safe?
- □ Does scheduled PyTorch training use script_scheduler_context, load_resume_checkpoint, and STEP/EPOCH safe_point calls with steps_per_epoch and a complete state_factory? Preserve these hooks during merges and repairs; keep setup inside the training entrypoint and hooks inactive when the context is None.
- Use the installed API exactly: context.control_hook.safe_point(SafePointType.STEP, epoch=epoch, global_step=global_step, steps_per_epoch=steps_per_epoch, state_factory=state_factory). Use the same keyword arguments for BEFORE_TRAIN and EPOCH. There is no payload= parameter and no second positional argument. Import script_scheduler_context from localml_scheduler.execution.script_context and SafePointType from localml_scheduler.domain. Guard hooks with `if context is not None`; do not catch and suppress hook/checkpoint exceptions. Define state_factory to capture actual model, optimizer, scaler, LR scheduler, RNG, global_step and exact step_in_epoch/sampler_state/data_state. Restore those same state objects and loader position before continuing. Use direct hook calls and TrainingDiagnostics construction so the static admission gate can verify the contract; do not add unreachable token-matching calls.
- 
- The code for this step must be self-contained and can be integrated with other steps.
- Use clear variable names that are consistent with previous steps.
- Do not duplicate code from previous steps - assume those parts already exist.
- Make sure to handle edge cases appropriately.


## Precision policy

- Normal mode permits FP32 and selective FP16 AMP only, not BF16, TF32, quantization or TE recipes. FP16 is an option, not a requirement or a whole-pipeline dtype. Keep model parameters and floating inputs and floating regression targets in FP32; preserve integer indices and labels. Use CUDA autocast only around eligible forward operations and compatible loss computation. Run backward and optimizer updates outside autocast, with GradScaler for FP16 and unscale before gradient clipping. For fragile losses/reductions, disable autocast locally and explicitly cast inputs to FP32. Keep metric accumulation and prediction export in FP32; validation/test forwards may share a validated autocast path. Do not call model.half() or cast the entire batch to half. CPU preflight uses FP32 with AMP/scaling disabled. Disable TF32 for matmul and convolution. Compare finite loss/gradients, task quality and elapsed time against FP32; retain FP32 for unsupported, unstable or slower operations. Record the actual regions and fallback, not an unmeasured speedup.


## 🚨 DATA LEAKAGE PREVENTION

- 
- ⚠️ **Strict Isolation Principle**: Validation/Test data must remain strictly unseen during training.
- 
- ✅ **Sequence**: Always **Split Data FIRST**, then apply processing.
- ✅ **Stateful Transformations**: Fit all Scalers, Encoders, Imputers, and Tokenizers **ONLY on Training data**, then use `.transform()` on Validation/Test.
- ✅ **Feature Engineering**: Calculate global statistics (e.g., mean, variance, vocabulary) solely from the Training set.
- ✅ **Target Leakage**: Never use target information (e.g., Target Encoding) from the validation set.
- 


## Installed Packages

The following packages are guaranteed in the execution environment: `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `xgboost`, `lightgbm`, `torch`, `torchvision`, `torch-geometric`, `bayesian-optimization`, `timm`, `transformers`, `sentence-transformers`, `opencv-python`, `Pillow`. Do not assume any unlisted third-party package is installed: either avoid it or use an ImportError-guarded fallback. Do not run pip from generated code. For neural networks we suggest using PyTorch rather than TensorFlow.

## 💡 Recommendation: Robustness & Generalization Strategy

- 
- **To improve model robustness and generalization on unseen data:**
- 
- ✅ **Architecture**: Match model inductive bias to data structure (e.g., CNNs/ViTs for spatial grids, Transformers/RNNs for sequences, GNNs/GCNs for graphs/topology)
- ✅ **Input Strategy**: Handle variable-length or large-scale inputs via **windowing strategies** or patch-based processing (consider overlap for smoother predictions)
- ✅ **Regularization**: Consider using Dropout, Batch/Layer Norm, Weight Decay, or Label Smoothing
- ✅ **Loss Function**: Inspect class distribution and adapt loss accordingly (e.g., weighted loss, FocalLoss, or task-specific objectives)
- ✅ **Learning Rate**: Consider using adaptive schedules like Cosine Annealing or ReduceLROnPlateau or Warmup with differential rates if needed
- ✅ **Data Augmentation**: Apply domain-appropriate augmentation based on data modality (e.g., geometric transforms, masking, mixup)
- ✅ **Validation**: Monitor validation metrics strictly and use early stopping to prevent overfitting
- 
- ⚠️ **Note**:
- Prioritize capturing the intrinsic structure of the data (Inductive Bias) over simply increasing model size.
- 


## training_evaluation guidelines

- - Your responsibility: Stage 3 hardware-aware training hyperparameter optimization. Write the training loop using the data/features, model or precision-adapted model, loss/interface, and datatype_precision variables from previous stages. Define optimizer, learning rate, weight decay, scheduler, physical batch size, gradient accumulation, dataloader workers, checkpoint cadence, validation, test inference, and `submission.csv`.
- Read the Cross-Stage Note Board first and align optimizer, batch, dataloader, runtime, and fallback choices with the Stage 1 candidate target and Stage 2 precision policy.
- CRITICAL: Assume all previous code steps have already been executed. Do NOT redefine or reload data/features, redesign the model/loss, undo precision-required model adapters, or replace the datatype_precision policy. Consume those variables and utilities AS-IS.
- CRITICAL: Own training hyperparameters in this step: batch size, effective batch size, accumulation steps, epochs, learning rate, weight decay, scheduler, early stopping, dataloader workers, pin_memory, persistent_workers, checkpointing, and runtime logging.
- CRITICAL: Use the datatype_precision variables/utilities and any precision-adapted model object for autocast, GradScaler, TF32, TE recipes, and fallback handling. Do NOT choose a different dtype policy unless the previous precision settings are impossible to use, and then keep a safe fallback.
- PyTorch runtime diagnostics: import TrainingDiagnostics from utils.training_diagnostics. Inside the training entrypoint, wrap training in `with TrainingDiagnostics(model, optimizer, scaler=scaler_or_None, settings={...}) as diagnostics:` after model/optimizer initialization. Settings must include actual physical_batch_size, effective_batch_size, planned_epochs, steps_per_epoch, model_family, split_seed and precision choice/reason. Call diagnostics.after_update() once after each optimizer update ATTEMPT, after scaler.step and scaler.update (or optimizer.step); never after accumulation microbatches. The helper observes real optimizer steps, including skips, and actual autocast dtype. Call diagnostics.report(epoch=one_based_epoch) once per epoch; keep the normal epoch metric line. Save diagnostics.state_dict() with the training checkpoint and restore it using diagnostics.load_state_dict(...). Preserve these calls during merges/repairs. Do not replace missing measurements with zero. Keep the final validation score as the last line, after leaving the diagnostics context. These diagnostics also run without the scheduler.
- Use the installed API exactly: context.control_hook.safe_point(SafePointType.STEP, epoch=epoch, global_step=global_step, steps_per_epoch=steps_per_epoch, state_factory=state_factory). Use the same keyword arguments for BEFORE_TRAIN and EPOCH. There is no payload= parameter and no second positional argument. Import script_scheduler_context from localml_scheduler.execution.script_context and SafePointType from localml_scheduler.domain. Guard hooks with `if context is not None`; do not catch and suppress hook/checkpoint exceptions. Define state_factory to capture actual model, optimizer, scaler, LR scheduler, RNG, global_step and exact step_in_epoch/sampler_state/data_state. Restore those same state objects and loader position before continuing. Use direct hook calls and TrainingDiagnostics construction so the static admission gate can verify the contract; do not add unreachable token-matching calls.
- Hardware graph contract: the stage-specific hardware node response is already attached in the Hardware/Profile Optimization Context. Do not query the hardware node again; query local feature-node details only for selected feature keys when deeper guidance is needed, and never fetch external URLs.
- Note board: record how training/runtime choices preserve the Stage 1 target and use the Stage 2 precision policy.
- CRITICAL: Validation metric computation must use the same prediction method as test inference, using training data only as reference, to avoid data leakage and ensure the metric reflects true generalization performance.
- CRITICAL CONSISTENCY REQUIREMENT: Ensure that validation and test inference use IDENTICAL processing logic. Any differences in how validation and test data are handled can cause large performance gaps between validation and test sets.
- CRITICAL: You MUST actively prevent overfitting. Use appropriate regularization, early stopping, and validation discipline without overfitting to the validation set.
- CRITICAL SCHEDULER CONTRACT: For iterative neural training, implement validation-metric-based early stopping with patience and best-checkpoint restore. Emit one parseable line per completed epoch as `MLEVOLVE_EPOCH_METRIC {"epoch": one_based_epoch_number, "metric": validation_metric, "metric_name": metric_name}` so planned, completed, and best epochs remain distinct.
- SCHEDULER STEP CONTROL: Inside the training entrypoint, before CUDA allocations, import and call script_scheduler_context from localml_scheduler.execution.script_context. It returns None outside scheduled execution. When a context exists, restore context.load_resume_checkpoint()['state'] and call context.control_hook.safe_point(SafePointType.BEFORE_TRAIN, ...) after initialization, SafePointType.STEP after each complete optimizer update (not accumulation microbatches), and SafePointType.EPOCH after each epoch. Supply epoch, global_step, steps_per_epoch and state_factory preserving model, optimizer, GradScaler, LR scheduler, Python/NumPy/Torch/CUDA RNG, and exact sampler/data position. Import SafePointType from localml_scheduler.domain. Restore data position before continuing and do not replay completed updates. Set context.job.max_epochs and context.job.config.max_epochs to the actual training budget and context.store.save_job(context.job) before the first safe point. The scheduler owns yielding, measurement and memory limits; do not implement custom process signals or change training semantics for a trial.
- CRITICAL CHECKPOINT COMPATIBILITY: When restoring a checkpoint written by this candidate and its payload includes optimizer state, metadata, NumPy values, or configuration dictionaries, call `torch.load(checkpoint_path, map_location=device, weights_only=False)`. Use this only for the candidate's own trusted local checkpoint; do not load arbitrary external files.
- CRITICAL BATCH CONTRACT: Define `QUALITY_SAFE_PHYSICAL_BATCH_SIZES` as the physical batch sizes you judge quality-safe, including the proposed batch. Also define `BATCH_LR_SCALING_POLICY` as `fixed`, `linear`, or `sqrt`. The scheduler may choose only inside this envelope and will jointly resolve accumulation, effective batch, learning rate, warmup, and scheduler steps.
- Do not reduce epoch count merely because physical batch increases. Keep planned epochs tied to data exposure and let validation early stopping reduce completed epochs; preserve effective batch through accumulation when possible.
- CRITICAL: You MUST implement the exact evaluation metric as specified in the task description's 'Evaluation' section. Do not use dummy, simplified, or approximate metrics.
- CRITICAL: The final line must be: `print(f'Final Validation Score: {score}')`. This is required for the score parser.
- Hardware-aware training: optimize runtime at fixed modeling intent. Do NOT increase epochs, folds, model size, input resolution, ensemble count, TTA, dataset size, or validation workload as a hardware-only optimization unless the user explicitly asks for score improvement.
- Allowed hardware optimizations in this stage: physical batch size, gradient accumulation while preserving effective batch size, dataloader workers, pin_memory, persistent_workers, channels_last, safe torch.compile, checkpointing, and runtime logging. Precision choices and precision-required model adapters must consume the datatype_precision policy.
- Scheduler-aware training: run as an independent subprocess and follow the one canonical effective backend in the Hardware/Profile Optimization Context. Keep scheduler-owned launch and resource controls out of job code, and never integrate with or launch sibling jobs.
- Hardware-aware training: choose physical batch size, accumulation, checkpoint cadence, and dataloader settings from evidence. On OOM/timeout, preserve model/input geometry, data exposure and effective batch; record smaller physical batch, accumulation or checkpoint-resume fallbacks instead of silently reducing the task.
- When feasible, log resolved batch size, selected precision, elapsed time, throughput, and peak CUDA memory so later scheduler graph evidence can learn from this run.
- Pipeline decision: consume `training_evaluation.optimizer`, `training_evaluation.scheduler`, `training_evaluation.batch_size_policy`, `training_evaluation.dataloader_policy`, `training_evaluation.fallbacks`, and `evidence` for training, validation, checkpointing, submission generation, and runtime logging.
- Pipeline decision: when evidence is missing, implement the recorded safe fallbacks rather than inventing hardware claims.
- Normal mode permits FP32 and selective FP16 AMP only, not BF16, TF32, quantization or TE recipes. FP16 is an option, not a requirement or a whole-pipeline dtype. Keep model parameters and floating inputs and floating regression targets in FP32; preserve integer indices and labels. Use CUDA autocast only around eligible forward operations and compatible loss computation. Run backward and optimizer updates outside autocast, with GradScaler for FP16 and unscale before gradient clipping. For fragile losses/reductions, disable autocast locally and explicitly cast inputs to FP32. Keep metric accumulation and prediction export in FP32; validation/test forwards may share a validated autocast path. Do not call model.half() or cast the entire batch to half. CPU preflight uses FP32 with AMP/scaling disabled. Disable TF32 for matmul and convolution. Compare finite loss/gradients, task quality and elapsed time against FP32; retain FP32 for unsupported, unstable or slower operations. Record the actual regions and fallback, not an unmeasured speedup.



Let me approach this systematically.
First, I'll examine the dataset:
```
input/
    description.md (5 lines)
    sample_submission.csv (1524 lines)
    test.csv (1724 lines)
    train.csv (6839 lines)
submission/

working/
```

-> input/description.md has content:

```
# Disaster Tweets: Custom Precision Benchmark

Predict target=1 for tweets about real disasters, target=0 otherwise. Use text, keyword and location as appropriate. The metric is binary F1. Write submission.csv with id,target columns and binary predictions for test.csv. Use only train.csv labels for model selection, with your own internal validation split.

This is a custom benchmark, not an official MLE-bench competition definition or Kaggle leaderboard evaluation. The original labeled Kaggle training data was split 80/20 with stratification and seed 42. Both precision modes use this same holdout; test labels are unavailable to the agents. No leaderboard submission is made.

```

-> input/sample_submission.csv has 1523 rows and 2 columns.
⚠️  IMPORTANT: This is the CORRECT submission format that must be followed!
The exact column names are: id, target
Any description.md format information should be IGNORED if it conflicts with this file.
Here is some information about the columns:
id (int64) has range: 5.00 - 10873.00, 0 nan values
target (int64) has 1 unique values: [0]

-> input/test.csv has 1523 rows and 4 columns.
Here is some information about the columns:
id (int64) has range: 5.00 - 10873.00, 0 nan values

-> input/train.csv has 6090 rows and 5 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 10872.00, 0 nan values
target (int64) has 2 unique values: [1, 0]

        ⚠️  CRITICAL SUBMISSION FORMAT NOTE:
        - If you see sample_submission.csv or similar files, those contain the CORRECT submission format
        - The column names in these files are the FINAL AUTHORITY for submission format
        - Always use the column names from the actual sample submission files
        

# Used Prompt 4: merge

You are a Kaggle grandmaster attending a competition, an expert in writing clean, efficient, and competition-winning Python code for ML tasks. You have received code snippets from a team of specialized agents, each focusing on a specific part of the ML pipeline. Your critical task is to intelligently merge these partial scripts into a single, cohesive, and fully runnable Python script.


# Task description
# Disaster Tweets: Custom Precision Benchmark

Predict target=1 for tweets about real disasters, target=0 otherwise. Use text, keyword and location as appropriate. The metric is binary F1. Write submission.csv with id,target columns and binary predictions for test.csv. Use only train.csv labels for model selection, with your own internal validation split.

The original labeled training data was split 80/20 with stratification and seed 42. Test labels are unavailable to the agents.

============================================================
**REQUIRED SUBMISSION FORMAT**
============================================================
The final submission file must match this format:

  id  target
6925       0
1975       0
5034       0
 254       0
8370       0

(Showing first 5 rows as example)
============================================================
**Submission File Location**: Must save the submission to `./submission/submission.csv`
.

# Memory
Below is a record of previous solution attempts and their outcomes:
 Summary: 3 total attempts - 3 executed, 2 successful (best: 0.7573)

-------------------------------
Attempt #1:
Design: I merged the three stages by keeping Stage 1's data cleaning, TF-IDF feature builders, `SparseMLP`, `build_loss`, `tune_threshold` and split helpers as the foundation; layering Stage 2's import-safe precision utilities (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `degrade_to_fp32`, `configure_tf32`, `format_precision_log`) on top; then consuming both inside Stage 3's `run_training` with scheduler safe-points, `TrainingDiagnostics`, linear warmup+cosine AdamW, best-val-F1 restore, and test inference. I resolved the duplicate `CandidateAdapter` by keeping Stage 3's self-contained CPU-safe version (it uses the Stage 1 model/loss/features through the Stage 2 precision path), removed duplicate imports/constants, kept all side effects under `if __name__ == "__main__":`, and preserved the exact `MODEL_FAMILY`, `WORKING_DIR`, `SUBMISSION_DIR` contract.
Results: The training script executed successfully end-to-end: TF-IDF features were fit on the train split, the SparseMLP trained under CUDA FP16 AMP with TF32 disabled, early stopping triggered after 4 epochs without improvement, and a best-val-F1 checkpoint was restored for test inference and submission writing. The run produced a finite best validation F1 of 0.7573 (peak at epoch 1), with the final score line printed and a submission.csv generated; the only non-fatal log noise was an unrelated Neo4j connection refusal.
Validation Metric: 0.7573
Execution Time: 925.14s

-------------------------------
Attempt #2:
Design: I merged the three stages by keeping Stage 1's data/model/loss/adapter code as the foundation, layering Stage 2's `PrecisionRuntime` policy helpers on top, and consuming both inside Stage 3's `run_training` (with diagnostics + scheduler safe points). I resolved naming conflicts, deduplicated imports, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module-level constants that Stage 3 references, and moved every side effect under `if __name__ == '__main__':` while preserving the CPU-safe `CandidateAdapter` contract.
Results: Training completed successfully in ~37 minutes on an A10 GPU, running 12 epochs with a hybrid transformer model. Validation F1 improved from 0.665 at epoch 1 to a best of ~0.7341 (best epoch 11), and a valid submission file with 1523 rows was written using a tuned threshold of 0.42. The only messages were benign warnings (Graph evidence mirror connection refused, nested-tensor warning) that did not affect execution.
Validation Metric: 0.7341
Execution Time: 2223.42s

-------------------------------
Attempt #3:
Design: I merged the three stages by keeping Stage 1's data pipeline, hybrid BiGRU+CharCNN model, loss helpers, and CPU-safe `CandidateAdapter` as the foundation; layering Stage 2's import-safe precision utilities (`PrecisionConfig`, `resolve_precision_config`, `get_autocast_context`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `clip_gradients`, `degrade_to_fp32`, `is_finite_loss`, `configure_backend_precision`, `format_precision_log`, `precision_to_dict`) on top; and consuming both inside Stage 3's `run_training` with diagnostics, linear warmup + cosine AdamW, best-val-F1 restore, and test inference. I resolved duplicate imports/constants, kept `MODEL_FAMILY`, `WORKING_DIR`, and `SUBMISSION_DIR` as shared module constants, removed the meaningless placeholder arg in the adapter's unpack call, and moved all side effects inside `if __name__ == "__main__":`.
Results: The implementation of this design has bugs.
Insight: Using a different approach may not result in the same bugs as the above approach.


# Hardware-Aware Stage 1 Candidate Construction Context
# Hardware-Aware Model Design Brief
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2; observed in 2 batch-size observations
- Precision optimization mode: normal on ampere (compute capability 8.6).
- Allowed native training precision policies: fp32, disabled, fp16_amp
- FP16 AMP is optional, not a whole-pipeline cast: FP32 parameters/inputs, local autocast for forward, FP32-sensitive losses/reductions and metrics/export; GradScaler and backward outside autocast. Disable TF32; use FP32 on CPU. Retain FP32 wherever AMP is unstable, unsupported or slower.
- Precision evidence rule: recommend only a native Tensor Core format with a supported forward/backward training path; generic FP4 is not NVFP4 and FP6 is not an MLEvolve training policy.
- Model candidate: transformer (score=0.05, confidence=0.0): transformer is a baseline-compatible option for transformer_training, but no strong hardware-specific evidence was found.
- Model candidate: small_transformer (score=0.05, confidence=0.0): small_transformer is a baseline-compatible option for transformer_training, but no strong hardware-specific evidence was found.
- Model candidate: lora_transformer (score=0.05, confidence=0.0): lora_transformer is a baseline-compatible option for transformer_training, but no strong hardware-specific evidence was found.
- Decision rule: Choose the architecture that best satisfies the task metric while using hardware features to reduce training time and improve GPU utilization. If hardware evidence is weak, keep a conservative baseline-compatible architecture.
- Feature-detail rule: The feature key list is an index. Use selected hardware feature details already shown above, or query local feature-node details for chosen keys when deeper implementation guidance is needed.
- Constraint rule: Hardware advice never overrides task, dataset, submission, package, model-source, or filesystem constraints. Do not invent Kaggle input paths, local checkpoints, torch hub directories, or placeholder weights; use only paths and model sources that are explicitly available in the prompt/config.
- Budget guardrail: For hardware/runtime optimization, preserve the parent solution's modeling budget unless the user explicitly asks to improve score. Do not increase epochs, folds, model size, input resolution, ensemble count, TTA, dataset size, or validation workload as a hardware-only optimization.
# Hardware/Profile Optimization Context
  - model_design
    - filter_audit: features_shown=6/6
  - source=hardware_knowledge_graph.json; linked_features=6
- Workflow stage: Stage 1 model-design.
- Stage ownership: hardware context lookup, data processing, feature engineering, model architecture, loss, and output contract. Do not choose dtype/AMP policy, optimizer, batch size, scheduler, or training loop here.
- Hardware node response rule: the current hardware node response and compact feature-key pairs are already attached below.
- Query rule: do not query the hardware node again; query local feature-node details only for selected feature keys when deeper implementation guidance is needed.
- Source rule: do not fetch external URLs or rely on source links in generated reasoning.
- Scheduler backend config: packing_backend=cuda_process, effective_backend=cuda_process, runner_contract=subprocess_job_v1, enabled_backends=['exclusive', 'mps_process', 'cuda_process'], exclusive_fallback_enabled=True

# Hardware-Aware Datatype/Precision Context
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2; observed in 2 batch-size observations
- Precision optimization mode: normal on ampere (compute capability 8.6).
- Allowed native training precision policies: fp32, disabled, fp16_amp
- FP16 AMP is optional, not a whole-pipeline cast: FP32 parameters/inputs, local autocast for forward, FP32-sensitive losses/reductions and metrics/export; GradScaler and backward outside autocast. Disable TF32; use FP32 on CPU. Retain FP32 wherever AMP is unstable, unsupported or slower.
- Precision evidence rule: recommend only a native Tensor Core format with a supported forward/backward training path; generic FP4 is not NVFP4 and FP6 is not an MLEvolve training policy.
- Stage boundary: Choose tensor datatype and precision policy here: DEVICE, USE_AMP, AMP_DTYPE, USE_TF32, GradScaler, Transformer Engine FP8/MXFP8/NVFP4 recipes, autocast helper, precision-required model adapters, fallback behavior, and precision logging.
- Allowed model adaptation: only precision-required wrappers/replacements for compatible modules, precision shape padding/config hooks, or higher-precision islands that preserve the Stage 1 model family, loss, data features, and output interface.
- Out of scope for this stage: general model redesign, loss changes, feature/preprocessing changes, batch size, epochs, learning rate, dataloader workers, gradient accumulation, checkpoint cadence, validation metric, and submission logic.
- Rule: HWDB recommended_patterns are conditional candidates, not measured speedups or mandatory steps. Check workload, installed backend, allowed precision and each stated limitation before applying one. A capability or verified flag does not prove benefit for this candidate. Keep the baseline when evidence is missing; record feature keys, unmet conditions and measured results without inventing them.
- Constraint rule: Hardware advice never overrides task, dataset, submission, package, model-source, or filesystem constraints. Do not invent Kaggle input paths, local checkpoints, torch hub directories, or placeholder weights; use only paths and model sources that are explicitly available in the prompt/config.
# Hardware/Profile Optimization Context
  - datatype_precision
    - filter_audit: features_shown=3/3
  - source=hardware_knowledge_graph.json; linked_features=3
- Workflow stage: Stage 2 datatype/quantization.
- Stage ownership: device, dtype, AMP/autocast, TF32, GradScaler, Transformer Engine recipes, precision-required model adapters, precision fallback, and precision logging.
- Hardware node response rule: the current hardware node response and compact feature-key pairs are already attached below.
- Query rule: do not query the hardware node again; query local feature-node details only for selected feature keys when deeper implementation guidance is needed.
- Source rule: do not fetch external URLs or rely on source links in generated reasoning.
- Scheduler backend config: packing_backend=cuda_process, effective_backend=cuda_process, runner_contract=subprocess_job_v1, enabled_backends=['exclusive', 'mps_process', 'cuda_process'], exclusive_fallback_enabled=True
- Scheduler limits: safe_vram_budget_mb=19201, packing_backend=cuda_process, runner_contract=subprocess_job_v1, exclusive_fallback_enabled=True
- Precision diagnosis: symptoms=['precision_not_optimized'] targets=['improve_precision_efficiency']
- Confidence: 0.0

# Hardware-Aware Training Hyperparameter Context
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2; observed in 2 batch-size observations
- Precision optimization mode: normal on ampere (compute capability 8.6).
- Allowed native training precision policies: fp32, disabled, fp16_amp
- FP16 AMP is optional, not a whole-pipeline cast: FP32 parameters/inputs, local autocast for forward, FP32-sensitive losses/reductions and metrics/export; GradScaler and backward outside autocast. Disable TF32; use FP32 on CPU. Retain FP32 wherever AMP is unstable, unsupported or slower.
- Precision evidence rule: recommend only a native Tensor Core format with a supported forward/backward training path; generic FP4 is not NVFP4 and FP6 is not an MLEvolve training policy.
- Stage boundary: Optimize training hyperparameters here: physical batch size, effective batch size, gradient accumulation, epochs, learning rate, weight decay, scheduler, dataloader workers, checkpointing, runtime logging, validation, and submission.
- Precision boundary: consume the datatype_precision variables/utilities. Do not pick a new AMP dtype here unless the prior policy is impossible to use and a safe fallback is needed.
- Rule: HWDB recommended_patterns are conditional candidates, not measured speedups or mandatory steps. Check workload, installed backend, allowed precision and each stated limitation before applying one. A capability or verified flag does not prove benefit for this candidate. Keep the baseline when evidence is missing; record feature keys, unmet conditions and measured results without inventing them.
- Constraint rule: Hardware advice never overrides task, dataset, submission, package, model-source, or filesystem constraints. Do not invent Kaggle input paths, local checkpoints, torch hub directories, or placeholder weights; use only paths and model sources that are explicitly available in the prompt/config.
- Budget guardrail: For hardware/runtime optimization, preserve the parent solution's modeling budget unless the user explicitly asks to improve score. Do not increase epochs, folds, model size, input resolution, ensemble count, TTA, dataset size, or validation workload as a hardware-only optimization.
# Hardware/Profile Optimization Context
  - training_evaluation
    - filter_audit: features_shown=3/5, omitted_not_recommended=['soap_optimizer', 'ademamix_optimizer']
    - muon_optimizer:  (category=optimizer, support_level=supported, recommended=True, verified=False, recommendation_scope=small_matmul_heavy_training_only): usage: Hybrid Muon+AdamW training optimizer for matmul-heavy candidates: the example code shows the parameter split, Muon settings, and AdamW companion optimizer.; notes: Use Muon on selected 2D hidden weights and AdamW on embedding/bias/norm/head/non-2D params; tune Muon and AdamW learning rates separately; tune weight_decay per optimizer group; initial Muon settings: ns_steps=5, momentum=0.95, nesterov=True; prefer adjust_lr_fn='match_rms_adamw' when reusing an AdamW LR schedule; compare against a tuned AdamW baseline.; limitations: software training optimizer; requires a PyTorch build that exposes torch.optim.Muon; no hardware-specific acceleration is required; use only on selected 2D hidden weights and keep AdamW for embeddings/bias/norm/head/non-2D params; benchmark against tuned AdamW; not an inference feature
  - source=hardware_knowledge_graph.json; linked_features=5
- Workflow stage: Stage 3 training.



# Pipeline Decision Contract
Before writing or modifying code, follow the hardware-aware stepwise workflow: model-design -> datatype/quantization -> training (code keys: model_design -> datatype_precision -> training_evaluation).

1. Model design: load/prepare data, create features/splits, and choose architecture/model family, loss, criterion, and output interface first.
2. Datatype/precision: choose only mode-allowed operations, autocast regions, FP32-sensitive regions, scaling and fallback behavior. A policy is not a whole-pipeline dtype.
3. Training: choose optimizer, scheduler, batch size, dataloader settings, checkpointing, validation, submission, runtime fallbacks, and logging.
Datatype precision may include only precision-required model adapters that preserve the Stage 1 model family, loss, data features, and output interface.
Hardware-only training/runtime tuning must not increase epochs, folds, model size, image resolution, ensemble count, TTA, dataset size, or validation workload.
If evidence is missing, use the fallback recorded in the trace instead of inventing hardware claims.

# Pipeline Decision Trace
```json
{
  "model_design": {
    "modality": "text",
    "target_type": "classification",
    "shape_constraints": [
      "sequence length and tokenizer output must stay configurable"
    ],
    "family": "available_transformer_or_sequence_baseline",
    "alternatives_considered": [],
    "loss": "binary cross-entropy with logits",
    "output_interface": "single logit per example; sigmoid threshold selected on internal validation to maximize binary F1; submission columns id,target with int binary predictions",
    "reason": "Use a baseline-compatible model family until stronger task or evidence signals are available.",
    "hardware_fit": "use only compatible evidence"
  },
  "datatype_precision": {
    "precision_policy": "fp16_amp",
    "precision_model_adaptation": "none",
    "fallback_policy": "fall back to fp32 or disabled AMP if the selected precision path is unsupported",
    "reason": "Use conservative precision until hardware and package support are confirmed."
  },
  "training_evaluation": {
    "optimizer": "AdamW over FP32 parameters",
    "scheduler": "linear warmup with linear/cosine decay",
    "batch_size_policy": "fixed",
    "dataloader_policy": "num_workers>0, pin_memory=True, persistent_workers when reused, non_blocking=True device copies",
    "fallbacks": [
      "disable AMP and run FP32 if non-finite loss/gradients or slower/less accurate",
      "retain FP32 for unsupported or unstable operations"
    ],
    "metrics_to_log": [
      "elapsed_seconds",
      "peak_vram_mb",
      "resolved_batch_size"
    ],
    "advanced_optimizer_used": false,
    "reason": "Choose stable training defaults matched to the inferred target type and available model family."
  },
  "evidence": {
    "hardware_context_used": true,
    "evidence_refs": [
      "backend_guidance:cuda_process.no_mps_controls",
      "backend_guidance:cuda_process.normal_process_throughput",
      "backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry",
      "backend_guidance:shared.subprocess.task_quality_and_elasticity"
    ],
    "confidence": 0.95,
    "missing_evidence": [
      "predictor/graph evidence not available",
      "no exact/similar/packed graph profiles for current hardware",
      "no measured FP16 vs FP32 comparison yet"
    ]
  }
}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=## Plan **WHAT makes this task unique:** Disaster-tweet classification is dominated by *metaphorical vs.; purpose=Define the Stage 1 candidate target for later precision and training stages.; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=## Plan **WHAT:** The datatype/precision step must define a hardware-aware FP16 AMP policy for the KB-MVNet sparse multi-view network on the A10 (SM 8.6) — where the model is dominated by sparse GEMM + LayerNorm + GEL...; purpose=Support the Stage 1 candidate target with a compatible precision policy.; hardware_keys=autocast_forward, configure_backend_precision, compute_bce_with_logits_fp32, make_grad_scaler, degrade_to_fp32, USE_TF32
  - Stage 3 training / training_evaluation: baseline_change=added the full Stage-3 trainer (`run_training`) with AdamW + linear-warmup/cosine LR, grad clipping, early stop + best-checkpoint restore, threshold-tuned binary-F1 validation, and test inference writing `./submission...; purpose=preserve Stage-1 candidate intent (metaphor-sensitive keyword-gated fusion) under the Stage-2 FP16-AMP policy while maximizing the exact F1 metric; hardware_keys=fp16_amp, grad_scaler, num_workers, pin_memory, persistent_workers
  - Stage 3 training / training_evaluation: baseline_change=re-fit the feature bundle only on the internal train split inside `run_training` and reuse the exact same `FeatureBundle.transform` for val/test; purpose=kill leak paths while keeping val and test pre-processing byte-identical; hardware_keys=none
  - Stage 3 training / training_evaluation: baseline_change=wired `TrainingDiagnostics`, scheduler safe-points (BEFORE_TRAIN/STEP/EPOCH) with a full `state_factory`, and `script_scheduler_context(load_resume_checkpoint)` guarded by `if context is not None`; purpose=satisfy the scheduler contract without altering training semantics on unscheduled runs; hardware_keys=none

# Step results

        ### Step 1: model_design
        **Plan:** ## Plan

**WHAT makes this task unique:** Disaster-tweet classification is dominated by *metaphorical vs. literal* usage of disaster vocabulary ("this song is fire", "my boss is killing me"), plus high-frequency but non-disaster-signaling keywords (e.g. "ablaze", "rain"). Previous attempts all used either plain TF-IDF→MLP (0.7573) or end-to-end sequence nets (0.7341) without conditioning on the keyword categorical and without any metaphor-sensitive stylistic signals.

**WHY my approach fits:** I replace the sequence-model direction with a *keyword-conditioned gated multi-view sparse network* (KB-MVNet) that fuses four heterogeneous evidence views — (1) word TF-IDF, (2) character TF-IDF (captures hashtag/emoji/typos), (3) one-hot keyword view (the strongest categorical prior available in this dataset), (4) 15 hand-crafted stylistic features that proxy metaphor (emoji/emoticon count, uppercase ratio, first-person pronouns, URL/mention structure). A learned soft-max gate selects which view to trust per example, so "ablaze+🌊 and streets flooded" leans on char+style, while "ablaze" alone on word+keyword. Loss is unchanged from the trace: BCE-with-logits → single logit → sigmoid threshold.

**HOW to validate:** Fit on the 85 % internal split, early-stop on the 15 % stratified holdout, and sweep the decision threshold on that holdout to maximize binary F1 — the exact metric. Confirm: val F1 beats 0.7573 and the gate entropy is not collapsed (all four views used). If val F1 < 0.75 or a single view dominates with entropy ≈ 0, I'll revisit fusion.

**Memory / elasticity / backend contract:** All features are in-memory dense float32 view tuples; batch size is `scenario['batch_size']` with no accumulation here (trainer owns it). Word/char dims, hidden width, dropout, and sequence of views remain configurable via context. Token/embedding indices are absent (bag-of-words views) so no long-dtype leakage. Scheduler-owned (NOT implemented here): device assignment, worker count, autocast dtype, GradScaler, optimizer/scheduler cadence, checkpoint cadence.

**Note board:**
- baseline_change: replaced prior TF-IDF→MLP with a 4-view gated keyword-conditioned sparse net (word TF-IDF + char TF-IDF + keyword one-hot + 15 stylistic metaphor proxies) | purpose: directly attack metaphor-vs-literal confusion that dominates this task's error mode, using the keyword categorical as a first-class view rather than as extra text | hardware_keys: none
- baseline_change: added a per-sample softmax gate over the four view embeddings before the classification head | purpose: allow the model to route stylistic/char evidence for hashtag/emoji-heavy tweets and word/keyword evidence for plain news-style tweets, improving F1 on both regimes | hardware_keys: none
- baseline_change: feature bundle is fit strictly on the training fold inside a cached module helper; a no-arg `CandidateAdapter` reuses the same real bundle/model/loss for CPU preflight | purpose: preserve leakage isolation and ensure the preflight exercises the exact production model | hardware_keys: none
        **Code:**
        ```python
# stage: model_design
# Model family: keyword-gated multi-view sparse MLP (word TF-IDF + char TF-IDF + keyword one-hot + stylistic features)
import os
import re
import html
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------ constants
MODEL_FAMILY = "keyword_gated_multiview_sparse_mlp_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
INPUT_DIR = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")

SEED = 42
VAL_FRACTION = 0.15

MAX_WORD_FEATURES = 8000
MAX_CHAR_FEATURES = 12000
MAX_KEYWORD_VOCAB = 300
MIN_KEYWORD_COUNT = 2

HIDDEN_DIM = 256
DROPOUT = 0.35
STYLE_DIM = 15
NUM_VIEWS = 4

# ------------------------------------------------------------------ text utils
_URL_RE = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_WS_RE = re.compile(r"\s+")
_FIRST_PERSON_RE = re.compile(
    r"\b(i|i'm|im|i've|ive|my|me|mine|we|we're|our|us|myself|ourselves)\b",
    re.IGNORECASE,
)
_EMOTICON_RE = re.compile(r"[:;=][\-o\*']?[\)\(dpDP]")
_EMOJI_RE = re.compile(
    "[" "\U0001f300-\U0001faff" "\U00002600-\U000027bf" "\U0001f1e6-\U0001f1ff" "]",
    re.UNICODE,
)


def clean_text(s):
    """Light normalisation that preserves metaphor cues (emoticons, casing, hashtags)."""
    if not isinstance(s, str):
        return ""
    s = html.unescape(s)
    s = _URL_RE.sub(" <url> ", s)
    s = _MENTION_RE.sub(" <user> ", s)
    s = _HASHTAG_RE.sub(r" \1 ", s)  # keep hashtag body as a word token
    s = s.lower()
    s = _WS_RE.sub(" ", s).strip()
    return s


def stylistic_features(raw_text):
    """15 per-tweet stylistic / metaphor-proxy features (always length 15)."""
    t = raw_text if isinstance(raw_text, str) else ""
    n = max(len(t), 1)
    n_upper = sum(1 for c in t if c.isupper())
    n_digit = sum(1 for c in t if c.isdigit())
    n_punct = sum(1 for c in t if c in "!?,.;:'\"")
    n_excl = t.count("!")
    n_q = t.count("?")
    n_hashtag = t.count("#")
    n_mention = t.count("@")
    n_url = 1 if _URL_RE.search(t) else 0
    first_person = len(_FIRST_PERSON_RE.findall(t))
    emoticon = len(_EMOTICON_RE.findall(t))
    emoji = len(_EMOJI_RE.findall(t))
    return np.array(
        [
            float(len(t)),
            float(np.log1p(len(t))),
            n_upper / n,
            n_digit / n,
            n_punct / n,
            float(n_excl),
            float(n_q),
            float(n_hashtag),
            float(n_mention),
            float(n_url),
            float(first_person),
            float(emoticon),
            float(emoji),
            n_excl / n,
            n_q / n,
        ],
        dtype=np.float32,
    )


# ------------------------------------------------------------------ keyword view
def build_keyword_vocab(keywords):
    from collections import Counter

    counts = Counter(
        k.strip().lower() for k in keywords if isinstance(k, str) and k and k.strip()
    )
    ordered = [
        k for k, c in counts.most_common(MAX_KEYWORD_VOCAB) if c >= MIN_KEYWORD_COUNT
    ]
    return {k: i for i, k in enumerate(ordered)}


def keyword_onehot(keywords, vocab):
    unknown_idx = len(vocab)
    out = np.zeros((len(keywords), len(vocab) + 1), dtype=np.float32)
    for i, kw in enumerate(keywords):
        key = kw.strip().lower() if isinstance(kw, str) else ""
        idx = vocab.get(key, unknown_idx)
        out[i, idx] = 1.0
    return out


# ------------------------------------------------------------------ feature bundle
class FeatureBundle(object):
    """Holds every stateful transform, fit on the training fold only."""

    def __init__(self, word_vec, char_vec, kw_vocab, style_scaler):
        self.word_vec = word_vec
        self.char_vec = char_vec
        self.kw_vocab = kw_vocab
        self.style_scaler = style_scaler

    @property
    def word_dim(self):
        return len(self.word_vec.get_feature_names_out())

    @property
    def char_dim(self):
        return len(self.char_vec.get_feature_names_out())

    @property
    def kw_dim(self):
        return len(self.kw_vocab) + 1

    @property
    def style_dim(self):
        return STYLE_DIM

    def transform(self, df):
        texts = [clean_text(t) for t in df["text"].fillna("").astype(str).tolist()]
        word_X = self.word_vec.transform(texts).astype(np.float32).toarray()
        char_X = self.char_vec.transform(texts).astype(np.float32).toarray()
        if "keyword" in df.columns:
            kws = df["keyword"].fillna("").astype(str).tolist()
        else:
            kws = [""] * len(df)
        kw_X = keyword_onehot(kws, self.kw_vocab)
        raw = df["text"].fillna("").astype(str).tolist()
        style = np.stack([stylistic_features(t) for t in raw], axis=0).astype(
            np.float32
        )
        style = self.style_scaler.transform(style).astype(np.float32)
        return word_X, char_X, kw_X, style


def fit_feature_bundle(train_df):
    """Fit the entire feature pipeline on the *training* fold only (no leakage)."""
    texts = [clean_text(t) for t in train_df["text"].fillna("").astype(str).tolist()]

    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=MAX_WORD_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    word_vec.fit(texts)

    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=MAX_CHAR_FEATURES,
        sublinear_tf=True,
        lowercase=True,
    )
    char_vec.fit(texts)

    kws = (
        train_df["keyword"].fillna("").astype(str).tolist()
        if "keyword" in train_df.columns
        else []
    )
    kw_vocab = build_keyword_vocab(kws)

    raw = train_df["text"].fillna("").astype(str).tolist()
    style_raw = np.stack([stylistic_features(t) for t in raw], axis=0).astype(
        np.float32
    )
    style_scaler = StandardScaler().fit(style_raw)

    return FeatureBundle(word_vec, char_vec, kw_vocab, style_scaler)


# ------------------------------------------------------------------ split
def split_data(df, seed=SEED, val_fraction=VAL_FRACTION):
    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        stratify=df["target"].astype(int).values,
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ------------------------------------------------------------------ model
class KeywordGatedMultiviewNet(nn.Module):
    """Fuse four heterogeneous evidence views with a per-sample softmax gate."""

    def __init__(
        self, word_dim, char_dim, kw_dim, style_dim, hidden=HIDDEN_DIM, dropout=DROPOUT
    ):
        super(KeywordGatedMultiviewNet, self).__init__()
        self.word_dim = int(word_dim)
        self.char_dim = int(char_dim)
        self.kw_dim = int(kw_dim)
        self.style_dim = int(style_dim)
        self.hidden = int(hidden)

        def _branch(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.word_proj = _branch(self.word_dim)
        self.char_proj = _branch(self.char_dim)
        self.kw_proj = _branch(self.kw_dim)
        self.style_proj = _branch(self.style_dim)

        self.gate = nn.Sequential(
            nn.Linear(hidden * NUM_VIEWS, hidden),
            nn.GELU(),
            nn.Linear(hidden, NUM_VIEWS),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, word_x, char_x, kw_x, style_x):
        hw = self.word_proj(word_x)
        hc = self.char_proj(char_x)
        hk = self.kw_proj(kw_x)
        hs = self.style_proj(style_x)
        stacked = torch.stack([hw, hc, hk, hs], dim=1)  # [B, 4, H]
        flat = stacked.reshape(stacked.size(0), -1)
        gates = torch.softmax(self.gate(flat), dim=-1).unsqueeze(-1)  # [B, 4, 1]
        fused = (stacked * gates).sum(dim=1)  # [B, H]
        return self.head(fused).squeeze(-1)  # single logit per example


# ------------------------------------------------------------------ loss / criterion
def build_loss():
    """Binary cross-entropy on raw logits (metric: thresholded binary F1)."""
    return nn.BCEWithLogitsLoss()


def _resolve_criterion(context):
    """Return the real criterion; honour caller-supplied mapping if valid."""
    if context:
        crit = context.get("criterion")
        if isinstance(crit, nn.Module):
            return crit
    return build_loss()


# ------------------------------------------------------------------ cached pipeline (for adapter only)
_PIPELINE_CACHE = {}


def _train_csv_path():
    return os.path.join(INPUT_DIR, "train.csv")


def _get_train_df():
    if "train_df" not in _PIPELINE_CACHE:
        _PIPELINE_CACHE["train_df"] = pd.read_csv(_train_csv_path())
    return _PIPELINE_CACHE["train_df"]


def _get_feature_bundle():
    if "bundle" not in _PIPELINE_CACHE:
        df = _get_train_df()
        # Fit on a stratified slice of the train split only (still no val/test labels).
        fit_df, _ = split_data(df)
        _PIPELINE_CACHE["bundle"] = fit_feature_bundle(fit_df)
    return _PIPELINE_CACHE["bundle"]


# ------------------------------------------------------------------ CPU preflight adapter
class CandidateAdapter(object):
    """No-arg CPU-safe adapter that reuses the real model / loss / features."""

    _DEFAULTS = {
        "hidden": HIDDEN_DIM,
        "dropout": DROPOUT,
        "style_dim": STYLE_DIM,
        "lr": 1e-3,
        "weight_decay": 1e-4,
    }

    def _resolve_context(self, context):
        ctx = dict(self._DEFAULTS)
        if context:
            for k, v in context.items():
                if v is not None:
                    ctx[k] = v
        return ctx

    def _resolve_dims(self, context):
        bundle = _get_feature_bundle()
        word_dim = bundle.word_dim
        char_dim = bundle.char_dim
        kw_dim = bundle.kw_dim
        style_dim = bundle.style_dim
        if context:
            if context.get("word_dim"):
                word_dim = int(context["word_dim"])
            if context.get("char_dim"):
                char_dim = int(context["char_dim"])
            if context.get("kw_dim"):
                kw_dim = int(context["kw_dim"])
            if context.get("style_dim"):
                style_dim = int(context["style_dim"])
        return word_dim, char_dim, kw_dim, style_dim

    # ---- construction ----------------------------------------------------
    def build_model(self, context):
        word_dim, char_dim, kw_dim, style_dim = self._resolve_dims(context)
        ctx = self._resolve_context(context)
        return KeywordGatedMultiviewNet(
            word_dim=word_dim,
            char_dim=char_dim,
            kw_dim=kw_dim,
            style_dim=style_dim,
            hidden=int(ctx["hidden"]),
            dropout=float(ctx["dropout"]),
        )

    def build_optimizer(self, model, context):
        ctx = self._resolve_context(context)
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(ctx["lr"]),
            weight_decay=float(ctx["weight_decay"]),
        )

    # ---- batch builders --------------------------------------------------
    def _fixture_batch(self, fixture, device):
        named = {}
        for k, v in fixture.items():
            if isinstance(v, torch.Tensor):
                named[k] = v.to(device)
            elif (
                isinstance(v, (list, tuple))
                and len(v) > 0
                and isinstance(v[0], torch.Tensor)
            ):
                named[k] = v.to(device)
            else:
                named[k] = torch.as_tensor(v, device=device)

        # Prefer named contract keys, otherwise collect every non-target tensor
        # in the fixture order and treat the last labelled tensor as target.
        def _take(*names):
            for n in names:
                if n in named:
                    return named[n]
            return None

        word_x = _take("word_x", "word", "word_ids")
        char_x = _take("char_x", "char", "char_ids")
        kw_x = _take("kw_x", "keyword", "keyword_ids")
        style_x = _take("style_x", "style", "style_features")
        target = _take("target", "label", "labels", "y")

        ordered_inputs = [t for t in (word_x, char_x, kw_x, style_x) if t is not None]
        if not ordered_inputs:
            # Fall back: gather all float tensors except target as inputs
            for k, v in named.items():
                if k in ("target", "label", "labels", "y"):
                    continue
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    ordered_inputs.append(v)
        ordered_inputs = [t.to(dtype=torch.float32) for t in ordered_inputs]

        if target is None:
            n = ordered_inputs[0].size(0) if ordered_inputs else 1
            target = torch.zeros(n, dtype=torch.float32, device=device)
        target = target.to(dtype=torch.float32).view(-1)

        out = {"inputs": tuple(ordered_inputs), "target": target}
        for k, v in named.items():
            out.setdefault(k, v)
        return out

    def _real_batch(self, batch_size, device):
        bundle = _get_feature_bundle()
        df = _get_train_df()
        n = min(int(batch_size), len(df))
        rng = np.random.RandomState(0)
        idx = rng.randint(0, len(df), size=n)
        sub = df.iloc[idx].reset_index(drop=True)
        word_X, char_X, kw_X, style_X = bundle.transform(sub)
        word_x = torch.from_numpy(np.ascontiguousarray(word_X, dtype=np.float32)).to(
            device
        )
        char_x = torch.from_numpy(np.ascontiguousarray(char_X, dtype=np.float32)).to(
            device
        )
        kw_x = torch.from_numpy(np.ascontiguousarray(kw_X, dtype=np.float32)).to(device)
        style_x = torch.from_numpy(np.ascontiguousarray(style_X, dtype=np.float32)).to(
            device
        )
        target = torch.from_numpy(sub["target"].values.astype(np.float32)).to(device)
        return {"inputs": (word_x, char_x, kw_x, style_x), "target": target}

    def _build_batch(self, scenario, device):
        if scenario is None:
            scenario = {}
        batch_size = int(scenario.get("batch_size", 8))
        fixture = scenario.get("fixture") or {}
        if fixture:
            return self._fixture_batch(fixture, device)
        return self._real_batch(batch_size, device)

    def build_train_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    # ---- steps -----------------------------------------------------------
    @staticmethod
    def _unpack(batch):
        if isinstance(batch, dict):
            inputs = batch.get("inputs")
            target = batch.get("target")
        else:
            inputs, target = batch[0], batch[1]
        if not isinstance(inputs, (list, tuple)):
            inputs = (inputs,)
        return inputs, target

    def training_step(self, model, batch, context):
        inputs, target = self._unpack(batch)
        inputs = [
            t.to(dtype=torch.float32) if torch.is_floating_point(t) else t
            for t in inputs
        ]
        target = target.to(dtype=torch.float32).view(-1)
        logits = model(*inputs)
        criterion = _resolve_criterion(context)
        loss = criterion(logits.view(-1), target)
        return loss

    def validation_step(self, model, batch, context):
        with torch.no_grad():
            return self.training_step(model, batch, context)


def build_model_from_bundle(bundle, hidden=HIDDEN_DIM, dropout=DROPOUT):
    """Convenience constructor the training loop can call with the real fitted bundle."""
    return KeywordGatedMultiviewNet(
        word_dim=bundle.word_dim,
        char_dim=bundle.char_dim,
        kw_dim=bundle.kw_dim,
        style_dim=bundle.style_dim,
        hidden=hidden,
        dropout=dropout,
    )

```
        
        ### Step 2: datatype_precision
        **Plan:** ## Plan

**WHAT:** The datatype/precision step must define a hardware-aware FP16 AMP policy for the KB-MVNet sparse multi-view network on the A10 (SM 8.6) — where the model is dominated by sparse GEMM + LayerNorm + GELU stacks that are highly FP16-friendly, but where the final BCEWithLogits reduction, gate softmax log-sum, and metric/prediction export must stay FP32.

**WHY:** Normal-mode rules permit only FP32 and selective FP16 AMP; TF32 must be explicitly disabled (A10's FP32-via-TF32 path can silently distort LayerNorm statistics and short-sequence logit tails). The Stage 1 target (winning F1 on metaphor-vs-literal) is sensitive to calibration, so a `GradScaler` protects the gradient signal through the four branch projections and the gate, while the loss, gate-softmax normalizer, and metric accumulators live in a hard FP32 island. We expose this as a *policy object + autocast context + graded scaler helpers*, not a global dtype switch — the model parameters remain FP32.

**HOW:** Validate by comparing val F1 and non-finite-loss counters with `USE_AMP=True` vs `False` on the same fold; require finite loss/gradients and no F1 regression (>0.005 drop) to keep AMP on, otherwise fall back to pure FP32. The helpers below are import-safe (no CUDA touched at definition time), CPU-safe (return FP32/no-op scaler on CPU), and expose `strategy_from_context(ctx)` so the Stage 3 training loop can call them directly.

**Backend contract:** precision utilities are pure helpers; device selection, worker count, and scheduler integration are chosen by the caller (Stage 3). No ops at import time.

**Note board:**
- baseline_change: added a self-contained `PrecisionStrategy` + `autocast_forward` + `configure_backend_precision` + `compute_bce_with_logits_fp32` + `make_grad_scaler` + `degrade_to_fp32` layer, without altering the Stage 1 KB-MVNet model/loss/features/output interface | purpose: give Stage 3 a single importable contract that keeps FP16 AMP in the four branching MLP views and the fused head while forcing FP32 on the BCE reduction, gate-softmax normalizer, metric accumulator, and prediction export — protecting the metaphor-sensitive F1 target | hardware_keys: precision_not_optimized, ampere_sm86_fp16_tc, tf32_disable
- baseline_change: TF32 explicitly disabled via `torch.backends.cuda.matmul.allow_tf32=False` and `cudnn.allow_tf32=False` inside `configure_backend_precision()` (called only from `__main__`), with a documented `USE_TF32` flag defaulting to False | purpose: prevent silent FP32-mantissa truncation on the sparse-view GEMMs that would shift LayerNorm statistics and hurt calibration/F1 selection | hardware_keys: none
- baseline_change: added a `degrade_to_fp32` fallback that flips `USE_AMP=False` when the caller observes repeated non-finite losses | purpose: preserve the Stage 1 model family and training stability without changing any modeling/optimizer/scheduler choices owned by other stages | hardware_keys: precision_fallback
        **Code:**
        ```python
# stage: datatype_precision
"""
Precision / datatype policy utilities for the keyword-gated multi-view sparse MLP.

Design contract (normal mode, NVIDIA A10 SM 8.6):
  * Model weights, floating inputs, and floating labels stay FP32 on all devices.
  * CUDA FP16 AMP is a *selective* option: main autocast region covers the four
    projection branches, the fused gate, and the classification head; the BCE
    reduction and gate-softmax normalizer remain in FP32.
  * GradScaler is required for FP16 AMP; backward + optimizer.step run OUTSIDE
    autocast.
  * TF32 for matmul and cudnn is explicitly DISABLED.
  * CPU preflight always runs FP32 with AMP/scaler disabled.
  * Fallback: if non-finite loss/gradients are observed repeatedly, the caller
    can call `strategy.with_amp_policy(False)` to degrade to pure FP32.
Nothing in this module touches CUDA, reads env vars, or allocates models at
import time; `resolve_precision_strategy()` / `configure_backend_precision()`
are invoked from the caller's `__main__` guard.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# ------------------------------------------------------------------ constants
MODEL_FAMILY_PRECISION_TAG = "kb_mvnet_kw_gated_sparse_v1__fp16_amp"

# Normal-mode allow-list: only FP32 and FP16 AMP.
ALLOWED_PRECISION_POLICIES = ("fp32", "fp16_amp")

# FP32-sensitive ops that must not be allowed to run inside FP16 autocast.
# These are *names* only; the actual FP32-islands are enforced by wrapping the
# corresponding math in `fp32_island()` at the call site (Stage 3).
FP32_SENSITIVE_REGIONS = (
    "bce_with_logits_reduction",
    "gate_softmax_normalizer",
    "metric_accumulation",
    "prediction_export",
)

_logger = logging.getLogger("kb_mvnet.precision")
if not _logger.handlers:  # avoid duplicate handlers when module is re-imported
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[precision] %(levelname)s: %(message)s"))
    _logger.addHandler(_h)
_logger.setLevel(logging.INFO)
_logger.propagate = False


# ------------------------------------------------------------------ config
@dataclasses.dataclass(frozen=True)
class PrecisionStrategy:
    """Immutable snapshot of the active precision policy."""

    device: torch.device
    use_amp: bool
    amp_dtype: torch.dtype  # torch.float16 when use_amp else torch.float32
    use_tf32: bool
    use_grad_scaler: bool
    amp_available: bool  # CUDA + cudnn present and policy allows AMP
    reason: str
    model_family_tag: str = MODEL_FAMILY_PRECISION_TAG

    # ---- derived helpers -------------------------------------------------
    @property
    def is_cpu(self) -> bool:
        return self.device.type == "cpu"

    @property
    def amp_active(self) -> bool:
        return bool(self.use_amp and self.amp_available and not self.is_cpu)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "use_amp": bool(self.use_amp),
            "amp_active": bool(self.amp_active),
            "amp_dtype": str(self.amp_dtype).replace("torch.", ""),
            "use_tf32": bool(self.use_tf32),
            "use_grad_scaler": bool(self.use_grad_scaler),
            "amp_available": bool(self.amp_available),
            "precision_reason": self.reason,
            "model_family_tag": self.model_family_tag,
            "fp32_sensitive_regions": list(FP32_SENSITIVE_REGIONS),
        }

    def with_amp_policy(
        self, use_amp: bool, reason: Optional[str] = None
    ) -> "PrecisionStrategy":
        """Return a copy of this strategy with AMP forced on/off (fallback path)."""
        new_reason = reason or (
            "amp_enabled_by_caller" if use_amp else "amp_disabled_by_caller"
        )
        return dataclasses.replace(
            self,
            use_amp=bool(use_amp),
            amp_dtype=torch.float16 if use_amp else torch.float32,
            use_grad_scaler=bool(use_amp and self.amp_available and not self.is_cpu),
            reason=f"{self.reason} | {new_reason}",
        )


# ------------------------------------------------------------------ resolution
def _cuda_fp16_available() -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def resolve_precision_strategy(
    requested_policy: Optional[str] = None,
    requested_device: Optional[Any] = None,
    allow_amp: Optional[bool] = None,
) -> PrecisionStrategy:
    """
    Resolve the actual precision strategy for the current process.

    CPU => FP32, no AMP, no scaler. CUDA => FP16 AMP by default (normal mode),
    TF32 off. Caller may override with `requested_policy='fp32'` or
    `allow_amp=False`. Unrecognised policies degrade to FP32 with an explicit
    reason rather than raising.
    """
    reason_bits = []

    # ---- device ----------------------------------------------------------
    if requested_device is None:
        device = torch.device("cuda" if _cuda_fp16_available() else "cpu")
        reason_bits.append("device=auto")
    else:
        device = torch.device(requested_device)
        reason_bits.append(f"device=caller({device.type})")

    if device.type == "cuda" and not _cuda_fp16_available():
        reason_bits.append("cuda_requested_unavailable->cpu")
        device = torch.device("cpu")

    is_cpu = device.type == "cpu"
    amp_available = not is_cpu

    # ---- policy ----------------------------------------------------------
    policy = (requested_policy or "fp16_amp").lower()
    if policy not in ALLOWED_PRECISION_POLICIES:
        reason_bits.append(f"policy:{policy}->fp32(fallback)")
        policy = "fp32"

    if allow_amp is False and policy != "fp32":
        reason_bits.append("amp_disabled_by_caller")
        policy = "fp32"

    if is_cpu:
        reason_bits.append("cpu_forces_fp32")
        policy = "fp32"
        amp_available = False

    use_amp = policy == "fp16_amp" and amp_available
    amp_dtype = torch.float16 if use_amp else torch.float32
    use_grad_scaler = bool(use_amp)
    use_tf32 = False  # normal-mode contract: TF32 disabled

    if use_amp:
        reason_bits.append("fp16_amp_selective_forward")
    else:
        reason_bits.append("fp32_full_precision")

    reason = "; ".join(reason_bits) or "default"

    return PrecisionStrategy(
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        use_tf32=use_tf32,
        use_grad_scaler=use_grad_scaler,
        amp_available=amp_available,
        reason=reason,
    )


def strategy_from_context(context: Optional[Dict[str, Any]]) -> PrecisionStrategy:
    """
    Build a PrecisionStrategy from a partial context mapping (as passed by the
    scheduler / training entrypoint). Missing keys fall through to defaults.
    """
    ctx = dict(context or {})
    requested_policy = ctx.get("precision_policy")
    if requested_policy is None:
        # Fall back to whatever the Pipeline Decision Trace recorded.
        requested_policy = ctx.get("requested_precision_policy", "fp16_amp")
    requested_device = ctx.get("device") or ctx.get("requested_device")
    allow_amp = ctx.get("allow_amp")
    if allow_amp is None and ctx.get("use_amp") is False:
        allow_amp = False
    return resolve_precision_strategy(
        requested_policy=requested_policy,
        requested_device=requested_device,
        allow_amp=allow_amp,
    )


# ------------------------------------------------------------------ backend
def configure_backend_precision(strategy: PrecisionStrategy) -> Dict[str, Any]:
    """
    Apply process-level precision flags. MUST be called from `__main__`, never
    at import time. Disables TF32 for both matmul and cudnn, and (when AMP is
    active) configures the cuDNN benchmark flags conservatively.
    """
    applied: Dict[str, Any] = {"tf32_matmul": False, "tf32_cudnn": False}

    # TF32 is always disabled in normal mode.
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    if strategy.device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = False  # shapes vary per batch
        except Exception:
            pass
        try:
            # Reduce fragmentation on A10's 22 GB VRAM.
            torch.cuda.empty_cache()
        except Exception:
            pass

    applied["torch_amp_available"] = bool(torch.cuda.is_available()) and (
        strategy.device.type == "cuda"
    )
    applied["strategy"] = strategy.as_dict()
    return applied


# ------------------------------------------------------------------ autocast
@contextlib.contextmanager
def autocast_forward(strategy: PrecisionStrategy):
    """
    Autocast context used strictly around eligible forward + loss-compatible
    computation. On CPU or when AMP is inactive this is a no-op context so that
    callers can use the same code path everywhere.
    """
    if not strategy.amp_active:
        yield
        return
    try:
        with torch.autocast(device_type="cuda", dtype=strategy.amp_dtype, enabled=True):
            yield
    except TypeError:
        # Older PyTorch signature.
        with torch.cuda.amp.autocast(dtype=strategy.amp_dtype, enabled=True):
            yield


@contextlib.contextmanager
def fp32_island(strategy: PrecisionStrategy):
    """
    Force FP32 math for FP32-sensitive regions (gate-softmax normalizer, BCE
    reduction, metric accumulation, prediction export). This is a no-op when
    AMP is inactive.
    """
    if not strategy.amp_active:
        yield
        return
    try:
        with torch.autocast(device_type="cuda", enabled=False):
            yield
    except TypeError:
        with torch.cuda.amp.autocast(enabled=False):
            yield


# ------------------------------------------------------------------ scaler
def make_grad_scaler(strategy: PrecisionStrategy):
    """
    Return a GradScaler when FP16 AMP is active, else None. Kept as a thin
    factory so the trainer can pass it straight into diagnostics.
    """
    if not strategy.use_grad_scaler or not strategy.amp_active:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def clip_gradients_(parameters, max_norm: float, scaler) -> None:
    """
    Unscale (when a scaler is active) then clip in-place. Must be called
    OUTSIDE autocast, right before optimizer.step().
    """
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        try:
            scaler.unscale_(parameters)
        except Exception:
            # If unscale_ errors (e.g. already unscaled), continue to clip on
            # current grads rather than crashing the training step.
            pass
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


# ------------------------------------------------------------------ FP32 loss
def compute_bce_with_logits_fp32(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    BCE-with-logits reduction run in FP32 regardless of ambient autocast, to
    protect the loss signal that drives threshold selection and F1 calibration.
    """
    logits_f = logits.float()
    target_f = target.float().view_as(logits_f)
    with fp32_island(resolve_precision_strategy(requested_policy="fp32")):
        return nn.functional.binary_cross_entropy_with_logits(logits_f, target_f)


def is_finite_loss(loss: torch.Tensor) -> bool:
    if loss is None:
        return False
    try:
        return bool(torch.isfinite(loss.detach()).all().item())
    except Exception:
        return False


# ------------------------------------------------------------------ fallback
class DegradeToFP32Tracker:
    """
    Watches for repeated non-finite losses; after `patience` bad steps the
    tracker flips to FP32 via `current_strategy()`. Simple, deterministic, and
    caller-driven (no background state).
    """

    def __init__(self, strategy: PrecisionStrategy, patience: int = 3):
        self._strategy = strategy
        self._patience = max(1, int(patience))
        self._bad = 0
        self.degraded = False

    @property
    def strategy(self) -> PrecisionStrategy:
        return self._strategy

    def observe(self, loss: Optional[torch.Tensor]) -> bool:
        """Record one loss value. Returns True when a fallback was triggered."""
        if self.degraded:
            return False
        if is_finite_loss(loss):
            self._bad = 0
            return False
        self._bad += 1
        if self._bad >= self._patience:
            self._strategy = self._strategy.with_amp_policy(
                False, reason="non_finite_loss_patience_exceeded"
            )
            self.degraded = True
            return True
        return False

    def current_strategy(self) -> PrecisionStrategy:
        return self._strategy


# ------------------------------------------------------------------ logging
def format_precision_log(strategy: PrecisionStrategy) -> str:
    d = strategy.as_dict()
    return (
        "precision | "
        f"device={d['device']} amp={d['amp_active']}({d['amp_dtype']}) "
        f"tf32={d['use_tf32']} scaler={d['use_grad_scaler']} "
        f"family={d['model_family_tag']} reason={d['precision_reason']}"
    )


def precision_to_dict(strategy: PrecisionStrategy) -> Dict[str, Any]:
    """Serialisable snapshot for checkpoints / diagnostics settings."""
    return strategy.as_dict()


# ------------------------------------------------------------------ public entry
def build_precision_context(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience bundle for the training entrypoint: strategy + scaler + log.
    Callers must invoke this inside `__main__`, not at import time (it touches
    CUDA when a GPU is present via `configure_backend_precision`).
    """
    strategy = strategy_from_context(context)
    applied = configure_backend_precision(strategy)
    scaler = make_grad_scaler(strategy)
    return {
        "strategy": strategy,
        "scaler": scaler,
        "backend_applied": applied,
        "log_line": format_precision_log(strategy),
        "settings": precision_to_dict(strategy),
    }

```
        
        ### Step 3: training_evaluation
        **Plan:** Looking at this, my role is Stage 3: implement the training loop that consumes the frozen Stage 1 `KeywordGatedMultiviewNet` (word TF-IDF + char TF-IDF + keyword one-hot + 15 style features, gated fusion) and Stage 2's FP16-AMP precision layer (selective autocast, FP32 island for BCE, GradScaler, `DegradeToFP32Tracker`), then produce a threshold-tuned binary-F1 submission.

The main modeling-hyperparameter choices here: AdamW with linear warmup + cosine decay on the pre-XLNet sparse feature stack (which trains in a handful of epochs), physical batch 64 with no accumulation, early stopping on the F1 metric with best-checkpoint restore, and threshold tuning on the internal holdout — the same threshold is then applied to test predictions to keep val/test processing identical.

## Plan

- **WHAT:** The Stage-1 model family is already wired to sparse TF-IDF views fit only on the train fold; Stage 3 must train it efficiently on A10 while preserving leakage isolation (feature bundle re-fit on the internal train split inside `run_training`).
- **WHY:** Sparse-feature MLPs converge fast, so the main lever is scheduler/regularization/batch rather than long training. I use AdamW + linear-warmup→cosine decay + gradient clipping + early stopping on binary F1, keeping the Stage 2 FP16 AMP path (autocast only around forward, GradScaler, FP32 BCE island) and TF32 disabled.
- **HOW:** Validate by tracking val F1 with the internal 15 % stratified split, threshold tuning on that same holdout, best-checkpoint restore, and test inference with the identical forward + threshold logic; fallback to FP32 if non-finite loss is repeatedly observed. Success = val F1 ≥ 0.75 with all four gates active.

## Note board

- baseline_change: added the full Stage-3 trainer (`run_training`) with AdamW + linear-warmup/cosine LR, grad clipping, early stop + best-checkpoint restore, threshold-tuned binary-F1 validation, and test inference writing `./submission/submission.csv` | purpose: preserve Stage-1 candidate intent (metaphor-sensitive keyword-gated fusion) under the Stage-2 FP16-AMP policy while maximizing the exact F1 metric | hardware_keys: fp16_amp, grad_scaler, num_workers, pin_memory, persistent_workers
- baseline_change: re-fit the feature bundle only on the internal train split inside `run_training` and reuse the exact same `FeatureBundle.transform` for val/test | purpose: kill leak paths while keeping val and test pre-processing byte-identical | hardware_keys: none
- baseline_change: wired `TrainingDiagnostics`, scheduler safe-points (BEFORE_TRAIN/STEP/EPOCH) with a full `state_factory`, and `script_scheduler_context(load_resume_checkpoint)` guarded by `if context is not None` | purpose: satisfy the scheduler contract without altering training semantics on unscheduled runs | hardware_keys: none
        **Code:**
        ```python
# stage: training_evaluation
"""
Stage 3: training, validation, threshold tuning, and submission for the
keyword-gated multi-view sparse MLP. Consumes the Stage 1 model/features/loss
and the Stage 2 precision utilities AS-IS.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

# --------------------------------------------------------------------- constants
MODEL_FAMILY = "keyword_gated_multiview_sparse_mlp_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
INPUT_DIR = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")

# --------------------------------------------------------------------- hyperparams
PHYSICAL_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 1
EFFECTIVE_BATCH_SIZE = PHYSICAL_BATCH_SIZE * GRAD_ACCUM_STEPS
PLANNED_EPOCHS = 20
LEARNING_RATE = 1.5e-3
WEIGHT_DECAY = 3e-4
WARMUP_FRACTION = 0.1
MAX_GRAD_NORM = 1.0
EARLY_STOP_PATIENCE = 4
NUM_WORKERS = 2
EVAL_BATCH_SIZE = 256
WARMUP_EPOCHS = 1

# Scheduler / quality contract constants.
QUALITY_SAFE_PHYSICAL_BATCH_SIZES = [32, 64, 128]
BATCH_LR_SCALING_POLICY = "sqrt"

SEED = 42
VAL_FRACTION = 0.15
HIDDEN_DIM = 256
DROPOUT = 0.35
STYLE_DIM = 15

# --------------------------------------------------------------------- optional imports
try:  # pragma: no cover
    from utils.training_diagnostics import TrainingDiagnostics
except Exception:  # pragma: no cover

    class TrainingDiagnostics(object):  # minimal no-op fallback
        def __init__(self, *args, **kwargs):
            self._state = {}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def after_update(self, *a, **k):
            return None

        def report(self, *a, **k):
            return None

        def state_dict(self):
            return dict(self._state)

        def load_state_dict(self, state):
            if isinstance(state, dict):
                self._state = dict(state)


try:  # pragma: no cover
    from localml_scheduler.execution.script_context import script_scheduler_context
    from localml_scheduler.domain import SafePointType
except Exception:  # pragma: no cover

    def script_scheduler_context():
        return None

    class SafePointType(object):
        BEFORE_TRAIN = "BEFORE_TRAIN"
        STEP = "STEP"
        EPOCH = "EPOCH"


# --------------------------------------------------------------------- import Stage 1/2 symbols
# These symbols are assumed to exist from the previous pipeline stages.
# We import them defensively so that a missing symbol yields an informative
# error inside the main guard rather than at import time.
from __main__ import (  # noqa: F401  (symbols provided by the merged scripts)
    FeatureBundle,
    KeywordGatedMultiviewNet,
    build_model_from_bundle,
    fit_feature_bundle,
    split_data,
    clean_text,
)  # type: ignore


def _precision_utils():
    """Resolve Stage 2 precision helpers from the merged namespace."""
    mod = sys.modules.get("__main__")
    if mod is None:
        return None
    names = (
        "strategy_from_context",
        "configure_backend_precision",
        "make_grad_scaler",
        "autocast_forward",
        "fp32_island",
        "compute_bce_with_logits_fp32",
        "clip_gradients_",
        "is_finite_loss",
        "DegradeToFP32Tracker",
        "format_precision_log",
        "precision_to_dict",
    )
    out = {n: getattr(mod, n, None) for n in names}
    if any(v is None for v in out.values()):
        return None
    return out


# --------------------------------------------------------------------- dataset
class FeatureTensorDataset(Dataset):
    """Wrap precomputed (word, char, kw, style) dense float32 arrays + float target."""

    def __init__(self, word, char, kw, style, target):
        self.word = np.asarray(word, dtype=np.float32)
        self.char = np.asarray(char, dtype=np.float32)
        self.kw = np.asarray(kw, dtype=np.float32)
        self.style = np.asarray(style, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)

    def __len__(self):
        return int(self.target.shape[0])

    def __getitem__(self, idx):
        return {
            "word_x": torch.from_numpy(self.word[idx]),
            "char_x": torch.from_numpy(self.char[idx]),
            "kw_x": torch.from_numpy(self.kw[idx]),
            "style_x": torch.from_numpy(self.style[idx]),
            "target": torch.tensor(float(self.target[idx]), dtype=torch.float32),
        }


# --------------------------------------------------------------------- metric helpers
def tune_threshold(probs: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """Sweep a fine threshold grid to maximise binary F1 (matches the task metric)."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    best_t = 0.5
    best_f1 = 0.0
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs >= t).astype(np.int64)
        score = f1_score(targets, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_t = float(t)
    return best_t, best_f1


# --------------------------------------------------------------------- train / eval steps
def _move_batch(batch, device):
    word = batch["word_x"].to(device, non_blocking=True)
    char = batch["char_x"].to(device, non_blocking=True)
    kw = batch["kw_x"].to(device, non_blocking=True)
    style = batch["style_x"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True).view(-1)
    return word, char, kw, style, target


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    strategy,
    device,
    accum_steps,
    max_grad_norm,
    compute_bce,
    autocast_forward,
    clip_gradients,
    is_finite_loss,
    lr_scheduler=None,
    diagnostics=None,
    on_optimizer_step=None,
):
    """One epoch of training. Returns mean (unscaled) soft loss over accepted batches."""
    model.train()
    total = 0.0
    n_used = 0
    accum = max(1, int(accum_steps))

    for i, batch in enumerate(loader):
        word, char, kw, style, target = _move_batch(batch, device)

        if (i % accum) == 0:
            optimizer.zero_grad(set_to_none=True)

        with autocast_forward(strategy):
            logits = model(word, char, kw, style)

        loss = compute_bce(logits, target)
        if not is_finite_loss(loss):
            # Skip this microbatch entirely (stale/unusable grads) and re-zero.
            optimizer.zero_grad(set_to_none=True)
            continue

        loss_scaled = loss / accum
        if scaler is not None:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        is_last = ((i + 1) % accum == 0) or ((i + 1) == len(loader))
        if is_last:
            clip_gradients(model.parameters(), max_grad_norm, scaler)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if lr_scheduler is not None:
                lr_scheduler.step()
            if diagnostics is not None:
                diagnostics.after_update()
            if on_optimizer_step is not None:
                on_optimizer_step()

        total += float(loss.detach().item())
        n_used += 1

    return total / max(1, n_used)


@torch.no_grad()
def predict_probs(model, loader, strategy, device, autocast_forward):
    """Forward pass producing per-example sigmoid probabilities in FP32."""
    model.eval()
    probs = []
    targets = []
    for batch in loader:
        word, char, kw, style, target = _move_batch(batch, device)
        with autocast_forward(strategy):
            logits = model(word, char, kw, style)
        p = torch.sigmoid(logits.float()).view(-1).cpu().numpy()
        probs.append(p)
        targets.append(target.detach().float().view(-1).cpu().numpy())
    if not probs:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    return np.concatenate(probs), np.concatenate(targets)


# --------------------------------------------------------------------- CPU preflight adapter
_CPU_ADAPTER_CACHE: Dict[str, Any] = {}


def _cpu_get_or_build_bundle():
    """Fit the feature bundle on the internal train split (lazy, CPU-only)."""
    if "bundle" not in _CPU_ADAPTER_CACHE:
        train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
        fit_df, _ = split_data(train_df, seed=SEED, val_fraction=VAL_FRACTION)
        bundle = fit_feature_bundle(fit_df)
        _CPU_ADAPTER_CACHE["bundle"] = bundle
        _CPU_ADAPTER_CACHE["df"] = fit_df.reset_index(drop=True)
    return _CPU_ADAPTER_CACHE["bundle"], _CPU_ADAPTER_CACHE["df"]


class CandidateAdapter(object):
    """No-arg CPU-safe adapter that exercises the real model/loss/features."""

    _DEFAULTS = {
        "hidden": HIDDEN_DIM,
        "dropout": DROPOUT,
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
    }

    def _resolve_context(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        ctx = dict(self._DEFAULTS)
        if context:
            for k, v in context.items():
                if v is not None:
                    ctx[k] = v
        return ctx

    # ---- model / optimizer ---------------------------------------------
    def build_model(self, context):
        bundle, _ = _cpu_get_or_build_bundle()
        ctx = self._resolve_context(context)
        return build_model_from_bundle(
            bundle, hidden=int(ctx["hidden"]), dropout=float(ctx["dropout"])
        )

    def build_optimizer(self, model, context):
        ctx = self._resolve_context(context)
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(ctx["lr"]),
            weight_decay=float(ctx["weight_decay"]),
        )

    # ---- batch builders -------------------------------------------------
    def _build_batch(self, scenario, device):
        scenario = scenario or {}
        batch_size = int(scenario.get("batch_size", 8))
        fixture = scenario.get("fixture") or {}

        if fixture:
            named: Dict[str, torch.Tensor] = {}
            for k, v in fixture.items():
                if isinstance(v, torch.Tensor):
                    named[k] = v.to(device)
                else:
                    named[k] = torch.as_tensor(v, device=device)

            def _take(*names):
                for nm in names:
                    if nm in named:
                        return named[nm]
                return None

            word = _take("word_x", "word", "word_ids")
            char = _take("char_x", "char", "char_ids")
            kw = _take("kw_x", "keyword", "keyword_ids")
            style = _take("style_x", "style", "style_features")
            target = _take("target", "label", "labels", "y")

            vecs = []
            for t in (word, char, kw, style):
                if t is None:
                    continue
                vecs.append(t.to(torch.float32) if t.is_floating_point() else t)
            if not vecs:
                for nm, v in named.items():
                    if nm in ("target", "label", "labels", "y"):
                        continue
                    if isinstance(v, torch.Tensor) and v.is_floating_point():
                        vecs.append(v.to(torch.float32))

            if target is None:
                n = vecs[0].size(0) if vecs else 1
                target = torch.zeros(n, dtype=torch.float32, device=device)

            inputs = tuple(vecs)
            out = {"inputs": inputs, "target": target.to(torch.float32).view(-1)}
            for k, v in named.items():
                out.setdefault(k, v)
            return out

        bundle, df = _cpu_get_or_build_bundle()
        n = min(batch_size, len(df))
        # Deterministic slice for preflight (no shuffle).
        sub = df.iloc[:n].reset_index(drop=True)
        word_X, char_X, kw_X, style_X = bundle.transform(sub)
        inputs = (
            torch.from_numpy(np.ascontiguousarray(word_X, dtype=np.float32)).to(device),
            torch.from_numpy(np.ascontiguousarray(char_X, dtype=np.float32)).to(device),
            torch.from_numpy(np.ascontiguousarray(kw_X, dtype=np.float32)).to(device),
            torch.from_numpy(np.ascontiguousarray(style_X, dtype=np.float32)).to(
                device
            ),
        )
        target = torch.from_numpy(
            sub["target"].values.astype(np.float32, copy=False)
        ).to(device)
        return {"inputs": inputs, "target": target}

    def build_train_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    # ---- steps ----------------------------------------------------------
    @staticmethod
    def _unpack(batch):
        if isinstance(batch, dict):
            inputs = batch.get("inputs")
            target = batch.get("target")
        else:
            inputs, target = batch[0], batch[1]
        if inputs is None:
            inputs = tuple()
        if not isinstance(inputs, (list, tuple)):
            inputs = (inputs,)
        return inputs, target

    def training_step(self, model, batch, context):
        inputs, target = self._unpack(batch)
        inputs = [
            t.to(torch.float32) if torch.is_floating_point(t) else t for t in inputs
        ]
        if target is None:
            n = inputs[0].size(0) if inputs else 1
            target = torch.zeros(
                n, dtype=torch.float32, device=inputs[0].device if inputs else "cpu"
            )
        target = target.to(torch.float32).view(-1)
        logits = model(*inputs)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits.view(-1), target
        )

    def validation_step(self, model, batch, context):
        with torch.no_grad():
            return self.training_step(model, batch, context)


# --------------------------------------------------------------------- main training entrypoint
def run_training():  # noqa: C901
    # ------------------------------------------------------------------
    # Scheduler context / resume (before any CUDA allocs)
    # ------------------------------------------------------------------
    context = script_scheduler_context()
    if context is not None:
        try:
            context.job.max_epochs = PLANNED_EPOCHS
        except Exception:
            pass
        try:
            if hasattr(context.job, "config") and context.job.config is not None:
                context.job.config.max_epochs = PLANNED_EPOCHS
        except Exception:
            pass
        try:
            context.store.save_job(context.job)
        except Exception:
            pass

    start_epoch = 0
    global_step = 0
    resume_state: Optional[Dict[str, Any]] = None
    if context is not None:
        resume = context.load_resume_checkpoint()
        if resume and resume.get("state"):
            resume_state = resume["state"]

    # ------------------------------------------------------------------
    # Data loading + leakage-safe feature fitting
    # ------------------------------------------------------------------
    train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

    train_split, val_split = split_data(train_df, seed=SEED, val_fraction=VAL_FRACTION)
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)

    bundle = fit_feature_bundle(train_split)

    train_word, train_char, train_kw, train_style = bundle.transform(train_split)
    val_word, val_char, val_kw, val_style = bundle.transform(val_split)
    test_word, test_char, test_kw, test_style = bundle.transform(test_df)

    y_train = train_split["target"].astype(np.float32).values
    y_val = val_split["target"].astype(np.float32).values

    train_ds = FeatureTensorDataset(
        train_word, train_char, train_kw, train_style, y_train
    )
    val_ds = FeatureTensorDataset(val_word, val_char, val_kw, val_style, y_val)
    test_ds = FeatureTensorDataset(
        test_word,
        test_char,
        test_kw,
        test_style,
        np.zeros(len(test_df), dtype=np.float32),
    )

    # ------------------------------------------------------------------
    # Precision (Stage 2 policy) + device
    # ------------------------------------------------------------------
    utils = _precision_utils()
    if utils is None:
        raise RuntimeError(
            "Stage 2 precision utilities (strategy_from_context, autocast_forward, "
            "compute_bce_with_logits_fp32, make_grad_scaler, clip_gradients_, "
            "is_finite_loss, DegradeToFP32Tracker, configure_backend_precision, "
            "format_precision_log) must be available in __main__."
        )

    requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    strategy = utils["strategy_from_context"](
        {"precision_policy": "fp16_amp", "device": requested_device}
    )
    utils["configure_backend_precision"](strategy)
    device = strategy.device
    scaler = utils["make_grad_scaler"](strategy)

    decode_tracker = utils["DegradeToFP32Tracker"](strategy, patience=3)
    strategy = decode_tracker.current_strategy()

    print(utils["format_precision_log"](strategy))
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(
            f"[runtime] device={props.name} cap={props.major}.{props.minor} "
            f"vram_mb={props.total_memory // (1024 * 1024)}"
        )

    # ------------------------------------------------------------------
    # Model / optimizer / scheduler
    # ------------------------------------------------------------------
    model = build_model_from_bundle(bundle, hidden=HIDDEN_DIM, dropout=DROPOUT).to(
        device
    )
    # Ensure FP32 parameter dtype (autocast handles fp16 activations locally).
    for p in model.parameters():
        if p.dtype != torch.float32:
            p.data = p.data.float()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    steps_per_epoch = max(1, math.ceil(len(train_ds) / PHYSICAL_BATCH_SIZE))
    total_steps = max(1, steps_per_epoch * PLANNED_EPOCHS)
    warmup_steps = int(total_steps * WARMUP_FRACTION)

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    persistent = bool(NUM_WORKERS > 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=persistent,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=persistent,
    )

    # ------------------------------------------------------------------
    # Resume (optimizer/scaler/scheduler)
    # ------------------------------------------------------------------
    if resume_state is not None:
        try:
            model.load_state_dict(resume_state["model"])
        except Exception:
            pass
        try:
            if resume_state.get("optimizer") is not None:
                optimizer.load_state_dict(resume_state["optimizer"])
        except Exception:
            pass
        try:
            if scaler is not None and resume_state.get("scaler") is not None:
                scaler.load_state_dict(resume_state["scaler"])
        except Exception:
            pass
        try:
            if resume_state.get("lr_scheduler") is not None:
                lr_scheduler.load_state_dict(resume_state["lr_scheduler"])
        except Exception:
            pass
        start_epoch = int(resume_state.get("epoch", 0))
        global_step = int(resume_state.get("global_step", 0))

    # ------------------------------------------------------------------
    # Scheduler state_factory
    # ------------------------------------------------------------------
    run_state: Dict[str, int] = {"epoch": start_epoch, "global_step": global_step}

    def state_factory():
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": run_state["epoch"],
            "global_step": run_state["global_step"],
            "python_rng": None,
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
        }
        try:
            state["python_rng"] = __import__("random").getstate()
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                state["cuda_rng"] = torch.cuda.get_rng_state_all()
            except Exception:
                pass
        return state

    if context is not None:
        context.control_hook.safe_point(
            SafePointType.BEFORE_TRAIN,
            epoch=run_state["epoch"],
            global_step=run_state["global_step"],
            steps_per_epoch=steps_per_epoch,
            state_factory=state_factory,
        )

    # ------------------------------------------------------------------
    # Runtime training diagnostics
    # ------------------------------------------------------------------
    diagnostics_settings = {
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "planned_epochs": PLANNED_EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "model_family": MODEL_FAMILY,
        "split_seed": SEED,
        "precision_choice": "fp16_amp" if strategy.amp_active else "fp32",
        "precision_reason": strategy.reason,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "num_workers": NUM_WORKERS,
    }

    best_f1 = 0.0
    best_threshold = 0.5
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_no_improve = 0
    t0 = time.time()

    with TrainingDiagnostics(
        model, optimizer, scaler=scaler, settings=diagnostics_settings
    ) as diagnostics:
        for epoch in range(start_epoch, PLANNED_EPOCHS):
            run_state["epoch"] = epoch

            def _on_step():
                run_state["global_step"] += 1
                if context is not None:
                    context.control_hook.safe_point(
                        SafePointType.STEP,
                        epoch=epoch,
                        global_step=run_state["global_step"],
                        steps_per_epoch=steps_per_epoch,
                        state_factory=state_factory,
                    )

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                strategy=strategy,
                device=device,
                accum_steps=GRAD_ACCUM_STEPS,
                max_grad_norm=MAX_GRAD_NORM,
                compute_bce=utils["compute_bce_with_logits_fp32"],
                autocast_forward=utils["autocast_forward"],
                clip_gradients=utils["clip_gradients_"],
                is_finite_loss=utils["is_finite_loss"],
                lr_scheduler=lr_scheduler,
                diagnostics=diagnostics,
                on_optimizer_step=_on_step,
            )

            val_probs, val_targets = predict_probs(
                model, val_loader, strategy, device, utils["autocast_forward"]
            )
            threshold, val_f1 = tune_threshold(val_probs, val_targets)

            print(
                f"MLEVOLVE_EPOCH_METRIC "
                f'{{"epoch": {epoch + 1}, "metric": {val_f1:.6f}, "metric_name": "f1"}}'
            )
            print(
                f"Epoch {epoch + 1}: train_loss={train_loss:.4f} "
                f"val_f1={val_f1:.4f} threshold={threshold:.3f} "
                f"lr={lr_scheduler.get_last_lr()[0]:.2e}"
            )

            if diagnostics is not None:
                diagnostics.report(epoch=epoch + 1)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_threshold = threshold
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if context is not None:
                context.control_hook.safe_point(
                    SafePointType.EPOCH,
                    epoch=epoch,
                    global_step=run_state["global_step"],
                    steps_per_epoch=steps_per_epoch,
                    state_factory=state_factory,
                )

            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(
                    f"[early-stop] no F1 improvement for {EARLY_STOP_PATIENCE} epochs"
                )
                break

        # Save best checkpoint including diagnostics state
        os.makedirs(WORKING_DIR, exist_ok=True)
        ckpt_path = os.path.join(WORKING_DIR, "best_model.pt")
        torch.save(
            {
                "model": best_state,
                "threshold": best_threshold,
                "best_f1": best_f1,
                "diagnostics": diagnostics.state_dict(),
                "model_family": MODEL_FAMILY,
                "precision": utils["precision_to_dict"](strategy),
            },
            ckpt_path,
        )

        # Restore best weights INSIDE the diagnostics context for a clean
        # book-keeping snapshot, then predict test.
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        test_probs, _ = predict_probs(
            model, test_loader, strategy, device, utils["autocast_forward"]
        )

    elapsed = time.time() - t0
    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        try:
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        except Exception:
            peak_vram_mb = 0.0

    print(
        f"[runtime] elapsed={elapsed:.1f}s peak_vram_mb={peak_vram_mb:.1f} "
        f"physical_bs={PHYSICAL_BATCH_SIZE} eff_bs={EFFECTIVE_BATCH_SIZE} "
        f"steps_per_epoch={steps_per_epoch} precision={diagnostics_settings['precision_choice']}"
    )

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------
    test_preds = (test_probs >= float(best_threshold)).astype(np.int64)
    sub = pd.DataFrame({"id": test_df["id"].values, "target": test_preds})
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    sub.to_csv(sub_path, index=False)
    print(
        f"[submission] wrote {sub_path} rows={len(sub)} "
        f"pos_rate={float(test_preds.mean()):.4f} threshold={best_threshold:.3f}"
    )

    # ------------------------------------------------------------------
    # Required final line
    # ------------------------------------------------------------------
    print(f"Final Validation Score: {best_f1}")


if __name__ == "__main__":
    run_training()

```
        


# Instructions

## Response format

Your response should be a brief summary (2-3 sentences) of how you merged the steps, followed by a single markdown code block (wrapped in ```) containing the complete merged code. There should be no additional headings or text in your response.

## Hardware/Profile reasoning rule

- HWDB recommended_patterns are conditional candidates, not measured speedups or mandatory steps. Check workload, installed backend, allowed precision and each stated limitation before applying one. A capability or verified flag does not prove benefit for this candidate. Keep the baseline when evidence is missing; record feature keys, unmet conditions and measured results without inventing them.
- Hardware advice never overrides task, dataset, submission, package, model-source, or filesystem constraints. Do not invent Kaggle input paths, local checkpoints, torch hub directories, or placeholder weights; use only paths and model sources that are explicitly available in the prompt/config.
- For hardware/runtime optimization, preserve the parent solution's modeling budget unless the user explicitly asks to improve score. Do not increase epochs, folds, model size, input resolution, ensemble count, TTA, dataset size, or validation workload as a hardware-only optimization.
- Generated code must run as one independent subprocess and follow the exact effective backend named in the backend-guidance section.
- Use backend coding preferences in job code, but leave daemon setup, launch policy, device assignment, active-thread allocation, priorities, and memory limits to the scheduler.
- Prefer scheduler-compatible code with configurable batch size, precision, dataloader workers, checkpoints, and runtime logging when using GPU training.
- The stage-specific hardware node response is already provided in the prompt. Do not query the hardware node again; query local feature-node details only for selected feature keys when deeper implementation guidance is needed.
- Use the Cross-Stage Note Board to keep Stage 2 precision and Stage 3 training choices aligned with the Stage 1 candidate target.


## Pipeline Decision Contract

- Use the Pipeline Decision Trace as the source of truth for code generation and planning.
- Follow the hardware-aware stepwise workflow: model-design -> datatype/quantization -> training (code keys: model_design -> datatype_precision -> training_evaluation).
- Do not jump directly to optimizer, precision, or batch-size choices before model family and output interface are decided.
- The model_design stage stores data modality, target shape, preprocessing/features, model family, loss, criterion, and output interface.
- Numeric precision such as fp32, fp16, bf16, tf32, or TE FP8/NVFP4 belongs under datatype_precision.precision_policy unless the task explicitly requires a numeric type.
- The datatype_precision step may make narrow precision-required model adapters such as Transformer Engine layer wrappers/replacements, padding/config hooks, autocast recipes, or higher-precision islands, but it must preserve the Stage 1 model family, loss, data features, and output interface.
- Hardware/profile evidence may influence model fit and tuning only when it is compatible with the task, installed packages, and available model sources.
- Hardware-only training/runtime tuning must not increase epochs, folds, model size, image resolution, ensemble count, TTA, dataset size, or validation workload.
- If execution feedback contradicts an earlier decision, update only the relevant part of the decision and explain why in the plan.


## 🔬 Critical: Scientific Approach to Design

- 
- Before designing your solution, you must answer three fundamental questions:
- 
- 1. **WHAT makes this task unique?**
- - Not generic observations like 'it's a classification task'
- - What SPECIFIC patterns, challenges, or domain characteristics?
- 
- 2. **WHY is your approach suitable for this task?**
- - Not just 'this model is good' - explain the MATCH between approach and task
- - What properties of your method address the task characteristics?
- 
- 3. **HOW will you validate your hypothesis?**
- - What outcome would confirm your approach is right?
- - What outcome would suggest you need to reconsider?
- 
- ---
- 
- ⚠️ This is not a template to fill - this is how scientists think.
- Blindly applying standard methods without understanding WHY is not acceptable.
- 
- Your plan should naturally reflect this reasoning process.


## Solution sketch guideline

- - This first solution design should be relatively simple — avoid complex ensemble strategies or extensive hyperparameter searches at this stage.
- - 🎯 **CRITICAL: NOVELTY & DIVERSITY REQUIREMENT**:
- • **Mandatory**: Your solution MUST be NOVEL compared to ALL existing attempts in Memory.
- • **Step 1**: Carefully analyze the core idea of EACH previous attempt in Memory.
- • **Step 2 - Choose Strategy**:
- → **Option A (Preferred)**: Propose a COMPLETELY DIFFERENT approach exploring an untried direction.
- → **Option B**: Build upon an existing approach BUT add significant novel insights that fundamentally change the solution.
- • **Forbidden**: Minor variations (changing hyperparameters, swapping similar models, tweaking preprocessing).
- • **Think**: 'Does my approach explore a fundamentally different hypothesis?' If NO → redesign.
- - Don't propose the same modelling solution but keep the evaluation the same.
- - Your plan should be concise but comprehensive: Must address WHAT/WHY/HOW (2-4 sentences each). Avoid verbosity - every sentence should add new insight. Natural length: around 8-12 sentences for a complete reasoning process.
- - Propose an evaluation metric that is reasonable for this task.
- - Don't suggest to do EDA.
- - The data is already prepared in `./input` directory. No need to unzip files.


## Coding & Execution Guidelines (CRITICAL)

- - **NO PROGRESS BARS**: You MUST NOT use `tqdm`. Assume `tqdm` is not installed. Use standard Python loops only. Do not use `verbose=1`.
- - **MINIMAL LOGGING**: Print ONLY 1 line per epoch (e.g. loss/accuracy). Do NOT print batch-level logs.
- - **FINAL OUTPUT**: The VERY LAST line of execution MUST be `print(f'Final Validation Score: {score}')`. This is required for the score parser.


## Backend design contract

- - In the plan, explicitly state the selected memory strategy and physical/effective batch elasticity strategy.
- - State operator and normalization considerations, the model dimensions that remain configurable, and the fallback behavior.
- - List scheduler-owned backend controls that the generated subprocess must not implement.
- - Do not assume code-level integration, shared framework state, or synchronized start time with another job.


## CPU preflight adapter contract

- - **CPU PREFLIGHT CONTRACT (MANDATORY)**: Define a no-argument `class CandidateAdapter` in the generated file. It must implement `build_model(context)`, `build_optimizer(model, context)`, `build_train_batch(scenario, device)`, `build_validation_batch(scenario, device)`, `training_step(model, batch, context)`, and `validation_step(model, batch, context)`.
- - The adapter must exercise the same real model, loss, inputs, and optimizer as the training pipeline; do not use mock tensors or a different toy model. Its batch builders must honor `scenario['batch_size']` and return all inputs required by the model plus a target.
- Batch builders must return (inputs, target) or a mapping with recognized `inputs` and `target` keys. For multiple positional model inputs, use {'inputs': (word_ids, char_ids, keyword_ids), 'target': targets}; training_step and validation_step must unpack those same inputs into the real model. Custom top-level keys alone (such as word_ids) are not recognized. Preserve any authoritative named fixture keys, adding an inputs tuple that references those same tensors when needed. Token/embedding indices remain torch.long and masks keep their required dtype; FP32 precision rules apply to floating-point computation, not integer indices. The checker owns zero_grad, backward and optimizer.step: training_step only computes and returns the connected scalar loss, without performing those updates itself.
- - Keep imports, constants, class/function definitions, and read-only device configuration import-safe. Put all training, validation, prediction, and submission side effects under `if __name__ == '__main__':`.


## Implementation guideline

- Milestone objective: complete one real candidate through review, CPU preflight, GPU training, validation and test submission. There is no overall wall-clock deadline or failed-node budget. Max execution time per run = an hour. Start with a compact trainable model and a complete executable pipeline; do not fabricate metrics, bypass checks, or use dummy predictions.
- 
- Code execution must complete within the configured per-run limit of an hour, including validation and test inference.
- 🎯 **CRITICAL REQUIREMENTS** (Non-Negotiable):
- 
- **1. Model Inference for ALL Predictions**
- • EVERY prediction (validation & test) MUST come from trained model's forward pass
- • Process: Load data → Preprocess → model.predict()/model.forward() → Save predictions
- • ❌ FORBIDDEN: Constants, placeholders, dummy values, empty arrays, statistics, random numbers
- • ❌ FORBIDDEN: Fake/mock metric functions (must use real sklearn.metrics or correct manual implementation)
- • Why: Shortcuts create fake high validation scores but fail on test (CRITICAL SYSTEM FAILURE)
- 
- **2. Generate submission.csv**
- • Path: `./submission/submission.csv` (NOT ./working/submission.csv)
- • Content: Model predictions on ALL test samples
- • Format: Follow task description exactly
- 
- **3. Print Validation Metric**
- • MUST print: `print(f'Final Validation Score: {score}')`
- • Score MUST be computed on hold-out validation set using proper metric formula
- • CRITICAL CONSISTENCY REQUIREMENT: Ensure that validation and test inference use IDENTICAL processing logic. Any differences in how validation and test data are handled (such as post-processing, reconstruction, or formatting) can cause large performance gaps between validation and test sets. Maintain consistency across all data processing steps for both validation and test phases.
- 
- **4. Scheduler Model Family Contract**
- • MUST define a top-level constant `MODEL_FAMILY = "family_name"` near the top of the file.
- • Use a stable, architecture-specific name, for example `resnet50_224`, `swin_b_384`, or `lightgbm_tabular_v1`.
- • If you switch the model architecture or input geometry in an improvement/evolution, update `MODEL_FAMILY` to a new distinct value.
- • If there is no more specific model/backbone name in the script, set the model name/key variable to the same value as `MODEL_FAMILY`.
- 
- **5. CPU Model-Preflight Adapter Contract**
- • The module MUST be import-safe: definitions and lightweight constants may be top-level, but all training, validation, inference, and submission execution MUST be called only inside `if __name__ == "__main__":`.
- • MUST define a no-argument top-level `CandidateAdapter` class with methods `build_model(context)`, `build_optimizer(model, context)`, `build_train_batch(scenario, device)`, `build_validation_batch(scenario, device)`, `training_step(model, batch, context)`, and `validation_step(model, batch, context)`. build_model(context) MUST return the real torch.nn.Module used by training, never a dict, tuple, or metadata wrapper.
- • `training_step` MUST return the real scalar loss tensor connected to model parameters; adapter methods must reuse the script's real model, loss, transforms, collate logic, and optimizer rather than mocks.
- • Treat every caller-supplied `context` as a partial mapping: merge it over adapter defaults before reading optional keys, and preserve caller-provided values.
- • CandidateAdapter context mutations do not persist between checker method calls; `training_step` and `validation_step` MUST resolve the script's real criterion when the partial context omits it or supplies `None`, rather than relying on `build_model` to mutate a local context.
- • Batch builders MUST honor `scenario['batch_size']` and `device`; use `os.environ.get('MLEVOLVE_INPUT_DIR', './input')` to locate input fixtures during the isolated CPU check.
- Batch builders must return (inputs, target) or a mapping with recognized `inputs` and `target` keys. For multiple positional model inputs, use {'inputs': (word_ids, char_ids, keyword_ids), 'target': targets}; training_step and validation_step must unpack those same inputs into the real model. Custom top-level keys alone (such as word_ids) are not recognized. Preserve any authoritative named fixture keys, adding an inputs tuple that references those same tensors when needed. Token/embedding indices remain torch.long and masks keep their required dtype; FP32 precision rules apply to floating-point computation, not integer indices. The checker owns zero_grad, backward and optimizer.step: training_step only computes and returns the connected scalar loss, without performing those updates itself.
- • When `scenario['fixture']` is non-empty, it is the authoritative named input contract: build every listed tensor with the declared non-batch shape, preserve those names in the returned mapping, and pass all model inputs to the real forward path. Do not collapse a multimodal fixture into one feature vector.
- • Adapter construction and a one-batch CPU step must be lightweight and must not download weights or require CUDA.
- 
- 📁 **Directories**: Input data in `./input/`, submission in `./submission/`, temp files in `./working/`
- 
- 📦 **Packages & Internet**: Guaranteed installed packages: numpy, pandas, scikit-learn, statsmodels, xgboost, lightgbm, torch, torchvision, torch-geometric, bayesian-optimization, timm, transformers, sentence-transformers, opencv-python, Pillow. Do not assume unlisted third-party packages are installed; avoid them or provide an ImportError-guarded fallback. Never run pip from generated code. Generated code runs offline: do not download weights or data through torch.hub.load(), Hugging Face, timm, or another remote loader. Use only verified local or cached weights.
- 
- ⚠️ **API Compatibility**: LightGBM/XGBoost: ❌ `fit(..., early_stopping_rounds=...)` → ✅ LightGBM: `fit(..., callbacks=[lgb.early_stopping(...)])` ✅ XGBoost: `XGBClassifier(early_stopping_rounds=...)`
- • AdamW: ❌ `from transformers import AdamW` (deprecated) → ✅ `from torch.optim import AdamW`
- 
- 🚫 **Execution Guidelines**:
- • NO tqdm (not installed), NO verbose=1
- • Print only 1 line per epoch (minimize logging)
- • Runtime diagnostics are an exception: one diagnostics line per epoch plus context entry/exit; no per-batch printing.
- PyTorch runtime diagnostics: import TrainingDiagnostics from utils.training_diagnostics. Inside the training entrypoint, wrap training in `with TrainingDiagnostics(model, optimizer, scaler=scaler_or_None, settings={...}) as diagnostics:` after model/optimizer initialization. Settings must include actual physical_batch_size, effective_batch_size, planned_epochs, steps_per_epoch, model_family, split_seed and precision choice/reason. Call diagnostics.after_update() once after each optimizer update ATTEMPT, after scaler.step and scaler.update (or optimizer.step); never after accumulation microbatches. The helper observes real optimizer steps, including skips, and actual autocast dtype. Call diagnostics.report(epoch=one_based_epoch) once per epoch; keep the normal epoch metric line. Save diagnostics.state_dict() with the training checkpoint and restore it using diagnostics.load_state_dict(...). Preserve these calls during merges/repairs. Do not replace missing measurements with zero. Keep the final validation score as the last line, after leaving the diagnostics context. These diagnostics also run without the scheduler.
- • Use DataLoader with num_workers>=2 for speed
- 
- ⚠️  **Self-Check Before Finalizing**:
- □ Did predictions pass through model's learned weights during inference? (If NO → INVALID)
- □ Did I generate submission.csv in correct path with ALL test predictions?
- □ Did I print validation metric as the last line?
- □ Did I use the COMPLETE training dataset (not a tiny subset)?
- □ Is execution behind the main guard and is CandidateAdapter complete and CPU-safe?
- □ Does scheduled PyTorch training use script_scheduler_context, load_resume_checkpoint, and STEP/EPOCH safe_point calls with steps_per_epoch and a complete state_factory? Preserve these hooks during merges and repairs; keep setup inside the training entrypoint and hooks inactive when the context is None.
- Use the installed API exactly: context.control_hook.safe_point(SafePointType.STEP, epoch=epoch, global_step=global_step, steps_per_epoch=steps_per_epoch, state_factory=state_factory). Use the same keyword arguments for BEFORE_TRAIN and EPOCH. There is no payload= parameter and no second positional argument. Import script_scheduler_context from localml_scheduler.execution.script_context and SafePointType from localml_scheduler.domain. Guard hooks with `if context is not None`; do not catch and suppress hook/checkpoint exceptions. Define state_factory to capture actual model, optimizer, scaler, LR scheduler, RNG, global_step and exact step_in_epoch/sampler_state/data_state. Restore those same state objects and loader position before continuing. Use direct hook calls and TrainingDiagnostics construction so the static admission gate can verify the contract; do not add unreachable token-matching calls.
- 


## Precision policy

- Normal mode permits FP32 and selective FP16 AMP only, not BF16, TF32, quantization or TE recipes. FP16 is an option, not a requirement or a whole-pipeline dtype. Keep model parameters and floating inputs and floating regression targets in FP32; preserve integer indices and labels. Use CUDA autocast only around eligible forward operations and compatible loss computation. Run backward and optimizer updates outside autocast, with GradScaler for FP16 and unscale before gradient clipping. For fragile losses/reductions, disable autocast locally and explicitly cast inputs to FP32. Keep metric accumulation and prediction export in FP32; validation/test forwards may share a validated autocast path. Do not call model.half() or cast the entire batch to half. CPU preflight uses FP32 with AMP/scaling disabled. Disable TF32 for matmul and convolution. Compare finite loss/gradients, task quality and elapsed time against FP32; retain FP32 for unsupported, unstable or slower operations. Record the actual regions and fallback, not an unmeasured speedup.


## 🚨 DATA LEAKAGE PREVENTION

- 
- ⚠️ **Strict Isolation Principle**: Validation/Test data must remain strictly unseen during training.
- 
- ✅ **Sequence**: Always **Split Data FIRST**, then apply processing.
- ✅ **Stateful Transformations**: Fit all Scalers, Encoders, Imputers, and Tokenizers **ONLY on Training data**, then use `.transform()` on Validation/Test.
- ✅ **Feature Engineering**: Calculate global statistics (e.g., mean, variance, vocabulary) solely from the Training set.
- ✅ **Target Leakage**: Never use target information (e.g., Target Encoding) from the validation set.
- 


## Installed Packages

The following packages are guaranteed in the execution environment: `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `xgboost`, `lightgbm`, `torch`, `torchvision`, `torch-geometric`, `bayesian-optimization`, `timm`, `transformers`, `sentence-transformers`, `opencv-python`, `Pillow`. Do not assume any unlisted third-party package is installed: either avoid it or use an ImportError-guarded fallback. Do not run pip from generated code. For neural networks we suggest using PyTorch rather than TensorFlow.

## 💡 Recommendation: Robustness & Generalization Strategy

- 
- **To improve model robustness and generalization on unseen data:**
- 
- ✅ **Architecture**: Match model inductive bias to data structure (e.g., CNNs/ViTs for spatial grids, Transformers/RNNs for sequences, GNNs/GCNs for graphs/topology)
- ✅ **Input Strategy**: Handle variable-length or large-scale inputs via **windowing strategies** or patch-based processing (consider overlap for smoother predictions)
- ✅ **Regularization**: Consider using Dropout, Batch/Layer Norm, Weight Decay, or Label Smoothing
- ✅ **Loss Function**: Inspect class distribution and adapt loss accordingly (e.g., weighted loss, FocalLoss, or task-specific objectives)
- ✅ **Learning Rate**: Consider using adaptive schedules like Cosine Annealing or ReduceLROnPlateau or Warmup with differential rates if needed
- ✅ **Data Augmentation**: Apply domain-appropriate augmentation based on data modality (e.g., geometric transforms, masking, mixup)
- ✅ **Validation**: Monitor validation metrics strictly and use early stopping to prevent overfitting
- 
- ⚠️ **Note**:
- Prioritize capturing the intrinsic structure of the data (Inductive Bias) over simply increasing model size.
- 


## Merge guidelines

- - Combine all code sections into a single, runnable Python script
- - CRITICAL: You are a MERGER, not a designer. Faithfully integrate the code from all steps. Do NOT introduce new models, algorithms, or approaches that were not in the original steps.
- - Ensure variable names are consistent across steps
- - Remove duplicate imports and definitions
- - Resolve conflicts between steps by following the earlier stage's design: model_design defines data preparation, features, model/loss/interface, and criterion; datatype_precision defines precision policy and any precision-required model adapters; training_evaluation consumes both.
- - Ensure the execution flow is logical: model_design -> datatype_precision -> training_evaluation
- - No module-level side effects: keep only imports, constants, classes, and helper definitions at import time. Put data access, environment reads, CUDA/TF32 and logging configuration, model/loss instantiation, DataLoader construction, training, evaluation, and submission writing in one `if __name__ == '__main__':` execution path; keep CandidateAdapter CPU-safe.
- - Make sure the final code prints validation metric (must match task's Evaluation section) and saves submission.csv
- - The code should be a single-file Python program that can be executed as-is
- - Assume previous steps have NOT been executed; do not skip execution steps and only read files or outputs.
- - All parts must work together seamlessly
- - Use hardware context only to preserve compatible precision, precision-required model adapters, batch, dataloader, and runtime choices. Do not redesign the selected model family, loss/interface, or optimizer during merge.
- - Never emit merge-conflict markers such as <<<<<<<, =======, or >>>>>>>. Resolve conflicting snippets into ordinary Python before returning the final code.



Let me approach this systematically.
First, I'll examine the dataset:
```
input/
    description.md (5 lines)
    sample_submission.csv (1524 lines)
    test.csv (1724 lines)
    train.csv (6839 lines)
submission/

working/
```

-> input/description.md has content:

```
# Disaster Tweets: Custom Precision Benchmark

Predict target=1 for tweets about real disasters, target=0 otherwise. Use text, keyword and location as appropriate. The metric is binary F1. Write submission.csv with id,target columns and binary predictions for test.csv. Use only train.csv labels for model selection, with your own internal validation split.

This is a custom benchmark, not an official MLE-bench competition definition or Kaggle leaderboard evaluation. The original labeled Kaggle training data was split 80/20 with stratification and seed 42. Both precision modes use this same holdout; test labels are unavailable to the agents. No leaderboard submission is made.

```

-> input/sample_submission.csv has 1523 rows and 2 columns.
⚠️  IMPORTANT: This is the CORRECT submission format that must be followed!
The exact column names are: id, target
Any description.md format information should be IGNORED if it conflicts with this file.
Here is some information about the columns:
id (int64) has range: 5.00 - 10873.00, 0 nan values
target (int64) has 1 unique values: [0]

-> input/test.csv has 1523 rows and 4 columns.
Here is some information about the columns:
id (int64) has range: 5.00 - 10873.00, 0 nan values

-> input/train.csv has 6090 rows and 5 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 10872.00, 0 nan values
target (int64) has 2 unique values: [1, 0]

        ⚠️  CRITICAL SUBMISSION FORMAT NOTE:
        - If you see sample_submission.csv or similar files, those contain the CORRECT submission format
        - The column names in these files are the FINAL AUTHORITY for submission format
        - Always use the column names from the actual sample submission files
        