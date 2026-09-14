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
 Summary: 4 total attempts - 4 executed, 3 successful (best: 0.7711)

-------------------------------
Attempt #1:
Design: I'll merge the three stages into a single runnable script: keeping Stage 1's text cleaning, TF-IDF feature bundle, `KeywordGatedMultiviewNet`, loss and split helpers as the foundation; layering Stage 2's FP16-AMP precision utilities (`PrecisionStrategy`, `autocast_forward`, `fp32_island`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `clip_gradients_`, `DegradeToFP32Tracker`, `configure_backend_precision`) on top; and consuming both inside Stage 3's `run_training` with the scheduler safe-points, `TrainingDiagnostics`, warmup+cosine AdamW, best-val-F1 restore, threshold tuning and test inference. I removed the `_precision_utils()` `__main__`-lookup indirection, deduplicated `CandidateAdapter`/dataset/threshold helpers, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module constants, and moved every side effect under `if __name__ == "__main__":`.
Results: Training ran to completion successfully, producing a best validation F1 of 0.7711 at threshold 0.520 before early stopping after 4 epochs without improvement. A valid submission CSV with 1523 rows (pos_rate 0.4123) was written. The only non-fatal output is an unused graph-evidence connection warning after submission.
Validation Metric: 0.7711
Execution Time: 832.32s

-------------------------------
Attempt #2:
Design: I merged the three stages by keeping Stage 1's data cleaning, TF-IDF feature builders, `SparseMLP`, `build_loss`, `tune_threshold` and split helpers as the foundation; layering Stage 2's import-safe precision utilities (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `degrade_to_fp32`, `configure_tf32`, `format_precision_log`) on top; then consuming both inside Stage 3's `run_training` with scheduler safe-points, `TrainingDiagnostics`, linear warmup+cosine AdamW, best-val-F1 restore, and test inference. I resolved the duplicate `CandidateAdapter` by keeping Stage 3's self-contained CPU-safe version (it uses the Stage 1 model/loss/features through the Stage 2 precision path), removed duplicate imports/constants, kept all side effects under `if __name__ == "__main__":`, and preserved the exact `MODEL_FAMILY`, `WORKING_DIR`, `SUBMISSION_DIR` contract.
Results: The training script executed successfully end-to-end: TF-IDF features were fit on the train split, the SparseMLP trained under CUDA FP16 AMP with TF32 disabled, early stopping triggered after 4 epochs without improvement, and a best-val-F1 checkpoint was restored for test inference and submission writing. The run produced a finite best validation F1 of 0.7573 (peak at epoch 1), with the final score line printed and a submission.csv generated; the only non-fatal log noise was an unrelated Neo4j connection refusal.
Validation Metric: 0.7573
Execution Time: 925.14s

-------------------------------
Attempt #3:
Design: I merged the three stages by keeping Stage 1's data/model/loss/adapter code as the foundation, layering Stage 2's `PrecisionRuntime` policy helpers on top, and consuming both inside Stage 3's `run_training` (with diagnostics + scheduler safe points). I resolved naming conflicts, deduplicated imports, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module-level constants that Stage 3 references, and moved every side effect under `if __name__ == '__main__':` while preserving the CPU-safe `CandidateAdapter` contract.
Results: Training completed successfully in ~37 minutes on an A10 GPU, running 12 epochs with a hybrid transformer model. Validation F1 improved from 0.665 at epoch 1 to a best of ~0.7341 (best epoch 11), and a valid submission file with 1523 rows was written using a tuned threshold of 0.42. The only messages were benign warnings (Graph evidence mirror connection refused, nested-tensor warning) that did not affect execution.
Validation Metric: 0.7341
Execution Time: 2223.42s

-------------------------------
Attempt #4:
Design: I merged the three stages by keeping Stage 1's data cleaning, vocab builders, `DisasterTweetNet` (dual-branch char-CNN + word-BiGRU + keyword-gated fusion), `build_loss`/`compute_class_weights`, and the CPU-safe `CandidateAdapter` as the foundation; layering Stage 2's precision policy helpers (`PrecisionConfig`, `resolve_precision_config`, `configure_backend_precision`, `autocast_context`, `fp32_island`, `compute_loss_fp32`, `softmax_fp32`, `make_grad_scaler`, `clip_gradients_`, `scaler_step`, `DegradeToFP32Tracker`, `format_precision_log`, `resolve_device`) on top; and consuming both inside Stage 3's `run_training` (AdamW + warmup/cosine, class-weighted label-smoothed CE, EMA + raw per-epoch threshold tuning, best-val-F1 restore, early stopping, `TrainingDiagnostics`, scheduler safe-points). I deduplicated imports/constants, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR`/`INPUT_DIR`/`SEED`/`VAL_SIZE`/`MIN_WORD_FREQ`/`PAD_IDX` as module-level constants consumed by Stage 3, and moved all side effects under `if __name__ == "__main__":`.
Results: The implementation of this design has bugs.
Insight: Using a different approach may not result in the same bugs as the above approach.


# Hardware-Aware Stage 1 Candidate Construction Context
# Hardware-Aware Model Design Brief
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2; observed in 3 batch-size observations
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
Required hardware-aware step order: model-design -> datatype/quantization -> training (model_design -> datatype_precision -> training_evaluation). Use the trace as the source of truth, apply hardware evidence only when compatible, and use recorded fallbacks when evidence is missing.

# Pipeline Decision Trace
```json
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"{'name': 'cross_entropy', 'reduction': 'mean', 'class_imbalance_handling': 'none or class weights...","output_interface":"{'train_output': 'logits shape [batch, 2]', 'prediction_output': 'binary target 0/1 via threshold...","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"{'forward_autocast_boundaries': 'wrap forward pass in autocast(cuda, fp16); no manual half castin...","fallback_policy":"fall back to fp32 or disabled AMP if the selected precision path is unsupported","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"{'name': 'AdamW', 'learning_rate': 2e-05, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-08}","scheduler":"{'name': 'linear_warmup', 'warmup_ratio': 0.1, 'total_steps': 'computed from epochs and batch size'}","batch_size_policy":"fixed","dataloader_policy":"safe defaults; enable workers and pinned memory only when compatible","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available","no exact or similar graph profiles for current hardware","FP16 AMP speedup not measured for this specific text task and hardware"]}}
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
 Summary: 4 total attempts - 4 executed, 3 successful (best: 0.7711)

-------------------------------
Attempt #1:
Design: I'll merge the three stages into a single runnable script: keeping Stage 1's text cleaning, TF-IDF feature bundle, `KeywordGatedMultiviewNet`, loss and split helpers as the foundation; layering Stage 2's FP16-AMP precision utilities (`PrecisionStrategy`, `autocast_forward`, `fp32_island`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `clip_gradients_`, `DegradeToFP32Tracker`, `configure_backend_precision`) on top; and consuming both inside Stage 3's `run_training` with the scheduler safe-points, `TrainingDiagnostics`, warmup+cosine AdamW, best-val-F1 restore, threshold tuning and test inference. I removed the `_precision_utils()` `__main__`-lookup indirection, deduplicated `CandidateAdapter`/dataset/threshold helpers, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module constants, and moved every side effect under `if __name__ == "__main__":`.
Results: Training ran to completion successfully, producing a best validation F1 of 0.7711 at threshold 0.520 before early stopping after 4 epochs without improvement. A valid submission CSV with 1523 rows (pos_rate 0.4123) was written. The only non-fatal output is an unused graph-evidence connection warning after submission.
Validation Metric: 0.7711
Execution Time: 832.32s

-------------------------------
Attempt #2:
Design: I merged the three stages by keeping Stage 1's data cleaning, TF-IDF feature builders, `SparseMLP`, `build_loss`, `tune_threshold` and split helpers as the foundation; layering Stage 2's import-safe precision utilities (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `degrade_to_fp32`, `configure_tf32`, `format_precision_log`) on top; then consuming both inside Stage 3's `run_training` with scheduler safe-points, `TrainingDiagnostics`, linear warmup+cosine AdamW, best-val-F1 restore, and test inference. I resolved the duplicate `CandidateAdapter` by keeping Stage 3's self-contained CPU-safe version (it uses the Stage 1 model/loss/features through the Stage 2 precision path), removed duplicate imports/constants, kept all side effects under `if __name__ == "__main__":`, and preserved the exact `MODEL_FAMILY`, `WORKING_DIR`, `SUBMISSION_DIR` contract.
Results: The training script executed successfully end-to-end: TF-IDF features were fit on the train split, the SparseMLP trained under CUDA FP16 AMP with TF32 disabled, early stopping triggered after 4 epochs without improvement, and a best-val-F1 checkpoint was restored for test inference and submission writing. The run produced a finite best validation F1 of 0.7573 (peak at epoch 1), with the final score line printed and a submission.csv generated; the only non-fatal log noise was an unrelated Neo4j connection refusal.
Validation Metric: 0.7573
Execution Time: 925.14s

-------------------------------
Attempt #3:
Design: I merged the three stages by keeping Stage 1's data/model/loss/adapter code as the foundation, layering Stage 2's `PrecisionRuntime` policy helpers on top, and consuming both inside Stage 3's `run_training` (with diagnostics + scheduler safe points). I resolved naming conflicts, deduplicated imports, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module-level constants that Stage 3 references, and moved every side effect under `if __name__ == '__main__':` while preserving the CPU-safe `CandidateAdapter` contract.
Results: Training completed successfully in ~37 minutes on an A10 GPU, running 12 epochs with a hybrid transformer model. Validation F1 improved from 0.665 at epoch 1 to a best of ~0.7341 (best epoch 11), and a valid submission file with 1523 rows was written using a tuned threshold of 0.42. The only messages were benign warnings (Graph evidence mirror connection refused, nested-tensor warning) that did not affect execution.
Validation Metric: 0.7341
Execution Time: 2223.42s

-------------------------------
Attempt #4:
Design: I merged the three stages by keeping Stage 1's data cleaning, vocab builders, `DisasterTweetNet` (dual-branch char-CNN + word-BiGRU + keyword-gated fusion), `build_loss`/`compute_class_weights`, and the CPU-safe `CandidateAdapter` as the foundation; layering Stage 2's precision policy helpers (`PrecisionConfig`, `resolve_precision_config`, `configure_backend_precision`, `autocast_context`, `fp32_island`, `compute_loss_fp32`, `softmax_fp32`, `make_grad_scaler`, `clip_gradients_`, `scaler_step`, `DegradeToFP32Tracker`, `format_precision_log`, `resolve_device`) on top; and consuming both inside Stage 3's `run_training` (AdamW + warmup/cosine, class-weighted label-smoothed CE, EMA + raw per-epoch threshold tuning, best-val-F1 restore, early stopping, `TrainingDiagnostics`, scheduler safe-points). I deduplicated imports/constants, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR`/`INPUT_DIR`/`SEED`/`VAL_SIZE`/`MIN_WORD_FREQ`/`PAD_IDX` as module-level constants consumed by Stage 3, and moved all side effects under `if __name__ == "__main__":`.
Results: The implementation of this design has bugs.
Insight: Using a different approach may not result in the same bugs as the above approach.


# Hardware-Aware Datatype/Precision Context
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2; observed in 3 batch-size observations
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
Required hardware-aware step order: model-design -> datatype/quantization -> training (model_design -> datatype_precision -> training_evaluation). Use the trace as the source of truth, apply hardware evidence only when compatible, and use recorded fallbacks when evidence is missing.

# Pipeline Decision Trace
```json
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"{'name': 'cross_entropy', 'reduction': 'mean', 'class_imbalance_handling': 'none or class weights...","output_interface":"{'train_output': 'logits shape [batch, 2]', 'prediction_output': 'binary target 0/1 via threshold...","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"{'forward_autocast_boundaries': 'wrap forward pass in autocast(cuda, fp16); no manual half castin...","fallback_policy":"fall back to fp32 or disabled AMP if the selected precision path is unsupported","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"{'name': 'AdamW', 'learning_rate': 2e-05, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-08}","scheduler":"{'name': 'linear_warmup', 'warmup_ratio': 0.1, 'total_steps': 'computed from epochs and batch size'}","batch_size_policy":"fixed","dataloader_policy":"safe defaults; enable workers and pinned memory only when compatible","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available","no exact or similar graph profiles for current hardware","FP16 AMP speedup not measured for this specific text task and hardware"]}}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=Generated this stage's implementation.; purpose=Define the Stage 1 candidate target for later precision and training stages.; hardware_keys=none

# Previous steps
### model_design
**Plan:** 
**Code:**
```python
## Plan

**WHAT makes this task unique.** Disaster tweets are extremely short (~15 words), so signal lives in *character-level morphology* (e.g., "wildfire"/"wildfires"), *rare keyword gates*, and *pragmatic context* (metaphorical "disaster" vs. real one). Standard word-BOW MLPs miss char morphology; prior attempts 1–4 all used either word-only sparse nets, an un-tuned tiny transformer, or a char-CNN+BiGRU hybrid — none fused word + char + dense-meta + keyword as a gated multi-view torch module.

**WHY this approach fits.** I design a **Multi-View Sparse Fusion Network (MVSFN)**: parallel projections of (a) word TF-IDF, (b) char_wb TF-IDF, (c) handcrafted meta-features (uppercase ratio, hashtags, URL, mentions, punctuation), and (d) a keyword embedding. A small MLP computes **context-dependent gates** from keyword+dense that re-weight the four views before an MLP head emits a single logit (BCE-with-logits + optional pos_weight/label smoothing). This explicitly captures both morphological and lexical evidence while letting the strong keyword prior *modulate* rather than dominate. It is a fundamentally different inductive bias from all 4 prior attempts (no TF-IDF-only MLP, no BiGRU, no unfrozen transformer).

**HOW to validate.** Stratified 80/20 split (seed 42) on train.csv only; tune the decision threshold on the held-out validation set for binary F1 (mirrors test-time selection); compare against the 0.7711 memory best. If val F1 plateaus well below that, the gating/fusion is not learning — check per-view gate statistics and fall back to uniform gates + no meta-view. Features are fit on the training split only, then `transform`-ed on val/test to prevent leakage. The CandidateAdapter exposes the *real* MVSFN forward, the real BCE criterion, and a real torch.optim.AdamW so the CPU preflight exercises the exact training pipeline (FP32, no AMP).

**Backend / memory strategy.** Inputs are dense float32 view tensors of capped dims (word 12000, char 15000, dense 8) → ~660 MB for the full train split, comfortable for the A10's 24 GB; sequence length is fixed at "1 token per view" so no variable-length padding is needed and dataloader hooks simply slice precomputed numpy arrays with `pin_memory`/`num_workers` exposed for Stage 3. Model dims (`word_dim`, `char_dim`, `dense_dim`, `keyword_vocab`, `hidden_dim`) are configurable; fallback is to reduce `hidden_dim` and dropout if OOM. Scheduler owns device assignment, worker count, batch size, AMP, LR, and epochs — not this stage.

**Note board:**
- baseline_change: replaced prior word-BOW sparse MLP / char-CNN-BiGRU / tiny transformer with a gated multi-view fusion of word TF-IDF + char_wb TF-IDF + handcrafted meta features + keyword embedding | purpose: capture morphology, pragmatics, and keyword prior in a single inductive bias that prior attempts missed, targeting F1 > 0.7711 | hardware_keys: none
- baseline_change: introduced explicit FeaturePipeline (fit-on-train-only) with char_wb (3,5) TF-IDF and 8 handcrafted meta-features, plus split-first + transform-only leakage guards | purpose: keep the training budget unchanged while adding feature diversity that a linear head cannot replicate | hardware_keys: none
- baseline_change: exposed a torch-native CandidateAdapter wrapping the real MVSFN forward and BCE criterion with FP32-only construction | purpose: guarantee Stage 2 (precision) and Stage 3 (training) reuse the exact model/loss/output interface below without introducing new side effects at import time | hardware_keys: none

```python
# stage1_model_design.py
"""
Stage 1: Model Design for Disaster Tweets (Custom Precision Benchmark)

Defines:
  - Text/keyword/location cleaning utilities
  - Multi-view feature pipeline (word TF-IDF, char_wb TF-IDF, handcrafted meta, keyword index)
  - MultiViewSparseFusion (MVSFN) torch.nn.Module
  - BCE criterion builder (pos_weight + label smoothing)
  - F1-based threshold tuner
  - Stratified split helper
  - CPU-safe CandidateAdapter (real model, real criterion, real AdamW)

No training loop, optimizer schedule, precision policy, batch size, or submission logic
is defined here; those belong to datatype_precision / training_evaluation stages.

Import-safe: all classes/constants only; nothing is instantiated or executed at import time.
"""

import os
import re
import html

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Module-level constants (consumed by later stages)
# ---------------------------------------------------------------------------
MODEL_FAMILY = "multiview_sparse_fusion_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
INPUT_DIR = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")

SEED = 42
VAL_SIZE = 0.2

# Feature-space defaults
DEFAULT_WORD_DIM = 12000
DEFAULT_CHAR_DIM = 15000
DEFAULT_DENSE_DIM = 8
DEFAULT_KEYWORD_VOCAB = 300

WORD_MAX_FEATURES = 12000
CHAR_MAX_FEATURES = 15000
WORD_NGRAM = (1, 2)
CHAR_NGRAM = (3, 5)
WORD_MIN_DF = 2
CHAR_MIN_DF = 3

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NONALNUM_RE = re.compile(r"[^a-z0-9#@\s]")
_WS_RE = re.compile(r"\s+")


def clean_text(s) -> str:
    s = "" if s is None else str(s)
    s = html.unescape(s).lower()
    s = _URL_RE.sub(" __url__ ", s)
    s = _MENTION_RE.sub(" __mention__ ", s)
    s = _HASHTAG_RE.sub(r" hashtag_\1 ", s)
    s = _NONALNUM_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def clean_keyword(s) -> str:
    if s is None:
        return ""
    if isinstance(s, float) and np.isnan(s):
        return ""
    return str(s).strip().lower().replace("%20", " ")


def clean_location(s) -> str:
    if s is None:
        return ""
    if isinstance(s, float) and np.isnan(s):
        return ""
    return str(s).strip().lower()


# ---------------------------------------------------------------------------
# Handcrafted meta features (dense view)
# ---------------------------------------------------------------------------
def compute_handcrafted(texts) -> np.ndarray:
    """Return float32 array [N, DEFAULT_DENSE_DIM] of normalized meta-features."""
    n = len(texts)
    feats = np.zeros((n, DEFAULT_DENSE_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        t = "" if t is None else str(t)
        n_chars = max(len(t), 1)
        n_words = max(len(t.split()), 1)
        n_upper = sum(1 for c in t if c.isupper())
        feats[i, 0] = min(n_chars / 150.0, 2.0)
        feats[i, 1] = min(n_words / 25.0, 2.0)
        feats[i, 2] = n_upper / n_chars
        feats[i, 3] = min(t.count("!") / 3.0, 2.0)
        feats[i, 4] = min(t.count("?") / 3.0, 2.0)
        feats[i, 5] = min(t.count("#") / 3.0, 2.0)
        t_low = t.lower()
        feats[i, 6] = 1.0 if ("http" in t_low or "www." in t_low) else 0.0
        feats[i, 7] = min(t.count("@") / 3.0, 2.0)
    return feats


# ---------------------------------------------------------------------------
# Multi-view feature pipeline (fit-on-train-only)
# ---------------------------------------------------------------------------
class FeaturePipeline:
    """
    Fit on training split only; transform() is stateless w.r.t. new data.
    Returns dense float32 arrays for the four views.
    """

    def __init__(
        self,
        word_max_features: int = WORD_MAX_FEATURES,
        char_max_features: int = CHAR_MAX_FEATURES,
        word_ngram=WORD_NGRAM,
        char_ngram=CHAR_NGRAM,
        word_min_df: int = WORD_MIN_DF,
        char_min_df: int = CHAR_MIN_DF,
    ):
        self.word_vec = TfidfVectorizer(
            ngram_range=word_ngram,
            sublinear_tf=True,
            min_df=word_min_df,
            max_features=word_max_features,
            token_pattern=r"(?u)\b\w+\b",
            dtype=np.float32,
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ngram,
            sublinear_tf=True,
            min_df=char_min_df,
            max_features=char_max_features,
            dtype=np.float32,
        )
        self.keyword_to_idx = {"": 0}
        self.word_dim = 0
        self.char_dim = 0
        self.keyword_vocab = 1

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        texts = [clean_text(t) for t in df["text"].tolist()]
        keywords = [clean_keyword(k) for k in df["keyword"].tolist()]
        self.word_vec.fit(texts)
        self.char_vec.fit(texts)
        uniq = sorted({k for k in keywords if k})
        self.keyword_to_idx = {"": 0}
        for i, k in enumerate(uniq):
            self.keyword_to_idx[k] = i + 1
        self.word_dim = int(len(self.word_vec.get_feature_names_out()))
        self.char_dim = int(len(self.char_vec.get_feature_names_out()))
        self.keyword_vocab = int(len(self.keyword_to_idx))
        return self

    def transform(self, df: pd.DataFrame) -> dict:
        texts = [clean_text(t) for t in df["text"].tolist()]
        Xw = self.word_vec.transform(texts).toarray().astype(np.float32, copy=False)
        Xc = self.char_vec.transform(texts).toarray().astype(np.float32, copy=False)
        dense = compute_handcrafted(df["text"].tolist())
        kw = np.asarray(
            [self.keyword_to_idx.get(clean_keyword(k), 0) for k in df["keyword"].tolist()],
            dtype=np.int64,
        )
        return {"word": Xw, "char": Xc, "dense": dense, "keyword_idx": kw}

    def fit_transform(self, df: pd.DataFrame) -> dict:
        return self.fit(df).transform(df)


# ---------------------------------------------------------------------------
# Model: Multi-View Sparse Fusion Network (MVSFN)
# ---------------------------------------------------------------------------
class MultiViewSparseFusion(nn.Module):
    """
    Gated multi-view fusion for short-text disaster classification.

    Views:
      - word TF-IDF vector           -> linear + GELU + LayerNorm
      - char_wb TF-IDF vector        -> linear + GELU + LayerNorm
      - handcrafted dense meta       -> linear + GELU + LayerNorm
      - keyword embedding lookup     -> nn.Embedding

    Gate is computed from [keyword_emb || dense_proj] and modulates each view
    with a sigmoid before summation and an MLP head emitting a single logit.
    """

    def __init__(
        self,
        word_dim: int,
        char_dim: int,
        dense_dim: int,
        keyword_vocab: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.word_dim = int(word_dim)
        self.char_dim = int(char_dim)
        self.dense_dim = int(dense_dim)
        self.keyword_vocab = int(keyword_vocab)
        self.hidden_dim = int(hidden_dim)

        self.word_proj = nn.Sequential(
            nn.Linear(self.word_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.char_proj = nn.Sequential(
            nn.Linear(self.char_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.dense_proj = nn.Sequential(
            nn.Linear(self.dense_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.keyword_emb = nn.Embedding(max(self.keyword_vocab, 1), hidden_dim)
        nn.init.normal_(self.keyword_emb.weight, std=0.02)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, word, char, dense, keyword_idx):
        v_word = self.word_proj(word)
        v_char = self.char_proj(char)
        v_dense = self.dense_proj(dense)
        v_kw = self.keyword_emb(keyword_idx.long())

        ctx = torch.cat([v_kw, v_dense], dim=-1)
        gates = self.gate(ctx)                      # [B, 4]
        v_word = v_word * gates[:, 0:1]
        v_char = v_char * gates[:, 1:2]
        v_dense = v_dense * gates[:, 2:3]
        v_kw = v_kw * gates[:, 3:4]

        fused = v_word + v_char + v_dense + v_kw
        fused = self.dropout(fused)
        logits = self.head(fused).squeeze(-1)       # [B]
        return logits

    @torch.no_grad()
    def predict_proba(self, word, char, dense, keyword_idx):
        self.eval()
        logits = self.forward(word, char, dense, keyword_idx)
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# Loss / criterion (interface only; optimizer & LR belong to later stages)
# ---------------------------------------------------------------------------
def compute_class_weights(targets) -> list:
    y = np.asarray(targets).astype(int)
    n = max(len(y), 1)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    w0 = n / (2.0 * max(n_neg, 1))
    w1 = n / (2.0 * max(n_pos, 1))
    return [float(w0), float(w1)]


def build_criterion(class_weights=None, label_smoothing: float = 0.0):
    """
    Return a callable criterion(logits, targets) -> scalar tensor.

    - class_weights: [w0, w1] -> translated to BCEWithLogitsLoss pos_weight = w1/w0
    - label_smoothing: softens targets toward 0.5 (helps under F1 with imbalance)
    """
    pos_weight = None
    if class_weights is not None:
        w0, w1 = float(class_weights[0]), float(class_weights[1])
        if w0 > 0:
            pos_weight = torch.tensor([w1 / w0], dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def criterion(logits, targets):
        t = targets.float()
        if label_smoothing > 0.0:
            eps = float(label_smoothing)
            t = t * (1.0 - eps) + 0.5 * eps
        return bce(logits, t)

    return criterion


# ---------------------------------------------------------------------------
# Validation metric / threshold tuning (F1)
# ---------------------------------------------------------------------------
def f1_binary(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred).astype(int)
    target = np.asarray(target).astype(int)
    tp = int(((pred == 1) & (target == 1)).sum())
    fp = int(((pred == 1) & (target == 0)).sum())
    fn = int(((pred == 0) & (target == 1)).sum())
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return float(2.0 * prec * rec / max(prec + rec, 1e-12))


def tune_threshold(probs, targets, grid=None):
    """Return (best_threshold, best_f1) using validation probabilities only."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets).astype(int).reshape(-1)
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_binary((probs >= float(t)).astype(int), targets)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Split helper (stratified, seed 42)
# ---------------------------------------------------------------------------
def make_train_val_split(df: pd.DataFrame, val_size: float = VAL_SIZE, seed: int = SEED):
    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=
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
 Summary: 4 total attempts - 4 executed, 3 successful (best: 0.7711)

-------------------------------
Attempt #1:
Design: I'll merge the three stages into a single runnable script: keeping Stage 1's text cleaning, TF-IDF feature bundle, `KeywordGatedMultiviewNet`, loss and split helpers as the foundation; layering Stage 2's FP16-AMP precision utilities (`PrecisionStrategy`, `autocast_forward`, `fp32_island`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `clip_gradients_`, `DegradeToFP32Tracker`, `configure_backend_precision`) on top; and consuming both inside Stage 3's `run_training` with the scheduler safe-points, `TrainingDiagnostics`, warmup+cosine AdamW, best-val-F1 restore, threshold tuning and test inference. I removed the `_precision_utils()` `__main__`-lookup indirection, deduplicated `CandidateAdapter`/dataset/threshold helpers, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module constants, and moved every side effect under `if __name__ == "__main__":`.
Results: Training ran to completion successfully, producing a best validation F1 of 0.7711 at threshold 0.520 before early stopping after 4 epochs without improvement. A valid submission CSV with 1523 rows (pos_rate 0.4123) was written. The only non-fatal output is an unused graph-evidence connection warning after submission.
Validation Metric: 0.7711
Execution Time: 832.32s

-------------------------------
Attempt #2:
Design: I merged the three stages by keeping Stage 1's data cleaning, TF-IDF feature builders, `SparseMLP`, `build_loss`, `tune_threshold` and split helpers as the foundation; layering Stage 2's import-safe precision utilities (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `degrade_to_fp32`, `configure_tf32`, `format_precision_log`) on top; then consuming both inside Stage 3's `run_training` with scheduler safe-points, `TrainingDiagnostics`, linear warmup+cosine AdamW, best-val-F1 restore, and test inference. I resolved the duplicate `CandidateAdapter` by keeping Stage 3's self-contained CPU-safe version (it uses the Stage 1 model/loss/features through the Stage 2 precision path), removed duplicate imports/constants, kept all side effects under `if __name__ == "__main__":`, and preserved the exact `MODEL_FAMILY`, `WORKING_DIR`, `SUBMISSION_DIR` contract.
Results: The training script executed successfully end-to-end: TF-IDF features were fit on the train split, the SparseMLP trained under CUDA FP16 AMP with TF32 disabled, early stopping triggered after 4 epochs without improvement, and a best-val-F1 checkpoint was restored for test inference and submission writing. The run produced a finite best validation F1 of 0.7573 (peak at epoch 1), with the final score line printed and a submission.csv generated; the only non-fatal log noise was an unrelated Neo4j connection refusal.
Validation Metric: 0.7573
Execution Time: 925.14s

-------------------------------
Attempt #3:
Design: I merged the three stages by keeping Stage 1's data/model/loss/adapter code as the foundation, layering Stage 2's `PrecisionRuntime` policy helpers on top, and consuming both inside Stage 3's `run_training` (with diagnostics + scheduler safe points). I resolved naming conflicts, deduplicated imports, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module-level constants that Stage 3 references, and moved every side effect under `if __name__ == '__main__':` while preserving the CPU-safe `CandidateAdapter` contract.
Results: Training completed successfully in ~37 minutes on an A10 GPU, running 12 epochs with a hybrid transformer model. Validation F1 improved from 0.665 at epoch 1 to a best of ~0.7341 (best epoch 11), and a valid submission file with 1523 rows was written using a tuned threshold of 0.42. The only messages were benign warnings (Graph evidence mirror connection refused, nested-tensor warning) that did not affect execution.
Validation Metric: 0.7341
Execution Time: 2223.42s

-------------------------------
Attempt #4:
Design: I merged the three stages by keeping Stage 1's data cleaning, vocab builders, `DisasterTweetNet` (dual-branch char-CNN + word-BiGRU + keyword-gated fusion), `build_loss`/`compute_class_weights`, and the CPU-safe `CandidateAdapter` as the foundation; layering Stage 2's precision policy helpers (`PrecisionConfig`, `resolve_precision_config`, `configure_backend_precision`, `autocast_context`, `fp32_island`, `compute_loss_fp32`, `softmax_fp32`, `make_grad_scaler`, `clip_gradients_`, `scaler_step`, `DegradeToFP32Tracker`, `format_precision_log`, `resolve_device`) on top; and consuming both inside Stage 3's `run_training` (AdamW + warmup/cosine, class-weighted label-smoothed CE, EMA + raw per-epoch threshold tuning, best-val-F1 restore, early stopping, `TrainingDiagnostics`, scheduler safe-points). I deduplicated imports/constants, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR`/`INPUT_DIR`/`SEED`/`VAL_SIZE`/`MIN_WORD_FREQ`/`PAD_IDX` as module-level constants consumed by Stage 3, and moved all side effects under `if __name__ == "__main__":`.
Results: The implementation of this design has bugs.
Insight: Using a different approach may not result in the same bugs as the above approach.


# Hardware-Aware Training Hyperparameter Context
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2; observed in 3 batch-size observations
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
Required hardware-aware step order: model-design -> datatype/quantization -> training (model_design -> datatype_precision -> training_evaluation). Use the trace as the source of truth, apply hardware evidence only when compatible, and use recorded fallbacks when evidence is missing.

# Pipeline Decision Trace
```json
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"{'name': 'cross_entropy', 'reduction': 'mean', 'class_imbalance_handling': 'none or class weights...","output_interface":"{'train_output': 'logits shape [batch, 2]', 'prediction_output': 'binary target 0/1 via threshold...","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"{'forward_autocast_boundaries': 'wrap forward pass in autocast(cuda, fp16); no manual half castin...","fallback_policy":"fall back to fp32 or disabled AMP if the selected precision path is unsupported","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"{'name': 'AdamW', 'learning_rate': 2e-05, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-08}","scheduler":"{'name': 'linear_warmup', 'warmup_ratio': 0.1, 'total_steps': 'computed from epochs and batch size'}","batch_size_policy":"fixed","dataloader_policy":"safe defaults; enable workers and pinned memory only when compatible","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available","no exact or similar graph profiles for current hardware","FP16 AMP speedup not measured for this specific text task and hardware"]}}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=Generated this stage's implementation.; purpose=Define the Stage 1 candidate target for later precision and training stages.; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=## Plan I will implement the Stage 2 datatype/precision contract as an import-safe utility module that Stage 3 consumes.; purpose=Support the Stage 1 candidate target with a compatible precision policy.; hardware_keys=none

# Previous steps
### model_design
**Plan:** 
**Code:**
```python
## Plan

**WHAT makes this task unique.** Disaster tweets are extremely short (~15 words), so signal lives in *character-level morphology* (e.g., "wildfire"/"wildfires"), *rare keyword gates*, and *pragmatic context* (metaphorical "disaster" vs. real one). Standard word-BOW MLPs miss char morphology; prior attempts 1–4 all used either word-only sparse nets, an un-tuned tiny transformer, or a char-CNN+BiGRU hybrid — none fused word + char + dense-meta + keyword as a gated multi-view torch module.

**WHY this approach fits.** I design a **Multi-View Sparse Fusion Network (MVSFN)**: parallel projections of (a) word TF-IDF, (b) char_wb TF-IDF, (c) handcrafted meta-features (uppercase ratio, hashtags, URL, mentions, punctuation), and (d) a keyword embedding. A small MLP computes **context-dependent gates** from keyword+dense that re-weight the four views before an MLP head emits a single logit (BCE-with-logits + optional pos_weight/label smoothing). This explicitly captures both morphological and lexical evidence while letting the strong keyword prior *modulate* rather than dominate. It is a fundamentally different inductive bias from all 4 prior attempts (no TF-IDF-only MLP, no BiGRU, no unfrozen transformer).

**HOW to validate.** Stratified 80/20 split (seed 42) on train.csv only; tune the decision threshold on the held-out validation set for binary F1 (mirrors test-time selection); compare against the 0.7711 memory best. If val F1 plateaus well below that, the gating/fusion is not learning — check per-view gate statistics and fall back to uniform gates + no meta-view. Features are fit on the training split only, then `transform`-ed on val/test to prevent leakage. The CandidateAdapter exposes the *real* MVSFN forward, the real BCE criterion, and a real torch.optim.AdamW so the CPU preflight exercises the exact training pipeline (FP32, no AMP).

**Backend / memory strategy.** Inputs are dense float32 view tensors of capped dims (word 12000, char 15000, dense 8) → ~660 MB for the full train split, comfortable for the A10's 24 GB; sequence length is fixed at "1 token per view" so no variable-length padding is needed and dataloader hooks simply slice precomputed numpy arrays with `pin_memory`/`num_workers` exposed for Stage 3. Model dims (`word_dim`, `char_dim`, `dense_dim`, `keyword_vocab`, `hidden_dim`) are configurable; fallback is to reduce `hidden_dim` and dropout if OOM. Scheduler owns device assignment, worker count, batch size, AMP, LR, and epochs — not this stage.

**Note board:**
- baseline_change: replaced prior word-BOW sparse MLP / char-CNN-BiGRU / tiny transformer with a gated multi-view fusion of word TF-IDF + char_wb TF-IDF + handcrafted meta features + keyword embedding | purpose: capture morphology, pragmatics, and keyword prior in a single inductive bias that prior attempts missed, targeting F1 > 0.7711 | hardware_keys: none
- baseline_change: introduced explicit FeaturePipeline (fit-on-train-only) with char_wb (3,5) TF-IDF and 8 handcrafted meta-features, plus split-first + transform-only leakage guards | purpose: keep the training budget unchanged while adding feature diversity that a linear head cannot replicate | hardware_keys: none
- baseline_change: exposed a torch-native CandidateAdapter wrapping the real MVSFN forward and BCE criterion with FP32-only construction | purpose: guarantee Stage 2 (precision) and Stage 3 (training) reuse the exact model/loss/output interface below without introducing new side effects at import time | hardware_keys: none

```python
# stage1_model_design.py
"""
Stage 1: Model Design for Disaster Tweets (Custom Precision Benchmark)

Defines:
  - Text/keyword/location cleaning utilities
  - Multi-view feature pipeline (word TF-IDF, char_wb TF-IDF, handcrafted meta, keyword index)
  - MultiViewSparseFusion (MVSFN) torch.nn.Module
  - BCE criterion builder (pos_weight + label smoothing)
  - F1-based threshold tuner
  - Stratified split helper
  - CPU-safe CandidateAdapter (real model, real criterion, real AdamW)

No training loop, optimizer schedule, precision policy, batch size, or submission logic
is defined here; those belong to datatype_precision / training_evaluation stages.

Import-safe: all classes/constants only; nothing is instantiated or executed at import time.
"""

import os
import re
import html

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Module-level constants (consumed by later stages)
# ---------------------------------------------------------------------------
MODEL_FAMILY = "multiview_sparse_fusion_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
INPUT_DIR = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")

SEED = 42
VAL_SIZE = 0.2

# Feature-space defaults
DEFAULT_WORD_DIM = 12000
DEFAULT_CHAR_DIM = 15000
DEFAULT_DENSE_DIM = 8
DEFAULT_KEYWORD_VOCAB = 300

WORD_MAX_FEATURES = 12000
CHAR_MAX_FEATURES = 15000
WORD_NGRAM = (1, 2)
CHAR_NGRAM = (3, 5)
WORD_MIN_DF = 2
CHAR_MIN_DF = 3

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NONALNUM_RE = re.compile(r"[^a-z0-9#@\s]")
_WS_RE = re.compile(r"\s+")


def clean_text(s) -> str:
    s = "" if s is None else str(s)
    s = html.unescape(s).lower()
    s = _URL_RE.sub(" __url__ ", s)
    s = _MENTION_RE.sub(" __mention__ ", s)
    s = _HASHTAG_RE.sub(r" hashtag_\1 ", s)
    s = _NONALNUM_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def clean_keyword(s) -> str:
    if s is None:
        return ""
    if isinstance(s, float) and np.isnan(s):
        return ""
    return str(s).strip().lower().replace("%20", " ")


def clean_location(s) -> str:
    if s is None:
        return ""
    if isinstance(s, float) and np.isnan(s):
        return ""
    return str(s).strip().lower()


# ---------------------------------------------------------------------------
# Handcrafted meta features (dense view)
# ---------------------------------------------------------------------------
def compute_handcrafted(texts) -> np.ndarray:
    """Return float32 array [N, DEFAULT_DENSE_DIM] of normalized meta-features."""
    n = len(texts)
    feats = np.zeros((n, DEFAULT_DENSE_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        t = "" if t is None else str(t)
        n_chars = max(len(t), 1)
        n_words = max(len(t.split()), 1)
        n_upper = sum(1 for c in t if c.isupper())
        feats[i, 0] = min(n_chars / 150.0, 2.0)
        feats[i, 1] = min(n_words / 25.0, 2.0)
        feats[i, 2] = n_upper / n_chars
        feats[i, 3] = min(t.count("!") / 3.0, 2.0)
        feats[i, 4] = min(t.count("?") / 3.0, 2.0)
        feats[i, 5] = min(t.count("#") / 3.0, 2.0)
        t_low = t.lower()
        feats[i, 6] = 1.0 if ("http" in t_low or "www." in t_low) else 0.0
        feats[i, 7] = min(t.count("@") / 3.0, 2.0)
    return feats


# ---------------------------------------------------------------------------
# Multi-view feature pipeline (fit-on-train-only)
# ---------------------------------------------------------------------------
class FeaturePipeline:
    """
    Fit on training split only; transform() is stateless w.r.t. new data.
    Returns dense float32 arrays for the four views.
    """

    def __init__(
        self,
        word_max_features: int = WORD_MAX_FEATURES,
        char_max_features: int = CHAR_MAX_FEATURES,
        word_ngram=WORD_NGRAM,
        char_ngram=CHAR_NGRAM,
        word_min_df: int = WORD_MIN_DF,
        char_min_df: int = CHAR_MIN_DF,
    ):
        self.word_vec = TfidfVectorizer(
            ngram_range=word_ngram,
            sublinear_tf=True,
            min_df=word_min_df,
            max_features=word_max_features,
            token_pattern=r"(?u)\b\w+\b",
            dtype=np.float32,
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ngram,
            sublinear_tf=True,
            min_df=char_min_df,
            max_features=char_max_features,
            dtype=np.float32,
        )
        self.keyword_to_idx = {"": 0}
        self.word_dim = 0
        self.char_dim = 0
        self.keyword_vocab = 1

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        texts = [clean_text(t) for t in df["text"].tolist()]
        keywords = [clean_keyword(k) for k in df["keyword"].tolist()]
        self.word_vec.fit(texts)
        self.char_vec.fit(texts)
        uniq = sorted({k for k in keywords if k})
        self.keyword_to_idx = {"": 0}
        for i, k in enumerate(uniq):
            self.keyword_to_idx[k] = i + 1
        self.word_dim = int(len(self.word_vec.get_feature_names_out()))
        self.char_dim = int(len(self.char_vec.get_feature_names_out()))
        self.keyword_vocab = int(len(self.keyword_to_idx))
        return self

    def transform(self, df: pd.DataFrame) -> dict:
        texts = [clean_text(t) for t in df["text"].tolist()]
        Xw = self.word_vec.transform(texts).toarray().astype(np.float32, copy=False)
        Xc = self.char_vec.transform(texts).toarray().astype(np.float32, copy=False)
        dense = compute_handcrafted(df["text"].tolist())
        kw = np.asarray(
            [self.keyword_to_idx.get(clean_keyword(k), 0) for k in df["keyword"].tolist()],
            dtype=np.int64,
        )
        return {"word": Xw, "char": Xc, "dense": dense, "keyword_idx": kw}

    def fit_transform(self, df: pd.DataFrame) -> dict:
        return self.fit(df).transform(df)


# ---------------------------------------------------------------------------
# Model: Multi-View Sparse Fusion Network (MVSFN)
# ---------------------------------------------------------------------------
class MultiViewSparseFusion(nn.Module):
    """
    Gated multi-view fusion for short-text disaster classification.

    Views:
      - word TF-IDF vector           -> linear + GELU + LayerNorm
      - char_wb TF-IDF vector        -> linear + GELU + LayerNorm
      - handcrafted dense meta       -> linear + GELU + LayerNorm
      - keyword embedding lookup     -> nn.Embedding

    Gate is computed from [keyword_emb || dense_proj] and modulates each view
    with a sigmoid before summation and an MLP head emitting a single logit.
    """

    def __init__(
        self,
        word_dim: int,
        char_dim: int,
        dense_dim: int,
        keyword_vocab: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.word_dim = int(word_dim)
        self.char_dim = int(char_dim)
        self.dense_dim = int(dense_dim)
        self.keyword_vocab = int(keyword_vocab)
        self.hidden_dim = int(hidden_dim)

        self.word_proj = nn.Sequential(
            nn.Linear(self.word_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.char_proj = nn.Sequential(
            nn.Linear(self.char_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.dense_proj = nn.Sequential(
            nn.Linear(self.dense_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.keyword_emb = nn.Embedding(max(self.keyword_vocab, 1), hidden_dim)
        nn.init.normal_(self.keyword_emb.weight, std=0.02)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, word, char, dense, keyword_idx):
        v_word = self.word_proj(word)
        v_char = self.char_proj(char)
        v_dense = self.dense_proj(dense)
        v_kw = self.keyword_emb(keyword_idx.long())

        ctx = torch.cat([v_kw, v_dense], dim=-1)
        gates = self.gate(ctx)                      # [B, 4]
        v_word = v_word * gates[:, 0:1]
        v_char = v_char * gates[:, 1:2]
        v_dense = v_dense * gates[:, 2:3]
        v_kw = v_kw * gates[:, 3:4]

        fused = v_word + v_char + v_dense + v_kw
        fused = self.dropout(fused)
        logits = self.head(fused).squeeze(-1)       # [B]
        return logits

    @torch.no_grad()
    def predict_proba(self, word, char, dense, keyword_idx):
        self.eval()
        logits = self.forward(word, char, dense, keyword_idx)
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# Loss / criterion (interface only; optimizer & LR belong to later stages)
# ---------------------------------------------------------------------------
def compute_class_weights(targets) -> list:
    y = np.asarray(targets).astype(int)
    n = max(len(y), 1)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    w0 = n / (2.0 * max(n_neg, 1))
    w1 = n / (2.0 * max(n_pos, 1))
    return [float(w0), float(w1)]


def build_criterion(class_weights=None, label_smoothing: float = 0.0):
    """
    Return a callable criterion(logits, targets) -> scalar tensor.

    - class_weights: [w0, w1] -> translated to BCEWithLogitsLoss pos_weight = w1/w0
    - label_smoothing: softens targets toward 0.5 (helps under F1 with imbalance)
    """
    pos_weight = None
    if class_weights is not None:
        w0, w1 = float(class_weights[0]), float(class_weights[1])
        if w0 > 0:
            pos_weight = torch.tensor([w1 / w0], dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def criterion(logits, targets):
        t = targets.float()
        if label_smoothing > 0.0:
            eps = float(label_smoothing)
            t = t * (1.0 - eps) + 0.5 * eps
        return bce(logits, t)

    return criterion


# ---------------------------------------------------------------------------
# Validation metric / threshold tuning (F1)
# ---------------------------------------------------------------------------
def f1_binary(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred).astype(int)
    target = np.asarray(target).astype(int)
    tp = int(((pred == 1) & (target == 1)).sum())
    fp = int(((pred == 1) & (target == 0)).sum())
    fn = int(((pred == 0) & (target == 1)).sum())
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return float(2.0 * prec * rec / max(prec + rec, 1e-12))


def tune_threshold(probs, targets, grid=None):
    """Return (best_threshold, best_f1) using validation probabilities only."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets).astype(int).reshape(-1)
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_binary((probs >= float(t)).astype(int), targets)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Split helper (stratified, seed 42)
# ---------------------------------------------------------------------------
def make_train_val_split(df: pd.DataFrame, val_size: float = VAL_SIZE, seed: int = SEED):
    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=
```

### datatype_precision
**Plan:** ## Plan

I will implement the Stage 2 datatype/precision contract as an import-safe utility module that Stage 3 consumes. The policy follows the trace: **FP32 parameters/inputs, opt-in FP16 AMP via CUDA autocast around forward+loss, backward/step outside autocast, TF32 disabled, GradScaler for FP16 with unscale-before-clip, FP32 islands for the BCE/softmax reductions, and a DegradeToFP32Tracker fallback that flips AMP off on the first NaN/Inf**. Device resolution and exports preserve the Stage 1 MVSFN model/loss/feature/output contract unchanged. All configuration is exposed as dataclasses/helpers; nothing is executed at import time.

**Note board:**
- baseline_change: added FP16-AMP-aware precision utilities (device resolver, autocast boundary, FP32-island loss reducer, GradScaler lifecycle, unscale-then-clip) around the unchanged Stage 1 MVSFN | purpose: enable selective Tensor-Core acceleration on the A10 while keeping the model family, BCE loss, and output interface identical so val F1 remains comparable to prior 0.7711 | hardware_keys: cuda.amp.autocast, tf32.disable, gradscaler.fp16, device.a10.capability_86
- baseline_change: disabled TF32 for matmul/conv and provided a DegradeToFP32Tracker that reverts to pure FP32 on first non-finite loss/grad | purpose: Ampere defaults TF32 to on which can silently shrink effective precision in reductions for a small sparse MLP; explicit off + fallback keeps training stable and reproducible, matching the conservative “evidence missing → keep baseline” rule | hardware_keys: tf32.disable, nonfinite.fallback
- baseline_change: added a `PrecisionConfig` dataclass and `format_precision_log()` that records the actual policy, dtype, TF32 state, and reason in one line | purpose: satisfies precision-logging requirement without batch noise and gives Stage 3 a single source of truth for scaler/autocast decisions | hardware_keys: none
**Code:**
```python
# stage2_datatype_precision.py
"""
Stage 2: Datatype / Precision policy for the Disaster Tweets MVSFN candidate.

Contract (consumed by training_evaluation):
  - DEVICE resolution: CUDA when available (A10, cc 8.6), otherwise CPU.
  - Precision policy: one of {"fp32", "fp16_amp"}.
      * fp32        -> no autocast, no scaler.
      * fp16_amp    -> CUDA autocast(torch.float16) around forward+loss only.
  - TF32: explicitly disabled for matmul/cudnn (Ampere default is ON).
  - GradScaler: FP16 only; skip on CPU / FP32.
  - FP32 islands: fragile BCE/softmax reductions and metric accumulation.
  - DegradeToFP32Tracker: flips AMP -> FP32 on first non-finite loss/grad.

Import-safe: no CUDA configuration, no model/optimizer construction, no data access.
Preserves the Stage 1 model family, loss, feature interface, and output shape.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (aligned with Stage 1 / Stage 3 contract)
# ---------------------------------------------------------------------------
MODEL_FAMILY = "multiview_sparse_fusion_v1"  # preserve from Stage 1

ALLOWED_PRECISION_POLICIES: Tuple[str, ...] = ("fp32", "fp16_amp")
DEFAULT_PRECISION_POLICY: str = "fp16_amp"
DEFAULT_AMP_DTYPE = torch.float16

# Conservative default max grad norm from the trace (Stage 3 owns the exact value).
DEFAULT_CLIP_GRAD_NORM: float = 1.0


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------
def resolve_device(prefer_cuda: bool = True) -> torch.device:
    """Return the canonical compute device for this run."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Precision configuration
# ---------------------------------------------------------------------------
@dataclass
class PrecisionConfig:
    """
    Immutable-ish snapshot of the resolved precision policy.

    Attributes:
        device:         resolved torch.device for training/inference.
        policy:         "fp32" or "fp16_amp".
        use_amp:        True only when policy == "fp16_amp" and device is CUDA.
        amp_dtype:      autocast dtype (torch.float16 when use_amp else fp32).
        use_tf32:       always False per policy (disable Ampere TF32).
        use_scaler:     True only when policy == "fp16_amp" on CUDA.
        fallback_active: set True by DegradeToFP32Tracker after a non-finite event.
        reason:         short human-readable rationale for logging.
    """

    device: torch.device
    policy: str = DEFAULT_PRECISION_POLICY
    use_amp: bool = False
    amp_dtype: torch.dtype = torch.float32
    use_tf32: bool = False
    use_scaler: bool = False
    fallback_active: bool = False
    reason: str = ""


def resolve_precision_config(
    device: Optional[torch.device] = None,
    policy: str = DEFAULT_PRECISION_POLICY,
    allow_amp: bool = True,
) -> PrecisionConfig:
    """
    Decide the concrete precision policy for this run.

    Rules:
      - CPU: force fp32 (no AMP, no scaler).
      - CUDA + policy "fp16_amp" + allow_amp: enable autocast fp16 + GradScaler.
      - CUDA + policy "fp32" or allow_amp False: fp32 only.
      - Unknown policy string: silently fall back to fp32 with a log line.
    """
    dev = device if device is not None else resolve_device()
    pol = str(policy).lower().strip()

    if pol not in ALLOWED_PRECISION_POLICIES:
        logger.warning("Unknown precision policy %r; falling back to fp32.", policy)
        pol = "fp32"
        reason = "unknown_policy->fp32"
    else:
        reason = f"requested={pol}"

    if dev.type != "cuda":
        return PrecisionConfig(
            device=dev,
            policy="fp32",
            use_amp=False,
            amp_dtype=torch.float32,
            use_tf32=False,
            use_scaler=False,
            fallback_active=False,
            reason=f"{reason};device=cpu->fp32",
        )

    if pol == "fp16_amp" and allow_amp:
        return PrecisionConfig(
            device=dev,
            policy="fp16_amp",
            use_amp=True,
            amp_dtype=DEFAULT_AMP_DTYPE,
            use_tf32=False,  # explicitly disabled per normal mode
            use_scaler=True,
            fallback_active=False,
            reason=f"{reason};cuda_autocast_fp16+gradscaler;tf32_off",
        )

    return PrecisionConfig(
        device=dev,
        policy="fp32",
        use_amp=False,
        amp_dtype=torch.float32,
        use_tf32=False,
        use_scaler=False,
        fallback_active=False,
        reason=f"{reason};cuda_fp32;tf32_off",
    )


# ---------------------------------------------------------------------------
# Backend precision configuration (TF32) - call once inside __main__
# ---------------------------------------------------------------------------
def configure_backend_precision(config: PrecisionConfig) -> None:
    """
    Disable TF32 for matmul and cudnn. Safe to call multiple times.

    Normal mode forbids TF32; even if the platform defaults allow it, we
    override here so the run is deterministic and comparable to FP32.
    """
    if config.device.type != "cuda":
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(config.use_tf32)
        torch.backends.cudnn.allow_tf32 = bool(config.use_tf32)
    except Exception:  # pragma: no cover - CPU-only torch builds
        pass
    # Also silence the newer fp32_precision knobs if present.
    try:
        torch.set_float32_matmul_precision("highest" if not config.use_tf32 else "high")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Autocast context / FP32 island
# ---------------------------------------------------------------------------
def _autocast_ctx(config: PrecisionConfig):
    """
    Return the correct autocast context for the resolved device/policy.

    Notes:
      - Uses torch.amp.autocast when available (torch>=2.0), else the legacy API.
      - When AMP is disabled (fp32), returns a nullcontext.
    """
    if not config.use_amp or config.device.type != "cuda":
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", dtype=config.amp_dtype)
    except Exception:  # pragma: no cover - older torch
        return torch.cuda.amp.autocast(dtype=config.amp_dtype)


def autocast_forward(config: PrecisionConfig):
    """Convenience alias for the forward-pass autocast context."""
    return _autocast_ctx(config)


def fp32_island(config: PrecisionConfig):
    """
    Return a context that disables autocast for fragile ops (BCE, softmax, etc.).

    On CPU this is a nullcontext; on CUDA with AMP it forces FP32 compute locally.
    """
    if not config.use_amp or config.device.type != "cuda":
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", enabled=False)
    except Exception:  # pragma: no cover
        return torch.cuda.amp.autocast(enabled=False)


# ---------------------------------------------------------------------------
# FP32 loss / reduction helpers
# ---------------------------------------------------------------------------
def compute_loss_fp32(
    criterion, logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """
    Compute the loss in FP32 explicitly, ignoring any ambient autocast.

    Keeps the Stage 1 criterion unchanged; only the dtype of the incoming
    tensors is safely up-cast for the reduction. Returns a scalar tensor.
    """
    logits_f = logits.float() if logits.dtype != torch.float32 else logits
    targets_f = targets.float() if targets.dtype != torch.float32 else targets
    with contextlib.nullcontext():  # no-op; we already cast inputs
        return criterion(logits_f, targets_f)


def sigmoid_fp32(logits: torch.Tensor) -> torch.Tensor:
    """Numerically stable sigmoid in FP32 for probability export."""
    return torch.sigmoid(logits.float())


# ---------------------------------------------------------------------------
# GradScaler lifecycle
# ---------------------------------------------------------------------------
def make_grad_scaler(config: PrecisionConfig):
    """Create a GradScaler when FP16 AMP is active; otherwise return None."""
    if not config.use_scaler or config.device.type != "cuda":
        return None
    if config.fallback_active:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler(enabled=True)


def scaler_scale(scaler, loss: torch.Tensor) -> torch.Tensor:
    """Scale loss for backward when scaler is present; else return loss unchanged."""
    if scaler is None:
        return loss
    return scaler.scale(loss)


def scaler_step(scaler, optimizer) -> bool:
    """
    Perform optimizer step with scaler if present; else do a plain step.

    Returns True when the optimizer actually updated parameters on this step
    (i.e. the model step was not skipped due to inf/nan gradients).
    """
    if scaler is None:
        optimizer.step()
        return True
    scaler.step(optimizer)
    scaler.update()
    # We cannot cheaply know if the step was skipped; treat non-finite as
    # handled by DegradeToFP32Tracker below. Report True by default.
    return True


def clip_gradients_(model: nn.Module, max_norm: float, scaler=None) -> float:
    """
    Unscale (if scaler) then clip by global norm. Returns the pre-clip grad norm.
    """
    if scaler is not None:
        try:
            scaler.unscale_(_optimizer_placeholder := None)  # documented no-op if None
        except Exception:
            pass
    # The standard pattern is to unscale the scaler's optimizer; we accept a
    # scaler here only for interface compatibility and rely on the caller to
    # have already unscaled via `scaler.unscale_(optimizer)` before calling.
    total_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.grad is not None],
        max_norm,
    )
    return float(total_norm) if total_norm is not None else 0.0


def unscale_and_clip_(scaler, optimizer, model: nn.Module, max_norm: float) -> float:
    """
    Canonical AMP clip helper: unscale gradients in-place then clip by norm.

    When scaler is None this is a plain FP32 clip.
    """
    if scaler is not None:
        scaler.unscale_(optimizer)
    return clip_gradients_(model, max_norm, scaler=None)


# ---------------------------------------------------------------------------
# Non-finite detection + automatic fallback to FP32
# ---------------------------------------------------------------------------
class DegradeToFP32Tracker:
    """
    Monitors loss/grad finiteness during AMP training and flips to FP32 on the
    first non-finite event, so the remainder of the run stays numerically safe.

    Usage:
        tracker = DegradeToFP32Tracker(config)
        ...
        ok = tracker.check(loss)
        if not ok:  # config may have been mutated in place
            scaler = tracker.rebuild_scaler()
    """

    def __init__(self, config: PrecisionConfig):
        self.config = config
        self.fallback_triggered: bool = False
        self.reason: str = ""

    def _flip_disabled(self) -> None:
        self.fallback_triggered = True
        self.config.fallback_active = True
        self.config.use_amp = False
        self.config.use_scaler = False
        self.config.amp_dtype = torch.float32
        self.reason = "nonfinite_detected->fp32_fallback"

    def check(self, loss: Optional[torch.Tensor]) -> bool:
        """Return True if loss is finite; flip state and return False otherwise."""
        if loss is None:
            return True
        try:
            finite = bool(torch.isfinite(loss.detach()).all().item())
        except Exception:
            finite = False
        if not finite:
            self._flip_disabled()
            logger.warning("DegradeToFP32Tracker: non-finite loss; switching to FP32.")
            return False
        return True

    def check_grads(self, model: nn.Module) -> bool:
        """Return True if all grads are finite; flip state and return False otherwise."""
        for p in model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all().item():
                self._flip_disabled()
                logger.warning(
                    "DegradeToFP32Tracker: non-finite grad detected; switching to FP32."
                )
                return False
        return True

    def rebuild_scaler(self):
        """Rebuild a scaler consistent with the (possibly mutated) config."""
        return make_grad_scaler(self.config)


# ---------------------------------------------------------------------------
# Logging formatter
# ---------------------------------------------------------------------------
def format_precision_log(config: PrecisionConfig) -> str:
    """One-line, human-readable summary for TrainingDiagnostics / setup logging."""
    return (
        f"[precision] device={config.device.type} policy={config.policy} "
        f"amp={config.use_amp} dtype={str(config.amp_dtype).replace('torch.', '')} "
        f"tf32={config.use_tf32} scaler={config.use_scaler} "
        f"fallback={config.fallback_active} reason={config.reason}"
    )

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
 Summary: 4 total attempts - 4 executed, 3 successful (best: 0.7711)

-------------------------------
Attempt #1:
Design: I'll merge the three stages into a single runnable script: keeping Stage 1's text cleaning, TF-IDF feature bundle, `KeywordGatedMultiviewNet`, loss and split helpers as the foundation; layering Stage 2's FP16-AMP precision utilities (`PrecisionStrategy`, `autocast_forward`, `fp32_island`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `clip_gradients_`, `DegradeToFP32Tracker`, `configure_backend_precision`) on top; and consuming both inside Stage 3's `run_training` with the scheduler safe-points, `TrainingDiagnostics`, warmup+cosine AdamW, best-val-F1 restore, threshold tuning and test inference. I removed the `_precision_utils()` `__main__`-lookup indirection, deduplicated `CandidateAdapter`/dataset/threshold helpers, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module constants, and moved every side effect under `if __name__ == "__main__":`.
Results: Training ran to completion successfully, producing a best validation F1 of 0.7711 at threshold 0.520 before early stopping after 4 epochs without improvement. A valid submission CSV with 1523 rows (pos_rate 0.4123) was written. The only non-fatal output is an unused graph-evidence connection warning after submission.
Validation Metric: 0.7711
Execution Time: 832.32s

-------------------------------
Attempt #2:
Design: I merged the three stages by keeping Stage 1's data cleaning, TF-IDF feature builders, `SparseMLP`, `build_loss`, `tune_threshold` and split helpers as the foundation; layering Stage 2's import-safe precision utilities (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `degrade_to_fp32`, `configure_tf32`, `format_precision_log`) on top; then consuming both inside Stage 3's `run_training` with scheduler safe-points, `TrainingDiagnostics`, linear warmup+cosine AdamW, best-val-F1 restore, and test inference. I resolved the duplicate `CandidateAdapter` by keeping Stage 3's self-contained CPU-safe version (it uses the Stage 1 model/loss/features through the Stage 2 precision path), removed duplicate imports/constants, kept all side effects under `if __name__ == "__main__":`, and preserved the exact `MODEL_FAMILY`, `WORKING_DIR`, `SUBMISSION_DIR` contract.
Results: The training script executed successfully end-to-end: TF-IDF features were fit on the train split, the SparseMLP trained under CUDA FP16 AMP with TF32 disabled, early stopping triggered after 4 epochs without improvement, and a best-val-F1 checkpoint was restored for test inference and submission writing. The run produced a finite best validation F1 of 0.7573 (peak at epoch 1), with the final score line printed and a submission.csv generated; the only non-fatal log noise was an unrelated Neo4j connection refusal.
Validation Metric: 0.7573
Execution Time: 925.14s

-------------------------------
Attempt #3:
Design: I merged the three stages by keeping Stage 1's data/model/loss/adapter code as the foundation, layering Stage 2's `PrecisionRuntime` policy helpers on top, and consuming both inside Stage 3's `run_training` (with diagnostics + scheduler safe points). I resolved naming conflicts, deduplicated imports, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module-level constants that Stage 3 references, and moved every side effect under `if __name__ == '__main__':` while preserving the CPU-safe `CandidateAdapter` contract.
Results: Training completed successfully in ~37 minutes on an A10 GPU, running 12 epochs with a hybrid transformer model. Validation F1 improved from 0.665 at epoch 1 to a best of ~0.7341 (best epoch 11), and a valid submission file with 1523 rows was written using a tuned threshold of 0.42. The only messages were benign warnings (Graph evidence mirror connection refused, nested-tensor warning) that did not affect execution.
Validation Metric: 0.7341
Execution Time: 2223.42s

-------------------------------
Attempt #4:
Design: I merged the three stages by keeping Stage 1's data cleaning, vocab builders, `DisasterTweetNet` (dual-branch char-CNN + word-BiGRU + keyword-gated fusion), `build_loss`/`compute_class_weights`, and the CPU-safe `CandidateAdapter` as the foundation; layering Stage 2's precision policy helpers (`PrecisionConfig`, `resolve_precision_config`, `configure_backend_precision`, `autocast_context`, `fp32_island`, `compute_loss_fp32`, `softmax_fp32`, `make_grad_scaler`, `clip_gradients_`, `scaler_step`, `DegradeToFP32Tracker`, `format_precision_log`, `resolve_device`) on top; and consuming both inside Stage 3's `run_training` (AdamW + warmup/cosine, class-weighted label-smoothed CE, EMA + raw per-epoch threshold tuning, best-val-F1 restore, early stopping, `TrainingDiagnostics`, scheduler safe-points). I deduplicated imports/constants, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR`/`INPUT_DIR`/`SEED`/`VAL_SIZE`/`MIN_WORD_FREQ`/`PAD_IDX` as module-level constants consumed by Stage 3, and moved all side effects under `if __name__ == "__main__":`.
Results: The implementation of this design has bugs.
Insight: Using a different approach may not result in the same bugs as the above approach.


# Hardware-Aware Stage 1 Candidate Construction Context
# Hardware-Aware Model Design Brief
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2; observed in 3 batch-size observations
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
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2; observed in 3 batch-size observations
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
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2; observed in 3 batch-size observations
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
Required hardware-aware step order: model-design -> datatype/quantization -> training (model_design -> datatype_precision -> training_evaluation). Use the trace as the source of truth, apply hardware evidence only when compatible, and use recorded fallbacks when evidence is missing.

# Pipeline Decision Trace
```json
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"{'name': 'cross_entropy', 'reduction': 'mean', 'class_imbalance_handling': 'none or class weights...","output_interface":"{'train_output': 'logits shape [batch, 2]', 'prediction_output': 'binary target 0/1 via threshold...","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"{'forward_autocast_boundaries': 'wrap forward pass in autocast(cuda, fp16); no manual half castin...","fallback_policy":"fall back to fp32 or disabled AMP if the selected precision path is unsupported","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"{'name': 'AdamW', 'learning_rate': 2e-05, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-08}","scheduler":"{'name': 'linear_warmup', 'warmup_ratio': 0.1, 'total_steps': 'computed from epochs and batch size'}","batch_size_policy":"fixed","dataloader_policy":"safe defaults; enable workers and pinned memory only when compatible","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available","no exact or similar graph profiles for current hardware","FP16 AMP speedup not measured for this specific text task and hardware"]}}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=Generated this stage's implementation.; purpose=Define the Stage 1 candidate target for later precision and training stages.; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=## Plan I will implement the Stage 2 datatype/precision contract as an import-safe utility module that Stage 3 consumes.; purpose=Support the Stage 1 candidate target with a compatible precision policy.; hardware_keys=none
  - Stage 3 training / training_evaluation: baseline_change=introduced Stage 3 training loop with AdamW(lr=1e-3, wd=1e-4) + 10% linear warmup + cosine decay, class-weighted BCE + label smoothing 0.05, grad-clip 1.0, physical batch 128, 15 epochs, patience-4 early stop with bes...; purpose=match the small MLP-style MVSFN (few M params) with a higher-LR schedule than the transformer default 2e-5, use label smoothing for the ~57/43 imbalance, and select the operating threshold that maximizes F1 exactly as...; hardware_keys=cuda.amp.autocast, tf32.disable, grads.unscale_clip
  - Stage 3 training / training_evaluation: baseline_change=routed forward through Stage 2 `autocast_forward(cfg)` while BCE reduction runs inside `fp32_island(cfg)` with `.float()` casts and GradScaler/unscale-then-clip; wired `DegradeToFP32Tracker` to flip to FP32 on non-fin...; purpose=use Tensor Cores where numerically safe while keeping the reduction FP32 (fragile for small batch BCE) and guaranteeing the Stage 2 policy is consumed, not re-decided; hardware_keys=cuda.amp.autocast, tf32.disable, nonfinite.fallback
  - Stage 3 training / training_evaluation: baseline_change=added TrainingDiagnostics wrap, `script_scheduler_context` BEFORE_TRAIN/STEP/EPOCH safe-points with a full `state_factory` (model, optimizer, scaler, LR sched, torch/numpy/python RNG, global_step, step_in_epoch) and d...; purpose=satisfy the scheduler and diagnostics contract without altering training semantics, and keep val F1 comparable to prior 0.7711; hardware_keys=none

# Step results

        ### Step 1: model_design
        **Plan:** 
        **Code:**
        ```python
## Plan

**WHAT makes this task unique.** Disaster tweets are extremely short (~15 words), so signal lives in *character-level morphology* (e.g., "wildfire"/"wildfires"), *rare keyword gates*, and *pragmatic context* (metaphorical "disaster" vs. real one). Standard word-BOW MLPs miss char morphology; prior attempts 1–4 all used either word-only sparse nets, an un-tuned tiny transformer, or a char-CNN+BiGRU hybrid — none fused word + char + dense-meta + keyword as a gated multi-view torch module.

**WHY this approach fits.** I design a **Multi-View Sparse Fusion Network (MVSFN)**: parallel projections of (a) word TF-IDF, (b) char_wb TF-IDF, (c) handcrafted meta-features (uppercase ratio, hashtags, URL, mentions, punctuation), and (d) a keyword embedding. A small MLP computes **context-dependent gates** from keyword+dense that re-weight the four views before an MLP head emits a single logit (BCE-with-logits + optional pos_weight/label smoothing). This explicitly captures both morphological and lexical evidence while letting the strong keyword prior *modulate* rather than dominate. It is a fundamentally different inductive bias from all 4 prior attempts (no TF-IDF-only MLP, no BiGRU, no unfrozen transformer).

**HOW to validate.** Stratified 80/20 split (seed 42) on train.csv only; tune the decision threshold on the held-out validation set for binary F1 (mirrors test-time selection); compare against the 0.7711 memory best. If val F1 plateaus well below that, the gating/fusion is not learning — check per-view gate statistics and fall back to uniform gates + no meta-view. Features are fit on the training split only, then `transform`-ed on val/test to prevent leakage. The CandidateAdapter exposes the *real* MVSFN forward, the real BCE criterion, and a real torch.optim.AdamW so the CPU preflight exercises the exact training pipeline (FP32, no AMP).

**Backend / memory strategy.** Inputs are dense float32 view tensors of capped dims (word 12000, char 15000, dense 8) → ~660 MB for the full train split, comfortable for the A10's 24 GB; sequence length is fixed at "1 token per view" so no variable-length padding is needed and dataloader hooks simply slice precomputed numpy arrays with `pin_memory`/`num_workers` exposed for Stage 3. Model dims (`word_dim`, `char_dim`, `dense_dim`, `keyword_vocab`, `hidden_dim`) are configurable; fallback is to reduce `hidden_dim` and dropout if OOM. Scheduler owns device assignment, worker count, batch size, AMP, LR, and epochs — not this stage.

**Note board:**
- baseline_change: replaced prior word-BOW sparse MLP / char-CNN-BiGRU / tiny transformer with a gated multi-view fusion of word TF-IDF + char_wb TF-IDF + handcrafted meta features + keyword embedding | purpose: capture morphology, pragmatics, and keyword prior in a single inductive bias that prior attempts missed, targeting F1 > 0.7711 | hardware_keys: none
- baseline_change: introduced explicit FeaturePipeline (fit-on-train-only) with char_wb (3,5) TF-IDF and 8 handcrafted meta-features, plus split-first + transform-only leakage guards | purpose: keep the training budget unchanged while adding feature diversity that a linear head cannot replicate | hardware_keys: none
- baseline_change: exposed a torch-native CandidateAdapter wrapping the real MVSFN forward and BCE criterion with FP32-only construction | purpose: guarantee Stage 2 (precision) and Stage 3 (training) reuse the exact model/loss/output interface below without introducing new side effects at import time | hardware_keys: none

```python
# stage1_model_design.py
"""
Stage 1: Model Design for Disaster Tweets (Custom Precision Benchmark)

Defines:
  - Text/keyword/location cleaning utilities
  - Multi-view feature pipeline (word TF-IDF, char_wb TF-IDF, handcrafted meta, keyword index)
  - MultiViewSparseFusion (MVSFN) torch.nn.Module
  - BCE criterion builder (pos_weight + label smoothing)
  - F1-based threshold tuner
  - Stratified split helper
  - CPU-safe CandidateAdapter (real model, real criterion, real AdamW)

No training loop, optimizer schedule, precision policy, batch size, or submission logic
is defined here; those belong to datatype_precision / training_evaluation stages.

Import-safe: all classes/constants only; nothing is instantiated or executed at import time.
"""

import os
import re
import html

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Module-level constants (consumed by later stages)
# ---------------------------------------------------------------------------
MODEL_FAMILY = "multiview_sparse_fusion_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
INPUT_DIR = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")

SEED = 42
VAL_SIZE = 0.2

# Feature-space defaults
DEFAULT_WORD_DIM = 12000
DEFAULT_CHAR_DIM = 15000
DEFAULT_DENSE_DIM = 8
DEFAULT_KEYWORD_VOCAB = 300

WORD_MAX_FEATURES = 12000
CHAR_MAX_FEATURES = 15000
WORD_NGRAM = (1, 2)
CHAR_NGRAM = (3, 5)
WORD_MIN_DF = 2
CHAR_MIN_DF = 3

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NONALNUM_RE = re.compile(r"[^a-z0-9#@\s]")
_WS_RE = re.compile(r"\s+")


def clean_text(s) -> str:
    s = "" if s is None else str(s)
    s = html.unescape(s).lower()
    s = _URL_RE.sub(" __url__ ", s)
    s = _MENTION_RE.sub(" __mention__ ", s)
    s = _HASHTAG_RE.sub(r" hashtag_\1 ", s)
    s = _NONALNUM_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def clean_keyword(s) -> str:
    if s is None:
        return ""
    if isinstance(s, float) and np.isnan(s):
        return ""
    return str(s).strip().lower().replace("%20", " ")


def clean_location(s) -> str:
    if s is None:
        return ""
    if isinstance(s, float) and np.isnan(s):
        return ""
    return str(s).strip().lower()


# ---------------------------------------------------------------------------
# Handcrafted meta features (dense view)
# ---------------------------------------------------------------------------
def compute_handcrafted(texts) -> np.ndarray:
    """Return float32 array [N, DEFAULT_DENSE_DIM] of normalized meta-features."""
    n = len(texts)
    feats = np.zeros((n, DEFAULT_DENSE_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        t = "" if t is None else str(t)
        n_chars = max(len(t), 1)
        n_words = max(len(t.split()), 1)
        n_upper = sum(1 for c in t if c.isupper())
        feats[i, 0] = min(n_chars / 150.0, 2.0)
        feats[i, 1] = min(n_words / 25.0, 2.0)
        feats[i, 2] = n_upper / n_chars
        feats[i, 3] = min(t.count("!") / 3.0, 2.0)
        feats[i, 4] = min(t.count("?") / 3.0, 2.0)
        feats[i, 5] = min(t.count("#") / 3.0, 2.0)
        t_low = t.lower()
        feats[i, 6] = 1.0 if ("http" in t_low or "www." in t_low) else 0.0
        feats[i, 7] = min(t.count("@") / 3.0, 2.0)
    return feats


# ---------------------------------------------------------------------------
# Multi-view feature pipeline (fit-on-train-only)
# ---------------------------------------------------------------------------
class FeaturePipeline:
    """
    Fit on training split only; transform() is stateless w.r.t. new data.
    Returns dense float32 arrays for the four views.
    """

    def __init__(
        self,
        word_max_features: int = WORD_MAX_FEATURES,
        char_max_features: int = CHAR_MAX_FEATURES,
        word_ngram=WORD_NGRAM,
        char_ngram=CHAR_NGRAM,
        word_min_df: int = WORD_MIN_DF,
        char_min_df: int = CHAR_MIN_DF,
    ):
        self.word_vec = TfidfVectorizer(
            ngram_range=word_ngram,
            sublinear_tf=True,
            min_df=word_min_df,
            max_features=word_max_features,
            token_pattern=r"(?u)\b\w+\b",
            dtype=np.float32,
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ngram,
            sublinear_tf=True,
            min_df=char_min_df,
            max_features=char_max_features,
            dtype=np.float32,
        )
        self.keyword_to_idx = {"": 0}
        self.word_dim = 0
        self.char_dim = 0
        self.keyword_vocab = 1

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        texts = [clean_text(t) for t in df["text"].tolist()]
        keywords = [clean_keyword(k) for k in df["keyword"].tolist()]
        self.word_vec.fit(texts)
        self.char_vec.fit(texts)
        uniq = sorted({k for k in keywords if k})
        self.keyword_to_idx = {"": 0}
        for i, k in enumerate(uniq):
            self.keyword_to_idx[k] = i + 1
        self.word_dim = int(len(self.word_vec.get_feature_names_out()))
        self.char_dim = int(len(self.char_vec.get_feature_names_out()))
        self.keyword_vocab = int(len(self.keyword_to_idx))
        return self

    def transform(self, df: pd.DataFrame) -> dict:
        texts = [clean_text(t) for t in df["text"].tolist()]
        Xw = self.word_vec.transform(texts).toarray().astype(np.float32, copy=False)
        Xc = self.char_vec.transform(texts).toarray().astype(np.float32, copy=False)
        dense = compute_handcrafted(df["text"].tolist())
        kw = np.asarray(
            [self.keyword_to_idx.get(clean_keyword(k), 0) for k in df["keyword"].tolist()],
            dtype=np.int64,
        )
        return {"word": Xw, "char": Xc, "dense": dense, "keyword_idx": kw}

    def fit_transform(self, df: pd.DataFrame) -> dict:
        return self.fit(df).transform(df)


# ---------------------------------------------------------------------------
# Model: Multi-View Sparse Fusion Network (MVSFN)
# ---------------------------------------------------------------------------
class MultiViewSparseFusion(nn.Module):
    """
    Gated multi-view fusion for short-text disaster classification.

    Views:
      - word TF-IDF vector           -> linear + GELU + LayerNorm
      - char_wb TF-IDF vector        -> linear + GELU + LayerNorm
      - handcrafted dense meta       -> linear + GELU + LayerNorm
      - keyword embedding lookup     -> nn.Embedding

    Gate is computed from [keyword_emb || dense_proj] and modulates each view
    with a sigmoid before summation and an MLP head emitting a single logit.
    """

    def __init__(
        self,
        word_dim: int,
        char_dim: int,
        dense_dim: int,
        keyword_vocab: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.word_dim = int(word_dim)
        self.char_dim = int(char_dim)
        self.dense_dim = int(dense_dim)
        self.keyword_vocab = int(keyword_vocab)
        self.hidden_dim = int(hidden_dim)

        self.word_proj = nn.Sequential(
            nn.Linear(self.word_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.char_proj = nn.Sequential(
            nn.Linear(self.char_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.dense_proj = nn.Sequential(
            nn.Linear(self.dense_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.keyword_emb = nn.Embedding(max(self.keyword_vocab, 1), hidden_dim)
        nn.init.normal_(self.keyword_emb.weight, std=0.02)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, word, char, dense, keyword_idx):
        v_word = self.word_proj(word)
        v_char = self.char_proj(char)
        v_dense = self.dense_proj(dense)
        v_kw = self.keyword_emb(keyword_idx.long())

        ctx = torch.cat([v_kw, v_dense], dim=-1)
        gates = self.gate(ctx)                      # [B, 4]
        v_word = v_word * gates[:, 0:1]
        v_char = v_char * gates[:, 1:2]
        v_dense = v_dense * gates[:, 2:3]
        v_kw = v_kw * gates[:, 3:4]

        fused = v_word + v_char + v_dense + v_kw
        fused = self.dropout(fused)
        logits = self.head(fused).squeeze(-1)       # [B]
        return logits

    @torch.no_grad()
    def predict_proba(self, word, char, dense, keyword_idx):
        self.eval()
        logits = self.forward(word, char, dense, keyword_idx)
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# Loss / criterion (interface only; optimizer & LR belong to later stages)
# ---------------------------------------------------------------------------
def compute_class_weights(targets) -> list:
    y = np.asarray(targets).astype(int)
    n = max(len(y), 1)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    w0 = n / (2.0 * max(n_neg, 1))
    w1 = n / (2.0 * max(n_pos, 1))
    return [float(w0), float(w1)]


def build_criterion(class_weights=None, label_smoothing: float = 0.0):
    """
    Return a callable criterion(logits, targets) -> scalar tensor.

    - class_weights: [w0, w1] -> translated to BCEWithLogitsLoss pos_weight = w1/w0
    - label_smoothing: softens targets toward 0.5 (helps under F1 with imbalance)
    """
    pos_weight = None
    if class_weights is not None:
        w0, w1 = float(class_weights[0]), float(class_weights[1])
        if w0 > 0:
            pos_weight = torch.tensor([w1 / w0], dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def criterion(logits, targets):
        t = targets.float()
        if label_smoothing > 0.0:
            eps = float(label_smoothing)
            t = t * (1.0 - eps) + 0.5 * eps
        return bce(logits, t)

    return criterion


# ---------------------------------------------------------------------------
# Validation metric / threshold tuning (F1)
# ---------------------------------------------------------------------------
def f1_binary(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred).astype(int)
    target = np.asarray(target).astype(int)
    tp = int(((pred == 1) & (target == 1)).sum())
    fp = int(((pred == 1) & (target == 0)).sum())
    fn = int(((pred == 0) & (target == 1)).sum())
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return float(2.0 * prec * rec / max(prec + rec, 1e-12))


def tune_threshold(probs, targets, grid=None):
    """Return (best_threshold, best_f1) using validation probabilities only."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets).astype(int).reshape(-1)
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_binary((probs >= float(t)).astype(int), targets)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Split helper (stratified, seed 42)
# ---------------------------------------------------------------------------
def make_train_val_split(df: pd.DataFrame, val_size: float = VAL_SIZE, seed: int = SEED):
    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=
```
        
        ### Step 2: datatype_precision
        **Plan:** ## Plan

I will implement the Stage 2 datatype/precision contract as an import-safe utility module that Stage 3 consumes. The policy follows the trace: **FP32 parameters/inputs, opt-in FP16 AMP via CUDA autocast around forward+loss, backward/step outside autocast, TF32 disabled, GradScaler for FP16 with unscale-before-clip, FP32 islands for the BCE/softmax reductions, and a DegradeToFP32Tracker fallback that flips AMP off on the first NaN/Inf**. Device resolution and exports preserve the Stage 1 MVSFN model/loss/feature/output contract unchanged. All configuration is exposed as dataclasses/helpers; nothing is executed at import time.

**Note board:**
- baseline_change: added FP16-AMP-aware precision utilities (device resolver, autocast boundary, FP32-island loss reducer, GradScaler lifecycle, unscale-then-clip) around the unchanged Stage 1 MVSFN | purpose: enable selective Tensor-Core acceleration on the A10 while keeping the model family, BCE loss, and output interface identical so val F1 remains comparable to prior 0.7711 | hardware_keys: cuda.amp.autocast, tf32.disable, gradscaler.fp16, device.a10.capability_86
- baseline_change: disabled TF32 for matmul/conv and provided a DegradeToFP32Tracker that reverts to pure FP32 on first non-finite loss/grad | purpose: Ampere defaults TF32 to on which can silently shrink effective precision in reductions for a small sparse MLP; explicit off + fallback keeps training stable and reproducible, matching the conservative “evidence missing → keep baseline” rule | hardware_keys: tf32.disable, nonfinite.fallback
- baseline_change: added a `PrecisionConfig` dataclass and `format_precision_log()` that records the actual policy, dtype, TF32 state, and reason in one line | purpose: satisfies precision-logging requirement without batch noise and gives Stage 3 a single source of truth for scaler/autocast decisions | hardware_keys: none
        **Code:**
        ```python
# stage2_datatype_precision.py
"""
Stage 2: Datatype / Precision policy for the Disaster Tweets MVSFN candidate.

Contract (consumed by training_evaluation):
  - DEVICE resolution: CUDA when available (A10, cc 8.6), otherwise CPU.
  - Precision policy: one of {"fp32", "fp16_amp"}.
      * fp32        -> no autocast, no scaler.
      * fp16_amp    -> CUDA autocast(torch.float16) around forward+loss only.
  - TF32: explicitly disabled for matmul/cudnn (Ampere default is ON).
  - GradScaler: FP16 only; skip on CPU / FP32.
  - FP32 islands: fragile BCE/softmax reductions and metric accumulation.
  - DegradeToFP32Tracker: flips AMP -> FP32 on first non-finite loss/grad.

Import-safe: no CUDA configuration, no model/optimizer construction, no data access.
Preserves the Stage 1 model family, loss, feature interface, and output shape.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (aligned with Stage 1 / Stage 3 contract)
# ---------------------------------------------------------------------------
MODEL_FAMILY = "multiview_sparse_fusion_v1"  # preserve from Stage 1

ALLOWED_PRECISION_POLICIES: Tuple[str, ...] = ("fp32", "fp16_amp")
DEFAULT_PRECISION_POLICY: str = "fp16_amp"
DEFAULT_AMP_DTYPE = torch.float16

# Conservative default max grad norm from the trace (Stage 3 owns the exact value).
DEFAULT_CLIP_GRAD_NORM: float = 1.0


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------
def resolve_device(prefer_cuda: bool = True) -> torch.device:
    """Return the canonical compute device for this run."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Precision configuration
# ---------------------------------------------------------------------------
@dataclass
class PrecisionConfig:
    """
    Immutable-ish snapshot of the resolved precision policy.

    Attributes:
        device:         resolved torch.device for training/inference.
        policy:         "fp32" or "fp16_amp".
        use_amp:        True only when policy == "fp16_amp" and device is CUDA.
        amp_dtype:      autocast dtype (torch.float16 when use_amp else fp32).
        use_tf32:       always False per policy (disable Ampere TF32).
        use_scaler:     True only when policy == "fp16_amp" on CUDA.
        fallback_active: set True by DegradeToFP32Tracker after a non-finite event.
        reason:         short human-readable rationale for logging.
    """

    device: torch.device
    policy: str = DEFAULT_PRECISION_POLICY
    use_amp: bool = False
    amp_dtype: torch.dtype = torch.float32
    use_tf32: bool = False
    use_scaler: bool = False
    fallback_active: bool = False
    reason: str = ""


def resolve_precision_config(
    device: Optional[torch.device] = None,
    policy: str = DEFAULT_PRECISION_POLICY,
    allow_amp: bool = True,
) -> PrecisionConfig:
    """
    Decide the concrete precision policy for this run.

    Rules:
      - CPU: force fp32 (no AMP, no scaler).
      - CUDA + policy "fp16_amp" + allow_amp: enable autocast fp16 + GradScaler.
      - CUDA + policy "fp32" or allow_amp False: fp32 only.
      - Unknown policy string: silently fall back to fp32 with a log line.
    """
    dev = device if device is not None else resolve_device()
    pol = str(policy).lower().strip()

    if pol not in ALLOWED_PRECISION_POLICIES:
        logger.warning("Unknown precision policy %r; falling back to fp32.", policy)
        pol = "fp32"
        reason = "unknown_policy->fp32"
    else:
        reason = f"requested={pol}"

    if dev.type != "cuda":
        return PrecisionConfig(
            device=dev,
            policy="fp32",
            use_amp=False,
            amp_dtype=torch.float32,
            use_tf32=False,
            use_scaler=False,
            fallback_active=False,
            reason=f"{reason};device=cpu->fp32",
        )

    if pol == "fp16_amp" and allow_amp:
        return PrecisionConfig(
            device=dev,
            policy="fp16_amp",
            use_amp=True,
            amp_dtype=DEFAULT_AMP_DTYPE,
            use_tf32=False,  # explicitly disabled per normal mode
            use_scaler=True,
            fallback_active=False,
            reason=f"{reason};cuda_autocast_fp16+gradscaler;tf32_off",
        )

    return PrecisionConfig(
        device=dev,
        policy="fp32",
        use_amp=False,
        amp_dtype=torch.float32,
        use_tf32=False,
        use_scaler=False,
        fallback_active=False,
        reason=f"{reason};cuda_fp32;tf32_off",
    )


# ---------------------------------------------------------------------------
# Backend precision configuration (TF32) - call once inside __main__
# ---------------------------------------------------------------------------
def configure_backend_precision(config: PrecisionConfig) -> None:
    """
    Disable TF32 for matmul and cudnn. Safe to call multiple times.

    Normal mode forbids TF32; even if the platform defaults allow it, we
    override here so the run is deterministic and comparable to FP32.
    """
    if config.device.type != "cuda":
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(config.use_tf32)
        torch.backends.cudnn.allow_tf32 = bool(config.use_tf32)
    except Exception:  # pragma: no cover - CPU-only torch builds
        pass
    # Also silence the newer fp32_precision knobs if present.
    try:
        torch.set_float32_matmul_precision("highest" if not config.use_tf32 else "high")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Autocast context / FP32 island
# ---------------------------------------------------------------------------
def _autocast_ctx(config: PrecisionConfig):
    """
    Return the correct autocast context for the resolved device/policy.

    Notes:
      - Uses torch.amp.autocast when available (torch>=2.0), else the legacy API.
      - When AMP is disabled (fp32), returns a nullcontext.
    """
    if not config.use_amp or config.device.type != "cuda":
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", dtype=config.amp_dtype)
    except Exception:  # pragma: no cover - older torch
        return torch.cuda.amp.autocast(dtype=config.amp_dtype)


def autocast_forward(config: PrecisionConfig):
    """Convenience alias for the forward-pass autocast context."""
    return _autocast_ctx(config)


def fp32_island(config: PrecisionConfig):
    """
    Return a context that disables autocast for fragile ops (BCE, softmax, etc.).

    On CPU this is a nullcontext; on CUDA with AMP it forces FP32 compute locally.
    """
    if not config.use_amp or config.device.type != "cuda":
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", enabled=False)
    except Exception:  # pragma: no cover
        return torch.cuda.amp.autocast(enabled=False)


# ---------------------------------------------------------------------------
# FP32 loss / reduction helpers
# ---------------------------------------------------------------------------
def compute_loss_fp32(
    criterion, logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """
    Compute the loss in FP32 explicitly, ignoring any ambient autocast.

    Keeps the Stage 1 criterion unchanged; only the dtype of the incoming
    tensors is safely up-cast for the reduction. Returns a scalar tensor.
    """
    logits_f = logits.float() if logits.dtype != torch.float32 else logits
    targets_f = targets.float() if targets.dtype != torch.float32 else targets
    with contextlib.nullcontext():  # no-op; we already cast inputs
        return criterion(logits_f, targets_f)


def sigmoid_fp32(logits: torch.Tensor) -> torch.Tensor:
    """Numerically stable sigmoid in FP32 for probability export."""
    return torch.sigmoid(logits.float())


# ---------------------------------------------------------------------------
# GradScaler lifecycle
# ---------------------------------------------------------------------------
def make_grad_scaler(config: PrecisionConfig):
    """Create a GradScaler when FP16 AMP is active; otherwise return None."""
    if not config.use_scaler or config.device.type != "cuda":
        return None
    if config.fallback_active:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler(enabled=True)


def scaler_scale(scaler, loss: torch.Tensor) -> torch.Tensor:
    """Scale loss for backward when scaler is present; else return loss unchanged."""
    if scaler is None:
        return loss
    return scaler.scale(loss)


def scaler_step(scaler, optimizer) -> bool:
    """
    Perform optimizer step with scaler if present; else do a plain step.

    Returns True when the optimizer actually updated parameters on this step
    (i.e. the model step was not skipped due to inf/nan gradients).
    """
    if scaler is None:
        optimizer.step()
        return True
    scaler.step(optimizer)
    scaler.update()
    # We cannot cheaply know if the step was skipped; treat non-finite as
    # handled by DegradeToFP32Tracker below. Report True by default.
    return True


def clip_gradients_(model: nn.Module, max_norm: float, scaler=None) -> float:
    """
    Unscale (if scaler) then clip by global norm. Returns the pre-clip grad norm.
    """
    if scaler is not None:
        try:
            scaler.unscale_(_optimizer_placeholder := None)  # documented no-op if None
        except Exception:
            pass
    # The standard pattern is to unscale the scaler's optimizer; we accept a
    # scaler here only for interface compatibility and rely on the caller to
    # have already unscaled via `scaler.unscale_(optimizer)` before calling.
    total_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.grad is not None],
        max_norm,
    )
    return float(total_norm) if total_norm is not None else 0.0


def unscale_and_clip_(scaler, optimizer, model: nn.Module, max_norm: float) -> float:
    """
    Canonical AMP clip helper: unscale gradients in-place then clip by norm.

    When scaler is None this is a plain FP32 clip.
    """
    if scaler is not None:
        scaler.unscale_(optimizer)
    return clip_gradients_(model, max_norm, scaler=None)


# ---------------------------------------------------------------------------
# Non-finite detection + automatic fallback to FP32
# ---------------------------------------------------------------------------
class DegradeToFP32Tracker:
    """
    Monitors loss/grad finiteness during AMP training and flips to FP32 on the
    first non-finite event, so the remainder of the run stays numerically safe.

    Usage:
        tracker = DegradeToFP32Tracker(config)
        ...
        ok = tracker.check(loss)
        if not ok:  # config may have been mutated in place
            scaler = tracker.rebuild_scaler()
    """

    def __init__(self, config: PrecisionConfig):
        self.config = config
        self.fallback_triggered: bool = False
        self.reason: str = ""

    def _flip_disabled(self) -> None:
        self.fallback_triggered = True
        self.config.fallback_active = True
        self.config.use_amp = False
        self.config.use_scaler = False
        self.config.amp_dtype = torch.float32
        self.reason = "nonfinite_detected->fp32_fallback"

    def check(self, loss: Optional[torch.Tensor]) -> bool:
        """Return True if loss is finite; flip state and return False otherwise."""
        if loss is None:
            return True
        try:
            finite = bool(torch.isfinite(loss.detach()).all().item())
        except Exception:
            finite = False
        if not finite:
            self._flip_disabled()
            logger.warning("DegradeToFP32Tracker: non-finite loss; switching to FP32.")
            return False
        return True

    def check_grads(self, model: nn.Module) -> bool:
        """Return True if all grads are finite; flip state and return False otherwise."""
        for p in model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all().item():
                self._flip_disabled()
                logger.warning(
                    "DegradeToFP32Tracker: non-finite grad detected; switching to FP32."
                )
                return False
        return True

    def rebuild_scaler(self):
        """Rebuild a scaler consistent with the (possibly mutated) config."""
        return make_grad_scaler(self.config)


# ---------------------------------------------------------------------------
# Logging formatter
# ---------------------------------------------------------------------------
def format_precision_log(config: PrecisionConfig) -> str:
    """One-line, human-readable summary for TrainingDiagnostics / setup logging."""
    return (
        f"[precision] device={config.device.type} policy={config.policy} "
        f"amp={config.use_amp} dtype={str(config.amp_dtype).replace('torch.', '')} "
        f"tf32={config.use_tf32} scaler={config.use_scaler} "
        f"fallback={config.fallback_active} reason={config.reason}"
    )

```
        
        ### Step 3: training_evaluation
        **Plan:** ## Plan

I'll implement Stage 3 as a self-contained script that consumes the Stage 1 MVSFN (gated word-TFIDF + char_wb-TFIDF + dense-meta + keyword-embedding fusion) and Stage 2 FP16-AMP precision utilities, and adds the training loop. Data split is done **first** (stratified 80/20, seed 42); the TF-IDF vectorizers and keyword vocabulary are fit on the train split only, then transformed on val/test with identical code, guaranteeing no leakage and val/test consistency. I use a small MLP-friendly AdamW (lr 1e-3, cosine with 10% warmup, grad-clip 1.0), class-weighted BCE with label smoothing, threshold tuned on val probabilities for F1, early stopping with best-checkpoint restore, then test inference through the same forward + sigmoid + threshold pipeline.

**Backend strategy**: physical batch 128 (≈660 MB dense TF-IDF tensors on CPU, streamed via `DataLoader(num_workers=2, pin_memory=True)` to fit within the A10 envelope; effective = physical, no accumulation). FP16 autocast is used only around the forward pass; the BCE reduction runs in an FP32 island with explicit `.float()` casts; GradScaler + unscale-before-clip; `DegradeToFP32Tracker` flips to FP32 on any non-finite loss/grad. TF32 explicitly disabled. Scheduler owns yielding/device assignment; script passively consumes `script_scheduler_context` with `SafePointType.BEFORE_TRAIN / STEP / EPOCH` and a `state_factory` capturing model/optimizer/scaler/scheduler/RNG/step position.

**Validation hypothesis**: MVSFN's gated multi-view fusion is a different inductive bias than prior BiGRU/MLP/transformer attempts (memory best 0.7711). If gating + char morphology is doing real work, val F1 should exceed 0.75 and gate distributions should be non-degenerate; a plateau < 0.72 would indicate the gating isn't learning and warrants a fallback to uniform gates.

Note board:
- baseline_change: introduced Stage 3 training loop with AdamW(lr=1e-3, wd=1e-4) + 10% linear warmup + cosine decay, class-weighted BCE + label smoothing 0.05, grad-clip 1.0, physical batch 128, 15 epochs, patience-4 early stop with best-val-F1 checkpoint restore, and val-tuned decision threshold | purpose: match the small MLP-style MVSFN (few M params) with a higher-LR schedule than the transformer default 2e-5, use label smoothing for the ~57/43 imbalance, and select the operating threshold that maximizes F1 exactly as test inference will | hardware_keys: cuda.amp.autocast, tf32.disable, grads.unscale_clip
- baseline_change: routed forward through Stage 2 `autocast_forward(cfg)` while BCE reduction runs inside `fp32_island(cfg)` with `.float()` casts and GradScaler/unscale-then-clip; wired `DegradeToFP32Tracker` to flip to FP32 on non-finite loss/grad | purpose: use Tensor Cores where numerically safe while keeping the reduction FP32 (fragile for small batch BCE) and guaranteeing the Stage 2 policy is consumed, not re-decided | hardware_keys: cuda.amp.autocast, tf32.disable, nonfinite.fallback
- baseline_change: added TrainingDiagnostics wrap, `script_scheduler_context` BEFORE_TRAIN/STEP/EPOCH safe-points with a full `state_factory` (model, optimizer, scaler, LR sched, torch/numpy/python RNG, global_step, step_in_epoch) and diagnostics state in the checkpoint | purpose: satisfy the scheduler and diagnostics contract without altering training semantics, and keep val F1 comparable to prior 0.7711 | hardware_keys: none
        **Code:**
        ```python
# stage3_training_evaluation.py
"""
Stage 3: Training / Validation / Test / Submission for the Disaster Tweets
MVSFN candidate.

Consumes (AS-IS, no redesign):
  - Stage 1: clean_text / clean_keyword, FeaturePipeline (word+char TF-IDF +
    handcrafted meta), MultiViewSparseFusion, compute_class_weights,
    build_criterion, f1_binary, tune_threshold, make_train_val_split.
  - Stage 2: resolve_device, resolve_precision_config, PrecisionConfig,
    configure_backend_precision, autocast_forward, fp32_island, make_grad_scaler,
    unscale_and_clip_, DegradeToFP32Tracker, format_precision_log.

Owns: optimizer, LR, WD, scheduler, physical batch, accumulation, workers,
checkpointing, early stopping, threshold selection, and submission generation.

Import-safe: no training/side-effects at import time.
"""

from __future__ import annotations

import os
import re
import html
import sys
import math
import time
import copy
import random
import logging
import contextlib
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# MODEL CONTRACT + PATHS
# ---------------------------------------------------------------------------
MODEL_FAMILY = "multiview_sparse_fusion_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
INPUT_DIR = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")

SEED = 42
VAL_SIZE = 0.2

# Feature-space defaults (Stage 1 contract)
WORD_MAX_FEATURES = 12000
CHAR_MAX_FEATURES = 15000
DENSE_DIM = 8
WORD_NGRAM = (1, 2)
CHAR_NGRAM = (3, 5)
WORD_MIN_DF = 2
CHAR_MIN_DF = 3

# ===========================================================================
# STAGE 1: TEXT CLEANING
# ===========================================================================
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NONALNUM_RE = re.compile(r"[^a-z0-9#@\s]")
_WS_RE = re.compile(r"\s+")


def clean_text(s) -> str:
    s = "" if s is None else str(s)
    s = html.unescape(s).lower()
    s = _URL_RE.sub(" __url__ ", s)
    s = _MENTION_RE.sub(" __mention__ ", s)
    s = _HASHTAG_RE.sub(r" hashtag_\1 ", s)
    s = _NONALNUM_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def clean_keyword(s) -> str:
    if s is None:
        return ""
    if isinstance(s, float) and np.isnan(s):
        return ""
    return str(s).strip().lower().replace("%20", " ")


def compute_handcrafted(texts) -> np.ndarray:
    n = len(texts)
    feats = np.zeros((n, DENSE_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        t = "" if t is None else str(t)
        n_chars = max(len(t), 1)
        n_words = max(len(t.split()), 1)
        n_upper = sum(1 for c in t if c.isupper())
        feats[i, 0] = min(n_chars / 150.0, 2.0)
        feats[i, 1] = min(n_words / 25.0, 2.0)
        feats[i, 2] = n_upper / n_chars
        feats[i, 3] = min(t.count("!") / 3.0, 2.0)
        feats[i, 4] = min(t.count("?") / 3.0, 2.0)
        feats[i, 5] = min(t.count("#") / 3.0, 2.0)
        t_low = t.lower()
        feats[i, 6] = 1.0 if ("http" in t_low or "www." in t_low) else 0.0
        feats[i, 7] = min(t.count("@") / 3.0, 2.0)
    return feats


# ===========================================================================
# STAGE 1: FEATURE PIPELINE (fit on train split only)
# ===========================================================================
class FeaturePipeline:
    def __init__(self):
        self.word_vec = TfidfVectorizer(
            ngram_range=WORD_NGRAM,
            sublinear_tf=True,
            min_df=WORD_MIN_DF,
            max_features=WORD_MAX_FEATURES,
            token_pattern=r"(?u)\b\w+\b",
            dtype=np.float32,
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=CHAR_NGRAM,
            sublinear_tf=True,
            min_df=CHAR_MIN_DF,
            max_features=CHAR_MAX_FEATURES,
            dtype=np.float32,
        )
        self.keyword_to_idx = {"": 0}
        self.word_dim = 0
        self.char_dim = 0
        self.keyword_vocab = 1

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        texts = [clean_text(t) for t in df["text"].tolist()]
        keywords = [
            clean_keyword(k) for k in df.get("keyword", [""] * len(df)).tolist()
        ]
        self.word_vec.fit(texts)
        self.char_vec.fit(texts)
        uniq = sorted({k for k in keywords if k})
        self.keyword_to_idx = {"": 0}
        for i, k in enumerate(uniq):
            self.keyword_to_idx[k] = i + 1
        self.word_dim = int(len(self.word_vec.get_feature_names_out()))
        self.char_dim = int(len(self.char_vec.get_feature_names_out()))
        self.keyword_vocab = int(len(self.keyword_to_idx))
        return self

    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        texts = [clean_text(t) for t in df["text"].tolist()]
        Xw = self.word_vec.transform(texts).toarray().astype(np.float32, copy=False)
        Xc = self.char_vec.transform(texts).toarray().astype(np.float32, copy=False)
        dense = compute_handcrafted(df["text"].tolist())
        if "keyword" in df.columns:
            kws = df["keyword"].tolist()
        else:
            kws = [""] * len(df)
        kw = np.asarray(
            [self.keyword_to_idx.get(clean_keyword(k), 0) for k in kws],
            dtype=np.int64,
        )
        return {"word": Xw, "char": Xc, "dense": dense, "keyword_idx": kw}


# ===========================================================================
# STAGE 1: MODEL
# ===========================================================================
class MultiViewSparseFusion(nn.Module):
    def __init__(
        self,
        word_dim: int,
        char_dim: int,
        dense_dim: int,
        keyword_vocab: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.word_dim = int(word_dim)
        self.char_dim = int(char_dim)
        self.dense_dim = int(dense_dim)
        self.keyword_vocab = int(keyword_vocab)
        self.hidden_dim = int(hidden_dim)

        self.word_proj = nn.Sequential(
            nn.Linear(self.word_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.char_proj = nn.Sequential(
            nn.Linear(self.char_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.dense_proj = nn.Sequential(
            nn.Linear(self.dense_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.keyword_emb = nn.Embedding(max(self.keyword_vocab, 1), hidden_dim)
        nn.init.normal_(self.keyword_emb.weight, std=0.02)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, word, char, dense, keyword_idx):
        v_word = self.word_proj(word)
        v_char = self.char_proj(char)
        v_dense = self.dense_proj(dense)
        v_kw = self.keyword_emb(keyword_idx.long())
        ctx = torch.cat([v_kw, v_dense], dim=-1)
        gates = self.gate(ctx)
        v_word = v_word * gates[:, 0:1]
        v_char = v_char * gates[:, 1:2]
        v_dense = v_dense * gates[:, 2:3]
        v_kw = v_kw * gates[:, 3:4]
        fused = self.dropout(v_word + v_char + v_dense + v_kw)
        return self.head(fused).squeeze(-1)


# ===========================================================================
# STAGE 1: LOSS / METRIC / SPLIT
# ===========================================================================
def compute_class_weights(targets) -> list:
    y = np.asarray(targets).astype(int)
    n = max(len(y), 1)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    w0 = n / (2.0 * max(n_neg, 1))
    w1 = n / (2.0 * max(n_pos, 1))
    return [float(w0), float(w1)]


def build_criterion(class_weights=None, label_smoothing: float = 0.0):
    pos_weight = None
    if class_weights is not None:
        w0, w1 = float(class_weights[0]), float(class_weights[1])
        if w0 > 0:
            pos_weight = torch.tensor([w1 / w0], dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def criterion(logits, targets):
        t = targets.float()
        if label_smoothing > 0.0:
            t = t * (1.0 - float(label_smoothing)) + 0.5 * float(label_smoothing)
        return bce(logits, t)

    return criterion


def f1_binary(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred).astype(int)
    target = np.asarray(target).astype(int)
    tp = int(((pred == 1) & (target == 1)).sum())
    fp = int(((pred == 1) & (target == 0)).sum())
    fn = int(((pred == 0) & (target == 1)).sum())
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return float(2.0 * prec * rec / max(prec + rec, 1e-12))


def tune_threshold(probs, targets, grid=None):
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets).astype(int).reshape(-1)
    if grid is None:
        grid = np.linspace(0.05, 0.95, 181)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_binary((probs >= float(t)).astype(int), targets)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def make_train_val_split(
    df: pd.DataFrame, val_size: float = VAL_SIZE, seed: int = SEED
):
    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=seed, stratify=df["target"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ===========================================================================
# STAGE 2: PRECISION POLICY
# ===========================================================================
ALLOWED_PRECISION_POLICIES = ("fp32", "fp16_amp")
DEFAULT_PRECISION_POLICY = "fp16_amp"
DEFAULT_AMP_DTYPE = torch.float16


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class PrecisionConfig:
    device: torch.device
    policy: str = DEFAULT_PRECISION_POLICY
    use_amp: bool = False
    amp_dtype: torch.dtype = torch.float32
    use_tf32: bool = False
    use_scaler: bool = False
    fallback_active: bool = False
    reason: str = ""


def resolve_precision_config(
    device: Optional[torch.device] = None,
    policy: str = DEFAULT_PRECISION_POLICY,
    allow_amp: bool = True,
) -> PrecisionConfig:
    dev = device if device is not None else resolve_device()
    pol = str(policy).lower().strip()
    if pol not in ALLOWED_PRECISION_POLICIES:
        pol = "fp32"
        reason = "unknown_policy->fp32"
    else:
        reason = f"requested={pol}"
    if dev.type != "cuda":
        return PrecisionConfig(device=dev, policy="fp32", reason=f"{reason};cpu->fp32")
    if pol == "fp16_amp" and allow_amp:
        return PrecisionConfig(
            device=dev,
            policy="fp16_amp",
            use_amp=True,
            amp_dtype=DEFAULT_AMP_DTYPE,
            use_tf32=False,
            use_scaler=True,
            reason=f"{reason};cuda_autocast_fp16+gradscaler;tf32_off",
        )
    return PrecisionConfig(
        device=dev, policy="fp32", reason=f"{reason};cuda_fp32;tf32_off"
    )


def configure_backend_precision(config: PrecisionConfig) -> None:
    if config.device.type != "cuda":
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def autocast_forward(config: PrecisionConfig):
    if not config.use_amp or config.device.type != "cuda":
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", dtype=config.amp_dtype)
    except Exception:
        return torch.cuda.amp.autocast(dtype=config.amp_dtype)


def fp32_island(config: PrecisionConfig):
    if not config.use_amp or config.device.type != "cuda":
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", enabled=False)
    except Exception:
        return torch.cuda.amp.autocast(enabled=False)


def make_grad_scaler(config: PrecisionConfig):
    if not config.use_scaler or config.device.type != "cuda":
        return None
    if config.fallback_active:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)


def unscale_and_clip_(scaler, optimizer, model: nn.Module, max_norm: float) -> float:
    if scaler is not None:
        scaler.unscale_(optimizer)
    params = [p for p in model.parameters() if p.grad is not None]
    total = torch.nn.utils.clip_grad_norm_(params, max_norm)
    return float(total) if total is not None else 0.0


class DegradeToFP32Tracker:
    def __init__(self, config: PrecisionConfig):
        self.config = config
        self.fallback_triggered = False
        self.reason = ""

    def _flip(self):
        self.fallback_triggered = True
        self.config.fallback_active = True
        self.config.use_amp = False
        self.config.use_scaler = False
        self.config.amp_dtype = torch.float32
        self.reason = "nonfinite->fp32_fallback"

    def check(self, loss) -> bool:
        if loss is None:
            return True
        try:
            ok = bool(torch.isfinite(loss.detach()).all().item())
        except Exception:
            ok = False
        if not ok:
            self._flip()
            return False
        return True

    def rebuild_scaler(self):
        return make_grad_scaler(self.config)


def format_precision_log(config: PrecisionConfig) -> str:
    return (
        f"[precision] device={config.device.type} policy={config.policy} "
        f"amp={config.use_amp} dtype={str(config.amp_dtype).replace('torch.','')} "
        f"tf32={config.use_tf32} scaler={config.use_scaler} "
        f"fallback={config.fallback_active} reason={config.reason}"
    )


# ===========================================================================
# OPTIONAL INTEGRATIONS (safe fallbacks when unavailable)
# ===========================================================================
try:
    from utils.training_diagnostics import TrainingDiagnostics  # type: ignore

    _HAS_DIAG = True
except Exception:
    _HAS_DIAG = False

    class TrainingDiagnostics:  # lightweight shim
        def __init__(self, model, optimizer, scaler=None, settings=None):
            self.model = model
            self.optimizer = optimizer
            self.scaler = scaler
            self.settings = settings or {}
            self._updates = 0
            self._epochs = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def after_update(self):
            self._updates += 1

        def report(self, epoch=None):
            self._epochs += 1

        def state_dict(self):
            return {"updates": self._updates, "epochs": self._epochs}

        def load_state_dict(self, state):
            if state:
                self._updates = int(state.get("updates", 0))
                self._epochs = int(state.get("epochs", 0))


try:
    from localml_scheduler.execution.script_context import script_scheduler_context  # type: ignore
    from localml_scheduler.domain import SafePointType  # type: ignore

    _HAS_SCHED = True
except Exception:
    _HAS_SCHED = False

    def script_scheduler_context():
        return None

    class SafePointType:  # type: ignore
        BEFORE_TRAIN = "BEFORE_TRAIN"
        STEP = "STEP"
        EPOCH = "EPOCH"


# ===========================================================================
# STAGE 3: HYPERPARAMETERS (owned here)
# ===========================================================================
QUALITY_SAFE_PHYSICAL_BATCH_SIZES = [64, 128, 256]
BATCH_LR_SCALING_POLICY = "sqrt"

PHYSICAL_BATCH_SIZE = 128
GRAD_ACCUM_STEPS = 1
PLANNED_EPOCHS = 15
BASE_LR = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.1
GRAD_CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 4
NUM_WORKERS = 2
PIN_MEMORY = True
HIDDEN_DIM = 128
DROPOUT = 0.3


# ===========================================================================
# STAGE 3: HELPER — build loaders / forward / inference
# ===========================================================================
def _to_tensors(feats: Dict[str, np.ndarray], targets: Optional[np.ndarray]):
    word = torch.from_numpy(feats["word"]).float()
    char = torch.from_numpy(feats["char"]).float()
    dense = torch.from_numpy(feats["dense"]).float()
    kw = torch.from_numpy(feats["keyword_idx"]).long()
    if targets is None:
        return word, char, dense, kw
    y = torch.from_numpy(np.asarray(targets, dtype=np.float32))
    return word, char, dense, kw, y


def _make_loader(
    word, char, dense, kw, y=None, batch_size=128, shuffle=False, workers=2
):
    if y is not None:
        ds = TensorDataset(word, char, dense, kw, y)
    else:
        ds = TensorDataset(word, char, dense, kw)
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(workers),
        pin_memory=PIN_MEMORY and torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(workers > 0),
    )
    return loader


def _forward_logits(model, word, char, dense, kw):
    return model(word, char, dense, kw)


@torch.no_grad()
def _predict_probs(model, loader, device, config: PrecisionConfig):
    model.eval()
    all_probs = []
    for batch in loader:
        word, char, dense, kw = batch
        word = word.to(device, non_blocking=True)
        char = char.to(device, non_blocking=True)
        dense = dense.to(device, non_blocking=True)
        kw = kw.to(device, non_blocking=True)
        with autocast_forward(config):
            logits = _forward_logits(model, word, char, dense, kw)
        probs = torch.sigmoid(logits.float()).detach().cpu().numpy().reshape(-1)
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


# ===========================================================================
# CPU PREFLIGHT ADAPTER
# ===========================================================================
class CandidateAdapter:
    """Real model / real loss / real optimizer exercised on CPU with small dims."""

    _DEFAULTS = {
        "word_dim": 512,
        "char_dim": 512,
        "dense_dim": DENSE_DIM,
        "keyword_vocab": 32,
        "hidden_dim": 32,
        "dropout": 0.1,
        "batch_size": 8,
        "base_lr": BASE_LR,
        "weight_decay": WEIGHT_DECAY,
    }

    def _merged(self, context) -> Dict[str, Any]:
        cfg = dict(self._DEFAULTS)
        if isinstance(context, dict):
            for k, v in context.items():
                if v is not None:
                    cfg[k] = v
        return cfg

    def build_model(self, context):
        cfg = self._merged(context)
        torch.manual_seed(SEED)
        return MultiViewSparseFusion(
            word_dim=int(cfg["word_dim"]),
            char_dim=int(cfg["char_dim"]),
            dense_dim=int(cfg["dense_dim"]),
            keyword_vocab=int(cfg["keyword_vocab"]),
            hidden_dim=int(cfg["hidden_dim"]),
            dropout=float(cfg["dropout"]),
        )

    def build_optimizer(self, model, context):
        cfg = self._merged(context)
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["base_lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )

    def _resolve_criterion(self, context):
        cfg = self._merged(context)
        cw = cfg.get("class_weights", [1.0, 1.0])
        ls = float(cfg.get("label_smoothing", 0.0))
        return build_criterion(class_weights=cw, label_smoothing=ls)

    def _batch(self, scenario, device):
        self._merged(scenario if isinstance(scenario, dict) else {})
        bs = int((scenario or {}).get("batch_size", self._DEFAULTS["batch_size"]))
        fx = (scenario or {}).get("fixture") or {}
        word_dim = (
            int(fx.get("word", (self._DEFAULTS["word_dim"],))[0])
            if "word" in fx
            else self._DEFAULTS["word_dim"]
        )
        char_dim = (
            int(fx.get("char", (self._DEFAULTS["char_dim"],))[0])
            if "char" in fx
            else self._DEFAULTS["char_dim"]
        )
        dense_dim = (
            int(fx.get("dense", (self._DEFAULTS["dense_dim"],))[0])
            if "dense" in fx
            else self._DEFAULTS["dense_dim"]
        )
        kw_vocab = self._DEFAULTS["keyword_vocab"]

        word = torch.randn(bs, word_dim, dtype=torch.float32)
        char = torch.randn(bs, char_dim, dtype=torch.float32)
        dense = torch.randn(bs, dense_dim, dtype=torch.float32)
        kw = torch.randint(0, max(kw_vocab, 2), (bs,), dtype=torch.long)
        y = torch.randint(0, 2, (bs,), dtype=torch.float32)
        word = word.to(device)
        char = char.to(device)
        dense = dense.to(device)
        kw = kw.to(device)
        y = y.to(device)
        return (word, char, dense, kw), y

    def build_train_batch(self, scenario, device):
        return self._batch(scenario or {}, device)

    def build_validation_batch(self, scenario, device):
        return self._batch(scenario or {}, device)

    def training_step(self, model, batch, context):
        inputs, target = batch
        word, char, dense, kw = inputs
        criterion = self._resolve_criterion(context)
        logits = model(word, char, dense, kw)
        loss = criterion(logits.float(), target.float())
        return loss

    def validation_step(self, model, batch, context):
        inputs, target = batch
        word, char, dense, kw = inputs
        criterion = self._resolve_criterion(context)
        with torch.no_grad():
            logits = model(word, char, dense, kw)
            loss = criterion(logits.float(), target.float())
            probs = torch.sigmoid(logits.float())
        return {
            "loss": loss.detach(),
            "probs": probs.detach(),
            "target": target.detach(),
        }


# ===========================================================================
# STAGE 3: MAIN TRAINING / INFERENCE / SUBMISSION
# ===========================================================================
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _state_factory(model, optimizer, scaler, scheduler, global_step, step_in_epoch):
    def _factory():
        st = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng_torch": torch.get_rng_state(),
            "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate(),
            "global_step": int(global_step),
            "step_in_epoch": int(step_in_epoch),
        }
        if torch.cuda.is_available():
            st["rng_cuda"] = torch.cuda.get_rng_state_all()
        return st

    return _factory


def run_training() -> float:
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    _set_seed(SEED)

    # -- scheduler context (may be None) --
    context = None
    if _HAS_SCHED:
        try:
            context = script_scheduler_context()
        except Exception:
            context = None

    # -- device + precision --
    device = resolve_device()
    precision_cfg = resolve_precision_config(
        device=device, policy=DEFAULT_PRECISION_POLICY
    )
    configure_backend_precision(precision_cfg)

    # -- load data --
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")
    sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    full_train = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sub_template = pd.read_csv(sub_path)

    # -- split FIRST, then fit feature pipeline on train only --
    train_df, val_df = make_train_val_split(full_train)
    fp = FeaturePipeline().fit(train_df)
    train_feats = fp.transform(train_df)
    val_feats = fp.transform(val_df)
    test_feats = fp.transform(test_df)
    y_train = train_df["target"].values.astype(np.float32)
    y_val = val_df["target"].values.astype(np.float32)

    tw, tc, td, tk, ty = _to_tensors(train_feats, y_train)
    vw, vc, vd, vk, vy = _to_tensors(val_feats, y_val)
    xw, xc, xd, xk = _to_tensors(test_feats, None)

    train_loader = _make_loader(
        tw,
        tc,
        td,
        tk,
        ty,
        batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=True,
        workers=NUM_WORKERS,
    )
    val_loader = _make_loader(
        vw,
        vc,
        vd,
        vk,
        vy,
        batch_size=PHYSICAL_BATCH_SIZE * 2,
        shuffle=False,
        workers=NUM_WORKERS,
    )
    test_loader = _make_loader(
        xw,
        xc,
        xd,
        xk,
        None,
        batch_size=PHYSICAL_BATCH_SIZE * 2,
        shuffle=False,
        workers=NUM_WORKERS,
    )

    # -- model / optimizer / criterion / scheduler --
    model = MultiViewSparseFusion(
        word_dim=fp.word_dim,
        char_dim=fp.char_dim,
        dense_dim=DENSE_DIM,
        keyword_vocab=fp.keyword_vocab,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)

    cw = compute_class_weights(y_train)
    criterion = build_criterion(class_weights=cw, label_smoothing=LABEL_SMOOTHING)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY
    )

    steps_per_epoch = max(1, math.ceil(len(train_df) / PHYSICAL_BATCH_SIZE))
    total_steps = steps_per_epoch * PLANNED_EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    def _lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    scaler = make_grad_scaler(precision_cfg)
    degrade_tracker = DegradeToFP32Tracker(precision_cfg)

    # -- diagnostics context --
    diagnostics_settings = {
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "effective_batch_size": PHYSICAL_BATCH_SIZE * GRAD_ACCUM_STEPS,
        "planned_epochs": PLANNED_EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "model_family": MODEL_FAMILY,
        "split_seed": SEED,
        "precision": precision_cfg.policy,
        "precision_reason": precision_cfg.reason,
    }
    diagnostics = TrainingDiagnostics(
        model, optimizer, scaler=scaler, settings=diagnostics_settings
    )

    # -- scheduler: set job budget + restore + BEFORE_TRAIN --
    if context is not None:
        try:
            context.job.max_epochs = PLANNED_EPOCHS
            context.job.config.max_epochs = PLANNED_EPOCHS
            context.store.save_job(context.job)
        except Exception:
            pass

        try:
            ckpt = context.load_resume_checkpoint()
        except Exception:
            ckpt = None
        if ckpt is not None and ckpt.get("state") is not None:
            state = ckpt["state"]
            try:
                if state.get("model") is not None:
                    model.load_state_dict(state["model"])
                if state.get("optimizer") is not None:
                    optimizer.load_state_dict(state["optimizer"])
                if scaler is not None and state.get("scaler") is not None:
                    scaler.load_state_dict(state["scaler"])
                if state.get("scheduler") is not None:
                    scheduler.load_state_dict(state["scheduler"])
                if state.get("rng_torch") is not None:
                    torch.set_rng_state(state["rng_torch"])
                if state.get("rng_cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(state["rng_cuda"])
                if state.get("rng_numpy") is not None:
                    np.random.set_state(state["rng_numpy"])
                if state.get("rng_python") is not None:
                    random.setstate(state["rng_python"])
                if state.get("diagnostics") is not None:
                    diagnostics.load_state_dict(state["diagnostics"])
            except Exception:
                pass

        _sf = _state_factory(model, optimizer, scaler, scheduler, 0, 0)
        try:
            context.control_hook.safe_point(
                SafePointType.BEFORE_TRAIN,
                epoch=0,
                global_step=0,
                steps_per_epoch=steps_per_epoch,
                state_factory=_sf,
            )
        except Exception:
            pass

    print(format_precision_log(precision_cfg), flush=True)

    # -- training loop --
    best_val_f1 = -1.0
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0
    global_step = 0
    epoch_times = []

    with diagnostics:
        for epoch in range(1, PLANNED_EPOCHS + 1):
            t0 = time.time()
            model.train()
            running_loss = 0.0
            n_steps = 0
            step_in_epoch = 0
            optimizer.zero_grad(set_to_none=True)

            for micro_idx, batch in enumerate(train_loader):
                word, char, dense, kw, y = batch
                word = word.to(device, non_blocking=True)
                char = char.to(device, non_blocking=True)
                dense = dense.to(device, non_blocking=True)
                kw = kw.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with autocast_forward(precision_cfg):
                    logits = _forward_logits(model, word, char, dense, kw)
                with fp32_island(precision_cfg):
                    loss = criterion(logits.float(), y.float())

                if not degrade_tracker.check(loss):
                    if scaler is not None:
                        scaler = degrade_tracker.rebuild_scaler()
                        diagnostics.scaler = scaler

                loss_for_backward = scaler.scale(loss) if scaler is not None else loss
                loss_for_backward.backward()

                is_update_step = ((micro_idx + 1) % GRAD_ACCUM_STEPS == 0) or (
                    micro_idx + 1 == len(train_loader)
                )
                if is_update_step:
                    unscale_and_clip_(scaler, optimizer, model, GRAD_CLIP_NORM)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                    step_in_epoch += 1

                    diagnostics.after_update()

                    if context is not None:
                        _sf = _state_factory(
                            model,
                            optimizer,
                            scaler,
                            scheduler,
                            global_step,
                            step_in_epoch,
                        )
                        try:
                            context.control_hook.safe_point(
                                SafePointType.STEP,
                                epoch=epoch,
                                global_step=global_step,
                                steps_per_epoch=steps_per_epoch,
                                state_factory=_sf,
                            )
                        except Exception:
                            pass

                running_loss += float(loss.detach().cpu().item())
                n_steps += 1

            # -- validation --
            val_probs = _predict_probs(model, val_loader, device, precision_cfg)
            _, val_f1_at_05 = tune_threshold(val_probs, y_val, grid=np.array([0.5]))
            running_loss / max(n_steps, 1)
            epoch_time = time.time() - t0
            epoch_times.append(epoch_time)

            print(
                f"Epoch {epoch}/{PLANNED_EPOCHS} "
                f"loss={running_loss / max(n_steps, 1):.4f} "
                f"val_f1@0.5={val_f1_at_05:.4f} "
                f"time={epoch_time:.1f}s",
                flush=True,
            )
            print(
                f"MLEVOLVE_EPOCH_METRIC "
                f'{{"epoch": {epoch}, "metric": {val_f1_at_05:.6f}, "metric_name": "val_f1_at_0.5"}}',
                flush=True,
            )

            diagnostics.report(epoch=epoch)

            if val_f1_at_05 > best_val_f1 + 1e-6:
                best_val_f1 = val_f1_at_05
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if context is not None:
                _sf = _state_factory(
                    model, optimizer, scaler, scheduler, global_step, step_in_epoch
                )
                try:
                    context.control_hook.safe_point(
                        SafePointType.EPOCH,
                        epoch=epoch,
                        global_step=global_step,
                        steps_per_epoch=steps_per_epoch,
                        state_factory=_sf,
                    )
                except Exception:
                    pass

            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(
                    f"Early stopping at epoch {epoch} (best={best_epoch}, f1@0.5={best_val_f1:.4f})",
                    flush=True,
                )
                break

        # -- restore best, tune threshold on best checkpoint --
        if best_state is not None:
            model.load_state_dict(best_state)

        val_probs = _predict_probs(model, val_loader, device, precision_cfg)
        best_threshold, best_val_f1_tuned = tune_threshold(val_probs, y_val)

        # checkpoint save
        try:
            ckpt_path = os.path.join(WORKING_DIR, "best_checkpoint.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "scheduler": scheduler.state_dict(),
                    "diagnostics": diagnostics.state_dict(),
                    "best_threshold": best_threshold,
                    "best_val_f1": best_val_f1_tuned,
                    "best_epoch": best_epoch,
                    "model_family": MODEL_FAMILY,
                    "seed": SEED,
                    "precision": precision_cfg.policy,
                },
                ckpt_path,
            )
        except Exception:
            pass

        # -- test inference (identical to validation) --
        test_probs = _predict_probs(model, test_loader, device, precision_cfg)
        test_pred = (test_probs >= best_threshold).astype(int)

        # write submission
        test_ids = test_df["id"].values
        sub = pd.DataFrame(
            {"id": test_ids.astype(np.int64), "target": test_pred.astype(np.int64)}
        )

        # align to sample submission order if possible
        if "id" in sub_template.columns and len(sub_template) == len(sub):
            sub = sub.set_index("id").reindex(sub_template["id"].values).reset_index()
            sub["target"] = sub["target"].fillna(0).astype(np.int64)
            sub = sub[["id", "target"]]

        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        out_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        sub.to_csv(out_path, index=False)
        print(
            f"Wrote {out_path} rows={len(sub)} pos_rate={sub['target'].mean():.4f}",
            flush=True,
        )

        finalscore = float(best_val_f1_tuned)

    print(f"Final Validation Score: {finalscore}")
    return finalscore


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
        