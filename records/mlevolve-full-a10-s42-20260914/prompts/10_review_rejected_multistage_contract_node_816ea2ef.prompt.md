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
 Summary: 3 total attempts - 2 pending execution, 1 executed, 0 successful (all failed or buggy)

-------------------------------
Attempt #1:
Design: I merged the three stages by keeping Stage 1's data/model/loss/adapter code as the foundation, layering Stage 2's import-safe precision policy (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `sigmoid_fp32`, `make_grad_scaler`, `to_fp32_policy`, `configure_tf32`, `fp32_island`) on top, and consuming both inside Stage 3's `run_training` (AdamW + cosine warmup, physical batch 32 × accum 2, val-F1 early stopping with best-state restore, scheduler safe-points and `TrainingDiagnostics`). I deduplicated imports/constants, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module-level constants referenced by Stage 3, preserved the CPU-safe `CandidateAdapter` contract, kept all side effects under `if __name__ == '__main__':`, and left the final `Final Validation Score:` print after the diagnostics context.
Results: The implementation of this design has bugs.
Insight: Using a different approach may not result in the same bugs as the above approach.

-------------------------------
Attempt #2:
Design: I merged the three stages by keeping Stage 1's data cleaning, TF-IDF feature builders, `SparseMLP`, `build_loss`, `tune_threshold` and split helpers as the foundation; layering Stage 2's import-safe precision utilities (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `degrade_to_fp32`, `configure_tf32`, `format_precision_log`) on top; then consuming both inside Stage 3's `run_training` with scheduler safe-points, `TrainingDiagnostics`, linear warmup+cosine AdamW, best-val-F1 restore, and test inference. I resolved the duplicate `CandidateAdapter` by keeping Stage 3's self-contained CPU-safe version (it uses the Stage 1 model/loss/features through the Stage 2 precision path), removed duplicate imports/constants, kept all side effects under `if __name__ == "__main__":`, and preserved the exact `MODEL_FAMILY`, `WORKING_DIR`, `SUBMISSION_DIR` contract.
Status: Code generated, execution pending (will run in parallel with other drafts).

-------------------------------
Attempt #3:
Design: I merged the three stages by keeping Stage 1's data/model/loss/adapter code as the foundation, layering Stage 2's `PrecisionRuntime` policy helpers on top, and consuming both inside Stage 3's `run_training` (with diagnostics + scheduler safe points). I resolved naming conflicts, deduplicated imports, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module-level constants that Stage 3 references, and moved every side effect under `if __name__ == '__main__':` while preserving the CPU-safe `CandidateAdapter` contract.
Status: Code generated, execution pending (will run in parallel with other drafts).


# Hardware-Aware Stage 1 Candidate Construction Context
# Hardware-Aware Model Design Brief
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2
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
- Workload: transformer_training



# Pipeline Decision Contract
Required hardware-aware step order: model-design -> datatype/quantization -> training (model_design -> datatype_precision -> training_evaluation). Use the trace as the source of truth, apply hardware evidence only when compatible, and use recorded fallbacks when evidence is missing.

# Pipeline Decision Trace
```json
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"{'criterion': 'BCEWithLogitsLoss', 'pos_weight': None, 'notes': 'Compute loss in FP32 island for...","output_interface":"{'raw_output': 'logits', 'probability': 'sigmoid(logits)', 'prediction_threshold': 0.5, 'threshol...","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"{'forward_autocast': {'enabled': True, 'backend': 'torch.amp.autocast', 'device_type': 'cuda', 'd...","fallback_policy":"fp32","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"{'name': 'AdamW', 'lr': 2e-05, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-08, 'notes'...","scheduler":"{'name': 'linear_with_warmup', 'warmup_ratio': 0.1, 'notes': 'Linear decay to zero after warmup.'}","batch_size_policy":"fixed","dataloader_policy":"safe defaults; enable workers and pinned memory only when compatible","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available","no exact or similar graph profiles for current hardware","no verified speedup measurement for FP16 AMP on this task"]}}
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
 Summary: 3 total attempts - 2 pending execution, 1 executed, 0 successful (all failed or buggy)

-------------------------------
Attempt #1:
Design: I merged the three stages by keeping Stage 1's data/model/loss/adapter code as the foundation, layering Stage 2's import-safe precision policy (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `sigmoid_fp32`, `make_grad_scaler`, `to_fp32_policy`, `configure_tf32`, `fp32_island`) on top, and consuming both inside Stage 3's `run_training` (AdamW + cosine warmup, physical batch 32 × accum 2, val-F1 early stopping with best-state restore, scheduler safe-points and `TrainingDiagnostics`). I deduplicated imports/constants, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module-level constants referenced by Stage 3, preserved the CPU-safe `CandidateAdapter` contract, kept all side effects under `if __name__ == '__main__':`, and left the final `Final Validation Score:` print after the diagnostics context.
Results: The implementation of this design has bugs.
Insight: Using a different approach may not result in the same bugs as the above approach.

-------------------------------
Attempt #2:
Design: I merged the three stages by keeping Stage 1's data cleaning, TF-IDF feature builders, `SparseMLP`, `build_loss`, `tune_threshold` and split helpers as the foundation; layering Stage 2's import-safe precision utilities (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `degrade_to_fp32`, `configure_tf32`, `format_precision_log`) on top; then consuming both inside Stage 3's `run_training` with scheduler safe-points, `TrainingDiagnostics`, linear warmup+cosine AdamW, best-val-F1 restore, and test inference. I resolved the duplicate `CandidateAdapter` by keeping Stage 3's self-contained CPU-safe version (it uses the Stage 1 model/loss/features through the Stage 2 precision path), removed duplicate imports/constants, kept all side effects under `if __name__ == "__main__":`, and preserved the exact `MODEL_FAMILY`, `WORKING_DIR`, `SUBMISSION_DIR` contract.
Status: Code generated, execution pending (will run in parallel with other drafts).

-------------------------------
Attempt #3:
Design: I merged the three stages by keeping Stage 1's data/model/loss/adapter code as the foundation, layering Stage 2's `PrecisionRuntime` policy helpers on top, and consuming both inside Stage 3's `run_training` (with diagnostics + scheduler safe points). I resolved naming conflicts, deduplicated imports, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module-level constants that Stage 3 references, and moved every side effect under `if __name__ == '__main__':` while preserving the CPU-safe `CandidateAdapter` contract.
Status: Code generated, execution pending (will run in parallel with other drafts).


# Hardware-Aware Datatype/Precision Context
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2
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
- Stage-filtered hardware knowledge:



# Pipeline Decision Contract
Required hardware-aware step order: model-design -> datatype/quantization -> training (model_design -> datatype_precision -> training_evaluation). Use the trace as the source of truth, apply hardware evidence only when compatible, and use recorded fallbacks when evidence is missing.

# Pipeline Decision Trace
```json
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"{'criterion': 'BCEWithLogitsLoss', 'pos_weight': None, 'notes': 'Compute loss in FP32 island for...","output_interface":"{'raw_output': 'logits', 'probability': 'sigmoid(logits)', 'prediction_threshold': 0.5, 'threshol...","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"{'forward_autocast': {'enabled': True, 'backend': 'torch.amp.autocast', 'device_type': 'cuda', 'd...","fallback_policy":"fp32","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"{'name': 'AdamW', 'lr': 2e-05, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-08, 'notes'...","scheduler":"{'name': 'linear_with_warmup', 'warmup_ratio': 0.1, 'notes': 'Linear decay to zero after warmup.'}","batch_size_policy":"fixed","dataloader_policy":"safe defaults; enable workers and pinned memory only when compatible","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available","no exact or similar graph profiles for current hardware","no verified speedup measurement for FP16 AMP on this task"]}}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=replaced sparse-TF-IDF MLP and vanilla transformer baselines with a BiGRU + additive-attention encoder that injects a learned keyword embedding as an additive bias into every word embedding; purpose=exploit word order and keyword context jointly for short, noisy disaster tweets under strict F1 evaluation; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=introduced explicit `prepare_text_field(text, keyword, location)` fusion that prepends location tokens and appends keyword tokens into the sequence, plus a separate categorical keyword id stream; purpose=preserve location as positional context while keeping keyword as a distinct, low-cardinality conditioning signal usable by Stage 2/3 without re-tokenization; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=exposed a CPU-safe `CandidateAdapter` that constructs real vocab + real model + real `BCEWithLogitsLoss`, plus an F1 threshold-tuner helper (`tune_threshold`) that mirrors the metric used at export; purpose=allow Stage 3 to consume the same model/loss/threshold path used for validation and test inference; hardware_keys=none

# Previous steps
### model_design
**Plan:** ## Plan

**WHAT makes this task unique**: Disaster tweets are short, noisy, and mix word order with strong keyword-level signals (e.g., "wildfire", "earthquake"); the label depends on whether the words describe a *real* disaster versus a metaphor, so sequence order and keyword context interact non-trivially. Prior baseline approaches in memory relied on bag-of-ngram sparse MLP or plain transformer towers, which either lose order signal or overfit at 6k examples.

**WHY this approach fits**: I'll build a compact hybrid **BiGRU + additive attention with token-level keyword-bias injection**: a trainable keyword embedding is added to every word embedding (a "keyword conditioning" bias) so attention can learn keyword-specific saliency, and a pooled keyword-only branch is concatenated with the sequence attention vector. This uses the strongly predictive `keyword` field as a *learned bias inside* the sequence encoder rather than a side feature, which is fundamentally different from the sparse-MLP (`TF-IDF + MLP`) and vanilla-transformer candidates in memory.

**HOW to validate**: Training/validation split is done first (stratified on target, seeded); vocabularies and the keyword map are fit only on the training fold. Confirmation = val F1 clearly above a constant-baseline and stable across epochs, plus monotone improvement as the keyword bias is enabled vs. disabled (ablated later). Refutation = F1 stuck near ~0.5 or strong train/val gap, indicating overfitting of the GRU/attention head and a need to shrink capacity or add stronger regularization.

**Memory strategy / batch elasticity**: Batches are built as (word_ids, keyword_ids, target) tuples with on-the-fly padding via `collate_batch`; the adapter honors `scenario['batch_size']` verbatim and never buffers global state inside the module. Physical vs. effective batch elasticity (accumulation) is a **training-loop decision** and is deliberately not encoded here. **Operator / normalization considerations**: `nn.GRU` + `tanh`-scored additive attention + ReLU MLP head; LayerNorm is *not* applied inside the recurrent block (only dropout), keeping the model stable under FP32/FP16-AMP. **Configurable dimensions**: `EMBED_DIM_DEFAULT`, `GRU_HIDDEN_DEFAULT`, `GRU_LAYERS_DEFAULT`, `DROPOUT_DEFAULT`, `MAX_SEQ_LEN_DEFAULT`, `WORD_VOCAB_SIZE_DEFAULT`, `KEYWORD_VOCAB_SIZE_DEFAULT` are constants that Stage 3 may override. **Fallback behavior**: if keyword/location columns are missing or NaN, they map to `<none>` token; if text is empty, sequence becomes `[UNK]`. **Scheduler-owned controls not implemented here**: device assignment, CUDA/TF32 policy, AMP precision, worker counts, optimizer LR/weight decay, epochs, checkpointing cadence — none are chosen in this file.

---

Note board:
- baseline_change: replaced sparse-TF-IDF MLP and vanilla transformer baselines with a BiGRU + additive-attention encoder that injects a learned keyword embedding as an additive bias into every word embedding | purpose: exploit word order and keyword context jointly for short, noisy disaster tweets under strict F1 evaluation | hardware_keys: none
- baseline_change: introduced explicit `prepare_text_field(text, keyword, location)` fusion that prepends location tokens and appends keyword tokens into the sequence, plus a separate categorical keyword id stream | purpose: preserve location as positional context while keeping keyword as a distinct, low-cardinality conditioning signal usable by Stage 2/3 without re-tokenization | hardware_keys: none
- baseline_change: exposed a CPU-safe `CandidateAdapter` that constructs real vocab + real model + real `BCEWithLogitsLoss`, plus an F1 threshold-tuner helper (`tune_threshold`) that mirrors the metric used at export | purpose: allow Stage 3 to consume the same model/loss/threshold path used for validation and test inference | hardware_keys: none
**Code:**
```python
"""Stage 1: data prep, features, model architecture, loss and output interface.

Model family: bilstm_attention_keyword_bias_v1
Input representation: cleaned word sequence (with location prepended and
keyword appended) + separate categorical keyword id stream.

Nothing in this module touches data or CUDA at import time. All side effects
(loading CSVs, building vocab, training, inference, submission) belong in
`if __name__ == "__main__":` of the final integrated pipeline.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MODEL_FAMILY = "bilstm_attention_keyword_bias_v1"

WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
INPUT_DIR_DEFAULT = "./input"

WORD_VOCAB_SIZE_DEFAULT = 20000
KEYWORD_VOCAB_SIZE_DEFAULT = 512
MAX_SEQ_LEN_DEFAULT = 48
EMBED_DIM_DEFAULT = 128
GRU_HIDDEN_DEFAULT = 128
GRU_LAYERS_DEFAULT = 2
DROPOUT_DEFAULT = 0.3

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
NONE_KEYWORD = "<none>"

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9#']+")
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Text cleaning / tokenization helpers
# ---------------------------------------------------------------------------
def clean_text(text: Any) -> str:
    """Lowercase, strip URLs/mentions/HTML entities and non-alphanumerics."""
    if text is None:
        return ""
    s = str(text).lower()
    s = _URL_RE.sub(" ", s)
    s = _MENTION_RE.sub(" ", s)
    s = s.replace("&amp;", " and ")
    s = s.replace("&lt;", " ").replace("&gt;", " ")
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def tokenize(text: Any) -> List[str]:
    s = clean_text(text)
    if not s:
        return []
    return [t for t in s.split(" ") if t]


def prepare_text_field(text: Any, keyword: Any, location: Any) -> str:
    """Fuse location + text + keyword into a single sequence string.

    Keyword is placed at the end so the recurrent encoder sees the topic hint
    after the main utterance, and keyword ids are also passed separately.
    """
    parts: List[str] = []
    if location is not None and not (
        isinstance(location, float) and np.isnan(location)
    ):
        loc = str(location).strip().replace("%20", " ")
        if loc and loc.lower() != "nan":
            parts.append(loc)
    if text is not None:
        t = str(text)
        if t and t.lower() != "nan":
            parts.append(t)
    if keyword is not None and not (isinstance(keyword, float) and np.isnan(keyword)):
        kw = str(keyword).strip().replace("%20", " ")
        if kw and kw.lower() != "nan":
            parts.append(kw)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Vocabulary builders (fit on the training split only)
# ---------------------------------------------------------------------------
def build_word_vocab(
    texts: List[str], max_size: int = WORD_VOCAB_SIZE_DEFAULT, min_freq: int = 1
) -> Dict[str, int]:
    counter: Dict[str, int] = {}
    for t in texts:
        for tok in tokenize(t):
            counter[tok] = counter.get(tok, 0) + 1
    items = [(w, c) for w, c in counter.items() if c >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    items = items[: max(0, max_size - 2)]
    vocab: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for w, _ in items:
        vocab[w] = len(vocab)
    return vocab


def build_keyword_vocab(
    keywords: List[Any], max_size: int = KEYWORD_VOCAB_SIZE_DEFAULT
) -> Dict[str, int]:
    vocab: Dict[str, int] = {PAD_TOKEN: 0, NONE_KEYWORD: 1}
    seen = set()
    for kw in keywords:
        if kw is None:
            continue
        if isinstance(kw, float) and np.isnan(kw):
            continue
        k = str(kw).strip().lower().replace("%20", " ")
        if not k or k == "nan":
            continue
        seen.add(k)
    for k in sorted(seen)[: max(0, max_size - 2)]:
        vocab[k] = len(vocab)
    return vocab


def encode_tokens(tokens: List[str], vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens]
    if len(ids) > max_len:
        ids = ids[:max_len]
    if len(ids) == 0:
        ids = [vocab[UNK_TOKEN]]
    return ids


def keyword_to_id(kw: Any, vocab: Dict[str, int]) -> int:
    if kw is None:
        return vocab[NONE_KEYWORD]
    if isinstance(kw, float) and np.isnan(kw):
        return vocab[NONE_KEYWORD]
    k = str(kw).strip().lower().replace("%20", " ")
    if not k or k == "nan":
        return vocab[NONE_KEYWORD]
    return vocab.get(k, vocab[UNK_TOKEN])


# ---------------------------------------------------------------------------
# Dataset / collate
# ---------------------------------------------------------------------------
class TweetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        word_ids: List[List[int]],
        keyword_ids: List[int],
        targets: Optional[List[float]] = None,
    ) -> None:
        self.word_ids = word_ids
        self.keyword_ids = keyword_ids
        self.targets = targets

    def __len__(self) -> int:
        return len(self.word_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "word_ids": torch.tensor(self.word_ids[idx], dtype=torch.long),
            "keyword_ids": torch.tensor(int(self.keyword_ids[idx]), dtype=torch.long),
        }
        if self.targets is not None:
            item["target"] = torch.tensor(float(self.targets[idx]), dtype=torch.float32)
        return item


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_len = max(int(b["word_ids"].shape[0]) for b in batch)
    word_rows = []
    for b in batch:
        w = b["word_ids"]
        if w.shape[0] < max_len:
            pad = torch.zeros(max_len - w.shape[0], dtype=torch.long)
            w = torch.cat([w, pad], dim=0)
        word_rows.append(w)
    out = {
        "word_ids": torch.stack(word_rows, dim=0),
        "keyword_ids": torch.stack([b["keyword_ids"] for b in batch], dim=0),
    }
    if "target" in batch[0]:
        out["target"] = torch.stack([b["target"] for b in batch], dim=0)
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DisasterHybridModel(nn.Module):
    """BiGRU + additive attention with keyword-bias injection.

    - `keyword_emb[keyword_ids]` is added to every word embedding in the
      sequence (learned topic conditioning).
    - Additive attention pools the sequence output into a fixed vector.
    - A separate projected keyword feature is concatenated before the MLP head.
    """

    def __init__(
        self,
        word_vocab_size: int,
        keyword_vocab_size: int,
        embed_dim: int = EMBED_DIM_DEFAULT,
        gru_hidden: int = GRU_HIDDEN_DEFAULT,
        gru_layers: int = GRU_LAYERS_DEFAULT,
        dropout: float = DROPOUT_DEFAULT,
        pad_index: int = 0,
    ) -> None:
        super().__init__()
        self.word_vocab_size = int(word_vocab_size)
        self.keyword_vocab_size = int(keyword_vocab_size)
        self.embed_dim = int(embed_dim)
        self.gru_hidden = int(gru_hidden)
        self.gru_layers = int(gru_layers)

        self.word_emb = nn.Embedding(
            self.word_vocab_size, self.embed_dim, padding_idx=pad_index
        )
        self.keyword_emb = nn.Embedding(
            self.keyword_vocab_size, self.embed_dim, padding_idx=pad_index
        )
        nn.init.normal_(self.word_emb.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.keyword_emb.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.word_emb.weight[pad_index].zero_()
            self.keyword_emb.weight[pad_index].zero_()

        self.word_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.gru_hidden,
            num_layers=self.gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if self.gru_layers > 1 else 0.0,
        )

        attn_dim = 2 * self.gru_hidden
        self.attn_proj = nn.Sequential(
            nn.Linear(attn_dim, max(32, attn_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(32, attn_dim // 2), 1),
        )

        self.keyword_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(attn_dim + self.embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self, word_ids: torch.Tensor, keyword_ids: torch.Tensor
    ) -> torch.Tensor:
        if word_ids.dim() == 1:
            word_ids = word_ids.unsqueeze(0)
        if keyword_ids.dim() == 0:
            keyword_ids = keyword_ids.unsqueeze(0)
        if keyword_ids.dim() == 1:
            keyword_ids = keyword_ids

        mask = word_ids != 0  # [B, L]
        word_vec = self.word_emb(word_ids)  # [B, L, D]
        kw_vec = self.keyword_emb(keyword_ids)  # [B, D]

        seq = word_vec + kw_vec.unsqueeze(1)  # broadcast topic bias
        seq = self.word_dropout(seq)

        gru_out, _ = self.gru(seq)  # [B, L, 2H]
        scores = self.attn_proj(gru_out).squeeze(-1)  # [B, L]
        scores = scores.masked_fill(~mask, -1e4)
        alpha = torch.softmax(scores, dim=-1)  # [B, L]
        pooled = torch.einsum("bl,blh->bh", alpha, gru_out)  # [B, 2H]

        kw_feat = self.keyword_proj(kw_vec)  # [B, D]
        x = torch.cat([pooled, kw_feat], dim=-1)  # [B, 2H + D]
        logits = self.head(x).squeeze(-1)  # [B]
        return logits


# ---------------------------------------------------------------------------
# Loss, output interface, metric-aligned threshold helper
# ---------------------------------------------------------------------------
def build_loss(pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
    """Binary-cross-entropy on logits. `pos_weight` may be supplied by Stage 3."""
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def sigmoid_logits(logits: torch.Tensor) -> torch.Tensor:
    """Output interface: probability = sigmoid(model logits)."""
    return torch.sigmoid(logits)


def tune_threshold(probs: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """Convenience F1 sweep used at validation/export time. Returns (best_t, best_f1)."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs >= t).astype(np.int64)
        tp = int(((preds == 1) & (targets == 1)).sum())
        fp = int(((preds == 1) & (targets == 0)).sum())
        fn = int(((preds == 0) & (targets == 1)).sum())
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


# ---------------------------------------------------------------------------
# CPU preflight adapter
# ---------------------------------------------------------------------------
class CandidateAdapter:
    """No-argument adapter exercising the real model/loss/inputs on CPU.

    The adapter lazily materializes a small slice of train.csv so its CPU
    construction is cheap and side-effect free at import time.
    """

    PREFLIGHT_ROWS = 256

    def __init__(self) -> None:
        self._cache: Optional[Dict[str, Any]] = None

    # -- internal -----------------------------------------------------------
    def _ensure_data(self) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache
        input_dir = os.environ.get("MLEVOLVE_INPUT_DIR", INPUT_DIR_DEFAULT)
        train_path = os.path.join(input_dir, "train.csv")
        df = pd.read_csv(train_path).head(self.PREFLIGHT_ROWS).copy()

        text_col = df["text"].tolist() if "text" in df.columns else [""] * len(df)
        keyword_col = (
            df["keyword"].tolist() if "keyword" in df.columns else [None] * len(df)
        )
        location_col = (
            df["location"].tolist() if "location" in df.columns else [None] * len(df)
        )
        target_col = (
            df["target"].astype(float).tolist() if "target" in df.columns else None
        )

        fused_text = [
            prepare_text_field(t, k, l)
            for t, k, l in zip(text_col, keyword_col, location_col)
        ]

        word_vocab = build_word_vocab(fused_text, WORD_VOCAB_SIZE_DEFAULT, min_freq=1)
        keyword_vocab = build_keyword_vocab(keyword_col, KEYWORD_VOCAB_SIZE_DEFAULT)

        word_ids = [
            encode_tokens(tokenize(t), word_vocab, MAX_SEQ_LEN_DEFAULT)
            for t in fused_text
        ]
        kw_ids = [keyword_to_id(k, keyword_vocab) for k in keyword_col]

        self._cache = {
            "word_vocab": word_vocab,
            "keyword_vocab": keyword_vocab,
            "word_ids": word_ids,
            "keyword_ids": kw_ids,
            "targets": target_col,
        }
        return self._cache

    @staticmethod
    def _merge_context(context: Any, defaults: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(defaults)
        if isinstance(context, dict):
            for k, v in context.items():
                if v is not None:
                    merged[k] = v
        return merged

    def _make_batch(self, batch_size: int, device: torch.device) -> Dict[str, Any]:
        cache = self._ensure_data()
        n = len(cache["word_ids"])
        bs = max(1, min(int(batch_size), n))
        idxs = list(range(bs))

        word_ids = torch.stack(
            [torch.tensor(cache["word_ids"][i], dtype=torch.long) for i in idxs], dim=0
        )
        keyword_ids = torch.stack(
            [
                torch.tensor(int(cache["keyword_ids"][i]), dtype=torch.long)
                for i in idxs
            ],
            dim=0,
        )
        if cache["targets"] is not None:
            targets = torch.tensor(
                [float(cache["targets"][i]) for i in idxs], dtype=torch.float32
            )
        else:
            targets = torch.zeros(bs, dtype=torch.float32)

        return {
            "inputs": (word_ids.to(device), keyword_ids.to(device)),
            "target": targets.to(device),
        }

    # -- contract -----------------------------------------------------------
    def build_model(self, context: Any) -> nn.Module:
        cache = self._ensure_data()
        model = DisasterHybridModel(
            word_vocab_size=len(cache["word_vocab"]),
            keyword_vocab_size=len(cache["keyword_vocab"]),
            embed_dim=EMBED_DIM_DEFAULT,
            gru_hidden=GRU_HIDDEN_DEFAULT,
            gru_layers=GRU_LAYERS_DEFAULT,
            dropout=DROPOUT_DEFAULT,
        )
        return model

    def build_optimizer(self, model: nn.Module, context: Any) -> torch.optim.Optimizer:
        # Stage 1 does not choose the real optimizer policy; this default only
        # exists so the CPU preflight can call training_step end-to-end.
        from torch.optim import AdamW

        merged = self._merge_context(context, {"lr": 1e-3, "weight_decay": 0.0})
        return AdamW(
            model.parameters(),
            lr=float(merged["lr"]),
            weight_decay=float(merged["weight_decay"]),
        )

    def build_train_batch(self, scenario: Any, device: torch.device) -> Dict[str, Any]:
        bs = 4
        if isinstance(scenario, dict) and scenario.get("batch_size") is not None:
            bs = int(scenario["batch_size"])
        return self._make_batch(bs, device)

    def build_validation_batch(
        self, scenario: Any, device: torch.device
    ) -> Dict[str, Any]:
        bs = 4
        if isinstance(scenario, dict) and scenario.get("batch_size") is not None:
            bs = int(scenario["batch_size"])
        return self._make_batch(bs, device)

    def training_step(
        self, model: nn.Module, batch: Dict[str, Any], context: Any
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        target = batch["target"]
        logits = model(*inputs)
        criterion = build_loss()
        loss = criterion(logits, target)
        return loss

    def validation_step(
        self, model: nn.Module, batch: Dict[str, Any], context: Any
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        target = batch["target"]
        logits = model(*inputs)
        criterion = build_loss()
        loss = criterion(logits, target)
        return loss

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
 Summary: 3 total attempts - 2 pending execution, 1 executed, 0 successful (all failed or buggy)

-------------------------------
Attempt #1:
Design: I merged the three stages by keeping Stage 1's data/model/loss/adapter code as the foundation, layering Stage 2's import-safe precision policy (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `sigmoid_fp32`, `make_grad_scaler`, `to_fp32_policy`, `configure_tf32`, `fp32_island`) on top, and consuming both inside Stage 3's `run_training` (AdamW + cosine warmup, physical batch 32 × accum 2, val-F1 early stopping with best-state restore, scheduler safe-points and `TrainingDiagnostics`). I deduplicated imports/constants, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module-level constants referenced by Stage 3, preserved the CPU-safe `CandidateAdapter` contract, kept all side effects under `if __name__ == '__main__':`, and left the final `Final Validation Score:` print after the diagnostics context.
Results: The implementation of this design has bugs.
Insight: Using a different approach may not result in the same bugs as the above approach.

-------------------------------
Attempt #2:
Design: I merged the three stages by keeping Stage 1's data cleaning, TF-IDF feature builders, `SparseMLP`, `build_loss`, `tune_threshold` and split helpers as the foundation; layering Stage 2's import-safe precision utilities (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `degrade_to_fp32`, `configure_tf32`, `format_precision_log`) on top; then consuming both inside Stage 3's `run_training` with scheduler safe-points, `TrainingDiagnostics`, linear warmup+cosine AdamW, best-val-F1 restore, and test inference. I resolved the duplicate `CandidateAdapter` by keeping Stage 3's self-contained CPU-safe version (it uses the Stage 1 model/loss/features through the Stage 2 precision path), removed duplicate imports/constants, kept all side effects under `if __name__ == "__main__":`, and preserved the exact `MODEL_FAMILY`, `WORKING_DIR`, `SUBMISSION_DIR` contract.
Status: Code generated, execution pending (will run in parallel with other drafts).

-------------------------------
Attempt #3:
Design: I merged the three stages by keeping Stage 1's data/model/loss/adapter code as the foundation, layering Stage 2's `PrecisionRuntime` policy helpers on top, and consuming both inside Stage 3's `run_training` (with diagnostics + scheduler safe points). I resolved naming conflicts, deduplicated imports, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module-level constants that Stage 3 references, and moved every side effect under `if __name__ == '__main__':` while preserving the CPU-safe `CandidateAdapter` contract.
Status: Code generated, execution pending (will run in parallel with other drafts).


# Hardware-Aware Training Hyperparameter Context
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2
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
- Confidence: 0.0
adamw_params = []



# Pipeline Decision Contract
Required hardware-aware step order: model-design -> datatype/quantization -> training (model_design -> datatype_precision -> training_evaluation). Use the trace as the source of truth, apply hardware evidence only when compatible, and use recorded fallbacks when evidence is missing.

# Pipeline Decision Trace
```json
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"{'criterion': 'BCEWithLogitsLoss', 'pos_weight': None, 'notes': 'Compute loss in FP32 island for...","output_interface":"{'raw_output': 'logits', 'probability': 'sigmoid(logits)', 'prediction_threshold': 0.5, 'threshol...","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"{'forward_autocast': {'enabled': True, 'backend': 'torch.amp.autocast', 'device_type': 'cuda', 'd...","fallback_policy":"fp32","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"{'name': 'AdamW', 'lr': 2e-05, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-08, 'notes'...","scheduler":"{'name': 'linear_with_warmup', 'warmup_ratio': 0.1, 'notes': 'Linear decay to zero after warmup.'}","batch_size_policy":"fixed","dataloader_policy":"safe defaults; enable workers and pinned memory only when compatible","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available","no exact or similar graph profiles for current hardware","no verified speedup measurement for FP16 AMP on this task"]}}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=replaced sparse-TF-IDF MLP and vanilla transformer baselines with a BiGRU + additive-attention encoder that injects a learned keyword embedding as an additive bias into every word embedding; purpose=exploit word order and keyword context jointly for short, noisy disaster tweets under strict F1 evaluation; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=introduced explicit `prepare_text_field(text, keyword, location)` fusion that prepends location tokens and appends keyword tokens into the sequence, plus a separate categorical keyword id stream; purpose=preserve location as positional context while keeping keyword as a distinct, low-cardinality conditioning signal usable by Stage 2/3 without re-tokenization; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=exposed a CPU-safe `CandidateAdapter` that constructs real vocab + real model + real `BCEWithLogitsLoss`, plus an F1 threshold-tuner helper (`tune_threshold`) that mirrors the metric used at export; purpose=allow Stage 3 to consume the same model/loss/threshold path used for validation and test inference; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=introduced `PrecisionRuntime` + helpers (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `sigmoid_fp32`, `fp32_island`, `make_grad_scaler`, `configure_tf32`, `format_precision_log`); purpose=give Stage 3 a single FP16-AMP-with-FP32-islands contract that keeps the BiGRU+keyword-bias forward stable while preserving the metric-aligned sigmoid/threshold export path; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=made every precision action a callable (no CUDA/TF32 flags set at import) and forced the CPU preflight path to FP32 with `scaler=None`; purpose=satisfy the import-safe contract and keep the CandidateAdapter usable on a CPU-only checker without AMP overhead; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=kept device/dtype configuration as resolved-at-runtime metadata (`device`, `use_amp`, `amp_dtype`, `use_tf32`, `grad_scaler`) rather than module globals; purpose=let Stage 3 choose device inside its main guard without coupling to a single hardcoded trainer; hardware_keys=none

# Previous steps
### model_design
**Plan:** ## Plan

**WHAT makes this task unique**: Disaster tweets are short, noisy, and mix word order with strong keyword-level signals (e.g., "wildfire", "earthquake"); the label depends on whether the words describe a *real* disaster versus a metaphor, so sequence order and keyword context interact non-trivially. Prior baseline approaches in memory relied on bag-of-ngram sparse MLP or plain transformer towers, which either lose order signal or overfit at 6k examples.

**WHY this approach fits**: I'll build a compact hybrid **BiGRU + additive attention with token-level keyword-bias injection**: a trainable keyword embedding is added to every word embedding (a "keyword conditioning" bias) so attention can learn keyword-specific saliency, and a pooled keyword-only branch is concatenated with the sequence attention vector. This uses the strongly predictive `keyword` field as a *learned bias inside* the sequence encoder rather than a side feature, which is fundamentally different from the sparse-MLP (`TF-IDF + MLP`) and vanilla-transformer candidates in memory.

**HOW to validate**: Training/validation split is done first (stratified on target, seeded); vocabularies and the keyword map are fit only on the training fold. Confirmation = val F1 clearly above a constant-baseline and stable across epochs, plus monotone improvement as the keyword bias is enabled vs. disabled (ablated later). Refutation = F1 stuck near ~0.5 or strong train/val gap, indicating overfitting of the GRU/attention head and a need to shrink capacity or add stronger regularization.

**Memory strategy / batch elasticity**: Batches are built as (word_ids, keyword_ids, target) tuples with on-the-fly padding via `collate_batch`; the adapter honors `scenario['batch_size']` verbatim and never buffers global state inside the module. Physical vs. effective batch elasticity (accumulation) is a **training-loop decision** and is deliberately not encoded here. **Operator / normalization considerations**: `nn.GRU` + `tanh`-scored additive attention + ReLU MLP head; LayerNorm is *not* applied inside the recurrent block (only dropout), keeping the model stable under FP32/FP16-AMP. **Configurable dimensions**: `EMBED_DIM_DEFAULT`, `GRU_HIDDEN_DEFAULT`, `GRU_LAYERS_DEFAULT`, `DROPOUT_DEFAULT`, `MAX_SEQ_LEN_DEFAULT`, `WORD_VOCAB_SIZE_DEFAULT`, `KEYWORD_VOCAB_SIZE_DEFAULT` are constants that Stage 3 may override. **Fallback behavior**: if keyword/location columns are missing or NaN, they map to `<none>` token; if text is empty, sequence becomes `[UNK]`. **Scheduler-owned controls not implemented here**: device assignment, CUDA/TF32 policy, AMP precision, worker counts, optimizer LR/weight decay, epochs, checkpointing cadence — none are chosen in this file.

---

Note board:
- baseline_change: replaced sparse-TF-IDF MLP and vanilla transformer baselines with a BiGRU + additive-attention encoder that injects a learned keyword embedding as an additive bias into every word embedding | purpose: exploit word order and keyword context jointly for short, noisy disaster tweets under strict F1 evaluation | hardware_keys: none
- baseline_change: introduced explicit `prepare_text_field(text, keyword, location)` fusion that prepends location tokens and appends keyword tokens into the sequence, plus a separate categorical keyword id stream | purpose: preserve location as positional context while keeping keyword as a distinct, low-cardinality conditioning signal usable by Stage 2/3 without re-tokenization | hardware_keys: none
- baseline_change: exposed a CPU-safe `CandidateAdapter` that constructs real vocab + real model + real `BCEWithLogitsLoss`, plus an F1 threshold-tuner helper (`tune_threshold`) that mirrors the metric used at export | purpose: allow Stage 3 to consume the same model/loss/threshold path used for validation and test inference | hardware_keys: none
**Code:**
```python
"""Stage 1: data prep, features, model architecture, loss and output interface.

Model family: bilstm_attention_keyword_bias_v1
Input representation: cleaned word sequence (with location prepended and
keyword appended) + separate categorical keyword id stream.

Nothing in this module touches data or CUDA at import time. All side effects
(loading CSVs, building vocab, training, inference, submission) belong in
`if __name__ == "__main__":` of the final integrated pipeline.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MODEL_FAMILY = "bilstm_attention_keyword_bias_v1"

WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
INPUT_DIR_DEFAULT = "./input"

WORD_VOCAB_SIZE_DEFAULT = 20000
KEYWORD_VOCAB_SIZE_DEFAULT = 512
MAX_SEQ_LEN_DEFAULT = 48
EMBED_DIM_DEFAULT = 128
GRU_HIDDEN_DEFAULT = 128
GRU_LAYERS_DEFAULT = 2
DROPOUT_DEFAULT = 0.3

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
NONE_KEYWORD = "<none>"

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9#']+")
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Text cleaning / tokenization helpers
# ---------------------------------------------------------------------------
def clean_text(text: Any) -> str:
    """Lowercase, strip URLs/mentions/HTML entities and non-alphanumerics."""
    if text is None:
        return ""
    s = str(text).lower()
    s = _URL_RE.sub(" ", s)
    s = _MENTION_RE.sub(" ", s)
    s = s.replace("&amp;", " and ")
    s = s.replace("&lt;", " ").replace("&gt;", " ")
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def tokenize(text: Any) -> List[str]:
    s = clean_text(text)
    if not s:
        return []
    return [t for t in s.split(" ") if t]


def prepare_text_field(text: Any, keyword: Any, location: Any) -> str:
    """Fuse location + text + keyword into a single sequence string.

    Keyword is placed at the end so the recurrent encoder sees the topic hint
    after the main utterance, and keyword ids are also passed separately.
    """
    parts: List[str] = []
    if location is not None and not (
        isinstance(location, float) and np.isnan(location)
    ):
        loc = str(location).strip().replace("%20", " ")
        if loc and loc.lower() != "nan":
            parts.append(loc)
    if text is not None:
        t = str(text)
        if t and t.lower() != "nan":
            parts.append(t)
    if keyword is not None and not (isinstance(keyword, float) and np.isnan(keyword)):
        kw = str(keyword).strip().replace("%20", " ")
        if kw and kw.lower() != "nan":
            parts.append(kw)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Vocabulary builders (fit on the training split only)
# ---------------------------------------------------------------------------
def build_word_vocab(
    texts: List[str], max_size: int = WORD_VOCAB_SIZE_DEFAULT, min_freq: int = 1
) -> Dict[str, int]:
    counter: Dict[str, int] = {}
    for t in texts:
        for tok in tokenize(t):
            counter[tok] = counter.get(tok, 0) + 1
    items = [(w, c) for w, c in counter.items() if c >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    items = items[: max(0, max_size - 2)]
    vocab: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for w, _ in items:
        vocab[w] = len(vocab)
    return vocab


def build_keyword_vocab(
    keywords: List[Any], max_size: int = KEYWORD_VOCAB_SIZE_DEFAULT
) -> Dict[str, int]:
    vocab: Dict[str, int] = {PAD_TOKEN: 0, NONE_KEYWORD: 1}
    seen = set()
    for kw in keywords:
        if kw is None:
            continue
        if isinstance(kw, float) and np.isnan(kw):
            continue
        k = str(kw).strip().lower().replace("%20", " ")
        if not k or k == "nan":
            continue
        seen.add(k)
    for k in sorted(seen)[: max(0, max_size - 2)]:
        vocab[k] = len(vocab)
    return vocab


def encode_tokens(tokens: List[str], vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens]
    if len(ids) > max_len:
        ids = ids[:max_len]
    if len(ids) == 0:
        ids = [vocab[UNK_TOKEN]]
    return ids


def keyword_to_id(kw: Any, vocab: Dict[str, int]) -> int:
    if kw is None:
        return vocab[NONE_KEYWORD]
    if isinstance(kw, float) and np.isnan(kw):
        return vocab[NONE_KEYWORD]
    k = str(kw).strip().lower().replace("%20", " ")
    if not k or k == "nan":
        return vocab[NONE_KEYWORD]
    return vocab.get(k, vocab[UNK_TOKEN])


# ---------------------------------------------------------------------------
# Dataset / collate
# ---------------------------------------------------------------------------
class TweetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        word_ids: List[List[int]],
        keyword_ids: List[int],
        targets: Optional[List[float]] = None,
    ) -> None:
        self.word_ids = word_ids
        self.keyword_ids = keyword_ids
        self.targets = targets

    def __len__(self) -> int:
        return len(self.word_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "word_ids": torch.tensor(self.word_ids[idx], dtype=torch.long),
            "keyword_ids": torch.tensor(int(self.keyword_ids[idx]), dtype=torch.long),
        }
        if self.targets is not None:
            item["target"] = torch.tensor(float(self.targets[idx]), dtype=torch.float32)
        return item


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_len = max(int(b["word_ids"].shape[0]) for b in batch)
    word_rows = []
    for b in batch:
        w = b["word_ids"]
        if w.shape[0] < max_len:
            pad = torch.zeros(max_len - w.shape[0], dtype=torch.long)
            w = torch.cat([w, pad], dim=0)
        word_rows.append(w)
    out = {
        "word_ids": torch.stack(word_rows, dim=0),
        "keyword_ids": torch.stack([b["keyword_ids"] for b in batch], dim=0),
    }
    if "target" in batch[0]:
        out["target"] = torch.stack([b["target"] for b in batch], dim=0)
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DisasterHybridModel(nn.Module):
    """BiGRU + additive attention with keyword-bias injection.

    - `keyword_emb[keyword_ids]` is added to every word embedding in the
      sequence (learned topic conditioning).
    - Additive attention pools the sequence output into a fixed vector.
    - A separate projected keyword feature is concatenated before the MLP head.
    """

    def __init__(
        self,
        word_vocab_size: int,
        keyword_vocab_size: int,
        embed_dim: int = EMBED_DIM_DEFAULT,
        gru_hidden: int = GRU_HIDDEN_DEFAULT,
        gru_layers: int = GRU_LAYERS_DEFAULT,
        dropout: float = DROPOUT_DEFAULT,
        pad_index: int = 0,
    ) -> None:
        super().__init__()
        self.word_vocab_size = int(word_vocab_size)
        self.keyword_vocab_size = int(keyword_vocab_size)
        self.embed_dim = int(embed_dim)
        self.gru_hidden = int(gru_hidden)
        self.gru_layers = int(gru_layers)

        self.word_emb = nn.Embedding(
            self.word_vocab_size, self.embed_dim, padding_idx=pad_index
        )
        self.keyword_emb = nn.Embedding(
            self.keyword_vocab_size, self.embed_dim, padding_idx=pad_index
        )
        nn.init.normal_(self.word_emb.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.keyword_emb.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.word_emb.weight[pad_index].zero_()
            self.keyword_emb.weight[pad_index].zero_()

        self.word_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.gru_hidden,
            num_layers=self.gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if self.gru_layers > 1 else 0.0,
        )

        attn_dim = 2 * self.gru_hidden
        self.attn_proj = nn.Sequential(
            nn.Linear(attn_dim, max(32, attn_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(32, attn_dim // 2), 1),
        )

        self.keyword_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(attn_dim + self.embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self, word_ids: torch.Tensor, keyword_ids: torch.Tensor
    ) -> torch.Tensor:
        if word_ids.dim() == 1:
            word_ids = word_ids.unsqueeze(0)
        if keyword_ids.dim() == 0:
            keyword_ids = keyword_ids.unsqueeze(0)
        if keyword_ids.dim() == 1:
            keyword_ids = keyword_ids

        mask = word_ids != 0  # [B, L]
        word_vec = self.word_emb(word_ids)  # [B, L, D]
        kw_vec = self.keyword_emb(keyword_ids)  # [B, D]

        seq = word_vec + kw_vec.unsqueeze(1)  # broadcast topic bias
        seq = self.word_dropout(seq)

        gru_out, _ = self.gru(seq)  # [B, L, 2H]
        scores = self.attn_proj(gru_out).squeeze(-1)  # [B, L]
        scores = scores.masked_fill(~mask, -1e4)
        alpha = torch.softmax(scores, dim=-1)  # [B, L]
        pooled = torch.einsum("bl,blh->bh", alpha, gru_out)  # [B, 2H]

        kw_feat = self.keyword_proj(kw_vec)  # [B, D]
        x = torch.cat([pooled, kw_feat], dim=-1)  # [B, 2H + D]
        logits = self.head(x).squeeze(-1)  # [B]
        return logits


# ---------------------------------------------------------------------------
# Loss, output interface, metric-aligned threshold helper
# ---------------------------------------------------------------------------
def build_loss(pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
    """Binary-cross-entropy on logits. `pos_weight` may be supplied by Stage 3."""
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def sigmoid_logits(logits: torch.Tensor) -> torch.Tensor:
    """Output interface: probability = sigmoid(model logits)."""
    return torch.sigmoid(logits)


def tune_threshold(probs: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """Convenience F1 sweep used at validation/export time. Returns (best_t, best_f1)."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs >= t).astype(np.int64)
        tp = int(((preds == 1) & (targets == 1)).sum())
        fp = int(((preds == 1) & (targets == 0)).sum())
        fn = int(((preds == 0) & (targets == 1)).sum())
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


# ---------------------------------------------------------------------------
# CPU preflight adapter
# ---------------------------------------------------------------------------
class CandidateAdapter:
    """No-argument adapter exercising the real model/loss/inputs on CPU.

    The adapter lazily materializes a small slice of train.csv so its CPU
    construction is cheap and side-effect free at import time.
    """

    PREFLIGHT_ROWS = 256

    def __init__(self) -> None:
        self._cache: Optional[Dict[str, Any]] = None

    # -- internal -----------------------------------------------------------
    def _ensure_data(self) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache
        input_dir = os.environ.get("MLEVOLVE_INPUT_DIR", INPUT_DIR_DEFAULT)
        train_path = os.path.join(input_dir, "train.csv")
        df = pd.read_csv(train_path).head(self.PREFLIGHT_ROWS).copy()

        text_col = df["text"].tolist() if "text" in df.columns else [""] * len(df)
        keyword_col = (
            df["keyword"].tolist() if "keyword" in df.columns else [None] * len(df)
        )
        location_col = (
            df["location"].tolist() if "location" in df.columns else [None] * len(df)
        )
        target_col = (
            df["target"].astype(float).tolist() if "target" in df.columns else None
        )

        fused_text = [
            prepare_text_field(t, k, l)
            for t, k, l in zip(text_col, keyword_col, location_col)
        ]

        word_vocab = build_word_vocab(fused_text, WORD_VOCAB_SIZE_DEFAULT, min_freq=1)
        keyword_vocab = build_keyword_vocab(keyword_col, KEYWORD_VOCAB_SIZE_DEFAULT)

        word_ids = [
            encode_tokens(tokenize(t), word_vocab, MAX_SEQ_LEN_DEFAULT)
            for t in fused_text
        ]
        kw_ids = [keyword_to_id(k, keyword_vocab) for k in keyword_col]

        self._cache = {
            "word_vocab": word_vocab,
            "keyword_vocab": keyword_vocab,
            "word_ids": word_ids,
            "keyword_ids": kw_ids,
            "targets": target_col,
        }
        return self._cache

    @staticmethod
    def _merge_context(context: Any, defaults: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(defaults)
        if isinstance(context, dict):
            for k, v in context.items():
                if v is not None:
                    merged[k] = v
        return merged

    def _make_batch(self, batch_size: int, device: torch.device) -> Dict[str, Any]:
        cache = self._ensure_data()
        n = len(cache["word_ids"])
        bs = max(1, min(int(batch_size), n))
        idxs = list(range(bs))

        word_ids = torch.stack(
            [torch.tensor(cache["word_ids"][i], dtype=torch.long) for i in idxs], dim=0
        )
        keyword_ids = torch.stack(
            [
                torch.tensor(int(cache["keyword_ids"][i]), dtype=torch.long)
                for i in idxs
            ],
            dim=0,
        )
        if cache["targets"] is not None:
            targets = torch.tensor(
                [float(cache["targets"][i]) for i in idxs], dtype=torch.float32
            )
        else:
            targets = torch.zeros(bs, dtype=torch.float32)

        return {
            "inputs": (word_ids.to(device), keyword_ids.to(device)),
            "target": targets.to(device),
        }

    # -- contract -----------------------------------------------------------
    def build_model(self, context: Any) -> nn.Module:
        cache = self._ensure_data()
        model = DisasterHybridModel(
            word_vocab_size=len(cache["word_vocab"]),
            keyword_vocab_size=len(cache["keyword_vocab"]),
            embed_dim=EMBED_DIM_DEFAULT,
            gru_hidden=GRU_HIDDEN_DEFAULT,
            gru_layers=GRU_LAYERS_DEFAULT,
            dropout=DROPOUT_DEFAULT,
        )
        return model

    def build_optimizer(self, model: nn.Module, context: Any) -> torch.optim.Optimizer:
        # Stage 1 does not choose the real optimizer policy; this default only
        # exists so the CPU preflight can call training_step end-to-end.
        from torch.optim import AdamW

        merged = self._merge_context(context, {"lr": 1e-3, "weight_decay": 0.0})
        return AdamW(
            model.parameters(),
            lr=float(merged["lr"]),
            weight_decay=float(merged["weight_decay"]),
        )

    def build_train_batch(self, scenario: Any, device: torch.device) -> Dict[str, Any]:
        bs = 4
        if isinstance(scenario, dict) and scenario.get("batch_size") is not None:
            bs = int(scenario["batch_size"])
        return self._make_batch(bs, device)

    def build_validation_batch(
        self, scenario: Any, device: torch.device
    ) -> Dict[str, Any]:
        bs = 4
        if isinstance(scenario, dict) and scenario.get("batch_size") is not None:
            bs = int(scenario["batch_size"])
        return self._make_batch(bs, device)

    def training_step(
        self, model: nn.Module, batch: Dict[str, Any], context: Any
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        target = batch["target"]
        logits = model(*inputs)
        criterion = build_loss()
        loss = criterion(logits, target)
        return loss

    def validation_step(
        self, model: nn.Module, batch: Dict[str, Any], context: Any
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        target = batch["target"]
        logits = model(*inputs)
        criterion = build_loss()
        loss = criterion(logits, target)
        return loss

```

### datatype_precision
**Plan:** ## Plan

This step locks in the Stage 2 precision policy decided by the pipeline trace (`fp16_amp` with `fp32` fallback) and wires it as import-safe helpers the Stage 3 trainer can consume. Because the Stage 1 model is a BiGRU + additive attention with LayerNorm-free recurrent block, FP16 AMP is safe for the forward path (embedding lookup, GRU, attention, MLP head) with the loss computed in a local FP32 island and sigmoid/reductions kept in FP32 for F1 stability. TF32 is explicitly disabled for matmul and convolution, GradScaler is created only when AMP is enabled on CUDA, and the CPU preflight path force-disables AMP/scaling; everything is a helper so no side effects run at import time.

Note board:
- baseline_change: introduced `PrecisionRuntime` + helpers (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `sigmoid_fp32`, `fp32_island`, `make_grad_scaler`, `configure_tf32`, `format_precision_log`) | purpose: give Stage 3 a single FP16-AMP-with-FP32-islands contract that keeps the BiGRU+keyword-bias forward stable while preserving the metric-aligned sigmoid/threshold export path | hardware_keys: none
- baseline_change: made every precision action a callable (no CUDA/TF32 flags set at import) and forced the CPU preflight path to FP32 with `scaler=None` | purpose: satisfy the import-safe contract and keep the CandidateAdapter usable on a CPU-only checker without AMP overhead | hardware_keys: none
- baseline_change: kept device/dtype configuration as resolved-at-runtime metadata (`device`, `use_amp`, `amp_dtype`, `use_tf32`, `grad_scaler`) rather than module globals | purpose: let Stage 3 choose device inside its main guard without coupling to a single hardcoded trainer | hardware_keys: none
**Code:**
```python
"""Stage 2: datatype / precision policy for the Stage 1 BiGRU+keyword-bias model.

Chosen policy (per Pipeline Decision Trace): ``fp16_amp`` with ``fp32`` fallback.
- FP32 is the storage dtype for parameters, floating inputs and floating targets.
- FP16 autocast is used ONLY around the forward pass (embedding -> GRU ->
  attention -> MLP head) on CUDA. It is never applied to Backward or optimizer.
- The loss and every F1-relevant reduction (sigmoid, metric accumulation,
  threshold sweep, export) run inside an explicit FP32 island.
- TF32 is disabled for matmul and convolution.
- GradScaler is created only when AMP is actually enabled on CUDA.
- CPU preflight is FP32 with AMP and scaling disabled.

Nothing in this module touches CUDA, TF32 state, data, or models at import time.
All configuration is exposed as helpers so the caller (or the CPU preflight
adapter) can opt in explicitly inside its own main guard.
"""

import contextlib
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn

# ---- import-safe autocast / GradScaler shims (both APIs supported) ----------
try:  # PyTorch >= 2.4 style
    from torch.amp import autocast as _torch_autocast
    from torch.amp import GradScaler as _TorchGradScaler

    _AMP_API = "torch.amp"
except ImportError:  # pragma: no cover - fallback for older torch
    from torch.cuda.amp import autocast as _torch_autocast  # type: ignore
    from torch.cuda.amp import GradScaler as _TorchGradScaler  # type: ignore

    _AMP_API = "torch.cuda.amp"


# ---------------------------------------------------------------------------
# Constants (import-safe; no CUDA queries executed at import time)
# ---------------------------------------------------------------------------
MODEL_FAMILY = "bilstm_attention_keyword_bias_v1"

# Defaults chosen by the Pipeline Decision Trace for Stage 2.
DEFAULT_USE_AMP = True
DEFAULT_AMP_DTYPE = "float16"
DEFAULT_USE_TF32 = False
DEFAULT_FALLBACK_POLICY = "fp32"
DEFAULT_SCALER_INIT_SCALE = 2.0**16
DEFAULT_SCALER_GROWTH_INTERVAL = 2000
DEFAULT_SCALER_BACKOFF_FACTOR = 0.5
DEFAULT_SCALER_GROWTH_FACTOR = 2.0

# Named precision labels used in logs (kept stable for downstream parsing).
PRECISION_FP32 = "fp32"
PRECISION_FP16_AMP = "fp16_amp"


# ---------------------------------------------------------------------------
# Device / TF32 helpers (call-only; no import-time side effects)
# ---------------------------------------------------------------------------
def resolve_device(prefer_cuda: bool = True) -> torch.device:
    """Return the compute device to use. Caller decides when to invoke it."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_tf32(enabled: bool = DEFAULT_USE_TF32) -> Dict[str, bool]:
    """Explicitly set TF32 flags. Returns the resulting state for logging.

    In normal mode we disable TF32 for both matmul and convolution; the trace
    marks TF32 as off, and FP16 AMP is the only accelerated path allowed here.
    """
    matmul_flag = bool(enabled)
    cudnn_flag = bool(enabled)
    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = matmul_flag
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = cudnn_flag
    except Exception:
        # Flag support depends on build; keep going without hard failure.
        pass
    return {"matmul_allow_tf32": matmul_flag, "cudnn_allow_tf32": cudnn_flag}


# ---------------------------------------------------------------------------
# FP32 islands
# ---------------------------------------------------------------------------
def fp32_island(device_type: Optional[str] = None):
    """Return a context manager that locally disables autocast.

    Use inside an ambient autocast block to compute numerically fragile ops
    (loss, reductions, sigmoid) in FP32. On CPU this is a no-op.
    """
    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if device_type == "cuda":
        try:
            return _torch_autocast(device_type="cuda", enabled=False)
        except TypeError:  # pragma: no cover - very old torch
            return _torch_autocast(enabled=False)
    return contextlib.nullcontext()


def to_fp32_policy(x: Any) -> Any:
    """Cast floating tensors to FP32; leave integer/index tensors untouched."""
    if torch.is_tensor(x):
        if x.is_floating_point() and x.dtype != torch.float32:
            return x.float()
        return x
    if isinstance(x, (list, tuple)):
        return type(x)(to_fp32_policy(v) for v in x)
    if isinstance(x, dict):
        return {k: to_fp32_policy(v) for k, v in x.items()}
    return x


# ---------------------------------------------------------------------------
# Precision-aware loss / output interface (mirror Stage 1 signatures)
# ---------------------------------------------------------------------------
def compute_bce_with_logits_fp32(
    criterion: Optional[nn.Module],
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy on logits, always computed in FP32.

    The Stage 1 criterion is `nn.BCEWithLogitsLoss`; if it is missing we fall
    back to a fresh instance so the CPU-preflight path never needs the caller
    to pre-materialize the loss.
    """
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    with fp32_island("cuda" if logits.is_cuda else "cpu"):
        logits_fp32 = logits.float() if logits.is_floating_point() else logits
        targets_fp32 = targets.float() if targets.is_floating_point() else targets
        loss = criterion(logits_fp32, targets_fp32)
    return loss


def sigmoid_fp32(logits: torch.Tensor) -> torch.Tensor:
    """Stage 1 output interface (probability = sigmoid(logits)) in FP32."""
    with fp32_island("cuda" if logits.is_cuda else "cpu"):
        logits_fp32 = logits.float() if logits.is_floating_point() else logits
        return torch.sigmoid(logits_fp32)


# ---------------------------------------------------------------------------
# GradScaler factory (only meaningful when AMP is enabled on CUDA)
# ---------------------------------------------------------------------------
def make_grad_scaler(
    enabled: bool,
    *,
    init_scale: float = DEFAULT_SCALER_INIT_SCALE,
    growth_interval: int = DEFAULT_SCALER_GROWTH_INTERVAL,
    backoff_factor: float = DEFAULT_SCALER_BACKOFF_FACTOR,
    growth_factor: float = DEFAULT_SCALER_GROWTH_FACTOR,
) -> Optional[Any]:
    """Create a GradScaler when AMP is enabled, else return None.

    The scaler is intentionally created WITHOUT a hardcoded device type so it
    remains compatible with both `torch.amp.GradScaler("cuda", ...)` and the
    legacy `torch.cuda.amp.GradScaler(...)` API.
    """
    if not enabled:
        return None
    kwargs = {
        "init_scale": float(init_scale),
        "growth_interval": int(growth_interval),
        "backoff_factor": float(backoff_factor),
        "growth_factor": float(growth_factor),
    }
    try:
        return _TorchGradScaler("cuda", **kwargs)
    except TypeError:
        # Legacy signature (no device_type positional argument).
        return _TorchGradScaler(**kwargs)


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------
def resolve_precision_config(
    device: Optional[torch.device] = None,
    override: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve the effective precision policy for the current run.

    Never raises on unsupported hardware: on CPU or when CUDA is unavailable
    we silently fall back to pure FP32 with `scaler=None`, matching the
    ``fallback_policy = fp32`` entry of the Pipeline Decision Trace.
    """
    if device is None:
        device = resolve_device(prefer_cuda=True)
    device = torch.device(device)

    cfg: Dict[str, Any] = {
        "device": device,
        "device_type": device.type,
        "cuda_available": bool(torch.cuda.is_available()),
        "requested_policy": PRECISION_FP16_AMP,
        "use_amp": DEFAULT_USE_AMP,
        "amp_dtype": DEFAULT_AMP_DTYPE,
        "amp_dtype_torch": torch.float16,
        "use_tf32": DEFAULT_USE_TF32,
        "fallback_policy": DEFAULT_FALLBACK_POLICY,
        "fp32_islands": ("loss", "sigmoid", "metrics", "export"),
        "autocast_regions": ("forward",),
        "scaler_initialized": False,
        "precision_label": PRECISION_FP16_AMP,
        "reason": "trace: fp16_amp on cuda with fp32 loss/metric/export islands",
    }

    if override:
        for k, v in override.items():
            if v is not None:
                cfg[k] = v

    # AMP is only valid on CUDA. Everything else falls back to FP32.
    if device.type != "cuda" or not cfg["cuda_available"]:
        cfg["use_amp"] = False
        cfg["amp_dtype"] = "float32"
        cfg["amp_dtype_torch"] = torch.float32
        cfg["precision_label"] = PRECISION_FP32
        cfg["reason"] = "cpu or cuda unavailable -> fp32 fallback"

    # TF32 stays disabled unless a caller explicitly overrides it (normal mode
    # does not permit TF32).
    cfg["tf32_state"] = {"matmul_allow_tf32": False, "cudnn_allow_tf32": False}
    return cfg


# ---------------------------------------------------------------------------
# Lightweight precision logger
# ---------------------------------------------------------------------------
def format_precision_log(cfg: Mapping[str, Any]) -> str:
    """One-line precision summary; safe to print once per run."""
    label = cfg.get("precision_label", PRECISION_FP32)
    device_type = cfg.get("device_type", "cpu")
    use_amp = bool(cfg.get("use_amp", False))
    amp_dtype = cfg.get("amp_dtype", "float32")
    use_tf32 = bool(cfg.get("use_tf32", False))
    scaler = "on" if cfg.get("scaler_initialized") else "off"
    reason = cfg.get("reason", "")
    return (
        f"[precision] policy={label} device={device_type} amp={use_amp} "
        f"amp_dtype={amp_dtype} tf32={use_tf32} scaler={scaler} "
        f"fp32_islands={','.join(cfg.get('fp32_islands', ()))} reason={reason}"
    )


# ---------------------------------------------------------------------------
# Runtime wrapper
# ---------------------------------------------------------------------------
class PrecisionRuntime:
    """Encapsulates device + AMP + scaler behavior for the Stage 3 trainer.

    The object is intentionally passive: it does not modify TF32 state at
    construction (callers do that explicitly via `configure_tf32`) and it does
    not own any data. It only brokers forward/loss/sigmoid calls under the
    correct precision, leaving backward/optimizer/scaler.step to the caller.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        cfg = resolve_precision_config(override=config)
        self.cfg: Dict[str, Any] = dict(cfg)

        self.device: torch.device = self.cfg["device"]
        self.device_type: str = self.cfg["device_type"]
        self.use_amp: bool = bool(self.cfg.get("use_amp", False))
        self.amp_dtype: torch.dtype = self.cfg.get("amp_dtype_torch", torch.float32)
        self.fallback_policy: str = str(
            self.cfg.get("fallback_policy", DEFAULT_FALLBACK_POLICY)
        )

        self._autocast_available = self.device_type == "cuda"
        self.grad_scaler: Optional[Any] = make_grad_scaler(
            self.use_amp and self._autocast_available
        )
        self.cfg["scaler_initialized"] = self.grad_scaler is not None
        if self.grad_scaler is not None:
            self.cfg["precision_label"] = PRECISION_FP16_AMP
        else:
            # No scaler -> we are in a pure-FP32 path.
            if not self.use_amp:
                self.cfg["precision_label"] = PRECISION_FP32

    # -- autocast context ---------------------------------------------------
    def autocast(self):
        """Forward-only autocast context. No-op on CPU or when AMP is off."""
        if self.use_amp and self._autocast_available:
            try:
                return _torch_autocast(device_type="cuda", dtype=self.amp_dtype)
            except TypeError:  # pragma: no cover - very old torch
                return _torch_autocast(dtype=self.amp_dtype)
        return contextlib.nullcontext()

    # -- forward under autocast --------------------------------------------
    def autocast_forward(self, model: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
        """Run the model forward under the resolved AMP region.

        Parameters and inputs stay in FP32; autocast handles the internal casts.
        """
        with self.autocast():
            return model(*inputs)

    # -- FP32-sensitive outputs --------------------------------------------
    def compute_loss(
        self,
        criterion: Optional[nn.Module],
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return compute_bce_with_logits_fp32(criterion, logits, targets)

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return sigmoid_fp32(logits)

    # -- scaler conveniences ------------------------------------------------
    @property
    def scaler_enabled(self) -> bool:
        return self.grad_scaler is not None

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        if self.grad_scaler is None:
            return loss
        return self.grad_scaler.scale(loss)

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(optimizer)

    def scaler_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.grad_scaler is not None:
            self.grad_scaler.step(optimizer)
        else:
            optimizer.step()

    def scaler_update(self) -> None:
        if self.grad_scaler is not None:
            self.grad_scaler.update()

    # -- diagnostics --------------------------------------------------------
    def describe(self) -> str:
        return format_precision_log(self.cfg)

    def as_settings(self) -> Dict[str, Any]:
        """Compact dict suitable for TrainingDiagnostics settings."""
        return {
            "precision_label": self.cfg.get("precision_label"),
            "use_amp": self.use_amp,
            "amp_dtype": str(self.amp_dtype).replace("torch.", ""),
            "use_tf32": bool(self.cfg.get("use_tf32", False)),
            "scaler_enabled": self.scaler_enabled,
            "fp32_islands": list(self.cfg.get("fp32_islands", ())),
            "device_type": self.device_type,
        }


# ---------------------------------------------------------------------------
# CPU-preflight helper: FP32-only, no AMP, no scaler.
# ---------------------------------------------------------------------------
def resolve_cpu_preflight_config() -> Dict[str, Any]:
    """Deterministic CPU-preflight config: pure FP32, no AMP, no TF32."""
    return {
        "device": torch.device("cpu"),
        "device_type": "cpu",
        "cuda_available": False,
        "requested_policy": PRECISION_FP32,
        "use_amp": False,
        "amp_dtype": "float32",
        "amp_dtype_torch": torch.float32,
        "use_tf32": False,
        "fallback_policy": DEFAULT_FALLBACK_POLICY,
        "fp32_islands": ("loss", "sigmoid", "metrics", "export"),
        "autocast_regions": (),
        "scaler_initialized": False,
        "precision_label": PRECISION_FP32,
        "reason": "cpu preflight -> fp32, amp and scaler disabled",
    }


def make_precision_runtime(
    device: Optional[torch.device] = None,
    override: Optional[Mapping[str, Any]] = None,
    cpu_preflight: bool = False,
) -> PrecisionRuntime:
    """Convenience constructor used by Stage 3's training entrypoint."""
    if cpu_preflight:
        runtime = PrecisionRuntime.__new__(PrecisionRuntime)
        cfg = resolve_cpu_preflight_config()
        runtime.cfg = dict(cfg)
        runtime.device = cfg["device"]
        runtime.device_type = cfg["device_type"]
        runtime.use_amp = False
        runtime.amp_dtype = torch.float32
        runtime.fallback_policy = cfg["fallback_policy"]
        runtime._autocast_available = False
        runtime.grad_scaler = None
        return runtime

    cfg = resolve_precision_config(device=device, override=override)
    return PrecisionRuntime(config=cfg)

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
 Summary: 3 total attempts - 2 pending execution, 1 executed, 0 successful (all failed or buggy)

-------------------------------
Attempt #1:
Design: I merged the three stages by keeping Stage 1's data/model/loss/adapter code as the foundation, layering Stage 2's import-safe precision policy (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `sigmoid_fp32`, `make_grad_scaler`, `to_fp32_policy`, `configure_tf32`, `fp32_island`) on top, and consuming both inside Stage 3's `run_training` (AdamW + cosine warmup, physical batch 32 × accum 2, val-F1 early stopping with best-state restore, scheduler safe-points and `TrainingDiagnostics`). I deduplicated imports/constants, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module-level constants referenced by Stage 3, preserved the CPU-safe `CandidateAdapter` contract, kept all side effects under `if __name__ == '__main__':`, and left the final `Final Validation Score:` print after the diagnostics context.
Results: The implementation of this design has bugs.
Insight: Using a different approach may not result in the same bugs as the above approach.

-------------------------------
Attempt #2:
Design: I merged the three stages by keeping Stage 1's data cleaning, TF-IDF feature builders, `SparseMLP`, `build_loss`, `tune_threshold` and split helpers as the foundation; layering Stage 2's import-safe precision utilities (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `degrade_to_fp32`, `configure_tf32`, `format_precision_log`) on top; then consuming both inside Stage 3's `run_training` with scheduler safe-points, `TrainingDiagnostics`, linear warmup+cosine AdamW, best-val-F1 restore, and test inference. I resolved the duplicate `CandidateAdapter` by keeping Stage 3's self-contained CPU-safe version (it uses the Stage 1 model/loss/features through the Stage 2 precision path), removed duplicate imports/constants, kept all side effects under `if __name__ == "__main__":`, and preserved the exact `MODEL_FAMILY`, `WORKING_DIR`, `SUBMISSION_DIR` contract.
Status: Code generated, execution pending (will run in parallel with other drafts).

-------------------------------
Attempt #3:
Design: I merged the three stages by keeping Stage 1's data/model/loss/adapter code as the foundation, layering Stage 2's `PrecisionRuntime` policy helpers on top, and consuming both inside Stage 3's `run_training` (with diagnostics + scheduler safe points). I resolved naming conflicts, deduplicated imports, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module-level constants that Stage 3 references, and moved every side effect under `if __name__ == '__main__':` while preserving the CPU-safe `CandidateAdapter` contract.
Status: Code generated, execution pending (will run in parallel with other drafts).


# Hardware-Aware Stage 1 Candidate Construction Context
# Hardware-Aware Model Design Brief
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2
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
- Workload: transformer_training

# Hardware-Aware Datatype/Precision Context
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2
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
- Stage-filtered hardware knowledge:

# Hardware-Aware Training Hyperparameter Context
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2
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
- Confidence: 0.0
adamw_params = []



# Pipeline Decision Contract
Required hardware-aware step order: model-design -> datatype/quantization -> training (model_design -> datatype_precision -> training_evaluation). Use the trace as the source of truth, apply hardware evidence only when compatible, and use recorded fallbacks when evidence is missing.

# Pipeline Decision Trace
```json
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"{'criterion': 'BCEWithLogitsLoss', 'pos_weight': None, 'notes': 'Compute loss in FP32 island for...","output_interface":"{'raw_output': 'logits', 'probability': 'sigmoid(logits)', 'prediction_threshold': 0.5, 'threshol...","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"{'forward_autocast': {'enabled': True, 'backend': 'torch.amp.autocast', 'device_type': 'cuda', 'd...","fallback_policy":"fp32","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"{'name': 'AdamW', 'lr': 2e-05, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-08, 'notes'...","scheduler":"{'name': 'linear_with_warmup', 'warmup_ratio': 0.1, 'notes': 'Linear decay to zero after warmup.'}","batch_size_policy":"fixed","dataloader_policy":"safe defaults; enable workers and pinned memory only when compatible","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available","no exact or similar graph profiles for current hardware","no verified speedup measurement for FP16 AMP on this task"]}}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=replaced sparse-TF-IDF MLP and vanilla transformer baselines with a BiGRU + additive-attention encoder that injects a learned keyword embedding as an additive bias into every word embedding; purpose=exploit word order and keyword context jointly for short, noisy disaster tweets under strict F1 evaluation; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=introduced explicit `prepare_text_field(text, keyword, location)` fusion that prepends location tokens and appends keyword tokens into the sequence, plus a separate categorical keyword id stream; purpose=preserve location as positional context while keeping keyword as a distinct, low-cardinality conditioning signal usable by Stage 2/3 without re-tokenization; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=exposed a CPU-safe `CandidateAdapter` that constructs real vocab + real model + real `BCEWithLogitsLoss`, plus an F1 threshold-tuner helper (`tune_threshold`) that mirrors the metric used at export; purpose=allow Stage 3 to consume the same model/loss/threshold path used for validation and test inference; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=introduced `PrecisionRuntime` + helpers (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `sigmoid_fp32`, `fp32_island`, `make_grad_scaler`, `configure_tf32`, `format_precision_log`); purpose=give Stage 3 a single FP16-AMP-with-FP32-islands contract that keeps the BiGRU+keyword-bias forward stable while preserving the metric-aligned sigmoid/threshold export path; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=made every precision action a callable (no CUDA/TF32 flags set at import) and forced the CPU preflight path to FP32 with `scaler=None`; purpose=satisfy the import-safe contract and keep the CandidateAdapter usable on a CPU-only checker without AMP overhead; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=kept device/dtype configuration as resolved-at-runtime metadata (`device`, `use_amp`, `amp_dtype`, `use_tf32`, `grad_scaler`) rather than module globals; purpose=let Stage 3 choose device inside its main guard without coupling to a single hardcoded trainer; hardware_keys=none
  - Stage 3 training / training_evaluation: baseline_change=Generated this stage's implementation.; purpose=Train and evaluate the Stage 1 candidate using the selected precision policy.; hardware_keys=none

# Step results

        ### Step 1: model_design
        **Plan:** ## Plan

**WHAT makes this task unique**: Disaster tweets are short, noisy, and mix word order with strong keyword-level signals (e.g., "wildfire", "earthquake"); the label depends on whether the words describe a *real* disaster versus a metaphor, so sequence order and keyword context interact non-trivially. Prior baseline approaches in memory relied on bag-of-ngram sparse MLP or plain transformer towers, which either lose order signal or overfit at 6k examples.

**WHY this approach fits**: I'll build a compact hybrid **BiGRU + additive attention with token-level keyword-bias injection**: a trainable keyword embedding is added to every word embedding (a "keyword conditioning" bias) so attention can learn keyword-specific saliency, and a pooled keyword-only branch is concatenated with the sequence attention vector. This uses the strongly predictive `keyword` field as a *learned bias inside* the sequence encoder rather than a side feature, which is fundamentally different from the sparse-MLP (`TF-IDF + MLP`) and vanilla-transformer candidates in memory.

**HOW to validate**: Training/validation split is done first (stratified on target, seeded); vocabularies and the keyword map are fit only on the training fold. Confirmation = val F1 clearly above a constant-baseline and stable across epochs, plus monotone improvement as the keyword bias is enabled vs. disabled (ablated later). Refutation = F1 stuck near ~0.5 or strong train/val gap, indicating overfitting of the GRU/attention head and a need to shrink capacity or add stronger regularization.

**Memory strategy / batch elasticity**: Batches are built as (word_ids, keyword_ids, target) tuples with on-the-fly padding via `collate_batch`; the adapter honors `scenario['batch_size']` verbatim and never buffers global state inside the module. Physical vs. effective batch elasticity (accumulation) is a **training-loop decision** and is deliberately not encoded here. **Operator / normalization considerations**: `nn.GRU` + `tanh`-scored additive attention + ReLU MLP head; LayerNorm is *not* applied inside the recurrent block (only dropout), keeping the model stable under FP32/FP16-AMP. **Configurable dimensions**: `EMBED_DIM_DEFAULT`, `GRU_HIDDEN_DEFAULT`, `GRU_LAYERS_DEFAULT`, `DROPOUT_DEFAULT`, `MAX_SEQ_LEN_DEFAULT`, `WORD_VOCAB_SIZE_DEFAULT`, `KEYWORD_VOCAB_SIZE_DEFAULT` are constants that Stage 3 may override. **Fallback behavior**: if keyword/location columns are missing or NaN, they map to `<none>` token; if text is empty, sequence becomes `[UNK]`. **Scheduler-owned controls not implemented here**: device assignment, CUDA/TF32 policy, AMP precision, worker counts, optimizer LR/weight decay, epochs, checkpointing cadence — none are chosen in this file.

---

Note board:
- baseline_change: replaced sparse-TF-IDF MLP and vanilla transformer baselines with a BiGRU + additive-attention encoder that injects a learned keyword embedding as an additive bias into every word embedding | purpose: exploit word order and keyword context jointly for short, noisy disaster tweets under strict F1 evaluation | hardware_keys: none
- baseline_change: introduced explicit `prepare_text_field(text, keyword, location)` fusion that prepends location tokens and appends keyword tokens into the sequence, plus a separate categorical keyword id stream | purpose: preserve location as positional context while keeping keyword as a distinct, low-cardinality conditioning signal usable by Stage 2/3 without re-tokenization | hardware_keys: none
- baseline_change: exposed a CPU-safe `CandidateAdapter` that constructs real vocab + real model + real `BCEWithLogitsLoss`, plus an F1 threshold-tuner helper (`tune_threshold`) that mirrors the metric used at export | purpose: allow Stage 3 to consume the same model/loss/threshold path used for validation and test inference | hardware_keys: none
        **Code:**
        ```python
"""Stage 1: data prep, features, model architecture, loss and output interface.

Model family: bilstm_attention_keyword_bias_v1
Input representation: cleaned word sequence (with location prepended and
keyword appended) + separate categorical keyword id stream.

Nothing in this module touches data or CUDA at import time. All side effects
(loading CSVs, building vocab, training, inference, submission) belong in
`if __name__ == "__main__":` of the final integrated pipeline.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MODEL_FAMILY = "bilstm_attention_keyword_bias_v1"

WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
INPUT_DIR_DEFAULT = "./input"

WORD_VOCAB_SIZE_DEFAULT = 20000
KEYWORD_VOCAB_SIZE_DEFAULT = 512
MAX_SEQ_LEN_DEFAULT = 48
EMBED_DIM_DEFAULT = 128
GRU_HIDDEN_DEFAULT = 128
GRU_LAYERS_DEFAULT = 2
DROPOUT_DEFAULT = 0.3

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
NONE_KEYWORD = "<none>"

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9#']+")
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Text cleaning / tokenization helpers
# ---------------------------------------------------------------------------
def clean_text(text: Any) -> str:
    """Lowercase, strip URLs/mentions/HTML entities and non-alphanumerics."""
    if text is None:
        return ""
    s = str(text).lower()
    s = _URL_RE.sub(" ", s)
    s = _MENTION_RE.sub(" ", s)
    s = s.replace("&amp;", " and ")
    s = s.replace("&lt;", " ").replace("&gt;", " ")
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def tokenize(text: Any) -> List[str]:
    s = clean_text(text)
    if not s:
        return []
    return [t for t in s.split(" ") if t]


def prepare_text_field(text: Any, keyword: Any, location: Any) -> str:
    """Fuse location + text + keyword into a single sequence string.

    Keyword is placed at the end so the recurrent encoder sees the topic hint
    after the main utterance, and keyword ids are also passed separately.
    """
    parts: List[str] = []
    if location is not None and not (
        isinstance(location, float) and np.isnan(location)
    ):
        loc = str(location).strip().replace("%20", " ")
        if loc and loc.lower() != "nan":
            parts.append(loc)
    if text is not None:
        t = str(text)
        if t and t.lower() != "nan":
            parts.append(t)
    if keyword is not None and not (isinstance(keyword, float) and np.isnan(keyword)):
        kw = str(keyword).strip().replace("%20", " ")
        if kw and kw.lower() != "nan":
            parts.append(kw)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Vocabulary builders (fit on the training split only)
# ---------------------------------------------------------------------------
def build_word_vocab(
    texts: List[str], max_size: int = WORD_VOCAB_SIZE_DEFAULT, min_freq: int = 1
) -> Dict[str, int]:
    counter: Dict[str, int] = {}
    for t in texts:
        for tok in tokenize(t):
            counter[tok] = counter.get(tok, 0) + 1
    items = [(w, c) for w, c in counter.items() if c >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    items = items[: max(0, max_size - 2)]
    vocab: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for w, _ in items:
        vocab[w] = len(vocab)
    return vocab


def build_keyword_vocab(
    keywords: List[Any], max_size: int = KEYWORD_VOCAB_SIZE_DEFAULT
) -> Dict[str, int]:
    vocab: Dict[str, int] = {PAD_TOKEN: 0, NONE_KEYWORD: 1}
    seen = set()
    for kw in keywords:
        if kw is None:
            continue
        if isinstance(kw, float) and np.isnan(kw):
            continue
        k = str(kw).strip().lower().replace("%20", " ")
        if not k or k == "nan":
            continue
        seen.add(k)
    for k in sorted(seen)[: max(0, max_size - 2)]:
        vocab[k] = len(vocab)
    return vocab


def encode_tokens(tokens: List[str], vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens]
    if len(ids) > max_len:
        ids = ids[:max_len]
    if len(ids) == 0:
        ids = [vocab[UNK_TOKEN]]
    return ids


def keyword_to_id(kw: Any, vocab: Dict[str, int]) -> int:
    if kw is None:
        return vocab[NONE_KEYWORD]
    if isinstance(kw, float) and np.isnan(kw):
        return vocab[NONE_KEYWORD]
    k = str(kw).strip().lower().replace("%20", " ")
    if not k or k == "nan":
        return vocab[NONE_KEYWORD]
    return vocab.get(k, vocab[UNK_TOKEN])


# ---------------------------------------------------------------------------
# Dataset / collate
# ---------------------------------------------------------------------------
class TweetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        word_ids: List[List[int]],
        keyword_ids: List[int],
        targets: Optional[List[float]] = None,
    ) -> None:
        self.word_ids = word_ids
        self.keyword_ids = keyword_ids
        self.targets = targets

    def __len__(self) -> int:
        return len(self.word_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "word_ids": torch.tensor(self.word_ids[idx], dtype=torch.long),
            "keyword_ids": torch.tensor(int(self.keyword_ids[idx]), dtype=torch.long),
        }
        if self.targets is not None:
            item["target"] = torch.tensor(float(self.targets[idx]), dtype=torch.float32)
        return item


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_len = max(int(b["word_ids"].shape[0]) for b in batch)
    word_rows = []
    for b in batch:
        w = b["word_ids"]
        if w.shape[0] < max_len:
            pad = torch.zeros(max_len - w.shape[0], dtype=torch.long)
            w = torch.cat([w, pad], dim=0)
        word_rows.append(w)
    out = {
        "word_ids": torch.stack(word_rows, dim=0),
        "keyword_ids": torch.stack([b["keyword_ids"] for b in batch], dim=0),
    }
    if "target" in batch[0]:
        out["target"] = torch.stack([b["target"] for b in batch], dim=0)
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DisasterHybridModel(nn.Module):
    """BiGRU + additive attention with keyword-bias injection.

    - `keyword_emb[keyword_ids]` is added to every word embedding in the
      sequence (learned topic conditioning).
    - Additive attention pools the sequence output into a fixed vector.
    - A separate projected keyword feature is concatenated before the MLP head.
    """

    def __init__(
        self,
        word_vocab_size: int,
        keyword_vocab_size: int,
        embed_dim: int = EMBED_DIM_DEFAULT,
        gru_hidden: int = GRU_HIDDEN_DEFAULT,
        gru_layers: int = GRU_LAYERS_DEFAULT,
        dropout: float = DROPOUT_DEFAULT,
        pad_index: int = 0,
    ) -> None:
        super().__init__()
        self.word_vocab_size = int(word_vocab_size)
        self.keyword_vocab_size = int(keyword_vocab_size)
        self.embed_dim = int(embed_dim)
        self.gru_hidden = int(gru_hidden)
        self.gru_layers = int(gru_layers)

        self.word_emb = nn.Embedding(
            self.word_vocab_size, self.embed_dim, padding_idx=pad_index
        )
        self.keyword_emb = nn.Embedding(
            self.keyword_vocab_size, self.embed_dim, padding_idx=pad_index
        )
        nn.init.normal_(self.word_emb.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.keyword_emb.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.word_emb.weight[pad_index].zero_()
            self.keyword_emb.weight[pad_index].zero_()

        self.word_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.gru_hidden,
            num_layers=self.gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if self.gru_layers > 1 else 0.0,
        )

        attn_dim = 2 * self.gru_hidden
        self.attn_proj = nn.Sequential(
            nn.Linear(attn_dim, max(32, attn_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(32, attn_dim // 2), 1),
        )

        self.keyword_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(attn_dim + self.embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self, word_ids: torch.Tensor, keyword_ids: torch.Tensor
    ) -> torch.Tensor:
        if word_ids.dim() == 1:
            word_ids = word_ids.unsqueeze(0)
        if keyword_ids.dim() == 0:
            keyword_ids = keyword_ids.unsqueeze(0)
        if keyword_ids.dim() == 1:
            keyword_ids = keyword_ids

        mask = word_ids != 0  # [B, L]
        word_vec = self.word_emb(word_ids)  # [B, L, D]
        kw_vec = self.keyword_emb(keyword_ids)  # [B, D]

        seq = word_vec + kw_vec.unsqueeze(1)  # broadcast topic bias
        seq = self.word_dropout(seq)

        gru_out, _ = self.gru(seq)  # [B, L, 2H]
        scores = self.attn_proj(gru_out).squeeze(-1)  # [B, L]
        scores = scores.masked_fill(~mask, -1e4)
        alpha = torch.softmax(scores, dim=-1)  # [B, L]
        pooled = torch.einsum("bl,blh->bh", alpha, gru_out)  # [B, 2H]

        kw_feat = self.keyword_proj(kw_vec)  # [B, D]
        x = torch.cat([pooled, kw_feat], dim=-1)  # [B, 2H + D]
        logits = self.head(x).squeeze(-1)  # [B]
        return logits


# ---------------------------------------------------------------------------
# Loss, output interface, metric-aligned threshold helper
# ---------------------------------------------------------------------------
def build_loss(pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
    """Binary-cross-entropy on logits. `pos_weight` may be supplied by Stage 3."""
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def sigmoid_logits(logits: torch.Tensor) -> torch.Tensor:
    """Output interface: probability = sigmoid(model logits)."""
    return torch.sigmoid(logits)


def tune_threshold(probs: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """Convenience F1 sweep used at validation/export time. Returns (best_t, best_f1)."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs >= t).astype(np.int64)
        tp = int(((preds == 1) & (targets == 1)).sum())
        fp = int(((preds == 1) & (targets == 0)).sum())
        fn = int(((preds == 0) & (targets == 1)).sum())
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


# ---------------------------------------------------------------------------
# CPU preflight adapter
# ---------------------------------------------------------------------------
class CandidateAdapter:
    """No-argument adapter exercising the real model/loss/inputs on CPU.

    The adapter lazily materializes a small slice of train.csv so its CPU
    construction is cheap and side-effect free at import time.
    """

    PREFLIGHT_ROWS = 256

    def __init__(self) -> None:
        self._cache: Optional[Dict[str, Any]] = None

    # -- internal -----------------------------------------------------------
    def _ensure_data(self) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache
        input_dir = os.environ.get("MLEVOLVE_INPUT_DIR", INPUT_DIR_DEFAULT)
        train_path = os.path.join(input_dir, "train.csv")
        df = pd.read_csv(train_path).head(self.PREFLIGHT_ROWS).copy()

        text_col = df["text"].tolist() if "text" in df.columns else [""] * len(df)
        keyword_col = (
            df["keyword"].tolist() if "keyword" in df.columns else [None] * len(df)
        )
        location_col = (
            df["location"].tolist() if "location" in df.columns else [None] * len(df)
        )
        target_col = (
            df["target"].astype(float).tolist() if "target" in df.columns else None
        )

        fused_text = [
            prepare_text_field(t, k, l)
            for t, k, l in zip(text_col, keyword_col, location_col)
        ]

        word_vocab = build_word_vocab(fused_text, WORD_VOCAB_SIZE_DEFAULT, min_freq=1)
        keyword_vocab = build_keyword_vocab(keyword_col, KEYWORD_VOCAB_SIZE_DEFAULT)

        word_ids = [
            encode_tokens(tokenize(t), word_vocab, MAX_SEQ_LEN_DEFAULT)
            for t in fused_text
        ]
        kw_ids = [keyword_to_id(k, keyword_vocab) for k in keyword_col]

        self._cache = {
            "word_vocab": word_vocab,
            "keyword_vocab": keyword_vocab,
            "word_ids": word_ids,
            "keyword_ids": kw_ids,
            "targets": target_col,
        }
        return self._cache

    @staticmethod
    def _merge_context(context: Any, defaults: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(defaults)
        if isinstance(context, dict):
            for k, v in context.items():
                if v is not None:
                    merged[k] = v
        return merged

    def _make_batch(self, batch_size: int, device: torch.device) -> Dict[str, Any]:
        cache = self._ensure_data()
        n = len(cache["word_ids"])
        bs = max(1, min(int(batch_size), n))
        idxs = list(range(bs))

        word_ids = torch.stack(
            [torch.tensor(cache["word_ids"][i], dtype=torch.long) for i in idxs], dim=0
        )
        keyword_ids = torch.stack(
            [
                torch.tensor(int(cache["keyword_ids"][i]), dtype=torch.long)
                for i in idxs
            ],
            dim=0,
        )
        if cache["targets"] is not None:
            targets = torch.tensor(
                [float(cache["targets"][i]) for i in idxs], dtype=torch.float32
            )
        else:
            targets = torch.zeros(bs, dtype=torch.float32)

        return {
            "inputs": (word_ids.to(device), keyword_ids.to(device)),
            "target": targets.to(device),
        }

    # -- contract -----------------------------------------------------------
    def build_model(self, context: Any) -> nn.Module:
        cache = self._ensure_data()
        model = DisasterHybridModel(
            word_vocab_size=len(cache["word_vocab"]),
            keyword_vocab_size=len(cache["keyword_vocab"]),
            embed_dim=EMBED_DIM_DEFAULT,
            gru_hidden=GRU_HIDDEN_DEFAULT,
            gru_layers=GRU_LAYERS_DEFAULT,
            dropout=DROPOUT_DEFAULT,
        )
        return model

    def build_optimizer(self, model: nn.Module, context: Any) -> torch.optim.Optimizer:
        # Stage 1 does not choose the real optimizer policy; this default only
        # exists so the CPU preflight can call training_step end-to-end.
        from torch.optim import AdamW

        merged = self._merge_context(context, {"lr": 1e-3, "weight_decay": 0.0})
        return AdamW(
            model.parameters(),
            lr=float(merged["lr"]),
            weight_decay=float(merged["weight_decay"]),
        )

    def build_train_batch(self, scenario: Any, device: torch.device) -> Dict[str, Any]:
        bs = 4
        if isinstance(scenario, dict) and scenario.get("batch_size") is not None:
            bs = int(scenario["batch_size"])
        return self._make_batch(bs, device)

    def build_validation_batch(
        self, scenario: Any, device: torch.device
    ) -> Dict[str, Any]:
        bs = 4
        if isinstance(scenario, dict) and scenario.get("batch_size") is not None:
            bs = int(scenario["batch_size"])
        return self._make_batch(bs, device)

    def training_step(
        self, model: nn.Module, batch: Dict[str, Any], context: Any
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        target = batch["target"]
        logits = model(*inputs)
        criterion = build_loss()
        loss = criterion(logits, target)
        return loss

    def validation_step(
        self, model: nn.Module, batch: Dict[str, Any], context: Any
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        target = batch["target"]
        logits = model(*inputs)
        criterion = build_loss()
        loss = criterion(logits, target)
        return loss

```
        
        ### Step 2: datatype_precision
        **Plan:** ## Plan

This step locks in the Stage 2 precision policy decided by the pipeline trace (`fp16_amp` with `fp32` fallback) and wires it as import-safe helpers the Stage 3 trainer can consume. Because the Stage 1 model is a BiGRU + additive attention with LayerNorm-free recurrent block, FP16 AMP is safe for the forward path (embedding lookup, GRU, attention, MLP head) with the loss computed in a local FP32 island and sigmoid/reductions kept in FP32 for F1 stability. TF32 is explicitly disabled for matmul and convolution, GradScaler is created only when AMP is enabled on CUDA, and the CPU preflight path force-disables AMP/scaling; everything is a helper so no side effects run at import time.

Note board:
- baseline_change: introduced `PrecisionRuntime` + helpers (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `sigmoid_fp32`, `fp32_island`, `make_grad_scaler`, `configure_tf32`, `format_precision_log`) | purpose: give Stage 3 a single FP16-AMP-with-FP32-islands contract that keeps the BiGRU+keyword-bias forward stable while preserving the metric-aligned sigmoid/threshold export path | hardware_keys: none
- baseline_change: made every precision action a callable (no CUDA/TF32 flags set at import) and forced the CPU preflight path to FP32 with `scaler=None` | purpose: satisfy the import-safe contract and keep the CandidateAdapter usable on a CPU-only checker without AMP overhead | hardware_keys: none
- baseline_change: kept device/dtype configuration as resolved-at-runtime metadata (`device`, `use_amp`, `amp_dtype`, `use_tf32`, `grad_scaler`) rather than module globals | purpose: let Stage 3 choose device inside its main guard without coupling to a single hardcoded trainer | hardware_keys: none
        **Code:**
        ```python
"""Stage 2: datatype / precision policy for the Stage 1 BiGRU+keyword-bias model.

Chosen policy (per Pipeline Decision Trace): ``fp16_amp`` with ``fp32`` fallback.
- FP32 is the storage dtype for parameters, floating inputs and floating targets.
- FP16 autocast is used ONLY around the forward pass (embedding -> GRU ->
  attention -> MLP head) on CUDA. It is never applied to Backward or optimizer.
- The loss and every F1-relevant reduction (sigmoid, metric accumulation,
  threshold sweep, export) run inside an explicit FP32 island.
- TF32 is disabled for matmul and convolution.
- GradScaler is created only when AMP is actually enabled on CUDA.
- CPU preflight is FP32 with AMP and scaling disabled.

Nothing in this module touches CUDA, TF32 state, data, or models at import time.
All configuration is exposed as helpers so the caller (or the CPU preflight
adapter) can opt in explicitly inside its own main guard.
"""

import contextlib
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn

# ---- import-safe autocast / GradScaler shims (both APIs supported) ----------
try:  # PyTorch >= 2.4 style
    from torch.amp import autocast as _torch_autocast
    from torch.amp import GradScaler as _TorchGradScaler

    _AMP_API = "torch.amp"
except ImportError:  # pragma: no cover - fallback for older torch
    from torch.cuda.amp import autocast as _torch_autocast  # type: ignore
    from torch.cuda.amp import GradScaler as _TorchGradScaler  # type: ignore

    _AMP_API = "torch.cuda.amp"


# ---------------------------------------------------------------------------
# Constants (import-safe; no CUDA queries executed at import time)
# ---------------------------------------------------------------------------
MODEL_FAMILY = "bilstm_attention_keyword_bias_v1"

# Defaults chosen by the Pipeline Decision Trace for Stage 2.
DEFAULT_USE_AMP = True
DEFAULT_AMP_DTYPE = "float16"
DEFAULT_USE_TF32 = False
DEFAULT_FALLBACK_POLICY = "fp32"
DEFAULT_SCALER_INIT_SCALE = 2.0**16
DEFAULT_SCALER_GROWTH_INTERVAL = 2000
DEFAULT_SCALER_BACKOFF_FACTOR = 0.5
DEFAULT_SCALER_GROWTH_FACTOR = 2.0

# Named precision labels used in logs (kept stable for downstream parsing).
PRECISION_FP32 = "fp32"
PRECISION_FP16_AMP = "fp16_amp"


# ---------------------------------------------------------------------------
# Device / TF32 helpers (call-only; no import-time side effects)
# ---------------------------------------------------------------------------
def resolve_device(prefer_cuda: bool = True) -> torch.device:
    """Return the compute device to use. Caller decides when to invoke it."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_tf32(enabled: bool = DEFAULT_USE_TF32) -> Dict[str, bool]:
    """Explicitly set TF32 flags. Returns the resulting state for logging.

    In normal mode we disable TF32 for both matmul and convolution; the trace
    marks TF32 as off, and FP16 AMP is the only accelerated path allowed here.
    """
    matmul_flag = bool(enabled)
    cudnn_flag = bool(enabled)
    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = matmul_flag
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = cudnn_flag
    except Exception:
        # Flag support depends on build; keep going without hard failure.
        pass
    return {"matmul_allow_tf32": matmul_flag, "cudnn_allow_tf32": cudnn_flag}


# ---------------------------------------------------------------------------
# FP32 islands
# ---------------------------------------------------------------------------
def fp32_island(device_type: Optional[str] = None):
    """Return a context manager that locally disables autocast.

    Use inside an ambient autocast block to compute numerically fragile ops
    (loss, reductions, sigmoid) in FP32. On CPU this is a no-op.
    """
    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if device_type == "cuda":
        try:
            return _torch_autocast(device_type="cuda", enabled=False)
        except TypeError:  # pragma: no cover - very old torch
            return _torch_autocast(enabled=False)
    return contextlib.nullcontext()


def to_fp32_policy(x: Any) -> Any:
    """Cast floating tensors to FP32; leave integer/index tensors untouched."""
    if torch.is_tensor(x):
        if x.is_floating_point() and x.dtype != torch.float32:
            return x.float()
        return x
    if isinstance(x, (list, tuple)):
        return type(x)(to_fp32_policy(v) for v in x)
    if isinstance(x, dict):
        return {k: to_fp32_policy(v) for k, v in x.items()}
    return x


# ---------------------------------------------------------------------------
# Precision-aware loss / output interface (mirror Stage 1 signatures)
# ---------------------------------------------------------------------------
def compute_bce_with_logits_fp32(
    criterion: Optional[nn.Module],
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy on logits, always computed in FP32.

    The Stage 1 criterion is `nn.BCEWithLogitsLoss`; if it is missing we fall
    back to a fresh instance so the CPU-preflight path never needs the caller
    to pre-materialize the loss.
    """
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    with fp32_island("cuda" if logits.is_cuda else "cpu"):
        logits_fp32 = logits.float() if logits.is_floating_point() else logits
        targets_fp32 = targets.float() if targets.is_floating_point() else targets
        loss = criterion(logits_fp32, targets_fp32)
    return loss


def sigmoid_fp32(logits: torch.Tensor) -> torch.Tensor:
    """Stage 1 output interface (probability = sigmoid(logits)) in FP32."""
    with fp32_island("cuda" if logits.is_cuda else "cpu"):
        logits_fp32 = logits.float() if logits.is_floating_point() else logits
        return torch.sigmoid(logits_fp32)


# ---------------------------------------------------------------------------
# GradScaler factory (only meaningful when AMP is enabled on CUDA)
# ---------------------------------------------------------------------------
def make_grad_scaler(
    enabled: bool,
    *,
    init_scale: float = DEFAULT_SCALER_INIT_SCALE,
    growth_interval: int = DEFAULT_SCALER_GROWTH_INTERVAL,
    backoff_factor: float = DEFAULT_SCALER_BACKOFF_FACTOR,
    growth_factor: float = DEFAULT_SCALER_GROWTH_FACTOR,
) -> Optional[Any]:
    """Create a GradScaler when AMP is enabled, else return None.

    The scaler is intentionally created WITHOUT a hardcoded device type so it
    remains compatible with both `torch.amp.GradScaler("cuda", ...)` and the
    legacy `torch.cuda.amp.GradScaler(...)` API.
    """
    if not enabled:
        return None
    kwargs = {
        "init_scale": float(init_scale),
        "growth_interval": int(growth_interval),
        "backoff_factor": float(backoff_factor),
        "growth_factor": float(growth_factor),
    }
    try:
        return _TorchGradScaler("cuda", **kwargs)
    except TypeError:
        # Legacy signature (no device_type positional argument).
        return _TorchGradScaler(**kwargs)


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------
def resolve_precision_config(
    device: Optional[torch.device] = None,
    override: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve the effective precision policy for the current run.

    Never raises on unsupported hardware: on CPU or when CUDA is unavailable
    we silently fall back to pure FP32 with `scaler=None`, matching the
    ``fallback_policy = fp32`` entry of the Pipeline Decision Trace.
    """
    if device is None:
        device = resolve_device(prefer_cuda=True)
    device = torch.device(device)

    cfg: Dict[str, Any] = {
        "device": device,
        "device_type": device.type,
        "cuda_available": bool(torch.cuda.is_available()),
        "requested_policy": PRECISION_FP16_AMP,
        "use_amp": DEFAULT_USE_AMP,
        "amp_dtype": DEFAULT_AMP_DTYPE,
        "amp_dtype_torch": torch.float16,
        "use_tf32": DEFAULT_USE_TF32,
        "fallback_policy": DEFAULT_FALLBACK_POLICY,
        "fp32_islands": ("loss", "sigmoid", "metrics", "export"),
        "autocast_regions": ("forward",),
        "scaler_initialized": False,
        "precision_label": PRECISION_FP16_AMP,
        "reason": "trace: fp16_amp on cuda with fp32 loss/metric/export islands",
    }

    if override:
        for k, v in override.items():
            if v is not None:
                cfg[k] = v

    # AMP is only valid on CUDA. Everything else falls back to FP32.
    if device.type != "cuda" or not cfg["cuda_available"]:
        cfg["use_amp"] = False
        cfg["amp_dtype"] = "float32"
        cfg["amp_dtype_torch"] = torch.float32
        cfg["precision_label"] = PRECISION_FP32
        cfg["reason"] = "cpu or cuda unavailable -> fp32 fallback"

    # TF32 stays disabled unless a caller explicitly overrides it (normal mode
    # does not permit TF32).
    cfg["tf32_state"] = {"matmul_allow_tf32": False, "cudnn_allow_tf32": False}
    return cfg


# ---------------------------------------------------------------------------
# Lightweight precision logger
# ---------------------------------------------------------------------------
def format_precision_log(cfg: Mapping[str, Any]) -> str:
    """One-line precision summary; safe to print once per run."""
    label = cfg.get("precision_label", PRECISION_FP32)
    device_type = cfg.get("device_type", "cpu")
    use_amp = bool(cfg.get("use_amp", False))
    amp_dtype = cfg.get("amp_dtype", "float32")
    use_tf32 = bool(cfg.get("use_tf32", False))
    scaler = "on" if cfg.get("scaler_initialized") else "off"
    reason = cfg.get("reason", "")
    return (
        f"[precision] policy={label} device={device_type} amp={use_amp} "
        f"amp_dtype={amp_dtype} tf32={use_tf32} scaler={scaler} "
        f"fp32_islands={','.join(cfg.get('fp32_islands', ()))} reason={reason}"
    )


# ---------------------------------------------------------------------------
# Runtime wrapper
# ---------------------------------------------------------------------------
class PrecisionRuntime:
    """Encapsulates device + AMP + scaler behavior for the Stage 3 trainer.

    The object is intentionally passive: it does not modify TF32 state at
    construction (callers do that explicitly via `configure_tf32`) and it does
    not own any data. It only brokers forward/loss/sigmoid calls under the
    correct precision, leaving backward/optimizer/scaler.step to the caller.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        cfg = resolve_precision_config(override=config)
        self.cfg: Dict[str, Any] = dict(cfg)

        self.device: torch.device = self.cfg["device"]
        self.device_type: str = self.cfg["device_type"]
        self.use_amp: bool = bool(self.cfg.get("use_amp", False))
        self.amp_dtype: torch.dtype = self.cfg.get("amp_dtype_torch", torch.float32)
        self.fallback_policy: str = str(
            self.cfg.get("fallback_policy", DEFAULT_FALLBACK_POLICY)
        )

        self._autocast_available = self.device_type == "cuda"
        self.grad_scaler: Optional[Any] = make_grad_scaler(
            self.use_amp and self._autocast_available
        )
        self.cfg["scaler_initialized"] = self.grad_scaler is not None
        if self.grad_scaler is not None:
            self.cfg["precision_label"] = PRECISION_FP16_AMP
        else:
            # No scaler -> we are in a pure-FP32 path.
            if not self.use_amp:
                self.cfg["precision_label"] = PRECISION_FP32

    # -- autocast context ---------------------------------------------------
    def autocast(self):
        """Forward-only autocast context. No-op on CPU or when AMP is off."""
        if self.use_amp and self._autocast_available:
            try:
                return _torch_autocast(device_type="cuda", dtype=self.amp_dtype)
            except TypeError:  # pragma: no cover - very old torch
                return _torch_autocast(dtype=self.amp_dtype)
        return contextlib.nullcontext()

    # -- forward under autocast --------------------------------------------
    def autocast_forward(self, model: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
        """Run the model forward under the resolved AMP region.

        Parameters and inputs stay in FP32; autocast handles the internal casts.
        """
        with self.autocast():
            return model(*inputs)

    # -- FP32-sensitive outputs --------------------------------------------
    def compute_loss(
        self,
        criterion: Optional[nn.Module],
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return compute_bce_with_logits_fp32(criterion, logits, targets)

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return sigmoid_fp32(logits)

    # -- scaler conveniences ------------------------------------------------
    @property
    def scaler_enabled(self) -> bool:
        return self.grad_scaler is not None

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        if self.grad_scaler is None:
            return loss
        return self.grad_scaler.scale(loss)

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(optimizer)

    def scaler_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.grad_scaler is not None:
            self.grad_scaler.step(optimizer)
        else:
            optimizer.step()

    def scaler_update(self) -> None:
        if self.grad_scaler is not None:
            self.grad_scaler.update()

    # -- diagnostics --------------------------------------------------------
    def describe(self) -> str:
        return format_precision_log(self.cfg)

    def as_settings(self) -> Dict[str, Any]:
        """Compact dict suitable for TrainingDiagnostics settings."""
        return {
            "precision_label": self.cfg.get("precision_label"),
            "use_amp": self.use_amp,
            "amp_dtype": str(self.amp_dtype).replace("torch.", ""),
            "use_tf32": bool(self.cfg.get("use_tf32", False)),
            "scaler_enabled": self.scaler_enabled,
            "fp32_islands": list(self.cfg.get("fp32_islands", ())),
            "device_type": self.device_type,
        }


# ---------------------------------------------------------------------------
# CPU-preflight helper: FP32-only, no AMP, no scaler.
# ---------------------------------------------------------------------------
def resolve_cpu_preflight_config() -> Dict[str, Any]:
    """Deterministic CPU-preflight config: pure FP32, no AMP, no TF32."""
    return {
        "device": torch.device("cpu"),
        "device_type": "cpu",
        "cuda_available": False,
        "requested_policy": PRECISION_FP32,
        "use_amp": False,
        "amp_dtype": "float32",
        "amp_dtype_torch": torch.float32,
        "use_tf32": False,
        "fallback_policy": DEFAULT_FALLBACK_POLICY,
        "fp32_islands": ("loss", "sigmoid", "metrics", "export"),
        "autocast_regions": (),
        "scaler_initialized": False,
        "precision_label": PRECISION_FP32,
        "reason": "cpu preflight -> fp32, amp and scaler disabled",
    }


def make_precision_runtime(
    device: Optional[torch.device] = None,
    override: Optional[Mapping[str, Any]] = None,
    cpu_preflight: bool = False,
) -> PrecisionRuntime:
    """Convenience constructor used by Stage 3's training entrypoint."""
    if cpu_preflight:
        runtime = PrecisionRuntime.__new__(PrecisionRuntime)
        cfg = resolve_cpu_preflight_config()
        runtime.cfg = dict(cfg)
        runtime.device = cfg["device"]
        runtime.device_type = cfg["device_type"]
        runtime.use_amp = False
        runtime.amp_dtype = torch.float32
        runtime.fallback_policy = cfg["fallback_policy"]
        runtime._autocast_available = False
        runtime.grad_scaler = None
        return runtime

    cfg = resolve_precision_config(device=device, override=override)
    return PrecisionRuntime(config=cfg)

```
        
        ### Step 3: training_evaluation
        **Plan:** 
        **Code:**
        ```python
# Plan

**WHAT makes this unique:** Every prior memory attempt either built a sparse MLP/transformer baseline or merged three stages and broke at import/merge boundaries. My Stage 1 design replaced those with a BiGRU + additive-attention encoder where a learned keyword embedding is injected as an *additive bias into every word embedding* — this couples the high-signal `keyword` field into the sequence encoder itself instead of treating it as a side feature. That's the modeling hypothesis to preserve.

**WHY this training design fits:** The Stage 1 model is a small (~500K-param) sequence encoder with no LayerNorm inside the ReLU-style recurrent block, so a compact AdamW + linear-warmup + cosine schedule across 8 short epochs at effective batch 32 (physical 32, accum 1) is enough to converge without thrashing on ~5.2K training samples. FP16 AMP is applied *only* via the Stage 2 `PrecisionRuntime.autocast()` wrapper around the forward pass, with `compute_loss`, `probabilities`, metrics, and threshold tuning forced into FP32 islands — matching the trace's `fp16_amp + fp32 fallback` policy exactly and preventing FP16 sigmoid denormalization from corrupting the F1 surface.

**HOW to validate:** Stratified 85/15 split (seed 42) — vocab/feature construction is fit on the training fold only, validation is scored with `tune_threshold` (91-point sweep on the metric used at export), and the same threshold+sigmoid pipeline is applied to test inference. Confirmation: val F1 clearly above 0.6 with an early-stopped best epoch < planned. Refutation: val F1 stuck near 0.5 or a large train/val gap would mean the keyword-bias injection is overfitting the small training fold.

**Memory strategy:** pad-on-collate in-memory tensors, all caching inside `CandidateAdapter` only. **Physical/effective batch:** physical 32, accum 1, effective 32 (inside `QUALITY_SAFE_PHYSICAL_BATCH_SIZES`). **Operator/normalization:** dropout inside GRU only when layers>1, no LayerNorm in the recurrent block; ReLU+dropout head — stable under AMP autocast. **Configurable dims:** vocab sizes, EMBED_DIM, GRU_HIDDEN, GRU_LAYERS, seq length (constants from Stage 1). **Scheduler-owned controls (NOT implemented here):** launch, device assignment, thread counts, process priorities, memory limits.

Note board:
- baseline_change: wrote a full AdamW + linear-warmup-cosine training loop with stratified 85/15 split, val-F1 early stopping with best-state restore, and threshold tuning on the validation fold | purpose: preserve the Stage 1 BiGRU+keyword-bias hypothesis as the modeling target while satisfying the strict F1 evaluation | hardware_keys: none
- baseline_change: consumed Stage 2 `PrecisionRuntime` for forward-only FP16 autocast and FP32 islands for loss / sigmoid / metric accumulation / export, with GradScaler path used only through `runtime.scaler_step`/`runtime.scaler_update` | purpose: keep the Stage 2 fp16_amp-with-fp32-islands contract intact including the CPU-preflight fp32 fallback | hardware_keys: none
- baseline_change: added `TrainingDiagnostics` wrapper (with graceful fallback), `script_scheduler_context` STEP/EPOCH/BEFORE_TRAIN safe points with a complete state_factory, and captured `QUALITY_SAFE_PHYSICAL_BATCH_SIZES` + `BATCH_LR_SCALING_POLICY` constants | purpose: make the candidate scheduler-compatible for hardware-aware batching while keeping effective batch and data exposure fixed | hardware_keys: none

```python
"""Stage 3: training loop, validation, threshold tuning, test inference, submission.

This file is self-contained: it defines the Stage 1 data/model/loss/interface and
the Stage 2 precision helpers locally (so the merged artifact runs as one
subprocess), then the Stage 3 training entrypoint.

Import-safe: no data loading, CUDA/TF32 configuration, model construction,
DataLoader creation, training, evaluation or submission writing happens at
module import time — everything runs inside ``if __name__ == "__main__":``.
"""

import contextlib
import copy
import math
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Module-level constants (import-safe)
# ---------------------------------------------------------------------------
MODEL_FAMILY = "bilstm_attention_keyword_bias_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
INPUT_DIR_DEFAULT = "./input"

# ---------------------------------------------------------------------------
# Stage 3 scheduler/HW contracts
# ---------------------------------------------------------------------------
QUALITY_SAFE_PHYSICAL_BATCH_SIZES = [16, 32, 64]
BATCH_LR_SCALING_POLICY = "fixed"  # lr does not scale with physical batch
SPLIT_SEED = 42
VAL_FRACTION = 0.15

# ---------------------------------------------------------------------------
# Stage 1 constants
# ---------------------------------------------------------------------------
WORD_VOCAB_SIZE_DEFAULT = 20000
KEYWORD_VOCAB_SIZE_DEFAULT = 512
MAX_SEQ_LEN_DEFAULT = 48
EMBED_DIM_DEFAULT = 128
GRU_HIDDEN_DEFAULT = 128
GRU_LAYERS_DEFAULT = 2
DROPOUT_DEFAULT = 0.3

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
NONE_KEYWORD = "<none>"

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9#']+")
_WS_RE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Stage 2 AMP imports (import-safe)
# ---------------------------------------------------------------------------
try:  # PyTorch >= 2.4
    from torch.amp import autocast as _torch_autocast
    from torch.amp import GradScaler as _TorchGradScaler

    _AMP_API = "torch.amp"
except ImportError:  # pragma: no cover
    from torch.cuda.amp import autocast as _torch_autocast  # type: ignore
    from torch.cuda.amp import GradScaler as _TorchGradScaler  # type: ignore

    _AMP_API = "torch.cuda.amp"

DEFAULT_USE_AMP = True
DEFAULT_AMP_DTYPE = "float16"
DEFAULT_USE_TF32 = False
DEFAULT_FALLBACK_POLICY = "fp32"
PRECISION_FP32 = "fp32"
PRECISION_FP16_AMP = "fp16_amp"

# ---------------------------------------------------------------------------
# Optional TrainingDiagnostics import (Stage 3 contract) with safe fallback
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from utils.training_diagnostics import TrainingDiagnostics  # type: ignore
except Exception:  # pragma: no cover
    class TrainingDiagnostics:  # type: ignore
        """No-op fallback used when the scheduler diagnostics helper is absent."""

        def __init__(self, model=None, optimizer=None, scaler=None, settings=None):
            self.model = model
            self.optimizer = optimizer
            self.scaler = scaler
            self.settings = dict(settings or {})
            self._reports: List[Dict[str, Any]] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def after_update(self):
            return None

        def report(self, epoch=1):
            self._reports.append({"epoch": int(epoch)})
            return None

        def state_dict(self):
            return {"reports": list(self._reports)}

        def load_state_dict(self, state):
            if isinstance(state, dict):
                self._reports = list(state.get("reports", []))


# ===========================================================================
# Stage 1: text cleaning / vocab / dataset / model / loss / threshold tuner
# ===========================================================================
def clean_text(text: Any) -> str:
    if text is None:
        return ""
    s = str(text).lower()
    s = _URL_RE.sub(" ", s)
    s = _MENTION_RE.sub(" ", s)
    s = s.replace("&amp;", " and ").replace("&lt;", " ").replace("&gt;", " ")
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def tokenize(text: Any) -> List[str]:
    s = clean_text(text)
    if not s:
        return []
    return [t for t in s.split(" ") if t]


def prepare_text_field(text: Any, keyword: Any, location: Any) -> str:
    parts: List[str] = []
    if location is not None and not (isinstance(location, float) and np.isnan(location)):
        loc = str(location).strip().replace("%20", " ")
        if loc and loc.lower() != "nan":
            parts.append(loc)
    if text is not None:
        t = str(text)
        if t and t.lower() != "nan":
            parts.append(t)
    if keyword is not None and not (isinstance(keyword, float) and np.isnan(keyword)):
        kw = str(keyword).strip().replace("%20", " ")
        if kw and kw.lower() != "nan":
            parts.append(kw)
    return " ".join(parts)


def build_word_vocab(texts, max_size=WORD_VOCAB_SIZE_DEFAULT, min_freq=1):
    counter: Dict[str, int] = {}
    for t in texts:
        for tok in tokenize(t):
            counter[tok] = counter.get(tok, 0) + 1
    items = [(w, c) for w, c in counter.items() if c >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    items = items[: max(0, max_size - 2)]
    vocab: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for w, _ in items:
        vocab[w] = len(vocab)
    return vocab


def build_keyword_vocab(keywords, max_size=KEYWORD_VOCAB_SIZE_DEFAULT):
    vocab: Dict[str, int] = {PAD_TOKEN: 0, NONE_KEYWORD: 1}
    seen = set()
    for kw in keywords:
        if kw is None:
            continue
        if isinstance(kw, float) and np.isnan(kw):
            continue
        k = str(kw).strip().lower().replace("%20", " ")
        if not k or k == "nan":
            continue
        seen.add(k)
    for k in sorted(seen)[: max(0, max_size - 2)]:
        vocab[k] = len(vocab)
    return vocab


def encode_tokens(tokens, vocab, max_len):
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens]
    if len(ids) > max_len:
        ids = ids[:max_len]
    if len(ids) == 0:
        ids = [vocab[UNK_TOKEN]]
    return ids


def keyword_to_id(kw, vocab):
    if kw is None:
        return vocab[NONE_KEYWORD]
    if isinstance(kw, float) and np.isnan(kw):
        return vocab[NONE_KEYWORD]
    k = str(kw).strip().lower().replace("%20", " ")
    if not k or k == "nan":
        return vocab[NONE_KEYWORD]
    return vocab.get(k, vocab[UNK_TOKEN])


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, word_ids, keyword_ids, targets=None):
        self.word_ids = word_ids
        self.keyword_ids = keyword_ids
        self.targets = targets

    def __len__(self):
        return len(self.word_ids)

    def __getitem__(self, idx):
        item = {
            "word_ids": torch.tensor(self.word_ids[idx], dtype=torch.long),
            "keyword_ids": torch.tensor(int(self.keyword_ids[idx]), dtype=torch.long),
        }
        if self.targets is not None:
            item["target"] = torch.tensor(float(self.targets[idx]), dtype=torch.float32)
        return item


def collate_batch(batch):
    max_len = max(int(b["word_ids"].shape[0]) for b in batch)
    rows = []
    for b in batch:
        w = b["word_ids"]
        if w.shape[0] < max_len:
            pad = torch.zeros(max_len - w.shape[0], dtype=torch.long)
            w = torch.cat([w, pad], dim=0)
        rows.append(w)
    out = {
        "word_ids": torch.stack(rows, dim=0),
        "keyword_ids": torch.stack([b["keyword_ids"] for b in batch], dim=0),
    }
    if "target" in batch[0]:
        out["target"] = torch.stack([b["target"] for b in batch], dim=0)
    return out


class DisasterHybridModel(nn.Module):
    """BiGRU + additive attention with learned keyword-bias injection."""

    def __init__(
        self,
        word_vocab_size: int,
        keyword_vocab_size: int,
        embed_dim: int = EMBED_DIM_DEFAULT,
        gru_hidden: int = GRU_HIDDEN_DEFAULT,
        gru_layers: int = GRU_LAYERS_DEFAULT,
        dropout: float = DROPOUT_DEFAULT,
        pad_index: int = 0,
    ) -> None:
        super().__init__()
        self.word_vocab_size = int(word_vocab_size)
        self.keyword_vocab_size = int(keyword_vocab_size)
        self.embed_dim = int(embed_dim)
        self.gru_hidden = int(gru_hidden)
        self.gru_layers = int(gru_layers)

        self.word_emb = nn.Embedding(self.word_vocab_size, self.embed_dim, padding_idx=pad_index)
        self.keyword_emb = nn.Embedding(self.keyword_vocab_size, self.embed_dim, padding_idx=pad_index)
        nn.init.normal_(self.word_emb.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.keyword_emb.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.word_emb.weight[pad_index].zero_()
            self.keyword_emb.weight[pad_index].zero_()

        self.word_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.gru_hidden,
            num_layers=self.gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if self.gru_layers > 1 else 0.0,
        )
        attn_dim = 2 * self.gru_hidden
        self.attn_proj = nn.Sequential(
            nn.Linear(attn_dim, max(32, attn_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(32, attn_dim // 2), 1),
        )
        self.keyword_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(attn_dim + self.embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, word_ids: torch.Tensor, keyword_ids: torch.Tensor) -> torch.Tensor:
        if word_ids.dim() == 1:
            word_ids = word_ids.unsqueeze(0)
        if keyword_ids.dim() == 0:
            keyword_ids = keyword_ids.unsqueeze(0)

        mask = word_ids != 0
        word_vec = self.word_emb(word_ids)
        kw_vec = self.keyword_emb(keyword_ids)

        seq = word_vec + kw_vec.unsqueeze(1)
        seq = self.word_dropout(seq)

        gru_out, _ = self.gru(seq)
        scores = self.attn_proj(gru_out).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        alpha = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bl,blh->bh", alpha, gru_out)

        kw_feat = self.keyword_proj(kw_vec)
        x = torch.cat([pooled, kw_feat], dim=-1)
        logits = self.head(x).squeeze(-1)
        return logits


def build_loss(pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def tune_threshold(probs, targets):
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs >= t).astype(np.int64)
        tp = int(((preds == 1) & (targets == 1)).sum())
        fp = int(((preds == 1) & (targets == 0)).sum())
        fn = int(((preds == 0) & (targets == 1)).sum())
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


# ===========================================================================
# Stage 2: precision policy helpers (import-safe, call-only side effects)
# ===========================================================================
def resolve_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_tf32(enabled: bool = DEFAULT_USE_TF32) -> Dict[str, bool]:
    matmul_flag = bool(enabled)
    cudnn_flag = bool(enabled)
    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = matmul_flag
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = cudnn_flag
    except Exception:
        pass
    return {"matmul_allow_tf32": matmul_flag, "cudnn_allow_tf32": cudnn_flag}


def fp32_island(device_type: Optional[str] = None):
    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if device_type == "cuda":
        try:
            return _torch_autocast(device_type="cuda", enabled=False)
        except TypeError:  # pragma: no cover
            return _torch_autocast(enabled=False)
    return contextlib.nullcontext()


def compute_bce_with_logits_fp32(criterion, logits, targets):
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    with fp32_island("cuda" if logits.is_cuda else "cpu"):
        logits_fp32 = logits.float() if logits.is_floating_point() else logits
        targets_fp32 = targets.float() if targets.is_floating_point() else targets
        loss = criterion(logits_fp32, targets_fp32)
    return loss


def sigmoid_fp32(logits):
    with fp32_island("cuda" if logits.is_cuda else "cpu"):
        logits_fp32 = logits.float() if logits.is_floating_point() else logits
        return torch.sigmoid(logits_fp32)


def make_grad_scaler(enabled):
    if not enabled:
        return None
    try:
        return _TorchGradScaler("cuda")
    except TypeError:  # pragma: no cover
        return _TorchGradScaler()


def resolve_precision_config(device=None, override=None):
    if device is None:
        device = resolve_device(prefer_cuda=True)
    device = torch.device(device)
    cfg: Dict[str, Any] = {
        "device": device,
        "device_type": device.type,
        "cuda_available": bool(torch.cuda.is_available()),
        "requested_policy": PRECISION_FP16_AMP,
        "use_amp": DEFAULT_USE_AMP,
        "amp_dtype": DEFAULT_AMP_DTYPE,
        "amp_dtype_torch": torch.float16,
        "use_tf32": DEFAULT_USE_TF32,
        "fallback_policy": DEFAULT_FALLBACK_POLICY,
        "fp32_islands": ("loss", "sigmoid", "metrics", "export"),
        "autocast_regions": ("forward",),
        "scaler_initialized": False,
        "precision_label": PRECISION_FP16_AMP,
        "reason": "trace: fp16_amp on cuda with fp32 loss/metric/export islands",
    }
    if override:
        for k, v in override.items():
            if v is not None:
                cfg[k] = v
    if device.type != "cuda" or not cfg["cuda_available"]:
        cfg["use_amp"] = False
        cfg["amp_dtype"] = "float32"
        cfg["amp_dtype_torch"] = torch.float32
        cfg["precision_label"] = PRECISION_FP32
        cfg["reason"] = "cpu or cuda unavailable -> fp32 fallback"
    return cfg


class PrecisionRuntime:
    def __init__(self, config=None):
        cfg = resolve_precision_config(override=config)
        self.cfg: Dict[str, Any] = dict(cfg)
        self.device = self.cfg["device"]
        self.device_type = self.cfg["device_type"]
        self.use_amp = bool(self.cfg.get("use_amp", False))
        self.amp_dtype = self.cfg.get("amp_dtype_torch", torch.float32)
        self._autocast_available = self.device_type == "cuda"
        self.grad_scaler = make_grad_scaler(self.use_amp and self._autocast_available)
        self.cfg["scaler_initialized"] = self.grad_scaler is not None
        if self.grad_scaler is None and not self.use_amp:
            self.cfg["precision_label"] = PRECISION_FP32

    def autocast(self):
        if self.use_amp and self._autocast_available:
            try:
                return _torch_autocast(device_type="cuda", dtype=self.amp_dtype)
            except TypeError:  # pragma: no cover
                return _torch_autocast(dtype=self.amp_dtype)
        return contextlib.nullcontext()

    def autocast_forward(self, model, *inputs):
        with self.autocast():
            return model(*inputs)

    def compute_loss(self, criterion, logits, targets):
        return compute_bce_with_logits_fp32(criterion, logits, targets)

    def probabilities(self, logits):
        return sigmoid_fp32(logits)

    @property
    def scaler_enabled(self):
        return self.grad_scaler is not None

    def scaler_step(self, optimizer):
        if self.grad_scaler is not None:
            self.grad_scaler.step(optimizer)
        else:
            optimizer.step()

    def scaler_update(self):
        if self.grad_scaler is not None:
            self.grad_scaler.update()

    def describe(self):
        label = self.cfg.get("precision_label")
        return (
            f"[precision] policy={label} device={self.device_type} amp={self.use_amp} "
            f"amp_dtype={self.cfg.get('amp_dtype')} tf32={self.cfg.get('use_tf32')} "
            f"scaler={'on' if self.scaler_enabled else 'off'} "
            f"fp32_islands={','.join(self.cfg.get('fp32_islands', ()))}"
        )

    def as_settings(self):
        return {
            "precision_label": self.cfg.get("precision_label"),
            "use_amp": self.use_amp,
            "amp_dtype": str(self.amp_dtype).replace("torch.", ""),
            "use_tf32": bool(self.cfg.get("use_tf32", False)),
            "scaler_enabled": self.scaler_enabled,
            "fp32_islands": list(self.cfg.get("fp32_islands", ())),
            "device_type": self.device_type,
        }


def make_precision_runtime(device=None, override=None):
    cfg = resolve_precision_config(device=device, override=override)
    return PrecisionRuntime(config=cfg)


# ===========================================================================
# Stage 3: data pipeline helpers (used only inside the training entrypoint)
# ===========================================================================
def _build_feature_streams(
    texts: List[str], keywords: List[Any], locations: List[Any],
    word_vocab: Dict[str, int], keyword_vocab: Dict[str, int],
):
    fused = [prepare_text_field(t, k, l) for t, k, l in zip(texts, keywords, locations)]
    word_ids = [encode_tokens(tokenize(t), word_vocab, MAX_SEQ_LEN_DEFAULT) for t in fused]
    kw_ids = [keyword_to_id(k, keyword_vocab) for k in keywords]
    return word_ids, kw_ids


def _make_loader(word_ids, kw_ids, targets, batch_size, shuffle, num_workers, pin_memory):
    ds = TweetDataset(word_ids, kw_ids, targets)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
        drop_last=False,
        persistent_workers=False,
    )


def _evaluate(model, loader, device, runtime, criterion):
    model.eval()
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            word_ids = batch["word_ids"].to(device, non_blocking=True)
            keyword_ids = batch["keyword_ids"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            with runtime.autocast():
                logits = model(word_ids, keyword_ids)
            loss = runtime.compute_loss(criterion, logits, targets)
            probs = runtime.probabilities(logits)
            total_loss += float(loss.detach().item()) * word_ids.shape[0]
            n += word_ids.shape[0]
            all_probs.append(probs.detach().cpu().numpy().astype(np.float64))
            all_targets.append(batch["target"].cpu().numpy().astype(np.float64))
    probs = np.concatenate(all_probs) if all_probs else np.zeros(0, dtype=np.float64)
    targs = np.concatenate(all_targets) if all_targets else np.zeros(0, dtype=np.float64)
    mean_loss = total_loss / max(1, n)
    return probs, targs.astype(np.int64), mean_loss


def _predict(model, loader, device, runtime):
    model.eval()
    probs_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            word_ids = batch["word_ids"].to(device, non_blocking=True)
            keyword_ids = batch["keyword_ids"].to(device, non_blocking=True)
            with runtime.autocast():
                logits = model(word_ids, keyword_ids)
            probs = runtime.probabilities(logits)
            probs_list.append(probs.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(probs_list) if probs_list else np.zeros(0, dtype=np.float64)


def _f1_score_binary(probs: np.ndarray, targets: np.ndarray, threshold: float) -> float:
    preds = (probs >= threshold).astype(np.int64)
    tp = int(((preds == 1) & (targets == 1)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom > 0 else 0.0


# ===========================================================================
# CPU preflight adapter
# ===========================================================================
class CandidateAdapter:
    """No-argument CPU preflight adapter for the Stage 3 training script.

    Lazily materializes a small slice of train.csv so import is cheap and
    side-effect free. Exercises the real model, loss, features and collate
    logic used by the training pipeline.
    """

    PREFLIGHT_ROWS = 256

    def __init__(self) -> None:
        self._cache: Optional[Dict[str, Any]] = None

    def _ensure_data(self) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache
        input_dir = os.environ.get("MLEVOLVE_INPUT_DIR", INPUT_DIR_DEFAULT)
        train_path = os.path.join(input_dir, "train.csv")
        df = pd.read_csv(train_path).head(self.PREFLIGHT_ROWS).copy()

        text_col = df["text"].tolist() if "text" in df.columns else [""] * len(df)
        keyword_col = df["keyword"].tolist() if "keyword" in df.columns else [None] * len(df)
        location_col = df["location"].tolist() if "location" in df.columns else [None] * len(df)
        target_col = df["target"].astype(float).tolist() if "target" in df.columns else None

        fused = [prepare_text_field(t, k, l) for t, k, l in zip(text_col, keyword_col, location_col)]
        word_vocab = build_word_vocab(fused, WORD_VOCAB_SIZE_DEFAULT, min_freq=1)
        keyword_vocab = build_keyword_vocab(keyword_col, KEYWORD_VOCAB_SIZE_DEFAULT)

        word_ids = [encode_tokens(tokenize(t), word_vocab, MAX_SEQ_LEN_DEFAULT) for t in fused]
        kw_ids = [keyword_to_id(k, keyword_vocab) for k in keyword_col]

        self._cache = {
            "word_vocab": word_vocab,
            "keyword_vocab": keyword_vocab,
            "word_ids": word_ids,
            "keyword_ids": kw_ids,
            "targets": target_col,
        }
        return self._cache

    @staticmethod
    def _merge_context(context: Any, defaults: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(defaults)
        if isinstance(context, dict):
            for k, v in context.items():
                if v is not None:
                    merged[k] = v
        return merged

    def _make_batch(self, batch_size: int, device: torch.device) -> Dict[str, Any]:
        cache = self._ensure_data()
        n = len(cache["word_ids"])
        bs = max(1, min(int(batch_size), n))
        idxs = list(range(bs))

        word_ids = torch.stack(
            [torch.tensor(cache["word_ids"][i], dtype=torch.long) for i in idxs], dim=0
        )
        keyword_ids = torch.stack(
            [torch.tensor(int(cache["keyword_ids"][i]), dtype=torch.long) for i in idxs],
            dim=0,
        )
        if cache["targets"] is not None:
            targets = torch.tensor(
                [float(cache["targets"][i]) for i in idxs], dtype=torch.float32
            )
        else:
            targets = torch.zeros(bs, dtype=torch.float32)

        word_ids = word_ids.to(device)
        keyword_ids = keyword_ids.to(device)
        targets = targets.to(device)
        return {
            "inputs": (word_ids, keyword_ids),
            "target": targets,
            "word_ids": word_ids,
            "keyword_ids": keyword_ids,
        }

    def build_model(self, context: Any) -> nn.Module:
        cache = self._ensure_data()
        return DisasterHybridModel(
            word_vocab_size=len(cache["word_vocab"]),
            keyword_vocab_size=len(cache["keyword_vocab"]),
            embed_dim=EMBED_DIM_DEFAULT,
            gru_hidden=GRU_HIDDEN_DEFAULT,
            gru_layers=GRU_LAYERS_DEFAULT,
            dropout=DROPOUT_DEFAULT,
        )

    def build_optimizer(self, model: nn.Module, context: Any) -> torch.optim.Optimizer:
        merged = self._merge_context(context, {"lr": 1e-3, "weight_decay": 0.0})
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(merged["lr"]),
            weight_decay=float(merged["weight_decay"]),
        )

    def build_train_batch(self, scenario: Any, device: torch.device) -> Dict[str, Any]:
        bs = 4
        if isinstance(scenario, dict) and scenario.get("batch_size") is not None:
            bs = int(scenario["batch_size"])
        return self._make_batch(bs, device)

    def build_validation_batch(self, scenario: Any, device: torch.device) -> Dict[str, Any]:
        bs = 4
        if isinstance(scenario, dict) and scenario.get("batch_size") is not None:
            bs = int(scenario["batch_size"])
        return self._make_batch(bs, device)

    def training_step(self, model: nn.Module, batch: Dict[str, Any], context: Any) -> torch.Tensor:
        inputs = batch["inputs"]
        target = batch["target"]
        logits = model(*inputs)
        criterion = build_loss()
        loss = compute_bce_with_logits_fp32(criterion, logits, target)
        return loss

    def validation_step(self, model: nn.Module, batch: Dict[str, Any], context: Any) -> torch.Tensor:
        inputs = batch["inputs"]
        target = batch["target"]
        logits = model(*inputs)
        criterion = build_loss()
        loss = compute_bce_with_logits_fp32(criterion, logits, target)
        return loss


# ===========================================================================
# Stage 3 training entrypoint
# ===========================================================================
EPOCHS = 8
PHYSICAL_BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 1
LR = 2e-3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
EARLY_STOP_PATIENCE = 3
NUM_WORKERS = 2


def run_training() -> float:
    # -- 1. Scheduler context -------------------------------------------------
    try:
        from localml_scheduler.execution.script_context import script_scheduler_context  # type: ignore
        from localml_scheduler.domain import SafePointType  # type: ignore
    except Exception:
        script_scheduler_context = None  # type: ignore
        SafePointType = None  # type: ignore

    context = None
    if script_scheduler_context is not None:
        context = script_scheduler_context()

    # -- 2. Device / precision / TF32 ----------------------------------------
    device = resolve_device(prefer_cuda=True)
    configure_tf32(DEFAULT_USE_TF32)
    runtime = make_precision_runtime(device=device)
    print(runtime.describe())
    scaler = runtime.grad_scaler

    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -- 3. Load CSVs --------------------------------------------------------
    input_dir = os.environ.get("MLEVOLVE_INPUT_DIR", INPUT_DIR_DEFAULT)
    train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(input_dir, "test.csv"))

    # -- 4. Stratified split (fit vocab only on training fold) --------------
    from sklearn.model_selection import train_test_split

    all_idx = np.arange(len(train_df))
    tr_idx, va_idx = train_test_split(
        all_idx,
        test_size=VAL_FRACTION,
        random_state=SPLIT_SEED,
        stratify=train_df["target"].values,
    )
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    def _cols(df):
        texts = df["text"].tolist() if "text" in df.columns else [""] * len(df)
        kws = df["keyword"].tolist() if "keyword" in df.columns else [None] * len(df)
        locs = df["location"].tolist() if "location" in df.columns else [None] * len(df)
        return texts, kws, locs

    tr_texts, tr_kws, tr_locs = _cols(tr_df)
    va_texts, va_kws, va_locs = _cols(va_df)
    te_texts, te_kws, te_locs = _cols(test_df)

    tr_fused = [prepare_text_field(t, k, l) for t, k, l in zip(tr_texts, tr_kws, tr_locs)]
    word_vocab = build_word_vocab(tr_fused, WORD_VOCAB_SIZE_DEFAULT, min_freq=1)
    keyword_vocab = build_keyword_vocab(tr_kws, KEYWORD_VOCAB_SIZE_DEFAULT)

    tr_word_ids, tr_kw_ids = _build_feature_streams(tr_texts, tr_kws, tr_locs, word_vocab, keyword_vocab)
    va_word_ids, va_kw_ids = _build_feature_streams(va_texts, va_kws, va_locs, word_vocab, keyword_vocab)
    te_word_ids, te_kw_ids = _build_feature_streams(te_texts, te_kws, te_locs, word_vocab, keyword_vocab)

    tr_targets = tr_df["target"].astype(float).tolist()
    va_targets = va_df["target"].astype(int).values

    pin_memory = device.type == "cuda"
    train_loader = _make_loader(
        tr_word_ids, tr_kw_ids, tr_targets,
        batch_size=PHYSICAL_BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin_memory,
    )
    val_loader = _make_loader(
        va_word_ids, va_kw_ids, va_targets.tolist(),
        batch_size=PHYSICAL_BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin_memory,
    )
    test_loader = _make_loader(
        te_word_ids, te_kw_ids, None,
        batch_size=PHYSICAL_BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin_memory,
    )

    steps_per_epoch = max(1, math.ceil(len(tr_word_ids) / PHYSICAL_BATCH_SIZE))
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    # -- 5. Model / optimizer / scheduler -----------------------------------
    model = DisasterHybridModel(
        word_vocab_size=len(word_vocab),
        keyword_vocab_size=len(keyword_vocab),
        embed_dim=EMBED_DIM_DEFAULT,
        gru_hidden=GRU_HIDDEN_DEFAULT,
        gru_layers=GRU_LAYERS_DEFAULT,
        dropout=DROPOUT_DEFAULT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8
    )
    criterion = build_loss()

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    effective_batch_size = PHYSICAL_BATCH_SIZE * GRAD_ACCUM_STEPS
    diagnostics_settings = {
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "effective_batch_size": effective_batch_size,
        "planned_epochs": EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "model_family": MODEL_FAMILY,
        "split_seed": SPLIT_SEED,
        "precision": runtime.as_settings(),
        "precision_reason": runtime.cfg.get("reason", ""),
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "quality_safe_physical_batch_sizes": list(QUALITY_SAFE_PHYSICAL_BATCH_SIZES),
        "batch_lr_scaling_policy": BATCH_LR_SCALING_POLICY,
    }

    # -- 6. Scheduler bookkeeping -------------------------------------------
    diag_state: Dict[str, int] = {"global_step": 0, "epoch": 0, "step_in_epoch": 0}

    if context is not None:
        try:
            context.job.max_epochs = EPOCHS
            context.job.config.max_epochs = EPOCHS
            context.store.save_job(context.job)
        except Exception:
            pass

    with TrainingDiagnostics(
        model, optimizer, scaler=scaler, settings=diagnostics_settings
    ) as diagnostics:
        if context is not None:
            state_factory = _make_state_factory(
                model, optimizer, scaler, scheduler, diag_state
            )
            context.control_hook.safe_point(
                SafePointType.BEFORE_TRAIN,
                epoch=0,
                global_step=0,
                steps_per_epoch=steps_per_epoch,
                state_factory=state_factory,
            )

        best_val_f1 = -1.0
        best_state = copy.deepcopy(model.state_dict())
        best_threshold = 0.5
        epochs_no_improve = 0

        for epoch in range(1, EPOCHS + 1):
            diag_state["epoch"] = epoch
            model.train()
            epoch_loss = 0.0
            n_seen = 0
            t0 = time.time()

            for step_in_epoch, batch in enumerate(train_loader):
                diag_state["step_in_epoch"] = step_in_epoch
                word_ids = batch["word_ids"].to(device, non_blocking=True)
                keyword_ids = batch["keyword_ids"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with runtime.autocast():
                    logits = model(word_ids, keyword_ids)
                loss = runtime.compute_loss(criterion, logits, targets)
                loss_scaled = scaler.scale(loss) if scaler is not None else loss
                loss_scaled.backward()
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                runtime.scaler_step(optimizer)
                runtime.scaler_update()
                scheduler.step()
                diagnostics.after_update()
                diag_state["global_step"] += 1

                bs = word_ids.shape[0]
                epoch_loss += float(loss.detach().item()) * bs
                n_seen += bs

                if context is not None:
                    state_factory = _make_state_factory(
                        model, optimizer, scaler, scheduler, diag_state
                    )
                    context.control_hook.safe_point(
                        SafePointType.STEP,
                        epoch=epoch,
                        global_step=diag_state["global_step"],
                        steps_per_epoch=steps_per_epoch,
                        state_factory=state_factory,
                    )

            # -- validation -------------------------------------------------
            val_probs, val_targs, val_loss = _evaluate(model, val_loader, device, runtime, criterion)
            best_t, val_f1 = tune_threshold(val_probs, val_targs)
            avg_loss = epoch_loss / max(1, n_seen)
            elapsed = time.time() - t0

            diagnostics.report(epoch=epoch)
            print(
                f"Epoch {epoch}/{EPOCHS} | loss={avg_loss:.4f} | val_loss={val_loss:.4f} "
                f"| val_f1={val_f1:.4f} (t={best_t:.2f}) | lr={scheduler.get_last_lr()[0]:.2e} "
                f"| {elapsed:.1f}s"
            )
            print(
                f"MLEVOLVE_EPOCH_METRIC "
                f"{{\"epoch\": {epoch}, \"metric\": {val_f1}, \"metric_name\": \"f1\"}}"
            )

            if val_f1 > best_val_f1 + 1e-6:
                best_val_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())
                best_threshold = float(best_t)
                epochs_no_improve = 0
                try:
                    ckpt_path = os.path.join(WORKING_DIR, "best_model.pt")
                    torch.save(
                        {
                            "model": best_state,
                            "threshold": best_threshold,
                            "val_f1": best_val_f1,
                            "epoch": epoch,
                            "diagnostics": diagnostics.state_dict(),
                        },
                        ckpt_path,
                    )
               
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
        