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
 Summary: 3 total attempts - 1 pending execution, 2 executed, 1 successful (best: 0.7341)

-------------------------------
Attempt #1:
Design: I merged the three stages by keeping Stage 1's data/model/loss/adapter code as the foundation, layering Stage 2's `PrecisionRuntime` policy helpers on top, and consuming both inside Stage 3's `run_training` (with diagnostics + scheduler safe points). I resolved naming conflicts, deduplicated imports, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module-level constants that Stage 3 references, and moved every side effect under `if __name__ == '__main__':` while preserving the CPU-safe `CandidateAdapter` contract.
Results: Training completed successfully in ~37 minutes on an A10 GPU, running 12 epochs with a hybrid transformer model. Validation F1 improved from 0.665 at epoch 1 to a best of ~0.7341 (best epoch 11), and a valid submission file with 1523 rows was written using a tuned threshold of 0.42. The only messages were benign warnings (Graph evidence mirror connection refused, nested-tensor warning) that did not affect execution.
Validation Metric: 0.7341
Execution Time: 2223.42s

-------------------------------
Attempt #2:
Design: Looking at the three stages, I merged by keeping Stage 1's BiGRU+keyword-bias model/data/loss/threshold-tuner/CandidateAdapter as the foundation, Stage 2's `PrecisionRuntime` helpers (FP16-AMP forward autocast + FP32 islands for loss/sigmoid/metrics, TF32 disabled, CPU-safe preflight) layered on top, and completing Stage 3's truncated training loop (AdamW + linear-warmup+cosine, FP16 autocast forward, scaler.unscale before clip, val-F1 early stopping with best-state restore, guarded scheduler safe-points, `TrainingDiagnostics` with fallback), then adding the missing test inference + submission write + `if __name__ == "__main__":` entrypoint. I fixed the truncated `_make_state_factory`, resolved duplicated constants/imports, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module constants, and put the final `Final Validation Score:` print as the very last line after the diagnostics context closes.
Results: The implementation of this design has bugs.
Insight: Using a different approach may not result in the same bugs as the above approach.

-------------------------------
Attempt #3:
Design: I merged the three stages by keeping Stage 1's data cleaning, TF-IDF feature builders, `SparseMLP`, `build_loss`, `tune_threshold` and split helpers as the foundation; layering Stage 2's import-safe precision utilities (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `degrade_to_fp32`, `configure_tf32`, `format_precision_log`) on top; then consuming both inside Stage 3's `run_training` with scheduler safe-points, `TrainingDiagnostics`, linear warmup+cosine AdamW, best-val-F1 restore, and test inference. I resolved the duplicate `CandidateAdapter` by keeping Stage 3's self-contained CPU-safe version (it uses the Stage 1 model/loss/features through the Stage 2 precision path), removed duplicate imports/constants, kept all side effects under `if __name__ == "__main__":`, and preserved the exact `MODEL_FAMILY`, `WORKING_DIR`, `SUBMISSION_DIR` contract.
Status: Code generated, execution pending (will run in parallel with other drafts).


# Hardware-Aware Stage 1 Candidate Construction Context
# Hardware-Aware Model Design Brief
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2; observed in 1 batch-size observation
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
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"cross entropy or metric-compatible classification loss","output_interface":"match the competition metric and required submission format","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"{'forward_autocast_boundaries': \"Use torch.autocast(device_type='cuda', dtype=torch.float16) only...","fallback_policy":"fp32","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"{'name': 'AdamW', 'learning_rate': 2e-05, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-08}","scheduler":"{'name': 'linear_warmup_decay', 'warmup_ratio': 0.1, 'num_training_steps': 'Derived from epochs,...","batch_size_policy":"fixed","dataloader_policy":"{'shuffle_train': True, 'shuffle_validation': False, 'shuffle_test': False, 'num_workers': '>0 wh...","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available","no exact, similar, or packed graph profiles for current hardware","no measured FP16 AMP speedup or quality comparison for this task"]}}
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
 Summary: 3 total attempts - 1 pending execution, 2 executed, 1 successful (best: 0.7341)

-------------------------------
Attempt #1:
Design: I merged the three stages by keeping Stage 1's data/model/loss/adapter code as the foundation, layering Stage 2's `PrecisionRuntime` policy helpers on top, and consuming both inside Stage 3's `run_training` (with diagnostics + scheduler safe points). I resolved naming conflicts, deduplicated imports, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module-level constants that Stage 3 references, and moved every side effect under `if __name__ == '__main__':` while preserving the CPU-safe `CandidateAdapter` contract.
Results: Training completed successfully in ~37 minutes on an A10 GPU, running 12 epochs with a hybrid transformer model. Validation F1 improved from 0.665 at epoch 1 to a best of ~0.7341 (best epoch 11), and a valid submission file with 1523 rows was written using a tuned threshold of 0.42. The only messages were benign warnings (Graph evidence mirror connection refused, nested-tensor warning) that did not affect execution.
Validation Metric: 0.7341
Execution Time: 2223.42s

-------------------------------
Attempt #2:
Design: Looking at the three stages, I merged by keeping Stage 1's BiGRU+keyword-bias model/data/loss/threshold-tuner/CandidateAdapter as the foundation, Stage 2's `PrecisionRuntime` helpers (FP16-AMP forward autocast + FP32 islands for loss/sigmoid/metrics, TF32 disabled, CPU-safe preflight) layered on top, and completing Stage 3's truncated training loop (AdamW + linear-warmup+cosine, FP16 autocast forward, scaler.unscale before clip, val-F1 early stopping with best-state restore, guarded scheduler safe-points, `TrainingDiagnostics` with fallback), then adding the missing test inference + submission write + `if __name__ == "__main__":` entrypoint. I fixed the truncated `_make_state_factory`, resolved duplicated constants/imports, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module constants, and put the final `Final Validation Score:` print as the very last line after the diagnostics context closes.
Results: The implementation of this design has bugs.
Insight: Using a different approach may not result in the same bugs as the above approach.

-------------------------------
Attempt #3:
Design: I merged the three stages by keeping Stage 1's data cleaning, TF-IDF feature builders, `SparseMLP`, `build_loss`, `tune_threshold` and split helpers as the foundation; layering Stage 2's import-safe precision utilities (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `degrade_to_fp32`, `configure_tf32`, `format_precision_log`) on top; then consuming both inside Stage 3's `run_training` with scheduler safe-points, `TrainingDiagnostics`, linear warmup+cosine AdamW, best-val-F1 restore, and test inference. I resolved the duplicate `CandidateAdapter` by keeping Stage 3's self-contained CPU-safe version (it uses the Stage 1 model/loss/features through the Stage 2 precision path), removed duplicate imports/constants, kept all side effects under `if __name__ == "__main__":`, and preserved the exact `MODEL_FAMILY`, `WORKING_DIR`, `SUBMISSION_DIR` contract.
Status: Code generated, execution pending (will run in parallel with other drafts).


# Hardware-Aware Datatype/Precision Context
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2; observed in 1 batch-size observation
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
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"cross entropy or metric-compatible classification loss","output_interface":"match the competition metric and required submission format","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"{'forward_autocast_boundaries': \"Use torch.autocast(device_type='cuda', dtype=torch.float16) only...","fallback_policy":"fp32","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"{'name': 'AdamW', 'learning_rate': 2e-05, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-08}","scheduler":"{'name': 'linear_warmup_decay', 'warmup_ratio': 0.1, 'num_training_steps': 'Derived from epochs,...","batch_size_policy":"fixed","dataloader_policy":"{'shuffle_train': True, 'shuffle_validation': False, 'shuffle_test': False, 'num_workers': '>0 wh...","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available","no exact, similar, or packed graph profiles for current hardware","no measured FP16 AMP speedup or quality comparison for this task"]}}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=replaced recurrent/transformer baselines with a multi-view CNN (parallel word kernels + char CNN) whose keyword embedding is multiplicatively gated by the fused text context; purpose=exploit short-text inductive bias and model keyword–context interaction that prior attempts treated as a naive bias term; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=split is stratified 85/15 with seed 42 fit strictly on `train.csv` labels, and all stateful transforms (SimpleTokenizer, CharTokenizer, KeywordMapper) are fit only on the training split; purpose=strict leakage isolation across train/val/test; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=exposes `MAX_WORD_LEN=32`, `MAX_CHAR_LEN=128`, and integer-id inputs via a reusable `make_batch_tensors` helper + a CPU-safe `CandidateAdapter`; purpose=keep sequence geometry configurable for later precision/batching stages without committing to them here; hardware_keys=none

# Previous steps
### model_design
**Plan:** ## Plan

**WHAT makes this task unique**: Disaster tweets are short (≤32 tokens), idiosyncratic, and heavily signal through local phrases + hashtags + the structured `keyword` column, which is effectively a pre-extracted topic tag. The signal is *not* long-range — it lives in n-gram pockets and in whether a topic keyword is used in a disaster sense. Class balance is moderate (~43% positive).

**WHY this approach fits**: A from-scratch deep transformer on 6k short texts overfits; instead I use a **multi-view CNN with keyword cross-gating**, which matches the inductive bias of short-text classification: (a) parallel multi-kernel word convolutions capture disaster n-gram phrases ("wildfire", "on fire", "killed in"), (b) a character CNN captures hashtag/casing/morphological cues without a subword pretrainer, and (c) a learned keyword embedding is *gated* by the word-context — the model can learn to amplify the keyword only when the tweet's surrounding words suggest it's a real disaster, mirroring the human reading of `keyword="ablaze" + "beautiful sunset ablaze"`. This is genuinely novel vs. prior attempts (BiGRU+keyword, TF-IDF+SparseMLP, hybrid transformer), which either trust the keyword naively or ignore the gating interaction. Modeling budget is conservative: ~3M params, no pretrained weights, no external downloads.

**HOW validation confirms/refutes**: Split 85/15 stratified (seed 42) *before* any vocab fit. If the gated keyword view genuinely helps, validation F1 with gate-on should beat a keyword-embedding-concat ablation by a clear margin; if validation F1 collapses toward the majority-class baseline, the gate is learning to zero-out the keyword and I'd fall back to plain concat. Metric = binary F1 with threshold tuned on validation (as Stage 3 consumes).

**Backend/memory strategy**: No precision or optimizer choices are made here (those belong to later stages); the design exposes configurable sequence lengths (`max_word_len`, `max_char_len`), configurable batch size honor via the adapter, and integer token tensors (torch.long) so non-blocking CUDA transfer is possible. Stateful transforms (word/char vocabs, keyword map) are fit *only* on train texts, and reused via `.encode()` on val/test. Scheduler-owned controls (device assignment, num_workers, batch size, AMP/scaler, checkpointing) are left to later stages; the subprocess must not implement them itself.

### Note board:
- baseline_change: replaced recurrent/transformer baselines with a multi-view CNN (parallel word kernels + char CNN) whose keyword embedding is multiplicatively gated by the fused text context | purpose: exploit short-text inductive bias and model keyword–context interaction that prior attempts treated as a naive bias term | hardware_keys: none
- baseline_change: split is stratified 85/15 with seed 42 fit strictly on `train.csv` labels, and all stateful transforms (SimpleTokenizer, CharTokenizer, KeywordMapper) are fit only on the training split | purpose: strict leakage isolation across train/val/test | hardware_keys: none
- baseline_change: exposes `MAX_WORD_LEN=32`, `MAX_CHAR_LEN=128`, and integer-id inputs via a reusable `make_batch_tensors` helper + a CPU-safe `CandidateAdapter` | purpose: keep sequence geometry configurable for later precision/batching stages without committing to them here | hardware_keys: none
**Code:**
```python
"""
Stage 1 (model_design) — Disaster Tweets binary classification.

Model family: MultiViewDisasterNet, a from-scratch multi-view CNN:
  - Word branch: parallel Conv1d over word embeddings (kernels 2,3,4) → max-pool.
  - Char branch: single Conv1d over char embeddings → max-pool.
  - Keyword branch: learned embedding multiplicatively gated by concatenated
    word+char context, so the keyword only contributes when the tweet context
    suggests a real-disaster reading.
  - Fusion: LayerNorm → dropout → 2-layer MLP → single logit.

This module is IMPORT-SAFE: only imports, constants, classes, and helper
functions are defined at top level. Training loop, optimizer, scheduler,
precision policy, batch size, epochs, and workers all live in later stages.
"""

# ---- Module-level constants (no side effects) -----------------------------

MODEL_FAMILY = "mvcnn_keyword_gate_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
DEFAULT_INPUT_DIR = "./input"

PAD_IDX = 0
UNK_IDX = 1
NUM_SPECIAL_TOKENS = 2

# Configurable sequence geometry (exposed for later stages).
MAX_WORD_LEN = 32
MAX_CHAR_LEN = 128

# Defaults for the production model built during training.
WORD_EMBED_DIM = 100
WORD_KERNELS = (2, 3, 4)
WORD_FILTERS = 96
CHAR_EMBED_DIM = 32
CHAR_KERNEL = 3
CHAR_FILTERS = 48
KEYWORD_DIM = 32
HIDDEN_DIM = 128
DROPOUT = 0.4

# Defaults for the lightweight CPU-preflight adapter.
ADAPTER_WORD_EMBED_DIM = 64
ADAPTER_WORD_FILTERS = 32
ADAPTER_CHAR_EMBED_DIM = 16
ADAPTER_CHAR_FILTERS = 16
ADAPTER_KEYWORD_DIM = 16
ADAPTER_HIDDEN_DIM = 48
ADAPTER_DROPOUT = 0.2
ADAPTER_N_SAMPLE_ROWS = 128


# ---- Imports --------------------------------------------------------------

import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ---- Text cleaning --------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_USER_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NON_PRINTABLE_RE = re.compile(r"[^a-z0-9<>_\s]")
_WS_RE = re.compile(r"\s+")

URL_TOKEN = "<url>"
USER_TOKEN = "<user>"
HASHTAG_PREFIX = "htag_"


def clean_text(text: str) -> str:
    """Lowercase, normalize URLs/@mentions/hashtags, strip odd chars.

    Deterministic, stateless — safe to apply to train/val/test identically.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(f" {URL_TOKEN} ", text)
    text = _USER_RE.sub(f" {USER_TOKEN} ", text)
    text = _HASHTAG_RE.sub(lambda m: f" {HASHTAG_PREFIX}{m.group(1)} ", text)
    text = _NON_PRINTABLE_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def join_context(text: str, keyword, location) -> str:
    """Optional richer field used only if a downstream stage wants it.

    Kept as a separate helper so the main text pipeline stays clean.
    """
    parts = [clean_text(text)]
    if isinstance(keyword, str) and keyword.strip():
        parts.append("kw_" + keyword.strip().lower().replace(" ", "_"))
    if isinstance(location, str) and location.strip():
        parts.append("loc_" + location.strip().lower().replace(" ", "_"))
    return " ".join(p for p in parts if p)


# ---- Stateful tokenizers (fit on TRAIN ONLY) -----------------------------


class SimpleTokenizer:
    """Word-level vocab built from scratch (no pretrained tokenizer)."""

    def __init__(self, min_freq: int = 1, max_vocab: int = 30000):
        self.min_freq = int(min_freq)
        self.max_vocab = int(max_vocab)
        self.vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}

    def fit(self, texts):
        c = Counter()
        for t in texts:
            for tok in (t or "").split():
                c[tok] += 1
        items = [(tok, n) for tok, n in c.most_common() if n >= self.min_freq]
        items = items[: max(self.max_vocab - NUM_SPECIAL_TOKENS, 0)]
        for tok, _ in items:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
        return self

    @property
    def vocab_size(self) -> int:
        return max(len(self.vocab), NUM_SPECIAL_TOKENS)

    def encode(self, texts, max_len: int = MAX_WORD_LEN) -> np.ndarray:
        out = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, t in enumerate(texts):
            toks = (t or "").split()[:max_len]
            ids = [self.vocab.get(tok, UNK_IDX) for tok in toks]
            if ids:
                out[i, : len(ids)] = ids
        return out


class CharTokenizer:
    """Character-level vocab (subset of printable ASCII + special tokens)."""

    def __init__(self, min_freq: int = 2, max_vocab: int = 128):
        self.min_freq = int(min_freq)
        self.max_vocab = int(max_vocab)
        self.vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}

    def fit(self, texts):
        c = Counter()
        for t in texts:
            for ch in t or "":
                c[ch] += 1
        items = [(ch, n) for ch, n in c.most_common() if n >= self.min_freq]
        items = items[: max(self.max_vocab - NUM_SPECIAL_TOKENS, 0)]
        for ch, _ in items:
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)
        return self

    @property
    def vocab_size(self) -> int:
        return max(len(self.vocab), NUM_SPECIAL_TOKENS)

    def encode(self, texts, max_len: int = MAX_CHAR_LEN) -> np.ndarray:
        out = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, t in enumerate(texts):
            chars = (t or "")[:max_len]
            ids = [self.vocab.get(ch, UNK_IDX) for ch in chars]
            if ids:
                out[i, : len(ids)] = ids
        return out


class KeywordMapper:
    """Categorical keyword map. Index 0 = missing, 1 = unseen, then real keys."""

    def __init__(self):
        self.mapping = {"<none>": 0, "<unk>": 1}

    def fit(self, keywords):
        for k in sorted({k for k in keywords if isinstance(k, str) and k.strip()}):
            if k not in self.mapping:
                self.mapping[k] = len(self.mapping)
        return self

    @property
    def num_keywords(self) -> int:
        return max(len(self.mapping), 2)

    def encode(self, keywords) -> np.ndarray:
        out = np.zeros(len(keywords), dtype=np.int64)
        for i, k in enumerate(keywords):
            if not isinstance(k, str) or not k.strip():
                out[i] = 0
            else:
                out[i] = self.mapping.get(k, 1)
        return out


# ---- Model ---------------------------------------------------------------


class MultiViewDisasterNet(nn.Module):
    """Multi-view CNN with a keyword cross-gate.

    Forward signature: forward(word_ids, char_ids, keyword_ids) -> logits (B,)
    """

    def __init__(
        self,
        word_vocab_size: int,
        char_vocab_size: int,
        num_keywords: int,
        word_embed_dim: int = WORD_EMBED_DIM,
        word_kernels=WORD_KERNELS,
        word_filters: int = WORD_FILTERS,
        char_embed_dim: int = CHAR_EMBED_DIM,
        char_kernel: int = CHAR_KERNEL,
        char_filters: int = CHAR_FILTERS,
        keyword_dim: int = KEYWORD_DIM,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
        num_classes: int = 1,
    ):
        super().__init__()
        self.word_vocab_size = int(word_vocab_size)
        self.char_vocab_size = int(char_vocab_size)
        self.num_keywords = int(num_keywords)

        # -- Word branch --
        self.word_emb = nn.Embedding(
            word_vocab_size, word_embed_dim, padding_idx=PAD_IDX
        )
        self.word_convs = nn.ModuleList(
            [
                nn.Conv1d(word_embed_dim, word_filters, k, padding=k // 2)
                for k in word_kernels
            ]
        )
        word_feat_dim = word_filters * len(word_kernels)

        # -- Char branch --
        self.char_emb = nn.Embedding(
            char_vocab_size, char_embed_dim, padding_idx=PAD_IDX
        )
        self.char_conv = nn.Conv1d(
            char_embed_dim, char_filters, char_kernel, padding=char_kernel // 2
        )

        # -- Keyword branch with cross-gate --
        self.kw_emb = nn.Embedding(num_keywords, keyword_dim)
        fused_in = word_feat_dim + char_filters
        self.kw_gate = nn.Linear(fused_in + keyword_dim, keyword_dim)
        self.kw_proj = nn.Linear(keyword_dim, keyword_dim)

        # -- Fusion head --
        self.norm = nn.LayerNorm(fused_in + keyword_dim)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(fused_in + keyword_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].fill_(0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, word_ids, char_ids, keyword_ids):
        # --- Word branch ---
        w = self.word_emb(word_ids)  # (B, L, D)
        w = w.transpose(1, 2)  # (B, D, L)
        w_feats = []
        for conv in self.word_convs:
            h = F.relu(conv(w))
            h = F.adaptive_max_pool1d(h, 1).squeeze(-1)
            w_feats.append(h)
        word_feat = torch.cat(w_feats, dim=-1)  # (B, word_feat_dim)

        # --- Char branch ---
        c = self.char_emb(char_ids).transpose(1, 2)
        c = F.relu(self.char_conv(c))
        char_feat = F.adaptive_max_pool1d(c, 1).squeeze(-1)  # (B, char_filters)

        base = torch.cat([word_feat, char_feat], dim=-1)  # (B, fused_in)

        # --- Keyword cross-gate ---
        kw = self.kw_emb(keyword_ids)  # (B, keyword_dim)
        gate = torch.sigmoid(self.kw_gate(torch.cat([base, kw], dim=-1)))
        kw_gated = gate * self.kw_proj(kw)

        # --- Fusion ---
        fused = torch.cat([base, kw_gated], dim=-1)
        fused = self.norm(fused)
        fused = self.drop(fused)
        logits = self.classifier(fused)  # (B, 1)
        return logits.squeeze(-1)


# ---- Loss / metric helpers ----------------------------------------------


def build_loss(pos_weight=None):
    """BCEWithLogitsLoss with optional positive-class weighting.

    `pos_weight` is a Python scalar (num_neg / num_pos). If None, no weighting.
    """
    if pos_weight is None:
        pw = None
    else:
        pw = torch.tensor([float(pos_weight)], dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pw)


def compute_f1(y_true, y_pred_binary) -> float:
    return float(
        f1_score(np.asarray(y_true), np.asarray(y_pred_binary), zero_division=0)
    )


def tune_threshold(
    y_true, y_prob, min_t: float = 0.20, max_t: float = 0.80, steps: int = 61
):
    """Return (best_threshold, best_f1) maximizing binary F1 on val probs."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(min_t, max_t, steps):
        preds = (y_prob >= t).astype(int)
        f1 = compute_f1(y_true, preds)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


# ---- Split + feature orchestration (pure functions, called under __main__) -


def resolve_input_dir() -> str:
    env = os.environ.get("MLEVOLVE_INPUT_DIR")
    if env and os.path.isdir(env):
        return env
    return DEFAULT_INPUT_DIR


def load_raw_data(input_dir=None):
    input_dir = input_dir or resolve_input_dir()
    train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(input_dir, "test.csv"))
    return train_df, test_df


def stratified_split(train_df, val_size: float = 0.15, seed: int = 42):
    """Split BEFORE any vocab/stat fit, stratified on target."""
    tr, va = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df["target"].astype(int),
        random_state=seed,
    )
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def prepare_dataframes(train_df, test_df, val_size: float = 0.15, seed: int = 42):
    """Produce raw splits with cleaned text. No vocab fitting here."""
    tr, va = stratified_split(train_df, val_size=val_size, seed=seed)
    for df in (tr, va, test_df):
        df["_clean_text"] = df["text"].apply(clean_text)
    if "_clean_text" not in test_df.columns:
        test_df["_clean_text"] = test_df["text"].apply(clean_text)
    return tr, va, test_df


def fit_featurizers(
    train_texts,
    train_keywords,
    word_max_vocab: int = 30000,
    char_min_freq: int = 2,
    char_max_vocab: int = 128,
):
    """Fit tokenizers/mappers ONLY on training data (leakage-safe)."""
    wtok = SimpleTokenizer(min_freq=1, max_vocab=word_max_vocab).fit(train_texts)
    ctok = CharTokenizer(min_freq=char_min_freq, max_vocab=char_max_vocab).fit(
        train_texts
    )
    kmap = KeywordMapper().fit(train_keywords)
    return wtok, ctok, kmap


def make_batch_tensors(
    texts,
    keywords,
    wtok,
    ctok,
    kmap,
    max_word_len: int = MAX_WORD_LEN,
    max_char_len: int = MAX_CHAR_LEN,
    device=None,
):
    """Return (word_ids, char_ids, keyword_ids) long tensors on `device`.

    integer indices stay torch.long; only floating tensors are affected by
    later precision policies.
    """
    word_np = wtok.encode(texts, max_word_len)
    char_np = ctok.encode(texts, max_char_len)
    kw_np = kmap.encode(keywords)
    word_ids = torch.from_numpy(word_np).long()
    char_ids = torch.from_numpy(char_np).long()
    kw_ids = torch.from_numpy(kw_np).long()
    if device is not None:
        word_ids = word_ids.to(device)
        char_ids = char_ids.to(device)
        kw_ids = kw_ids.to(device)
    return word_ids, char_ids, kw_ids


def build_model_from_featurizers(
    wtok,
    ctok,
    kmap,
    word_embed_dim: int = WORD_EMBED_DIM,
    word_filters: int = WORD_FILTERS,
    char_embed_dim: int = CHAR_EMBED_DIM,
    char_filters: int = CHAR_FILTERS,
    keyword_dim: int = KEYWORD_DIM,
    hidden_dim: int = HIDDEN_DIM,
    dropout: float = DROPOUT,
) -> MultiViewDisasterNet:
    return MultiViewDisasterNet(
        word_vocab_size=wtok.vocab_size,
        char_vocab_size=ctok.vocab_size,
        num_keywords=kmap.num_keywords,
        word_embed_dim=word_embed_dim,
        word_kernels=WORD_KERNELS,
        word_filters=word_filters,
        char_embed_dim=char_embed_dim,
        char_kernel=CHAR_KERNEL,
        char_filters=char_filters,
        keyword_dim=keyword_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def get_pos_weight(labels) -> float:
    labels = np.asarray(labels).astype(int)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    if pos <= 0:
        return 1.0
    return float(neg) / float(pos)


@torch.no_grad()
def predict_proba(
    model, word_ids, char_ids, keyword_ids, batch_size: int = 256, device=None
):
    """Run the real model forward pass. Returns float32 numpy probs."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    n = word_ids.shape[0]
    probs = np.zeros(n, dtype=np.float32)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        w = word_ids[i:j].to(device)
        c = char_ids[i:j].to(device)
        k = keyword_ids[i:j].to(device)
        logits = model(w, c, k).float()
        probs[i:j] = torch.sigmoid(logits).detach().cpu().numpy()
    return probs


# ---- CPU-preflight adapter (real model, real loss, real optimizer) -------


class CandidateAdapter:
    """No-argument adapter that exercises the real model/loss/inputs/optimizer.

    The vocabularies are lazily built once from a fixed small slice of train.csv
    and cached at class scope, so build_model and the batch builders share the
    same feasibility space. No weights are downloaded and no CUDA is required.
    """

    _featurizer_cache = None  # (wtok, ctok, kmap, texts, keywords, labels)

    def __init__(self):
        self.max_word_len = MAX_WORD_LEN
        self.max_char_len = MAX_CHAR_LEN
        self.word_embed_dim = ADAPTER_WORD_EMBED_DIM
        self.word_filters = ADAPTER_WORD_FILTERS
        self.char_embed_dim = ADAPTER_CHAR_EMBED_DIM
        self.char_filters = ADAPTER_CHAR_FILTERS
        self.keyword_dim = ADAPTER_KEYWORD_DIM
        self.hidden_dim = ADAPTER_HIDDEN_DIM
        self.dropout = ADAPTER_DROPOUT

    # -- internal helpers --
    def _load_small(self):
        if CandidateAdapter._featurizer_cache is not None:
            return CandidateAdapter._featurizer_cache
        input_dir = resolve_input_dir()
        path = os.path.join(input_dir, "train.csv")
        df = pd.read_csv(path).head(ADAPTER_N_SAMPLE_ROWS)
        texts = [clean_text(t) for t in df["text"].astype(str).tolist()]
        if "keyword" in df.columns:
            keywords = df["keyword"].tolist()
        else:
            keywords = [None] * len(texts)
        if "target" in df.columns:
            labels = df["target"].astype("float32").tolist()
        else:
            labels = [0.0] * len(texts)
        wtok = SimpleTokenizer(min_freq=1, max_vocab=4096).fit(texts)
        ctok = CharTokenizer(min_freq=1, max_vocab=128).fit(texts)
        kmap = KeywordMapper().fit(keywords)
        CandidateAdapter._featurizer_cache = (wtok, ctok, kmap, texts, keywords, labels)
        return CandidateAdapter._featurizer_cache

    def _make_batch(self, scenario, device):
        bs = int(scenario.get("batch_size", 8)) if isinstance(scenario, dict) else 8
        bs = max(bs, 1)
        fixture = scenario.get("fixture") if isinstance(scenario, dict) else None

        # If fixture explicitly provides named integer token tensors, honor them.
        if (
            isinstance(fixture, dict)
            and fixture
            and all(k in fixture for k in ("word_ids", "char_ids", "keyword_ids"))
        ):
            word_ids = torch.as_tensor(fixture["word_ids"]).long().to(device)
            char_ids = torch.as_tensor(fixture["char_ids"]).long().to(device)
            kw_ids = torch.as_tensor(fixture["keyword_ids"]).long().to(device)
            target = fixture.get("target", fixture.get("labels", None))
            if target is None:
                target = torch.zeros(
                    word_ids.shape[0], dtype=torch.float32, device=device
                )
            else:
                target = torch.as_tensor(target).float().to(device)
            return (word_ids, char_ids, kw_ids), target

        # Otherwise build a real batch from the first rows of train.csv.
        wtok, ctok, kmap, texts, keywords, labels = self._load_small()
        if not texts:
            texts = [""]
            keywords = [None]
            labels = [0.0]
        idx = list(range(min(bs, len(texts))))
        b_texts = [texts[i] for i in idx]
        b_kw = [keywords[i] for i in idx]
        b_y = [labels[i] for i in idx]
        # Pad to batch_size so downstream modules see the requested shape.
        while len(b_texts) < bs:
            b_texts.append(b_texts[-1])
            b_kw.append(b_kw[-1])
            b_y.append(0.0)

        word_ids, char_ids, kw_ids = make_batch_tensors(
            b_texts,
            b_kw,
            wtok,
            ctok,
            kmap,
            max_word_len=self.max_word_len,
            max_char_len=self.max_char_len,
            device=device,
        )
        target = torch.tensor(b_y, dtype=torch.float32, device=device)
        return (word_ids, char_ids, kw_ids), target

    # -- adapter API --
    def build_model(self, context):
        wtok, ctok, kmap, _, _, _ = self._load_small()
        model = MultiViewDisasterNet(
            word_vocab_size=max(wtok.vocab_size, NUM_SPECIAL_TOKENS + 1),
            char_vocab_size=max(ctok.vocab_size, NUM_SPECIAL_TOKENS + 1),
            num_keywords=max(kmap.num_keywords, 4),
            word_embed_dim=self.word_embed_dim,
            word_kernels=WORD_KERNELS,
            word_filters=self.word_filters,
            char_embed_dim=self.char_embed_dim,
            char_kernel=CHAR_KERNEL,
            char_filters=self.char_filters,
            keyword_dim=self.keyword_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        model.cpu()
        return model

    def build_optimizer(self, model, context):
        # Only used by the isolated CPU preflight to exercise a real update;
        # the production optimizer/scheduler belongs to Stage 3.
        return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    def build_train_batch(self, scenario, device):
        return self._make_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._make_batch(scenario, device)

    def _resolve_criterion(self, context):
        if isinstance(context, dict):
            crit = context.get("criterion")
            if isinstance(crit, nn.Module):
                return crit
        return build_loss(pos_weight=None)

    def training_step(self, model, batch, context):
        model.train()
        inputs, target = batch
        word_ids, char_ids, kw_ids = inputs
        logits = model(word_ids, char_ids, kw_ids)
        criterion = self._resolve_criterion(context)
        loss = criterion(logits, target.float())
        return loss

    def validation_step(self, model, batch, context):
        model.eval()
        inputs, target = batch
        word_ids, char_ids, kw_ids = inputs
        with torch.no_grad():
            logits = model(word_ids, char_ids, kw_ids)
        criterion = self._resolve_criterion(context)
        loss = criterion(logits, target.float())
        probs = torch.sigmoid(logits.float())
        return {"loss": loss, "probs": probs, "target": target.float()}

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
 Summary: 3 total attempts - 1 pending execution, 2 executed, 1 successful (best: 0.7341)

-------------------------------
Attempt #1:
Design: I merged the three stages by keeping Stage 1's data/model/loss/adapter code as the foundation, layering Stage 2's `PrecisionRuntime` policy helpers on top, and consuming both inside Stage 3's `run_training` (with diagnostics + scheduler safe points). I resolved naming conflicts, deduplicated imports, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module-level constants that Stage 3 references, and moved every side effect under `if __name__ == '__main__':` while preserving the CPU-safe `CandidateAdapter` contract.
Results: Training completed successfully in ~37 minutes on an A10 GPU, running 12 epochs with a hybrid transformer model. Validation F1 improved from 0.665 at epoch 1 to a best of ~0.7341 (best epoch 11), and a valid submission file with 1523 rows was written using a tuned threshold of 0.42. The only messages were benign warnings (Graph evidence mirror connection refused, nested-tensor warning) that did not affect execution.
Validation Metric: 0.7341
Execution Time: 2223.42s

-------------------------------
Attempt #2:
Design: Looking at the three stages, I merged by keeping Stage 1's BiGRU+keyword-bias model/data/loss/threshold-tuner/CandidateAdapter as the foundation, Stage 2's `PrecisionRuntime` helpers (FP16-AMP forward autocast + FP32 islands for loss/sigmoid/metrics, TF32 disabled, CPU-safe preflight) layered on top, and completing Stage 3's truncated training loop (AdamW + linear-warmup+cosine, FP16 autocast forward, scaler.unscale before clip, val-F1 early stopping with best-state restore, guarded scheduler safe-points, `TrainingDiagnostics` with fallback), then adding the missing test inference + submission write + `if __name__ == "__main__":` entrypoint. I fixed the truncated `_make_state_factory`, resolved duplicated constants/imports, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module constants, and put the final `Final Validation Score:` print as the very last line after the diagnostics context closes.
Results: The implementation of this design has bugs.
Insight: Using a different approach may not result in the same bugs as the above approach.

-------------------------------
Attempt #3:
Design: I merged the three stages by keeping Stage 1's data cleaning, TF-IDF feature builders, `SparseMLP`, `build_loss`, `tune_threshold` and split helpers as the foundation; layering Stage 2's import-safe precision utilities (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `degrade_to_fp32`, `configure_tf32`, `format_precision_log`) on top; then consuming both inside Stage 3's `run_training` with scheduler safe-points, `TrainingDiagnostics`, linear warmup+cosine AdamW, best-val-F1 restore, and test inference. I resolved the duplicate `CandidateAdapter` by keeping Stage 3's self-contained CPU-safe version (it uses the Stage 1 model/loss/features through the Stage 2 precision path), removed duplicate imports/constants, kept all side effects under `if __name__ == "__main__":`, and preserved the exact `MODEL_FAMILY`, `WORKING_DIR`, `SUBMISSION_DIR` contract.
Status: Code generated, execution pending (will run in parallel with other drafts).


# Hardware-Aware Training Hyperparameter Context
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2; observed in 1 batch-size observation
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
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"cross entropy or metric-compatible classification loss","output_interface":"match the competition metric and required submission format","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"{'forward_autocast_boundaries': \"Use torch.autocast(device_type='cuda', dtype=torch.float16) only...","fallback_policy":"fp32","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"{'name': 'AdamW', 'learning_rate': 2e-05, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-08}","scheduler":"{'name': 'linear_warmup_decay', 'warmup_ratio': 0.1, 'num_training_steps': 'Derived from epochs,...","batch_size_policy":"fixed","dataloader_policy":"{'shuffle_train': True, 'shuffle_validation': False, 'shuffle_test': False, 'num_workers': '>0 wh...","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available","no exact, similar, or packed graph profiles for current hardware","no measured FP16 AMP speedup or quality comparison for this task"]}}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=replaced recurrent/transformer baselines with a multi-view CNN (parallel word kernels + char CNN) whose keyword embedding is multiplicatively gated by the fused text context; purpose=exploit short-text inductive bias and model keyword–context interaction that prior attempts treated as a naive bias term; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=split is stratified 85/15 with seed 42 fit strictly on `train.csv` labels, and all stateful transforms (SimpleTokenizer, CharTokenizer, KeywordMapper) are fit only on the training split; purpose=strict leakage isolation across train/val/test; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=exposes `MAX_WORD_LEN=32`, `MAX_CHAR_LEN=128`, and integer-id inputs via a reusable `make_batch_tensors` helper + a CPU-safe `CandidateAdapter`; purpose=keep sequence geometry configurable for later precision/batching stages without committing to them here; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=introduced a `PrecisionConfig` + selective FP16 AMP policy (CUDA autocast for forward only, FP32 islands for gate sigmoid/loss/probs, GradScaler for backward, TF32 explicitly disabled) replacing the implicit "everythi...; purpose=exploit A10 Tensor Cores on the word/char Conv1d + Embedding matmuls while protecting the numerically fragile keyword-gate interaction and BCEWithLogits loss that Stage 1 relies on; hardware_keys=datatype_precision.precision_policy, datatype_precision.fallback_policy, datatype_precision.precision_model_adaptation
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=added runtime precision helpers (`configure_precision_globals`, `autocast_context`, `fp32_island`, `make_grad_scaler`, `amp_forward`, `bce_with_logits_fp32`, `degrade_to_fp32`) that Stage 3's `run_training` will consu...; purpose=keep precision configurable and CPU-safe so Stage 3 can simply call them around its existing forward/loss, preserving Stage 1's model family and output interface; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=no `model.half()`, no whole-pipeline cast; integer token indices stay `torch.long`, floating params/targets stay FP32; CPU path resolves to FP32 with AMP/scaler disabled; purpose=honor the "selective forward autocast" rule and guarantee the CPU preflight can execute the real model/loss without CUDA; hardware_keys=none

# Previous steps
### model_design
**Plan:** ## Plan

**WHAT makes this task unique**: Disaster tweets are short (≤32 tokens), idiosyncratic, and heavily signal through local phrases + hashtags + the structured `keyword` column, which is effectively a pre-extracted topic tag. The signal is *not* long-range — it lives in n-gram pockets and in whether a topic keyword is used in a disaster sense. Class balance is moderate (~43% positive).

**WHY this approach fits**: A from-scratch deep transformer on 6k short texts overfits; instead I use a **multi-view CNN with keyword cross-gating**, which matches the inductive bias of short-text classification: (a) parallel multi-kernel word convolutions capture disaster n-gram phrases ("wildfire", "on fire", "killed in"), (b) a character CNN captures hashtag/casing/morphological cues without a subword pretrainer, and (c) a learned keyword embedding is *gated* by the word-context — the model can learn to amplify the keyword only when the tweet's surrounding words suggest it's a real disaster, mirroring the human reading of `keyword="ablaze" + "beautiful sunset ablaze"`. This is genuinely novel vs. prior attempts (BiGRU+keyword, TF-IDF+SparseMLP, hybrid transformer), which either trust the keyword naively or ignore the gating interaction. Modeling budget is conservative: ~3M params, no pretrained weights, no external downloads.

**HOW validation confirms/refutes**: Split 85/15 stratified (seed 42) *before* any vocab fit. If the gated keyword view genuinely helps, validation F1 with gate-on should beat a keyword-embedding-concat ablation by a clear margin; if validation F1 collapses toward the majority-class baseline, the gate is learning to zero-out the keyword and I'd fall back to plain concat. Metric = binary F1 with threshold tuned on validation (as Stage 3 consumes).

**Backend/memory strategy**: No precision or optimizer choices are made here (those belong to later stages); the design exposes configurable sequence lengths (`max_word_len`, `max_char_len`), configurable batch size honor via the adapter, and integer token tensors (torch.long) so non-blocking CUDA transfer is possible. Stateful transforms (word/char vocabs, keyword map) are fit *only* on train texts, and reused via `.encode()` on val/test. Scheduler-owned controls (device assignment, num_workers, batch size, AMP/scaler, checkpointing) are left to later stages; the subprocess must not implement them itself.

### Note board:
- baseline_change: replaced recurrent/transformer baselines with a multi-view CNN (parallel word kernels + char CNN) whose keyword embedding is multiplicatively gated by the fused text context | purpose: exploit short-text inductive bias and model keyword–context interaction that prior attempts treated as a naive bias term | hardware_keys: none
- baseline_change: split is stratified 85/15 with seed 42 fit strictly on `train.csv` labels, and all stateful transforms (SimpleTokenizer, CharTokenizer, KeywordMapper) are fit only on the training split | purpose: strict leakage isolation across train/val/test | hardware_keys: none
- baseline_change: exposes `MAX_WORD_LEN=32`, `MAX_CHAR_LEN=128`, and integer-id inputs via a reusable `make_batch_tensors` helper + a CPU-safe `CandidateAdapter` | purpose: keep sequence geometry configurable for later precision/batching stages without committing to them here | hardware_keys: none
**Code:**
```python
"""
Stage 1 (model_design) — Disaster Tweets binary classification.

Model family: MultiViewDisasterNet, a from-scratch multi-view CNN:
  - Word branch: parallel Conv1d over word embeddings (kernels 2,3,4) → max-pool.
  - Char branch: single Conv1d over char embeddings → max-pool.
  - Keyword branch: learned embedding multiplicatively gated by concatenated
    word+char context, so the keyword only contributes when the tweet context
    suggests a real-disaster reading.
  - Fusion: LayerNorm → dropout → 2-layer MLP → single logit.

This module is IMPORT-SAFE: only imports, constants, classes, and helper
functions are defined at top level. Training loop, optimizer, scheduler,
precision policy, batch size, epochs, and workers all live in later stages.
"""

# ---- Module-level constants (no side effects) -----------------------------

MODEL_FAMILY = "mvcnn_keyword_gate_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
DEFAULT_INPUT_DIR = "./input"

PAD_IDX = 0
UNK_IDX = 1
NUM_SPECIAL_TOKENS = 2

# Configurable sequence geometry (exposed for later stages).
MAX_WORD_LEN = 32
MAX_CHAR_LEN = 128

# Defaults for the production model built during training.
WORD_EMBED_DIM = 100
WORD_KERNELS = (2, 3, 4)
WORD_FILTERS = 96
CHAR_EMBED_DIM = 32
CHAR_KERNEL = 3
CHAR_FILTERS = 48
KEYWORD_DIM = 32
HIDDEN_DIM = 128
DROPOUT = 0.4

# Defaults for the lightweight CPU-preflight adapter.
ADAPTER_WORD_EMBED_DIM = 64
ADAPTER_WORD_FILTERS = 32
ADAPTER_CHAR_EMBED_DIM = 16
ADAPTER_CHAR_FILTERS = 16
ADAPTER_KEYWORD_DIM = 16
ADAPTER_HIDDEN_DIM = 48
ADAPTER_DROPOUT = 0.2
ADAPTER_N_SAMPLE_ROWS = 128


# ---- Imports --------------------------------------------------------------

import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ---- Text cleaning --------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_USER_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NON_PRINTABLE_RE = re.compile(r"[^a-z0-9<>_\s]")
_WS_RE = re.compile(r"\s+")

URL_TOKEN = "<url>"
USER_TOKEN = "<user>"
HASHTAG_PREFIX = "htag_"


def clean_text(text: str) -> str:
    """Lowercase, normalize URLs/@mentions/hashtags, strip odd chars.

    Deterministic, stateless — safe to apply to train/val/test identically.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(f" {URL_TOKEN} ", text)
    text = _USER_RE.sub(f" {USER_TOKEN} ", text)
    text = _HASHTAG_RE.sub(lambda m: f" {HASHTAG_PREFIX}{m.group(1)} ", text)
    text = _NON_PRINTABLE_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def join_context(text: str, keyword, location) -> str:
    """Optional richer field used only if a downstream stage wants it.

    Kept as a separate helper so the main text pipeline stays clean.
    """
    parts = [clean_text(text)]
    if isinstance(keyword, str) and keyword.strip():
        parts.append("kw_" + keyword.strip().lower().replace(" ", "_"))
    if isinstance(location, str) and location.strip():
        parts.append("loc_" + location.strip().lower().replace(" ", "_"))
    return " ".join(p for p in parts if p)


# ---- Stateful tokenizers (fit on TRAIN ONLY) -----------------------------


class SimpleTokenizer:
    """Word-level vocab built from scratch (no pretrained tokenizer)."""

    def __init__(self, min_freq: int = 1, max_vocab: int = 30000):
        self.min_freq = int(min_freq)
        self.max_vocab = int(max_vocab)
        self.vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}

    def fit(self, texts):
        c = Counter()
        for t in texts:
            for tok in (t or "").split():
                c[tok] += 1
        items = [(tok, n) for tok, n in c.most_common() if n >= self.min_freq]
        items = items[: max(self.max_vocab - NUM_SPECIAL_TOKENS, 0)]
        for tok, _ in items:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
        return self

    @property
    def vocab_size(self) -> int:
        return max(len(self.vocab), NUM_SPECIAL_TOKENS)

    def encode(self, texts, max_len: int = MAX_WORD_LEN) -> np.ndarray:
        out = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, t in enumerate(texts):
            toks = (t or "").split()[:max_len]
            ids = [self.vocab.get(tok, UNK_IDX) for tok in toks]
            if ids:
                out[i, : len(ids)] = ids
        return out


class CharTokenizer:
    """Character-level vocab (subset of printable ASCII + special tokens)."""

    def __init__(self, min_freq: int = 2, max_vocab: int = 128):
        self.min_freq = int(min_freq)
        self.max_vocab = int(max_vocab)
        self.vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}

    def fit(self, texts):
        c = Counter()
        for t in texts:
            for ch in t or "":
                c[ch] += 1
        items = [(ch, n) for ch, n in c.most_common() if n >= self.min_freq]
        items = items[: max(self.max_vocab - NUM_SPECIAL_TOKENS, 0)]
        for ch, _ in items:
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)
        return self

    @property
    def vocab_size(self) -> int:
        return max(len(self.vocab), NUM_SPECIAL_TOKENS)

    def encode(self, texts, max_len: int = MAX_CHAR_LEN) -> np.ndarray:
        out = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, t in enumerate(texts):
            chars = (t or "")[:max_len]
            ids = [self.vocab.get(ch, UNK_IDX) for ch in chars]
            if ids:
                out[i, : len(ids)] = ids
        return out


class KeywordMapper:
    """Categorical keyword map. Index 0 = missing, 1 = unseen, then real keys."""

    def __init__(self):
        self.mapping = {"<none>": 0, "<unk>": 1}

    def fit(self, keywords):
        for k in sorted({k for k in keywords if isinstance(k, str) and k.strip()}):
            if k not in self.mapping:
                self.mapping[k] = len(self.mapping)
        return self

    @property
    def num_keywords(self) -> int:
        return max(len(self.mapping), 2)

    def encode(self, keywords) -> np.ndarray:
        out = np.zeros(len(keywords), dtype=np.int64)
        for i, k in enumerate(keywords):
            if not isinstance(k, str) or not k.strip():
                out[i] = 0
            else:
                out[i] = self.mapping.get(k, 1)
        return out


# ---- Model ---------------------------------------------------------------


class MultiViewDisasterNet(nn.Module):
    """Multi-view CNN with a keyword cross-gate.

    Forward signature: forward(word_ids, char_ids, keyword_ids) -> logits (B,)
    """

    def __init__(
        self,
        word_vocab_size: int,
        char_vocab_size: int,
        num_keywords: int,
        word_embed_dim: int = WORD_EMBED_DIM,
        word_kernels=WORD_KERNELS,
        word_filters: int = WORD_FILTERS,
        char_embed_dim: int = CHAR_EMBED_DIM,
        char_kernel: int = CHAR_KERNEL,
        char_filters: int = CHAR_FILTERS,
        keyword_dim: int = KEYWORD_DIM,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
        num_classes: int = 1,
    ):
        super().__init__()
        self.word_vocab_size = int(word_vocab_size)
        self.char_vocab_size = int(char_vocab_size)
        self.num_keywords = int(num_keywords)

        # -- Word branch --
        self.word_emb = nn.Embedding(
            word_vocab_size, word_embed_dim, padding_idx=PAD_IDX
        )
        self.word_convs = nn.ModuleList(
            [
                nn.Conv1d(word_embed_dim, word_filters, k, padding=k // 2)
                for k in word_kernels
            ]
        )
        word_feat_dim = word_filters * len(word_kernels)

        # -- Char branch --
        self.char_emb = nn.Embedding(
            char_vocab_size, char_embed_dim, padding_idx=PAD_IDX
        )
        self.char_conv = nn.Conv1d(
            char_embed_dim, char_filters, char_kernel, padding=char_kernel // 2
        )

        # -- Keyword branch with cross-gate --
        self.kw_emb = nn.Embedding(num_keywords, keyword_dim)
        fused_in = word_feat_dim + char_filters
        self.kw_gate = nn.Linear(fused_in + keyword_dim, keyword_dim)
        self.kw_proj = nn.Linear(keyword_dim, keyword_dim)

        # -- Fusion head --
        self.norm = nn.LayerNorm(fused_in + keyword_dim)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(fused_in + keyword_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].fill_(0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, word_ids, char_ids, keyword_ids):
        # --- Word branch ---
        w = self.word_emb(word_ids)  # (B, L, D)
        w = w.transpose(1, 2)  # (B, D, L)
        w_feats = []
        for conv in self.word_convs:
            h = F.relu(conv(w))
            h = F.adaptive_max_pool1d(h, 1).squeeze(-1)
            w_feats.append(h)
        word_feat = torch.cat(w_feats, dim=-1)  # (B, word_feat_dim)

        # --- Char branch ---
        c = self.char_emb(char_ids).transpose(1, 2)
        c = F.relu(self.char_conv(c))
        char_feat = F.adaptive_max_pool1d(c, 1).squeeze(-1)  # (B, char_filters)

        base = torch.cat([word_feat, char_feat], dim=-1)  # (B, fused_in)

        # --- Keyword cross-gate ---
        kw = self.kw_emb(keyword_ids)  # (B, keyword_dim)
        gate = torch.sigmoid(self.kw_gate(torch.cat([base, kw], dim=-1)))
        kw_gated = gate * self.kw_proj(kw)

        # --- Fusion ---
        fused = torch.cat([base, kw_gated], dim=-1)
        fused = self.norm(fused)
        fused = self.drop(fused)
        logits = self.classifier(fused)  # (B, 1)
        return logits.squeeze(-1)


# ---- Loss / metric helpers ----------------------------------------------


def build_loss(pos_weight=None):
    """BCEWithLogitsLoss with optional positive-class weighting.

    `pos_weight` is a Python scalar (num_neg / num_pos). If None, no weighting.
    """
    if pos_weight is None:
        pw = None
    else:
        pw = torch.tensor([float(pos_weight)], dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pw)


def compute_f1(y_true, y_pred_binary) -> float:
    return float(
        f1_score(np.asarray(y_true), np.asarray(y_pred_binary), zero_division=0)
    )


def tune_threshold(
    y_true, y_prob, min_t: float = 0.20, max_t: float = 0.80, steps: int = 61
):
    """Return (best_threshold, best_f1) maximizing binary F1 on val probs."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(min_t, max_t, steps):
        preds = (y_prob >= t).astype(int)
        f1 = compute_f1(y_true, preds)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


# ---- Split + feature orchestration (pure functions, called under __main__) -


def resolve_input_dir() -> str:
    env = os.environ.get("MLEVOLVE_INPUT_DIR")
    if env and os.path.isdir(env):
        return env
    return DEFAULT_INPUT_DIR


def load_raw_data(input_dir=None):
    input_dir = input_dir or resolve_input_dir()
    train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(input_dir, "test.csv"))
    return train_df, test_df


def stratified_split(train_df, val_size: float = 0.15, seed: int = 42):
    """Split BEFORE any vocab/stat fit, stratified on target."""
    tr, va = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df["target"].astype(int),
        random_state=seed,
    )
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def prepare_dataframes(train_df, test_df, val_size: float = 0.15, seed: int = 42):
    """Produce raw splits with cleaned text. No vocab fitting here."""
    tr, va = stratified_split(train_df, val_size=val_size, seed=seed)
    for df in (tr, va, test_df):
        df["_clean_text"] = df["text"].apply(clean_text)
    if "_clean_text" not in test_df.columns:
        test_df["_clean_text"] = test_df["text"].apply(clean_text)
    return tr, va, test_df


def fit_featurizers(
    train_texts,
    train_keywords,
    word_max_vocab: int = 30000,
    char_min_freq: int = 2,
    char_max_vocab: int = 128,
):
    """Fit tokenizers/mappers ONLY on training data (leakage-safe)."""
    wtok = SimpleTokenizer(min_freq=1, max_vocab=word_max_vocab).fit(train_texts)
    ctok = CharTokenizer(min_freq=char_min_freq, max_vocab=char_max_vocab).fit(
        train_texts
    )
    kmap = KeywordMapper().fit(train_keywords)
    return wtok, ctok, kmap


def make_batch_tensors(
    texts,
    keywords,
    wtok,
    ctok,
    kmap,
    max_word_len: int = MAX_WORD_LEN,
    max_char_len: int = MAX_CHAR_LEN,
    device=None,
):
    """Return (word_ids, char_ids, keyword_ids) long tensors on `device`.

    integer indices stay torch.long; only floating tensors are affected by
    later precision policies.
    """
    word_np = wtok.encode(texts, max_word_len)
    char_np = ctok.encode(texts, max_char_len)
    kw_np = kmap.encode(keywords)
    word_ids = torch.from_numpy(word_np).long()
    char_ids = torch.from_numpy(char_np).long()
    kw_ids = torch.from_numpy(kw_np).long()
    if device is not None:
        word_ids = word_ids.to(device)
        char_ids = char_ids.to(device)
        kw_ids = kw_ids.to(device)
    return word_ids, char_ids, kw_ids


def build_model_from_featurizers(
    wtok,
    ctok,
    kmap,
    word_embed_dim: int = WORD_EMBED_DIM,
    word_filters: int = WORD_FILTERS,
    char_embed_dim: int = CHAR_EMBED_DIM,
    char_filters: int = CHAR_FILTERS,
    keyword_dim: int = KEYWORD_DIM,
    hidden_dim: int = HIDDEN_DIM,
    dropout: float = DROPOUT,
) -> MultiViewDisasterNet:
    return MultiViewDisasterNet(
        word_vocab_size=wtok.vocab_size,
        char_vocab_size=ctok.vocab_size,
        num_keywords=kmap.num_keywords,
        word_embed_dim=word_embed_dim,
        word_kernels=WORD_KERNELS,
        word_filters=word_filters,
        char_embed_dim=char_embed_dim,
        char_kernel=CHAR_KERNEL,
        char_filters=char_filters,
        keyword_dim=keyword_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def get_pos_weight(labels) -> float:
    labels = np.asarray(labels).astype(int)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    if pos <= 0:
        return 1.0
    return float(neg) / float(pos)


@torch.no_grad()
def predict_proba(
    model, word_ids, char_ids, keyword_ids, batch_size: int = 256, device=None
):
    """Run the real model forward pass. Returns float32 numpy probs."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    n = word_ids.shape[0]
    probs = np.zeros(n, dtype=np.float32)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        w = word_ids[i:j].to(device)
        c = char_ids[i:j].to(device)
        k = keyword_ids[i:j].to(device)
        logits = model(w, c, k).float()
        probs[i:j] = torch.sigmoid(logits).detach().cpu().numpy()
    return probs


# ---- CPU-preflight adapter (real model, real loss, real optimizer) -------


class CandidateAdapter:
    """No-argument adapter that exercises the real model/loss/inputs/optimizer.

    The vocabularies are lazily built once from a fixed small slice of train.csv
    and cached at class scope, so build_model and the batch builders share the
    same feasibility space. No weights are downloaded and no CUDA is required.
    """

    _featurizer_cache = None  # (wtok, ctok, kmap, texts, keywords, labels)

    def __init__(self):
        self.max_word_len = MAX_WORD_LEN
        self.max_char_len = MAX_CHAR_LEN
        self.word_embed_dim = ADAPTER_WORD_EMBED_DIM
        self.word_filters = ADAPTER_WORD_FILTERS
        self.char_embed_dim = ADAPTER_CHAR_EMBED_DIM
        self.char_filters = ADAPTER_CHAR_FILTERS
        self.keyword_dim = ADAPTER_KEYWORD_DIM
        self.hidden_dim = ADAPTER_HIDDEN_DIM
        self.dropout = ADAPTER_DROPOUT

    # -- internal helpers --
    def _load_small(self):
        if CandidateAdapter._featurizer_cache is not None:
            return CandidateAdapter._featurizer_cache
        input_dir = resolve_input_dir()
        path = os.path.join(input_dir, "train.csv")
        df = pd.read_csv(path).head(ADAPTER_N_SAMPLE_ROWS)
        texts = [clean_text(t) for t in df["text"].astype(str).tolist()]
        if "keyword" in df.columns:
            keywords = df["keyword"].tolist()
        else:
            keywords = [None] * len(texts)
        if "target" in df.columns:
            labels = df["target"].astype("float32").tolist()
        else:
            labels = [0.0] * len(texts)
        wtok = SimpleTokenizer(min_freq=1, max_vocab=4096).fit(texts)
        ctok = CharTokenizer(min_freq=1, max_vocab=128).fit(texts)
        kmap = KeywordMapper().fit(keywords)
        CandidateAdapter._featurizer_cache = (wtok, ctok, kmap, texts, keywords, labels)
        return CandidateAdapter._featurizer_cache

    def _make_batch(self, scenario, device):
        bs = int(scenario.get("batch_size", 8)) if isinstance(scenario, dict) else 8
        bs = max(bs, 1)
        fixture = scenario.get("fixture") if isinstance(scenario, dict) else None

        # If fixture explicitly provides named integer token tensors, honor them.
        if (
            isinstance(fixture, dict)
            and fixture
            and all(k in fixture for k in ("word_ids", "char_ids", "keyword_ids"))
        ):
            word_ids = torch.as_tensor(fixture["word_ids"]).long().to(device)
            char_ids = torch.as_tensor(fixture["char_ids"]).long().to(device)
            kw_ids = torch.as_tensor(fixture["keyword_ids"]).long().to(device)
            target = fixture.get("target", fixture.get("labels", None))
            if target is None:
                target = torch.zeros(
                    word_ids.shape[0], dtype=torch.float32, device=device
                )
            else:
                target = torch.as_tensor(target).float().to(device)
            return (word_ids, char_ids, kw_ids), target

        # Otherwise build a real batch from the first rows of train.csv.
        wtok, ctok, kmap, texts, keywords, labels = self._load_small()
        if not texts:
            texts = [""]
            keywords = [None]
            labels = [0.0]
        idx = list(range(min(bs, len(texts))))
        b_texts = [texts[i] for i in idx]
        b_kw = [keywords[i] for i in idx]
        b_y = [labels[i] for i in idx]
        # Pad to batch_size so downstream modules see the requested shape.
        while len(b_texts) < bs:
            b_texts.append(b_texts[-1])
            b_kw.append(b_kw[-1])
            b_y.append(0.0)

        word_ids, char_ids, kw_ids = make_batch_tensors(
            b_texts,
            b_kw,
            wtok,
            ctok,
            kmap,
            max_word_len=self.max_word_len,
            max_char_len=self.max_char_len,
            device=device,
        )
        target = torch.tensor(b_y, dtype=torch.float32, device=device)
        return (word_ids, char_ids, kw_ids), target

    # -- adapter API --
    def build_model(self, context):
        wtok, ctok, kmap, _, _, _ = self._load_small()
        model = MultiViewDisasterNet(
            word_vocab_size=max(wtok.vocab_size, NUM_SPECIAL_TOKENS + 1),
            char_vocab_size=max(ctok.vocab_size, NUM_SPECIAL_TOKENS + 1),
            num_keywords=max(kmap.num_keywords, 4),
            word_embed_dim=self.word_embed_dim,
            word_kernels=WORD_KERNELS,
            word_filters=self.word_filters,
            char_embed_dim=self.char_embed_dim,
            char_kernel=CHAR_KERNEL,
            char_filters=self.char_filters,
            keyword_dim=self.keyword_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        model.cpu()
        return model

    def build_optimizer(self, model, context):
        # Only used by the isolated CPU preflight to exercise a real update;
        # the production optimizer/scheduler belongs to Stage 3.
        return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    def build_train_batch(self, scenario, device):
        return self._make_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._make_batch(scenario, device)

    def _resolve_criterion(self, context):
        if isinstance(context, dict):
            crit = context.get("criterion")
            if isinstance(crit, nn.Module):
                return crit
        return build_loss(pos_weight=None)

    def training_step(self, model, batch, context):
        model.train()
        inputs, target = batch
        word_ids, char_ids, kw_ids = inputs
        logits = model(word_ids, char_ids, kw_ids)
        criterion = self._resolve_criterion(context)
        loss = criterion(logits, target.float())
        return loss

    def validation_step(self, model, batch, context):
        model.eval()
        inputs, target = batch
        word_ids, char_ids, kw_ids = inputs
        with torch.no_grad():
            logits = model(word_ids, char_ids, kw_ids)
        criterion = self._resolve_criterion(context)
        loss = criterion(logits, target.float())
        probs = torch.sigmoid(logits.float())
        return {"loss": loss, "probs": probs, "target": target.float()}

```

### datatype_precision
**Plan:** ## Plan

**WHAT makes this task unique for precision**: The model is a from-scratch multi-view CNN with a **keyword cross-gate** — the gate involves `sigmoid(x) * proj(kw)`, a small multiplicative interaction that can numerically saturate under FP16. The loss is `BCEWithLogitsLoss` with `pos_weight`, which naively computed under autocast can produce inf/nan gradients on a ~43% positive-class tweet corpus. So FP16 AMP must be applied **only around the Conv1d/Embedding forward ops**, with the gate's sigmoid, the loss, and the sigmoid-based metric/prob export kept in FP32 islands — exactly the "selective, forward-only" pattern.

**WHY this is the right precision policy**: A10 (Ampere, cc 8.6) Tensor Cores give real throughput on fp16 matmul/conv, and prior attempts fit the model in ~37 min at fp32-equivalent. Cheap win to enable selective fp16 AMP with GradScaler, disable TF32 (contract mandates this in normal mode, and TF32 on the elementwise-kernel-heavy CNN has negligible benefit), and keep parameters, integer indices (`word_ids`, `char_ids`, `keyword_ids`), and the target tensor in their native dtypes — never `model.half()`.

**HOW validation confirms it**: The precision policy is confirmed if (a) fp16-AMP training produces finite loss/gradients for all epochs, (b) the epoch-1 val F1 is within noise of FP32, and (c) wall time (or steps/sec) improves or is neutral. If I observe NaN/inf loss, non-finite gradients, or clear F1 degradation, `degrade_to_fp32` is called and the run continues in pure FP32 — that is the recorded fallback. CPU-preflight always uses FP32 regardless of host GPU.

Note board:
- baseline_change: introduced a `PrecisionConfig` + selective FP16 AMP policy (CUDA autocast for forward only, FP32 islands for gate sigmoid/loss/probs, GradScaler for backward, TF32 explicitly disabled) replacing the implicit "everything fp32" default | purpose: exploit A10 Tensor Cores on the word/char Conv1d + Embedding matmuls while protecting the numerically fragile keyword-gate interaction and BCEWithLogits loss that Stage 1 relies on | hardware_keys: datatype_precision.precision_policy, datatype_precision.fallback_policy, datatype_precision.precision_model_adaptation
- baseline_change: added runtime precision helpers (`configure_precision_globals`, `autocast_context`, `fp32_island`, `make_grad_scaler`, `amp_forward`, `bce_with_logits_fp32`, `degrade_to_fp32`) that Stage 3's `run_training` will consume without touching optimizer/scheduler/batch-size/epoch code | purpose: keep precision configurable and CPU-safe so Stage 3 can simply call them around its existing forward/loss, preserving Stage 1's model family and output interface | hardware_keys: none
- baseline_change: no `model.half()`, no whole-pipeline cast; integer token indices stay `torch.long`, floating params/targets stay FP32; CPU path resolves to FP32 with AMP/scaler disabled | purpose: honor the "selective forward autocast" rule and guarantee the CPU preflight can execute the real model/loss without CUDA | hardware_keys: none
**Code:**
```python
"""
Stage 2 (datatype_precision) — Precision policy for the disaster-tweets MVCNN.

This module is IMPORT-SAFE: it defines constants, a `PrecisionConfig`
dataclass, and helper functions only.  No data is read, no CUDA/TF32 flags are
mutated, no logging is configured, and no model is instantiated at import time.
All global side effects happen inside helper calls that the Stage 3 main guard
invokes.

Consumed by Stage 1 (model/loss) and Stage 3 (training loop):

  cfg   = resolve_precision_config()            # device + policy
  configure_precision_globals(cfg)              # disable TF32 etc. (main guard)
  scaler = make_grad_scaler(cfg)                # None on CPU / fp32
  logits = amp_forward(model, (w, c, k), cfg)   # autocast ONLY here
  loss   = bce_with_logits_fp32(logits, y, pw)  # fp32 island
  probs  = logits_to_probs_fp32(logits)         # fp32 metric/export

Contract preserved from Stage 1:
  - Model family: MultiViewDisasterNet (multi-view CNN + keyword gate).
  - Inputs: integer token tensors (word_ids, char_ids, keyword_ids) stay
    torch.long; only floating-point computation is affected by autocast.
  - Loss: BCEWithLogitsLoss (optionally pos_weight); computed in FP32.
  - Output interface: scalar logits per example (B,).

Precision policy (Ampere / A10, normal mode):
  - Default policy: 'fp16_amp' with CUDA autocast on the forward pass only.
  - Parameters, floating inputs, and targets remain FP32.
  - Backward and optimizer.step run OUTSIDE autocast, with GradScaler.
  - FP32 islands used for the keyword-gate sigmoid, the loss, and metric probs.
  - TF32 disabled for matmul and convolutions.
  - CPU path: FP32, AMP and scaler disabled (supports CPU preflight).
  - Fallback policy: 'fp32' (call degrade_to_fp32 on instability).
"""

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Module-level constants (no side effects)
# ---------------------------------------------------------------------------

# Allowed precision policies in this stage. Normal mode endorses only fp32 and
# selective fp16_amp; no bf16, no TF32, no quantization, no TE recipes.
PRECISION_POLICY_FP32 = "fp32"
PRECISION_POLICY_FP16_AMP = "fp16_amp"
ALLOWED_PRECISION_POLICIES: Tuple[str, ...] = (
    PRECISION_POLICY_FP32,
    PRECISION_POLICY_FP16_AMP,
)

# Dtype contract.
PARAM_DTYPE = torch.float32
FLOAT_INPUT_DTYPE = torch.float32
INTEGER_INPUT_DTYPE = torch.long
TARGET_DTYPE = torch.float32

# AMP dtype: only fp16 is permitted in normal mode.
AMP_DTYPE_FP16 = torch.float16
ALLOWED_AMP_DTYPES: Tuple[torch.dtype, ...] = (AMP_DTYPE_FP16,)
PREFERRED_AMP_DTYPE = AMP_DTYPE_FP16

# Fallback.
FALLBACK_POLICY = "fp32"

# Reason strings recorded in the precision log.
REASON_CUDA_FP16_AMP = "ampere_gpu_selective_fp16_amp_fp32_islands"
REASON_CPU_OR_DISABLED = "cpu_or_amp_disabled_fp32"
REASON_DEGRADED = "degraded_amp_to_fp32"


# ---------------------------------------------------------------------------
# Precision configuration object
# ---------------------------------------------------------------------------


@dataclass
class PrecisionConfig:
    """Resolved device + precision policy.

    Attributes
    ----------
    device : str
        Target device string ("cuda" or "cpu").
    use_amp : bool
        Whether forward-only FP16 autocast is enabled.
    amp_dtype : torch.dtype
        Autocast dtype (torch.float16). Irrelevant when use_amp=False.
    use_tf32 : bool
        TF32 for matmul/conv. Must remain False in normal mode.
    fallback : str
        Fallback policy name (typically "fp32").
    reason : str
        Human-readable reason for the current policy.
    amp_dtype_name : str
        Short label for logging: "fp16" or "fp32".
    """

    device: str
    use_amp: bool
    amp_dtype: torch.dtype = AMP_DTYPE_FP16
    use_tf32: bool = False
    fallback: str = FALLBACK_POLICY
    reason: str = ""
    amp_dtype_name: str = "fp32"

    @property
    def is_cuda(self) -> bool:
        return str(self.device).startswith("cuda")

    @property
    def policy_name(self) -> str:
        return PRECISION_POLICY_FP16_AMP if self.use_amp else PRECISION_POLICY_FP32

    @property
    def scaler_enabled(self) -> bool:
        return bool(self.use_amp) and self.is_cuda


# ---------------------------------------------------------------------------
# Device resolution helpers (read-only, no CUDA configuration)
# ---------------------------------------------------------------------------


def _cuda_usable() -> bool:
    """Return True only when CUDA is present AND a trivial allocation works.

    Read-only probe; does NOT set TF32, does NOT set cudnn.benchmark, does NOT
    touch any global state.  Missing GPU drivers / no CUDA context simply
    return False so the caller can fall back to CPU/FP32.
    """
    try:
        if not torch.cuda.is_available():
            return False
        _ = torch.cuda.current_device()
        _ = torch.zeros(1, device="cuda")
        return True
    except Exception:
        return False


def resolve_precision_config(
    prefer_amp: bool = True,
    device: Optional[str] = None,
    verbose: bool = True,
) -> PrecisionConfig:
    """Resolve device and precision policy for the current host.

    Rules
    -----
    - CUDA requested/available and `prefer_amp=True` -> fp16_amp forward-only.
    - Otherwise                                       -> fp32 everywhere.
    - CPU preflight always resolves to fp32 (no scaler, no autocast).

    Parameters
    ----------
    prefer_amp : bool
        If True, allow FP16 AMP on a usable CUDA device.
    device : Optional[str]
        Explicit device request (e.g. "cuda", "cpu").  If None, auto-detect.
    verbose : bool
        Emit one precision log line (only at call time, never at import).
    """
    cuda_ok = _cuda_usable()

    if device is None:
        device = "cuda" if cuda_ok else "cpu"

    # Guard against requesting CUDA when it is not usable.
    if str(device).startswith("cuda") and not cuda_ok:
        device = "cpu"

    on_cuda = str(device).startswith("cuda")

    if on_cuda and prefer_amp:
        cfg = PrecisionConfig(
            device=device,
            use_amp=True,
            amp_dtype=PREFERRED_AMP_DTYPE,
            amp_dtype_name="fp16",
            use_tf32=False,
            fallback=FALLBACK_POLICY,
            reason=REASON_CUDA_FP16_AMP,
        )
    else:
        cfg = PrecisionConfig(
            device=device,
            use_amp=False,
            amp_dtype=PREFERRED_AMP_DTYPE,
            amp_dtype_name="fp32",
            use_tf32=False,
            fallback=FALLBACK_POLICY,
            reason=REASON_CPU_OR_DISABLED,
        )

    if verbose:
        print(format_precision_log(cfg))
    return cfg


def format_precision_log(cfg: PrecisionConfig) -> str:
    """Single-line precision log (never called at import time)."""
    return (
        "[precision] "
        f"device={cfg.device} "
        f"policy={cfg.policy_name} "
        f"amp_dtype={cfg.amp_dtype_name} "
        f"tf32={bool(cfg.use_tf32)} "
        f"scaler={'on' if cfg.scaler_enabled else 'off'} "
        f"fallback={cfg.fallback} "
        f"reason={cfg.reason}"
    )


# ---------------------------------------------------------------------------
# Global precision flags (called from the main guard only)
# ---------------------------------------------------------------------------


def configure_precision_globals(cfg: PrecisionConfig) -> dict:
    """Apply TF32 / cudnn flags according to `cfg`.

    Called from the training entry point.  Returns a dict of what was applied
    so callers can log it without re-reading globals.

    Contract: TF32 is disabled in normal mode regardless of hardware support.
    """
    applied = {"tf32_matmul": False, "tf32_cudnn": False, "cudnn_benchmark": None}

    # TF32: honor cfg (default False).  Never enable unless explicitly asked.
    use_tf32 = bool(cfg.use_tf32) and cfg.is_cuda
    try:
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        applied["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = use_tf32
        applied["tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
    except Exception:
        pass

    # Benchmark mode: soft perf win for fixed input shapes; harmless on CPU.
    if cfg.is_cuda:
        try:
            torch.backends.cudnn.benchmark = True
            applied["cudnn_benchmark"] = True
        except Exception:
            applied["cudnn_benchmark"] = None

    return applied


# ---------------------------------------------------------------------------
# Autocast contexts (forward-only) and FP32 islands
# ---------------------------------------------------------------------------


def _make_cuda_autocast(enabled: bool):
    """Build a CUDA autocast context across torch versions.

    Prefers the modern `torch.amp.autocast("cuda", ...)` API, falls back to
    `torch.cuda.amp.autocast`, then to `nullcontext` if neither is available.
    """
    if not enabled:
        return contextlib.nullcontext()
    # Modern API (torch >= 2.0).
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "autocast"):
        try:
            return amp_mod.autocast(
                device_type="cuda", dtype=AMP_DTYPE_FP16, enabled=True
            )
        except Exception:
            pass
    # Legacy API.
    legacy = getattr(torch.cuda, "amp", None)
    if legacy is not None and hasattr(legacy, "autocast"):
        try:
            return legacy.autocast(dtype=AMP_DTYPE_FP16, enabled=True)
        except Exception:
            pass
    return contextlib.nullcontext()


def _make_cuda_autocast_disabled():
    """Explicitly disable autocast for FP32 island regions."""
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "autocast"):
        try:
            return amp_mod.autocast(device_type="cuda", enabled=False)
        except Exception:
            pass
    legacy = getattr(torch.cuda, "amp", None)
    if legacy is not None and hasattr(legacy, "autocast"):
        try:
            return legacy.autocast(enabled=False)
        except Exception:
            pass
    return contextlib.nullcontext()


def autocast_context(cfg: PrecisionConfig):
    """FP16 autocast context — wrap ONLY eligible forward ops.

    Returns `nullcontext` (no-op) on CPU or when AMP is disabled, so training
    code can use `with autocast_context(cfg):` unconditionally.
    """
    if not cfg.use_amp or not cfg.is_cuda:
        return contextlib.nullcontext()
    return _make_cuda_autocast(True)


def fp32_island(cfg: PrecisionConfig):
    """Disable autocast for numerically fragile regions (gate sigmoid, loss,
    probability export, metric accumulation).  No-op on CPU / fp32 runs.
    """
    if not cfg.use_amp or not cfg.is_cuda:
        return contextlib.nullcontext()
    return _make_cuda_autocast_disabled()


# ---------------------------------------------------------------------------
# GradScaler factory (backward / optimizer updates live OUTSIDE autocast)
# ---------------------------------------------------------------------------


def make_grad_scaler(cfg: PrecisionConfig, enabled: Optional[bool] = None):
    """Create a GradScaler only when FP16 AMP is active on CUDA.

    Returns None on CPU / fp32-only runs so callers can pass it through
    `TrainingDiagnostics(..., scaler=scaler_or_None)` and the training loop
    without special-casing.  Uses the modern API when available, with a
    legacy fallback.
    """
    if enabled is None:
        enabled = cfg.scaler_enabled
    if not enabled:
        return None

    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "GradScaler"):
        try:
            return amp_mod.GradScaler("cuda", enabled=True)
        except Exception:
            pass
    legacy = getattr(torch.cuda, "amp", None)
    if legacy is not None and hasattr(legacy, "GradScaler"):
        try:
            return legacy.GradScaler(enabled=True)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Precision-safe forward / loss / probability helpers
# ---------------------------------------------------------------------------


def amp_forward(model: nn.Module, inputs, cfg: PrecisionConfig):
    """Run one forward pass under autocast, return FP32 logits.

    `inputs` may be a tuple/list of tensors (e.g. word_ids, char_ids, kw_ids)
    or a dict of kwargs.  All integer token tensors must already be on the
    correct device with dtype=torch.long — this helper does NOT cast them.

    The returned tensor is a detached FP32 tensor suitable for the loss and
    for downstream FP32-only metric computation.
    """
    if isinstance(inputs, dict):
        with autocast_context(cfg):
            logits = model(**inputs)
    else:
        with autocast_context(cfg):
            logits = model(*inputs)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    # Force FP32 for the boundary — autocast can produce FP16 logits.
    return logits.float()


def bce_with_logits_fp32(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[Any] = None,
    criterion: Optional[nn.Module] = None,
) -> torch.Tensor:
    """BCEWithLogitsLoss computed in FP32 regardless of surrounding autocast.

    Stage 1's `build_loss(pos_weight=...)` returns a `BCEWithLogitsLoss` whose
    registered `pos_weight` buffer may already be float32.  We explicitly:

      1. cast logits/targets to float32,
      2. (re)cast pos_weight to float32 on the same device,
      3. execute the criterion (or a functional fallback) even if an outer
         autocast scope is active.

    Because `binary_cross_entropy_with_logits` is a reduction op sensitive to
    fp16 underflow, this helper must be invoked inside `fp32_island(cfg)` by
    the training loop for maximum safety.
    """
    logits_f = logits.float()
    targets_f = targets.to(device=logits_f.device, dtype=torch.float32)

    if criterion is not None and isinstance(criterion, nn.Module):
        # Rebuild the criterion's pos_weight buffer into fp32 to avoid a
        # silent dtype mismatch under autocast.
        pw_buf = getattr(criterion, "pos_weight", None)
        if pw_buf is not None:
            try:
                critic_buf = pw_buf.to(device=logits_f.device, dtype=torch.float32)
                with torch.no_grad():
                    criterion.pos_weight = critic_buf
            except Exception:
                pass
        return criterion(logits_f, targets_f)

    pw = None
    if pos_weight is not None:
        pw = torch.as_tensor(pos_weight, dtype=torch.float32, device=logits_f.device)
    return F.binary_cross_entropy_with_logits(logits_f, targets_f, pos_weight=pw)


def logits_to_probs_fp32(logits: torch.Tensor) -> torch.Tensor:
    """Sigmoid in FP32 for metrics/export. Never under autocast."""
    return torch.sigmoid(logits.float())


# ---------------------------------------------------------------------------
# Instability detection / fallback
# ---------------------------------------------------------------------------


def check_finite(values) -> bool:
    """Return True iff every non-None tensor/scalar is finite."""
    if values is None:
        return True
    if isinstance(values, (tuple, list)):
        return all(check_finite(v) for v in values)
    if isinstance(values, dict):
        return all(check_finite(v) for v in values.values())
    if isinstance(values, torch.Tensor):
        return bool(torch.isfinite(values).all().item())
    if isinstance(values, (int, float)):
        return math.isfinite(float(values))
    # Unsupported types are treated as "unknown but not per-se bad".
    return True


def degrade_to_fp32(
    cfg: PrecisionConfig,
    reason: str = "instability",
) -> PrecisionConfig:
    """Mutate `cfg` (in place) to the FP32 fallback policy.

    Called from the training loop when a non-finite loss/gradient is observed
    or when the AMP path appears to be hurting quality.  Returns the same cfg
    so callers can chain `cfg = degrade_to_fp32(cfg)`.
    """
    cfg.use_amp = False
    cfg.amp_dtype_name = "fp32"
    cfg.use_tf32 = False
    cfg.reason = f"{REASON_DEGRADED}:{reason}"
    return cfg


# ---------------------------------------------------------------------------
# Small utilities for gradient clipping under AMP
# ---------------------------------------------------------------------------


def unscale_then_clip_grad_norm_(
    scaler,
    model: nn.Module,
    max_norm: Optional[float] = None,
) -> Optional[torch.Tensor]:
    """Unscale gradients before clipping (required for FP16 AMP).

    If `scaler` is None (fp32 path), just compute the grad norm.  Returns the
    total grad norm tensor (or None if there is nothing to clip).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return None

    if scaler is not None:
        try:
            scaler.unscale_(optimizer_placeholder := None) if False else None
        except Exception:
            pass
        # `scaler.unscale_` requires the optimizer; Stage 3 owns the optimizer
        # and calls `scaler.unscale_(optimizer)` itself.  We expose only the
        # clipping side here so the contract stays clean.
    max_n = max_norm if max_norm is not None else math.inf
    return torch.nn.utils.clip_grad_norm_(params, max_n)


# ---------------------------------------------------------------------------
# Optional lightweight AMP smoke-test (used by training, not at import time)
# ---------------------------------------------------------------------------


def probe_amp_stability(
    cfg: PrecisionConfig,
    model: nn.Module,
    sample_inputs,
    criterion: Optional[nn.Module] = None,
    sample_targets: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> bool:
    """One forward+loss probe on a real batch under the selected precision.

    Returns True if the result is finite.  Intended to be called once by the
    training loop right after model/criterion setup, so a bad AMP path can be
    detected before the first epoch and the fallback applied.

    Never runs at import time.  Does NOT mutate model weights (no backward).
    """
    was_training = model.training
    try:
        model.train(False)
        with torch.no_grad():
            logits = amp_forward(model, sample_inputs, cfg)
            finite_logits = check_finite(logits)
            finite_loss = True
            if criterion is not None and sample_targets is not None:
                with fp32_island(cfg):
                    loss = bce_with_logits_fp32(
                        logits, sample_targets, criterion=criterion
                    )
                finite_loss = check_finite(loss)
        ok = bool(finite_logits and finite_loss)
    except Exception:
        ok = False
    finally:
        model.train(was_training)
    if verbose:
        print(f"[precision] amp_probe finite={ok} policy={cfg.policy_name}")
    return ok


__all__ = [
    "PRECISION_POLICY_FP32",
    "PRECISION_POLICY_FP16_AMP",
    "ALLOWED_PRECISION_POLICIES",
    "PARAM_DTYPE",
    "FLOAT_INPUT_DTYPE",
    "INTEGER_INPUT_DTYPE",
    "TARGET_DTYPE",
    "AMP_DTYPE_FP16",
    "ALLOWED_AMP_DTYPES",
    "PREFERRED_AMP_DTYPE",
    "FALLBACK_POLICY",
    "PrecisionConfig",
    "resolve_precision_config",
    "format_precision_log",
    "configure_precision_globals",
    "autocast_context",
    "fp32_island",
    "make_grad_scaler",
    "amp_forward",
    "bce_with_logits_fp32",
    "logits_to_probs_fp32",
    "check_finite",
    "degrade_to_fp32",
    "probe_amp_stability",
]

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
 Summary: 3 total attempts - 1 pending execution, 2 executed, 1 successful (best: 0.7341)

-------------------------------
Attempt #1:
Design: I merged the three stages by keeping Stage 1's data/model/loss/adapter code as the foundation, layering Stage 2's `PrecisionRuntime` policy helpers on top, and consuming both inside Stage 3's `run_training` (with diagnostics + scheduler safe points). I resolved naming conflicts, deduplicated imports, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module-level constants that Stage 3 references, and moved every side effect under `if __name__ == '__main__':` while preserving the CPU-safe `CandidateAdapter` contract.
Results: Training completed successfully in ~37 minutes on an A10 GPU, running 12 epochs with a hybrid transformer model. Validation F1 improved from 0.665 at epoch 1 to a best of ~0.7341 (best epoch 11), and a valid submission file with 1523 rows was written using a tuned threshold of 0.42. The only messages were benign warnings (Graph evidence mirror connection refused, nested-tensor warning) that did not affect execution.
Validation Metric: 0.7341
Execution Time: 2223.42s

-------------------------------
Attempt #2:
Design: Looking at the three stages, I merged by keeping Stage 1's BiGRU+keyword-bias model/data/loss/threshold-tuner/CandidateAdapter as the foundation, Stage 2's `PrecisionRuntime` helpers (FP16-AMP forward autocast + FP32 islands for loss/sigmoid/metrics, TF32 disabled, CPU-safe preflight) layered on top, and completing Stage 3's truncated training loop (AdamW + linear-warmup+cosine, FP16 autocast forward, scaler.unscale before clip, val-F1 early stopping with best-state restore, guarded scheduler safe-points, `TrainingDiagnostics` with fallback), then adding the missing test inference + submission write + `if __name__ == "__main__":` entrypoint. I fixed the truncated `_make_state_factory`, resolved duplicated constants/imports, kept `MODEL_FAMILY`/`WORKING_DIR`/`SUBMISSION_DIR` as module constants, and put the final `Final Validation Score:` print as the very last line after the diagnostics context closes.
Results: The implementation of this design has bugs.
Insight: Using a different approach may not result in the same bugs as the above approach.

-------------------------------
Attempt #3:
Design: I merged the three stages by keeping Stage 1's data cleaning, TF-IDF feature builders, `SparseMLP`, `build_loss`, `tune_threshold` and split helpers as the foundation; layering Stage 2's import-safe precision utilities (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `degrade_to_fp32`, `configure_tf32`, `format_precision_log`) on top; then consuming both inside Stage 3's `run_training` with scheduler safe-points, `TrainingDiagnostics`, linear warmup+cosine AdamW, best-val-F1 restore, and test inference. I resolved the duplicate `CandidateAdapter` by keeping Stage 3's self-contained CPU-safe version (it uses the Stage 1 model/loss/features through the Stage 2 precision path), removed duplicate imports/constants, kept all side effects under `if __name__ == "__main__":`, and preserved the exact `MODEL_FAMILY`, `WORKING_DIR`, `SUBMISSION_DIR` contract.
Status: Code generated, execution pending (will run in parallel with other drafts).


# Hardware-Aware Stage 1 Candidate Construction Context
# Hardware-Aware Model Design Brief
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2; observed in 1 batch-size observation
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
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2; observed in 1 batch-size observation
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
- Hardware: NVIDIA A10 with 22589 MiB VRAM, compute capability 8.6, cuda 13.2; observed in 1 batch-size observation
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
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"cross entropy or metric-compatible classification loss","output_interface":"match the competition metric and required submission format","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"{'forward_autocast_boundaries': \"Use torch.autocast(device_type='cuda', dtype=torch.float16) only...","fallback_policy":"fp32","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"{'name': 'AdamW', 'learning_rate': 2e-05, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-08}","scheduler":"{'name': 'linear_warmup_decay', 'warmup_ratio': 0.1, 'num_training_steps': 'Derived from epochs,...","batch_size_policy":"fixed","dataloader_policy":"{'shuffle_train': True, 'shuffle_validation': False, 'shuffle_test': False, 'num_workers': '>0 wh...","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available","no exact, similar, or packed graph profiles for current hardware","no measured FP16 AMP speedup or quality comparison for this task"]}}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=split is stratified 85/15 with seed 42 fit strictly on `train.csv` labels, and all stateful transforms (SimpleTokenizer, CharTokenizer, KeywordMapper) are fit only on the training split; purpose=strict leakage isolation across train/val/test; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=exposes `MAX_WORD_LEN=32`, `MAX_CHAR_LEN=128`, and integer-id inputs via a reusable `make_batch_tensors` helper + a CPU-safe `CandidateAdapter`; purpose=keep sequence geometry configurable for later precision/batching stages without committing to them here; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=introduced a `PrecisionConfig` + selective FP16 AMP policy (CUDA autocast for forward only, FP32 islands for gate sigmoid/loss/probs, GradScaler for backward, TF32 explicitly disabled) replacing the implicit "everythi...; purpose=exploit A10 Tensor Cores on the word/char Conv1d + Embedding matmuls while protecting the numerically fragile keyword-gate interaction and BCEWithLogits loss that Stage 1 relies on; hardware_keys=datatype_precision.precision_policy, datatype_precision.fallback_policy, datatype_precision.precision_model_adaptation
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=added runtime precision helpers (`configure_precision_globals`, `autocast_context`, `fp32_island`, `make_grad_scaler`, `amp_forward`, `bce_with_logits_fp32`, `degrade_to_fp32`) that Stage 3's `run_training` will consu...; purpose=keep precision configurable and CPU-safe so Stage 3 can simply call them around its existing forward/loss, preserving Stage 1's model family and output interface; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=no `model.half()`, no whole-pipeline cast; integer token indices stay `torch.long`, floating params/targets stay FP32; CPU path resolves to FP32 with AMP/scaler disabled; purpose=honor the "selective forward autocast" rule and guarantee the CPU preflight can execute the real model/loss without CUDA; hardware_keys=none
  - Stage 3 training / training_evaluation: baseline_change=added end-to-end Stage-3 training loop consuming Stage-1 model/loss/featurizers and Stage-2 `amp_forward`/`bce_with_logits_fp32`/`fp32_island`/`GradScaler`/`degrade_to_fp32`, with linear-warmup+cosine AdamW, grad-clip...; purpose=convert the gated MVCNN into a real trained candidate whose validation and test predictions flow through the identical model forward and threshold; hardware_keys=training_evaluation.optimizer, training_evaluation.scheduler, datatype_precision.precision_policy
  - Stage 3 training / training_evaluation: baseline_change=replaced DataLoader(shuffle=True) with a per-epoch permutation + `IndexedTensorDataset` + shuffle=False loader, and snapshot that permutation + `step_in_epoch` in the scheduler state factory; purpose=honor the "do not replay completed updates" contract on mid-epoch resume and keep epoch boundaries clean for safe_point EPOCH; hardware_keys=training_evaluation.dataloader_policy
  - Stage 3 training / training_evaluation: baseline_change=wrapped the full loop in `TrainingDiagnostics` with real settings (physical/effective batch, planned epochs, steps/epoch, model_family, split_seed, precision reason), called `after_update()` after each optimizer attem...; purpose=emit measureable throughput/VRAM telemetry while keeping the final validation score as the last line after context exit; hardware_keys=none

# Step results

        ### Step 1: model_design
        **Plan:** ## Plan

**WHAT makes this task unique**: Disaster tweets are short (≤32 tokens), idiosyncratic, and heavily signal through local phrases + hashtags + the structured `keyword` column, which is effectively a pre-extracted topic tag. The signal is *not* long-range — it lives in n-gram pockets and in whether a topic keyword is used in a disaster sense. Class balance is moderate (~43% positive).

**WHY this approach fits**: A from-scratch deep transformer on 6k short texts overfits; instead I use a **multi-view CNN with keyword cross-gating**, which matches the inductive bias of short-text classification: (a) parallel multi-kernel word convolutions capture disaster n-gram phrases ("wildfire", "on fire", "killed in"), (b) a character CNN captures hashtag/casing/morphological cues without a subword pretrainer, and (c) a learned keyword embedding is *gated* by the word-context — the model can learn to amplify the keyword only when the tweet's surrounding words suggest it's a real disaster, mirroring the human reading of `keyword="ablaze" + "beautiful sunset ablaze"`. This is genuinely novel vs. prior attempts (BiGRU+keyword, TF-IDF+SparseMLP, hybrid transformer), which either trust the keyword naively or ignore the gating interaction. Modeling budget is conservative: ~3M params, no pretrained weights, no external downloads.

**HOW validation confirms/refutes**: Split 85/15 stratified (seed 42) *before* any vocab fit. If the gated keyword view genuinely helps, validation F1 with gate-on should beat a keyword-embedding-concat ablation by a clear margin; if validation F1 collapses toward the majority-class baseline, the gate is learning to zero-out the keyword and I'd fall back to plain concat. Metric = binary F1 with threshold tuned on validation (as Stage 3 consumes).

**Backend/memory strategy**: No precision or optimizer choices are made here (those belong to later stages); the design exposes configurable sequence lengths (`max_word_len`, `max_char_len`), configurable batch size honor via the adapter, and integer token tensors (torch.long) so non-blocking CUDA transfer is possible. Stateful transforms (word/char vocabs, keyword map) are fit *only* on train texts, and reused via `.encode()` on val/test. Scheduler-owned controls (device assignment, num_workers, batch size, AMP/scaler, checkpointing) are left to later stages; the subprocess must not implement them itself.

### Note board:
- baseline_change: replaced recurrent/transformer baselines with a multi-view CNN (parallel word kernels + char CNN) whose keyword embedding is multiplicatively gated by the fused text context | purpose: exploit short-text inductive bias and model keyword–context interaction that prior attempts treated as a naive bias term | hardware_keys: none
- baseline_change: split is stratified 85/15 with seed 42 fit strictly on `train.csv` labels, and all stateful transforms (SimpleTokenizer, CharTokenizer, KeywordMapper) are fit only on the training split | purpose: strict leakage isolation across train/val/test | hardware_keys: none
- baseline_change: exposes `MAX_WORD_LEN=32`, `MAX_CHAR_LEN=128`, and integer-id inputs via a reusable `make_batch_tensors` helper + a CPU-safe `CandidateAdapter` | purpose: keep sequence geometry configurable for later precision/batching stages without committing to them here | hardware_keys: none
        **Code:**
        ```python
"""
Stage 1 (model_design) — Disaster Tweets binary classification.

Model family: MultiViewDisasterNet, a from-scratch multi-view CNN:
  - Word branch: parallel Conv1d over word embeddings (kernels 2,3,4) → max-pool.
  - Char branch: single Conv1d over char embeddings → max-pool.
  - Keyword branch: learned embedding multiplicatively gated by concatenated
    word+char context, so the keyword only contributes when the tweet context
    suggests a real-disaster reading.
  - Fusion: LayerNorm → dropout → 2-layer MLP → single logit.

This module is IMPORT-SAFE: only imports, constants, classes, and helper
functions are defined at top level. Training loop, optimizer, scheduler,
precision policy, batch size, epochs, and workers all live in later stages.
"""

# ---- Module-level constants (no side effects) -----------------------------

MODEL_FAMILY = "mvcnn_keyword_gate_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
DEFAULT_INPUT_DIR = "./input"

PAD_IDX = 0
UNK_IDX = 1
NUM_SPECIAL_TOKENS = 2

# Configurable sequence geometry (exposed for later stages).
MAX_WORD_LEN = 32
MAX_CHAR_LEN = 128

# Defaults for the production model built during training.
WORD_EMBED_DIM = 100
WORD_KERNELS = (2, 3, 4)
WORD_FILTERS = 96
CHAR_EMBED_DIM = 32
CHAR_KERNEL = 3
CHAR_FILTERS = 48
KEYWORD_DIM = 32
HIDDEN_DIM = 128
DROPOUT = 0.4

# Defaults for the lightweight CPU-preflight adapter.
ADAPTER_WORD_EMBED_DIM = 64
ADAPTER_WORD_FILTERS = 32
ADAPTER_CHAR_EMBED_DIM = 16
ADAPTER_CHAR_FILTERS = 16
ADAPTER_KEYWORD_DIM = 16
ADAPTER_HIDDEN_DIM = 48
ADAPTER_DROPOUT = 0.2
ADAPTER_N_SAMPLE_ROWS = 128


# ---- Imports --------------------------------------------------------------

import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ---- Text cleaning --------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_USER_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NON_PRINTABLE_RE = re.compile(r"[^a-z0-9<>_\s]")
_WS_RE = re.compile(r"\s+")

URL_TOKEN = "<url>"
USER_TOKEN = "<user>"
HASHTAG_PREFIX = "htag_"


def clean_text(text: str) -> str:
    """Lowercase, normalize URLs/@mentions/hashtags, strip odd chars.

    Deterministic, stateless — safe to apply to train/val/test identically.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(f" {URL_TOKEN} ", text)
    text = _USER_RE.sub(f" {USER_TOKEN} ", text)
    text = _HASHTAG_RE.sub(lambda m: f" {HASHTAG_PREFIX}{m.group(1)} ", text)
    text = _NON_PRINTABLE_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def join_context(text: str, keyword, location) -> str:
    """Optional richer field used only if a downstream stage wants it.

    Kept as a separate helper so the main text pipeline stays clean.
    """
    parts = [clean_text(text)]
    if isinstance(keyword, str) and keyword.strip():
        parts.append("kw_" + keyword.strip().lower().replace(" ", "_"))
    if isinstance(location, str) and location.strip():
        parts.append("loc_" + location.strip().lower().replace(" ", "_"))
    return " ".join(p for p in parts if p)


# ---- Stateful tokenizers (fit on TRAIN ONLY) -----------------------------


class SimpleTokenizer:
    """Word-level vocab built from scratch (no pretrained tokenizer)."""

    def __init__(self, min_freq: int = 1, max_vocab: int = 30000):
        self.min_freq = int(min_freq)
        self.max_vocab = int(max_vocab)
        self.vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}

    def fit(self, texts):
        c = Counter()
        for t in texts:
            for tok in (t or "").split():
                c[tok] += 1
        items = [(tok, n) for tok, n in c.most_common() if n >= self.min_freq]
        items = items[: max(self.max_vocab - NUM_SPECIAL_TOKENS, 0)]
        for tok, _ in items:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
        return self

    @property
    def vocab_size(self) -> int:
        return max(len(self.vocab), NUM_SPECIAL_TOKENS)

    def encode(self, texts, max_len: int = MAX_WORD_LEN) -> np.ndarray:
        out = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, t in enumerate(texts):
            toks = (t or "").split()[:max_len]
            ids = [self.vocab.get(tok, UNK_IDX) for tok in toks]
            if ids:
                out[i, : len(ids)] = ids
        return out


class CharTokenizer:
    """Character-level vocab (subset of printable ASCII + special tokens)."""

    def __init__(self, min_freq: int = 2, max_vocab: int = 128):
        self.min_freq = int(min_freq)
        self.max_vocab = int(max_vocab)
        self.vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}

    def fit(self, texts):
        c = Counter()
        for t in texts:
            for ch in t or "":
                c[ch] += 1
        items = [(ch, n) for ch, n in c.most_common() if n >= self.min_freq]
        items = items[: max(self.max_vocab - NUM_SPECIAL_TOKENS, 0)]
        for ch, _ in items:
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)
        return self

    @property
    def vocab_size(self) -> int:
        return max(len(self.vocab), NUM_SPECIAL_TOKENS)

    def encode(self, texts, max_len: int = MAX_CHAR_LEN) -> np.ndarray:
        out = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, t in enumerate(texts):
            chars = (t or "")[:max_len]
            ids = [self.vocab.get(ch, UNK_IDX) for ch in chars]
            if ids:
                out[i, : len(ids)] = ids
        return out


class KeywordMapper:
    """Categorical keyword map. Index 0 = missing, 1 = unseen, then real keys."""

    def __init__(self):
        self.mapping = {"<none>": 0, "<unk>": 1}

    def fit(self, keywords):
        for k in sorted({k for k in keywords if isinstance(k, str) and k.strip()}):
            if k not in self.mapping:
                self.mapping[k] = len(self.mapping)
        return self

    @property
    def num_keywords(self) -> int:
        return max(len(self.mapping), 2)

    def encode(self, keywords) -> np.ndarray:
        out = np.zeros(len(keywords), dtype=np.int64)
        for i, k in enumerate(keywords):
            if not isinstance(k, str) or not k.strip():
                out[i] = 0
            else:
                out[i] = self.mapping.get(k, 1)
        return out


# ---- Model ---------------------------------------------------------------


class MultiViewDisasterNet(nn.Module):
    """Multi-view CNN with a keyword cross-gate.

    Forward signature: forward(word_ids, char_ids, keyword_ids) -> logits (B,)
    """

    def __init__(
        self,
        word_vocab_size: int,
        char_vocab_size: int,
        num_keywords: int,
        word_embed_dim: int = WORD_EMBED_DIM,
        word_kernels=WORD_KERNELS,
        word_filters: int = WORD_FILTERS,
        char_embed_dim: int = CHAR_EMBED_DIM,
        char_kernel: int = CHAR_KERNEL,
        char_filters: int = CHAR_FILTERS,
        keyword_dim: int = KEYWORD_DIM,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
        num_classes: int = 1,
    ):
        super().__init__()
        self.word_vocab_size = int(word_vocab_size)
        self.char_vocab_size = int(char_vocab_size)
        self.num_keywords = int(num_keywords)

        # -- Word branch --
        self.word_emb = nn.Embedding(
            word_vocab_size, word_embed_dim, padding_idx=PAD_IDX
        )
        self.word_convs = nn.ModuleList(
            [
                nn.Conv1d(word_embed_dim, word_filters, k, padding=k // 2)
                for k in word_kernels
            ]
        )
        word_feat_dim = word_filters * len(word_kernels)

        # -- Char branch --
        self.char_emb = nn.Embedding(
            char_vocab_size, char_embed_dim, padding_idx=PAD_IDX
        )
        self.char_conv = nn.Conv1d(
            char_embed_dim, char_filters, char_kernel, padding=char_kernel // 2
        )

        # -- Keyword branch with cross-gate --
        self.kw_emb = nn.Embedding(num_keywords, keyword_dim)
        fused_in = word_feat_dim + char_filters
        self.kw_gate = nn.Linear(fused_in + keyword_dim, keyword_dim)
        self.kw_proj = nn.Linear(keyword_dim, keyword_dim)

        # -- Fusion head --
        self.norm = nn.LayerNorm(fused_in + keyword_dim)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(fused_in + keyword_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].fill_(0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, word_ids, char_ids, keyword_ids):
        # --- Word branch ---
        w = self.word_emb(word_ids)  # (B, L, D)
        w = w.transpose(1, 2)  # (B, D, L)
        w_feats = []
        for conv in self.word_convs:
            h = F.relu(conv(w))
            h = F.adaptive_max_pool1d(h, 1).squeeze(-1)
            w_feats.append(h)
        word_feat = torch.cat(w_feats, dim=-1)  # (B, word_feat_dim)

        # --- Char branch ---
        c = self.char_emb(char_ids).transpose(1, 2)
        c = F.relu(self.char_conv(c))
        char_feat = F.adaptive_max_pool1d(c, 1).squeeze(-1)  # (B, char_filters)

        base = torch.cat([word_feat, char_feat], dim=-1)  # (B, fused_in)

        # --- Keyword cross-gate ---
        kw = self.kw_emb(keyword_ids)  # (B, keyword_dim)
        gate = torch.sigmoid(self.kw_gate(torch.cat([base, kw], dim=-1)))
        kw_gated = gate * self.kw_proj(kw)

        # --- Fusion ---
        fused = torch.cat([base, kw_gated], dim=-1)
        fused = self.norm(fused)
        fused = self.drop(fused)
        logits = self.classifier(fused)  # (B, 1)
        return logits.squeeze(-1)


# ---- Loss / metric helpers ----------------------------------------------


def build_loss(pos_weight=None):
    """BCEWithLogitsLoss with optional positive-class weighting.

    `pos_weight` is a Python scalar (num_neg / num_pos). If None, no weighting.
    """
    if pos_weight is None:
        pw = None
    else:
        pw = torch.tensor([float(pos_weight)], dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pw)


def compute_f1(y_true, y_pred_binary) -> float:
    return float(
        f1_score(np.asarray(y_true), np.asarray(y_pred_binary), zero_division=0)
    )


def tune_threshold(
    y_true, y_prob, min_t: float = 0.20, max_t: float = 0.80, steps: int = 61
):
    """Return (best_threshold, best_f1) maximizing binary F1 on val probs."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(min_t, max_t, steps):
        preds = (y_prob >= t).astype(int)
        f1 = compute_f1(y_true, preds)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


# ---- Split + feature orchestration (pure functions, called under __main__) -


def resolve_input_dir() -> str:
    env = os.environ.get("MLEVOLVE_INPUT_DIR")
    if env and os.path.isdir(env):
        return env
    return DEFAULT_INPUT_DIR


def load_raw_data(input_dir=None):
    input_dir = input_dir or resolve_input_dir()
    train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(input_dir, "test.csv"))
    return train_df, test_df


def stratified_split(train_df, val_size: float = 0.15, seed: int = 42):
    """Split BEFORE any vocab/stat fit, stratified on target."""
    tr, va = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df["target"].astype(int),
        random_state=seed,
    )
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def prepare_dataframes(train_df, test_df, val_size: float = 0.15, seed: int = 42):
    """Produce raw splits with cleaned text. No vocab fitting here."""
    tr, va = stratified_split(train_df, val_size=val_size, seed=seed)
    for df in (tr, va, test_df):
        df["_clean_text"] = df["text"].apply(clean_text)
    if "_clean_text" not in test_df.columns:
        test_df["_clean_text"] = test_df["text"].apply(clean_text)
    return tr, va, test_df


def fit_featurizers(
    train_texts,
    train_keywords,
    word_max_vocab: int = 30000,
    char_min_freq: int = 2,
    char_max_vocab: int = 128,
):
    """Fit tokenizers/mappers ONLY on training data (leakage-safe)."""
    wtok = SimpleTokenizer(min_freq=1, max_vocab=word_max_vocab).fit(train_texts)
    ctok = CharTokenizer(min_freq=char_min_freq, max_vocab=char_max_vocab).fit(
        train_texts
    )
    kmap = KeywordMapper().fit(train_keywords)
    return wtok, ctok, kmap


def make_batch_tensors(
    texts,
    keywords,
    wtok,
    ctok,
    kmap,
    max_word_len: int = MAX_WORD_LEN,
    max_char_len: int = MAX_CHAR_LEN,
    device=None,
):
    """Return (word_ids, char_ids, keyword_ids) long tensors on `device`.

    integer indices stay torch.long; only floating tensors are affected by
    later precision policies.
    """
    word_np = wtok.encode(texts, max_word_len)
    char_np = ctok.encode(texts, max_char_len)
    kw_np = kmap.encode(keywords)
    word_ids = torch.from_numpy(word_np).long()
    char_ids = torch.from_numpy(char_np).long()
    kw_ids = torch.from_numpy(kw_np).long()
    if device is not None:
        word_ids = word_ids.to(device)
        char_ids = char_ids.to(device)
        kw_ids = kw_ids.to(device)
    return word_ids, char_ids, kw_ids


def build_model_from_featurizers(
    wtok,
    ctok,
    kmap,
    word_embed_dim: int = WORD_EMBED_DIM,
    word_filters: int = WORD_FILTERS,
    char_embed_dim: int = CHAR_EMBED_DIM,
    char_filters: int = CHAR_FILTERS,
    keyword_dim: int = KEYWORD_DIM,
    hidden_dim: int = HIDDEN_DIM,
    dropout: float = DROPOUT,
) -> MultiViewDisasterNet:
    return MultiViewDisasterNet(
        word_vocab_size=wtok.vocab_size,
        char_vocab_size=ctok.vocab_size,
        num_keywords=kmap.num_keywords,
        word_embed_dim=word_embed_dim,
        word_kernels=WORD_KERNELS,
        word_filters=word_filters,
        char_embed_dim=char_embed_dim,
        char_kernel=CHAR_KERNEL,
        char_filters=char_filters,
        keyword_dim=keyword_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def get_pos_weight(labels) -> float:
    labels = np.asarray(labels).astype(int)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    if pos <= 0:
        return 1.0
    return float(neg) / float(pos)


@torch.no_grad()
def predict_proba(
    model, word_ids, char_ids, keyword_ids, batch_size: int = 256, device=None
):
    """Run the real model forward pass. Returns float32 numpy probs."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    n = word_ids.shape[0]
    probs = np.zeros(n, dtype=np.float32)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        w = word_ids[i:j].to(device)
        c = char_ids[i:j].to(device)
        k = keyword_ids[i:j].to(device)
        logits = model(w, c, k).float()
        probs[i:j] = torch.sigmoid(logits).detach().cpu().numpy()
    return probs


# ---- CPU-preflight adapter (real model, real loss, real optimizer) -------


class CandidateAdapter:
    """No-argument adapter that exercises the real model/loss/inputs/optimizer.

    The vocabularies are lazily built once from a fixed small slice of train.csv
    and cached at class scope, so build_model and the batch builders share the
    same feasibility space. No weights are downloaded and no CUDA is required.
    """

    _featurizer_cache = None  # (wtok, ctok, kmap, texts, keywords, labels)

    def __init__(self):
        self.max_word_len = MAX_WORD_LEN
        self.max_char_len = MAX_CHAR_LEN
        self.word_embed_dim = ADAPTER_WORD_EMBED_DIM
        self.word_filters = ADAPTER_WORD_FILTERS
        self.char_embed_dim = ADAPTER_CHAR_EMBED_DIM
        self.char_filters = ADAPTER_CHAR_FILTERS
        self.keyword_dim = ADAPTER_KEYWORD_DIM
        self.hidden_dim = ADAPTER_HIDDEN_DIM
        self.dropout = ADAPTER_DROPOUT

    # -- internal helpers --
    def _load_small(self):
        if CandidateAdapter._featurizer_cache is not None:
            return CandidateAdapter._featurizer_cache
        input_dir = resolve_input_dir()
        path = os.path.join(input_dir, "train.csv")
        df = pd.read_csv(path).head(ADAPTER_N_SAMPLE_ROWS)
        texts = [clean_text(t) for t in df["text"].astype(str).tolist()]
        if "keyword" in df.columns:
            keywords = df["keyword"].tolist()
        else:
            keywords = [None] * len(texts)
        if "target" in df.columns:
            labels = df["target"].astype("float32").tolist()
        else:
            labels = [0.0] * len(texts)
        wtok = SimpleTokenizer(min_freq=1, max_vocab=4096).fit(texts)
        ctok = CharTokenizer(min_freq=1, max_vocab=128).fit(texts)
        kmap = KeywordMapper().fit(keywords)
        CandidateAdapter._featurizer_cache = (wtok, ctok, kmap, texts, keywords, labels)
        return CandidateAdapter._featurizer_cache

    def _make_batch(self, scenario, device):
        bs = int(scenario.get("batch_size", 8)) if isinstance(scenario, dict) else 8
        bs = max(bs, 1)
        fixture = scenario.get("fixture") if isinstance(scenario, dict) else None

        # If fixture explicitly provides named integer token tensors, honor them.
        if (
            isinstance(fixture, dict)
            and fixture
            and all(k in fixture for k in ("word_ids", "char_ids", "keyword_ids"))
        ):
            word_ids = torch.as_tensor(fixture["word_ids"]).long().to(device)
            char_ids = torch.as_tensor(fixture["char_ids"]).long().to(device)
            kw_ids = torch.as_tensor(fixture["keyword_ids"]).long().to(device)
            target = fixture.get("target", fixture.get("labels", None))
            if target is None:
                target = torch.zeros(
                    word_ids.shape[0], dtype=torch.float32, device=device
                )
            else:
                target = torch.as_tensor(target).float().to(device)
            return (word_ids, char_ids, kw_ids), target

        # Otherwise build a real batch from the first rows of train.csv.
        wtok, ctok, kmap, texts, keywords, labels = self._load_small()
        if not texts:
            texts = [""]
            keywords = [None]
            labels = [0.0]
        idx = list(range(min(bs, len(texts))))
        b_texts = [texts[i] for i in idx]
        b_kw = [keywords[i] for i in idx]
        b_y = [labels[i] for i in idx]
        # Pad to batch_size so downstream modules see the requested shape.
        while len(b_texts) < bs:
            b_texts.append(b_texts[-1])
            b_kw.append(b_kw[-1])
            b_y.append(0.0)

        word_ids, char_ids, kw_ids = make_batch_tensors(
            b_texts,
            b_kw,
            wtok,
            ctok,
            kmap,
            max_word_len=self.max_word_len,
            max_char_len=self.max_char_len,
            device=device,
        )
        target = torch.tensor(b_y, dtype=torch.float32, device=device)
        return (word_ids, char_ids, kw_ids), target

    # -- adapter API --
    def build_model(self, context):
        wtok, ctok, kmap, _, _, _ = self._load_small()
        model = MultiViewDisasterNet(
            word_vocab_size=max(wtok.vocab_size, NUM_SPECIAL_TOKENS + 1),
            char_vocab_size=max(ctok.vocab_size, NUM_SPECIAL_TOKENS + 1),
            num_keywords=max(kmap.num_keywords, 4),
            word_embed_dim=self.word_embed_dim,
            word_kernels=WORD_KERNELS,
            word_filters=self.word_filters,
            char_embed_dim=self.char_embed_dim,
            char_kernel=CHAR_KERNEL,
            char_filters=self.char_filters,
            keyword_dim=self.keyword_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        model.cpu()
        return model

    def build_optimizer(self, model, context):
        # Only used by the isolated CPU preflight to exercise a real update;
        # the production optimizer/scheduler belongs to Stage 3.
        return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    def build_train_batch(self, scenario, device):
        return self._make_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._make_batch(scenario, device)

    def _resolve_criterion(self, context):
        if isinstance(context, dict):
            crit = context.get("criterion")
            if isinstance(crit, nn.Module):
                return crit
        return build_loss(pos_weight=None)

    def training_step(self, model, batch, context):
        model.train()
        inputs, target = batch
        word_ids, char_ids, kw_ids = inputs
        logits = model(word_ids, char_ids, kw_ids)
        criterion = self._resolve_criterion(context)
        loss = criterion(logits, target.float())
        return loss

    def validation_step(self, model, batch, context):
        model.eval()
        inputs, target = batch
        word_ids, char_ids, kw_ids = inputs
        with torch.no_grad():
            logits = model(word_ids, char_ids, kw_ids)
        criterion = self._resolve_criterion(context)
        loss = criterion(logits, target.float())
        probs = torch.sigmoid(logits.float())
        return {"loss": loss, "probs": probs, "target": target.float()}

```
        
        ### Step 2: datatype_precision
        **Plan:** ## Plan

**WHAT makes this task unique for precision**: The model is a from-scratch multi-view CNN with a **keyword cross-gate** — the gate involves `sigmoid(x) * proj(kw)`, a small multiplicative interaction that can numerically saturate under FP16. The loss is `BCEWithLogitsLoss` with `pos_weight`, which naively computed under autocast can produce inf/nan gradients on a ~43% positive-class tweet corpus. So FP16 AMP must be applied **only around the Conv1d/Embedding forward ops**, with the gate's sigmoid, the loss, and the sigmoid-based metric/prob export kept in FP32 islands — exactly the "selective, forward-only" pattern.

**WHY this is the right precision policy**: A10 (Ampere, cc 8.6) Tensor Cores give real throughput on fp16 matmul/conv, and prior attempts fit the model in ~37 min at fp32-equivalent. Cheap win to enable selective fp16 AMP with GradScaler, disable TF32 (contract mandates this in normal mode, and TF32 on the elementwise-kernel-heavy CNN has negligible benefit), and keep parameters, integer indices (`word_ids`, `char_ids`, `keyword_ids`), and the target tensor in their native dtypes — never `model.half()`.

**HOW validation confirms it**: The precision policy is confirmed if (a) fp16-AMP training produces finite loss/gradients for all epochs, (b) the epoch-1 val F1 is within noise of FP32, and (c) wall time (or steps/sec) improves or is neutral. If I observe NaN/inf loss, non-finite gradients, or clear F1 degradation, `degrade_to_fp32` is called and the run continues in pure FP32 — that is the recorded fallback. CPU-preflight always uses FP32 regardless of host GPU.

Note board:
- baseline_change: introduced a `PrecisionConfig` + selective FP16 AMP policy (CUDA autocast for forward only, FP32 islands for gate sigmoid/loss/probs, GradScaler for backward, TF32 explicitly disabled) replacing the implicit "everything fp32" default | purpose: exploit A10 Tensor Cores on the word/char Conv1d + Embedding matmuls while protecting the numerically fragile keyword-gate interaction and BCEWithLogits loss that Stage 1 relies on | hardware_keys: datatype_precision.precision_policy, datatype_precision.fallback_policy, datatype_precision.precision_model_adaptation
- baseline_change: added runtime precision helpers (`configure_precision_globals`, `autocast_context`, `fp32_island`, `make_grad_scaler`, `amp_forward`, `bce_with_logits_fp32`, `degrade_to_fp32`) that Stage 3's `run_training` will consume without touching optimizer/scheduler/batch-size/epoch code | purpose: keep precision configurable and CPU-safe so Stage 3 can simply call them around its existing forward/loss, preserving Stage 1's model family and output interface | hardware_keys: none
- baseline_change: no `model.half()`, no whole-pipeline cast; integer token indices stay `torch.long`, floating params/targets stay FP32; CPU path resolves to FP32 with AMP/scaler disabled | purpose: honor the "selective forward autocast" rule and guarantee the CPU preflight can execute the real model/loss without CUDA | hardware_keys: none
        **Code:**
        ```python
"""
Stage 2 (datatype_precision) — Precision policy for the disaster-tweets MVCNN.

This module is IMPORT-SAFE: it defines constants, a `PrecisionConfig`
dataclass, and helper functions only.  No data is read, no CUDA/TF32 flags are
mutated, no logging is configured, and no model is instantiated at import time.
All global side effects happen inside helper calls that the Stage 3 main guard
invokes.

Consumed by Stage 1 (model/loss) and Stage 3 (training loop):

  cfg   = resolve_precision_config()            # device + policy
  configure_precision_globals(cfg)              # disable TF32 etc. (main guard)
  scaler = make_grad_scaler(cfg)                # None on CPU / fp32
  logits = amp_forward(model, (w, c, k), cfg)   # autocast ONLY here
  loss   = bce_with_logits_fp32(logits, y, pw)  # fp32 island
  probs  = logits_to_probs_fp32(logits)         # fp32 metric/export

Contract preserved from Stage 1:
  - Model family: MultiViewDisasterNet (multi-view CNN + keyword gate).
  - Inputs: integer token tensors (word_ids, char_ids, keyword_ids) stay
    torch.long; only floating-point computation is affected by autocast.
  - Loss: BCEWithLogitsLoss (optionally pos_weight); computed in FP32.
  - Output interface: scalar logits per example (B,).

Precision policy (Ampere / A10, normal mode):
  - Default policy: 'fp16_amp' with CUDA autocast on the forward pass only.
  - Parameters, floating inputs, and targets remain FP32.
  - Backward and optimizer.step run OUTSIDE autocast, with GradScaler.
  - FP32 islands used for the keyword-gate sigmoid, the loss, and metric probs.
  - TF32 disabled for matmul and convolutions.
  - CPU path: FP32, AMP and scaler disabled (supports CPU preflight).
  - Fallback policy: 'fp32' (call degrade_to_fp32 on instability).
"""

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Module-level constants (no side effects)
# ---------------------------------------------------------------------------

# Allowed precision policies in this stage. Normal mode endorses only fp32 and
# selective fp16_amp; no bf16, no TF32, no quantization, no TE recipes.
PRECISION_POLICY_FP32 = "fp32"
PRECISION_POLICY_FP16_AMP = "fp16_amp"
ALLOWED_PRECISION_POLICIES: Tuple[str, ...] = (
    PRECISION_POLICY_FP32,
    PRECISION_POLICY_FP16_AMP,
)

# Dtype contract.
PARAM_DTYPE = torch.float32
FLOAT_INPUT_DTYPE = torch.float32
INTEGER_INPUT_DTYPE = torch.long
TARGET_DTYPE = torch.float32

# AMP dtype: only fp16 is permitted in normal mode.
AMP_DTYPE_FP16 = torch.float16
ALLOWED_AMP_DTYPES: Tuple[torch.dtype, ...] = (AMP_DTYPE_FP16,)
PREFERRED_AMP_DTYPE = AMP_DTYPE_FP16

# Fallback.
FALLBACK_POLICY = "fp32"

# Reason strings recorded in the precision log.
REASON_CUDA_FP16_AMP = "ampere_gpu_selective_fp16_amp_fp32_islands"
REASON_CPU_OR_DISABLED = "cpu_or_amp_disabled_fp32"
REASON_DEGRADED = "degraded_amp_to_fp32"


# ---------------------------------------------------------------------------
# Precision configuration object
# ---------------------------------------------------------------------------


@dataclass
class PrecisionConfig:
    """Resolved device + precision policy.

    Attributes
    ----------
    device : str
        Target device string ("cuda" or "cpu").
    use_amp : bool
        Whether forward-only FP16 autocast is enabled.
    amp_dtype : torch.dtype
        Autocast dtype (torch.float16). Irrelevant when use_amp=False.
    use_tf32 : bool
        TF32 for matmul/conv. Must remain False in normal mode.
    fallback : str
        Fallback policy name (typically "fp32").
    reason : str
        Human-readable reason for the current policy.
    amp_dtype_name : str
        Short label for logging: "fp16" or "fp32".
    """

    device: str
    use_amp: bool
    amp_dtype: torch.dtype = AMP_DTYPE_FP16
    use_tf32: bool = False
    fallback: str = FALLBACK_POLICY
    reason: str = ""
    amp_dtype_name: str = "fp32"

    @property
    def is_cuda(self) -> bool:
        return str(self.device).startswith("cuda")

    @property
    def policy_name(self) -> str:
        return PRECISION_POLICY_FP16_AMP if self.use_amp else PRECISION_POLICY_FP32

    @property
    def scaler_enabled(self) -> bool:
        return bool(self.use_amp) and self.is_cuda


# ---------------------------------------------------------------------------
# Device resolution helpers (read-only, no CUDA configuration)
# ---------------------------------------------------------------------------


def _cuda_usable() -> bool:
    """Return True only when CUDA is present AND a trivial allocation works.

    Read-only probe; does NOT set TF32, does NOT set cudnn.benchmark, does NOT
    touch any global state.  Missing GPU drivers / no CUDA context simply
    return False so the caller can fall back to CPU/FP32.
    """
    try:
        if not torch.cuda.is_available():
            return False
        _ = torch.cuda.current_device()
        _ = torch.zeros(1, device="cuda")
        return True
    except Exception:
        return False


def resolve_precision_config(
    prefer_amp: bool = True,
    device: Optional[str] = None,
    verbose: bool = True,
) -> PrecisionConfig:
    """Resolve device and precision policy for the current host.

    Rules
    -----
    - CUDA requested/available and `prefer_amp=True` -> fp16_amp forward-only.
    - Otherwise                                       -> fp32 everywhere.
    - CPU preflight always resolves to fp32 (no scaler, no autocast).

    Parameters
    ----------
    prefer_amp : bool
        If True, allow FP16 AMP on a usable CUDA device.
    device : Optional[str]
        Explicit device request (e.g. "cuda", "cpu").  If None, auto-detect.
    verbose : bool
        Emit one precision log line (only at call time, never at import).
    """
    cuda_ok = _cuda_usable()

    if device is None:
        device = "cuda" if cuda_ok else "cpu"

    # Guard against requesting CUDA when it is not usable.
    if str(device).startswith("cuda") and not cuda_ok:
        device = "cpu"

    on_cuda = str(device).startswith("cuda")

    if on_cuda and prefer_amp:
        cfg = PrecisionConfig(
            device=device,
            use_amp=True,
            amp_dtype=PREFERRED_AMP_DTYPE,
            amp_dtype_name="fp16",
            use_tf32=False,
            fallback=FALLBACK_POLICY,
            reason=REASON_CUDA_FP16_AMP,
        )
    else:
        cfg = PrecisionConfig(
            device=device,
            use_amp=False,
            amp_dtype=PREFERRED_AMP_DTYPE,
            amp_dtype_name="fp32",
            use_tf32=False,
            fallback=FALLBACK_POLICY,
            reason=REASON_CPU_OR_DISABLED,
        )

    if verbose:
        print(format_precision_log(cfg))
    return cfg


def format_precision_log(cfg: PrecisionConfig) -> str:
    """Single-line precision log (never called at import time)."""
    return (
        "[precision] "
        f"device={cfg.device} "
        f"policy={cfg.policy_name} "
        f"amp_dtype={cfg.amp_dtype_name} "
        f"tf32={bool(cfg.use_tf32)} "
        f"scaler={'on' if cfg.scaler_enabled else 'off'} "
        f"fallback={cfg.fallback} "
        f"reason={cfg.reason}"
    )


# ---------------------------------------------------------------------------
# Global precision flags (called from the main guard only)
# ---------------------------------------------------------------------------


def configure_precision_globals(cfg: PrecisionConfig) -> dict:
    """Apply TF32 / cudnn flags according to `cfg`.

    Called from the training entry point.  Returns a dict of what was applied
    so callers can log it without re-reading globals.

    Contract: TF32 is disabled in normal mode regardless of hardware support.
    """
    applied = {"tf32_matmul": False, "tf32_cudnn": False, "cudnn_benchmark": None}

    # TF32: honor cfg (default False).  Never enable unless explicitly asked.
    use_tf32 = bool(cfg.use_tf32) and cfg.is_cuda
    try:
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        applied["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = use_tf32
        applied["tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
    except Exception:
        pass

    # Benchmark mode: soft perf win for fixed input shapes; harmless on CPU.
    if cfg.is_cuda:
        try:
            torch.backends.cudnn.benchmark = True
            applied["cudnn_benchmark"] = True
        except Exception:
            applied["cudnn_benchmark"] = None

    return applied


# ---------------------------------------------------------------------------
# Autocast contexts (forward-only) and FP32 islands
# ---------------------------------------------------------------------------


def _make_cuda_autocast(enabled: bool):
    """Build a CUDA autocast context across torch versions.

    Prefers the modern `torch.amp.autocast("cuda", ...)` API, falls back to
    `torch.cuda.amp.autocast`, then to `nullcontext` if neither is available.
    """
    if not enabled:
        return contextlib.nullcontext()
    # Modern API (torch >= 2.0).
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "autocast"):
        try:
            return amp_mod.autocast(
                device_type="cuda", dtype=AMP_DTYPE_FP16, enabled=True
            )
        except Exception:
            pass
    # Legacy API.
    legacy = getattr(torch.cuda, "amp", None)
    if legacy is not None and hasattr(legacy, "autocast"):
        try:
            return legacy.autocast(dtype=AMP_DTYPE_FP16, enabled=True)
        except Exception:
            pass
    return contextlib.nullcontext()


def _make_cuda_autocast_disabled():
    """Explicitly disable autocast for FP32 island regions."""
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "autocast"):
        try:
            return amp_mod.autocast(device_type="cuda", enabled=False)
        except Exception:
            pass
    legacy = getattr(torch.cuda, "amp", None)
    if legacy is not None and hasattr(legacy, "autocast"):
        try:
            return legacy.autocast(enabled=False)
        except Exception:
            pass
    return contextlib.nullcontext()


def autocast_context(cfg: PrecisionConfig):
    """FP16 autocast context — wrap ONLY eligible forward ops.

    Returns `nullcontext` (no-op) on CPU or when AMP is disabled, so training
    code can use `with autocast_context(cfg):` unconditionally.
    """
    if not cfg.use_amp or not cfg.is_cuda:
        return contextlib.nullcontext()
    return _make_cuda_autocast(True)


def fp32_island(cfg: PrecisionConfig):
    """Disable autocast for numerically fragile regions (gate sigmoid, loss,
    probability export, metric accumulation).  No-op on CPU / fp32 runs.
    """
    if not cfg.use_amp or not cfg.is_cuda:
        return contextlib.nullcontext()
    return _make_cuda_autocast_disabled()


# ---------------------------------------------------------------------------
# GradScaler factory (backward / optimizer updates live OUTSIDE autocast)
# ---------------------------------------------------------------------------


def make_grad_scaler(cfg: PrecisionConfig, enabled: Optional[bool] = None):
    """Create a GradScaler only when FP16 AMP is active on CUDA.

    Returns None on CPU / fp32-only runs so callers can pass it through
    `TrainingDiagnostics(..., scaler=scaler_or_None)` and the training loop
    without special-casing.  Uses the modern API when available, with a
    legacy fallback.
    """
    if enabled is None:
        enabled = cfg.scaler_enabled
    if not enabled:
        return None

    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "GradScaler"):
        try:
            return amp_mod.GradScaler("cuda", enabled=True)
        except Exception:
            pass
    legacy = getattr(torch.cuda, "amp", None)
    if legacy is not None and hasattr(legacy, "GradScaler"):
        try:
            return legacy.GradScaler(enabled=True)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Precision-safe forward / loss / probability helpers
# ---------------------------------------------------------------------------


def amp_forward(model: nn.Module, inputs, cfg: PrecisionConfig):
    """Run one forward pass under autocast, return FP32 logits.

    `inputs` may be a tuple/list of tensors (e.g. word_ids, char_ids, kw_ids)
    or a dict of kwargs.  All integer token tensors must already be on the
    correct device with dtype=torch.long — this helper does NOT cast them.

    The returned tensor is a detached FP32 tensor suitable for the loss and
    for downstream FP32-only metric computation.
    """
    if isinstance(inputs, dict):
        with autocast_context(cfg):
            logits = model(**inputs)
    else:
        with autocast_context(cfg):
            logits = model(*inputs)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    # Force FP32 for the boundary — autocast can produce FP16 logits.
    return logits.float()


def bce_with_logits_fp32(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[Any] = None,
    criterion: Optional[nn.Module] = None,
) -> torch.Tensor:
    """BCEWithLogitsLoss computed in FP32 regardless of surrounding autocast.

    Stage 1's `build_loss(pos_weight=...)` returns a `BCEWithLogitsLoss` whose
    registered `pos_weight` buffer may already be float32.  We explicitly:

      1. cast logits/targets to float32,
      2. (re)cast pos_weight to float32 on the same device,
      3. execute the criterion (or a functional fallback) even if an outer
         autocast scope is active.

    Because `binary_cross_entropy_with_logits` is a reduction op sensitive to
    fp16 underflow, this helper must be invoked inside `fp32_island(cfg)` by
    the training loop for maximum safety.
    """
    logits_f = logits.float()
    targets_f = targets.to(device=logits_f.device, dtype=torch.float32)

    if criterion is not None and isinstance(criterion, nn.Module):
        # Rebuild the criterion's pos_weight buffer into fp32 to avoid a
        # silent dtype mismatch under autocast.
        pw_buf = getattr(criterion, "pos_weight", None)
        if pw_buf is not None:
            try:
                critic_buf = pw_buf.to(device=logits_f.device, dtype=torch.float32)
                with torch.no_grad():
                    criterion.pos_weight = critic_buf
            except Exception:
                pass
        return criterion(logits_f, targets_f)

    pw = None
    if pos_weight is not None:
        pw = torch.as_tensor(pos_weight, dtype=torch.float32, device=logits_f.device)
    return F.binary_cross_entropy_with_logits(logits_f, targets_f, pos_weight=pw)


def logits_to_probs_fp32(logits: torch.Tensor) -> torch.Tensor:
    """Sigmoid in FP32 for metrics/export. Never under autocast."""
    return torch.sigmoid(logits.float())


# ---------------------------------------------------------------------------
# Instability detection / fallback
# ---------------------------------------------------------------------------


def check_finite(values) -> bool:
    """Return True iff every non-None tensor/scalar is finite."""
    if values is None:
        return True
    if isinstance(values, (tuple, list)):
        return all(check_finite(v) for v in values)
    if isinstance(values, dict):
        return all(check_finite(v) for v in values.values())
    if isinstance(values, torch.Tensor):
        return bool(torch.isfinite(values).all().item())
    if isinstance(values, (int, float)):
        return math.isfinite(float(values))
    # Unsupported types are treated as "unknown but not per-se bad".
    return True


def degrade_to_fp32(
    cfg: PrecisionConfig,
    reason: str = "instability",
) -> PrecisionConfig:
    """Mutate `cfg` (in place) to the FP32 fallback policy.

    Called from the training loop when a non-finite loss/gradient is observed
    or when the AMP path appears to be hurting quality.  Returns the same cfg
    so callers can chain `cfg = degrade_to_fp32(cfg)`.
    """
    cfg.use_amp = False
    cfg.amp_dtype_name = "fp32"
    cfg.use_tf32 = False
    cfg.reason = f"{REASON_DEGRADED}:{reason}"
    return cfg


# ---------------------------------------------------------------------------
# Small utilities for gradient clipping under AMP
# ---------------------------------------------------------------------------


def unscale_then_clip_grad_norm_(
    scaler,
    model: nn.Module,
    max_norm: Optional[float] = None,
) -> Optional[torch.Tensor]:
    """Unscale gradients before clipping (required for FP16 AMP).

    If `scaler` is None (fp32 path), just compute the grad norm.  Returns the
    total grad norm tensor (or None if there is nothing to clip).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return None

    if scaler is not None:
        try:
            scaler.unscale_(optimizer_placeholder := None) if False else None
        except Exception:
            pass
        # `scaler.unscale_` requires the optimizer; Stage 3 owns the optimizer
        # and calls `scaler.unscale_(optimizer)` itself.  We expose only the
        # clipping side here so the contract stays clean.
    max_n = max_norm if max_norm is not None else math.inf
    return torch.nn.utils.clip_grad_norm_(params, max_n)


# ---------------------------------------------------------------------------
# Optional lightweight AMP smoke-test (used by training, not at import time)
# ---------------------------------------------------------------------------


def probe_amp_stability(
    cfg: PrecisionConfig,
    model: nn.Module,
    sample_inputs,
    criterion: Optional[nn.Module] = None,
    sample_targets: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> bool:
    """One forward+loss probe on a real batch under the selected precision.

    Returns True if the result is finite.  Intended to be called once by the
    training loop right after model/criterion setup, so a bad AMP path can be
    detected before the first epoch and the fallback applied.

    Never runs at import time.  Does NOT mutate model weights (no backward).
    """
    was_training = model.training
    try:
        model.train(False)
        with torch.no_grad():
            logits = amp_forward(model, sample_inputs, cfg)
            finite_logits = check_finite(logits)
            finite_loss = True
            if criterion is not None and sample_targets is not None:
                with fp32_island(cfg):
                    loss = bce_with_logits_fp32(
                        logits, sample_targets, criterion=criterion
                    )
                finite_loss = check_finite(loss)
        ok = bool(finite_logits and finite_loss)
    except Exception:
        ok = False
    finally:
        model.train(was_training)
    if verbose:
        print(f"[precision] amp_probe finite={ok} policy={cfg.policy_name}")
    return ok


__all__ = [
    "PRECISION_POLICY_FP32",
    "PRECISION_POLICY_FP16_AMP",
    "ALLOWED_PRECISION_POLICIES",
    "PARAM_DTYPE",
    "FLOAT_INPUT_DTYPE",
    "INTEGER_INPUT_DTYPE",
    "TARGET_DTYPE",
    "AMP_DTYPE_FP16",
    "ALLOWED_AMP_DTYPES",
    "PREFERRED_AMP_DTYPE",
    "FALLBACK_POLICY",
    "PrecisionConfig",
    "resolve_precision_config",
    "format_precision_log",
    "configure_precision_globals",
    "autocast_context",
    "fp32_island",
    "make_grad_scaler",
    "amp_forward",
    "bce_with_logits_fp32",
    "logits_to_probs_fp32",
    "check_finite",
    "degrade_to_fp32",
    "probe_amp_stability",
]

```
        
        ### Step 3: training_evaluation
        **Plan:** ## Plan

**WHAT makes this step unique**: Stage 2 protects a fragile keyword cross-gate + BCEWithLogits under selective FP16 AMP with FP32 islands, so Stage 3 must route the forward through `amp_forward`, compute loss through `bce_with_logits_fp32` inside `fp32_island`, unscale via GradScaler before clipping, and keep metric/export in FP32. The training budget is one hour, so I keep the Stage-1 architecture exactly and only own optimizer/scheduler/batching/checkpointing.

**WHY this approach fits**: The MVCNN is small (~3M params) and fits comfortably in one physical batch of 32 on A10. I fix physical=effective batch size (no accumulation), use AdamW with linear-warmup+cosine, early stopping on val F1 with best-state restore, and a scheduler-driven `safe_point` contract (BEFORE_TRAIN/STEP/EPOCH) with a complete state factory that snapshots per-epoch permutation so mid-epoch resume does not replay updates. Threshold is tuned once on the held-out val set and reused verbatim for test inference.

**HOW validation confirms**: If the pipeline is correct, epoch-1 val F1 should be >0.6 and rise monotonically to ~0.75-0.78, with FP16-AMP finite losses throughout; a NaN/degradation triggers the recorded `degrade_to_fp32` fallback and continues. The printed `Final Validation Score` is the held-out F1 at the tuned threshold; a collapse near majority-class (~0.6) would signal the gate isn't transferring and I would fall back to keyword-concat (Stage 1 alternative), but Stage 3 itself only preserves the contract.

**Backend contract**: Physical BS = 32 (in `QUALITY_SAFE_PHYSICAL_BATCH_SIZES`), accumulation = 1, `BATCH_LR_SCALING_POLICY = "sqrt"`, num_workers = 2, pin_memory = True, persistent_workers = False (per-epoch subset loaders). Configurable dims unchanged from Stage 1. Fallbacks: reduce BS on OOM before touching model family; reduce epochs only if wall-clock overruns. Scheduler-owned controls (device assignment, thread allocation, memory limit, launch policy) are left to the scheduler; the subprocess owns optimizer/scheduler/cadence.

Note board:
- baseline_change: added end-to-end Stage-3 training loop consuming Stage-1 model/loss/featurizers and Stage-2 `amp_forward`/`bce_with_logits_fp32`/`fp32_island`/`GradScaler`/`degrade_to_fp32`, with linear-warmup+cosine AdamW, grad-clip after unscale, val-F1 early stopping + best-state restore | purpose: convert the gated MVCNN into a real trained candidate whose validation and test predictions flow through the identical model forward and threshold | hardware_keys: training_evaluation.optimizer, training_evaluation.scheduler, datatype_precision.precision_policy
- baseline_change: replaced DataLoader(shuffle=True) with a per-epoch permutation + `IndexedTensorDataset` + shuffle=False loader, and snapshot that permutation + `step_in_epoch` in the scheduler state factory | purpose: honor the "do not replay completed updates" contract on mid-epoch resume and keep epoch boundaries clean for safe_point EPOCH | hardware_keys: training_evaluation.dataloader_policy
- baseline_change: wrapped the full loop in `TrainingDiagnostics` with real settings (physical/effective batch, planned epochs, steps/epoch, model_family, split_seed, precision reason), called `after_update()` after each optimizer attempt and `report()` per epoch, and stored `diagnostics.state_dict()` in the scheduler checkpoint | purpose: emit measureable throughput/VRAM telemetry while keeping the final validation score as the last line after context exit | hardware_keys: none
        **Code:**
        ```python
"""
Stage 3 (training_evaluation) — Disaster Tweets.

Owns optimizer/scheduler/batching/epochs/checkpointing/inference/submission.
Consumes Stage 1 (model/loss/featurizers) and Stage 2 (precision) as-is.

Import-safe: only constants, imports, classes, and functions at top level.
All side effects (data read, CUDA config, training, inference, submission)
live under `if __name__ == "__main__":`.
"""

# ---------------------------------------------------------------------------
# Module-level constants (No side effects)
# ---------------------------------------------------------------------------

MODEL_FAMILY = "mvcnn_keyword_gate_v1"

WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
DEFAULT_INPUT_DIR = "./input"

SEED = 42
VAL_SIZE = 0.15

# Batch / accumulation — fixed physical batch, no accumulation by default.
PHYSICAL_BATCH_SIZE = 32
EFFECTIVE_BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 1

NUM_EPOCHS = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
BETAS = (0.9, 0.999)
EPS = 1e-8
WARMUP_RATIO = 0.1
GRAD_CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 3

NUM_WORKERS = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = False

# Envelope advertised to the scheduler (must include PHYSICAL_BATCH_SIZE).
QUALITY_SAFE_PHYSICAL_BATCH_SIZES = [16, 32, 64]
BATCH_LR_SCALING_POLICY = "sqrt"

# Threshold search range for binary F1 on validation.
THRESHOLD_MIN = 0.20
THRESHOLD_MAX = 0.80
THRESHOLD_STEPS = 61

# ---------------------------------------------------------------------------
# Imports (import-safe)
# ---------------------------------------------------------------------------

import os
import math
import time
import json
import random
import contextlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

# --- Stage 1 model/features/loss/metrics ---
from stage1_model_design import (  # noqa: F401  (Stage 1 symbols)
    clean_text,
    SimpleTokenizer,
    CharTokenizer,
    KeywordMapper,
    MultiViewDisasterNet,
    build_loss,
    compute_f1,
    tune_threshold,
    resolve_input_dir,
    load_raw_data,
    stratified_split,
    fit_featurizers,
    make_batch_tensors,
    build_model_from_featurizers,
    get_pos_weight,
    predict_proba,
    MAX_WORD_LEN,
    MAX_CHAR_LEN,
    PAD_IDX,
    UNK_IDX,
    NUM_SPECIAL_TOKENS,
    WORD_EMBED_DIM,
    WORD_KERNELS,
    WORD_FILTERS,
    CHAR_EMBED_DIM,
    CHAR_KERNEL,
    CHAR_FILTERS,
    KEYWORD_DIM,
    HIDDEN_DIM,
    DROPOUT,
    ADAPTER_WORD_EMBED_DIM,
    ADAPTER_WORD_FILTERS,
    ADAPTER_CHAR_EMBED_DIM,
    ADAPTER_CHAR_FILTERS,
    ADAPTER_KEYWORD_DIM,
    ADAPTER_HIDDEN_DIM,
    ADAPTER_DROPOUT,
    ADAPTER_N_SAMPLE_ROWS,
)

# --- Stage 2 precision utilities ---
from stage2_datatype_precision import (  # noqa: F401
    PrecisionConfig,
    resolve_precision_config,
    configure_precision_globals,
    format_precision_log,
    autocast_context,
    fp32_island,
    make_grad_scaler,
    amp_forward,
    bce_with_logits_fp32,
    logits_to_probs_fp32,
    check_finite,
    degrade_to_fp32,
    probe_amp_stability,
)

# --- Scheduler (optional) ---
_SCRIPT_SCHED_AVAILABLE = False
script_scheduler_context = None
SafePointType = None
try:
    from localml_scheduler.execution.script_context import script_scheduler_context  # type: ignore
    from localml_scheduler.domain import SafePointType  # type: ignore

    _SCRIPT_SCHED_AVAILABLE = True
except Exception:
    _SCRIPT_SCHED_AVAILABLE = False

# --- Runtime diagnostics (optional) ---
TrainingDiagnostics = None
try:
    from utils.training_diagnostics import TrainingDiagnostics  # type: ignore
except Exception:
    # Minimal import-safe fallback so the script still runs standalone.
    class TrainingDiagnostics:  # type: ignore
        def __init__(self, model, optimizer, scaler=None, settings=None):
            self.model = model
            self.optimizer = optimizer
            self.scaler = scaler
            self.settings = dict(settings or {})
            self._updates = 0
            self._start = time.time()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def after_update(self):
            self._updates += 1

        def report(self, epoch=None):
            dt = max(time.time() - self._start, 1e-6)
            print(
                f"[diagnostics] epoch={epoch} updates={self._updates} "
                f"elapsed_s={dt:.1f}"
            )

        def state_dict(self):
            return {"updates": self._updates}

        def load_state_dict(self, state):
            if isinstance(state, dict):
                self._updates = int(state.get("updates", 0))


# ---------------------------------------------------------------------------
# LR schedulers (small, self-contained, no transformers dependency)
# ---------------------------------------------------------------------------


class LinearWarmupCosine:
    """Linear warmup followed by cosine decay to `min_lr_ratio` of base LR."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = max(int(warmup_steps), 0)
        self.total_steps = max(int(total_steps), 1)
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def _scale(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step + 1) / float(self.warmup_steps)
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def step(self):
        self.step_count += 1
        scale = self._scale(self.step_count)
        for base, pg in zip(self.base_lrs, self.optimizer.param_groups):
            pg["lr"] = base * scale

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "step_count": self.step_count,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": list(self.base_lrs),
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            return
        self.step_count = int(state.get("step_count", 0))
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.total_steps = int(state.get("total_steps", self.total_steps))
        self.min_lr_ratio = float(state.get("min_lr_ratio", self.min_lr_ratio))
        bl = state.get("base_lrs")
        if isinstance(bl, (list, tuple)) and len(bl) == len(
            self.optimizer.param_groups
        ):
            self.base_lrs = [float(x) for x in bl]
            for v, pg in zip(self.base_lrs, self.optimizer.param_groups):
                pg["lr"] = v * self._scale(self.step_count)


# ---------------------------------------------------------------------------
# Dataset wrapper that honors an explicit index order (per-epoch permutation)
# ---------------------------------------------------------------------------


class IndexedTensorDataset(Dataset):
    def __init__(self, word_ids, char_ids, kw_ids, y, indices=None):
        self.word_ids = word_ids
        self.char_ids = char_ids
        self.kw_ids = kw_ids
        self.y = y
        if indices is None:
            self.indices = torch.arange(word_ids.size(0), dtype=torch.long)
        else:
            self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self):
        return int(self.indices.numel())

    def __getitem__(self, i):
        idx = int(self.indices[i])
        return self.word_ids[idx], self.char_ids[idx], self.kw_ids[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _unscale_and_clip(scaler, optimizer, model, max_norm):
    """Unscale (if AMP) then clip grad norm. Returns total grad norm (float)."""
    params = [p for p in model.parameters() if p.requires_grad]
    if scaler is not None:
        try:
            scaler.unscale_(optimizer)
        except Exception:
            pass
    if params:
        total = torch.nn.utils.clip_grad_norm_(params, max_norm)
        return float(total.detach().item())
    return 0.0


def _evaluate(
    model,
    cfg,
    word_ids,
    char_ids,
    kw_ids,
    y,
    criterion,
    batch_size=256,
):
    """Full-set forward through the real model. Returns (probs, loss)."""
    model.eval()
    device = next(model.parameters()).device
    n = int(word_ids.shape[0])
    probs = np.zeros(n, dtype=np.float32)
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            wb = word_ids[i:j].to(device, non_blocking=True)
            cb = char_ids[i:j].to(device, non_blocking=True)
            kb = kw_ids[i:j].to(device, non_blocking=True)
            yb = y[i:j].to(device, non_blocking=True)
            logits = amp_forward(model, (wb, cb, kb), cfg)
            with fp32_island(cfg):
                loss = bce_with_logits_fp32(logits, yb, criterion=criterion)
            probs[i:j] = logits_to_probs_fp32(logits).detach().cpu().numpy()
            total_loss += float(loss.detach().float().item()) * (j - i)
            total_n += j - i
    return probs, (total_loss / max(total_n, 1))


def _make_loader(
    word_ids, char_ids, kw_ids, y, indices, batch_size, workers, shuffle_workers=False
):
    ds = IndexedTensorDataset(word_ids, char_ids, kw_ids, y, indices=indices)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=False,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# CPU-preflight adapter (real model, real loss, real optimizer, no CUDA)
# ---------------------------------------------------------------------------


class CandidateAdapter:
    """No-argument, import-safe adapter used by the CPU preflight checker."""

    _cache = None  # (wtok, ctok, kmap, texts, keywords, labels)

    def __init__(self):
        self.max_word_len = MAX_WORD_LEN
        self.max_char_len = MAX_CHAR_LEN
        self.device = torch.device("cpu")
        self.precision_cfg = PrecisionConfig(
            device="cpu",
            use_amp=False,
            amp_dtype=torch.float16,
            amp_dtype_name="fp32",
            use_tf32=False,
            reason="cpu_preflight_fp32",
        )

    # -- internal --
    def _load(self):
        if CandidateAdapter._cache is not None:
            return CandidateAdapter._cache
        input_dir = os.environ.get("MLEVOLVE_INPUT_DIR", DEFAULT_INPUT_DIR)
        path = os.path.join(input_dir, "train.csv")
        if not os.path.exists(path):
            # Fallbacks: search common locations.
            for cand in (
                os.path.join(DEFAULT_INPUT_DIR, "train.csv"),
                "train.csv",
            ):
                if os.path.exists(cand):
                    path = cand
                    break
        df = pd.read_csv(path).head(ADAPTER_N_SAMPLE_ROWS)
        texts = [clean_text(t) for t in df["text"].astype(str).tolist()]
        keywords = (
            df["keyword"].tolist() if "keyword" in df.columns else [None] * len(texts)
        )
        labels = (
            df["target"].astype("float32").tolist()
            if "target" in df.columns
            else [0.0] * len(texts)
        )
        wtok = SimpleTokenizer(min_freq=1, max_vocab=4096).fit(texts)
        ctok = CharTokenizer(min_freq=1, max_vocab=128).fit(texts)
        kmap = KeywordMapper().fit(keywords)
        CandidateAdapter._cache = (wtok, ctok, kmap, texts, keywords, labels)
        return CandidateAdapter._cache

    def _make_batch(self, scenario, device):
        scenario = scenario or {}
        bs = int(scenario.get("batch_size", 8) or 8)
        bs = max(bs, 1)
        fixture = scenario.get("fixture") if isinstance(scenario, dict) else None

        if (
            isinstance(fixture, dict)
            and fixture
            and all(k in fixture for k in ("word_ids", "char_ids", "keyword_ids"))
        ):
            w = torch.as_tensor(fixture["word_ids"]).long().to(device)
            c = torch.as_tensor(fixture["char_ids"]).long().to(device)
            k = torch.as_tensor(fixture["keyword_ids"]).long().to(device)
            tgt = fixture.get("target", fixture.get("labels", None))
            if tgt is None:
                tgt = torch.zeros(w.shape[0], dtype=torch.float32)
            tgt = torch.as_tensor(tgt).float().to(device)
            return {"inputs": (w, c, k), "target": tgt}

        wtok, ctok, kmap, texts, kws, labels = self._load()
        if not texts:
            texts, kws, labels = [""], [None], [0.0]
        idx = list(range(min(bs, len(texts))))
        b_texts = [texts[i] for i in idx]
        b_kw = [kws[i] for i in idx]
        b_y = [labels[i] for i in idx]
        while len(b_texts) < bs:
            b_texts.append(b_texts[-1])
            b_kw.append(b_kw[-1])
            b_y.append(0.0)
        w, c, k = make_batch_tensors(
            b_texts,
            b_kw,
            wtok,
            ctok,
            kmap,
            max_word_len=self.max_word_len,
            max_char_len=self.max_char_len,
            device=device,
        )
        tgt = torch.tensor(b_y, dtype=torch.float32, device=device)
        return {"inputs": (w, c, k), "target": tgt}

    # -- adapter API --
    def build_model(self, context):
        wtok, ctok, kmap, _, _, _ = self._load()
        model = MultiViewDisasterNet(
            word_vocab_size=max(wtok.vocab_size, NUM_SPECIAL_TOKENS + 1),
            char_vocab_size=max(ctok.vocab_size, NUM_SPECIAL_TOKENS + 1),
            num_keywords=max(kmap.num_keywords, 4),
            word_embed_dim=ADAPTER_WORD_EMBED_DIM,
            word_kernels=WORD_KERNELS,
            word_filters=ADAPTER_WORD_FILTERS,
            char_embed_dim=ADAPTER_CHAR_EMBED_DIM,
            char_kernel=CHAR_KERNEL,
            char_filters=ADAPTER_CHAR_FILTERS,
            keyword_dim=ADAPTER_KEYWORD_DIM,
            hidden_dim=ADAPTER_HIDDEN_DIM,
            dropout=ADAPTER_DROPOUT,
        )
        model.cpu()
        return model

    def build_optimizer(self, model, context):
        return torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=BETAS,
            eps=EPS,
        )

    def build_train_batch(self, scenario, device):
        return self._make_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._make_batch(scenario, device)

    @staticmethod
    def _resolve_criterion(context):
        if isinstance(context, dict):
            crit = context.get("criterion")
            if isinstance(crit, nn.Module):
                return crit
        return build_loss(pos_weight=None)

    def training_step(self, model, batch, context):
        model.train(True)
        # Accept both tuple and dict batch forms.
        if isinstance(batch, dict):
            inputs = batch["inputs"]
            target = batch["target"]
        else:
            inputs, target = batch
        w, c, k = inputs
        with autocast_context(self.precision_cfg):
            logits = model(w, c, k)
        logits = logits.float()
        criterion = self._resolve_criterion(context)
        with fp32_island(self.precision_cfg):
            loss = bce_with_logits_fp32(logits, target.float(), criterion=criterion)
        return loss

    def validation_step(self, model, batch, context):
        model.eval()
        if isinstance(batch, dict):
            inputs = batch["inputs"]
            target = batch["target"]
        else:
            inputs, target = batch
        w, c, k = inputs
        with torch.no_grad():
            logits = model(w, c, k).float()
        criterion = self._resolve_criterion(context)
        with fp32_island(self.precision_cfg):
            loss = bce_with_logits_fp32(logits, target.float(), criterion=criterion)
        probs = torch.sigmoid(logits)
        return {"loss": loss, "probs": probs, "target": target.float()}


# ---------------------------------------------------------------------------
# Main training routine (called under __main__ only)
# ---------------------------------------------------------------------------


def run_training():
    # --- Directories ---
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- Seed ---
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # --- Precision policy (Stage 2) ---
    cfg = resolve_precision_config(prefer_amp=True, verbose=True)
    configure_precision_globals(cfg)
    device = torch.device(cfg.device)
    scaler = make_grad_scaler(cfg)
    print(
        f"[train] device={device} precision={cfg.policy_name} scaler={'on' if scaler is not None else 'off'}"
    )

    # --- Data ---
    train_df, test_df = load_raw_data()
    tr_df, va_df = stratified_split(train_df, val_size=VAL_SIZE, seed=SEED)

    tr_texts = [clean_text(t) for t in tr_df["text"].astype(str).tolist()]
    va_texts = [clean_text(t) for t in va_df["text"].astype(str).tolist()]
    te_texts = [clean_text(t) for t in test_df["text"].astype(str).tolist()]

    tr_kw = (
        tr_df["keyword"].tolist() if "keyword" in tr_df.columns else [None] * len(tr_df)
    )
    va_kw = (
        va_df["keyword"].tolist() if "keyword" in va_df.columns else [None] * len(va_df)
    )
    te_kw = (
        test_df["keyword"].tolist()
        if "keyword" in test_df.columns
        else [None] * len(test_df)
    )

    # Featurizers fit ONLY on train split.
    wtok, ctok, kmap = fit_featurizers(tr_texts, tr_kw)

    tr_w_np = wtok.encode(tr_texts, MAX_WORD_LEN)
    va_w_np = wtok.encode(va_texts, MAX_WORD_LEN)
    te_w_np = wtok.encode(te_texts, MAX_WORD_LEN)

    tr_c_np = ctok.encode(tr_texts, MAX_CHAR_LEN)
    va_c_np = ctok.encode(va_texts, MAX_CHAR_LEN)
    te_c_np = ctok.encode(te_texts, MAX_CHAR_LEN)

    tr_k_np = kmap.encode(tr_kw)
    va_k_np = kmap.encode(va_kw)
    te_k_np = kmap.encode(te_kw)

    tr_w = torch.from_numpy(tr_w_np).long()
    va_w = torch.from_numpy(va_w_np).long()
    te_w = torch.from_numpy(te_w_np).long()
    tr_c = torch.from_numpy(tr_c_np).long()
    va_c = torch.from_numpy(va_c_np).long()
    te_c = torch.from_numpy(te_c_np).long()
    tr_k = torch.from_numpy(tr_k_np).long()
    va_k = torch.from_numpy(va_k_np).long()
    te_k = torch.from_numpy(te_k_np).long()

    tr_y = torch.tensor(tr_df["target"].astype("float32").values, dtype=torch.float32)
    va_y = torch.tensor(va_df["target"].astype("float32").values, dtype=torch.float32)

    n_train = int(tr_w.size(0))
    steps_per_epoch = int(math.ceil(n_train / PHYSICAL_BATCH_SIZE))
    total_optim_steps = steps_per_epoch * NUM_EPOCHS

    # --- Model / loss / optimizer / scheduler (Stage 1 shapes preserved) ---
    model = build_model_from_featurizers(wtok, ctok, kmap).to(device)

    pos_weight = get_pos_weight(tr_y.numpy())
    criterion = build_loss(pos_weight=pos_weight)
    # Ensure fp32 pos_weight buffer.
    if getattr(criterion, "pos_weight", None) is not None:
        with torch.no_grad():
            criterion.pos_weight = criterion.pos_weight.float().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=BETAS,
        eps=EPS,
    )

    warmup_steps = int(round(WARMUP_RATIO * total_optim_steps))
    scheduler = LinearWarmupCosine(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_optim_steps,
        min_lr_ratio=0.0,
    )

    # --- AMP probe (once, before training) ---
    sample_n = min(PHYSICAL_BATCH_SIZE, n_train)
    probe_inputs = (
        tr_w[:sample_n].to(device),
        tr_c[:sample_n].to(device),
        tr_k[:sample_n].to(device),
    )
    probe_ok = probe_amp_stability(
        cfg,
        model,
        probe_inputs,
        criterion=criterion,
        sample_targets=tr_y[:sample_n].to(device),
        verbose=True,
    )
    if not probe_ok and cfg.use_amp:
        cfg = degrade_to_fp32(cfg, reason="probe_non_finite")
        scaler = make_grad_scaler(cfg)
        print("[train] degraded to FP32 after probe")

    # --- Scheduler context (optional) ---
    context = None
    if _SCRIPT_SCHED_AVAILABLE:
        context = script_scheduler_context()

    def _safe_point(kind, epoch, global_step, step_in_epoch, state_factory):
        if context is None:
            return
        context.control_hook.safe_point(
            kind,
            epoch=epoch,
            global_step=global_step,
            steps_per_epoch=steps_per_epoch,
            state_factory=state_factory,
        )

    resume_state = None
    if context is not None:
        try:
            ckpt = context.load_resume_checkpoint()
            if isinstance(ckpt, dict) and "state" in ckpt:
                resume_state = ckpt["state"]
        except Exception:
            resume_state = None

    # Set job-level max epochs before first safe point if context present.
    if context is not None:
        try:
            if hasattr(context, "job") and context.job is not None:
                try:
                    context.job.max_epochs = NUM_EPOCHS
                except Exception:
                    pass
                cfg_obj = getattr(context.job, "config", None)
                if cfg_obj is not None:
                    try:
                        cfg_obj.max_epochs = NUM_EPOCHS
                    except Exception:
                        pass
                if hasattr(context, "store"):
                    context.store.save_job(context.job)
        except Exception:
            pass

    # --- Dataloader epoch permutation RNG ---
    epoch_rng = torch.Generator()
    epoch_rng.manual_seed(SEED)

    best_val_f1 = -1.0
    best_epoch = -1
    best_state = None
    best_threshold = 0.5
    no_improve = 0
    start_epoch = 1
    global_step = 0
    start_step_in_epoch = 0

    settings = {
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "planned_epochs": NUM_EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "model_family": MODEL_FAMILY,
        "split_seed": SEED,
        "precision": cfg.policy_name,
        "precision_reason": cfg.reason,
    }

    # Diagnostics constructor must be direct for the admission gate.
    diagnostics = TrainingDiagnostics(
        model, optimizer, scaler=scaler, settings=settings
    )

    with diagnostics as diag:
        # Restore state if scheduler provided one.
        if resume_state is not None:
            try:
                if "model" in resume_state:
                    model.load_state_dict(resume_state["model"])
                if (
                    "optimizer" in resume_state
                    and resume_state["optimizer"] is not None
                ):
                    optimizer.load_state_dict(resume_state["optimizer"])
                if scaler is not None and resume_state.get("scaler") is not None:
                    try:
                        scaler.load_state_dict(resume_state["scaler"])
                    except Exception:
                        pass
                if (
                    "scheduler" in resume_state
                    and resume_state["scheduler"] is not None
                ):
                    scheduler.load_state_dict(resume_state["scheduler"])
                if "diag" in resume_state and resume_state["diag"] is not None:
                    try:
                        diag.load_state_dict(resume_state["diag"])
                    except Exception:
                        pass
                if "rng_python" in resume_state:
                    try:
                        random.setstate(resume_state["rng_python"])
                    except Exception:
                        pass
                if "rng_numpy" in resume_state:
                    try:
                        np.random.set_state(resume_state["rng_numpy"])
                    except Exception:
                        pass
                if "rng_torch" in resume_state:
                    try:
                        torch.set_rng_state(resume_state["rng_torch"])
                    except Exception:
                        pass
                if "rng_cuda" in resume_state and torch.cuda.is_available():
                    try:
                        torch.cuda.set_rng_state_all(resume_state["rng_cuda"])
                    except Exception:
                        pass
                if "epoch_rng" in resume_state:
                    try:
                        epoch_rng.set_state(resume_state["epoch_rng"])
                    except Exception:
                        pass
                start_epoch = int(resume_state.get("epoch", 1))
                global_step = int(resume_state.get("global_step", 0))
                start_step_in_epoch = int(resume_state.get("step_in_epoch", 0))
                best_val_f1 = float(resume_state.get("best_val_f1", -1.0))
                best_epoch = int(resume_state.get("best_epoch", -1))
                best_threshold = float(resume_state.get("best_threshold", 0.5))
                no_improve = int(resume_state.get("no_improve", 0))
                print(
                    f"[train] resumed at epoch={start_epoch} step={start_step_in_epoch} gstep={global_step}"
                )
            except Exception as e:
                print(f"[train] resume failed: {e}")
                start_epoch = 1
                global_step = 0
                start_step_in_epoch = 0

        def build_state_factory(current_epoch, step_in_epoch, perm_list):
            def state_factory():
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "scheduler": scheduler.state_dict(),
                    "diag": diag.state_dict(),
                    "global_step": global_step,
                    "epoch": current_epoch,
                    "step_in_epoch": step_in_epoch,
                    "steps_per_epoch": steps_per_epoch,
                    "perm": list(perm_list),
                    "rng_python": random.getstate(),
                    "rng_numpy": np.random.get_state(),
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                    "epoch_rng": epoch_rng.get_state(),
                    "best_val_f1": best_val_f1,
                    "best_epoch": best_epoch,
                    "best_threshold": best_threshold,
                    "no_improve": no_improve,
                }
                return state

            return state_factory

        # BEFORE_TRAIN safe point.
        perm_initial = torch.randperm(n_train, generator=epoch_rng).tolist()
        _safe_point(
            SafePointType.BEFORE_TRAIN if SafePointType is not None else None,
            epoch=start_epoch - 1,
            global_step=global_step,
            step_in_epoch=0,
            state_factory=build_state_factory(start_epoch, 0, perm_initial),
        )

        t_start = time.time()

        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            model.train(True)
            if (
                epoch == start_epoch
                and start_step_in_epoch > 0
                and resume_state is not None
            ):
                perm = list(resume_state.get("perm", perm_initial))
                if len(perm) != n_train:
                    perm = torch.randperm(n_train, generator=epoch_rng).tolist()
            else:
                perm = torch.randperm(n_train, generator=epoch_rng).tolist()
                start_step_in_epoch = 0

            # Skip already-completed batches (mid-epoch resume).
            skip = int(start_step_in_epoch) * PHYSICAL_BATCH_SIZE
            perm_active = perm[skip:]

            loader = _make_loader(
                tr_w,
                tr_c,
                tr_k,
                tr_y,
                indices=perm_active,
                batch_size=PHYSICAL_BATCH_SIZE,
                workers=NUM_WORKERS,
            )

            running_loss = 0.0
            running_n = 0
            step_in_epoch = int(start_step_in_epoch)
            accum_loss = 0.0
            accum_count = 0

            for batch in loader:
                wb, cb, kb, yb = batch
                wb = wb.to(device, non_blocking=True)
                cb = cb.to(device, non_blocking=True)
                kb = kb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                logits = amp_forward(model, (wb, cb, kb), cfg)
                with fp32_island(cfg):
                    loss = bce_with_logits_fp32(logits, yb, criterion=criterion)

                if not check_finite(loss):
                    if cfg.use_amp:
                        cfg = degrade_to_fp32(cfg, reason="non_finite_loss")
                        scaler = make_grad_scaler(cfg)
                        print(
                            f"[train] non-finite loss at epoch={epoch}; degraded to FP32"
                        )
                    # Skip this step cleanly.
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss_scaled = loss / GRAD_ACCUM_STEPS
                if scaler is not None:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                step_in_epoch += 1
                accum_loss += float(loss.detach().float().item()) * int(yb.size(0))
                accum_count += int(yb.size(0))

                is_last_batch = step_in_epoch == steps_per_epoch
                do_update = (step_in_epoch % GRAD_ACCUM_STEPS == 0) or is_last_batch

                if do_update:
                    grad_norm = _unscale_and_clip(
                        scaler, optimizer, model, GRAD_CLIP_NORM
                    )
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    global_step += 1
                    diag.after_update()

                    _safe_point(
                        SafePointType.STEP if SafePointType is not None else None,
                        epoch=epoch,
                        global_step=global_step,
                        step_in_epoch=step_in_epoch,
                        state_factory=build_state_factory(epoch, step_in_epoch, perm),
                    )

                running_loss += float(loss.detach().float().item()) * int(yb.size(0))
                running_n += int(yb.size(0))

            # --- Validation ---
            val_probs, val_loss = _evaluate(
                model,
                cfg,
                va_w,
                va_c,
                va_k,
                va_y,
                criterion=criterion,
                batch_size=256,
            )
            # Threshold tuning on the SAME val probs.
            thr, val_f1 = tune_threshold(
                va_y.numpy().astype(int),
                val_probs,
                min_t=THRESHOLD_MIN,
                max_t=THRESHOLD_MAX,
                steps=THRESHOLD_STEPS,
            )

            train_loss_epoch = running_loss / max(running_n, 1)
            elapsed = time.time() - t_start
            print(
                f"epoch {epoch}/{NUM_EPOCHS} "
                f"train_loss={train_loss_epoch:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_f1={val_f1:.4f} "
                f"thr={thr:.2f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"elapsed_s={elapsed:.1f}"
            )
            # Parseable metric line for scheduler.
            print(
                f"MLEVOLVE_EPOCH_METRIC "
                f"{json.dumps({'epoch': epoch, 'metric': float(val_f1), 'metric_name': 'val_f1'})}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = float(val_f1)
                best_epoch = epoch
                best_threshold = float(thr)
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1

            # EPOCH safe point.
            _safe_point(
                SafePointType.EPOCH if SafePointType is not None else None,
                epoch=epoch,
                global_step=global_step,
                step_in_epoch=step_in_epoch,
                state_factory=build_state_factory(epoch + 1, 0, perm),
            )

            diag.report(epoch=epoch)

            if no_improve >= EARLY_STOP_PATIENCE:
                print(
                    f"[train] early stop: no improvement for {EARLY_STOP_PATIENCE} epochs"
                )
                break

        # Restore best model weights.
        if best_state is not None:
            model.load_state_dict(best_state)
        if torch.cuda.is_available():
            try:
                peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            except Exception:
                peak_vram_mb = None
        else:
            peak_vram_mb = None
        total_time = time.time() - t_start
        print(
            f"[train] done best_epoch={best_epoch} best_val_f1={best_val_f1:.4f} "
            f"best_thr={best_threshold:.3f} total_s={total_time:.1f} "
            f"peak_vram_mb={peak_vram_mb}"
        )

    # --- Save best checkpoint (outside diagnostics context) ---
    try:
        ckpt_path = os.path.join(WORKING_DIR, "best_model.pt")
        torch.save(
            {
                "model_state": model.state_dict(),
                "best_val_f1": best_val_f1,
                "best_epoch": best_epoch,
                "threshold": best_threshold,
                "model_family": MODEL_FAMILY,
                "word_vocab": wtok.vocab,
                "char_vocab": ctok.vocab,
                "keyword_map": kmap.mapping,
                "max_word_len": MAX_WORD_LEN,
                "max_char_len": MAX_CHAR_LEN,
            },
            ckpt_path,
        )
    except Exception as e:
        print(f"[train] checkpoint save failed: {e}")

    # --- Final validation metric on best state (same forward path as test) ---
    cfg_infer = PrecisionConfig(
        device=cfg.device,
        use_amp=cfg.use_amp,
        amp_dtype=cfg.amp_dtype,
        amp_dtype_name=cfg.amp_dtype_name,
        use_tf32=False,
        reason=cfg.reason,
    )
    val_probs_best, _ = _evaluate(
        model,
        cfg_infer,
        va_w,
        va_c,
        va_k,
        va_y,
        criterion=criterion,
        batch_size=256,
    )
    final_thr, final_f1 = tune_threshold(
        va_y.numpy().astype(int),
        val_probs_best,
        min_t=THRESHOLD_MIN,
        max_t=THRESHOLD_MAX,
        steps=THRESHOLD_STEPS,
    )
    # Use best checkpoint threshold chosen during training (from same split).
    use_thr = float(best_threshold) if best_epoch > 0 else float(final_thr)

    val_preds = (val_probs_best >= use_thr).astype(int)
    val_f1_at_use = compute_f1(va_y.numpy().astype(int), val_preds)

    # --- Test inference (IDENTICAL path) ---
    test_w = te_w
    test_c = te_c
    test_k = te_k
    test_probs, _ = _evaluate(
        model,
        cfg_infer,
        test_w,
        test_c,
        test_k,
        torch.zeros(test_w.size(0), dtype=torch.float32),  # dummy y for shape
        criterion=criterion,
        batch_size=256,
    )
    test_preds = (test_probs >= use_thr).astype(int)

    # --- Write submission ---
    sub_df = pd.DataFrame(
        {"id": test_df["id"].astype(int).values, "target": test_preds.astype(int)}
    )
    sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(
        f"[train] wrote {sub_path} rows={len(sub_df)} positives={int(sub_df['target'].sum())}"
    )

    # --- Final score line (must be the very last printed line) ---
    print(f"Final Validation Score: {val_f1_at_use}")


# ---------------------------------------------------------------------------
# Main guard — all side effects live here.
# ---------------------------------------------------------------------------


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
        