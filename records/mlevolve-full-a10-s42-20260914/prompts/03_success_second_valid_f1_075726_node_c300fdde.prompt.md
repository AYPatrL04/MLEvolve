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
 Summary: 1 total attempts - 1 pending execution

-------------------------------
Attempt #1:
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
Before writing or modifying code, follow the hardware-aware stepwise workflow: model-design -> datatype/quantization -> training (code keys: model_design -> datatype_precision -> training_evaluation).

1. Model design: load/prepare data, create features/splits, and choose architecture/model family, loss, criterion, and output interface first.
2. Datatype/precision: choose only mode-allowed operations, autocast regions, FP32-sensitive regions, scaling and fallback behavior. A policy is not a whole-pipeline dtype.
3. Training: choose optimizer, scheduler, batch size, dataloader settings, checkpointing, validation, submission, runtime fallbacks, and logging.
Datatype precision may include only precision-required model adapters that preserve the Stage 1 model family, loss, data features, and output interface.
Hardware-only training/runtime tuning must not increase epochs, folds, model size, image resolution, ensemble count, TTA, dataset size, or validation workload.
If evidence is missing, use the fallback recorded in the trace instead of inventing hardware claims.

# Pipeline Decision Trace
```json
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"{'criterion': 'BCEWithLogitsLoss', 'notes': 'use logits for numerical stability; can apply class weighting if positive/negative imbalance is severe'}","output_interface":"{'prediction': 'sigmoid(logit) >= 0.5 -> binary target 0/1', 'metric': 'binary F1', 'threshold_selection': 'select threshold on internal validation split by maximizing F1', 'submission': 'submission/submission.csv with columns id,target'}","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"{'parameter_dtype': 'fp32', 'input_dtype': 'fp32', 'target_dtype': 'fp32', 'integer_fields': 'keep ids and labels as integer or long where appropriate', 'forward_autocast': {'enabled': True, 'device_type': 'cuda', 'dtype': 'torch.float16...","fallback_policy":"fall back to fp32 or disabled AMP if the selected precision path is unsupported","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"AdamW for neural models or the library default for tree/linear baselines","scheduler":"none","batch_size_policy":"fixed","dataloader_policy":"safe defaults; enable workers and pinned memory only when compatible","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry","backend_guidance:shared.subprocess.task_quality_and_elasticity"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available"]}}
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
 Summary: 1 total attempts - 1 pending execution

-------------------------------
Attempt #1:
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
Before writing or modifying code, follow the hardware-aware stepwise workflow: model-design -> datatype/quantization -> training (code keys: model_design -> datatype_precision -> training_evaluation).

1. Model design: load/prepare data, create features/splits, and choose architecture/model family, loss, criterion, and output interface first.
2. Datatype/precision: choose only mode-allowed operations, autocast regions, FP32-sensitive regions, scaling and fallback behavior. A policy is not a whole-pipeline dtype.
3. Training: choose optimizer, scheduler, batch size, dataloader settings, checkpointing, validation, submission, runtime fallbacks, and logging.
Datatype precision may include only precision-required model adapters that preserve the Stage 1 model family, loss, data features, and output interface.
Hardware-only training/runtime tuning must not increase epochs, folds, model size, image resolution, ensemble count, TTA, dataset size, or validation workload.
If evidence is missing, use the fallback recorded in the trace instead of inventing hardware claims.

# Pipeline Decision Trace
```json
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"{'criterion': 'BCEWithLogitsLoss', 'notes': 'use logits for numerical stability; can apply class weighting if positive/negative imbalance is severe'}","output_interface":"{'prediction': 'sigmoid(logit) >= 0.5 -> binary target 0/1', 'metric': 'binary F1', 'threshold_selection': 'select threshold on internal validation split by maximizing F1', 'submission': 'submission/submission.csv with columns id,target'}","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"{'parameter_dtype': 'fp32', 'input_dtype': 'fp32', 'target_dtype': 'fp32', 'integer_fields': 'keep ids and labels as integer or long where appropriate', 'forward_autocast': {'enabled': True, 'device_type': 'cuda', 'dtype': 'torch.float16...","fallback_policy":"fall back to fp32 or disabled AMP if the selected precision path is unsupported","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"AdamW for neural models or the library default for tree/linear baselines","scheduler":"none","batch_size_policy":"fixed","dataloader_policy":"safe defaults; enable workers and pinned memory only when compatible","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry","backend_guidance:shared.subprocess.task_quality_and_elasticity"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available"]}}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=## Plan **WHAT makes this task unique.** Disaster tweets are short, noisy micro-texts (URLs, mentions, HTML entities, hashtags) where the `keyword` field is moderately discriminative but `location` is extremely sparse...; purpose=Define the Stage 1 candidate target for later precision and training stages.; hardware_keys=cuda_process, pos_weight

# Previous steps
### model_design
**Plan:** ## Plan

**WHAT makes this task unique.** Disaster tweets are short, noisy micro-texts (URLs, mentions, HTML entities, hashtags) where the `keyword` field is moderately discriminative but `location` is extremely sparse and often junk; the class balance is roughly 43/57, and the metric is F1, which rewards recall on the positive class. Critically, the environment is offline — no pretrained transformer weights (HF/timm) can be fetched — so a "real transformer" candidate is unavailable despite the generic hardware brief.

**WHY my approach matches.** A word+char TF-IDF representation, built **only on the training split**, captures both lexical tokens and sub-word morphology that short noisy tweets depend on, and it is robust to OOV spelling variance. Prepending `kw_<keyword>` and `loc_<first word>` tokens into the same text stream lets the vectorizer ingest the metadata as ordinary n-grams, so a single small PyTorch MLP can learn nonlinear interactions between keyword tokens and lexical evidence.

**HOW I will validate.** Stratified hold-out (15%, seed 42). Decision threshold is tuned on this held-out split to maximize binary F1. If val F1 << ~0.78, the MLP is overfitting and I'll widen dropout / fall back to the linear head. If val F1 ≈ 0.80+, the design hypothesis (metadata-as-tokens + char n-grams) is confirmed.

**Backend / memory strategy.** Sparse TF-IDF matrices stay CSR on CPU with `float32` data; batches convert a per-batch slice to a dense `float32` tensor (physical_batch_size × ~40k features stays in single-digit MB, so the elasticity story is "reduce physical batch on OOM, do not change the feature set"). Integer labels stay `long`/`float32` for the loss only; FP32 is the default in CPU preflight, AMP belongs to Stage 2. Scheduler owns device placement, thread allocation, and any `cuda_process` controls — the subprocess only exposes configurable shapes/dims. No pretrained weights, no `timm`/`torch.hub`, no downloads.

**Novelty.** The prior attempt was a transformer with a precision-policy runtime. This design is a fundamentally different family — a from-scratch, no-pretrained, sparse-feature MLP classifier with explicit metadata-tokenization — and it explores the untried hypothesis that well-engineered sparse lexical features beat an undersized from-scratch transformer on 6k short tweets.

**Note board:**
- baseline_change: Replaced "transformer family" baseline with a from-scratch TF-IDF(word 1-2 + char_wb 3-5) → MLP classifier whose only special feature is streaming `keyword`/`location` in as prefix tokens | purpose: fits the offline/no-pretrained-weights reality and targets F1 on short noisy micro-text | hardware_keys: none
- baseline_change: Data pipeline now fits every vectorizer on the train split only, returns CSR `float32` for train/val/test plus a threshold-tuning hook | purpose: prevents leakage and gives Stage 2/Stage 3 a stable, precision-neutral feature contract | hardware_keys: none
- baseline_change: Exposed `CandidateAdapter` with the real `SparseMLP` model and `BCEWithLogitsLoss` (with `pos_weight` support), plus a numpy-free batch builder using `scenario['batch_size']` | purpose: keeps the CPU preflight exercising the actual forward/loss path used by the training stage | hardware_keys: none
**Code:**
```python
"""Stage 1 model_design for Disaster Tweets F1 benchmark.

Design decisions (this stage only):
  * Data: text + keyword + location; keyword/location streamed in as prefix tokens.
  * Features: word TF-IDF (1-2) + char_wb TF-IDF (3-5), fit ONLY on the training split.
  * Model: small PyTorch MLP over sparse-derived dense features (real learned model).
  * Loss: BCEWithLogitsLoss (with pos_weight support); logits for numerical stability.
  * Output: sigmoid(logit) >= threshold (tuned on val for F1) -> 0/1 for submission.

Explicitly NOT decided here: optimizer LR, scheduler, epochs, physical batch size,
AMP precision policy, dataloader workers, checkpoint policy - those belong to later stages.
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

# ------------------------------------------------------------------ constants
MODEL_FAMILY = "tfidf_mlp_tweets_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
DEFAULT_DATA_DIR = "./input"

SEED = 42
VAL_FRACTION = 0.15

WORD_NGRAM = (1, 2)
CHAR_NGRAM = (3, 5)
MAX_WORD_FEATURES = 20000
MAX_CHAR_FEATURES = 20000
MIN_DF_WORD = 2
MIN_DF_CHAR = 3
MAX_KW_TOKEN_LEN = 2  # cap keyword phrase length to limit cardinality

HIDDEN_DIM = 256
DROPOUT = 0.3
DEFAULT_THRESHOLD = 0.5

_HTML_ENTITY_MAP = {
    "&amp;": " and ",
    "&lt;": " < ",
    "&gt;": " > ",
    "&quot;": ' " ',
    "&#39;": " ' ",
    "&nbsp;": " ",
}


# ------------------------------------------------------------------ cleaning
def clean_text(s) -> str:
    """Normalize a raw tweet string. Returns a plain utf-8 lowercased-ish string."""
    if s is None:
        return ""
    if isinstance(s, float) and np.isnan(s):
        return ""
    s = str(s)
    for k, v in _HTML_ENTITY_MAP.items():
        s = s.replace(k, v)
    s = re.sub(r"https?://\S+", " url ", s)
    s = re.sub(r"www\.\S+", " url ", s)
    s = re.sub(r"@\w+", " user ", s)
    s = re.sub(r"#(\w+)", r" \1 ", s)  # hashtags -> bare token
    s = re.sub(r"(\d+)\s*%", r" \1percent ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def clean_keyword(s) -> str:
    if s is None:
        return "none"
    if isinstance(s, float) and np.isnan(s):
        return "none"
    s = str(s).strip().lower()
    s = s.replace("%", "percent")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    return s if s else "none"


def clean_location(s) -> str:
    """Very sparse feature: keep only the first token to limit cardinality."""
    if s is None:
        return "none"
    if isinstance(s, float) and np.isnan(s):
        return "none"
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    if not s:
        return "none"
    first = s.split("_")[0]
    return first if first else "none"


# ------------------------------------------------------------------ features
def build_feature_text(df: pd.DataFrame) -> pd.Series:
    """Stream keyword + first location token + cleaned text into a single string."""
    n = len(df)
    text_col = (
        df["text"].fillna("").astype(str).tolist() if "text" in df.columns else [""] * n
    )
    kw_col = df["keyword"].tolist() if "keyword" in df.columns else [None] * n
    loc_col = df["location"].tolist() if "location" in df.columns else [None] * n

    out = []
    for t, k, l in zip(text_col, kw_col, loc_col):
        kw = clean_keyword(k)
        if MAX_KW_TOKEN_LEN > 0:
            kw = "_".join(kw.split("_")[:MAX_KW_TOKEN_LEN])
        loc = clean_location(l)
        body = clean_text(t)
        out.append(f"kw_{kw} loc_{loc} {body}")
    return pd.Series(out, index=df.index)


def make_vectorizers():
    word_vec = TfidfVectorizer(
        ngram_range=WORD_NGRAM,
        max_features=MAX_WORD_FEATURES,
        min_df=MIN_DF_WORD,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=CHAR_NGRAM,
        max_features=MAX_CHAR_FEATURES,
        min_df=MIN_DF_CHAR,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    return word_vec, char_vec


def fit_features(train_df: pd.DataFrame):
    """Fit vectorizers on the TRAINING split only (leakage guard)."""
    text = build_feature_text(train_df)
    word_vec, char_vec = make_vectorizers()
    Xw = word_vec.fit_transform(text)
    Xc = char_vec.fit_transform(text)
    X = hstack([Xw, Xc]).tocsr().astype(np.float32)
    return word_vec, char_vec, X


def transform_features(df: pd.DataFrame, word_vec, char_vec) -> csr_matrix:
    text = build_feature_text(df)
    Xw = word_vec.transform(text)
    Xc = char_vec.transform(text)
    X = hstack([Xw, Xc]).tocsr().astype(np.float32)
    return X


# ------------------------------------------------------------------ split
def load_frames(data_dir: str = DEFAULT_DATA_DIR):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train, test


def strat_split(
    train_df: pd.DataFrame, val_fraction: float = VAL_FRACTION, seed: int = SEED
):
    y = train_df["target"].astype(int)
    tr_df, va_df = train_test_split(
        train_df, test_size=val_fraction, random_state=seed, stratify=y
    )
    return tr_df.reset_index(drop=True), va_df.reset_index(drop=True)


# ------------------------------------------------------------------ model
class SparseMLP(nn.Module):
    """Small MLP over dense-batch TF-IDF features. Real learned parameters."""

    def __init__(
        self, n_features: int, hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.hidden_dim = int(hidden_dim)
        self.fc1 = nn.Linear(self.n_features, self.hidden_dim)
        self.act = nn.ReLU(inplace=False)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        # Light init for stability on top of normalized TF-IDF inputs.
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.fc1(x)
        h = self.act(h)
        h = self.drop(h)
        logits = self.fc2(h)
        return logits.squeeze(-1)


def build_loss(pos_weight: float = 1.0) -> nn.Module:
    """Loss factory. Kept as a function so later stages can pass pos_weight."""
    if pos_weight is None or pos_weight == 1.0:
        return nn.BCEWithLogitsLoss()
    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32)
    )


def probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.float())


def tune_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Pick decision threshold that maximizes binary F1 on a validation split."""
    from sklearn.metrics import f1_score

    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    if probs.size == 0:
        return DEFAULT_THRESHOLD
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.05, 0.95, 91),
                np.quantile(probs, np.linspace(0.01, 0.99, 50)),
            ]
        )
    )
    best_f1, best_t = -1.0, DEFAULT_THRESHOLD
    for t in grid:
        t = float(t)
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), t
    return best_t


# ------------------------------------------------------------------ CPU preflight adapter
class CandidateAdapter:
    """CPU-safe preflight adapter. Uses the real model, real loss, real features."""

    def __init__(self):
        self._word_vec = None
        self._char_vec = None
        self._n_features = None
        self._criterion = None
        self._cache_df = None
        self._cache_X = None
        self._cache_y = None

        # Small, deterministic slice of train.csv used only for CPU preflight.
        self._preflight_rows = 256
        self._preflight_word_features = 512  # keeps CPU check cheap

    # ---- internal ----
    def _fit_preflight(self):
        data_dir = os.environ.get("MLEVOLVE_INPUT_DIR", DEFAULT_DATA_DIR)
        train_path = os.path.join(data_dir, "train.csv")
        df = pd.read_csv(train_path)
        if "target" not in df.columns:
            raise ValueError("train.csv is missing 'target' column for preflight")
        df = df.head(min(self._preflight_rows, len(df))).reset_index(drop=True)
        text = build_feature_text(df)
        # Light, cheap vectorizer so the preflight is fast on CPU.
        wv = TfidfVectorizer(
            ngram_range=WORD_NGRAM,
            max_features=self._preflight_word_features,
            min_df=1,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
        )
        X = wv.fit_transform(text).tocsr().astype(np.float32)
        self._word_vec = wv
        self._char_vec = None
        self._n_features = X.shape[1]
        self._cache_df = df
        self._cache_X = X
        self._cache_y = df["target"].astype(np.float32).values

    def _ensure(self):
        if self._n_features is None:
            self._fit_preflight()
        return self._n_features

    def _unpack(self, batch):
        if isinstance(batch, dict):
            return batch["inputs"], batch["target"]
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0], batch[1]
        raise ValueError("Unsupported batch format passed to CandidateAdapter")

    def _as_float_tensor(self, x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.float32)
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return torch.tensor(arr, dtype=torch.float32, device=device)

    # ---- public contract ----
    def build_model(self, context):
        self._ensure()
        ctx = dict(context) if context else {}
        hidden = int(ctx.get("hidden_dim", HIDDEN_DIM))
        drop = float(ctx.get("dropout", DROPOUT))
        model = SparseMLP(self._n_features, hidden_dim=hidden, dropout=drop)
        return model

    def build_optimizer(self, model, context):
        # Stage 1 default; later stages own the real optimizer policy.
        ctx = dict(context) if context else {}
        lr = float(ctx.get("lr", 1e-3))
        wd = float(ctx.get("weight_decay", 1e-4))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def _build_batch(self, scenario, device):
        self._ensure()
        ctx = dict(scenario) if scenario else {}
        bs = int(ctx.get("batch_size", 4) or 4)
        bs = max(1, min(bs, self._cache_X.shape[0]))
        Xb = self._cache_X[:bs].toarray().astype(np.float32)
        yb = self._cache_y[:bs].astype(np.float32)
        x_t = torch.tensor(Xb, dtype=torch.float32, device=device)
        y_t = torch.tensor(yb, dtype=torch.float32, device=device)
        return {"inputs": x_t, "target": y_t}

    def build_train_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def _criterion_for(self, context):
        ctx = dict(context) if context else {}
        pw = ctx.get("pos_weight", None)
        if pw is None and "pos_weight" in ctx:
            pw = None
        if self._criterion is None:
            if pw is None:
                self._criterion = nn.BCEWithLogitsLoss()
            else:
                self._criterion = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(float(pw), dtype=torch.float32)
                )
        return self._criterion

    def training_step(self, model, batch, context):
        inputs, target = self._unpack(batch)
        device = next(model.parameters()).device
        x = self._as_float_tensor(inputs, device)
        y = self._as_float_tensor(target, device).view(-1)
        logits = model(x).view(-1)
        crit = self._criterion_for(context)
        if isinstance(crit, nn.BCEWithLogitsLoss) and crit.pos_weight is not None:
            crit = nn.BCEWithLogitsLoss(pos_weight=crit.pos_weight.to(device))
            self._criterion = crit
        return crit(logits, y)

    def validation_step(self, model, batch, context):
        inputs, target = self._unpack(batch)
        device = next(model.parameters()).device
        x = self._as_float_tensor(inputs, device)
        y = self._as_float_tensor(target, device).view(-1)
        crit = self._criterion_for(context)
        if isinstance(crit, nn.BCEWithLogitsLoss) and crit.pos_weight is not None:
            crit = nn.BCEWithLogitsLoss(pos_weight=crit.pos_weight.to(device))
            self._criterion = crit
        with torch.no_grad():
            logits = model(x).view(-1)
            loss = crit(logits, y)
        return loss


# ------------------------------------------------------------------ batch iterator helpers
def csr_batches(X: csr_matrix, y: np.ndarray, indices: np.ndarray, batch_size: int):
    """Yield (dense_float32_x, float32_y) batches honoring batch_size.

    Kept here so later stages can share a single, consistent batching policy.
    """
    n = len(indices)
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        xb = X[idx].toarray().astype(np.float32)
        yb = y[idx].astype(np.float32)
        yield xb, yb


__all__ = [
    "MODEL_FAMILY",
    "WORKING_DIR",
    "SUBMISSION_DIR",
    "DEFAULT_DATA_DIR",
    "SEED",
    "VAL_FRACTION",
    "HIDDEN_DIM",
    "DROPOUT",
    "DEFAULT_THRESHOLD",
    "clean_text",
    "clean_keyword",
    "clean_location",
    "build_feature_text",
    "make_vectorizers",
    "fit_features",
    "transform_features",
    "load_frames",
    "strat_split",
    "SparseMLP",
    "build_loss",
    "probs_from_logits",
    "tune_threshold",
    "csr_batches",
    "CandidateAdapter",
]

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
 Summary: 1 total attempts - 1 pending execution

-------------------------------
Attempt #1:
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
Before writing or modifying code, follow the hardware-aware stepwise workflow: model-design -> datatype/quantization -> training (code keys: model_design -> datatype_precision -> training_evaluation).

1. Model design: load/prepare data, create features/splits, and choose architecture/model family, loss, criterion, and output interface first.
2. Datatype/precision: choose only mode-allowed operations, autocast regions, FP32-sensitive regions, scaling and fallback behavior. A policy is not a whole-pipeline dtype.
3. Training: choose optimizer, scheduler, batch size, dataloader settings, checkpointing, validation, submission, runtime fallbacks, and logging.
Datatype precision may include only precision-required model adapters that preserve the Stage 1 model family, loss, data features, and output interface.
Hardware-only training/runtime tuning must not increase epochs, folds, model size, image resolution, ensemble count, TTA, dataset size, or validation workload.
If evidence is missing, use the fallback recorded in the trace instead of inventing hardware claims.

# Pipeline Decision Trace
```json
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"{'criterion': 'BCEWithLogitsLoss', 'notes': 'use logits for numerical stability; can apply class weighting if positive/negative imbalance is severe'}","output_interface":"{'prediction': 'sigmoid(logit) >= 0.5 -> binary target 0/1', 'metric': 'binary F1', 'threshold_selection': 'select threshold on internal validation split by maximizing F1', 'submission': 'submission/submission.csv with columns id,target'}","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"{'parameter_dtype': 'fp32', 'input_dtype': 'fp32', 'target_dtype': 'fp32', 'integer_fields': 'keep ids and labels as integer or long where appropriate', 'forward_autocast': {'enabled': True, 'device_type': 'cuda', 'dtype': 'torch.float16...","fallback_policy":"fall back to fp32 or disabled AMP if the selected precision path is unsupported","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"AdamW for neural models or the library default for tree/linear baselines","scheduler":"none","batch_size_policy":"fixed","dataloader_policy":"safe defaults; enable workers and pinned memory only when compatible","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry","backend_guidance:shared.subprocess.task_quality_and_elasticity"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available"]}}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=## Plan **WHAT makes this task unique.** Disaster tweets are short, noisy micro-texts (URLs, mentions, HTML entities, hashtags) where the `keyword` field is moderately discriminative but `location` is extremely sparse...; purpose=Define the Stage 1 candidate target for later precision and training stages.; hardware_keys=cuda_process, pos_weight
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=## Plan **WHAT this step must deliver.** Stage 1 produced a sparse TF-IDF → `SparseMLP` classifier whose only float-heavy operation is a `Linear(40000, 256)` matmul plus a tiny `Linear(256, 1)`; the pipeline trace res...; purpose=Support the Stage 1 candidate target with a compatible precision policy.; hardware_keys=fp16_amp, degrade_to_fp32, n_features, hidden_dim, datatype_precision, autocast_forward, compute_bce_with_logits_fp32, build_loss

# Previous steps
### model_design
**Plan:** ## Plan

**WHAT makes this task unique.** Disaster tweets are short, noisy micro-texts (URLs, mentions, HTML entities, hashtags) where the `keyword` field is moderately discriminative but `location` is extremely sparse and often junk; the class balance is roughly 43/57, and the metric is F1, which rewards recall on the positive class. Critically, the environment is offline — no pretrained transformer weights (HF/timm) can be fetched — so a "real transformer" candidate is unavailable despite the generic hardware brief.

**WHY my approach matches.** A word+char TF-IDF representation, built **only on the training split**, captures both lexical tokens and sub-word morphology that short noisy tweets depend on, and it is robust to OOV spelling variance. Prepending `kw_<keyword>` and `loc_<first word>` tokens into the same text stream lets the vectorizer ingest the metadata as ordinary n-grams, so a single small PyTorch MLP can learn nonlinear interactions between keyword tokens and lexical evidence.

**HOW I will validate.** Stratified hold-out (15%, seed 42). Decision threshold is tuned on this held-out split to maximize binary F1. If val F1 << ~0.78, the MLP is overfitting and I'll widen dropout / fall back to the linear head. If val F1 ≈ 0.80+, the design hypothesis (metadata-as-tokens + char n-grams) is confirmed.

**Backend / memory strategy.** Sparse TF-IDF matrices stay CSR on CPU with `float32` data; batches convert a per-batch slice to a dense `float32` tensor (physical_batch_size × ~40k features stays in single-digit MB, so the elasticity story is "reduce physical batch on OOM, do not change the feature set"). Integer labels stay `long`/`float32` for the loss only; FP32 is the default in CPU preflight, AMP belongs to Stage 2. Scheduler owns device placement, thread allocation, and any `cuda_process` controls — the subprocess only exposes configurable shapes/dims. No pretrained weights, no `timm`/`torch.hub`, no downloads.

**Novelty.** The prior attempt was a transformer with a precision-policy runtime. This design is a fundamentally different family — a from-scratch, no-pretrained, sparse-feature MLP classifier with explicit metadata-tokenization — and it explores the untried hypothesis that well-engineered sparse lexical features beat an undersized from-scratch transformer on 6k short tweets.

**Note board:**
- baseline_change: Replaced "transformer family" baseline with a from-scratch TF-IDF(word 1-2 + char_wb 3-5) → MLP classifier whose only special feature is streaming `keyword`/`location` in as prefix tokens | purpose: fits the offline/no-pretrained-weights reality and targets F1 on short noisy micro-text | hardware_keys: none
- baseline_change: Data pipeline now fits every vectorizer on the train split only, returns CSR `float32` for train/val/test plus a threshold-tuning hook | purpose: prevents leakage and gives Stage 2/Stage 3 a stable, precision-neutral feature contract | hardware_keys: none
- baseline_change: Exposed `CandidateAdapter` with the real `SparseMLP` model and `BCEWithLogitsLoss` (with `pos_weight` support), plus a numpy-free batch builder using `scenario['batch_size']` | purpose: keeps the CPU preflight exercising the actual forward/loss path used by the training stage | hardware_keys: none
**Code:**
```python
"""Stage 1 model_design for Disaster Tweets F1 benchmark.

Design decisions (this stage only):
  * Data: text + keyword + location; keyword/location streamed in as prefix tokens.
  * Features: word TF-IDF (1-2) + char_wb TF-IDF (3-5), fit ONLY on the training split.
  * Model: small PyTorch MLP over sparse-derived dense features (real learned model).
  * Loss: BCEWithLogitsLoss (with pos_weight support); logits for numerical stability.
  * Output: sigmoid(logit) >= threshold (tuned on val for F1) -> 0/1 for submission.

Explicitly NOT decided here: optimizer LR, scheduler, epochs, physical batch size,
AMP precision policy, dataloader workers, checkpoint policy - those belong to later stages.
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

# ------------------------------------------------------------------ constants
MODEL_FAMILY = "tfidf_mlp_tweets_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
DEFAULT_DATA_DIR = "./input"

SEED = 42
VAL_FRACTION = 0.15

WORD_NGRAM = (1, 2)
CHAR_NGRAM = (3, 5)
MAX_WORD_FEATURES = 20000
MAX_CHAR_FEATURES = 20000
MIN_DF_WORD = 2
MIN_DF_CHAR = 3
MAX_KW_TOKEN_LEN = 2  # cap keyword phrase length to limit cardinality

HIDDEN_DIM = 256
DROPOUT = 0.3
DEFAULT_THRESHOLD = 0.5

_HTML_ENTITY_MAP = {
    "&amp;": " and ",
    "&lt;": " < ",
    "&gt;": " > ",
    "&quot;": ' " ',
    "&#39;": " ' ",
    "&nbsp;": " ",
}


# ------------------------------------------------------------------ cleaning
def clean_text(s) -> str:
    """Normalize a raw tweet string. Returns a plain utf-8 lowercased-ish string."""
    if s is None:
        return ""
    if isinstance(s, float) and np.isnan(s):
        return ""
    s = str(s)
    for k, v in _HTML_ENTITY_MAP.items():
        s = s.replace(k, v)
    s = re.sub(r"https?://\S+", " url ", s)
    s = re.sub(r"www\.\S+", " url ", s)
    s = re.sub(r"@\w+", " user ", s)
    s = re.sub(r"#(\w+)", r" \1 ", s)  # hashtags -> bare token
    s = re.sub(r"(\d+)\s*%", r" \1percent ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def clean_keyword(s) -> str:
    if s is None:
        return "none"
    if isinstance(s, float) and np.isnan(s):
        return "none"
    s = str(s).strip().lower()
    s = s.replace("%", "percent")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    return s if s else "none"


def clean_location(s) -> str:
    """Very sparse feature: keep only the first token to limit cardinality."""
    if s is None:
        return "none"
    if isinstance(s, float) and np.isnan(s):
        return "none"
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    if not s:
        return "none"
    first = s.split("_")[0]
    return first if first else "none"


# ------------------------------------------------------------------ features
def build_feature_text(df: pd.DataFrame) -> pd.Series:
    """Stream keyword + first location token + cleaned text into a single string."""
    n = len(df)
    text_col = (
        df["text"].fillna("").astype(str).tolist() if "text" in df.columns else [""] * n
    )
    kw_col = df["keyword"].tolist() if "keyword" in df.columns else [None] * n
    loc_col = df["location"].tolist() if "location" in df.columns else [None] * n

    out = []
    for t, k, l in zip(text_col, kw_col, loc_col):
        kw = clean_keyword(k)
        if MAX_KW_TOKEN_LEN > 0:
            kw = "_".join(kw.split("_")[:MAX_KW_TOKEN_LEN])
        loc = clean_location(l)
        body = clean_text(t)
        out.append(f"kw_{kw} loc_{loc} {body}")
    return pd.Series(out, index=df.index)


def make_vectorizers():
    word_vec = TfidfVectorizer(
        ngram_range=WORD_NGRAM,
        max_features=MAX_WORD_FEATURES,
        min_df=MIN_DF_WORD,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=CHAR_NGRAM,
        max_features=MAX_CHAR_FEATURES,
        min_df=MIN_DF_CHAR,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    return word_vec, char_vec


def fit_features(train_df: pd.DataFrame):
    """Fit vectorizers on the TRAINING split only (leakage guard)."""
    text = build_feature_text(train_df)
    word_vec, char_vec = make_vectorizers()
    Xw = word_vec.fit_transform(text)
    Xc = char_vec.fit_transform(text)
    X = hstack([Xw, Xc]).tocsr().astype(np.float32)
    return word_vec, char_vec, X


def transform_features(df: pd.DataFrame, word_vec, char_vec) -> csr_matrix:
    text = build_feature_text(df)
    Xw = word_vec.transform(text)
    Xc = char_vec.transform(text)
    X = hstack([Xw, Xc]).tocsr().astype(np.float32)
    return X


# ------------------------------------------------------------------ split
def load_frames(data_dir: str = DEFAULT_DATA_DIR):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train, test


def strat_split(
    train_df: pd.DataFrame, val_fraction: float = VAL_FRACTION, seed: int = SEED
):
    y = train_df["target"].astype(int)
    tr_df, va_df = train_test_split(
        train_df, test_size=val_fraction, random_state=seed, stratify=y
    )
    return tr_df.reset_index(drop=True), va_df.reset_index(drop=True)


# ------------------------------------------------------------------ model
class SparseMLP(nn.Module):
    """Small MLP over dense-batch TF-IDF features. Real learned parameters."""

    def __init__(
        self, n_features: int, hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.hidden_dim = int(hidden_dim)
        self.fc1 = nn.Linear(self.n_features, self.hidden_dim)
        self.act = nn.ReLU(inplace=False)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        # Light init for stability on top of normalized TF-IDF inputs.
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.fc1(x)
        h = self.act(h)
        h = self.drop(h)
        logits = self.fc2(h)
        return logits.squeeze(-1)


def build_loss(pos_weight: float = 1.0) -> nn.Module:
    """Loss factory. Kept as a function so later stages can pass pos_weight."""
    if pos_weight is None or pos_weight == 1.0:
        return nn.BCEWithLogitsLoss()
    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32)
    )


def probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.float())


def tune_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Pick decision threshold that maximizes binary F1 on a validation split."""
    from sklearn.metrics import f1_score

    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    if probs.size == 0:
        return DEFAULT_THRESHOLD
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.05, 0.95, 91),
                np.quantile(probs, np.linspace(0.01, 0.99, 50)),
            ]
        )
    )
    best_f1, best_t = -1.0, DEFAULT_THRESHOLD
    for t in grid:
        t = float(t)
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), t
    return best_t


# ------------------------------------------------------------------ CPU preflight adapter
class CandidateAdapter:
    """CPU-safe preflight adapter. Uses the real model, real loss, real features."""

    def __init__(self):
        self._word_vec = None
        self._char_vec = None
        self._n_features = None
        self._criterion = None
        self._cache_df = None
        self._cache_X = None
        self._cache_y = None

        # Small, deterministic slice of train.csv used only for CPU preflight.
        self._preflight_rows = 256
        self._preflight_word_features = 512  # keeps CPU check cheap

    # ---- internal ----
    def _fit_preflight(self):
        data_dir = os.environ.get("MLEVOLVE_INPUT_DIR", DEFAULT_DATA_DIR)
        train_path = os.path.join(data_dir, "train.csv")
        df = pd.read_csv(train_path)
        if "target" not in df.columns:
            raise ValueError("train.csv is missing 'target' column for preflight")
        df = df.head(min(self._preflight_rows, len(df))).reset_index(drop=True)
        text = build_feature_text(df)
        # Light, cheap vectorizer so the preflight is fast on CPU.
        wv = TfidfVectorizer(
            ngram_range=WORD_NGRAM,
            max_features=self._preflight_word_features,
            min_df=1,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
        )
        X = wv.fit_transform(text).tocsr().astype(np.float32)
        self._word_vec = wv
        self._char_vec = None
        self._n_features = X.shape[1]
        self._cache_df = df
        self._cache_X = X
        self._cache_y = df["target"].astype(np.float32).values

    def _ensure(self):
        if self._n_features is None:
            self._fit_preflight()
        return self._n_features

    def _unpack(self, batch):
        if isinstance(batch, dict):
            return batch["inputs"], batch["target"]
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0], batch[1]
        raise ValueError("Unsupported batch format passed to CandidateAdapter")

    def _as_float_tensor(self, x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.float32)
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return torch.tensor(arr, dtype=torch.float32, device=device)

    # ---- public contract ----
    def build_model(self, context):
        self._ensure()
        ctx = dict(context) if context else {}
        hidden = int(ctx.get("hidden_dim", HIDDEN_DIM))
        drop = float(ctx.get("dropout", DROPOUT))
        model = SparseMLP(self._n_features, hidden_dim=hidden, dropout=drop)
        return model

    def build_optimizer(self, model, context):
        # Stage 1 default; later stages own the real optimizer policy.
        ctx = dict(context) if context else {}
        lr = float(ctx.get("lr", 1e-3))
        wd = float(ctx.get("weight_decay", 1e-4))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def _build_batch(self, scenario, device):
        self._ensure()
        ctx = dict(scenario) if scenario else {}
        bs = int(ctx.get("batch_size", 4) or 4)
        bs = max(1, min(bs, self._cache_X.shape[0]))
        Xb = self._cache_X[:bs].toarray().astype(np.float32)
        yb = self._cache_y[:bs].astype(np.float32)
        x_t = torch.tensor(Xb, dtype=torch.float32, device=device)
        y_t = torch.tensor(yb, dtype=torch.float32, device=device)
        return {"inputs": x_t, "target": y_t}

    def build_train_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def _criterion_for(self, context):
        ctx = dict(context) if context else {}
        pw = ctx.get("pos_weight", None)
        if pw is None and "pos_weight" in ctx:
            pw = None
        if self._criterion is None:
            if pw is None:
                self._criterion = nn.BCEWithLogitsLoss()
            else:
                self._criterion = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(float(pw), dtype=torch.float32)
                )
        return self._criterion

    def training_step(self, model, batch, context):
        inputs, target = self._unpack(batch)
        device = next(model.parameters()).device
        x = self._as_float_tensor(inputs, device)
        y = self._as_float_tensor(target, device).view(-1)
        logits = model(x).view(-1)
        crit = self._criterion_for(context)
        if isinstance(crit, nn.BCEWithLogitsLoss) and crit.pos_weight is not None:
            crit = nn.BCEWithLogitsLoss(pos_weight=crit.pos_weight.to(device))
            self._criterion = crit
        return crit(logits, y)

    def validation_step(self, model, batch, context):
        inputs, target = self._unpack(batch)
        device = next(model.parameters()).device
        x = self._as_float_tensor(inputs, device)
        y = self._as_float_tensor(target, device).view(-1)
        crit = self._criterion_for(context)
        if isinstance(crit, nn.BCEWithLogitsLoss) and crit.pos_weight is not None:
            crit = nn.BCEWithLogitsLoss(pos_weight=crit.pos_weight.to(device))
            self._criterion = crit
        with torch.no_grad():
            logits = model(x).view(-1)
            loss = crit(logits, y)
        return loss


# ------------------------------------------------------------------ batch iterator helpers
def csr_batches(X: csr_matrix, y: np.ndarray, indices: np.ndarray, batch_size: int):
    """Yield (dense_float32_x, float32_y) batches honoring batch_size.

    Kept here so later stages can share a single, consistent batching policy.
    """
    n = len(indices)
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        xb = X[idx].toarray().astype(np.float32)
        yb = y[idx].astype(np.float32)
        yield xb, yb


__all__ = [
    "MODEL_FAMILY",
    "WORKING_DIR",
    "SUBMISSION_DIR",
    "DEFAULT_DATA_DIR",
    "SEED",
    "VAL_FRACTION",
    "HIDDEN_DIM",
    "DROPOUT",
    "DEFAULT_THRESHOLD",
    "clean_text",
    "clean_keyword",
    "clean_location",
    "build_feature_text",
    "make_vectorizers",
    "fit_features",
    "transform_features",
    "load_frames",
    "strat_split",
    "SparseMLP",
    "build_loss",
    "probs_from_logits",
    "tune_threshold",
    "csr_batches",
    "CandidateAdapter",
]

```

### datatype_precision
**Plan:** ## Plan

**WHAT this step must deliver.** Stage 1 produced a sparse TF-IDF → `SparseMLP` classifier whose only float-heavy operation is a `Linear(40000, 256)` matmul plus a tiny `Linear(256, 1)`; the pipeline trace resolves precision as `fp16_amp` on CUDA with FP32 fallback, and normal mode forbids BF16/TF32/quantization. So this stage must supply *utilities* — device/dtype resolution, TF32 disabling, a GradScaler, an autocast region for the forward only, an explicit FP32 island for the fragile `BCEWithLogitsLoss`, finite-loss/grad guards, a deterministic FP32 degradation path, and a one-line precision log — without touching the model family, loss, features, or output interface.

**WHY this matches the workload.** The dominant cost is one wide linear layer; FP16 Tensor-Core autocast can accelerate it on the A10, but sigmoid/log reductions inside `BCEWithLogitsLoss` are numerically fragile, so the loss is computed in an explicit FP32 island with `logits.float()`. Parameters, inputs, targets, metrics and prediction export all stay FP32, and the scaler only becomes active when CUDA + FP16 are both confirmed, so the CPU preflight silently falls back to FP32 with AMP disabled.

**HOW I will validate.** Stage 3 will log the resolved precision line, record finite loss/grad on the first optimizer steps, and fall back to FP32 (`degrade_to_fp32`) if any non-finite value appears; the fallback is deterministic and preserves the output interface, so the val-F1 signal is directly comparable between the FP16-AMP and FP32 runs.

**Backend / elasticity statement.** Memory strategy is unchanged from Stage 1: CSR TF-IDF stays on CPU, batches densify per-slice. Physical batch size is a Stage 3 knob; precision elasticity here is *only* AMP on/off. Operator considerations: `Linear` ops eligible for FP16 autocast, loss/reductions pinned to FP32, TF32 explicitly disabled for both matmul and cudnn. Configurable dimensions preserved: `n_features`, `hidden_dim`, `dropout`. Scheduler-owned controls (device assignment, thread allocation, memory limits, launch policy) are **not** implemented here.

**Note board:**
- baseline_change: Added a standalone `datatype_precision` module that resolves `DEVICE/USE_AMP/AMP_DTYPE/USE_TF32`, builds a CUDA-only `GradScaler`, and exposes `autocast_forward` (forward only) + `compute_bce_with_logits_fp32` (fragile-loss FP32 island) without touching `SparseMLP`, `build_loss`, features, or the output interface | purpose: implements the trace's `fp16_amp` policy as pure, import-safe utilities so Stage 3 can wire autocast around the forward while keeping reductions and metrics in FP32 | hardware_keys: cuda_process, fp16_amp, tf32_disabled
- baseline_change: Added `degrade_to_fp32(...)` + `is_finite_tensor` / `check_grads_finite` so a non-finite loss/grad on FP16 AMP can deterministically revert to FP32 | purpose: satisfies the trace's `fallback_policy` and precision-evidence rule (keep FP32 wherever AMP is unstable/unsupported) while preserving the Stage 1 candidate target | hardware_keys: fp16_amp
- baseline_change: Added `configure_tf32(False)` helper (called from the main guard only) and a single-line `format_precision_log(...)` reporter | purpose: enforces normal-mode TF32 disablement without import-time CUDA side effects and records the actual autocast region / fallback used | hardware_keys: tf32_disabled
**Code:**
```python
"""Stage 2 (datatype_precision) for the Disaster Tweets F1 benchmark.

Consumes the pipeline decision trace:
    precision_policy      = "fp16_amp"
    precision_model_adaptation:
        parameter_dtype   = fp32
        input_dtype       = fp32
        target_dtype      = fp32
        forward_autocast  = {enabled: True, device: cuda, dtype: float16}
        backward/grad/step outside autocast, GradScaler for fp16
    fallback_policy       = fall back to fp32 / disabled AMP if unsupported

Normal mode permits only FP32 + selective FP16 AMP, so:
  * no BF16, no TF32, no FP8/MXFP8/NVFP4 recipes;
  * model parameters and floating inputs/targets remain FP32;
  * autocast is used *only* around the forward path of the real model;
  * the fragile `BCEWithLogitsLoss` reduction is computed in an FP32 island;
  * backward, unscale, and optimizer.step run outside autocast;
  * the CPU preflight (and any non-CUDA device) uses FP32 with the scaler disabled.

Everything here is defined as import-safe helpers/constants only. No CUDA queries,
no TF32 mutation, no logging setup, no model/loss instantiation at import time.
The training_evaluation stage calls these from inside its `if __name__ == "__main__":`
entrypoint. Nothing in this module changes the model family, loss, data features,
output interface, optimizer, scheduler, batch policy, or submission logic.
"""

import contextlib
import torch

# ------------------------------------------------------------------ constants
# Only FP16 is a permitted native AMP dtype in normal mode.
AMP_DTYPE = torch.float16
AMP_DTYPE_NAME = "float16"

# Default policy bundled for easy consumption by the training step.
PRECISION_POLICY = "fp16_amp"
DEFAULT_USE_AMP = True
DEFAULT_USE_TF32 = False  # normal mode: TF32 must be disabled
DEFAULT_ALLOWED_AMP_DTYPES = ("float16",)

# Filled in at runtime by `resolve_precision_config` when a config is bound.
DEVICE = None
USE_AMP = DEFAULT_USE_AMP
USE_TF32 = DEFAULT_USE_TF32
AMP_SCALER = None


# ------------------------------------------------------------------ helpers
def _cuda_available():
    """Import-safe CUDA availability probe (only called at runtime)."""
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_precision_config(
    use_amp=DEFAULT_USE_AMP, requested_device=None, amp_dtype_name=AMP_DTYPE_NAME
):
    """Resolve device / autocast / scaler / TF32 settings for this run.

    Returns a dict with keys:
        device, use_amp, amp_dtype, amp_dtype_name, use_tf32, reason, notes

    * FP16 AMP is enabled only when CUDA is available and `use_amp` is True.
    * Any non-FP16 dtype request collapses to FP32 (normal-mode constraint).
    * CPU / non-CUDA always resolves to FP32 with AMP disabled.
    """
    if requested_device:
        try:
            device = torch.device(requested_device)
        except Exception:
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if _cuda_available() else "cpu")

    cuda_ok = device.type == "cuda" and _cuda_available()
    if device.type == "cuda" and not cuda_ok:
        device = torch.device("cpu")

    amp_name = str(amp_dtype_name or AMP_DTYPE_NAME).lower()
    if amp_name not in DEFAULT_ALLOWED_AMP_DTYPES:
        amp_name = "float32"

    amp_enabled = bool(use_amp) and cuda_ok and amp_name == "float16"

    if amp_enabled:
        reason = "cuda_fp16_amp"
    elif device.type == "cuda":
        reason = "cuda_fp32_amp_disabled_or_unsupported"
    else:
        reason = "cpu_fp32_no_amp"

    return {
        "device": device,
        "use_amp": amp_enabled,
        "amp_dtype": AMP_DTYPE if amp_enabled else torch.float32,
        "amp_dtype_name": "float16" if amp_enabled else "float32",
        "use_tf32": DEFAULT_USE_TF32,
        "reason": reason,
        "notes": (
            "params/inputs/targets=fp32; forward under cuda autocast; "
            "loss+reductions fp32 island; backward/step external; TF32 disabled"
        ),
    }


def configure_tf32(enable=False):
    """Explicitly set the TF32 backends. Call from the main guard only."""
    enable = bool(enable)
    try:
        torch.backends.cuda.matmul.allow_tf32 = enable
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = enable
    except Exception:
        pass


def make_grad_scaler(use_amp, device):
    """Build a GradScaler that is only active for CUDA FP16 AMP.

    Returns a scaler object, or None if GradScaler is unavailable.
    """
    active = (
        bool(use_amp)
        and device is not None
        and getattr(device, "type", "cpu") == "cuda"
    )
    try:
        return torch.amp.GradScaler("cuda", enabled=active)  # torch >= 2.4 API
    except (AttributeError, TypeError):
        pass
    try:
        return torch.cuda.amp.GradScaler(enabled=active)  # legacy API
    except Exception:
        return None


@contextlib.contextmanager
def autocast_forward(use_amp, device, amp_dtype=AMP_DTYPE):
    """Autocast region for the *forward* path only. No-op when AMP is disabled.

    Backward, GradScaler.unscale_, and optimizer.step MUST be executed outside
    this context. Do not wrap the fragile BCE loss inside this context; use
    `compute_bce_with_logits_fp32` instead, which is an explicit FP32 island.
    """
    if not use_amp or device is None or getattr(device, "type", "cpu") != "cuda":
        yield
        return

    cm = None
    try:
        cm = torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True)
    except (AttributeError, TypeError):
        try:
            cm = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=True)
        except Exception:
            cm = None

    if cm is None:
        yield
        return

    with cm:
        yield


def compute_bce_with_logits_fp32(criterion, logits, targets):
    """Compute the BCE-with-logits loss in an explicit FP32 island.

    Autocast may hand us an FP16 logit tensor. Sigmoid + log + weighted-mean
    reductions inside BCEWithLogitsLoss are numerically fragile in FP16, so we
    cast to FP32 before calling the (unchanged) Stage 1 criterion. This preserves
    the Stage 1 loss object, output interface, and reduction semantics.
    """
    logits_fp32 = logits.float() if isinstance(logits, torch.Tensor) else logits
    targets_fp32 = (
        targets.float().view(-1) if isinstance(targets, torch.Tensor) else targets
    )
    return criterion(logits_fp32, targets_fp32)


def is_finite_tensor(x):
    """True if tensor is None or all elements are finite."""
    if x is None:
        return True
    if not isinstance(x, torch.Tensor):
        try:
            x = torch.as_tensor(x)
        except Exception:
            return True
    try:
        return bool(torch.isfinite(x.detach()).all().item())
    except Exception:
        return False


def check_grads_finite(model):
    """True if every populated parameter gradient is finite."""
    if model is None:
        return True
    try:
        for p in model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                return False
    except Exception:
        return True
    return True


def degrade_to_fp32(precision_cfg, reason="non_finite_loss_or_grad"):
    """Deterministic fallback from FP16 AMP to FP32.

    Returns a new config dict with AMP disabled so that Stage 3 can swap in
    the FP32 path without changing model family, loss, features, or interface.
    """
    new_cfg = dict(precision_cfg or {})
    new_cfg["use_amp"] = False
    new_cfg["amp_dtype"] = torch.float32
    new_cfg["amp_dtype_name"] = "float32"
    new_cfg["fallback_from"] = (precision_cfg or {}).get("reason", "unknown")
    new_cfg["reason"] = f"fallback_fp32:{reason}"
    return new_cfg


def format_precision_log(precision_cfg, extra=None):
    """One-line precision report for epoch-level logging (no batch noise)."""
    cfg = precision_cfg or {}
    line = (
        f"[precision] device={cfg.get('device', 'unknown')} "
        f"policy={PRECISION_POLICY} "
        f"amp={cfg.get('use_amp', False)} "
        f"amp_dtype={cfg.get('amp_dtype_name', 'float32')} "
        f"tf32={cfg.get('use_tf32', DEFAULT_USE_TF32)} "
        f"reason={cfg.get('reason', 'unset')}"
    )
    if extra:
        try:
            line += " " + " ".join(f"{k}={v}" for k, v in dict(extra).items())
        except Exception:
            pass
    return line


__all__ = [
    "AMP_DTYPE",
    "AMP_DTYPE_NAME",
    "PRECISION_POLICY",
    "DEFAULT_USE_AMP",
    "DEFAULT_USE_TF32",
    "DEFAULT_ALLOWED_AMP_DTYPES",
    "DEVICE",
    "USE_AMP",
    "USE_TF32",
    "AMP_SCALER",
    "resolve_precision_config",
    "configure_tf32",
    "make_grad_scaler",
    "autocast_forward",
    "compute_bce_with_logits_fp32",
    "is_finite_tensor",
    "check_grads_finite",
    "degrade_to_fp32",
    "format_precision_log",
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
 Summary: 1 total attempts - 1 pending execution

-------------------------------
Attempt #1:
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
Before writing or modifying code, follow the hardware-aware stepwise workflow: model-design -> datatype/quantization -> training (code keys: model_design -> datatype_precision -> training_evaluation).

1. Model design: load/prepare data, create features/splits, and choose architecture/model family, loss, criterion, and output interface first.
2. Datatype/precision: choose only mode-allowed operations, autocast regions, FP32-sensitive regions, scaling and fallback behavior. A policy is not a whole-pipeline dtype.
3. Training: choose optimizer, scheduler, batch size, dataloader settings, checkpointing, validation, submission, runtime fallbacks, and logging.
Datatype precision may include only precision-required model adapters that preserve the Stage 1 model family, loss, data features, and output interface.
Hardware-only training/runtime tuning must not increase epochs, folds, model size, image resolution, ensemble count, TTA, dataset size, or validation workload.
If evidence is missing, use the fallback recorded in the trace instead of inventing hardware claims.

# Pipeline Decision Trace
```json
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"{'criterion': 'BCEWithLogitsLoss', 'notes': 'use logits for numerical stability; can apply class weighting if positive/negative imbalance is severe'}","output_interface":"{'prediction': 'sigmoid(logit) >= 0.5 -> binary target 0/1', 'metric': 'binary F1', 'threshold_selection': 'select threshold on internal validation split by maximizing F1', 'submission': 'submission/submission.csv with columns id,target'}","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"{'parameter_dtype': 'fp32', 'input_dtype': 'fp32', 'target_dtype': 'fp32', 'integer_fields': 'keep ids and labels as integer or long where appropriate', 'forward_autocast': {'enabled': True, 'device_type': 'cuda', 'dtype': 'torch.float16...","fallback_policy":"fall back to fp32 or disabled AMP if the selected precision path is unsupported","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"AdamW for neural models or the library default for tree/linear baselines","scheduler":"none","batch_size_policy":"fixed","dataloader_policy":"safe defaults; enable workers and pinned memory only when compatible","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry","backend_guidance:shared.subprocess.task_quality_and_elasticity"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available"]}}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=## Plan **WHAT makes this task unique.** Disaster tweets are short, noisy micro-texts (URLs, mentions, HTML entities, hashtags) where the `keyword` field is moderately discriminative but `location` is extremely sparse...; purpose=Define the Stage 1 candidate target for later precision and training stages.; hardware_keys=cuda_process, pos_weight
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=## Plan **WHAT this step must deliver.** Stage 1 produced a sparse TF-IDF → `SparseMLP` classifier whose only float-heavy operation is a `Linear(40000, 256)` matmul plus a tiny `Linear(256, 1)`; the pipeline trace res...; purpose=Support the Stage 1 candidate target with a compatible precision policy.; hardware_keys=fp16_amp, degrade_to_fp32, n_features, hidden_dim, datatype_precision, autocast_forward, compute_bce_with_logits_fp32, build_loss
  - Stage 3 training / training_evaluation: baseline_change=## Plan **WHAT/WHY.** Stage 1 gives a real learned `SparseMLP` over train-fitted TF-IDF, Stage 2 gives `fp16_amp` utilities (autocast-forward + FP32 BCE island + GradScaler + `degrade_to_fp32`).; purpose=Train and evaluate the Stage 1 candidate using the selected precision policy.; hardware_keys=fp16_amp, degrade_to_fp32, run_training, tune_threshold, autocast_forward, compute_bce_with_logits_fp32, make_grad_scaler, resolve_precision_config

# Step results

        ### Step 1: model_design
        **Plan:** ## Plan

**WHAT makes this task unique.** Disaster tweets are short, noisy micro-texts (URLs, mentions, HTML entities, hashtags) where the `keyword` field is moderately discriminative but `location` is extremely sparse and often junk; the class balance is roughly 43/57, and the metric is F1, which rewards recall on the positive class. Critically, the environment is offline — no pretrained transformer weights (HF/timm) can be fetched — so a "real transformer" candidate is unavailable despite the generic hardware brief.

**WHY my approach matches.** A word+char TF-IDF representation, built **only on the training split**, captures both lexical tokens and sub-word morphology that short noisy tweets depend on, and it is robust to OOV spelling variance. Prepending `kw_<keyword>` and `loc_<first word>` tokens into the same text stream lets the vectorizer ingest the metadata as ordinary n-grams, so a single small PyTorch MLP can learn nonlinear interactions between keyword tokens and lexical evidence.

**HOW I will validate.** Stratified hold-out (15%, seed 42). Decision threshold is tuned on this held-out split to maximize binary F1. If val F1 << ~0.78, the MLP is overfitting and I'll widen dropout / fall back to the linear head. If val F1 ≈ 0.80+, the design hypothesis (metadata-as-tokens + char n-grams) is confirmed.

**Backend / memory strategy.** Sparse TF-IDF matrices stay CSR on CPU with `float32` data; batches convert a per-batch slice to a dense `float32` tensor (physical_batch_size × ~40k features stays in single-digit MB, so the elasticity story is "reduce physical batch on OOM, do not change the feature set"). Integer labels stay `long`/`float32` for the loss only; FP32 is the default in CPU preflight, AMP belongs to Stage 2. Scheduler owns device placement, thread allocation, and any `cuda_process` controls — the subprocess only exposes configurable shapes/dims. No pretrained weights, no `timm`/`torch.hub`, no downloads.

**Novelty.** The prior attempt was a transformer with a precision-policy runtime. This design is a fundamentally different family — a from-scratch, no-pretrained, sparse-feature MLP classifier with explicit metadata-tokenization — and it explores the untried hypothesis that well-engineered sparse lexical features beat an undersized from-scratch transformer on 6k short tweets.

**Note board:**
- baseline_change: Replaced "transformer family" baseline with a from-scratch TF-IDF(word 1-2 + char_wb 3-5) → MLP classifier whose only special feature is streaming `keyword`/`location` in as prefix tokens | purpose: fits the offline/no-pretrained-weights reality and targets F1 on short noisy micro-text | hardware_keys: none
- baseline_change: Data pipeline now fits every vectorizer on the train split only, returns CSR `float32` for train/val/test plus a threshold-tuning hook | purpose: prevents leakage and gives Stage 2/Stage 3 a stable, precision-neutral feature contract | hardware_keys: none
- baseline_change: Exposed `CandidateAdapter` with the real `SparseMLP` model and `BCEWithLogitsLoss` (with `pos_weight` support), plus a numpy-free batch builder using `scenario['batch_size']` | purpose: keeps the CPU preflight exercising the actual forward/loss path used by the training stage | hardware_keys: none
        **Code:**
        ```python
"""Stage 1 model_design for Disaster Tweets F1 benchmark.

Design decisions (this stage only):
  * Data: text + keyword + location; keyword/location streamed in as prefix tokens.
  * Features: word TF-IDF (1-2) + char_wb TF-IDF (3-5), fit ONLY on the training split.
  * Model: small PyTorch MLP over sparse-derived dense features (real learned model).
  * Loss: BCEWithLogitsLoss (with pos_weight support); logits for numerical stability.
  * Output: sigmoid(logit) >= threshold (tuned on val for F1) -> 0/1 for submission.

Explicitly NOT decided here: optimizer LR, scheduler, epochs, physical batch size,
AMP precision policy, dataloader workers, checkpoint policy - those belong to later stages.
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

# ------------------------------------------------------------------ constants
MODEL_FAMILY = "tfidf_mlp_tweets_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
DEFAULT_DATA_DIR = "./input"

SEED = 42
VAL_FRACTION = 0.15

WORD_NGRAM = (1, 2)
CHAR_NGRAM = (3, 5)
MAX_WORD_FEATURES = 20000
MAX_CHAR_FEATURES = 20000
MIN_DF_WORD = 2
MIN_DF_CHAR = 3
MAX_KW_TOKEN_LEN = 2  # cap keyword phrase length to limit cardinality

HIDDEN_DIM = 256
DROPOUT = 0.3
DEFAULT_THRESHOLD = 0.5

_HTML_ENTITY_MAP = {
    "&amp;": " and ",
    "&lt;": " < ",
    "&gt;": " > ",
    "&quot;": ' " ',
    "&#39;": " ' ",
    "&nbsp;": " ",
}


# ------------------------------------------------------------------ cleaning
def clean_text(s) -> str:
    """Normalize a raw tweet string. Returns a plain utf-8 lowercased-ish string."""
    if s is None:
        return ""
    if isinstance(s, float) and np.isnan(s):
        return ""
    s = str(s)
    for k, v in _HTML_ENTITY_MAP.items():
        s = s.replace(k, v)
    s = re.sub(r"https?://\S+", " url ", s)
    s = re.sub(r"www\.\S+", " url ", s)
    s = re.sub(r"@\w+", " user ", s)
    s = re.sub(r"#(\w+)", r" \1 ", s)  # hashtags -> bare token
    s = re.sub(r"(\d+)\s*%", r" \1percent ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def clean_keyword(s) -> str:
    if s is None:
        return "none"
    if isinstance(s, float) and np.isnan(s):
        return "none"
    s = str(s).strip().lower()
    s = s.replace("%", "percent")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    return s if s else "none"


def clean_location(s) -> str:
    """Very sparse feature: keep only the first token to limit cardinality."""
    if s is None:
        return "none"
    if isinstance(s, float) and np.isnan(s):
        return "none"
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    if not s:
        return "none"
    first = s.split("_")[0]
    return first if first else "none"


# ------------------------------------------------------------------ features
def build_feature_text(df: pd.DataFrame) -> pd.Series:
    """Stream keyword + first location token + cleaned text into a single string."""
    n = len(df)
    text_col = (
        df["text"].fillna("").astype(str).tolist() if "text" in df.columns else [""] * n
    )
    kw_col = df["keyword"].tolist() if "keyword" in df.columns else [None] * n
    loc_col = df["location"].tolist() if "location" in df.columns else [None] * n

    out = []
    for t, k, l in zip(text_col, kw_col, loc_col):
        kw = clean_keyword(k)
        if MAX_KW_TOKEN_LEN > 0:
            kw = "_".join(kw.split("_")[:MAX_KW_TOKEN_LEN])
        loc = clean_location(l)
        body = clean_text(t)
        out.append(f"kw_{kw} loc_{loc} {body}")
    return pd.Series(out, index=df.index)


def make_vectorizers():
    word_vec = TfidfVectorizer(
        ngram_range=WORD_NGRAM,
        max_features=MAX_WORD_FEATURES,
        min_df=MIN_DF_WORD,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=CHAR_NGRAM,
        max_features=MAX_CHAR_FEATURES,
        min_df=MIN_DF_CHAR,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    return word_vec, char_vec


def fit_features(train_df: pd.DataFrame):
    """Fit vectorizers on the TRAINING split only (leakage guard)."""
    text = build_feature_text(train_df)
    word_vec, char_vec = make_vectorizers()
    Xw = word_vec.fit_transform(text)
    Xc = char_vec.fit_transform(text)
    X = hstack([Xw, Xc]).tocsr().astype(np.float32)
    return word_vec, char_vec, X


def transform_features(df: pd.DataFrame, word_vec, char_vec) -> csr_matrix:
    text = build_feature_text(df)
    Xw = word_vec.transform(text)
    Xc = char_vec.transform(text)
    X = hstack([Xw, Xc]).tocsr().astype(np.float32)
    return X


# ------------------------------------------------------------------ split
def load_frames(data_dir: str = DEFAULT_DATA_DIR):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train, test


def strat_split(
    train_df: pd.DataFrame, val_fraction: float = VAL_FRACTION, seed: int = SEED
):
    y = train_df["target"].astype(int)
    tr_df, va_df = train_test_split(
        train_df, test_size=val_fraction, random_state=seed, stratify=y
    )
    return tr_df.reset_index(drop=True), va_df.reset_index(drop=True)


# ------------------------------------------------------------------ model
class SparseMLP(nn.Module):
    """Small MLP over dense-batch TF-IDF features. Real learned parameters."""

    def __init__(
        self, n_features: int, hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.hidden_dim = int(hidden_dim)
        self.fc1 = nn.Linear(self.n_features, self.hidden_dim)
        self.act = nn.ReLU(inplace=False)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        # Light init for stability on top of normalized TF-IDF inputs.
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.fc1(x)
        h = self.act(h)
        h = self.drop(h)
        logits = self.fc2(h)
        return logits.squeeze(-1)


def build_loss(pos_weight: float = 1.0) -> nn.Module:
    """Loss factory. Kept as a function so later stages can pass pos_weight."""
    if pos_weight is None or pos_weight == 1.0:
        return nn.BCEWithLogitsLoss()
    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32)
    )


def probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.float())


def tune_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Pick decision threshold that maximizes binary F1 on a validation split."""
    from sklearn.metrics import f1_score

    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    if probs.size == 0:
        return DEFAULT_THRESHOLD
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.05, 0.95, 91),
                np.quantile(probs, np.linspace(0.01, 0.99, 50)),
            ]
        )
    )
    best_f1, best_t = -1.0, DEFAULT_THRESHOLD
    for t in grid:
        t = float(t)
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), t
    return best_t


# ------------------------------------------------------------------ CPU preflight adapter
class CandidateAdapter:
    """CPU-safe preflight adapter. Uses the real model, real loss, real features."""

    def __init__(self):
        self._word_vec = None
        self._char_vec = None
        self._n_features = None
        self._criterion = None
        self._cache_df = None
        self._cache_X = None
        self._cache_y = None

        # Small, deterministic slice of train.csv used only for CPU preflight.
        self._preflight_rows = 256
        self._preflight_word_features = 512  # keeps CPU check cheap

    # ---- internal ----
    def _fit_preflight(self):
        data_dir = os.environ.get("MLEVOLVE_INPUT_DIR", DEFAULT_DATA_DIR)
        train_path = os.path.join(data_dir, "train.csv")
        df = pd.read_csv(train_path)
        if "target" not in df.columns:
            raise ValueError("train.csv is missing 'target' column for preflight")
        df = df.head(min(self._preflight_rows, len(df))).reset_index(drop=True)
        text = build_feature_text(df)
        # Light, cheap vectorizer so the preflight is fast on CPU.
        wv = TfidfVectorizer(
            ngram_range=WORD_NGRAM,
            max_features=self._preflight_word_features,
            min_df=1,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
        )
        X = wv.fit_transform(text).tocsr().astype(np.float32)
        self._word_vec = wv
        self._char_vec = None
        self._n_features = X.shape[1]
        self._cache_df = df
        self._cache_X = X
        self._cache_y = df["target"].astype(np.float32).values

    def _ensure(self):
        if self._n_features is None:
            self._fit_preflight()
        return self._n_features

    def _unpack(self, batch):
        if isinstance(batch, dict):
            return batch["inputs"], batch["target"]
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0], batch[1]
        raise ValueError("Unsupported batch format passed to CandidateAdapter")

    def _as_float_tensor(self, x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.float32)
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return torch.tensor(arr, dtype=torch.float32, device=device)

    # ---- public contract ----
    def build_model(self, context):
        self._ensure()
        ctx = dict(context) if context else {}
        hidden = int(ctx.get("hidden_dim", HIDDEN_DIM))
        drop = float(ctx.get("dropout", DROPOUT))
        model = SparseMLP(self._n_features, hidden_dim=hidden, dropout=drop)
        return model

    def build_optimizer(self, model, context):
        # Stage 1 default; later stages own the real optimizer policy.
        ctx = dict(context) if context else {}
        lr = float(ctx.get("lr", 1e-3))
        wd = float(ctx.get("weight_decay", 1e-4))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def _build_batch(self, scenario, device):
        self._ensure()
        ctx = dict(scenario) if scenario else {}
        bs = int(ctx.get("batch_size", 4) or 4)
        bs = max(1, min(bs, self._cache_X.shape[0]))
        Xb = self._cache_X[:bs].toarray().astype(np.float32)
        yb = self._cache_y[:bs].astype(np.float32)
        x_t = torch.tensor(Xb, dtype=torch.float32, device=device)
        y_t = torch.tensor(yb, dtype=torch.float32, device=device)
        return {"inputs": x_t, "target": y_t}

    def build_train_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def _criterion_for(self, context):
        ctx = dict(context) if context else {}
        pw = ctx.get("pos_weight", None)
        if pw is None and "pos_weight" in ctx:
            pw = None
        if self._criterion is None:
            if pw is None:
                self._criterion = nn.BCEWithLogitsLoss()
            else:
                self._criterion = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(float(pw), dtype=torch.float32)
                )
        return self._criterion

    def training_step(self, model, batch, context):
        inputs, target = self._unpack(batch)
        device = next(model.parameters()).device
        x = self._as_float_tensor(inputs, device)
        y = self._as_float_tensor(target, device).view(-1)
        logits = model(x).view(-1)
        crit = self._criterion_for(context)
        if isinstance(crit, nn.BCEWithLogitsLoss) and crit.pos_weight is not None:
            crit = nn.BCEWithLogitsLoss(pos_weight=crit.pos_weight.to(device))
            self._criterion = crit
        return crit(logits, y)

    def validation_step(self, model, batch, context):
        inputs, target = self._unpack(batch)
        device = next(model.parameters()).device
        x = self._as_float_tensor(inputs, device)
        y = self._as_float_tensor(target, device).view(-1)
        crit = self._criterion_for(context)
        if isinstance(crit, nn.BCEWithLogitsLoss) and crit.pos_weight is not None:
            crit = nn.BCEWithLogitsLoss(pos_weight=crit.pos_weight.to(device))
            self._criterion = crit
        with torch.no_grad():
            logits = model(x).view(-1)
            loss = crit(logits, y)
        return loss


# ------------------------------------------------------------------ batch iterator helpers
def csr_batches(X: csr_matrix, y: np.ndarray, indices: np.ndarray, batch_size: int):
    """Yield (dense_float32_x, float32_y) batches honoring batch_size.

    Kept here so later stages can share a single, consistent batching policy.
    """
    n = len(indices)
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        xb = X[idx].toarray().astype(np.float32)
        yb = y[idx].astype(np.float32)
        yield xb, yb


__all__ = [
    "MODEL_FAMILY",
    "WORKING_DIR",
    "SUBMISSION_DIR",
    "DEFAULT_DATA_DIR",
    "SEED",
    "VAL_FRACTION",
    "HIDDEN_DIM",
    "DROPOUT",
    "DEFAULT_THRESHOLD",
    "clean_text",
    "clean_keyword",
    "clean_location",
    "build_feature_text",
    "make_vectorizers",
    "fit_features",
    "transform_features",
    "load_frames",
    "strat_split",
    "SparseMLP",
    "build_loss",
    "probs_from_logits",
    "tune_threshold",
    "csr_batches",
    "CandidateAdapter",
]

```
        
        ### Step 2: datatype_precision
        **Plan:** ## Plan

**WHAT this step must deliver.** Stage 1 produced a sparse TF-IDF → `SparseMLP` classifier whose only float-heavy operation is a `Linear(40000, 256)` matmul plus a tiny `Linear(256, 1)`; the pipeline trace resolves precision as `fp16_amp` on CUDA with FP32 fallback, and normal mode forbids BF16/TF32/quantization. So this stage must supply *utilities* — device/dtype resolution, TF32 disabling, a GradScaler, an autocast region for the forward only, an explicit FP32 island for the fragile `BCEWithLogitsLoss`, finite-loss/grad guards, a deterministic FP32 degradation path, and a one-line precision log — without touching the model family, loss, features, or output interface.

**WHY this matches the workload.** The dominant cost is one wide linear layer; FP16 Tensor-Core autocast can accelerate it on the A10, but sigmoid/log reductions inside `BCEWithLogitsLoss` are numerically fragile, so the loss is computed in an explicit FP32 island with `logits.float()`. Parameters, inputs, targets, metrics and prediction export all stay FP32, and the scaler only becomes active when CUDA + FP16 are both confirmed, so the CPU preflight silently falls back to FP32 with AMP disabled.

**HOW I will validate.** Stage 3 will log the resolved precision line, record finite loss/grad on the first optimizer steps, and fall back to FP32 (`degrade_to_fp32`) if any non-finite value appears; the fallback is deterministic and preserves the output interface, so the val-F1 signal is directly comparable between the FP16-AMP and FP32 runs.

**Backend / elasticity statement.** Memory strategy is unchanged from Stage 1: CSR TF-IDF stays on CPU, batches densify per-slice. Physical batch size is a Stage 3 knob; precision elasticity here is *only* AMP on/off. Operator considerations: `Linear` ops eligible for FP16 autocast, loss/reductions pinned to FP32, TF32 explicitly disabled for both matmul and cudnn. Configurable dimensions preserved: `n_features`, `hidden_dim`, `dropout`. Scheduler-owned controls (device assignment, thread allocation, memory limits, launch policy) are **not** implemented here.

**Note board:**
- baseline_change: Added a standalone `datatype_precision` module that resolves `DEVICE/USE_AMP/AMP_DTYPE/USE_TF32`, builds a CUDA-only `GradScaler`, and exposes `autocast_forward` (forward only) + `compute_bce_with_logits_fp32` (fragile-loss FP32 island) without touching `SparseMLP`, `build_loss`, features, or the output interface | purpose: implements the trace's `fp16_amp` policy as pure, import-safe utilities so Stage 3 can wire autocast around the forward while keeping reductions and metrics in FP32 | hardware_keys: cuda_process, fp16_amp, tf32_disabled
- baseline_change: Added `degrade_to_fp32(...)` + `is_finite_tensor` / `check_grads_finite` so a non-finite loss/grad on FP16 AMP can deterministically revert to FP32 | purpose: satisfies the trace's `fallback_policy` and precision-evidence rule (keep FP32 wherever AMP is unstable/unsupported) while preserving the Stage 1 candidate target | hardware_keys: fp16_amp
- baseline_change: Added `configure_tf32(False)` helper (called from the main guard only) and a single-line `format_precision_log(...)` reporter | purpose: enforces normal-mode TF32 disablement without import-time CUDA side effects and records the actual autocast region / fallback used | hardware_keys: tf32_disabled
        **Code:**
        ```python
"""Stage 2 (datatype_precision) for the Disaster Tweets F1 benchmark.

Consumes the pipeline decision trace:
    precision_policy      = "fp16_amp"
    precision_model_adaptation:
        parameter_dtype   = fp32
        input_dtype       = fp32
        target_dtype      = fp32
        forward_autocast  = {enabled: True, device: cuda, dtype: float16}
        backward/grad/step outside autocast, GradScaler for fp16
    fallback_policy       = fall back to fp32 / disabled AMP if unsupported

Normal mode permits only FP32 + selective FP16 AMP, so:
  * no BF16, no TF32, no FP8/MXFP8/NVFP4 recipes;
  * model parameters and floating inputs/targets remain FP32;
  * autocast is used *only* around the forward path of the real model;
  * the fragile `BCEWithLogitsLoss` reduction is computed in an FP32 island;
  * backward, unscale, and optimizer.step run outside autocast;
  * the CPU preflight (and any non-CUDA device) uses FP32 with the scaler disabled.

Everything here is defined as import-safe helpers/constants only. No CUDA queries,
no TF32 mutation, no logging setup, no model/loss instantiation at import time.
The training_evaluation stage calls these from inside its `if __name__ == "__main__":`
entrypoint. Nothing in this module changes the model family, loss, data features,
output interface, optimizer, scheduler, batch policy, or submission logic.
"""

import contextlib
import torch

# ------------------------------------------------------------------ constants
# Only FP16 is a permitted native AMP dtype in normal mode.
AMP_DTYPE = torch.float16
AMP_DTYPE_NAME = "float16"

# Default policy bundled for easy consumption by the training step.
PRECISION_POLICY = "fp16_amp"
DEFAULT_USE_AMP = True
DEFAULT_USE_TF32 = False  # normal mode: TF32 must be disabled
DEFAULT_ALLOWED_AMP_DTYPES = ("float16",)

# Filled in at runtime by `resolve_precision_config` when a config is bound.
DEVICE = None
USE_AMP = DEFAULT_USE_AMP
USE_TF32 = DEFAULT_USE_TF32
AMP_SCALER = None


# ------------------------------------------------------------------ helpers
def _cuda_available():
    """Import-safe CUDA availability probe (only called at runtime)."""
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_precision_config(
    use_amp=DEFAULT_USE_AMP, requested_device=None, amp_dtype_name=AMP_DTYPE_NAME
):
    """Resolve device / autocast / scaler / TF32 settings for this run.

    Returns a dict with keys:
        device, use_amp, amp_dtype, amp_dtype_name, use_tf32, reason, notes

    * FP16 AMP is enabled only when CUDA is available and `use_amp` is True.
    * Any non-FP16 dtype request collapses to FP32 (normal-mode constraint).
    * CPU / non-CUDA always resolves to FP32 with AMP disabled.
    """
    if requested_device:
        try:
            device = torch.device(requested_device)
        except Exception:
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if _cuda_available() else "cpu")

    cuda_ok = device.type == "cuda" and _cuda_available()
    if device.type == "cuda" and not cuda_ok:
        device = torch.device("cpu")

    amp_name = str(amp_dtype_name or AMP_DTYPE_NAME).lower()
    if amp_name not in DEFAULT_ALLOWED_AMP_DTYPES:
        amp_name = "float32"

    amp_enabled = bool(use_amp) and cuda_ok and amp_name == "float16"

    if amp_enabled:
        reason = "cuda_fp16_amp"
    elif device.type == "cuda":
        reason = "cuda_fp32_amp_disabled_or_unsupported"
    else:
        reason = "cpu_fp32_no_amp"

    return {
        "device": device,
        "use_amp": amp_enabled,
        "amp_dtype": AMP_DTYPE if amp_enabled else torch.float32,
        "amp_dtype_name": "float16" if amp_enabled else "float32",
        "use_tf32": DEFAULT_USE_TF32,
        "reason": reason,
        "notes": (
            "params/inputs/targets=fp32; forward under cuda autocast; "
            "loss+reductions fp32 island; backward/step external; TF32 disabled"
        ),
    }


def configure_tf32(enable=False):
    """Explicitly set the TF32 backends. Call from the main guard only."""
    enable = bool(enable)
    try:
        torch.backends.cuda.matmul.allow_tf32 = enable
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = enable
    except Exception:
        pass


def make_grad_scaler(use_amp, device):
    """Build a GradScaler that is only active for CUDA FP16 AMP.

    Returns a scaler object, or None if GradScaler is unavailable.
    """
    active = (
        bool(use_amp)
        and device is not None
        and getattr(device, "type", "cpu") == "cuda"
    )
    try:
        return torch.amp.GradScaler("cuda", enabled=active)  # torch >= 2.4 API
    except (AttributeError, TypeError):
        pass
    try:
        return torch.cuda.amp.GradScaler(enabled=active)  # legacy API
    except Exception:
        return None


@contextlib.contextmanager
def autocast_forward(use_amp, device, amp_dtype=AMP_DTYPE):
    """Autocast region for the *forward* path only. No-op when AMP is disabled.

    Backward, GradScaler.unscale_, and optimizer.step MUST be executed outside
    this context. Do not wrap the fragile BCE loss inside this context; use
    `compute_bce_with_logits_fp32` instead, which is an explicit FP32 island.
    """
    if not use_amp or device is None or getattr(device, "type", "cpu") != "cuda":
        yield
        return

    cm = None
    try:
        cm = torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True)
    except (AttributeError, TypeError):
        try:
            cm = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=True)
        except Exception:
            cm = None

    if cm is None:
        yield
        return

    with cm:
        yield


def compute_bce_with_logits_fp32(criterion, logits, targets):
    """Compute the BCE-with-logits loss in an explicit FP32 island.

    Autocast may hand us an FP16 logit tensor. Sigmoid + log + weighted-mean
    reductions inside BCEWithLogitsLoss are numerically fragile in FP16, so we
    cast to FP32 before calling the (unchanged) Stage 1 criterion. This preserves
    the Stage 1 loss object, output interface, and reduction semantics.
    """
    logits_fp32 = logits.float() if isinstance(logits, torch.Tensor) else logits
    targets_fp32 = (
        targets.float().view(-1) if isinstance(targets, torch.Tensor) else targets
    )
    return criterion(logits_fp32, targets_fp32)


def is_finite_tensor(x):
    """True if tensor is None or all elements are finite."""
    if x is None:
        return True
    if not isinstance(x, torch.Tensor):
        try:
            x = torch.as_tensor(x)
        except Exception:
            return True
    try:
        return bool(torch.isfinite(x.detach()).all().item())
    except Exception:
        return False


def check_grads_finite(model):
    """True if every populated parameter gradient is finite."""
    if model is None:
        return True
    try:
        for p in model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                return False
    except Exception:
        return True
    return True


def degrade_to_fp32(precision_cfg, reason="non_finite_loss_or_grad"):
    """Deterministic fallback from FP16 AMP to FP32.

    Returns a new config dict with AMP disabled so that Stage 3 can swap in
    the FP32 path without changing model family, loss, features, or interface.
    """
    new_cfg = dict(precision_cfg or {})
    new_cfg["use_amp"] = False
    new_cfg["amp_dtype"] = torch.float32
    new_cfg["amp_dtype_name"] = "float32"
    new_cfg["fallback_from"] = (precision_cfg or {}).get("reason", "unknown")
    new_cfg["reason"] = f"fallback_fp32:{reason}"
    return new_cfg


def format_precision_log(precision_cfg, extra=None):
    """One-line precision report for epoch-level logging (no batch noise)."""
    cfg = precision_cfg or {}
    line = (
        f"[precision] device={cfg.get('device', 'unknown')} "
        f"policy={PRECISION_POLICY} "
        f"amp={cfg.get('use_amp', False)} "
        f"amp_dtype={cfg.get('amp_dtype_name', 'float32')} "
        f"tf32={cfg.get('use_tf32', DEFAULT_USE_TF32)} "
        f"reason={cfg.get('reason', 'unset')}"
    )
    if extra:
        try:
            line += " " + " ".join(f"{k}={v}" for k, v in dict(extra).items())
        except Exception:
            pass
    return line


__all__ = [
    "AMP_DTYPE",
    "AMP_DTYPE_NAME",
    "PRECISION_POLICY",
    "DEFAULT_USE_AMP",
    "DEFAULT_USE_TF32",
    "DEFAULT_ALLOWED_AMP_DTYPES",
    "DEVICE",
    "USE_AMP",
    "USE_TF32",
    "AMP_SCALER",
    "resolve_precision_config",
    "configure_tf32",
    "make_grad_scaler",
    "autocast_forward",
    "compute_bce_with_logits_fp32",
    "is_finite_tensor",
    "check_grads_finite",
    "degrade_to_fp32",
    "format_precision_log",
]

```
        
        ### Step 3: training_evaluation
        **Plan:** ## Plan

**WHAT/WHY.** Stage 1 gives a real learned `SparseMLP` over train-fitted TF-IDF, Stage 2 gives `fp16_amp` utilities (autocast-forward + FP32 BCE island + GradScaler + `degrade_to_fp32`). This stage must train that exact model on the internal stratified train/val split, tune the F1 decision threshold on the held-out val set, and drive test inference through the same forward path. The only *new* modelling decisions allowed are training hyperparameters: AdamW with decoupled weight decay, linear-warmup+cosine LR, physical batch 64 (quality-safe envelope 32–128), full-data exposure budget, gradient accumulation to preserve the effective batch, early stopping on val F1 with best-checkpoint restore.

**HOW I will validate.** Theoretically the "sparse lexical MLP" hypothesis is confirmed if val F1 reaches ≈0.78–0.82 while exceeding the 0.5-threshold baseline, and if the epoch metric trace is monotone-ish before plateauing (no overfitting). If val F1 < ~0.75 or collapses, dropout/regularization are too weak or the LR schedule is wrong; best-checkpoint restore + early stopping guards test quality regardless.

**Backend / elasticity.** CSR TF-IDF stays on CPU; batches densify to `float32` per slice (physical 64 × 40k floats ≈ 10 MB). Physical batch is the only elasticity knob; effective batch is fixed at 64 via accumulation. AMP wraps *only* the forward; BCE is computed in an explicit FP32 island; backward/step outside autocast; TF32 disabled; fallback to FP32 if any non-finite loss/grad appears. Scheduler owns launch/device/threads — this subprocess never sets them.

**Note board:**
- baseline_change: Added the real `run_training` loop (AdamW + linear-warmup/cosine, physical batch 64, accum 1), `QUALITY_SAFE_PHYSICAL_BATCH_SIZES=[32,64,128]`, `BATCH_LR_SCALING_POLICY="linear"`, best-val-F1 checkpoint restore, val-F1 threshold tuning via Stage 1's `tune_threshold`, and test inference through the same `autocast_forward` + `SparseMLP` path | purpose: complete the Stage 1 candidate target with a training/eval loop that shares the exact preprocessing with test inference | hardware_keys: none
- baseline_change: Consumed Stage 2 utilities (`autocast_forward`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `resolve_precision_config`, `degrade_to_fp32`, `configure_tf32(False)`, `format_precision_log`) without redefining dtype policy; AMP region = forward only, BCE forced to FP32, backward/unscale/step outside autocast | purpose: honor `fp16_amp` policy + `BCEWithLogitsLoss` FP32 island while retaining FP32 params/inputs/targets and true sigmoid→threshold inference | hardware_keys: fp16_amp, degrade_to_fp32, tf32_disabled
- baseline_change: Wired `script_scheduler_context` / `SafePointType.{BEFORE_TRAIN,STEP,EPOCH}` hooks, `context.load_resume_checkpoint`, `context.job.max_epochs` + `store.save_job`, per-epoch `MLEVOLVE_EPOCH_METRIC` line, and `TrainingDiagnostics` (`after_update` post-step, `report` per epoch, `state_dict` inside the safe-point factory) | purpose: satisfy the scheduler/diagnostics contract so a later run can learn from measured throughput, VRAM, and batch/LR elasticity | hardware_keys: none
        **Code:**
        ```python
"""Stage 3 (training_evaluation) for the Disaster Tweets F1 benchmark.

Consumes (assumed already imported/defined by earlier stages in this same module):
    Stage 1: MODEL_FAMILY, WORKING_DIR, SUBMISSION_DIR, DEFAULT_DATA_DIR, SEED,
             VAL_FRACTION, HIDDEN_DIM, DROPOUT, DEFAULT_THRESHOLD,
             build_feature_text, make_vectorizers, fit_features,
             transform_features, load_frames, strat_split, SparseMLP,
             build_loss, probs_from_logits, tune_threshold, csr_batches.
    Stage 2: resolve_precision_config, configure_tf32, make_grad_scaler,
             autocast_forward, compute_bce_with_logits_fp32, is_finite_tensor,
             check_grads_finite, degrade_to_fp32, format_precision_log, AMP_DTYPE.

This stage owns ONLY training hyperparameters: optimizer, lr, weight decay,
scheduler, physical batch, gradient accumulation, epochs, dataloader settings,
checkpointing, early stopping, validation, diagnostics, test inference, and the
submission file. Model family, features, loss, output interface, and precision
policy are unchanged from prior stages.

Import-safe: no data IO, no CUDA configuration, no model/loss/loader creation,
no training, no submission writing at module import time. All side effects live
under `if __name__ == "__main__":`.
"""

import contextlib
import copy
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

# Re-assert the model family constant (must match Stage 1's value).
MODEL_FAMILY = "tfidf_mlp_tweets_v1"

# ------------------------------------------------------------------ Stage 3 hyperparameters
PLANNED_EPOCHS = 14
PHYSICAL_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 1
EFFECTIVE_BATCH_SIZE = PHYSICAL_BATCH_SIZE * GRAD_ACCUM_STEPS
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_FRACTION = 0.10
EARLY_STOP_PATIENCE = 4
GRAD_CLIP_NORM = 1.0
DATALOADER_WORKERS = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True
CHECKPOINT_EVERY_EPOCH = True
POS_WEIGHT = 1.0  # class balance ~43/57; F1 threshold tuning handles imbalance

# Scheduler envelope (do not silently widen; scheduler may pick inside this set)
QUALITY_SAFE_PHYSICAL_BATCH_SIZES = [32, 64, 128]
BATCH_LR_SCALING_POLICY = "linear"


# ------------------------------------------------------------------ dataset / loader helpers
class _SparseTfidfDataset(torch.utils.data.Dataset):
    """Wrap a CSR TF-IDF matrix + float labels. Densifies per-item to float32."""

    def __init__(self, X_csr, y):
        self.X = X_csr
        self.y = np.asarray(y, dtype=np.float32).reshape(-1)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        row = self.X[idx]
        dense = np.asarray(row.todense(), dtype=np.float32).ravel()
        return torch.from_numpy(dense), torch.tensor(
            float(self.y[idx]), dtype=torch.float32
        )


def _make_loader(X_csr, y, batch_size, shuffle):
    ds = _SparseTfidfDataset(X_csr, y)
    workers = max(0, int(DATALOADER_WORKERS))
    kwargs = dict(
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=workers,
        pin_memory=bool(PIN_MEMORY),
        drop_last=False,
    )
    if workers > 0 and PERSISTENT_WORKERS:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return ds, torch.utils.data.DataLoader(ds, **kwargs)


def _make_lr_schedule(optimizer, total_steps, warmup_steps):
    """Linear warmup -> cosine decay. total_steps/warmup_steps are in optimizer steps."""
    total_steps = max(1, int(total_steps))
    warmup_steps = max(1, int(warmup_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        progress = min(1.0, max(0.0, progress))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _sigmoid_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    return 1.0 / (1.0 + np.exp(-logits))


def _forward_logits(model, x_tensor, use_amp, device, amp_dtype):
    """Forward only, under the Stage 2 autocast region. Returns FP32 CPU logits."""
    with autocast_forward(use_amp, device, amp_dtype):
        logits = model(x_tensor).view(-1)
    return logits.float()


@torch.no_grad()
def _predict_logits(model, loader, device, use_amp, amp_dtype):
    model.eval()
    chunks = []
    for batch in loader:
        xb = batch[0] if isinstance(batch, (tuple, list)) else batch["inputs"]
        xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
        logits = _forward_logits(model, xb, use_amp, device, amp_dtype)
        chunks.append(logits.detach().cpu())
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return torch.cat(chunks).numpy().astype(np.float32)


# ------------------------------------------------------------------ CandidateAdapter (CPU preflight)
class CandidateAdapter:
    """CPU-safe preflight adapter. Uses the real model, loss, features, optimizer.

    Stage 1 already provides a CandidateAdapter; this version is written to be
    self-contained so the module can also be executed standalone for preflight.
    """

    def __init__(self):
        self._word_vec = None
        self._features = None
        self._y = None
        self._n_features = None
        self._criterion = None
        self._preflight_rows = 256
        self._preflight_word_features = 512

    def _sparse_dataset_from_df(self, df):
        text = build_feature_text(df)
        wv = make_vectorizers()[0]
        wv.set_params(max_features=self._preflight_word_features, min_df=1)
        X = wv.fit_transform(text).tocsr().astype(np.float32)
        self._word_vec = wv
        return X

    def _ensure(self):
        if self._n_features is not None:
            return
        data_dir = os.environ.get("MLEVOLVE_INPUT_DIR", DEFAULT_DATA_DIR)
        df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        if "target" not in df.columns:
            raise ValueError("train.csv is missing 'target' column for preflight")
        df = df.head(min(self._preflight_rows, len(df))).reset_index(drop=True)
        X = self._sparse_dataset_from_df(df)
        self._features = X
        self._y = df["target"].astype(np.float32).values
        self._n_features = int(X.shape[1])

    def _criterion_for(self, context):
        ctx = dict(context) if context else {}
        pw = ctx.get("pos_weight", POS_WEIGHT)
        if pw is None:
            pw = POS_WEIGHT
        if self._criterion is None or (
            isinstance(self._criterion, nn.BCEWithLogitsLoss)
            and (self._criterion.pos_weight is None) != (float(pw) == 1.0)
        ):
            self._criterion = build_loss(pos_weight=float(pw))
        return self._criterion

    def build_model(self, context):
        self._ensure()
        ctx = dict(context) if context else {}
        hidden = int(ctx.get("hidden_dim", HIDDEN_DIM))
        dropout = float(ctx.get("dropout", DROPOUT))
        return SparseMLP(self._n_features, hidden_dim=hidden, dropout=dropout)

    def build_optimizer(self, model, context):
        ctx = dict(context) if context else {}
        lr = float(ctx.get("lr", LEARNING_RATE))
        wd = float(ctx.get("weight_decay", WEIGHT_DECAY))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def _build_batch(self, scenario, device):
        self._ensure()
        ctx = dict(scenario) if scenario else {}
        bs = int(ctx.get("batch_size", 4) or 4)
        bs = max(1, min(bs, self._features.shape[0]))
        Xb = self._features[:bs].toarray().astype(np.float32)
        yb = self._y[:bs].astype(np.float32)
        x_t = torch.tensor(Xb, dtype=torch.float32, device=device)
        y_t = torch.tensor(yb, dtype=torch.float32, device=device)
        return {"inputs": x_t, "target": y_t}

    def build_train_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def _unpack(self, batch):
        if isinstance(batch, dict):
            return batch["inputs"], batch["target"]
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0], batch[1]
        raise ValueError("Unsupported batch format in CandidateAdapter")

    def _as_float(self, x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.float32)
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return torch.tensor(arr, dtype=torch.float32, device=device)

    def training_step(self, model, batch, context):
        inputs, target = self._unpack(batch)
        device = next(model.parameters()).device
        x = self._as_float(inputs, device)
        y = self._as_float(target, device).view(-1)
        crit = self._criterion_for(context)
        # Autocast only around forward; BCE kept in FP32 island (Stage 2 contract).
        logits = _forward_logits(
            model, x, use_amp=False, device=device, amp_dtype=torch.float32
        )
        return compute_bce_with_logits_fp32(crit, logits, y)

    def validation_step(self, model, batch, context):
        inputs, target = self._unpack(batch)
        device = next(model.parameters()).device
        x = self._as_float(inputs, device)
        y = self._as_float(target, device).view(-1)
        crit = self._criterion_for(context)
        with torch.no_grad():
            logits = _forward_logits(
                model, x, use_amp=False, device=device, amp_dtype=torch.float32
            )
            loss = compute_bce_with_logits_fp32(crit, logits, y)
        return loss


# ------------------------------------------------------------------ training entrypoint
def run_training():
    """Full Stage 3 training + validation + test inference + submission."""
    # ---- Precision: disable TF32 explicitly (normal mode) -----------------
    configure_tf32(False)

    # ---- Determinism ------------------------------------------------------
    seed = int(SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    data_dir = os.environ.get("MLEVOLVE_INPUT_DIR", DEFAULT_DATA_DIR)

    # ---- Scheduler context (BEFORE any CUDA allocation) ------------------
    context = None
    SafePointType = None
    try:
        from localml_scheduler.execution.script_context import script_scheduler_context
        from localml_scheduler.domain import SafePointType as _SafePointType

        SafePointType = _SafePointType
        context = script_scheduler_context()
    except Exception:
        context = None
        SafePointType = None

    # ---- Data + Stage 1 features (fit on train split only) ---------------
    train_df, test_df = load_frames(data_dir)
    tr_df, va_df = strat_split(train_df, VAL_FRACTION, SEED)
    word_vec, char_vec, X_tr = fit_features(tr_df)
    X_va = transform_features(va_df, word_vec, char_vec)
    X_te = transform_features(test_df, word_vec, char_vec)
    y_tr = tr_df["target"].astype(np.float32).values
    y_va = va_df["target"].astype(np.float32).values
    n_features = int(X_tr.shape[1])

    # ---- Precision resolution + scaler -----------------------------------
    prec_cfg = resolve_precision_config(use_amp=True)
    device = prec_cfg["device"]
    use_amp = bool(prec_cfg["use_amp"])
    amp_dtype = prec_cfg["amp_dtype"]
    scaler = make_grad_scaler(use_amp, device)
    print(format_precision_log(prec_cfg, {"n_features": n_features}))

    # ---- Model / loss / optimizer ----------------------------------------
    model = SparseMLP(n_features, hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
    criterion = build_loss(pos_weight=POS_WEIGHT)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # ---- DataLoaders ------------------------------------------------------
    tr_ds, tr_loader = _make_loader(X_tr, y_tr, PHYSICAL_BATCH_SIZE, shuffle=True)
    _, va_loader = _make_loader(X_va, y_va, PHYSICAL_BATCH_SIZE, shuffle=False)

    steps_per_epoch = int(math.ceil(len(tr_ds) / float(PHYSICAL_BATCH_SIZE)))
    optim_steps_per_epoch = max(
        1, int(math.ceil(steps_per_epoch / float(GRAD_ACCUM_STEPS)))
    )
    total_optim_steps = optim_steps_per_epoch * PLANNED_EPOCHS
    warmup_steps = max(1, int(WARMUP_FRACTION * total_optim_steps))
    scheduler = _make_lr_schedule(optimizer, total_optim_steps, warmup_steps)

    # ---- Diagnostics (context-managed) -----------------------------------
    TrainingDiagnostics = None
    try:
        from utils.training_diagnostics import TrainingDiagnostics as _TD

        TrainingDiagnostics = _TD
    except Exception:
        TrainingDiagnostics = None

    diag_settings = {
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "planned_epochs": PLANNED_EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "model_family": MODEL_FAMILY,
        "split_seed": SEED,
        "precision": prec_cfg.get("amp_dtype_name", "float32"),
        "precision_reason": prec_cfg.get("reason", "unknown"),
    }

    stack = contextlib.ExitStack()
    if TrainingDiagnostics is not None:
        try:
            diagnostics = stack.enter_context(
                TrainingDiagnostics(
                    model, optimizer, scaler=scaler, settings=diag_settings
                )
            )
        except Exception:
            diagnostics = None
    else:
        diagnostics = None

    # closure-based state_factory (captures current state objects)
    def make_state_factory(epoch_val, step_val):
        def _state():
            st = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "epoch": int(epoch_val),
                "global_step": int(step_val),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
            }
            if torch.cuda.is_available():
                st["cuda_rng"] = torch.cuda.get_rng_state_all()
            if diagnostics is not None:
                try:
                    st["diagnostics"] = diagnostics.state_dict()
                except Exception:
                    pass
            return st

        return _state

    global_step = 0
    start_epoch = 0

    # ---- Scheduler: configure budget and resume --------------------------
    if context is not None:
        try:
            context.job.max_epochs = PLANNED_EPOCHS
        except Exception:
            pass
        try:
            context.job.config.max_epochs = PLANNED_EPOCHS
        except Exception:
            pass
        try:
            context.store.save_job(context.job)
        except Exception:
            pass

        resume = context.load_resume_checkpoint()
        if resume and isinstance(resume, dict) and resume.get("state"):
            state = resume["state"]
            try:
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                if scaler is not None and state.get("scaler"):
                    scaler.load_state_dict(state["scaler"])
                if state.get("scheduler"):
                    scheduler.load_state_dict(state["scheduler"])
                if diagnostics is not None and state.get("diagnostics"):
                    diagnostics.load_state_dict(state["diagnostics"])
                global_step = int(state.get("global_step", 0))
                start_epoch = int(state.get("epoch", 0))
            except Exception:
                start_epoch = 0
                global_step = 0

        context.control_hook.safe_point(
            SafePointType.BEFORE_TRAIN,
            epoch=start_epoch,
            global_step=global_step,
            steps_per_epoch=steps_per_epoch,
            state_factory=make_state_factory(start_epoch, global_step),
        )

    # ---- Training loop ----------------------------------------------------
    best_val_f1 = -1.0
    best_threshold = float(DEFAULT_THRESHOLD)
    best_state = None
    epochs_no_improve = 0
    t_start = time.time()
    amp_fallback_done = False

    for epoch in range(start_epoch, PLANNED_EPOCHS):
        model.train()
        running_loss = 0.0
        seen = 0
        accum = 0
        optimizer.zero_grad(set_to_none=True)

        for xb, yb in tr_loader:
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
            yb = yb.to(device=device, dtype=torch.float32, non_blocking=True).view(-1)

            logits = _forward_logits(model, xb, use_amp, device, amp_dtype)
            loss = compute_bce_with_logits_fp32(criterion, logits, yb)

            if not is_finite_tensor(loss):
                if use_amp and not amp_fallback_done:
                    prec_cfg = degrade_to_fp32(prec_cfg, reason="non_finite_train_loss")
                    use_amp = False
                    amp_dtype = torch.float32
                    scaler = make_grad_scaler(False, device)
                    amp_fallback_done = True
                    print(
                        format_precision_log(
                            prec_cfg, {"fallback": "non_finite_train_loss"}
                        )
                    )
                    continue
                raise RuntimeError("Non-finite training loss after FP32 fallback")

            loss_scaled = loss / float(GRAD_ACCUM_STEPS)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            running_loss += float(loss.item()) * xb.shape[0]
            seen += int(xb.shape[0])
            accum += 1

            if accum >= GRAD_ACCUM_STEPS:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)

                if not check_grads_finite(model):
                    if use_amp and not amp_fallback_done:
                        prec_cfg = degrade_to_fp32(prec_cfg, reason="non_finite_grads")
                        use_amp = False
                        amp_dtype = torch.float32
                        scaler = make_grad_scaler(False, device)
                        amp_fallback_done = True
                        optimizer.zero_grad(set_to_none=True)
                        accum = 0
                        print(
                            format_precision_log(
                                prec_cfg, {"fallback": "non_finite_grads"}
                            )
                        )
                        continue
                    raise RuntimeError("Non-finite gradients after FP32 fallback")

                if GRAD_CLIP_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                global_step += 1

                if diagnostics is not None:
                    diagnostics.after_update()

                if context is not None:
                    context.control_hook.safe_point(
                        SafePointType.STEP,
                        epoch=epoch + 1,
                        global_step=global_step,
                        steps_per_epoch=steps_per_epoch,
                        state_factory=make_state_factory(epoch + 1, global_step),
                    )

        train_loss = running_loss / max(1, seen)

        # ---- Validation: forward through the model, tune F1 threshold ----
        val_logits = _predict_logits(model, va_loader, device, use_amp, amp_dtype)
        val_probs = _sigmoid_np(val_logits)
        thr = float(tune_threshold(val_probs, y_va))
        val_preds = (val_probs >= thr).astype(int)
        val_f1 = float(f1_score(y_va.astype(int), val_preds, zero_division=0))

        elapsed = time.time() - t_start
        print(
            f"MLEVOLVE_EPOCH_METRIC "
            f'{{"epoch": {epoch + 1}, "metric": {val_f1}, "metric_name": "f1"}}'
        )
        print(
            f"[epoch {epoch + 1}/{PLANNED_EPOCHS}] loss={train_loss:.4f} "
            f"val_f1={val_f1:.4f} thr={thr:.3f} elapsed={elapsed:.1f}s"
        )

        if diagnostics is not None:
            try:
                diagnostics.report(epoch=epoch + 1)
            except Exception:
                pass

        # ---- Best-checkpoint tracking -----------------------------------
        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_threshold = thr
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # ---- Checkpoint (working) ----------------------------------------
        if CHECKPOINT_EVERY_EPOCH:
            ckpt = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "best_val_f1": best_val_f1,
                "best_threshold": best_threshold,
                "model_family": MODEL_FAMILY,
                "n_features": n_features,
            }
            if diagnostics is not None:
                try:
                    ckpt["diagnostics"] = diagnostics.state_dict()
                except Exception:
                    pass
            try:
                torch.save(ckpt, os.path.join(WORKING_DIR, "checkpoint.pt"))
            except Exception:
                pass

        if context is not None:
            context.control_hook.safe_point(
                SafePointType.EPOCH,
                epoch=epoch + 1,
                global_step=global_step,
                steps_per_epoch=steps_per_epoch,
                state_factory=make_state_factory(epoch + 1, global_step),
            )

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"[early_stop] no improvement for {EARLY_STOP_PATIENCE} epochs")
            break

    # ---- Restore best weights --------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # ---- Test inference (identical preprocessing + forward path) ---------
    _, te_loader = _make_loader(
        X_te,
        np.zeros(X_te.shape[0], dtype=np.float32),
        PHYSICAL_BATCH_SIZE,
        shuffle=False,
    )
    test_logits = _predict_logits(model, te_loader, device, use_amp, amp_dtype)
    test_probs = _sigmoid_np(test_logits)
    test_preds = (test_probs >= float(best_threshold)).astype(int)

    sub = pd.DataFrame(
        {"id": test_df["id"].astype(int).values, "target": test_preds.astype(int)}
    )
    sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    sub.to_csv(sub_path, index=False)

    # Leave diagnostics context before the final score line
    stack.close()

    # Final validation score (must be the last printed line)
    final_score = float(best_val_f1 if best_val_f1 >= 0 else 0.0)
    print(f"Final Validation Score: {final_score}")


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
        