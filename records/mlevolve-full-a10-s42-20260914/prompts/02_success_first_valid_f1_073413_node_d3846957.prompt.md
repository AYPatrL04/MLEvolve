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
 Summary: 1 total attempts - 1 executed, 0 successful (all failed or buggy)

-------------------------------
Attempt #1:
Design: I merged the three stages into one script: Stage 1 provides data prep, tokenizer, dataset, `DisasterTransformer`, criterion, threshold search, and `CandidateAdapter`; Stage 2 adds `PrecisionConfig`/`resolve_precision`/`PrecisionRuntime`/autocast+scaler plumbing and TF32-off policy; Stage 3 wraps these into the full train/validate/test loop with scheduler hooks, diagnostics, AdamW + warmup-linear schedule, best-checkpoint restore, and submission writing at `./submission/submission.csv`. I deduplicated imports and the single shared `MODEL_FAMILY` constant, and kept all side effects under `if __name__ == "__main__":` with `Final Validation Score:` as the last line.
Results: The implementation of this design has bugs.
Insight: Using a different approach may not result in the same bugs as the above approach.


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
{
  "model_design": {
    "modality": "text",
    "target_type": "classification",
    "shape_constraints": [
      "sequence length and tokenizer output must stay configurable"
    ],
    "family": "available_transformer_or_sequence_baseline",
    "alternatives_considered": [],
    "loss": "cross entropy or metric-compatible classification loss",
    "output_interface": "match the competition metric and required submission format",
    "reason": "Use a baseline-compatible model family until stronger task or evidence signals are available.",
    "hardware_fit": "use only compatible evidence"
  },
  "datatype_precision": {
    "precision_policy": "disabled",
    "precision_model_adaptation": "none",
    "fallback_policy": "fall back to fp32 or disabled AMP if the selected precision path is unsupported",
    "reason": "Use conservative precision until hardware and package support are confirmed."
  },
  "training_evaluation": {
    "optimizer": "AdamW for neural models or the library default for tree/linear baselines",
    "scheduler": "none",
    "batch_size_policy": "fixed",
    "dataloader_policy": "safe defaults; enable workers and pinned memory only when compatible",
    "fallbacks": [
      "reduce physical batch size on OOM",
      "reduce epochs before changing model family on timeout"
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
      "predictor/graph evidence not available"
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
 Summary: 1 total attempts - 1 executed, 0 successful (all failed or buggy)

-------------------------------
Attempt #1:
Design: I merged the three stages into one script: Stage 1 provides data prep, tokenizer, dataset, `DisasterTransformer`, criterion, threshold search, and `CandidateAdapter`; Stage 2 adds `PrecisionConfig`/`resolve_precision`/`PrecisionRuntime`/autocast+scaler plumbing and TF32-off policy; Stage 3 wraps these into the full train/validate/test loop with scheduler hooks, diagnostics, AdamW + warmup-linear schedule, best-checkpoint restore, and submission writing at `./submission/submission.csv`. I deduplicated imports and the single shared `MODEL_FAMILY` constant, and kept all side effects under `if __name__ == "__main__":` with `Final Validation Score:` as the last line.
Results: The implementation of this design has bugs.
Insight: Using a different approach may not result in the same bugs as the above approach.


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
{
  "model_design": {
    "modality": "text",
    "target_type": "classification",
    "shape_constraints": [
      "sequence length and tokenizer output must stay configurable"
    ],
    "family": "available_transformer_or_sequence_baseline",
    "alternatives_considered": [],
    "loss": "cross entropy or metric-compatible classification loss",
    "output_interface": "match the competition metric and required submission format",
    "reason": "Use a baseline-compatible model family until stronger task or evidence signals are available.",
    "hardware_fit": "use only compatible evidence"
  },
  "datatype_precision": {
    "precision_policy": "disabled",
    "precision_model_adaptation": "none",
    "fallback_policy": "fall back to fp32 or disabled AMP if the selected precision path is unsupported",
    "reason": "Use conservative precision until hardware and package support are confirmed."
  },
  "training_evaluation": {
    "optimizer": "AdamW for neural models or the library default for tree/linear baselines",
    "scheduler": "none",
    "batch_size_policy": "fixed",
    "dataloader_policy": "safe defaults; enable workers and pinned memory only when compatible",
    "fallbacks": [
      "reduce physical batch size on OOM",
      "reduce epochs before changing model family on timeout"
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
      "predictor/graph evidence not available"
    ]
  }
}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=## Plan **WHAT**: Tweets are short, noisy, informal text with strong keyword signals (e.g., "wildfire", "derailment") and distinct distributional patterns between real disasters and figurative/hyperbolic uses of disas...; purpose=Define the Stage 1 candidate target for later precision and training stages.; hardware_keys=max_len, batch_size, emb_dim, n_layers, load_all_data, MLEVOLVE_INPUT_DIR

# Previous steps
### model_design
**Plan:** ## Plan

**WHAT**: Tweets are short, noisy, informal text with strong keyword signals (e.g., "wildfire", "derailment") and distinct distributional patterns between real disasters and figurative/hyperbolic uses of disaster words. Location fields are extremely sparse/noisy, and tweet text length varies from a few tokens to a paragraph. Binary F1 on ~6k training samples is the metric, so both ranking quality and threshold calibration matter.

**WHY**: I'll build a **hybrid from-scratch classifier** that jointly learns token-level interactions (word-level Transformer encoder with padding-masked multi-head attention pooling) and non-textual cues (a dedicated keyword embedding + hand-crafted numeric features like caps ratio, digit/URL counts, hashtag presence). This is deliberately novel relative to prior attempts: it avoids dependency on any potentially-unavailable pretrained HF checkpoint (offline constraint), while still capturing the task-specific structure that generic bag-of-words baselines miss — namely the interaction between *which* disaster keyword appears and *how* it is used (caps, exclamations, URL presence, hashtag patterns). Training-set-only vocab construction and a stratified seed-42 split mirror the benchmark.

**HOW**: Confirmation = the hybrid model achieves F1 clearly above a keyword-only or TF-IDF logistic baseline on the held-out split when its threshold is tuned post-hoc. Reconsideration trigger = if validation F1 collapses near the class prior (~0.57) or plateaus at the level of the keyword indicator alone, then the from-scratch transformer is not extracting signal beyond the keyword and I'd fall back to a subword/character-augmented variant.

**Memory/backend notes**: Memory strategy = fixed-shape `max_len` sequence tensor (configurable) plus small dense feature vectors, so batches are homogeneous and CUDA-friendly; no gradient accumulation policy is chosen here (Stage 3 handles it). Elasticity = `batch_size` is honored from `scenario`/`context` by the adapter. Operator/normalization: `LayerNorm` pre-norm transformer blocks + dropout; embedding dim and layer count stay configurable. Fallbacks = `max_len`, `emb_dim`, `n_layers` all configurable for OOM/timeout. Scheduler-owned controls (daemon, device assignment, thread caps, memory limits) are **not** implemented here.

**Note board**:
- baseline_change: Replace a generic sequence baseline with a hybrid word-level Transformer encoder (pre-norm, padding-masked, attention-pooled) fused with a keyword embedding and 10 hand-crafted numeric features, all trained from scratch (no external weights). | purpose: matches the text modality, removes offline-checkpoint risk, and lets the model learn keyword-usage interactions that drive real-vs-figurative detection for F1. | hardware_keys: none
- baseline_change: Fit tokenizer/vocab/keyword-index ONLY on the training split; keep `max_len` and model width configurable and expose a leak-free `load_all_data` returning (train, val, test, vocab, keyword2idx). | purpose: prevents leakage into validation/test and gives Stage 2/3 a stable, configurable tensor contract. | hardware_keys: none
- baseline_change: Expose `CandidateAdapter` with real forward/loss/batch builders using real CSV rows from `MLEVOLVE_INPUT_DIR`. | purpose: satisfies the CPU preflight contract while exercising the actual model + Focal-BCE loss. | hardware_keys: none
**Code:**
```python
import os
import re
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Top-level constants (import-safe)
# ---------------------------------------------------------------------------
MODEL_FAMILY = "disaster_hybrid_transformer_v1"

INPUT_DIR_DEFAULT = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"

SEED = 42
VAL_FRACTION = 0.20
MAX_LEN = 48
MIN_FREQ = 2
PAD_IDX = 0
UNK_IDX = 1
NUM_NUMERIC = 10


# ---------------------------------------------------------------------------
# Text cleaning / tokenization
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s\-\'!?]")
_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = _URL_RE.sub(" url ", t)
    t = _MENTION_RE.sub(" user ", t)
    t = _HASHTAG_RE.sub(r" \1 ", t)
    t = _NON_ALNUM_RE.sub(" ", t)
    t = _SPACE_RE.sub(" ", t).strip()
    return t


def tokenize(text: str) -> List[str]:
    return clean_text(text).split()


def extract_numeric_features(text: str, keyword: str, location: str) -> np.ndarray:
    t = text if isinstance(text, str) else ""
    k = keyword if isinstance(keyword, str) else ""
    l = location if isinstance(location, str) else ""
    n_char = float(len(t))
    n_word = float(len(t.split()))
    n_upper = sum(1 for c in t if c.isupper())
    upper_ratio = n_upper / max(n_char, 1.0)
    n_digit = sum(1 for c in t if c.isdigit())
    digit_ratio = n_digit / max(n_char, 1.0)
    n_exclam = float(t.count("!"))
    n_question = float(t.count("?"))
    n_hashtag = float(t.count("#"))
    n_url = 1.0 if ("http" in t.lower()) else 0.0
    has_keyword = 1.0 if k.strip() else 0.0
    has_location = 1.0 if l.strip() else 0.0
    return np.asarray(
        [
            n_char,
            n_word,
            upper_ratio,
            digit_ratio,
            n_exclam,
            n_question,
            n_hashtag,
            n_url,
            has_keyword,
            has_location,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Vocabulary (fit on TRAIN only)
# ---------------------------------------------------------------------------
class Vocab:
    def __init__(self, min_freq: int = MIN_FREQ):
        self.min_freq = int(min_freq)
        self.token2idx: Dict[str, int] = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}

    def fit(self, texts: List[str]) -> "Vocab":
        counter: Dict[str, int] = {}
        for t in texts:
            for tok in tokenize(t):
                counter[tok] = counter.get(tok, 0) + 1
        for tok, cnt in counter.items():
            if cnt >= self.min_freq and tok not in self.token2idx:
                self.token2idx[tok] = len(self.token2idx)
        return self

    def encode(self, text: str, max_len: int = MAX_LEN) -> List[int]:
        toks = tokenize(text)
        if not toks:
            toks = ["<unk>"]
        ids = [self.token2idx.get(tok, UNK_IDX) for tok in toks[:max_len]]
        if len(ids) < max_len:
            ids = ids + [PAD_IDX] * (max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.token2idx)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TweetDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocab,
        keyword2idx: Dict[str, int],
        max_len: int = MAX_LEN,
        has_target: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.keyword2idx = keyword2idx
        self.max_len = int(max_len)
        self.has_target = bool(has_target)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = row["text"] if "text" in row and isinstance(row["text"], str) else ""
        kw = (
            row["keyword"]
            if "keyword" in row and isinstance(row["keyword"], str)
            else ""
        )
        loc = (
            row["location"]
            if "location" in row and isinstance(row["location"], str)
            else ""
        )

        input_ids = self.vocab.encode(text, self.max_len)
        kw_key = kw.lower().strip()
        kw_id = self.keyword2idx.get(kw_key, 0)
        num_feats = extract_numeric_features(text, kw, loc)

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "keyword_id": torch.tensor(kw_id, dtype=torch.long),
            "num_feats": torch.tensor(num_feats, dtype=torch.float32),
        }
        if self.has_target and "target" in row:
            item["target"] = torch.tensor(float(row["target"]), dtype=torch.float32)
        else:
            item["target"] = torch.tensor(0.0, dtype=torch.float32)
        if "id" in row:
            item["id"] = torch.tensor(int(row["id"]), dtype=torch.long)
        return item


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    keyword_id = torch.stack([b["keyword_id"] for b in batch], dim=0)
    num_feats = torch.stack([b["num_feats"] for b in batch], dim=0)
    target = torch.stack([b["target"] for b in batch], dim=0)
    out: Dict[str, object] = {
        "inputs": (input_ids, keyword_id, num_feats),
        "target": target,
    }
    if "id" in batch[0]:
        out["id"] = torch.stack([b["id"] for b in batch], dim=0)
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MultiHeadAttentionPooling(nn.Module):
    """Attention pooling that ignores padding positions."""

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = int(n_heads)
        self.attn = nn.Linear(dim, self.n_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D); mask: (B, L) bool, True for real tokens
        scores = self.attn(x)  # (B, L, H)
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask.unsqueeze(-1), neg_inf)
        weights = torch.softmax(scores, dim=1)  # (B, L, H)
        weights = weights.transpose(1, 2)  # (B, H, L)
        pooled = torch.bmm(weights, x)  # (B, H, D)
        return pooled.mean(dim=1)  # (B, D)


class DisasterNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_keywords: int,
        num_numeric: int = NUM_NUMERIC,
        emb_dim: int = 128,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.3,
        max_len: int = MAX_LEN,
    ):
        super().__init__()
        self.max_len = int(max_len)
        self.emb_dim = int(emb_dim)
        self.word_emb = nn.Embedding(vocab_size, self.emb_dim, padding_idx=PAD_IDX)
        self.pos_emb = nn.Embedding(self.max_len, self.emb_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(self.emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=n_heads,
            dim_feedforward=max(hidden_dim * 2, self.emb_dim * 2),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.attn_pool = MultiHeadAttentionPooling(self.emb_dim, n_heads=4)

        kw_dim = 32
        num_dim = 32
        self.keyword_emb = nn.Embedding(num_keywords, kw_dim, padding_idx=0)
        self.num_mlp = nn.Sequential(
            nn.Linear(num_numeric, num_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_dim, num_dim),
        )

        feat_dim = self.emb_dim + kw_dim + num_dim
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        keyword_id: torch.Tensor,
        num_feats: torch.Tensor,
    ) -> torch.Tensor:
        # input_ids: (B, L) long; keyword_id: (B,) long; num_feats: (B, F) float
        B, L = input_ids.shape
        if L > self.max_len:
            input_ids = input_ids[:, : self.max_len]
            L = self.max_len

        mask = input_ids != PAD_IDX  # (B, L) True = real
        if not bool(mask.any(dim=1).all().item()):
            # ensure at least one unmasked position per sample (safety net)
            mask = mask.clone()
            mask[:, 0] = True

        pos = (
            torch.arange(L, device=input_ids.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(B, L)
        )
        x = self.word_emb(input_ids) + self.pos_emb(pos)
        x = self.emb_norm(self.emb_dropout(x))

        pad_mask = ~mask  # True where padding
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        pooled = self.attn_pool(x, mask)

        kw = self.keyword_emb(keyword_id)
        num = self.num_mlp(num_feats)
        feat = torch.cat([pooled, kw, num], dim=-1)
        logit = self.head(feat).squeeze(-1)  # (B,)
        return logit


# ---------------------------------------------------------------------------
# Loss / criterion
# ---------------------------------------------------------------------------
class FocalBCELoss(nn.Module):
    """Focal BCE-with-logits: strong on imbalanced binary text classification."""

    def __init__(
        self,
        gamma: float = 1.5,
        pos_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.02,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        if self.label_smoothing > 0.0:
            smooth_targets = (
                targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            )
        else:
            smooth_targets = targets
        bce = F.binary_cross_entropy_with_logits(
            logits, smooth_targets, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * smooth_targets + (1.0 - p) * (1.0 - smooth_targets)
        focal = (1.0 - p_t).clamp(min=1e-6) ** self.gamma
        return (bce * focal).mean()


# ---------------------------------------------------------------------------
# Threshold search helper (F1)
# ---------------------------------------------------------------------------
def find_best_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (probs >= float(t)).astype(int)
        f1 = f1_score(targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Data loading helper (fit vocab on TRAIN split only)
# ---------------------------------------------------------------------------
def load_all_data(
    input_dir: str = INPUT_DIR_DEFAULT,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
    min_freq: int = MIN_FREQ,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Vocab, Dict[str, int]]:
    train_path = os.path.join(input_dir, "train.csv")
    test_path = os.path.join(input_dir, "test.csv")

    full_train = pd.read_csv(train_path)
    if "text" in full_train.columns:
        full_train = full_train.dropna(subset=["text"]).reset_index(drop=True)
    if "target" not in full_train.columns:
        raise ValueError("train.csv must contain a 'target' column")

    test_df = pd.read_csv(test_path).reset_index(drop=True)

    tr, va = train_test_split(
        full_train,
        test_size=float(val_fraction),
        stratify=full_train["target"].astype(int).values,
        random_state=int(seed),
    )
    tr = tr.reset_index(drop=True)
    va = va.reset_index(drop=True)

    vocab = Vocab(min_freq=min_freq).fit(tr["text"].fillna("").astype(str).tolist())

    kw_lower = (
        tr["keyword"].fillna("").astype(str).str.lower().str.strip()
        if "keyword" in tr.columns
        else pd.Series([], dtype=str)
    )
    kw_set = sorted({k for k in kw_lower.unique().tolist() if k})
    keyword2idx: Dict[str, int] = {
        k: i + 1 for i, k in enumerate(kw_set)
    }  # 0 = no keyword

    return tr, va, test_df, vocab, keyword2idx


def build_datasets(
    tr: pd.DataFrame,
    va: pd.DataFrame,
    test_df: pd.DataFrame,
    vocab: Vocab,
    keyword2idx: Dict[str, int],
    max_len: int = MAX_LEN,
) -> Tuple[TweetDataset, TweetDataset, TweetDataset]:
    ds_tr = TweetDataset(tr, vocab, keyword2idx, max_len=max_len, has_target=True)
    ds_va = TweetDataset(va, vocab, keyword2idx, max_len=max_len, has_target=True)
    ds_te = TweetDataset(test_df, vocab, keyword2idx, max_len=max_len, has_target=False)
    return ds_tr, ds_va, ds_te


# ---------------------------------------------------------------------------
# CPU preflight adapter
# ---------------------------------------------------------------------------
class CandidateAdapter:
    """No-argument adapter used by the CPU preflight checker.

    Exposes the real model, real loss, real CSV-derived batches, and the
    optimizer/view used by the training pipeline.
    """

    def __init__(self):
        self.defaults: Dict[str, object] = {
            "emb_dim": 128,
            "hidden_dim": 128,
            "n_layers": 2,
            "n_heads": 4,
            "dropout": 0.3,
            "max_len": MAX_LEN,
        }
        self._vocab_cache: Optional[Vocab] = None
        self._keyword_cache: Optional[Dict[str, int]] = None
        self._df_cache: Optional[pd.DataFrame] = None

    # --- internal helpers -------------------------------------------------
    def _resolve_input_dir(self) -> str:
        return os.environ.get("MLEVOLVE_INPUT_DIR", INPUT_DIR_DEFAULT)

    def _load_caches(self) -> None:
        if self._vocab_cache is not None and self._keyword_cache is not None:
            return
        input_dir = self._resolve_input_dir()
        train_path = os.path.join(input_dir, "train.csv")
        df = pd.read_csv(train_path)
        if "text" in df.columns:
            df = df.dropna(subset=["text"]).reset_index(drop=True)
        # Fit vocab + keyword index on the FULL training file here — this is
        # only the preflight adapter path, not the pipeline train/val split.
        vocab = Vocab(min_freq=MIN_FREQ).fit(df["text"].fillna("").astype(str).tolist())
        if "keyword" in df.columns:
            kw_lower = df["keyword"].fillna("").astype(str).str.lower().str.strip()
        else:
            kw_lower = pd.Series([], dtype=str)
        kw_set = sorted({k for k in kw_lower.unique().tolist() if k})
        keyword2idx = {k: i + 1 for i, k in enumerate(kw_set)}
        self._vocab_cache = vocab
        self._keyword_cache = keyword2idx
        self._df_cache = df

    def _merged_context(
        self, context: Optional[Dict[str, object]]
    ) -> Dict[str, object]:
        ctx = dict(self.defaults)
        if context:
            for k, v in dict(context).items():
                if v is not None:
                    ctx[k] = v
        return ctx

    # --- contract ---------------------------------------------------------
    def build_model(self, context) -> nn.Module:
        ctx = self._merged_context(context)
        self._load_caches()
        vocab = self._vocab_cache
        keyword2idx = self._keyword_cache
        assert vocab is not None and keyword2idx is not None
        model = DisasterNet(
            vocab_size=len(vocab),
            num_keywords=len(keyword2idx) + 1,
            num_numeric=NUM_NUMERIC,
            emb_dim=int(ctx.get("emb_dim", 128)),
            hidden_dim=int(ctx.get("hidden_dim", 128)),
            n_layers=int(ctx.get("n_layers", 2)),
            n_heads=int(ctx.get("n_heads", 4)),
            dropout=float(ctx.get("dropout", 0.3)),
            max_len=int(ctx.get("max_len", MAX_LEN)),
        )
        return model

    def build_optimizer(self, model, context):
        ctx = self._merged_context(context)
        lr = float(ctx.get("lr", 1e-3))
        wd = float(ctx.get("weight_decay", 0.01))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def _make_batch(self, bs: int, device: torch.device) -> Dict[str, object]:
        self._load_caches()
        vocab = self._vocab_cache
        keyword2idx = self._keyword_cache
        df = self._df_cache
        assert vocab is not None and keyword2idx is not None and df is not None

        bs = max(1, int(bs))
        sub = df.head(bs)
        max_len = int(self.defaults["max_len"])
        ids_list: List[List[int]] = []
        kw_list: List[int] = []
        num_list: List[np.ndarray] = []
        tgt_list: List[float] = []
        for _, row in sub.iterrows():
            text = row["text"] if isinstance(row.get("text", None), str) else ""
            kw = row["keyword"] if isinstance(row.get("keyword", None), str) else ""
            loc = row["location"] if isinstance(row.get("location", None), str) else ""
            ids_list.append(vocab.encode(text, max_len))
            kw_list.append(keyword2idx.get(kw.lower().strip(), 0))
            num_list.append(extract_numeric_features(text, kw, loc))
            if "target" in row:
                tgt_list.append(float(row["target"]))
            else:
                tgt_list.append(0.0)

        input_ids = torch.tensor(ids_list, dtype=torch.long, device=device)
        keyword_id = torch.tensor(kw_list, dtype=torch.long, device=device)
        num_feats = torch.tensor(
            np.stack(num_list, axis=0), dtype=torch.float32, device=device
        )
        target = torch.tensor(tgt_list, dtype=torch.float32, device=device)
        return {"inputs": (input_ids, keyword_id, num_feats), "target": target}

    def build_train_batch(self, scenario, device):
        bs = int(scenario.get("batch_size", 8)) if isinstance(scenario, dict) else 8
        return self._make_batch(bs, device)

    def build_validation_batch(self, scenario, device):
        bs = int(scenario.get("batch_size", 8)) if isinstance(scenario, dict) else 8
        return self._make_batch(bs, device)

    def _resolve_criterion(self, context) -> nn.Module:
        crit = None
        if context:
            crit = dict(context).get("criterion")
        if crit is None:
            crit = FocalBCELoss()
        return crit

    def training_step(self, model, batch, context):
        criterion = self._resolve_criterion(context)
        inputs = batch["inputs"]
        target = batch["target"]
        logits = model(*inputs)
        loss = criterion(logits, target)
        return loss

    def validation_step(self, model, batch, context):
        criterion = self._resolve_criterion(context)
        inputs = batch["inputs"]
        target = batch["target"]
        was_training = model.training
        model.eval()
        with torch.no_grad():
            logits = model(*inputs)
            loss = criterion(logits, target)
        if was_training:
            model.train()
        return {"loss": loss, "logits": logits, "target": target}


if __name__ == "__main__":
    # Full training / validation / submission pipeline is owned by later stages.
    pass

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
 Summary: 1 total attempts - 1 executed, 0 successful (all failed or buggy)

-------------------------------
Attempt #1:
Design: I merged the three stages into one script: Stage 1 provides data prep, tokenizer, dataset, `DisasterTransformer`, criterion, threshold search, and `CandidateAdapter`; Stage 2 adds `PrecisionConfig`/`resolve_precision`/`PrecisionRuntime`/autocast+scaler plumbing and TF32-off policy; Stage 3 wraps these into the full train/validate/test loop with scheduler hooks, diagnostics, AdamW + warmup-linear schedule, best-checkpoint restore, and submission writing at `./submission/submission.csv`. I deduplicated imports and the single shared `MODEL_FAMILY` constant, and kept all side effects under `if __name__ == "__main__":` with `Final Validation Score:` as the last line.
Results: The implementation of this design has bugs.
Insight: Using a different approach may not result in the same bugs as the above approach.


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
{
  "model_design": {
    "modality": "text",
    "target_type": "classification",
    "shape_constraints": [
      "sequence length and tokenizer output must stay configurable"
    ],
    "family": "available_transformer_or_sequence_baseline",
    "alternatives_considered": [],
    "loss": "cross entropy or metric-compatible classification loss",
    "output_interface": "match the competition metric and required submission format",
    "reason": "Use a baseline-compatible model family until stronger task or evidence signals are available.",
    "hardware_fit": "use only compatible evidence"
  },
  "datatype_precision": {
    "precision_policy": "disabled",
    "precision_model_adaptation": "none",
    "fallback_policy": "fall back to fp32 or disabled AMP if the selected precision path is unsupported",
    "reason": "Use conservative precision until hardware and package support are confirmed."
  },
  "training_evaluation": {
    "optimizer": "AdamW for neural models or the library default for tree/linear baselines",
    "scheduler": "none",
    "batch_size_policy": "fixed",
    "dataloader_policy": "safe defaults; enable workers and pinned memory only when compatible",
    "fallbacks": [
      "reduce physical batch size on OOM",
      "reduce epochs before changing model family on timeout"
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
      "predictor/graph evidence not available"
    ]
  }
}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=## Plan **WHAT**: Tweets are short, noisy, informal text with strong keyword signals (e.g., "wildfire", "derailment") and distinct distributional patterns between real disasters and figurative/hyperbolic uses of disas...; purpose=Define the Stage 1 candidate target for later precision and training stages.; hardware_keys=max_len, batch_size, emb_dim, n_layers, load_all_data, MLEVOLVE_INPUT_DIR
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=## Plan **Datatype/precision step**: I'll lock down a conservative precision tier for the Stage 1 hybrid word-level Transformer (FP32 default, FP16 AMP as an *opt-in* policy) that the training stage can consume.; purpose=Support the Stage 1 candidate target with a compatible precision policy.; hardware_keys=max_len

# Previous steps
### model_design
**Plan:** ## Plan

**WHAT**: Tweets are short, noisy, informal text with strong keyword signals (e.g., "wildfire", "derailment") and distinct distributional patterns between real disasters and figurative/hyperbolic uses of disaster words. Location fields are extremely sparse/noisy, and tweet text length varies from a few tokens to a paragraph. Binary F1 on ~6k training samples is the metric, so both ranking quality and threshold calibration matter.

**WHY**: I'll build a **hybrid from-scratch classifier** that jointly learns token-level interactions (word-level Transformer encoder with padding-masked multi-head attention pooling) and non-textual cues (a dedicated keyword embedding + hand-crafted numeric features like caps ratio, digit/URL counts, hashtag presence). This is deliberately novel relative to prior attempts: it avoids dependency on any potentially-unavailable pretrained HF checkpoint (offline constraint), while still capturing the task-specific structure that generic bag-of-words baselines miss — namely the interaction between *which* disaster keyword appears and *how* it is used (caps, exclamations, URL presence, hashtag patterns). Training-set-only vocab construction and a stratified seed-42 split mirror the benchmark.

**HOW**: Confirmation = the hybrid model achieves F1 clearly above a keyword-only or TF-IDF logistic baseline on the held-out split when its threshold is tuned post-hoc. Reconsideration trigger = if validation F1 collapses near the class prior (~0.57) or plateaus at the level of the keyword indicator alone, then the from-scratch transformer is not extracting signal beyond the keyword and I'd fall back to a subword/character-augmented variant.

**Memory/backend notes**: Memory strategy = fixed-shape `max_len` sequence tensor (configurable) plus small dense feature vectors, so batches are homogeneous and CUDA-friendly; no gradient accumulation policy is chosen here (Stage 3 handles it). Elasticity = `batch_size` is honored from `scenario`/`context` by the adapter. Operator/normalization: `LayerNorm` pre-norm transformer blocks + dropout; embedding dim and layer count stay configurable. Fallbacks = `max_len`, `emb_dim`, `n_layers` all configurable for OOM/timeout. Scheduler-owned controls (daemon, device assignment, thread caps, memory limits) are **not** implemented here.

**Note board**:
- baseline_change: Replace a generic sequence baseline with a hybrid word-level Transformer encoder (pre-norm, padding-masked, attention-pooled) fused with a keyword embedding and 10 hand-crafted numeric features, all trained from scratch (no external weights). | purpose: matches the text modality, removes offline-checkpoint risk, and lets the model learn keyword-usage interactions that drive real-vs-figurative detection for F1. | hardware_keys: none
- baseline_change: Fit tokenizer/vocab/keyword-index ONLY on the training split; keep `max_len` and model width configurable and expose a leak-free `load_all_data` returning (train, val, test, vocab, keyword2idx). | purpose: prevents leakage into validation/test and gives Stage 2/3 a stable, configurable tensor contract. | hardware_keys: none
- baseline_change: Expose `CandidateAdapter` with real forward/loss/batch builders using real CSV rows from `MLEVOLVE_INPUT_DIR`. | purpose: satisfies the CPU preflight contract while exercising the actual model + Focal-BCE loss. | hardware_keys: none
**Code:**
```python
import os
import re
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Top-level constants (import-safe)
# ---------------------------------------------------------------------------
MODEL_FAMILY = "disaster_hybrid_transformer_v1"

INPUT_DIR_DEFAULT = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"

SEED = 42
VAL_FRACTION = 0.20
MAX_LEN = 48
MIN_FREQ = 2
PAD_IDX = 0
UNK_IDX = 1
NUM_NUMERIC = 10


# ---------------------------------------------------------------------------
# Text cleaning / tokenization
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s\-\'!?]")
_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = _URL_RE.sub(" url ", t)
    t = _MENTION_RE.sub(" user ", t)
    t = _HASHTAG_RE.sub(r" \1 ", t)
    t = _NON_ALNUM_RE.sub(" ", t)
    t = _SPACE_RE.sub(" ", t).strip()
    return t


def tokenize(text: str) -> List[str]:
    return clean_text(text).split()


def extract_numeric_features(text: str, keyword: str, location: str) -> np.ndarray:
    t = text if isinstance(text, str) else ""
    k = keyword if isinstance(keyword, str) else ""
    l = location if isinstance(location, str) else ""
    n_char = float(len(t))
    n_word = float(len(t.split()))
    n_upper = sum(1 for c in t if c.isupper())
    upper_ratio = n_upper / max(n_char, 1.0)
    n_digit = sum(1 for c in t if c.isdigit())
    digit_ratio = n_digit / max(n_char, 1.0)
    n_exclam = float(t.count("!"))
    n_question = float(t.count("?"))
    n_hashtag = float(t.count("#"))
    n_url = 1.0 if ("http" in t.lower()) else 0.0
    has_keyword = 1.0 if k.strip() else 0.0
    has_location = 1.0 if l.strip() else 0.0
    return np.asarray(
        [
            n_char,
            n_word,
            upper_ratio,
            digit_ratio,
            n_exclam,
            n_question,
            n_hashtag,
            n_url,
            has_keyword,
            has_location,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Vocabulary (fit on TRAIN only)
# ---------------------------------------------------------------------------
class Vocab:
    def __init__(self, min_freq: int = MIN_FREQ):
        self.min_freq = int(min_freq)
        self.token2idx: Dict[str, int] = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}

    def fit(self, texts: List[str]) -> "Vocab":
        counter: Dict[str, int] = {}
        for t in texts:
            for tok in tokenize(t):
                counter[tok] = counter.get(tok, 0) + 1
        for tok, cnt in counter.items():
            if cnt >= self.min_freq and tok not in self.token2idx:
                self.token2idx[tok] = len(self.token2idx)
        return self

    def encode(self, text: str, max_len: int = MAX_LEN) -> List[int]:
        toks = tokenize(text)
        if not toks:
            toks = ["<unk>"]
        ids = [self.token2idx.get(tok, UNK_IDX) for tok in toks[:max_len]]
        if len(ids) < max_len:
            ids = ids + [PAD_IDX] * (max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.token2idx)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TweetDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocab,
        keyword2idx: Dict[str, int],
        max_len: int = MAX_LEN,
        has_target: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.keyword2idx = keyword2idx
        self.max_len = int(max_len)
        self.has_target = bool(has_target)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = row["text"] if "text" in row and isinstance(row["text"], str) else ""
        kw = (
            row["keyword"]
            if "keyword" in row and isinstance(row["keyword"], str)
            else ""
        )
        loc = (
            row["location"]
            if "location" in row and isinstance(row["location"], str)
            else ""
        )

        input_ids = self.vocab.encode(text, self.max_len)
        kw_key = kw.lower().strip()
        kw_id = self.keyword2idx.get(kw_key, 0)
        num_feats = extract_numeric_features(text, kw, loc)

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "keyword_id": torch.tensor(kw_id, dtype=torch.long),
            "num_feats": torch.tensor(num_feats, dtype=torch.float32),
        }
        if self.has_target and "target" in row:
            item["target"] = torch.tensor(float(row["target"]), dtype=torch.float32)
        else:
            item["target"] = torch.tensor(0.0, dtype=torch.float32)
        if "id" in row:
            item["id"] = torch.tensor(int(row["id"]), dtype=torch.long)
        return item


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    keyword_id = torch.stack([b["keyword_id"] for b in batch], dim=0)
    num_feats = torch.stack([b["num_feats"] for b in batch], dim=0)
    target = torch.stack([b["target"] for b in batch], dim=0)
    out: Dict[str, object] = {
        "inputs": (input_ids, keyword_id, num_feats),
        "target": target,
    }
    if "id" in batch[0]:
        out["id"] = torch.stack([b["id"] for b in batch], dim=0)
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MultiHeadAttentionPooling(nn.Module):
    """Attention pooling that ignores padding positions."""

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = int(n_heads)
        self.attn = nn.Linear(dim, self.n_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D); mask: (B, L) bool, True for real tokens
        scores = self.attn(x)  # (B, L, H)
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask.unsqueeze(-1), neg_inf)
        weights = torch.softmax(scores, dim=1)  # (B, L, H)
        weights = weights.transpose(1, 2)  # (B, H, L)
        pooled = torch.bmm(weights, x)  # (B, H, D)
        return pooled.mean(dim=1)  # (B, D)


class DisasterNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_keywords: int,
        num_numeric: int = NUM_NUMERIC,
        emb_dim: int = 128,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.3,
        max_len: int = MAX_LEN,
    ):
        super().__init__()
        self.max_len = int(max_len)
        self.emb_dim = int(emb_dim)
        self.word_emb = nn.Embedding(vocab_size, self.emb_dim, padding_idx=PAD_IDX)
        self.pos_emb = nn.Embedding(self.max_len, self.emb_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(self.emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=n_heads,
            dim_feedforward=max(hidden_dim * 2, self.emb_dim * 2),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.attn_pool = MultiHeadAttentionPooling(self.emb_dim, n_heads=4)

        kw_dim = 32
        num_dim = 32
        self.keyword_emb = nn.Embedding(num_keywords, kw_dim, padding_idx=0)
        self.num_mlp = nn.Sequential(
            nn.Linear(num_numeric, num_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_dim, num_dim),
        )

        feat_dim = self.emb_dim + kw_dim + num_dim
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        keyword_id: torch.Tensor,
        num_feats: torch.Tensor,
    ) -> torch.Tensor:
        # input_ids: (B, L) long; keyword_id: (B,) long; num_feats: (B, F) float
        B, L = input_ids.shape
        if L > self.max_len:
            input_ids = input_ids[:, : self.max_len]
            L = self.max_len

        mask = input_ids != PAD_IDX  # (B, L) True = real
        if not bool(mask.any(dim=1).all().item()):
            # ensure at least one unmasked position per sample (safety net)
            mask = mask.clone()
            mask[:, 0] = True

        pos = (
            torch.arange(L, device=input_ids.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(B, L)
        )
        x = self.word_emb(input_ids) + self.pos_emb(pos)
        x = self.emb_norm(self.emb_dropout(x))

        pad_mask = ~mask  # True where padding
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        pooled = self.attn_pool(x, mask)

        kw = self.keyword_emb(keyword_id)
        num = self.num_mlp(num_feats)
        feat = torch.cat([pooled, kw, num], dim=-1)
        logit = self.head(feat).squeeze(-1)  # (B,)
        return logit


# ---------------------------------------------------------------------------
# Loss / criterion
# ---------------------------------------------------------------------------
class FocalBCELoss(nn.Module):
    """Focal BCE-with-logits: strong on imbalanced binary text classification."""

    def __init__(
        self,
        gamma: float = 1.5,
        pos_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.02,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        if self.label_smoothing > 0.0:
            smooth_targets = (
                targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            )
        else:
            smooth_targets = targets
        bce = F.binary_cross_entropy_with_logits(
            logits, smooth_targets, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * smooth_targets + (1.0 - p) * (1.0 - smooth_targets)
        focal = (1.0 - p_t).clamp(min=1e-6) ** self.gamma
        return (bce * focal).mean()


# ---------------------------------------------------------------------------
# Threshold search helper (F1)
# ---------------------------------------------------------------------------
def find_best_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (probs >= float(t)).astype(int)
        f1 = f1_score(targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Data loading helper (fit vocab on TRAIN split only)
# ---------------------------------------------------------------------------
def load_all_data(
    input_dir: str = INPUT_DIR_DEFAULT,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
    min_freq: int = MIN_FREQ,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Vocab, Dict[str, int]]:
    train_path = os.path.join(input_dir, "train.csv")
    test_path = os.path.join(input_dir, "test.csv")

    full_train = pd.read_csv(train_path)
    if "text" in full_train.columns:
        full_train = full_train.dropna(subset=["text"]).reset_index(drop=True)
    if "target" not in full_train.columns:
        raise ValueError("train.csv must contain a 'target' column")

    test_df = pd.read_csv(test_path).reset_index(drop=True)

    tr, va = train_test_split(
        full_train,
        test_size=float(val_fraction),
        stratify=full_train["target"].astype(int).values,
        random_state=int(seed),
    )
    tr = tr.reset_index(drop=True)
    va = va.reset_index(drop=True)

    vocab = Vocab(min_freq=min_freq).fit(tr["text"].fillna("").astype(str).tolist())

    kw_lower = (
        tr["keyword"].fillna("").astype(str).str.lower().str.strip()
        if "keyword" in tr.columns
        else pd.Series([], dtype=str)
    )
    kw_set = sorted({k for k in kw_lower.unique().tolist() if k})
    keyword2idx: Dict[str, int] = {
        k: i + 1 for i, k in enumerate(kw_set)
    }  # 0 = no keyword

    return tr, va, test_df, vocab, keyword2idx


def build_datasets(
    tr: pd.DataFrame,
    va: pd.DataFrame,
    test_df: pd.DataFrame,
    vocab: Vocab,
    keyword2idx: Dict[str, int],
    max_len: int = MAX_LEN,
) -> Tuple[TweetDataset, TweetDataset, TweetDataset]:
    ds_tr = TweetDataset(tr, vocab, keyword2idx, max_len=max_len, has_target=True)
    ds_va = TweetDataset(va, vocab, keyword2idx, max_len=max_len, has_target=True)
    ds_te = TweetDataset(test_df, vocab, keyword2idx, max_len=max_len, has_target=False)
    return ds_tr, ds_va, ds_te


# ---------------------------------------------------------------------------
# CPU preflight adapter
# ---------------------------------------------------------------------------
class CandidateAdapter:
    """No-argument adapter used by the CPU preflight checker.

    Exposes the real model, real loss, real CSV-derived batches, and the
    optimizer/view used by the training pipeline.
    """

    def __init__(self):
        self.defaults: Dict[str, object] = {
            "emb_dim": 128,
            "hidden_dim": 128,
            "n_layers": 2,
            "n_heads": 4,
            "dropout": 0.3,
            "max_len": MAX_LEN,
        }
        self._vocab_cache: Optional[Vocab] = None
        self._keyword_cache: Optional[Dict[str, int]] = None
        self._df_cache: Optional[pd.DataFrame] = None

    # --- internal helpers -------------------------------------------------
    def _resolve_input_dir(self) -> str:
        return os.environ.get("MLEVOLVE_INPUT_DIR", INPUT_DIR_DEFAULT)

    def _load_caches(self) -> None:
        if self._vocab_cache is not None and self._keyword_cache is not None:
            return
        input_dir = self._resolve_input_dir()
        train_path = os.path.join(input_dir, "train.csv")
        df = pd.read_csv(train_path)
        if "text" in df.columns:
            df = df.dropna(subset=["text"]).reset_index(drop=True)
        # Fit vocab + keyword index on the FULL training file here — this is
        # only the preflight adapter path, not the pipeline train/val split.
        vocab = Vocab(min_freq=MIN_FREQ).fit(df["text"].fillna("").astype(str).tolist())
        if "keyword" in df.columns:
            kw_lower = df["keyword"].fillna("").astype(str).str.lower().str.strip()
        else:
            kw_lower = pd.Series([], dtype=str)
        kw_set = sorted({k for k in kw_lower.unique().tolist() if k})
        keyword2idx = {k: i + 1 for i, k in enumerate(kw_set)}
        self._vocab_cache = vocab
        self._keyword_cache = keyword2idx
        self._df_cache = df

    def _merged_context(
        self, context: Optional[Dict[str, object]]
    ) -> Dict[str, object]:
        ctx = dict(self.defaults)
        if context:
            for k, v in dict(context).items():
                if v is not None:
                    ctx[k] = v
        return ctx

    # --- contract ---------------------------------------------------------
    def build_model(self, context) -> nn.Module:
        ctx = self._merged_context(context)
        self._load_caches()
        vocab = self._vocab_cache
        keyword2idx = self._keyword_cache
        assert vocab is not None and keyword2idx is not None
        model = DisasterNet(
            vocab_size=len(vocab),
            num_keywords=len(keyword2idx) + 1,
            num_numeric=NUM_NUMERIC,
            emb_dim=int(ctx.get("emb_dim", 128)),
            hidden_dim=int(ctx.get("hidden_dim", 128)),
            n_layers=int(ctx.get("n_layers", 2)),
            n_heads=int(ctx.get("n_heads", 4)),
            dropout=float(ctx.get("dropout", 0.3)),
            max_len=int(ctx.get("max_len", MAX_LEN)),
        )
        return model

    def build_optimizer(self, model, context):
        ctx = self._merged_context(context)
        lr = float(ctx.get("lr", 1e-3))
        wd = float(ctx.get("weight_decay", 0.01))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def _make_batch(self, bs: int, device: torch.device) -> Dict[str, object]:
        self._load_caches()
        vocab = self._vocab_cache
        keyword2idx = self._keyword_cache
        df = self._df_cache
        assert vocab is not None and keyword2idx is not None and df is not None

        bs = max(1, int(bs))
        sub = df.head(bs)
        max_len = int(self.defaults["max_len"])
        ids_list: List[List[int]] = []
        kw_list: List[int] = []
        num_list: List[np.ndarray] = []
        tgt_list: List[float] = []
        for _, row in sub.iterrows():
            text = row["text"] if isinstance(row.get("text", None), str) else ""
            kw = row["keyword"] if isinstance(row.get("keyword", None), str) else ""
            loc = row["location"] if isinstance(row.get("location", None), str) else ""
            ids_list.append(vocab.encode(text, max_len))
            kw_list.append(keyword2idx.get(kw.lower().strip(), 0))
            num_list.append(extract_numeric_features(text, kw, loc))
            if "target" in row:
                tgt_list.append(float(row["target"]))
            else:
                tgt_list.append(0.0)

        input_ids = torch.tensor(ids_list, dtype=torch.long, device=device)
        keyword_id = torch.tensor(kw_list, dtype=torch.long, device=device)
        num_feats = torch.tensor(
            np.stack(num_list, axis=0), dtype=torch.float32, device=device
        )
        target = torch.tensor(tgt_list, dtype=torch.float32, device=device)
        return {"inputs": (input_ids, keyword_id, num_feats), "target": target}

    def build_train_batch(self, scenario, device):
        bs = int(scenario.get("batch_size", 8)) if isinstance(scenario, dict) else 8
        return self._make_batch(bs, device)

    def build_validation_batch(self, scenario, device):
        bs = int(scenario.get("batch_size", 8)) if isinstance(scenario, dict) else 8
        return self._make_batch(bs, device)

    def _resolve_criterion(self, context) -> nn.Module:
        crit = None
        if context:
            crit = dict(context).get("criterion")
        if crit is None:
            crit = FocalBCELoss()
        return crit

    def training_step(self, model, batch, context):
        criterion = self._resolve_criterion(context)
        inputs = batch["inputs"]
        target = batch["target"]
        logits = model(*inputs)
        loss = criterion(logits, target)
        return loss

    def validation_step(self, model, batch, context):
        criterion = self._resolve_criterion(context)
        inputs = batch["inputs"]
        target = batch["target"]
        was_training = model.training
        model.eval()
        with torch.no_grad():
            logits = model(*inputs)
            loss = criterion(logits, target)
        if was_training:
            model.train()
        return {"loss": loss, "logits": logits, "target": target}


if __name__ == "__main__":
    # Full training / validation / submission pipeline is owned by later stages.
    pass

```

### datatype_precision
**Plan:** ## Plan

**Datatype/precision step**: I'll lock down a conservative precision tier for the Stage 1 hybrid word-level Transformer (FP32 default, FP16 AMP as an *opt-in* policy) that the training stage can consume. The Stage 1 candidate already keeps Focal-BCE numerically robust (it explicitly casts logits/targets to `float32`), so the autocast region is confined to the Transformer forward path while the loss, metric accumulation, and prediction export stay FP32. Backward/optimizer run outside autocast with GradScaler only when FP16 is actually active; TF32 is force-disabled and CPU preflight is guaranteed FP32 with scaling off.

**WHAT/WHY/HOW**: The competition is short-text binary classification with F1 — where a single mis-scaled logit near the threshold flips precision/recall heavily — so the safe move on an A10 (compute capability 8.6, fp16-safe tensor cores) is selective FP16 autocast *around* the encoder, FP32 elsewhere. This is only enabled when the caller explicitly requests it (default "disabled"), matching the decision trace's `precision_policy="disabled"` and the HWDB `confidence=0.0` (no verified speedup evidence). Confirmation = finite loss/grads, F1 within noise of FP32, and non-regressed epoch time; reconsider trigger = any NaN/Inf grad, unstable early-epoch loss, or slower wall-time → fallback to FP32. Memory strategy is unchanged from Stage 1 (fixed-shape `max_len` tensor); elasticity is unchanged (batch_size from caller); no operator/normalization redesign — only autocast region and TF32/scaler toggles are added. Scheduler-owned controls (device assignment, thread caps, memory limits, daemon) are explicitly *not* implemented here.
**Code:**
```python
# =============================================================================
# Stage 2 — datatype / precision policy
# -----------------------------------------------------------------------------
# Import-safe: defines dataclasses, resolvers and helpers only.
# No device selection, no model construction, no training, no side effects.
# Consumed by Stage 3 (training_evaluation) and by CandidateAdapter.
# =============================================================================

import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

# ---------------------------------------------------------------------------
# Top-level policy defaults (conservative; decision trace: precision_policy="disabled")
# ---------------------------------------------------------------------------
PRECISION_POLICY_DEFAULT: str = "disabled"  # "disabled" (fp32) | "fp16_amp"
AMP_DTYPE_NAME_DEFAULT: str = "float16"
USE_TF32_DEFAULT: bool = False  # forced off in normal mode
DEVICE_OVERRIDE_DEFAULT: Optional[str] = None  # let caller/scheduler choose


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class PrecisionConfig:
    """Caller-facing knobs. All optional; merged over defaults in resolve_precision."""

    policy: str = PRECISION_POLICY_DEFAULT
    amp_dtype: str = AMP_DTYPE_NAME_DEFAULT
    use_tf32: bool = USE_TF32_DEFAULT
    device: Optional[str] = DEVICE_OVERRIDE_DEFAULT
    fallback_to_fp32: bool = True
    reason: Optional[str] = None


@dataclass
class PrecisionRuntime:
    """Resolved, immutable-ish precision state consumed by training/eval code."""

    device: torch.device
    policy: str  # "fp32" or "fp16_amp"
    amp_enabled: bool
    amp_dtype: torch.dtype
    use_tf32: bool
    scaler_enabled: bool
    fallback_to_fp32: bool
    reason: str
    notes: Dict[str, Any] = field(default_factory=dict)

    # -- context managers ---------------------------------------------------
    def autocast(self):
        """Autocast around eligible forward ops. No-op when AMP disabled."""
        if self.amp_enabled and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        return contextlib.nullcontext()

    def autocast_off(self):
        """Explicit FP32 island — use inside autocast for fragile reductions/loss."""
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", enabled=False)
        return contextlib.nullcontext()

    # -- scaler -------------------------------------------------------------
    def make_grad_scaler(self):
        """Return a GradScaler for FP16 or None. Uses new torch.amp API when available."""
        if not self.scaler_enabled or self.device.type != "cuda":
            return None
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            # Older torch fallback
            try:
                return torch.cuda.amp.GradScaler(enabled=True)
            except Exception:
                return None


# ---------------------------------------------------------------------------
# Device resolution (no CUDA calls at import time)
# ---------------------------------------------------------------------------
def resolve_device(explicit: Optional[str] = None) -> torch.device:
    """Resolve target device. Scheduler-owned assignment is honored when given."""
    if explicit:
        try:
            return torch.device(explicit)
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Precision resolution
# ---------------------------------------------------------------------------
def _normalize_policy(policy: Optional[str]) -> str:
    if policy is None:
        return PRECISION_POLICY_DEFAULT
    p = str(policy).strip().lower()
    if p in ("fp16_amp", "fp16", "amp", "fp16-amp"):
        return "fp16_amp"
    # Anything else (including "fp32", "disabled", unknown) → conservative FP32
    return "disabled"


def resolve_precision(
    context: Optional[Dict[str, Any]] = None,
    config: Optional[PrecisionConfig] = None,
) -> PrecisionRuntime:
    """Resolve a PrecisionRuntime from an optional context mapping (partial-merge).

    Never raises: any unsupported path falls back to FP32 with a recorded note.
    """
    ctx: Dict[str, Any] = dict(context or {})
    cfg = config or PrecisionConfig()

    # Merge config <- context (context wins when present)
    policy_raw = ctx.get("precision_policy", ctx.get("policy", cfg.policy))
    amp_dtype_raw = ctx.get("amp_dtype", cfg.amp_dtype)
    use_tf32_raw = ctx.get("use_tf32", cfg.use_tf32)
    device_raw = ctx.get("device", cfg.device)
    fallback_ok = bool(ctx.get("fallback_to_fp32", cfg.fallback_to_fp32))
    reason_raw = ctx.get("reason", cfg.reason)

    policy = _normalize_policy(policy_raw)
    device = resolve_device(device_raw)
    notes: Dict[str, Any] = {}

    # TF32 is force-off in normal mode regardless of caller ask.
    if bool(use_tf32_raw):
        notes["tf32_forced_off"] = True
    use_tf32 = False

    amp_enabled = False
    amp_dtype = torch.float32
    scaler_enabled = False

    if policy == "fp16_amp":
        if device.type != "cuda":
            notes["amp_disabled"] = "device_not_cuda"
        elif not torch.cuda.is_available():
            notes["amp_disabled"] = "cuda_unavailable"
        elif str(amp_dtype_raw).lower() not in ("float16", "fp16", "half"):
            notes["amp_disabled"] = f"unsupported_amp_dtype:{amp_dtype_raw}"
        else:
            # Verify GPU can actually run FP16 tensor-core autocast.
            try:
                cap_major, _cap_minor = torch.cuda.get_device_capability(device)
                if cap_major >= 7:
                    amp_enabled = True
                    amp_dtype = torch.float16
                    scaler_enabled = True
                else:
                    notes["amp_disabled"] = f"capability_below_7:{cap_major}"
            except Exception as e:  # noqa: BLE001
                notes["amp_disabled"] = f"cap_check_failed:{type(e).__name__}"

    if amp_enabled:
        final_policy = "fp16_amp"
    else:
        final_policy = "fp32"
        amp_dtype = torch.float32
        scaler_enabled = False

    if reason_raw:
        reason = str(reason_raw)
    elif amp_enabled:
        reason = "caller_requested_fp16_amp;verified_cuda_cc>=7"
    else:
        reason = "conservative_fp32_default(no_verified_speedup_evidence)"

    return PrecisionRuntime(
        device=device,
        policy=final_policy,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        use_tf32=use_tf32,
        scaler_enabled=scaler_enabled,
        fallback_to_fp32=bool(fallback_ok),
        reason=reason,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Backend flag application (safe subset; no device placement, no daemon, no
# thread/memory controls owned by the scheduler).
# ---------------------------------------------------------------------------
def apply_precision_backend(precision: PrecisionRuntime) -> None:
    """Apply only precision-related torch.backends flags. Idempotent, non-fatal."""
    # TF32 must be off per normal-mode policy.
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    # Also silence the new fine-grained tf32 toggles when present.
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch.backends.cudnn.conv.fp32_precision = "ieee"  # type: ignore[attr-defined]
    except Exception:
        pass
    # Hint: cuDNN benchmark is a scheduling/timing concern; do not toggle here.
    _ = precision  # reserved for future per-runtime flags


# ---------------------------------------------------------------------------
# Convenience helpers used by the training loop
# ---------------------------------------------------------------------------
def should_use_amp(precision: PrecisionRuntime) -> bool:
    return bool(precision.amp_enabled and precision.device.type == "cuda")


def safe_autocast(precision: PrecisionRuntime):
    """Alias so training code has a single, grep-friendly entrypoint."""
    return precision.autocast()


def fp32_island(precision: PrecisionRuntime):
    """Run a fragile block (loss reduction, metrics, threshold search) in FP32."""
    return precision.autocast_off()


def ensure_fp32(t: torch.Tensor) -> torch.Tensor:
    """Lightweight guard for FP32-sensitive ops (loss/metrics). No-op if already fp32."""
    if not torch.is_floating_point(t):
        return t
    return t if t.dtype == torch.float32 else t.float()


def unscale_if_needed_(optimizer, scaler, model) -> None:
    """Unscale gradients before clipping when running under FP16 GradScaler."""
    if scaler is not None and scaler.is_enabled():
        scaler.unscale_(optimizer)


# ---------------------------------------------------------------------------
# Logging helper (single line; called once by training entrypoint)
# ---------------------------------------------------------------------------
def precision_log_line(precision: PrecisionRuntime) -> str:
    amp_dtype_name = str(precision.amp_dtype).replace("torch.", "")
    return (
        f"[precision] device={precision.device.type} "
        f"policy={precision.policy} dtype={amp_dtype_name} "
        f"amp={precision.amp_enabled} tf32={precision.use_tf32} "
        f"scaler={precision.scaler_enabled} fallback_fp32={precision.fallback_to_fp32} "
        f"reason={precision.reason} notes={precision.notes}"
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
 Summary: 1 total attempts - 1 executed, 0 successful (all failed or buggy)

-------------------------------
Attempt #1:
Design: I merged the three stages into one script: Stage 1 provides data prep, tokenizer, dataset, `DisasterTransformer`, criterion, threshold search, and `CandidateAdapter`; Stage 2 adds `PrecisionConfig`/`resolve_precision`/`PrecisionRuntime`/autocast+scaler plumbing and TF32-off policy; Stage 3 wraps these into the full train/validate/test loop with scheduler hooks, diagnostics, AdamW + warmup-linear schedule, best-checkpoint restore, and submission writing at `./submission/submission.csv`. I deduplicated imports and the single shared `MODEL_FAMILY` constant, and kept all side effects under `if __name__ == "__main__":` with `Final Validation Score:` as the last line.
Results: The implementation of this design has bugs.
Insight: Using a different approach may not result in the same bugs as the above approach.


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
{
  "model_design": {
    "modality": "text",
    "target_type": "classification",
    "shape_constraints": [
      "sequence length and tokenizer output must stay configurable"
    ],
    "family": "available_transformer_or_sequence_baseline",
    "alternatives_considered": [],
    "loss": "cross entropy or metric-compatible classification loss",
    "output_interface": "match the competition metric and required submission format",
    "reason": "Use a baseline-compatible model family until stronger task or evidence signals are available.",
    "hardware_fit": "use only compatible evidence"
  },
  "datatype_precision": {
    "precision_policy": "disabled",
    "precision_model_adaptation": "none",
    "fallback_policy": "fall back to fp32 or disabled AMP if the selected precision path is unsupported",
    "reason": "Use conservative precision until hardware and package support are confirmed."
  },
  "training_evaluation": {
    "optimizer": "AdamW for neural models or the library default for tree/linear baselines",
    "scheduler": "none",
    "batch_size_policy": "fixed",
    "dataloader_policy": "safe defaults; enable workers and pinned memory only when compatible",
    "fallbacks": [
      "reduce physical batch size on OOM",
      "reduce epochs before changing model family on timeout"
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
      "predictor/graph evidence not available"
    ]
  }
}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=## Plan **WHAT**: Tweets are short, noisy, informal text with strong keyword signals (e.g., "wildfire", "derailment") and distinct distributional patterns between real disasters and figurative/hyperbolic uses of disas...; purpose=Define the Stage 1 candidate target for later precision and training stages.; hardware_keys=max_len, batch_size, emb_dim, n_layers, load_all_data, MLEVOLVE_INPUT_DIR
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=## Plan **Datatype/precision step**: I'll lock down a conservative precision tier for the Stage 1 hybrid word-level Transformer (FP32 default, FP16 AMP as an *opt-in* policy) that the training stage can consume.; purpose=Support the Stage 1 candidate target with a compatible precision policy.; hardware_keys=max_len
  - Stage 3 training / training_evaluation: baseline_change=## Plan **WHAT/WHY/HOW**: Stage 3 locks in the hyperparameter envelope for the Stage 1 hybrid word-level Transformer under the Stage 2 conservative precision tier (FP32 default, FP16 AMP opt-in).; purpose=Train and evaluate the Stage 1 candidate using the selected precision policy.; hardware_keys=GRAD_ACCUM_STEPS, clip_grad_norm_, script_scheduler_context

# Step results

        ### Step 1: model_design
        **Plan:** ## Plan

**WHAT**: Tweets are short, noisy, informal text with strong keyword signals (e.g., "wildfire", "derailment") and distinct distributional patterns between real disasters and figurative/hyperbolic uses of disaster words. Location fields are extremely sparse/noisy, and tweet text length varies from a few tokens to a paragraph. Binary F1 on ~6k training samples is the metric, so both ranking quality and threshold calibration matter.

**WHY**: I'll build a **hybrid from-scratch classifier** that jointly learns token-level interactions (word-level Transformer encoder with padding-masked multi-head attention pooling) and non-textual cues (a dedicated keyword embedding + hand-crafted numeric features like caps ratio, digit/URL counts, hashtag presence). This is deliberately novel relative to prior attempts: it avoids dependency on any potentially-unavailable pretrained HF checkpoint (offline constraint), while still capturing the task-specific structure that generic bag-of-words baselines miss — namely the interaction between *which* disaster keyword appears and *how* it is used (caps, exclamations, URL presence, hashtag patterns). Training-set-only vocab construction and a stratified seed-42 split mirror the benchmark.

**HOW**: Confirmation = the hybrid model achieves F1 clearly above a keyword-only or TF-IDF logistic baseline on the held-out split when its threshold is tuned post-hoc. Reconsideration trigger = if validation F1 collapses near the class prior (~0.57) or plateaus at the level of the keyword indicator alone, then the from-scratch transformer is not extracting signal beyond the keyword and I'd fall back to a subword/character-augmented variant.

**Memory/backend notes**: Memory strategy = fixed-shape `max_len` sequence tensor (configurable) plus small dense feature vectors, so batches are homogeneous and CUDA-friendly; no gradient accumulation policy is chosen here (Stage 3 handles it). Elasticity = `batch_size` is honored from `scenario`/`context` by the adapter. Operator/normalization: `LayerNorm` pre-norm transformer blocks + dropout; embedding dim and layer count stay configurable. Fallbacks = `max_len`, `emb_dim`, `n_layers` all configurable for OOM/timeout. Scheduler-owned controls (daemon, device assignment, thread caps, memory limits) are **not** implemented here.

**Note board**:
- baseline_change: Replace a generic sequence baseline with a hybrid word-level Transformer encoder (pre-norm, padding-masked, attention-pooled) fused with a keyword embedding and 10 hand-crafted numeric features, all trained from scratch (no external weights). | purpose: matches the text modality, removes offline-checkpoint risk, and lets the model learn keyword-usage interactions that drive real-vs-figurative detection for F1. | hardware_keys: none
- baseline_change: Fit tokenizer/vocab/keyword-index ONLY on the training split; keep `max_len` and model width configurable and expose a leak-free `load_all_data` returning (train, val, test, vocab, keyword2idx). | purpose: prevents leakage into validation/test and gives Stage 2/3 a stable, configurable tensor contract. | hardware_keys: none
- baseline_change: Expose `CandidateAdapter` with real forward/loss/batch builders using real CSV rows from `MLEVOLVE_INPUT_DIR`. | purpose: satisfies the CPU preflight contract while exercising the actual model + Focal-BCE loss. | hardware_keys: none
        **Code:**
        ```python
import os
import re
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Top-level constants (import-safe)
# ---------------------------------------------------------------------------
MODEL_FAMILY = "disaster_hybrid_transformer_v1"

INPUT_DIR_DEFAULT = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"

SEED = 42
VAL_FRACTION = 0.20
MAX_LEN = 48
MIN_FREQ = 2
PAD_IDX = 0
UNK_IDX = 1
NUM_NUMERIC = 10


# ---------------------------------------------------------------------------
# Text cleaning / tokenization
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s\-\'!?]")
_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = _URL_RE.sub(" url ", t)
    t = _MENTION_RE.sub(" user ", t)
    t = _HASHTAG_RE.sub(r" \1 ", t)
    t = _NON_ALNUM_RE.sub(" ", t)
    t = _SPACE_RE.sub(" ", t).strip()
    return t


def tokenize(text: str) -> List[str]:
    return clean_text(text).split()


def extract_numeric_features(text: str, keyword: str, location: str) -> np.ndarray:
    t = text if isinstance(text, str) else ""
    k = keyword if isinstance(keyword, str) else ""
    l = location if isinstance(location, str) else ""
    n_char = float(len(t))
    n_word = float(len(t.split()))
    n_upper = sum(1 for c in t if c.isupper())
    upper_ratio = n_upper / max(n_char, 1.0)
    n_digit = sum(1 for c in t if c.isdigit())
    digit_ratio = n_digit / max(n_char, 1.0)
    n_exclam = float(t.count("!"))
    n_question = float(t.count("?"))
    n_hashtag = float(t.count("#"))
    n_url = 1.0 if ("http" in t.lower()) else 0.0
    has_keyword = 1.0 if k.strip() else 0.0
    has_location = 1.0 if l.strip() else 0.0
    return np.asarray(
        [
            n_char,
            n_word,
            upper_ratio,
            digit_ratio,
            n_exclam,
            n_question,
            n_hashtag,
            n_url,
            has_keyword,
            has_location,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Vocabulary (fit on TRAIN only)
# ---------------------------------------------------------------------------
class Vocab:
    def __init__(self, min_freq: int = MIN_FREQ):
        self.min_freq = int(min_freq)
        self.token2idx: Dict[str, int] = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}

    def fit(self, texts: List[str]) -> "Vocab":
        counter: Dict[str, int] = {}
        for t in texts:
            for tok in tokenize(t):
                counter[tok] = counter.get(tok, 0) + 1
        for tok, cnt in counter.items():
            if cnt >= self.min_freq and tok not in self.token2idx:
                self.token2idx[tok] = len(self.token2idx)
        return self

    def encode(self, text: str, max_len: int = MAX_LEN) -> List[int]:
        toks = tokenize(text)
        if not toks:
            toks = ["<unk>"]
        ids = [self.token2idx.get(tok, UNK_IDX) for tok in toks[:max_len]]
        if len(ids) < max_len:
            ids = ids + [PAD_IDX] * (max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.token2idx)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TweetDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocab,
        keyword2idx: Dict[str, int],
        max_len: int = MAX_LEN,
        has_target: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.keyword2idx = keyword2idx
        self.max_len = int(max_len)
        self.has_target = bool(has_target)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = row["text"] if "text" in row and isinstance(row["text"], str) else ""
        kw = (
            row["keyword"]
            if "keyword" in row and isinstance(row["keyword"], str)
            else ""
        )
        loc = (
            row["location"]
            if "location" in row and isinstance(row["location"], str)
            else ""
        )

        input_ids = self.vocab.encode(text, self.max_len)
        kw_key = kw.lower().strip()
        kw_id = self.keyword2idx.get(kw_key, 0)
        num_feats = extract_numeric_features(text, kw, loc)

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "keyword_id": torch.tensor(kw_id, dtype=torch.long),
            "num_feats": torch.tensor(num_feats, dtype=torch.float32),
        }
        if self.has_target and "target" in row:
            item["target"] = torch.tensor(float(row["target"]), dtype=torch.float32)
        else:
            item["target"] = torch.tensor(0.0, dtype=torch.float32)
        if "id" in row:
            item["id"] = torch.tensor(int(row["id"]), dtype=torch.long)
        return item


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    keyword_id = torch.stack([b["keyword_id"] for b in batch], dim=0)
    num_feats = torch.stack([b["num_feats"] for b in batch], dim=0)
    target = torch.stack([b["target"] for b in batch], dim=0)
    out: Dict[str, object] = {
        "inputs": (input_ids, keyword_id, num_feats),
        "target": target,
    }
    if "id" in batch[0]:
        out["id"] = torch.stack([b["id"] for b in batch], dim=0)
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MultiHeadAttentionPooling(nn.Module):
    """Attention pooling that ignores padding positions."""

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = int(n_heads)
        self.attn = nn.Linear(dim, self.n_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D); mask: (B, L) bool, True for real tokens
        scores = self.attn(x)  # (B, L, H)
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask.unsqueeze(-1), neg_inf)
        weights = torch.softmax(scores, dim=1)  # (B, L, H)
        weights = weights.transpose(1, 2)  # (B, H, L)
        pooled = torch.bmm(weights, x)  # (B, H, D)
        return pooled.mean(dim=1)  # (B, D)


class DisasterNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_keywords: int,
        num_numeric: int = NUM_NUMERIC,
        emb_dim: int = 128,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.3,
        max_len: int = MAX_LEN,
    ):
        super().__init__()
        self.max_len = int(max_len)
        self.emb_dim = int(emb_dim)
        self.word_emb = nn.Embedding(vocab_size, self.emb_dim, padding_idx=PAD_IDX)
        self.pos_emb = nn.Embedding(self.max_len, self.emb_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(self.emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=n_heads,
            dim_feedforward=max(hidden_dim * 2, self.emb_dim * 2),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.attn_pool = MultiHeadAttentionPooling(self.emb_dim, n_heads=4)

        kw_dim = 32
        num_dim = 32
        self.keyword_emb = nn.Embedding(num_keywords, kw_dim, padding_idx=0)
        self.num_mlp = nn.Sequential(
            nn.Linear(num_numeric, num_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_dim, num_dim),
        )

        feat_dim = self.emb_dim + kw_dim + num_dim
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        keyword_id: torch.Tensor,
        num_feats: torch.Tensor,
    ) -> torch.Tensor:
        # input_ids: (B, L) long; keyword_id: (B,) long; num_feats: (B, F) float
        B, L = input_ids.shape
        if L > self.max_len:
            input_ids = input_ids[:, : self.max_len]
            L = self.max_len

        mask = input_ids != PAD_IDX  # (B, L) True = real
        if not bool(mask.any(dim=1).all().item()):
            # ensure at least one unmasked position per sample (safety net)
            mask = mask.clone()
            mask[:, 0] = True

        pos = (
            torch.arange(L, device=input_ids.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(B, L)
        )
        x = self.word_emb(input_ids) + self.pos_emb(pos)
        x = self.emb_norm(self.emb_dropout(x))

        pad_mask = ~mask  # True where padding
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        pooled = self.attn_pool(x, mask)

        kw = self.keyword_emb(keyword_id)
        num = self.num_mlp(num_feats)
        feat = torch.cat([pooled, kw, num], dim=-1)
        logit = self.head(feat).squeeze(-1)  # (B,)
        return logit


# ---------------------------------------------------------------------------
# Loss / criterion
# ---------------------------------------------------------------------------
class FocalBCELoss(nn.Module):
    """Focal BCE-with-logits: strong on imbalanced binary text classification."""

    def __init__(
        self,
        gamma: float = 1.5,
        pos_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.02,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        if self.label_smoothing > 0.0:
            smooth_targets = (
                targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            )
        else:
            smooth_targets = targets
        bce = F.binary_cross_entropy_with_logits(
            logits, smooth_targets, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * smooth_targets + (1.0 - p) * (1.0 - smooth_targets)
        focal = (1.0 - p_t).clamp(min=1e-6) ** self.gamma
        return (bce * focal).mean()


# ---------------------------------------------------------------------------
# Threshold search helper (F1)
# ---------------------------------------------------------------------------
def find_best_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (probs >= float(t)).astype(int)
        f1 = f1_score(targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Data loading helper (fit vocab on TRAIN split only)
# ---------------------------------------------------------------------------
def load_all_data(
    input_dir: str = INPUT_DIR_DEFAULT,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
    min_freq: int = MIN_FREQ,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Vocab, Dict[str, int]]:
    train_path = os.path.join(input_dir, "train.csv")
    test_path = os.path.join(input_dir, "test.csv")

    full_train = pd.read_csv(train_path)
    if "text" in full_train.columns:
        full_train = full_train.dropna(subset=["text"]).reset_index(drop=True)
    if "target" not in full_train.columns:
        raise ValueError("train.csv must contain a 'target' column")

    test_df = pd.read_csv(test_path).reset_index(drop=True)

    tr, va = train_test_split(
        full_train,
        test_size=float(val_fraction),
        stratify=full_train["target"].astype(int).values,
        random_state=int(seed),
    )
    tr = tr.reset_index(drop=True)
    va = va.reset_index(drop=True)

    vocab = Vocab(min_freq=min_freq).fit(tr["text"].fillna("").astype(str).tolist())

    kw_lower = (
        tr["keyword"].fillna("").astype(str).str.lower().str.strip()
        if "keyword" in tr.columns
        else pd.Series([], dtype=str)
    )
    kw_set = sorted({k for k in kw_lower.unique().tolist() if k})
    keyword2idx: Dict[str, int] = {
        k: i + 1 for i, k in enumerate(kw_set)
    }  # 0 = no keyword

    return tr, va, test_df, vocab, keyword2idx


def build_datasets(
    tr: pd.DataFrame,
    va: pd.DataFrame,
    test_df: pd.DataFrame,
    vocab: Vocab,
    keyword2idx: Dict[str, int],
    max_len: int = MAX_LEN,
) -> Tuple[TweetDataset, TweetDataset, TweetDataset]:
    ds_tr = TweetDataset(tr, vocab, keyword2idx, max_len=max_len, has_target=True)
    ds_va = TweetDataset(va, vocab, keyword2idx, max_len=max_len, has_target=True)
    ds_te = TweetDataset(test_df, vocab, keyword2idx, max_len=max_len, has_target=False)
    return ds_tr, ds_va, ds_te


# ---------------------------------------------------------------------------
# CPU preflight adapter
# ---------------------------------------------------------------------------
class CandidateAdapter:
    """No-argument adapter used by the CPU preflight checker.

    Exposes the real model, real loss, real CSV-derived batches, and the
    optimizer/view used by the training pipeline.
    """

    def __init__(self):
        self.defaults: Dict[str, object] = {
            "emb_dim": 128,
            "hidden_dim": 128,
            "n_layers": 2,
            "n_heads": 4,
            "dropout": 0.3,
            "max_len": MAX_LEN,
        }
        self._vocab_cache: Optional[Vocab] = None
        self._keyword_cache: Optional[Dict[str, int]] = None
        self._df_cache: Optional[pd.DataFrame] = None

    # --- internal helpers -------------------------------------------------
    def _resolve_input_dir(self) -> str:
        return os.environ.get("MLEVOLVE_INPUT_DIR", INPUT_DIR_DEFAULT)

    def _load_caches(self) -> None:
        if self._vocab_cache is not None and self._keyword_cache is not None:
            return
        input_dir = self._resolve_input_dir()
        train_path = os.path.join(input_dir, "train.csv")
        df = pd.read_csv(train_path)
        if "text" in df.columns:
            df = df.dropna(subset=["text"]).reset_index(drop=True)
        # Fit vocab + keyword index on the FULL training file here — this is
        # only the preflight adapter path, not the pipeline train/val split.
        vocab = Vocab(min_freq=MIN_FREQ).fit(df["text"].fillna("").astype(str).tolist())
        if "keyword" in df.columns:
            kw_lower = df["keyword"].fillna("").astype(str).str.lower().str.strip()
        else:
            kw_lower = pd.Series([], dtype=str)
        kw_set = sorted({k for k in kw_lower.unique().tolist() if k})
        keyword2idx = {k: i + 1 for i, k in enumerate(kw_set)}
        self._vocab_cache = vocab
        self._keyword_cache = keyword2idx
        self._df_cache = df

    def _merged_context(
        self, context: Optional[Dict[str, object]]
    ) -> Dict[str, object]:
        ctx = dict(self.defaults)
        if context:
            for k, v in dict(context).items():
                if v is not None:
                    ctx[k] = v
        return ctx

    # --- contract ---------------------------------------------------------
    def build_model(self, context) -> nn.Module:
        ctx = self._merged_context(context)
        self._load_caches()
        vocab = self._vocab_cache
        keyword2idx = self._keyword_cache
        assert vocab is not None and keyword2idx is not None
        model = DisasterNet(
            vocab_size=len(vocab),
            num_keywords=len(keyword2idx) + 1,
            num_numeric=NUM_NUMERIC,
            emb_dim=int(ctx.get("emb_dim", 128)),
            hidden_dim=int(ctx.get("hidden_dim", 128)),
            n_layers=int(ctx.get("n_layers", 2)),
            n_heads=int(ctx.get("n_heads", 4)),
            dropout=float(ctx.get("dropout", 0.3)),
            max_len=int(ctx.get("max_len", MAX_LEN)),
        )
        return model

    def build_optimizer(self, model, context):
        ctx = self._merged_context(context)
        lr = float(ctx.get("lr", 1e-3))
        wd = float(ctx.get("weight_decay", 0.01))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def _make_batch(self, bs: int, device: torch.device) -> Dict[str, object]:
        self._load_caches()
        vocab = self._vocab_cache
        keyword2idx = self._keyword_cache
        df = self._df_cache
        assert vocab is not None and keyword2idx is not None and df is not None

        bs = max(1, int(bs))
        sub = df.head(bs)
        max_len = int(self.defaults["max_len"])
        ids_list: List[List[int]] = []
        kw_list: List[int] = []
        num_list: List[np.ndarray] = []
        tgt_list: List[float] = []
        for _, row in sub.iterrows():
            text = row["text"] if isinstance(row.get("text", None), str) else ""
            kw = row["keyword"] if isinstance(row.get("keyword", None), str) else ""
            loc = row["location"] if isinstance(row.get("location", None), str) else ""
            ids_list.append(vocab.encode(text, max_len))
            kw_list.append(keyword2idx.get(kw.lower().strip(), 0))
            num_list.append(extract_numeric_features(text, kw, loc))
            if "target" in row:
                tgt_list.append(float(row["target"]))
            else:
                tgt_list.append(0.0)

        input_ids = torch.tensor(ids_list, dtype=torch.long, device=device)
        keyword_id = torch.tensor(kw_list, dtype=torch.long, device=device)
        num_feats = torch.tensor(
            np.stack(num_list, axis=0), dtype=torch.float32, device=device
        )
        target = torch.tensor(tgt_list, dtype=torch.float32, device=device)
        return {"inputs": (input_ids, keyword_id, num_feats), "target": target}

    def build_train_batch(self, scenario, device):
        bs = int(scenario.get("batch_size", 8)) if isinstance(scenario, dict) else 8
        return self._make_batch(bs, device)

    def build_validation_batch(self, scenario, device):
        bs = int(scenario.get("batch_size", 8)) if isinstance(scenario, dict) else 8
        return self._make_batch(bs, device)

    def _resolve_criterion(self, context) -> nn.Module:
        crit = None
        if context:
            crit = dict(context).get("criterion")
        if crit is None:
            crit = FocalBCELoss()
        return crit

    def training_step(self, model, batch, context):
        criterion = self._resolve_criterion(context)
        inputs = batch["inputs"]
        target = batch["target"]
        logits = model(*inputs)
        loss = criterion(logits, target)
        return loss

    def validation_step(self, model, batch, context):
        criterion = self._resolve_criterion(context)
        inputs = batch["inputs"]
        target = batch["target"]
        was_training = model.training
        model.eval()
        with torch.no_grad():
            logits = model(*inputs)
            loss = criterion(logits, target)
        if was_training:
            model.train()
        return {"loss": loss, "logits": logits, "target": target}


if __name__ == "__main__":
    # Full training / validation / submission pipeline is owned by later stages.
    pass

```
        
        ### Step 2: datatype_precision
        **Plan:** ## Plan

**Datatype/precision step**: I'll lock down a conservative precision tier for the Stage 1 hybrid word-level Transformer (FP32 default, FP16 AMP as an *opt-in* policy) that the training stage can consume. The Stage 1 candidate already keeps Focal-BCE numerically robust (it explicitly casts logits/targets to `float32`), so the autocast region is confined to the Transformer forward path while the loss, metric accumulation, and prediction export stay FP32. Backward/optimizer run outside autocast with GradScaler only when FP16 is actually active; TF32 is force-disabled and CPU preflight is guaranteed FP32 with scaling off.

**WHAT/WHY/HOW**: The competition is short-text binary classification with F1 — where a single mis-scaled logit near the threshold flips precision/recall heavily — so the safe move on an A10 (compute capability 8.6, fp16-safe tensor cores) is selective FP16 autocast *around* the encoder, FP32 elsewhere. This is only enabled when the caller explicitly requests it (default "disabled"), matching the decision trace's `precision_policy="disabled"` and the HWDB `confidence=0.0` (no verified speedup evidence). Confirmation = finite loss/grads, F1 within noise of FP32, and non-regressed epoch time; reconsider trigger = any NaN/Inf grad, unstable early-epoch loss, or slower wall-time → fallback to FP32. Memory strategy is unchanged from Stage 1 (fixed-shape `max_len` tensor); elasticity is unchanged (batch_size from caller); no operator/normalization redesign — only autocast region and TF32/scaler toggles are added. Scheduler-owned controls (device assignment, thread caps, memory limits, daemon) are explicitly *not* implemented here.
        **Code:**
        ```python
# =============================================================================
# Stage 2 — datatype / precision policy
# -----------------------------------------------------------------------------
# Import-safe: defines dataclasses, resolvers and helpers only.
# No device selection, no model construction, no training, no side effects.
# Consumed by Stage 3 (training_evaluation) and by CandidateAdapter.
# =============================================================================

import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

# ---------------------------------------------------------------------------
# Top-level policy defaults (conservative; decision trace: precision_policy="disabled")
# ---------------------------------------------------------------------------
PRECISION_POLICY_DEFAULT: str = "disabled"  # "disabled" (fp32) | "fp16_amp"
AMP_DTYPE_NAME_DEFAULT: str = "float16"
USE_TF32_DEFAULT: bool = False  # forced off in normal mode
DEVICE_OVERRIDE_DEFAULT: Optional[str] = None  # let caller/scheduler choose


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class PrecisionConfig:
    """Caller-facing knobs. All optional; merged over defaults in resolve_precision."""

    policy: str = PRECISION_POLICY_DEFAULT
    amp_dtype: str = AMP_DTYPE_NAME_DEFAULT
    use_tf32: bool = USE_TF32_DEFAULT
    device: Optional[str] = DEVICE_OVERRIDE_DEFAULT
    fallback_to_fp32: bool = True
    reason: Optional[str] = None


@dataclass
class PrecisionRuntime:
    """Resolved, immutable-ish precision state consumed by training/eval code."""

    device: torch.device
    policy: str  # "fp32" or "fp16_amp"
    amp_enabled: bool
    amp_dtype: torch.dtype
    use_tf32: bool
    scaler_enabled: bool
    fallback_to_fp32: bool
    reason: str
    notes: Dict[str, Any] = field(default_factory=dict)

    # -- context managers ---------------------------------------------------
    def autocast(self):
        """Autocast around eligible forward ops. No-op when AMP disabled."""
        if self.amp_enabled and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        return contextlib.nullcontext()

    def autocast_off(self):
        """Explicit FP32 island — use inside autocast for fragile reductions/loss."""
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", enabled=False)
        return contextlib.nullcontext()

    # -- scaler -------------------------------------------------------------
    def make_grad_scaler(self):
        """Return a GradScaler for FP16 or None. Uses new torch.amp API when available."""
        if not self.scaler_enabled or self.device.type != "cuda":
            return None
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            # Older torch fallback
            try:
                return torch.cuda.amp.GradScaler(enabled=True)
            except Exception:
                return None


# ---------------------------------------------------------------------------
# Device resolution (no CUDA calls at import time)
# ---------------------------------------------------------------------------
def resolve_device(explicit: Optional[str] = None) -> torch.device:
    """Resolve target device. Scheduler-owned assignment is honored when given."""
    if explicit:
        try:
            return torch.device(explicit)
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Precision resolution
# ---------------------------------------------------------------------------
def _normalize_policy(policy: Optional[str]) -> str:
    if policy is None:
        return PRECISION_POLICY_DEFAULT
    p = str(policy).strip().lower()
    if p in ("fp16_amp", "fp16", "amp", "fp16-amp"):
        return "fp16_amp"
    # Anything else (including "fp32", "disabled", unknown) → conservative FP32
    return "disabled"


def resolve_precision(
    context: Optional[Dict[str, Any]] = None,
    config: Optional[PrecisionConfig] = None,
) -> PrecisionRuntime:
    """Resolve a PrecisionRuntime from an optional context mapping (partial-merge).

    Never raises: any unsupported path falls back to FP32 with a recorded note.
    """
    ctx: Dict[str, Any] = dict(context or {})
    cfg = config or PrecisionConfig()

    # Merge config <- context (context wins when present)
    policy_raw = ctx.get("precision_policy", ctx.get("policy", cfg.policy))
    amp_dtype_raw = ctx.get("amp_dtype", cfg.amp_dtype)
    use_tf32_raw = ctx.get("use_tf32", cfg.use_tf32)
    device_raw = ctx.get("device", cfg.device)
    fallback_ok = bool(ctx.get("fallback_to_fp32", cfg.fallback_to_fp32))
    reason_raw = ctx.get("reason", cfg.reason)

    policy = _normalize_policy(policy_raw)
    device = resolve_device(device_raw)
    notes: Dict[str, Any] = {}

    # TF32 is force-off in normal mode regardless of caller ask.
    if bool(use_tf32_raw):
        notes["tf32_forced_off"] = True
    use_tf32 = False

    amp_enabled = False
    amp_dtype = torch.float32
    scaler_enabled = False

    if policy == "fp16_amp":
        if device.type != "cuda":
            notes["amp_disabled"] = "device_not_cuda"
        elif not torch.cuda.is_available():
            notes["amp_disabled"] = "cuda_unavailable"
        elif str(amp_dtype_raw).lower() not in ("float16", "fp16", "half"):
            notes["amp_disabled"] = f"unsupported_amp_dtype:{amp_dtype_raw}"
        else:
            # Verify GPU can actually run FP16 tensor-core autocast.
            try:
                cap_major, _cap_minor = torch.cuda.get_device_capability(device)
                if cap_major >= 7:
                    amp_enabled = True
                    amp_dtype = torch.float16
                    scaler_enabled = True
                else:
                    notes["amp_disabled"] = f"capability_below_7:{cap_major}"
            except Exception as e:  # noqa: BLE001
                notes["amp_disabled"] = f"cap_check_failed:{type(e).__name__}"

    if amp_enabled:
        final_policy = "fp16_amp"
    else:
        final_policy = "fp32"
        amp_dtype = torch.float32
        scaler_enabled = False

    if reason_raw:
        reason = str(reason_raw)
    elif amp_enabled:
        reason = "caller_requested_fp16_amp;verified_cuda_cc>=7"
    else:
        reason = "conservative_fp32_default(no_verified_speedup_evidence)"

    return PrecisionRuntime(
        device=device,
        policy=final_policy,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        use_tf32=use_tf32,
        scaler_enabled=scaler_enabled,
        fallback_to_fp32=bool(fallback_ok),
        reason=reason,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Backend flag application (safe subset; no device placement, no daemon, no
# thread/memory controls owned by the scheduler).
# ---------------------------------------------------------------------------
def apply_precision_backend(precision: PrecisionRuntime) -> None:
    """Apply only precision-related torch.backends flags. Idempotent, non-fatal."""
    # TF32 must be off per normal-mode policy.
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    # Also silence the new fine-grained tf32 toggles when present.
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch.backends.cudnn.conv.fp32_precision = "ieee"  # type: ignore[attr-defined]
    except Exception:
        pass
    # Hint: cuDNN benchmark is a scheduling/timing concern; do not toggle here.
    _ = precision  # reserved for future per-runtime flags


# ---------------------------------------------------------------------------
# Convenience helpers used by the training loop
# ---------------------------------------------------------------------------
def should_use_amp(precision: PrecisionRuntime) -> bool:
    return bool(precision.amp_enabled and precision.device.type == "cuda")


def safe_autocast(precision: PrecisionRuntime):
    """Alias so training code has a single, grep-friendly entrypoint."""
    return precision.autocast()


def fp32_island(precision: PrecisionRuntime):
    """Run a fragile block (loss reduction, metrics, threshold search) in FP32."""
    return precision.autocast_off()


def ensure_fp32(t: torch.Tensor) -> torch.Tensor:
    """Lightweight guard for FP32-sensitive ops (loss/metrics). No-op if already fp32."""
    if not torch.is_floating_point(t):
        return t
    return t if t.dtype == torch.float32 else t.float()


def unscale_if_needed_(optimizer, scaler, model) -> None:
    """Unscale gradients before clipping when running under FP16 GradScaler."""
    if scaler is not None and scaler.is_enabled():
        scaler.unscale_(optimizer)


# ---------------------------------------------------------------------------
# Logging helper (single line; called once by training entrypoint)
# ---------------------------------------------------------------------------
def precision_log_line(precision: PrecisionRuntime) -> str:
    amp_dtype_name = str(precision.amp_dtype).replace("torch.", "")
    return (
        f"[precision] device={precision.device.type} "
        f"policy={precision.policy} dtype={amp_dtype_name} "
        f"amp={precision.amp_enabled} tf32={precision.use_tf32} "
        f"scaler={precision.scaler_enabled} fallback_fp32={precision.fallback_to_fp32} "
        f"reason={precision.reason} notes={precision.notes}"
    )

```
        
        ### Step 3: training_evaluation
        **Plan:** ## Plan

**WHAT/WHY/HOW**: Stage 3 locks in the hyperparameter envelope for the Stage 1 hybrid word-level Transformer under the Stage 2 conservative precision tier (FP32 default, FP16 AMP opt-in). The training regime: AdamW (LR 1.5e-3, weight decay 0.02, betas (0.9, 0.98)), a 10% linear-warmup + cosine-decay schedule stepped per *optimizer update* (not per microbatch), physical batch 64 with `GRAD_ACCUM_STEPS=1`, grad-norm clipping 1.0, and early stopping (patience 4) with best-F1-threshold checkpoint restore. AMP autocast is confined to the encoder forward and the loss is explicitly run in FP32 via `precision.autocast_off()`; the GradScaler path unscales **before** clipping. Validation/test use a single shared `_evaluate` routine, so val and test probs come from identical processing; the threshold is search-fitted on val only. **Confirmation** = F1 strictly above the keyword-only prior (~0.57) on internal val; **reconsider trigger** = plateaus near prior or diverges → I would rather fall back to FP32 than enlarge the model family.

**Memory & backend**: fixed-shape `(B, max_len)` token tensor + small dense feature vectors → near-constant VRAM per batch; physical batch elasticity via `QUALITY_SAFE_PHYSICAL_BATCH_SIZES=[16,32,64]` with `sqrt` LR scaling policy, `GRAD_ACCUM_STEPS` preserved as 1 unless the scheduler overrides. Loader workers = 2, pinned memory, persistent workers, deterministic seed 42. **Not implemented here** (scheduler-owned): process daemon, device assignment, thread caps, memory limits, sibling job launch.

**Note board:**
- baseline_change: Adds Stage 3 hyperparameters — AdamW (betas 0.9/0.98, wd 0.02), 10% linear warmup + cosine decay stepped per optimizer update, physical batch 64 with `GRAD_ACCUM_STEPS=1` and `BATCH_LR_SCALING_POLICY="sqrt"` — replacing the previous merged-script approach that used a fixed linear-warmup schedule with different plumbing. | purpose: stable convergence of the pre-norm transformer + Focal-BCE head, with an LR-scale policy that the scheduler can safely re-resolve if batch changes. | hardware_keys: physical_batch_size, effective_batch_size, grad_accum_steps, epoch_count, learning_rate, weight_decay, num_workers, pin_memory
- baseline_change: Consumes Stage 2 `PrecisionRuntime` as-is — AMP autocast only around the encoder forward, `autocast_off()` island for Focal-BCE, `GradScaler.unscale_` before `clip_grad_norm_`, backward/optimizer strictly outside autocast; TF32 left disabled. | purpose: preserves the Stage 1 loss/data/output interface, protects FP32-sensitive threshold/export math, and keeps FP16 as an opt-in with an explicit FP32 fallback path. | hardware_keys: precision_policy, amp_dtype, use_tf32, scaler, fallback_fp32
- baseline_change: Single shared `_evaluate`/threshold search drives validation and test predictions; best-model restore + `TrainingDiagnostics` state saved in checkpoint and wired into `script_scheduler_context` STEP/EPOCH/BEFORE_TRAIN safe points. | purpose: guarantees val/test consistency (identical processing), implements metric-based early stopping without leaking test, and satisfies the scheduler resume contract. | hardware_keys: steps_per_epoch, checkpoint_cadence, split_seed
        **Code:**
        ```python
# =============================================================================
# Stage 3 — training_evaluation
# -----------------------------------------------------------------------------
# Import-safe. Defines Stage-3 hyperparameters, training/eval helpers, and the
# __main__ entrypoint that consumes Stage 1 (model_design) and Stage 2
# (datatype_precision) code as-is.
# =============================================================================

import os
import math
import time
import random
import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Optional backend integrations (kept import-safe; guarded for CPU preflight)
# ---------------------------------------------------------------------------
try:
    from utils.training_diagnostics import TrainingDiagnostics
except Exception:  # pragma: no cover - fallback only used when module missing

    class TrainingDiagnostics:  # noqa: D401 - drop-in no-op
        def __init__(self, model=None, optimizer=None, scaler=None, settings=None):
            self.model = model
            self.optimizer = optimizer
            self.scaler = scaler
            self.settings = dict(settings or {})

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def after_update(self):
            return None

        def report(self, epoch=None):
            return None

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            return None


try:
    from localml_scheduler.execution.script_context import script_scheduler_context
except Exception:  # pragma: no cover

    def script_scheduler_context():  # type: ignore
        return None


try:
    from localml_scheduler.domain import SafePointType
except Exception:  # pragma: no cover

    class SafePointType:  # noqa: D401 - local enum-like fallback
        BEFORE_TRAIN = "BEFORE_TRAIN"
        STEP = "STEP"
        EPOCH = "EPOCH"


# =============================================================================
# Stage 3 — hyperparameters
# =============================================================================
PHYSICAL_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 1
EFFECTIVE_BATCH_SIZE = PHYSICAL_BATCH_SIZE * GRAD_ACCUM_STEPS
NUM_EPOCHS = 12
LEARNING_RATE = 1.5e-3
WEIGHT_DECAY = 0.02
ADAMW_BETAS: Tuple[float, float] = (0.9, 0.98)
ADAMW_EPS = 1e-8
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0
EARLY_STOP_PATIENCE = 4
DATALOADER_WORKERS = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True
EVAL_BATCH_MULT = 2

CHECKPOINT_PATH = (
    os.path.join(WORKING_DIR, "best_checkpoint.pt")
    if "WORKING_DIR" in globals()
    else "./working/best_checkpoint.pt"
)
SUBMISSION_PATH = (
    os.path.join(SUBMISSION_DIR, "submission.csv")
    if "SUBMISSION_DIR" in globals()
    else "./submission/submission.csv"
)

# Batch / LR envelope contract exposed to the scheduler.
QUALITY_SAFE_PHYSICAL_BATCH_SIZES = [16, 32, 64]
BATCH_LR_SCALING_POLICY = "sqrt"  # one of: "fixed", "linear", "sqrt"


# =============================================================================
# Helper: seeding
# =============================================================================
def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# =============================================================================
# Helper: optimizer / scheduler
# =============================================================================
def make_optimizer(
    model: nn.Module, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer, total_updates: int, warmup_updates: int
):
    total_updates = max(1, int(total_updates))
    warmup_updates = max(0, int(warmup_updates))

    def lr_lambda(step: int) -> float:
        if step < warmup_updates:
            return float(step + 1) / max(1, warmup_updates)
        progress = float(step - warmup_updates) / max(1, total_updates - warmup_updates)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Helper: dataloaders
# =============================================================================
def make_loaders(
    train_ds,
    val_ds,
    test_ds,
    batch_size: int,
    workers: int,
    pin_memory: bool,
):
    common = dict(
        num_workers=int(workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(workers and workers > 0),
        collate_fn=collate_batch,
    )
    tr = DataLoader(
        train_ds, batch_size=int(batch_size), shuffle=True, drop_last=False, **common
    )
    ev_bs = max(1, int(batch_size) * EVAL_BATCH_MULT)
    va = DataLoader(val_ds, batch_size=ev_bs, shuffle=False, **common)
    te = DataLoader(test_ds, batch_size=ev_bs, shuffle=False, **common)
    return tr, va, te


# =============================================================================
# Helper: checkpoint state capture / restore
# =============================================================================
def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["rng_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return
    if "rng_python" in state:
        try:
            random.setstate(state["rng_python"])
        except Exception:
            pass
    if "rng_numpy" in state:
        try:
            np.random.set_state(state["rng_numpy"])
        except Exception:
            pass
    if "rng_torch" in state:
        try:
            torch.set_rng_state(state["rng_torch"])
        except Exception:
            pass
    if torch.cuda.is_available() and "rng_cuda" in state:
        try:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        except Exception:
            pass


def _make_checkpoint_dict(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    scheduler,
    global_step: int,
    epoch: int,
    step_in_epoch: int,
    diagnostics: "TrainingDiagnostics",
    best_f1: float,
    best_state: Optional[Dict[str, torch.Tensor]],
    best_threshold: float,
    best_epoch: int,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "step_in_epoch": int(step_in_epoch),
        "best": {
            "f1": float(best_f1),
            "threshold": float(best_threshold),
            "epoch": int(best_epoch),
            "model_state": best_state,
        },
    }
    if scaler is not None:
        try:
            state["scaler"] = scaler.state_dict()
        except Exception:
            pass
    if scheduler is not None:
        try:
            state["scheduler"] = scheduler.state_dict()
        except Exception:
            pass
    rng_state = _capture_rng_state()
    state.update(rng_state)
    try:
        state["diagnostics"] = diagnostics.state_dict()
    except Exception:
        pass
    return state


def _restore_state(
    saved: Dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    scheduler,
    diagnostics: "TrainingDiagnostics",
) -> Tuple[int, int, int, float, Optional[Dict[str, torch.Tensor]], float, int]:
    """Returns (start_epoch, global_step, step_in_epoch, best_f1, best_state, best_threshold, best_epoch)."""
    if not saved:
        return 0, 0, 0, -1.0, None, 0.5, 0
    try:
        if "model" in saved:
            model.load_state_dict(saved["model"])
    except Exception as e:
        print(f"[resume_warn] model_load_failed:{type(e).__name__}")
    try:
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])
    except Exception as e:
        print(f"[resume_warn] optimizer_load_failed:{type(e).__name__}")
    if scaler is not None and "scaler" in saved:
        try:
            scaler.load_state_dict(saved["scaler"])
        except Exception:
            pass
    if scheduler is not None and "scheduler" in saved:
        try:
            scheduler.load_state_dict(saved["scheduler"])
        except Exception:
            pass
    _restore_rng_state(saved)
    if diagnostics is not None and "diagnostics" in saved:
        try:
            diagnostics.load_state_dict(saved["diagnostics"])
        except Exception:
            pass
    best_f1 = -1.0
    best_state = None
    best_threshold = 0.5
    best_epoch = 0
    if isinstance(saved.get("best"), dict):
        best_f1 = float(saved["best"].get("f1", -1.0))
        best_state = saved["best"].get("model_state", None)
        best_threshold = float(saved["best"].get("threshold", 0.5))
        best_epoch = int(saved["best"].get("epoch", 0))
    return (
        int(saved.get("epoch", 0)),
        int(saved.get("global_step", 0)),
        int(saved.get("step_in_epoch", 0)),
        best_f1,
        best_state,
        best_threshold,
        best_epoch,
    )


# =============================================================================
# Batch-to-device helper
# =============================================================================
def _move_batch_to_device(batch: Dict[str, Any], device: torch.device):
    input_ids, keyword_id, num_feats = batch["inputs"]
    input_ids = input_ids.to(device, non_blocking=True)
    keyword_id = keyword_id.to(device, non_blocking=True)
    num_feats = num_feats.to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True).float()
    return (input_ids, keyword_id, num_feats), target


# =============================================================================
# Forward / evaluate routines (shared by val and test → identical processing)
# =============================================================================
def _forward_logits(model: nn.Module, inputs, precision):
    input_ids, keyword_id, num_feats = inputs
    with precision.autocast():
        logits = model(input_ids, keyword_id, num_feats)
    return logits.float()


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    precision,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    all_probs = []
    all_targets = []
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            inputs, target = _move_batch_to_device(batch, device)
            logits = _forward_logits(model, inputs, precision)
            with precision.autocast_off():
                loss = criterion(logits, target)
            probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            all_probs.append(probs)
            all_targets.append(target.detach().cpu().numpy().astype(np.float32))
            total_loss += float(loss.item()) * int(target.size(0))
            total_n += int(target.size(0))
    probs = np.concatenate(all_probs) if all_probs else np.zeros((0,), dtype=np.float32)
    targets = (
        np.concatenate(all_targets) if all_targets else np.zeros((0,), dtype=np.float32)
    )
    avg_loss = total_loss / max(1, total_n)
    return probs, targets, avg_loss


# =============================================================================
# Training entrypoint
# =============================================================================
def run_training() -> float:
    # ---- Scheduler context (must be called before CUDA allocations) -------
    context = script_scheduler_context()

    resume_payload = None
    if context is not None:
        try:
            resume_payload = context.load_resume_checkpoint()
        except Exception as e:
            print(f"[scheduler] resume_load_error:{type(e).__name__}")
            resume_payload = None

    # ---- Seed ------------------------------------------------------------
    _set_seed(SEED)

    # ---- Precision (Stage 2 policy, consumed as-is) -----------------------
    precision = resolve_precision(context=None)
    apply_precision_backend(precision)
    print(precision_log_line(precision))
    device = precision.device

    # ---- Data (Stage 1 helpers; fit vocab on train-split only) ------------
    tr_df, va_df, te_df, vocab, keyword2idx = load_all_data()
    ds_tr, ds_va, ds_te = build_datasets(tr_df, va_df, te_df, vocab, keyword2idx)

    # ---- Model (Stage 1 family, unchanged) --------------------------------
    model = DisasterNet(
        vocab_size=len(vocab),
        num_keywords=len(keyword2idx) + 1,
        num_numeric=NUM_NUMERIC,
        emb_dim=128,
        hidden_dim=128,
        n_layers=2,
        n_heads=4,
        dropout=0.3,
        max_len=MAX_LEN,
    ).to(device)

    # ---- Loaders / optimizer / criterion ----------------------------------
    tr_loader, va_loader, te_loader = make_loaders(
        ds_tr,
        ds_va,
        ds_te,
        batch_size=PHYSICAL_BATCH_SIZE,
        workers=DATALOADER_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda",
    )
    criterion = FocalBCELoss(gamma=1.5, label_smoothing=0.02)
    optimizer = make_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)

    steps_per_epoch = max(1, len(tr_loader) // max(1, GRAD_ACCUM_STEPS))
    total_updates = steps_per_epoch * NUM_EPOCHS
    warmup_updates = int(total_updates * WARMUP_RATIO)
    scheduler = make_scheduler(optimizer, total_updates, warmup_updates)
    scaler = precision.make_grad_scaler()
    use_amp_scaler = scaler is not None and scaler.is_enabled()

    # ---- Update scheduler job budget --------------------------------------
    if context is not None:
        try:
            context.job.max_epochs = NUM_EPOCHS
            context.job.config.max_epochs = NUM_EPOCHS
            context.store.save_job(context.job)
        except Exception:
            pass

    # ---- Runtime state (shared for state_factory closures) ----------------
    runtime = {"epoch": 0, "global_step": 0, "step_in_epoch": 0}
    best = {"f1": -1.0, "state": None, "threshold": 0.5, "epoch": 0}
    epochs_no_improve = 0

    diag_settings = {
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "planned_epochs": NUM_EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "model_family": MODEL_FAMILY,
        "split_seed": SEED,
        "precision_policy": precision.policy,
        "precision_reason": precision.reason,
    }

    print(
        f"[train] device={device.type} family={MODEL_FAMILY} "
        f"phys_bs={PHYSICAL_BATCH_SIZE} accum={GRAD_ACCUM_STEPS} "
        f"eff_bs={EFFECTIVE_BATCH_SIZE} epochs={NUM_EPOCHS} lr={LEARNING_RATE} "
        f"wd={WEIGHT_DECAY} steps_per_epoch={steps_per_epoch} "
        f"train_n={len(ds_tr)} val_n={len(ds_va)} test_n={len(ds_te)}"
    )

    start_time = time.time()

    with TrainingDiagnostics(
        model, optimizer, scaler=scaler, settings=diag_settings
    ) as diagnostics:
        # ---- Optional resume -------------------------------------------
        if (
            context is not None
            and isinstance(resume_payload, dict)
            and resume_payload.get("state")
        ):
            (
                start_epoch,
                gstep,
                sin_ep,
                bf1,
                bstate,
                bthr,
                bepoch,
            ) = _restore_state(
                resume_payload["state"],
                model,
                optimizer,
                scaler,
                scheduler,
                diagnostics,
            )
            runtime["epoch"] = start_epoch
            runtime["global_step"] = gstep
            runtime["step_in_epoch"] = sin_ep
            best["f1"] = bf1
            best["state"] = bstate
            best["threshold"] = bthr
            best["epoch"] = bepoch
            print(f"[resume] start_epoch={start_epoch} global_step={gstep}")

        # ---- BEFORE_TRAIN safe point ------------------------------------
        def _bt_state_factory():
            return _make_checkpoint_dict(
                model,
                optimizer,
                scaler,
                scheduler,
                runtime["global_step"],
                runtime["epoch"],
                runtime["step_in_epoch"],
                diagnostics,
                best["f1"],
                best["state"],
                best["threshold"],
                best["epoch"],
            )

        if context is not None:
            context.control_hook.safe_point(
                SafePointType.BEFORE_TRAIN,
                epoch=runtime["epoch"],
                global_step=runtime["global_step"],
                steps_per_epoch=steps_per_epoch,
                state_factory=_bt_state_factory,
            )

        # ---- Epoch loop ---------------------------------------------------
        for epoch_one_based in range(runtime["epoch"] + 1, NUM_EPOCHS + 1):
            epoch_t0 = time.time()
            model.train()
            running_loss = 0.0
            running_n = 0
            micro_count = 0
            optimizer.zero_grad(set_to_none=True)
            update_counter_in_epoch = 0

            for batch in tr_loader:
                inputs, target = _move_batch_to_device(batch, device)
                with precision.autocast():
                    logits = model(*inputs)
                with precision.autocast_off():
                    loss = criterion(logits.float(), target)
                loss_for_backward = loss / max(1, GRAD_ACCUM_STEPS)

                if use_amp_scaler:
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

                running_loss += float(loss.item()) * int(target.size(0))
                running_n += int(target.size(0))
                micro_count += 1

                do_step = (micro_count % GRAD_ACCUM_STEPS == 0) or (
                    micro_count == len(tr_loader)
                )
                if not do_step:
                    continue

                # --- optimizer update attempt ---------------------------
                if use_amp_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                diagnostics.after_update()

                if scheduler is not None:
                    try:
                        scheduler.step()
                    except Exception:
                        pass

                runtime["global_step"] += 1
                update_counter_in_epoch += 1
                runtime["step_in_epoch"] = update_counter_in_epoch

                # --- STEP safe point ------------------------------------
                def _step_state_factory():
                    return _make_checkpoint_dict(
                        model,
                        optimizer,
                        scaler,
                        scheduler,
                        runtime["global_step"],
                        epoch_one_based,
                        runtime["step_in_epoch"],
                        diagnostics,
                        best["f1"],
                        best["state"],
                        best["threshold"],
                        best["epoch"],
                    )

                if context is not None:
                    context.control_hook.safe_point(
                        SafePointType.STEP,
                        epoch=epoch_one_based,
                        global_step=runtime["global_step"],
                        steps_per_epoch=steps_per_epoch,
                        state_factory=_step_state_factory,
                    )

            runtime["epoch"] = epoch_one_based
            train_loss = running_loss / max(1, running_n)

            # ---- Validation + threshold search --------------------------
            val_probs, val_targets, val_loss = _evaluate(
                model, va_loader, criterion, precision, device
            )
            val_thr, val_f1 = find_best_threshold(val_probs, val_targets)

            # ---- Diagnostics per-epoch report ---------------------------
            try:
                diagnostics.report(epoch=epoch_one_based)
            except Exception:
                pass

            elapsed = time.time() - epoch_t0
            print(
                f"[epoch {epoch_one_based:02d}/{NUM_EPOCHS}] "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_f1={val_f1:.4f} thr={val_thr:.3f} time={elapsed:.1f}s"
            )

            # ---- Parseable scheduler metric line ------------------------
            print(
                "MLEVOLVE_EPOCH_METRIC "
                f'{{"epoch": {epoch_one_based}, "metric": {val_f1:.6f}, "metric_name": "val_f1"}}'
            )

            # ---- Best-checkpoint update ---------------------------------
            if val_f1 > best["f1"] + 1e-6:
                best["f1"] = float(val_f1)
                best["threshold"] = float(val_thr)
                best["epoch"] = int(epoch_one_based)
                best["state"] = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                try:
                    torch.save(
                        {
                            "model": best["state"],
                            "threshold": best["threshold"],
                            "f1": best["f1"],
                            "epoch": best["epoch"],
                            "model_family": MODEL_FAMILY,
                        },
                        CHECKPOINT_PATH,
                    )
                except Exception as e:
                    print(f"[ckpt_warn] save_failed:{type(e).__name__}")
            else:
                epochs_no_improve += 1

            # ---- EPOCH safe point ---------------------------------------
            def _epoch_state_factory():
                return _make_checkpoint_dict(
                    model,
                    optimizer,
                    scaler,
                    scheduler,
                    runtime["global_step"],
                    epoch_one_based,
                    runtime["step_in_epoch"],
                    diagnostics,
                    best["f1"],
                    best["state"],
                    best["threshold"],
                    best["epoch"],
                )

            if context is not None:
                context.control_hook.safe_point(
                    SafePointType.EPOCH,
                    epoch=epoch_one_based,
                    global_step=runtime["global_step"],
                    steps_per_epoch=steps_per_epoch,
                    state_factory=_epoch_state_factory,
                )

            # ---- Early stopping (still after EPOCH safe point) ----------
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"[early_stop] no improvement for {EARLY_STOP_PATIENCE} epochs")
                break

    # ---- Restore best checkpoint & final scoring --------------------------
    if best["state"] is not None:
        try:
            model.load_state_dict(best["state"])
        except Exception as e:
            print(f"[final_warn] best_state_load_failed:{type(e).__name__}")

    # Final validation with the best model + val-tuned threshold (val only).
    val_probs, val_targets, val_loss = _evaluate(
        model, va_loader, criterion, precision, device
    )
    final_thr, final_f1 = find_best_threshold(val_probs, val_targets)
    final_f1 = float(final_f1)
    final_thr = float(final_thr)

    # Test inference through the SAME _evaluate path (identical processing).
    test_probs, _, _ = _evaluate(model, te_loader, criterion, precision, device)
    test_preds = (test_probs >= final_thr).astype(int)

    # ---- Write submission -------------------------------------------------
    try:
        os.makedirs(os.path.dirname(SUBMISSION_PATH) or ".", exist_ok=True)
    except Exception:
        pass
    sub_df = pd.DataFrame({"id": te_df["id"].values, "target": test_preds})
    sub_df.to_csv(SUBMISSION_PATH, index=False)
    print(
        f"[submission] wrote {SUBMISSION_PATH} rows={len(sub_df)} "
        f"pos_rate={float(test_preds.mean()):.4f} threshold={final_thr:.3f}"
    )

    total_time = time.time() - start_time
    print(
        f"[done] elapsed={total_time:.1f}s best_epoch={best['epoch']} "
        f"final_val_f1={final_f1:.4f} final_thr={final_thr:.3f}"
    )
    return final_f1


# =============================================================================
# Main entrypoint
# =============================================================================
if __name__ == "__main__":
    score = run_training()
    print(f"Final Validation Score: {score}")

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
        