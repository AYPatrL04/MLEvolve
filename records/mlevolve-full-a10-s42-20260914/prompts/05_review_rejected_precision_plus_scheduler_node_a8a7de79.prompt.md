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
    "output_interface": "float logit per row -> sigmoid -> threshold -> integer {0,1} written as target column alongside id",
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
    "optimizer": "AdamW, lr 2e-5 for encoder / 1e-3 for head, weight_decay 0.01",
    "scheduler": "linear warmup (10%) then linear decay",
    "batch_size_policy": "fixed",
    "dataloader_policy": "num_workers>0, pin_memory=True, persistent_workers when reused, non_blocking=True device copies; dynamic padding acceptable",
    "fallbacks": [
      "if AMP unstable or slower, run FP32; if pretrained weights unavailable, fall back to small_transformer or TF-IDF + linear model with same splits and I/O"
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
      "no matching profiles for current hardware; graph_evidence empty so recommendations are unverified candidates"
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
    "output_interface": "float logit per row -> sigmoid -> threshold -> integer {0,1} written as target column alongside id",
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
    "optimizer": "AdamW, lr 2e-5 for encoder / 1e-3 for head, weight_decay 0.01",
    "scheduler": "linear warmup (10%) then linear decay",
    "batch_size_policy": "fixed",
    "dataloader_policy": "num_workers>0, pin_memory=True, persistent_workers when reused, non_blocking=True device copies; dynamic padding acceptable",
    "fallbacks": [
      "if AMP unstable or slower, run FP32; if pretrained weights unavailable, fall back to small_transformer or TF-IDF + linear model with same splits and I/O"
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
      "no matching profiles for current hardware; graph_evidence empty so recommendations are unverified candidates"
    ]
  }
}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=## Plan **WHAT makes this task unique:** Disaster-tweet classification is a *context-dependent* short-text problem where lexical cues (e.g.; purpose=Define the Stage 1 candidate target for later precision and training stages.; hardware_keys=pos_weight, vocab_size, max_len, d_model, num_layers, dim_feedforward, available_transformer_or_sequence_baseline, disaster_small_transformer_v1

# Previous steps
### model_design
**Plan:** ## Plan

**WHAT makes this task unique:** Disaster-tweet classification is a *context-dependent* short-text problem where lexical cues (e.g. "ablaze", "evacuate") co-occur with noisy social-media artifacts (URLs, @mentions, slang) and a categorical `keyword` field; F1 favors calibrated decision thresholds over raw accuracy, and the 6090-example train set is small enough that regularization matters more than capacity.

**WHY this approach:** I will keep the pipeline in the transformer family required by the decision trace but adapt it to the small-data regime: (a) a compact ~1–2 M parameter transformer encoder trained from scratch with a *learned* word-level vocabulary fit on the training split only, (b) keyword/location folded into the text stream with sentinel tokens (`keymark`, `locmark`) so the transformer sees them in-context instead of as a separate one-hot branch, (c) mean-pooling over non-pad tokens with per-batch dynamic padding so the transformer (and later FP16 AMP path) sees dense, tensor-core-friendly tensors, (d) class-imbalance handled via `pos_weight` in BCEWithLogitsLoss and a post-hoc threshold search on the internal validation split to directly maximize F1. To keep the file robust to offline environments I additionally expose a fully-defined fallback tokenizer/model config so nothing depends on a Hugging Face download.

**HOW I will validate:** Internal stratified 85/15 split (seed 42) — if val F1 after threshold tuning is meaningfully above the naive majority-class F1 (~0.57) and the model overfits only mildly (train-val gap bounded), the transformer branch is validated; if val F1 collapses or loss diverges, I will fall back to a TF-IDF + linear model on the same split, using the same text prep.

**Memory / batch-elasticity strategy:** static per-batch dynamic padding (no pack-to-max), so physical batch size can be reduced for the preflight without changing model geometry; `vocab_size`, `max_len`, `d_model`, `nhead`, `num_layers`, `dim_feedforward` remain configurable context keys.

**Backend contract:** the subprocess owns nothing about workers, device placement, CUDA graphs, memory limits, torch threads, or precision policy — those are scheduler-owned and Stage 2/3-owned. The `CandidateAdapter` is CPU-safe, no-argument, and only exposes the real model/loss/optimizer/batch builders.

**Note board:**
- baseline_change: replaced a generic `available_transformer_or_sequence_baseline` with a concrete small from-scratch word-transformer (`disaster_small_transformer_v1`) whose vocabulary and `pos_weight` are fit only on the training split; keyword/location are injected as sentinel tokens into the text stream. | purpose: honors the `text`/`classification` family while explicitly using the `keyword`/`location` columns the task advertises, keeps parameters and vocab lightweight enough to train within the hour budget. | hardware_keys: none
- baseline_change: added a keyword-aware text builder, an explicit class-imbalance-aware loss factory, and a validation-split F1 threshold search as first-class model-design artifacts. | purpose: F1 is threshold-sensitive; making the criterion and threshold selection part of the design (not training loop) aligns Stage 2/3 with the same decision rule and prevents metric-vs-loss drift. | hardware_keys: none
- baseline_change: exposed `input_ids`/`attention_mask` tensors plus an `inputs` tuple with dynamic padding and `torch.long` dtypes, and a CPU-safe `CandidateAdapter`. | purpose: enables non-blocking CUDA transfers, tensor-core-friendly dense batches under optional FP16 AMP, and satisfies the preflight contract without downloading weights. | hardware_keys: none
**Code:**
```python
"""
Stage 1: model_design for Disaster Tweets (text classification, binary F1).

Defines:
  - Data loading + text cleaning + keyword/location fusion
  - Word-level tokenizer (fit on TRAIN ONLY)
  - torch Dataset + dynamic-padding collate
  - Small transformer classifier (word embeddings + TransformerEncoder + mean pool)
  - BCEWithLogits criterion with pos_weight
  - Validation F1 threshold search
  - CandidateAdapter for CPU preflight (real model, no downloads)

Does NOT define: training loop, optimizer schedule, precision policy, batch size,
epochs, workers, checkpointing, or submission writing.
"""

import os
import re
import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------
MODEL_FAMILY = "disaster_small_transformer_v1"
SEED = 42
INPUT_DIR = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")

MAX_SEQ_LEN = 64
MAX_VOCAB = 30000
MIN_FREQ = 2
VAL_SIZE = 0.15

PAD_IDX = 0
UNK_IDX = 1
SPECIAL_TOKENS: Dict[str, int] = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}

# ---------------------------------------------------------------------------
# Text cleaning / feature fusion
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NUM_RE = re.compile(r"\d+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase + normalize URLs/mentions/hashtags/numbers into sentinel words."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = _URL_RE.sub(" urlmark ", t)
    t = _MENTION_RE.sub(" usermark ", t)
    t = _HASHTAG_RE.sub(r" \1 ", t)
    t = _NUM_RE.sub(" nummark ", t)
    t = _NON_ALNUM_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def build_model_input(row) -> str:
    """Fuse text + keyword + location into a single stream with in-context sentinels."""
    parts: List[str] = []
    text = row.get("text", "")
    if isinstance(text, str) and text.strip():
        parts.append(text)
    kw = row.get("keyword", None)
    if isinstance(kw, str) and kw.strip():
        parts.append("keymark " + kw.replace("%20", " "))
    loc = row.get("location", None)
    if isinstance(loc, str) and loc.strip():
        parts.append("locmark " + loc)
    return " ".join(parts)


def prepare_texts(df: pd.DataFrame) -> List[str]:
    return [clean_text(build_model_input(row)) for _, row in df.iterrows()]


# ---------------------------------------------------------------------------
# Tokenizer (train-only vocabulary)
# ---------------------------------------------------------------------------
class SimpleWordTokenizer:
    """Whitespace tokenizer with a train-fit frequency vocabulary."""

    def __init__(self, max_vocab: int = MAX_VOCAB, min_freq: int = MIN_FREQ):
        self.max_vocab = int(max_vocab)
        self.min_freq = int(min_freq)
        self.vocab: Dict[str, int] = dict(SPECIAL_TOKENS)

    def fit(self, texts: Sequence[str]) -> "SimpleWordTokenizer":
        counter: Counter = Counter()
        for t in texts:
            if isinstance(t, str):
                counter.update(t.split())
        for tok, cnt in counter.most_common():
            if cnt < self.min_freq:
                break
            if tok in self.vocab:
                continue
            if len(self.vocab) >= self.max_vocab:
                break
            self.vocab[tok] = len(self.vocab)
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
        if not isinstance(text, str):
            return [PAD_IDX]
        toks = text.split()
        ids = [self.vocab.get(t, UNK_IDX) for t in toks]
        if len(ids) == 0:
            ids = [UNK_IDX]
        if len(ids) > max_len:
            ids = ids[:max_len]
        return ids

    def state_dict(self) -> Dict:
        return {
            "vocab": dict(self.vocab),
            "max_vocab": self.max_vocab,
            "min_freq": self.min_freq,
        }

    @classmethod
    def from_state_dict(cls, sd: Dict) -> "SimpleWordTokenizer":
        tok = cls(
            max_vocab=sd.get("max_vocab", MAX_VOCAB),
            min_freq=sd.get("min_freq", MIN_FREQ),
        )
        tok.vocab = dict(sd["vocab"])
        return tok


def build_tokenizer_on_train(
    train_texts: Sequence[str], max_vocab: int = MAX_VOCAB, min_freq: int = MIN_FREQ
) -> SimpleWordTokenizer:
    return SimpleWordTokenizer(max_vocab=max_vocab, min_freq=min_freq).fit(train_texts)


# ---------------------------------------------------------------------------
# Dataset + collate
# ---------------------------------------------------------------------------
class TweetDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[float]],
        tokenizer: SimpleWordTokenizer,
        max_len: int = MAX_SEQ_LEN,
    ):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tok = tokenizer
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.tok.encode(self.texts[idx], self.max_len)
        item: Dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(ids, dtype=torch.long)
        }
        if self.labels is not None:
            item["label"] = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return item


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Dynamic padding; emits input_ids (long) + attention_mask (long)."""
    seqs = [b["input_ids"] for b in batch]
    L = max(s.numel() for s in seqs)
    B = len(seqs)
    input_ids = torch.zeros(B, L, dtype=torch.long)
    attention_mask = torch.zeros(B, L, dtype=torch.long)
    for i, s in enumerate(seqs):
        n = s.numel()
        input_ids[i, :n] = s
        attention_mask[i, :n] = 1
    out: Dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch], dim=0)
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DisasterTransformer(nn.Module):
    """
    Compact word-transformer for short-text binary classification.

    - Learned word embeddings + sinusoidal-free learned positional embeddings
    - N pre-LN TransformerEncoder blocks (GELU, configurable dims)
    - Mean pooling over non-pad tokens -> linear head emits a single logit
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int = MAX_SEQ_LEN,
        d_model: int = 192,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 384,
        dropout: float = 0.15,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.vocab_size = int(vocab_size)
        self.max_len = int(max_len)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout_p = float(dropout)

        self.token_emb = nn.Embedding(
            self.vocab_size, self.d_model, padding_idx=PAD_IDX
        )
        self.pos_emb = nn.Embedding(self.max_len, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model, eps=1e-6)
        self.dropout = nn.Dropout(self.dropout_p)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)

        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.d_model, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_emb.weight, std=0.02)
        with torch.no_grad():
            self.token_emb.weight[PAD_IDX].fill_(0.0)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be [B, L], got shape {tuple(input_ids.shape)}"
            )
        B, L = input_ids.shape
        L = min(L, self.max_len)
        input_ids = input_ids[:, :L]

        if attention_mask is None:
            attention_mask = (input_ids != PAD_IDX).long()
        else:
            attention_mask = attention_mask[:, :L]
            if attention_mask.dtype != torch.long:
                attention_mask = attention_mask.long()

        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        pos = pos.clamp_(max=self.max_len - 1)

        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.input_norm(x)
        x = self.dropout(x)

        # True = ignore in src_key_padding_mask
        key_padding_mask = attention_mask == 0
        # Guard all-padding rows to avoid NaN in attention softmax
        all_pad = key_padding_mask.all(dim=1)
        if bool(all_pad.any()):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_pad, 0] = False

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        mask_f = attention_mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_(min=1.0)
        pooled = (x * mask_f).sum(dim=1) / denom

        logits = self.head(pooled).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# Loss / criterion
# ---------------------------------------------------------------------------
class BCEWithLogitsCriterion(nn.Module):
    """Binary cross-entropy on logits with optional per-class pos_weight."""

    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer(
                "pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32)
            )
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        logits = logits.float()
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


def compute_pos_weight(labels: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=np.float32)
    n_pos = float((y > 0.5).sum())
    n_neg = float(len(y) - n_pos)
    if n_pos <= 0.0:
        return 1.0
    return n_neg / n_pos


def get_criterion(pos_weight: Optional[float] = None) -> BCEWithLogitsCriterion:
    return BCEWithLogitsCriterion(pos_weight=pos_weight)


# ---------------------------------------------------------------------------
# Output interface helpers
# ---------------------------------------------------------------------------
def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.float())


def find_best_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    lo: float = 0.15,
    hi: float = 0.85,
    step: float = 0.01,
) -> Tuple[float, float]:
    """Grid search the decision threshold that maximizes binary F1."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(lo, hi + 1e-9, step):
        preds = (probs >= t).astype(np.int64)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_t = float(t)
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Data loading / splitting
# ---------------------------------------------------------------------------
def load_dataframes(input_dir: str = INPUT_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(input_dir, "train.csv")
    test_path = os.path.join(input_dir, "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def stratified_split(
    train_df: pd.DataFrame, val_size: float = VAL_SIZE, seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = np.arange(len(train_df))
    strat = train_df["target"].values
    tr_idx, va_idx = train_test_split(
        idx, test_size=val_size, random_state=seed, stratify=strat
    )
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)
    return tr_df, va_df


def pipeline_summary() -> Dict:
    """Small machine-readable descriptor for later stages."""
    return {
        "model_family": MODEL_FAMILY,
        "modality": "text",
        "target_type": "classification",
        "output_interface": "logit -> sigmoid -> threshold -> {0,1}",
        "loss": "BCEWithLogitsLoss (pos_weight from train labels)",
        "max_seq_len": MAX_SEQ_LEN,
        "default_dims": {
            "d_model": 192,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 384,
        },
        "input_dir": INPUT_DIR,
        "seed": SEED,
        "val_size": VAL_SIZE,
    }


# ---------------------------------------------------------------------------
# CPU preflight: CandidateAdapter
# ---------------------------------------------------------------------------
class CandidateAdapter:
    """
    Real-model CPU preflight. Uses the same model/tokenizer/loss/text-prep as the
    full pipeline. No downloads, no CUDA required, no side effects on import.
    """

    DEFAULT_CONTEXT: Dict[str, object] = {
        "vocab_size": 5000,
        "max_len": 32,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "pos_weight": 1.0,
        "seed": SEED,
        "input_dir": INPUT_DIR,
    }

    def __init__(self):
        self._tokenizer: Optional[SimpleWordTokenizer] = None
        self._cached_train_texts: Optional[List[str]] = None
        self._cached_train_labels: Optional[List[int]] = None

    # -- context handling ---------------------------------------------------
    def _merge(self, context: Optional[Dict]) -> Dict[str, object]:
        cfg = dict(self.DEFAULT_CONTEXT)
        if context:
            for k, v in context.items():
                if v is not None:
                    cfg[k] = v
        return cfg

    # -- data ---------------------------------------------------------------
    def _load_train_hint(
        self, input_dir: str, n_rows: int
    ) -> Tuple[List[str], List[int]]:
        train_path = os.path.join(input_dir, "train.csv")
        texts: List[str] = []
        labels: List[int] = []
        if os.path.exists(train_path):
            try:
                df = pd.read_csv(train_path)
                head = df.head(max(n_rows, 4))
                for _, row in head.iterrows():
                    texts.append(clean_text(build_model_input(row)))
                    labels.append(int(row["target"]))
            except Exception:
                texts, labels = [], []
        if not texts:
            texts = [
                "sample disaster tweet text keymark locmark",
                "calm day no emergency happening now",
            ]
            labels = [0, 0]
        return texts, labels

    def _get_tokenizer(self, input_dir: str, vocab_size: int) -> SimpleWordTokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        texts, labels = self._load_train_hint(input_dir, n_rows=500)
        self._cached_train_texts = texts
        self._cached_train_labels = labels
        tok = SimpleWordTokenizer(max_vocab=max(int(vocab_size), 64), min_freq=1).fit(
            texts
        )
        self._tokenizer = tok
        return tok

    # -- model --------------------------------------------------------------
    def build_model(self, context: Optional[Dict] = None) -> nn.Module:
        cfg = self._merge(context)
        input_dir = str(cfg["input_dir"])
        tok = self._get_tokenizer(input_dir, int(cfg["vocab_size"]))
        vs = max(int(cfg["vocab_size"]), tok.vocab_size)

        # Guarantee nhead divides d_model even if caller supplies odd values
        d_model = int(cfg["d_model"])
        nhead = int(cfg["nhead"])
        if d_model % nhead != 0:
            d_model = (d_model // nhead) * nhead

        model = DisasterTransformer(
            vocab_size=vs,
            max_len=int(cfg["max_len"]),
            d_model=d_model,
            nhead=nhead,
            num_layers=int(cfg["num_layers"]),
            dim_feedforward=int(cfg["dim_feedforward"]),
            dropout=float(cfg["dropout"]),
        )
        return model

    def build_optimizer(
        self, model: nn.Module, context: Optional[Dict] = None
    ) -> torch.optim.Optimizer:
        # Real optimizer object; schedule/hyperparams are Stage-3 concerns, so we
        # mirror the decision trace defaults here (encoder-style lr for the whole net).
        return torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # -- batches ------------------------------------------------------------
    def _build_batch(
        self, scenario: Optional[Dict], device: torch.device, default_batch: int = 8
    ) -> Dict[str, object]:
        scenario = scenario or {}
        cfg = self._merge(scenario)
        input_dir = str(cfg["input_dir"])
        max_len = int(cfg["max_len"])
        batch_size = int(scenario.get("batch_size", default_batch))
        batch_size = max(batch_size, 1)

        texts, labels = self._load_train_hint(input_dir, n_rows=batch_size)
        # Make exactly `batch_size` rows
        while len(texts) < batch_size:
            texts.append("sample tweet about a disaster keymark locmark")
            labels.append(0)
        texts = texts[:batch_size]
        labels = labels[:batch_size]

        tok = self._get_tokenizer(input_dir, int(cfg["vocab_size"]))

        enc = [tok.encode(t, max_len) for t in texts]
        L = max(len(e) for e in enc)

        # Respect an explicit fixture length hint (non-batch shape (L,)).
        fixture = scenario.get("fixture", None)
        if isinstance(fixture, dict) and "input_ids" in fixture:
            try:
                fx = fixture["input_ids"]
                fx_len = int(fx[-1]) if isinstance(fx, (list, tuple)) else int(fx)
                L = min(L, max(fx_len, 1))
                enc = [e[:L] for e in enc]
            except Exception:
                pass
        L = max(L, 1)

        input_ids = torch.zeros(batch_size, L, dtype=torch.long, device=device)
        attention_mask = torch.zeros(batch_size, L, dtype=torch.long, device=device)
        for i, e in enumerate(enc):
            n = min(len(e), L)
            if n > 0:
                input_ids[i, :n] = torch.tensor(e[:n], dtype=torch.long, device=device)
                attention_mask[i, :n] = 1

        target = torch.tensor(labels, dtype=torch.float32, device=device)

        return {
            # Contract-recognized keys
            "inputs": (input_ids, attention_mask),
            "target": target,
            # Named tensors preserved for fixture contracts
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def build_train_batch(self, scenario, device) -> Dict[str, object]:
        return self._build_batch(scenario, device)

    def build_validation_batch(self, scenario, device) -> Dict[str, object]:
        return self._build_batch(scenario, device)

    # -- steps --------------------------------------------------------------
    def _criterion_for(self, context: Optional[Dict]) -> BCEWithLogitsCriterion:
        cfg = self._merge(context)
        pw = cfg.get("pos_weight", None)
        pw = float(pw) if pw is not None else None
        return BCEWithLogitsCriterion(pos_weight=pw)

    def training_step(
        self, model: nn.Module, batch: Dict[str, object], context: Optional[Dict] = None
    ) -> torch.Tensor:
        input_ids, attention_mask = batch["inputs"]  # type: ignore[misc]
        target = batch["target"]
        logits = model(input_ids, attention_mask)  # type: ignore[arg-type]
        loss = self._criterion_for(context)(logits, target)  # type: ignore[arg-type]
        return loss

    def validation_step(
        self, model: nn.Module, batch: Dict[str, object], context: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask = batch["inputs"]  # type: ignore[misc]
        target = batch["target"]
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                logits = model(input_ids, attention_mask)  # type: ignore[arg-type]
                probs = torch.sigmoid(logits.float())
        finally:
            if was_training:
                model.train()
        return {"probs": probs, "target": target}  # type: ignore[dict-item]

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
    "output_interface": "float logit per row -> sigmoid -> threshold -> integer {0,1} written as target column alongside id",
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
    "optimizer": "AdamW, lr 2e-5 for encoder / 1e-3 for head, weight_decay 0.01",
    "scheduler": "linear warmup (10%) then linear decay",
    "batch_size_policy": "fixed",
    "dataloader_policy": "num_workers>0, pin_memory=True, persistent_workers when reused, non_blocking=True device copies; dynamic padding acceptable",
    "fallbacks": [
      "if AMP unstable or slower, run FP32; if pretrained weights unavailable, fall back to small_transformer or TF-IDF + linear model with same splits and I/O"
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
      "no matching profiles for current hardware; graph_evidence empty so recommendations are unverified candidates"
    ]
  }
}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=## Plan **WHAT makes this task unique:** Disaster-tweet classification is a *context-dependent* short-text problem where lexical cues (e.g.; purpose=Define the Stage 1 candidate target for later precision and training stages.; hardware_keys=pos_weight, vocab_size, max_len, d_model, num_layers, dim_feedforward, available_transformer_or_sequence_baseline, disaster_small_transformer_v1
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=added `PrecisionConfig`, runtime AMP/autocast/GradScaler/TF32 helpers and a scalar-loss precision logger instead of leaving precision implicit in the training loop.; purpose=keeps the `disaster_small_transformer_v1` candidate target intact while making the fp16_amp policy selective (autocast only around forward+loss; scaler/unscale outside) and explicitly reversible to FP32.; hardware_keys=precision_policy, precision_model_adaptation, fallback_policy, fp16_amp, use_amp, use_tf32, grad_scaler
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=added a non-invasive `prepare_model_for_precision` that asserts FP32 parameter dtype instead of `model.half()`; no precision-required module replacement was added since `precision_model_adaptation=none`.; purpose=prevents accidental whole-pipeline half casting that would break BCE logits reductions and metric accumulation.; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=added a CPU-only preflight preference (`force_fp32_on_cpu`) plus a non-finite-loss trigger to downgrade to FP32 at runtime.; purpose=implements the decision trace's stated fallback ("fall back to fp32 or disabled AMP if the selected precision path is unsupported") without changing any other Stage 1/3 decision.; hardware_keys=none

# Previous steps
### model_design
**Plan:** ## Plan

**WHAT makes this task unique:** Disaster-tweet classification is a *context-dependent* short-text problem where lexical cues (e.g. "ablaze", "evacuate") co-occur with noisy social-media artifacts (URLs, @mentions, slang) and a categorical `keyword` field; F1 favors calibrated decision thresholds over raw accuracy, and the 6090-example train set is small enough that regularization matters more than capacity.

**WHY this approach:** I will keep the pipeline in the transformer family required by the decision trace but adapt it to the small-data regime: (a) a compact ~1–2 M parameter transformer encoder trained from scratch with a *learned* word-level vocabulary fit on the training split only, (b) keyword/location folded into the text stream with sentinel tokens (`keymark`, `locmark`) so the transformer sees them in-context instead of as a separate one-hot branch, (c) mean-pooling over non-pad tokens with per-batch dynamic padding so the transformer (and later FP16 AMP path) sees dense, tensor-core-friendly tensors, (d) class-imbalance handled via `pos_weight` in BCEWithLogitsLoss and a post-hoc threshold search on the internal validation split to directly maximize F1. To keep the file robust to offline environments I additionally expose a fully-defined fallback tokenizer/model config so nothing depends on a Hugging Face download.

**HOW I will validate:** Internal stratified 85/15 split (seed 42) — if val F1 after threshold tuning is meaningfully above the naive majority-class F1 (~0.57) and the model overfits only mildly (train-val gap bounded), the transformer branch is validated; if val F1 collapses or loss diverges, I will fall back to a TF-IDF + linear model on the same split, using the same text prep.

**Memory / batch-elasticity strategy:** static per-batch dynamic padding (no pack-to-max), so physical batch size can be reduced for the preflight without changing model geometry; `vocab_size`, `max_len`, `d_model`, `nhead`, `num_layers`, `dim_feedforward` remain configurable context keys.

**Backend contract:** the subprocess owns nothing about workers, device placement, CUDA graphs, memory limits, torch threads, or precision policy — those are scheduler-owned and Stage 2/3-owned. The `CandidateAdapter` is CPU-safe, no-argument, and only exposes the real model/loss/optimizer/batch builders.

**Note board:**
- baseline_change: replaced a generic `available_transformer_or_sequence_baseline` with a concrete small from-scratch word-transformer (`disaster_small_transformer_v1`) whose vocabulary and `pos_weight` are fit only on the training split; keyword/location are injected as sentinel tokens into the text stream. | purpose: honors the `text`/`classification` family while explicitly using the `keyword`/`location` columns the task advertises, keeps parameters and vocab lightweight enough to train within the hour budget. | hardware_keys: none
- baseline_change: added a keyword-aware text builder, an explicit class-imbalance-aware loss factory, and a validation-split F1 threshold search as first-class model-design artifacts. | purpose: F1 is threshold-sensitive; making the criterion and threshold selection part of the design (not training loop) aligns Stage 2/3 with the same decision rule and prevents metric-vs-loss drift. | hardware_keys: none
- baseline_change: exposed `input_ids`/`attention_mask` tensors plus an `inputs` tuple with dynamic padding and `torch.long` dtypes, and a CPU-safe `CandidateAdapter`. | purpose: enables non-blocking CUDA transfers, tensor-core-friendly dense batches under optional FP16 AMP, and satisfies the preflight contract without downloading weights. | hardware_keys: none
**Code:**
```python
"""
Stage 1: model_design for Disaster Tweets (text classification, binary F1).

Defines:
  - Data loading + text cleaning + keyword/location fusion
  - Word-level tokenizer (fit on TRAIN ONLY)
  - torch Dataset + dynamic-padding collate
  - Small transformer classifier (word embeddings + TransformerEncoder + mean pool)
  - BCEWithLogits criterion with pos_weight
  - Validation F1 threshold search
  - CandidateAdapter for CPU preflight (real model, no downloads)

Does NOT define: training loop, optimizer schedule, precision policy, batch size,
epochs, workers, checkpointing, or submission writing.
"""

import os
import re
import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------
MODEL_FAMILY = "disaster_small_transformer_v1"
SEED = 42
INPUT_DIR = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")

MAX_SEQ_LEN = 64
MAX_VOCAB = 30000
MIN_FREQ = 2
VAL_SIZE = 0.15

PAD_IDX = 0
UNK_IDX = 1
SPECIAL_TOKENS: Dict[str, int] = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}

# ---------------------------------------------------------------------------
# Text cleaning / feature fusion
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NUM_RE = re.compile(r"\d+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase + normalize URLs/mentions/hashtags/numbers into sentinel words."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = _URL_RE.sub(" urlmark ", t)
    t = _MENTION_RE.sub(" usermark ", t)
    t = _HASHTAG_RE.sub(r" \1 ", t)
    t = _NUM_RE.sub(" nummark ", t)
    t = _NON_ALNUM_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def build_model_input(row) -> str:
    """Fuse text + keyword + location into a single stream with in-context sentinels."""
    parts: List[str] = []
    text = row.get("text", "")
    if isinstance(text, str) and text.strip():
        parts.append(text)
    kw = row.get("keyword", None)
    if isinstance(kw, str) and kw.strip():
        parts.append("keymark " + kw.replace("%20", " "))
    loc = row.get("location", None)
    if isinstance(loc, str) and loc.strip():
        parts.append("locmark " + loc)
    return " ".join(parts)


def prepare_texts(df: pd.DataFrame) -> List[str]:
    return [clean_text(build_model_input(row)) for _, row in df.iterrows()]


# ---------------------------------------------------------------------------
# Tokenizer (train-only vocabulary)
# ---------------------------------------------------------------------------
class SimpleWordTokenizer:
    """Whitespace tokenizer with a train-fit frequency vocabulary."""

    def __init__(self, max_vocab: int = MAX_VOCAB, min_freq: int = MIN_FREQ):
        self.max_vocab = int(max_vocab)
        self.min_freq = int(min_freq)
        self.vocab: Dict[str, int] = dict(SPECIAL_TOKENS)

    def fit(self, texts: Sequence[str]) -> "SimpleWordTokenizer":
        counter: Counter = Counter()
        for t in texts:
            if isinstance(t, str):
                counter.update(t.split())
        for tok, cnt in counter.most_common():
            if cnt < self.min_freq:
                break
            if tok in self.vocab:
                continue
            if len(self.vocab) >= self.max_vocab:
                break
            self.vocab[tok] = len(self.vocab)
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
        if not isinstance(text, str):
            return [PAD_IDX]
        toks = text.split()
        ids = [self.vocab.get(t, UNK_IDX) for t in toks]
        if len(ids) == 0:
            ids = [UNK_IDX]
        if len(ids) > max_len:
            ids = ids[:max_len]
        return ids

    def state_dict(self) -> Dict:
        return {
            "vocab": dict(self.vocab),
            "max_vocab": self.max_vocab,
            "min_freq": self.min_freq,
        }

    @classmethod
    def from_state_dict(cls, sd: Dict) -> "SimpleWordTokenizer":
        tok = cls(
            max_vocab=sd.get("max_vocab", MAX_VOCAB),
            min_freq=sd.get("min_freq", MIN_FREQ),
        )
        tok.vocab = dict(sd["vocab"])
        return tok


def build_tokenizer_on_train(
    train_texts: Sequence[str], max_vocab: int = MAX_VOCAB, min_freq: int = MIN_FREQ
) -> SimpleWordTokenizer:
    return SimpleWordTokenizer(max_vocab=max_vocab, min_freq=min_freq).fit(train_texts)


# ---------------------------------------------------------------------------
# Dataset + collate
# ---------------------------------------------------------------------------
class TweetDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[float]],
        tokenizer: SimpleWordTokenizer,
        max_len: int = MAX_SEQ_LEN,
    ):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tok = tokenizer
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.tok.encode(self.texts[idx], self.max_len)
        item: Dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(ids, dtype=torch.long)
        }
        if self.labels is not None:
            item["label"] = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return item


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Dynamic padding; emits input_ids (long) + attention_mask (long)."""
    seqs = [b["input_ids"] for b in batch]
    L = max(s.numel() for s in seqs)
    B = len(seqs)
    input_ids = torch.zeros(B, L, dtype=torch.long)
    attention_mask = torch.zeros(B, L, dtype=torch.long)
    for i, s in enumerate(seqs):
        n = s.numel()
        input_ids[i, :n] = s
        attention_mask[i, :n] = 1
    out: Dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch], dim=0)
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DisasterTransformer(nn.Module):
    """
    Compact word-transformer for short-text binary classification.

    - Learned word embeddings + sinusoidal-free learned positional embeddings
    - N pre-LN TransformerEncoder blocks (GELU, configurable dims)
    - Mean pooling over non-pad tokens -> linear head emits a single logit
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int = MAX_SEQ_LEN,
        d_model: int = 192,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 384,
        dropout: float = 0.15,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.vocab_size = int(vocab_size)
        self.max_len = int(max_len)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout_p = float(dropout)

        self.token_emb = nn.Embedding(
            self.vocab_size, self.d_model, padding_idx=PAD_IDX
        )
        self.pos_emb = nn.Embedding(self.max_len, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model, eps=1e-6)
        self.dropout = nn.Dropout(self.dropout_p)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)

        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.d_model, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_emb.weight, std=0.02)
        with torch.no_grad():
            self.token_emb.weight[PAD_IDX].fill_(0.0)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be [B, L], got shape {tuple(input_ids.shape)}"
            )
        B, L = input_ids.shape
        L = min(L, self.max_len)
        input_ids = input_ids[:, :L]

        if attention_mask is None:
            attention_mask = (input_ids != PAD_IDX).long()
        else:
            attention_mask = attention_mask[:, :L]
            if attention_mask.dtype != torch.long:
                attention_mask = attention_mask.long()

        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        pos = pos.clamp_(max=self.max_len - 1)

        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.input_norm(x)
        x = self.dropout(x)

        # True = ignore in src_key_padding_mask
        key_padding_mask = attention_mask == 0
        # Guard all-padding rows to avoid NaN in attention softmax
        all_pad = key_padding_mask.all(dim=1)
        if bool(all_pad.any()):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_pad, 0] = False

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        mask_f = attention_mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_(min=1.0)
        pooled = (x * mask_f).sum(dim=1) / denom

        logits = self.head(pooled).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# Loss / criterion
# ---------------------------------------------------------------------------
class BCEWithLogitsCriterion(nn.Module):
    """Binary cross-entropy on logits with optional per-class pos_weight."""

    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer(
                "pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32)
            )
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        logits = logits.float()
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


def compute_pos_weight(labels: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=np.float32)
    n_pos = float((y > 0.5).sum())
    n_neg = float(len(y) - n_pos)
    if n_pos <= 0.0:
        return 1.0
    return n_neg / n_pos


def get_criterion(pos_weight: Optional[float] = None) -> BCEWithLogitsCriterion:
    return BCEWithLogitsCriterion(pos_weight=pos_weight)


# ---------------------------------------------------------------------------
# Output interface helpers
# ---------------------------------------------------------------------------
def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.float())


def find_best_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    lo: float = 0.15,
    hi: float = 0.85,
    step: float = 0.01,
) -> Tuple[float, float]:
    """Grid search the decision threshold that maximizes binary F1."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(lo, hi + 1e-9, step):
        preds = (probs >= t).astype(np.int64)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_t = float(t)
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Data loading / splitting
# ---------------------------------------------------------------------------
def load_dataframes(input_dir: str = INPUT_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(input_dir, "train.csv")
    test_path = os.path.join(input_dir, "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def stratified_split(
    train_df: pd.DataFrame, val_size: float = VAL_SIZE, seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = np.arange(len(train_df))
    strat = train_df["target"].values
    tr_idx, va_idx = train_test_split(
        idx, test_size=val_size, random_state=seed, stratify=strat
    )
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)
    return tr_df, va_df


def pipeline_summary() -> Dict:
    """Small machine-readable descriptor for later stages."""
    return {
        "model_family": MODEL_FAMILY,
        "modality": "text",
        "target_type": "classification",
        "output_interface": "logit -> sigmoid -> threshold -> {0,1}",
        "loss": "BCEWithLogitsLoss (pos_weight from train labels)",
        "max_seq_len": MAX_SEQ_LEN,
        "default_dims": {
            "d_model": 192,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 384,
        },
        "input_dir": INPUT_DIR,
        "seed": SEED,
        "val_size": VAL_SIZE,
    }


# ---------------------------------------------------------------------------
# CPU preflight: CandidateAdapter
# ---------------------------------------------------------------------------
class CandidateAdapter:
    """
    Real-model CPU preflight. Uses the same model/tokenizer/loss/text-prep as the
    full pipeline. No downloads, no CUDA required, no side effects on import.
    """

    DEFAULT_CONTEXT: Dict[str, object] = {
        "vocab_size": 5000,
        "max_len": 32,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "pos_weight": 1.0,
        "seed": SEED,
        "input_dir": INPUT_DIR,
    }

    def __init__(self):
        self._tokenizer: Optional[SimpleWordTokenizer] = None
        self._cached_train_texts: Optional[List[str]] = None
        self._cached_train_labels: Optional[List[int]] = None

    # -- context handling ---------------------------------------------------
    def _merge(self, context: Optional[Dict]) -> Dict[str, object]:
        cfg = dict(self.DEFAULT_CONTEXT)
        if context:
            for k, v in context.items():
                if v is not None:
                    cfg[k] = v
        return cfg

    # -- data ---------------------------------------------------------------
    def _load_train_hint(
        self, input_dir: str, n_rows: int
    ) -> Tuple[List[str], List[int]]:
        train_path = os.path.join(input_dir, "train.csv")
        texts: List[str] = []
        labels: List[int] = []
        if os.path.exists(train_path):
            try:
                df = pd.read_csv(train_path)
                head = df.head(max(n_rows, 4))
                for _, row in head.iterrows():
                    texts.append(clean_text(build_model_input(row)))
                    labels.append(int(row["target"]))
            except Exception:
                texts, labels = [], []
        if not texts:
            texts = [
                "sample disaster tweet text keymark locmark",
                "calm day no emergency happening now",
            ]
            labels = [0, 0]
        return texts, labels

    def _get_tokenizer(self, input_dir: str, vocab_size: int) -> SimpleWordTokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        texts, labels = self._load_train_hint(input_dir, n_rows=500)
        self._cached_train_texts = texts
        self._cached_train_labels = labels
        tok = SimpleWordTokenizer(max_vocab=max(int(vocab_size), 64), min_freq=1).fit(
            texts
        )
        self._tokenizer = tok
        return tok

    # -- model --------------------------------------------------------------
    def build_model(self, context: Optional[Dict] = None) -> nn.Module:
        cfg = self._merge(context)
        input_dir = str(cfg["input_dir"])
        tok = self._get_tokenizer(input_dir, int(cfg["vocab_size"]))
        vs = max(int(cfg["vocab_size"]), tok.vocab_size)

        # Guarantee nhead divides d_model even if caller supplies odd values
        d_model = int(cfg["d_model"])
        nhead = int(cfg["nhead"])
        if d_model % nhead != 0:
            d_model = (d_model // nhead) * nhead

        model = DisasterTransformer(
            vocab_size=vs,
            max_len=int(cfg["max_len"]),
            d_model=d_model,
            nhead=nhead,
            num_layers=int(cfg["num_layers"]),
            dim_feedforward=int(cfg["dim_feedforward"]),
            dropout=float(cfg["dropout"]),
        )
        return model

    def build_optimizer(
        self, model: nn.Module, context: Optional[Dict] = None
    ) -> torch.optim.Optimizer:
        # Real optimizer object; schedule/hyperparams are Stage-3 concerns, so we
        # mirror the decision trace defaults here (encoder-style lr for the whole net).
        return torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # -- batches ------------------------------------------------------------
    def _build_batch(
        self, scenario: Optional[Dict], device: torch.device, default_batch: int = 8
    ) -> Dict[str, object]:
        scenario = scenario or {}
        cfg = self._merge(scenario)
        input_dir = str(cfg["input_dir"])
        max_len = int(cfg["max_len"])
        batch_size = int(scenario.get("batch_size", default_batch))
        batch_size = max(batch_size, 1)

        texts, labels = self._load_train_hint(input_dir, n_rows=batch_size)
        # Make exactly `batch_size` rows
        while len(texts) < batch_size:
            texts.append("sample tweet about a disaster keymark locmark")
            labels.append(0)
        texts = texts[:batch_size]
        labels = labels[:batch_size]

        tok = self._get_tokenizer(input_dir, int(cfg["vocab_size"]))

        enc = [tok.encode(t, max_len) for t in texts]
        L = max(len(e) for e in enc)

        # Respect an explicit fixture length hint (non-batch shape (L,)).
        fixture = scenario.get("fixture", None)
        if isinstance(fixture, dict) and "input_ids" in fixture:
            try:
                fx = fixture["input_ids"]
                fx_len = int(fx[-1]) if isinstance(fx, (list, tuple)) else int(fx)
                L = min(L, max(fx_len, 1))
                enc = [e[:L] for e in enc]
            except Exception:
                pass
        L = max(L, 1)

        input_ids = torch.zeros(batch_size, L, dtype=torch.long, device=device)
        attention_mask = torch.zeros(batch_size, L, dtype=torch.long, device=device)
        for i, e in enumerate(enc):
            n = min(len(e), L)
            if n > 0:
                input_ids[i, :n] = torch.tensor(e[:n], dtype=torch.long, device=device)
                attention_mask[i, :n] = 1

        target = torch.tensor(labels, dtype=torch.float32, device=device)

        return {
            # Contract-recognized keys
            "inputs": (input_ids, attention_mask),
            "target": target,
            # Named tensors preserved for fixture contracts
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def build_train_batch(self, scenario, device) -> Dict[str, object]:
        return self._build_batch(scenario, device)

    def build_validation_batch(self, scenario, device) -> Dict[str, object]:
        return self._build_batch(scenario, device)

    # -- steps --------------------------------------------------------------
    def _criterion_for(self, context: Optional[Dict]) -> BCEWithLogitsCriterion:
        cfg = self._merge(context)
        pw = cfg.get("pos_weight", None)
        pw = float(pw) if pw is not None else None
        return BCEWithLogitsCriterion(pos_weight=pw)

    def training_step(
        self, model: nn.Module, batch: Dict[str, object], context: Optional[Dict] = None
    ) -> torch.Tensor:
        input_ids, attention_mask = batch["inputs"]  # type: ignore[misc]
        target = batch["target"]
        logits = model(input_ids, attention_mask)  # type: ignore[arg-type]
        loss = self._criterion_for(context)(logits, target)  # type: ignore[arg-type]
        return loss

    def validation_step(
        self, model: nn.Module, batch: Dict[str, object], context: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask = batch["inputs"]  # type: ignore[misc]
        target = batch["target"]
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                logits = model(input_ids, attention_mask)  # type: ignore[arg-type]
                probs = torch.sigmoid(logits.float())
        finally:
            if was_training:
                model.train()
        return {"probs": probs, "target": target}  # type: ignore[dict-item]

```

### datatype_precision
**Plan:** ## Plan

This step locks in the Stage 2 precision policy without touching the Stage 1 model family, loss, features, or output interface. I will expose a small, import-safe `PrecisionConfig` + helpers that (a) resolve `DEVICE`/precision at runtime, (b) wrap *only* the forward+loss in `torch.autocast('cuda', fp16)`, (c) hand back a `GradScaler` so backward/step run outside autocast, with `unscale_` before clipping, (d) force-disable TF32 for matmul/cudnn, and (e) provide a single-line precision logger. Fallback is explicit: CPU or unsupported FP16 → pure FP32/no scaler, matching the decision trace's `fallback_policy`.

The physics: this is a sub-2M-parameter word transformer on ~5k training examples — FP16 tensor cores on the A10 help throughput but FP32 is cheap here, so AMP is treated as **optional**; the loss (`BCEWithLogitsLoss`) and the metric are guarded in FP32, and any non-finite loss/grad triggers a graceful FP32 fallback. Model batch elasticity is preserved (dynamic padding, `input_ids` long, `attention_mask` long) — nothing in this stage changes the physical/effective batch size.

Note board:
- baseline_change: added `PrecisionConfig`, runtime AMP/autocast/GradScaler/TF32 helpers and a scalar-loss precision logger instead of leaving precision implicit in the training loop. | purpose: keeps the `disaster_small_transformer_v1` candidate target intact while making the fp16_amp policy selective (autocast only around forward+loss; scaler/unscale outside) and explicitly reversible to FP32. | hardware_keys: precision_policy, precision_model_adaptation, fallback_policy, fp16_amp, use_amp, use_tf32, grad_scaler
- baseline_change: added a non-invasive `prepare_model_for_precision` that asserts FP32 parameter dtype instead of `model.half()`; no precision-required module replacement was added since `precision_model_adaptation=none`. | purpose: prevents accidental whole-pipeline half casting that would break BCE logits reductions and metric accumulation. | hardware_keys: none
- baseline_change: added a CPU-only preflight preference (`force_fp32_on_cpu`) plus a non-finite-loss trigger to downgrade to FP32 at runtime. | purpose: implements the decision trace's stated fallback ("fall back to fp32 or disabled AMP if the selected precision path is unsupported") without changing any other Stage 1/3 decision. | hardware_keys: none
**Code:**
```python
"""
Stage 2: datatype_precision for Disaster Tweets (disaster_small_transformer_v1).

Defines:
  - DEVICE resolution
  - PrecisionConfig (fp16_amp by default, fp32 fallback)
  - selective torch.autocast forward+loss helper
  - GradScaler factory (backward/step run outside autocast)
  - unscale-before-clip helper
  - TF32 disable helper (matmul + cudnn)
  - precision logging (single line, no per-batch noise)
  - PrecisionRuntime bundle for the training loop to consume directly

Does NOT redefine: model family, loss, optimizer, scheduler, batch size, epochs,
data features, validation metric, or submission logic.
No side effects at import time; every helper is called from the training
entrypoint under `if __name__ == "__main__":`.
"""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Policy constants (mirror Pipeline Decision Trace)
# ---------------------------------------------------------------------------
MODEL_FAMILY_PRECISION = "disaster_small_transformer_v1"
PRECISION_POLICY = "fp16_amp"  # from datatype_precision.precision_policy
PRECISION_MODEL_ADAPTATION = "none"  # decision trace: no model replacement
ALLOWED_POLICIES = ("fp32", "disabled", "fp16_amp")
DEFAULT_FALLBACK = "fp32 or disabled AMP if the selected precision path is unsupported"

# BF16 and TF32 are explicitly disallowed in the current normal mode.
_FORBIDDEN_POLICIES = ("bf16", "tf32", "fp8", "nvfp4", "mxfp8")


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class PrecisionConfig:
    """Resolved precision configuration for one training run."""

    device: torch.device
    policy: str  # requested policy
    effective_policy: str  # policy actually used
    use_amp: bool
    amp_dtype: torch.dtype
    use_tf32: bool
    use_scaler: bool
    reason: str = ""
    fallback: str = DEFAULT_FALLBACK
    fallback_triggered: bool = False
    # Runtime diagnostics (updated by PrecisionRuntime / training loop)
    forward_autocast_used: bool = False
    nonfinite_events: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["device"] = str(self.device)
        d["amp_dtype"] = str(self.amp_dtype)
        return d


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------
def resolve_device(preferred: Optional[str] = None) -> torch.device:
    """Resolve the execution device. Scheduler owns assignment; this is a hint."""
    if preferred:
        try:
            dev = torch.device(preferred)
            if dev.type == "cuda" and not torch.cuda.is_available():
                return torch.device("cpu")
            return dev
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# TF32 controls (must be off for the current normal mode)
# ---------------------------------------------------------------------------
def disable_tf32() -> None:
    """Disable TF32 on matmul and cudnn (best-effort, no-op if unsupported)."""
    try:
        torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = False  # type: ignore[attr-defined]
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Autocast + scaler plumbing
# ---------------------------------------------------------------------------
def _cuda_autocast(dtype: torch.dtype):
    """Best-effort CUDA autocast context (fallback to legacy API on old torch)."""
    try:
        return torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(dtype=dtype, enabled=True)


def autocast_context(cfg: PrecisionConfig):
    """Return a context manager for the forward+loss autocast region.

    Only the forward pass and the compatible loss computation should live inside
    this context. Backward, optimizer.step, metric accumulation, and any FP32
    reductions must run outside.
    """
    if cfg.use_amp and cfg.device.type == "cuda" and cfg.effective_policy == "fp16_amp":
        return _cuda_autocast(cfg.amp_dtype)
    return nullcontext()


def build_grad_scaler(cfg: PrecisionConfig):
    """Create a GradScaler when FP16 AMP is active on CUDA, else None."""
    if not (cfg.use_amp and cfg.use_scaler and cfg.device.type == "cuda"):
        return None
    if cfg.effective_policy != "fp16_amp":
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


# ---------------------------------------------------------------------------
# Model preparation (kept FP32 — no half/fp16 cast of parameters)
# ---------------------------------------------------------------------------
def prepare_model_for_precision(model: nn.Module, cfg: PrecisionConfig) -> nn.Module:
    """Validate the Stage 1 model is staying in FP32 parameters/buffers.

    Only precision-safe adaptations are allowed here; we do NOT replace modules
    (`precision_model_adaptation=none` in the decision trace) and we do NOT call
    `model.half()`. If a parameter is found in a forbidden half dtype we restore
    it to FP32, which preserves the Stage 1 model family and output interface.
    """
    for name, p in model.named_parameters():
        if p.dtype in (torch.float16, torch.bfloat16):
            with torch.no_grad():
                p.data = p.data.float()
        # Ensure parameters are actually FP32 going in.
        if p.dtype not in (torch.float32,):
            raise RuntimeError(
                f"parameter {name} has dtype {p.dtype}; expected float32 for fp16_amp policy"
            )
    for name, b in model.named_buffers():
        if b.dtype in (torch.float16, torch.bfloat16):
            model.register_buffer(name, b.float())
    return model


# ---------------------------------------------------------------------------
# Forward + loss under autocast (returns FP32 logits/loss)
# ---------------------------------------------------------------------------
def precision_forward_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    cfg: PrecisionConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run forward + loss inside autocast; return logits/loss in FP32.

    Integer index tensors (`input_ids`, `attention_mask`) are preserved as long.
    Targets are float32 already; they will pass through the loss under autocast
    automatically (BCEWithLogitsLoss is autocast-compatible) and we cast the
    result back to FP32 for downstream math and metric accumulation.
    """
    with autocast_context(cfg):
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, targets)
    # Metrics and downstream reductions live in FP32.
    logits_fp32 = logits.float()
    loss_fp32 = loss.float()
    if cfg.use_amp:
        cfg.forward_autocast_used = True
    return logits_fp32, loss_fp32


def check_finite_for_fallback(loss: torch.Tensor, cfg: PrecisionConfig) -> bool:
    """Return True if loss is finite; flip the fallback flag otherwise."""
    ok = bool(torch.isfinite(loss).all().item())
    if not ok:
        cfg.nonfinite_events += 1
        if cfg.effective_policy != "fp32":
            cfg.fallback_triggered = True
            cfg.effective_policy = "fp32"
            cfg.use_amp = False
            cfg.use_scaler = False
            cfg.reason = cfg.reason + " | runtime fallback to fp32 (non-finite loss)"
    return ok


# ---------------------------------------------------------------------------
# Scaled backward / step / unscale-before-clip
# ---------------------------------------------------------------------------
def scaled_backward(loss: torch.Tensor, scaler) -> None:
    """Backward must run OUTSIDE autocast; scale only for FP16."""
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()


def unscale_and_clip(model: nn.Module, scaler, max_norm: Optional[float]) -> None:
    """Unscale grads (FP16) and then clip; no-op if max_norm is None."""
    if max_norm is None:
        return
    if scaler is not None:
        scaler.unscale_(model)  # must be called before clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)


def scaler_step_and_update(optimizer, scaler) -> bool:
    """Run optimizer step (scaled if FP16). Returns True if a step was applied."""
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
        return True
    optimizer.step()
    return True


# ---------------------------------------------------------------------------
# Runtime bundle (consumed by the training loop)
# ---------------------------------------------------------------------------
class PrecisionRuntime:
    """Bundle of resolved precision config + scaler for the training loop.

    The training loop is expected to:
        runtime = PrecisionRuntime(cfg)
        with autocast via runtime.forward(...)  # or precision_forward_loss
        runtime.backward(loss)
        runtime.unscale_and_clip(model, max_norm)
        runtime.step(optimizer)
    """

    def __init__(self, cfg: PrecisionConfig):
        self.cfg = cfg
        self.scaler = build_grad_scaler(cfg)

    # Pass-through helpers that respect runtime drift (e.g., fallback triggered)
    def forward(self, model, input_ids, attention_mask, targets, criterion):
        return precision_forward_loss(
            model, input_ids, attention_mask, targets, criterion, self.cfg
        )

    def backward(self, loss: torch.Tensor) -> None:
        scaled_backward(loss, self.scaler)

    def unscale_and_clip(self, model: nn.Module, max_norm: Optional[float]) -> None:
        unscale_and_clip(model, self.scaler, max_norm)

    def step(self, optimizer) -> bool:
        return scaler_step_and_update(optimizer, self.scaler)

    def state_dict(self) -> Dict[str, Any]:
        sd: Dict[str, Any] = {
            "cfg": self.cfg.to_dict(),
            "scaler": (self.scaler.state_dict() if self.scaler is not None else None),
        }
        return sd

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        if not isinstance(sd, dict):
            return
        if self.scaler is not None and sd.get("scaler") is not None:
            try:
                self.scaler.load_state_dict(sd["scaler"])
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Policy resolution
# ---------------------------------------------------------------------------
def _can_use_fp16_amp(device: torch.device) -> bool:
    """FP16 AMP on CUDA only; any modern CUDA GPU qualifies (A10 cc8.6 included)."""
    if device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        # Touch device to make sure it's healthy; capability check is informational.
        torch.cuda.get_device_capability(device)
        return True
    except Exception:
        return False


def resolve_precision(
    preferred_device: Optional[str] = None,
    requested_policy: Optional[str] = None,
    force_fp32: bool = False,
    context: Optional[Dict[str, Any]] = None,
) -> PrecisionConfig:
    """Resolve precision for the run. Pure function; call from the training entrypoint.

    Order of precedence:
      1. explicit `requested_policy` argument
      2. `context['precision_policy']` (from Pipeline Decision Trace)
      3. module constant `PRECISION_POLICY`
    Any unsupported/forbidden policy downgrades to `fp32`.
    """
    device = resolve_device(preferred_device)

    policy = (
        requested_policy or (context or {}).get("precision_policy") or PRECISION_POLICY
    )
    if isinstance(policy, str) and policy.lower() in _FORBIDDEN_POLICIES:
        policy = "fp32"
    if policy not in ALLOWED_POLICIES:
        policy = "fp32"

    cpu_only = device.type != "cuda"

    if cpu_only or force_fp32 or policy in ("fp32", "disabled"):
        cfg = PrecisionConfig(
            device=device,
            policy=policy,
            effective_policy="fp32",
            use_amp=False,
            amp_dtype=torch.float32,
            use_tf32=False,
            use_scaler=False,
            reason=(
                "CPU or explicit fp32: AMP/scaling disabled (CPU preflight uses FP32)."
                if cpu_only or force_fp32
                else "Requested policy disabled by caller."
            ),
            fallback=DEFAULT_FALLBACK,
            fallback_triggered=(policy not in ("fp32", "disabled")),
        )
    else:
        if not _can_use_fp16_amp(device):
            # Should be unreachable given `cpu_only` check above, but keep as
            # a defensive fallback so we never silently enable AMP on a broken
            # CUDA path.
            cfg = PrecisionConfig(
                device=device,
                policy="fp16_amp",
                effective_policy="fp32",
                use_amp=False,
                amp_dtype=torch.float32,
                use_tf32=False,
                use_scaler=False,
                reason="FP16 AMP unsupported on this device; falling back to FP32.",
                fallback=DEFAULT_FALLBACK,
                fallback_triggered=True,
            )
        else:
            cfg = PrecisionConfig(
                device=device,
                policy="fp16_amp",
                effective_policy="fp16_amp",
                use_amp=True,
                amp_dtype=torch.float16,
                use_tf32=False,
                use_scaler=True,
                reason=(
                    "Selective FP16 AMP: autocast around forward+loss only; "
                    "backward/optimizer/metrics in FP32; GradScaler with unscale "
                    "before clipping. TF32 disabled."
                ),
                fallback=DEFAULT_FALLBACK,
                fallback_triggered=False,
            )

    # Always enforce TF32 = off (normal mode forbids it).
    disable_tf32()
    return cfg


# ---------------------------------------------------------------------------
# Precision logging (single line; no batch-level noise)
# ---------------------------------------------------------------------------
def format_precision_banner(cfg: PrecisionConfig) -> str:
    return (
        f"[precision] device={cfg.device} policy={cfg.policy} "
        f"effective={cfg.effective_policy} amp={cfg.use_amp} "
        f"amp_dtype={cfg.amp_dtype} scaler={cfg.use_scaler} tf32={cfg.use_tf32} "
        f"fallback_triggered={cfg.fallback_triggered} reason={cfg.reason}"
    )


def log_precision(cfg: PrecisionConfig, logger=None) -> str:
    """Print (or log) one precision banner. Safe to call once per run."""
    msg = format_precision_banner(cfg)
    if logger is not None:
        try:
            logger.info(msg)
            return msg
        except Exception:
            pass
    print(msg)
    return msg

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
    "output_interface": "float logit per row -> sigmoid -> threshold -> integer {0,1} written as target column alongside id",
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
    "optimizer": "AdamW, lr 2e-5 for encoder / 1e-3 for head, weight_decay 0.01",
    "scheduler": "linear warmup (10%) then linear decay",
    "batch_size_policy": "fixed",
    "dataloader_policy": "num_workers>0, pin_memory=True, persistent_workers when reused, non_blocking=True device copies; dynamic padding acceptable",
    "fallbacks": [
      "if AMP unstable or slower, run FP32; if pretrained weights unavailable, fall back to small_transformer or TF-IDF + linear model with same splits and I/O"
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
      "no matching profiles for current hardware; graph_evidence empty so recommendations are unverified candidates"
    ]
  }
}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=## Plan **WHAT makes this task unique:** Disaster-tweet classification is a *context-dependent* short-text problem where lexical cues (e.g.; purpose=Define the Stage 1 candidate target for later precision and training stages.; hardware_keys=pos_weight, vocab_size, max_len, d_model, num_layers, dim_feedforward, available_transformer_or_sequence_baseline, disaster_small_transformer_v1
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=added `PrecisionConfig`, runtime AMP/autocast/GradScaler/TF32 helpers and a scalar-loss precision logger instead of leaving precision implicit in the training loop.; purpose=keeps the `disaster_small_transformer_v1` candidate target intact while making the fp16_amp policy selective (autocast only around forward+loss; scaler/unscale outside) and explicitly reversible to FP32.; hardware_keys=precision_policy, precision_model_adaptation, fallback_policy, fp16_amp, use_amp, use_tf32, grad_scaler
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=added a non-invasive `prepare_model_for_precision` that asserts FP32 parameter dtype instead of `model.half()`; no precision-required module replacement was added since `precision_model_adaptation=none`.; purpose=prevents accidental whole-pipeline half casting that would break BCE logits reductions and metric accumulation.; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=added a CPU-only preflight preference (`force_fp32_on_cpu`) plus a non-finite-loss trigger to downgrade to FP32 at runtime.; purpose=implements the decision trace's stated fallback ("fall back to fp32 or disabled AMP if the selected precision path is unsupported") without changing any other Stage 1/3 decision.; hardware_keys=none
  - Stage 3 training / training_evaluation: baseline_change=## Plan **WHAT makes this task unique:** F1 on disaster tweets is threshold-sensitive on a small (6090) labeled set with class imbalance and noisy social-media text; the from-scratch small transformer defined in Stage...; purpose=Train and evaluate the Stage 1 candidate using the selected precision policy.; hardware_keys=steps_per_epoch, script_scheduler_context, MLEVOLVE_EPOCH_METRIC

# Step results

        ### Step 1: model_design
        **Plan:** ## Plan

**WHAT makes this task unique:** Disaster-tweet classification is a *context-dependent* short-text problem where lexical cues (e.g. "ablaze", "evacuate") co-occur with noisy social-media artifacts (URLs, @mentions, slang) and a categorical `keyword` field; F1 favors calibrated decision thresholds over raw accuracy, and the 6090-example train set is small enough that regularization matters more than capacity.

**WHY this approach:** I will keep the pipeline in the transformer family required by the decision trace but adapt it to the small-data regime: (a) a compact ~1–2 M parameter transformer encoder trained from scratch with a *learned* word-level vocabulary fit on the training split only, (b) keyword/location folded into the text stream with sentinel tokens (`keymark`, `locmark`) so the transformer sees them in-context instead of as a separate one-hot branch, (c) mean-pooling over non-pad tokens with per-batch dynamic padding so the transformer (and later FP16 AMP path) sees dense, tensor-core-friendly tensors, (d) class-imbalance handled via `pos_weight` in BCEWithLogitsLoss and a post-hoc threshold search on the internal validation split to directly maximize F1. To keep the file robust to offline environments I additionally expose a fully-defined fallback tokenizer/model config so nothing depends on a Hugging Face download.

**HOW I will validate:** Internal stratified 85/15 split (seed 42) — if val F1 after threshold tuning is meaningfully above the naive majority-class F1 (~0.57) and the model overfits only mildly (train-val gap bounded), the transformer branch is validated; if val F1 collapses or loss diverges, I will fall back to a TF-IDF + linear model on the same split, using the same text prep.

**Memory / batch-elasticity strategy:** static per-batch dynamic padding (no pack-to-max), so physical batch size can be reduced for the preflight without changing model geometry; `vocab_size`, `max_len`, `d_model`, `nhead`, `num_layers`, `dim_feedforward` remain configurable context keys.

**Backend contract:** the subprocess owns nothing about workers, device placement, CUDA graphs, memory limits, torch threads, or precision policy — those are scheduler-owned and Stage 2/3-owned. The `CandidateAdapter` is CPU-safe, no-argument, and only exposes the real model/loss/optimizer/batch builders.

**Note board:**
- baseline_change: replaced a generic `available_transformer_or_sequence_baseline` with a concrete small from-scratch word-transformer (`disaster_small_transformer_v1`) whose vocabulary and `pos_weight` are fit only on the training split; keyword/location are injected as sentinel tokens into the text stream. | purpose: honors the `text`/`classification` family while explicitly using the `keyword`/`location` columns the task advertises, keeps parameters and vocab lightweight enough to train within the hour budget. | hardware_keys: none
- baseline_change: added a keyword-aware text builder, an explicit class-imbalance-aware loss factory, and a validation-split F1 threshold search as first-class model-design artifacts. | purpose: F1 is threshold-sensitive; making the criterion and threshold selection part of the design (not training loop) aligns Stage 2/3 with the same decision rule and prevents metric-vs-loss drift. | hardware_keys: none
- baseline_change: exposed `input_ids`/`attention_mask` tensors plus an `inputs` tuple with dynamic padding and `torch.long` dtypes, and a CPU-safe `CandidateAdapter`. | purpose: enables non-blocking CUDA transfers, tensor-core-friendly dense batches under optional FP16 AMP, and satisfies the preflight contract without downloading weights. | hardware_keys: none
        **Code:**
        ```python
"""
Stage 1: model_design for Disaster Tweets (text classification, binary F1).

Defines:
  - Data loading + text cleaning + keyword/location fusion
  - Word-level tokenizer (fit on TRAIN ONLY)
  - torch Dataset + dynamic-padding collate
  - Small transformer classifier (word embeddings + TransformerEncoder + mean pool)
  - BCEWithLogits criterion with pos_weight
  - Validation F1 threshold search
  - CandidateAdapter for CPU preflight (real model, no downloads)

Does NOT define: training loop, optimizer schedule, precision policy, batch size,
epochs, workers, checkpointing, or submission writing.
"""

import os
import re
import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------
MODEL_FAMILY = "disaster_small_transformer_v1"
SEED = 42
INPUT_DIR = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")

MAX_SEQ_LEN = 64
MAX_VOCAB = 30000
MIN_FREQ = 2
VAL_SIZE = 0.15

PAD_IDX = 0
UNK_IDX = 1
SPECIAL_TOKENS: Dict[str, int] = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}

# ---------------------------------------------------------------------------
# Text cleaning / feature fusion
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NUM_RE = re.compile(r"\d+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase + normalize URLs/mentions/hashtags/numbers into sentinel words."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = _URL_RE.sub(" urlmark ", t)
    t = _MENTION_RE.sub(" usermark ", t)
    t = _HASHTAG_RE.sub(r" \1 ", t)
    t = _NUM_RE.sub(" nummark ", t)
    t = _NON_ALNUM_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def build_model_input(row) -> str:
    """Fuse text + keyword + location into a single stream with in-context sentinels."""
    parts: List[str] = []
    text = row.get("text", "")
    if isinstance(text, str) and text.strip():
        parts.append(text)
    kw = row.get("keyword", None)
    if isinstance(kw, str) and kw.strip():
        parts.append("keymark " + kw.replace("%20", " "))
    loc = row.get("location", None)
    if isinstance(loc, str) and loc.strip():
        parts.append("locmark " + loc)
    return " ".join(parts)


def prepare_texts(df: pd.DataFrame) -> List[str]:
    return [clean_text(build_model_input(row)) for _, row in df.iterrows()]


# ---------------------------------------------------------------------------
# Tokenizer (train-only vocabulary)
# ---------------------------------------------------------------------------
class SimpleWordTokenizer:
    """Whitespace tokenizer with a train-fit frequency vocabulary."""

    def __init__(self, max_vocab: int = MAX_VOCAB, min_freq: int = MIN_FREQ):
        self.max_vocab = int(max_vocab)
        self.min_freq = int(min_freq)
        self.vocab: Dict[str, int] = dict(SPECIAL_TOKENS)

    def fit(self, texts: Sequence[str]) -> "SimpleWordTokenizer":
        counter: Counter = Counter()
        for t in texts:
            if isinstance(t, str):
                counter.update(t.split())
        for tok, cnt in counter.most_common():
            if cnt < self.min_freq:
                break
            if tok in self.vocab:
                continue
            if len(self.vocab) >= self.max_vocab:
                break
            self.vocab[tok] = len(self.vocab)
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
        if not isinstance(text, str):
            return [PAD_IDX]
        toks = text.split()
        ids = [self.vocab.get(t, UNK_IDX) for t in toks]
        if len(ids) == 0:
            ids = [UNK_IDX]
        if len(ids) > max_len:
            ids = ids[:max_len]
        return ids

    def state_dict(self) -> Dict:
        return {
            "vocab": dict(self.vocab),
            "max_vocab": self.max_vocab,
            "min_freq": self.min_freq,
        }

    @classmethod
    def from_state_dict(cls, sd: Dict) -> "SimpleWordTokenizer":
        tok = cls(
            max_vocab=sd.get("max_vocab", MAX_VOCAB),
            min_freq=sd.get("min_freq", MIN_FREQ),
        )
        tok.vocab = dict(sd["vocab"])
        return tok


def build_tokenizer_on_train(
    train_texts: Sequence[str], max_vocab: int = MAX_VOCAB, min_freq: int = MIN_FREQ
) -> SimpleWordTokenizer:
    return SimpleWordTokenizer(max_vocab=max_vocab, min_freq=min_freq).fit(train_texts)


# ---------------------------------------------------------------------------
# Dataset + collate
# ---------------------------------------------------------------------------
class TweetDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[float]],
        tokenizer: SimpleWordTokenizer,
        max_len: int = MAX_SEQ_LEN,
    ):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tok = tokenizer
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.tok.encode(self.texts[idx], self.max_len)
        item: Dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(ids, dtype=torch.long)
        }
        if self.labels is not None:
            item["label"] = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return item


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Dynamic padding; emits input_ids (long) + attention_mask (long)."""
    seqs = [b["input_ids"] for b in batch]
    L = max(s.numel() for s in seqs)
    B = len(seqs)
    input_ids = torch.zeros(B, L, dtype=torch.long)
    attention_mask = torch.zeros(B, L, dtype=torch.long)
    for i, s in enumerate(seqs):
        n = s.numel()
        input_ids[i, :n] = s
        attention_mask[i, :n] = 1
    out: Dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch], dim=0)
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DisasterTransformer(nn.Module):
    """
    Compact word-transformer for short-text binary classification.

    - Learned word embeddings + sinusoidal-free learned positional embeddings
    - N pre-LN TransformerEncoder blocks (GELU, configurable dims)
    - Mean pooling over non-pad tokens -> linear head emits a single logit
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int = MAX_SEQ_LEN,
        d_model: int = 192,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 384,
        dropout: float = 0.15,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.vocab_size = int(vocab_size)
        self.max_len = int(max_len)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout_p = float(dropout)

        self.token_emb = nn.Embedding(
            self.vocab_size, self.d_model, padding_idx=PAD_IDX
        )
        self.pos_emb = nn.Embedding(self.max_len, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model, eps=1e-6)
        self.dropout = nn.Dropout(self.dropout_p)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)

        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.d_model, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_emb.weight, std=0.02)
        with torch.no_grad():
            self.token_emb.weight[PAD_IDX].fill_(0.0)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be [B, L], got shape {tuple(input_ids.shape)}"
            )
        B, L = input_ids.shape
        L = min(L, self.max_len)
        input_ids = input_ids[:, :L]

        if attention_mask is None:
            attention_mask = (input_ids != PAD_IDX).long()
        else:
            attention_mask = attention_mask[:, :L]
            if attention_mask.dtype != torch.long:
                attention_mask = attention_mask.long()

        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        pos = pos.clamp_(max=self.max_len - 1)

        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.input_norm(x)
        x = self.dropout(x)

        # True = ignore in src_key_padding_mask
        key_padding_mask = attention_mask == 0
        # Guard all-padding rows to avoid NaN in attention softmax
        all_pad = key_padding_mask.all(dim=1)
        if bool(all_pad.any()):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_pad, 0] = False

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        mask_f = attention_mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_(min=1.0)
        pooled = (x * mask_f).sum(dim=1) / denom

        logits = self.head(pooled).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# Loss / criterion
# ---------------------------------------------------------------------------
class BCEWithLogitsCriterion(nn.Module):
    """Binary cross-entropy on logits with optional per-class pos_weight."""

    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer(
                "pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32)
            )
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        logits = logits.float()
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


def compute_pos_weight(labels: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=np.float32)
    n_pos = float((y > 0.5).sum())
    n_neg = float(len(y) - n_pos)
    if n_pos <= 0.0:
        return 1.0
    return n_neg / n_pos


def get_criterion(pos_weight: Optional[float] = None) -> BCEWithLogitsCriterion:
    return BCEWithLogitsCriterion(pos_weight=pos_weight)


# ---------------------------------------------------------------------------
# Output interface helpers
# ---------------------------------------------------------------------------
def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.float())


def find_best_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    lo: float = 0.15,
    hi: float = 0.85,
    step: float = 0.01,
) -> Tuple[float, float]:
    """Grid search the decision threshold that maximizes binary F1."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(lo, hi + 1e-9, step):
        preds = (probs >= t).astype(np.int64)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_t = float(t)
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Data loading / splitting
# ---------------------------------------------------------------------------
def load_dataframes(input_dir: str = INPUT_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(input_dir, "train.csv")
    test_path = os.path.join(input_dir, "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def stratified_split(
    train_df: pd.DataFrame, val_size: float = VAL_SIZE, seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = np.arange(len(train_df))
    strat = train_df["target"].values
    tr_idx, va_idx = train_test_split(
        idx, test_size=val_size, random_state=seed, stratify=strat
    )
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)
    return tr_df, va_df


def pipeline_summary() -> Dict:
    """Small machine-readable descriptor for later stages."""
    return {
        "model_family": MODEL_FAMILY,
        "modality": "text",
        "target_type": "classification",
        "output_interface": "logit -> sigmoid -> threshold -> {0,1}",
        "loss": "BCEWithLogitsLoss (pos_weight from train labels)",
        "max_seq_len": MAX_SEQ_LEN,
        "default_dims": {
            "d_model": 192,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 384,
        },
        "input_dir": INPUT_DIR,
        "seed": SEED,
        "val_size": VAL_SIZE,
    }


# ---------------------------------------------------------------------------
# CPU preflight: CandidateAdapter
# ---------------------------------------------------------------------------
class CandidateAdapter:
    """
    Real-model CPU preflight. Uses the same model/tokenizer/loss/text-prep as the
    full pipeline. No downloads, no CUDA required, no side effects on import.
    """

    DEFAULT_CONTEXT: Dict[str, object] = {
        "vocab_size": 5000,
        "max_len": 32,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "pos_weight": 1.0,
        "seed": SEED,
        "input_dir": INPUT_DIR,
    }

    def __init__(self):
        self._tokenizer: Optional[SimpleWordTokenizer] = None
        self._cached_train_texts: Optional[List[str]] = None
        self._cached_train_labels: Optional[List[int]] = None

    # -- context handling ---------------------------------------------------
    def _merge(self, context: Optional[Dict]) -> Dict[str, object]:
        cfg = dict(self.DEFAULT_CONTEXT)
        if context:
            for k, v in context.items():
                if v is not None:
                    cfg[k] = v
        return cfg

    # -- data ---------------------------------------------------------------
    def _load_train_hint(
        self, input_dir: str, n_rows: int
    ) -> Tuple[List[str], List[int]]:
        train_path = os.path.join(input_dir, "train.csv")
        texts: List[str] = []
        labels: List[int] = []
        if os.path.exists(train_path):
            try:
                df = pd.read_csv(train_path)
                head = df.head(max(n_rows, 4))
                for _, row in head.iterrows():
                    texts.append(clean_text(build_model_input(row)))
                    labels.append(int(row["target"]))
            except Exception:
                texts, labels = [], []
        if not texts:
            texts = [
                "sample disaster tweet text keymark locmark",
                "calm day no emergency happening now",
            ]
            labels = [0, 0]
        return texts, labels

    def _get_tokenizer(self, input_dir: str, vocab_size: int) -> SimpleWordTokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        texts, labels = self._load_train_hint(input_dir, n_rows=500)
        self._cached_train_texts = texts
        self._cached_train_labels = labels
        tok = SimpleWordTokenizer(max_vocab=max(int(vocab_size), 64), min_freq=1).fit(
            texts
        )
        self._tokenizer = tok
        return tok

    # -- model --------------------------------------------------------------
    def build_model(self, context: Optional[Dict] = None) -> nn.Module:
        cfg = self._merge(context)
        input_dir = str(cfg["input_dir"])
        tok = self._get_tokenizer(input_dir, int(cfg["vocab_size"]))
        vs = max(int(cfg["vocab_size"]), tok.vocab_size)

        # Guarantee nhead divides d_model even if caller supplies odd values
        d_model = int(cfg["d_model"])
        nhead = int(cfg["nhead"])
        if d_model % nhead != 0:
            d_model = (d_model // nhead) * nhead

        model = DisasterTransformer(
            vocab_size=vs,
            max_len=int(cfg["max_len"]),
            d_model=d_model,
            nhead=nhead,
            num_layers=int(cfg["num_layers"]),
            dim_feedforward=int(cfg["dim_feedforward"]),
            dropout=float(cfg["dropout"]),
        )
        return model

    def build_optimizer(
        self, model: nn.Module, context: Optional[Dict] = None
    ) -> torch.optim.Optimizer:
        # Real optimizer object; schedule/hyperparams are Stage-3 concerns, so we
        # mirror the decision trace defaults here (encoder-style lr for the whole net).
        return torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # -- batches ------------------------------------------------------------
    def _build_batch(
        self, scenario: Optional[Dict], device: torch.device, default_batch: int = 8
    ) -> Dict[str, object]:
        scenario = scenario or {}
        cfg = self._merge(scenario)
        input_dir = str(cfg["input_dir"])
        max_len = int(cfg["max_len"])
        batch_size = int(scenario.get("batch_size", default_batch))
        batch_size = max(batch_size, 1)

        texts, labels = self._load_train_hint(input_dir, n_rows=batch_size)
        # Make exactly `batch_size` rows
        while len(texts) < batch_size:
            texts.append("sample tweet about a disaster keymark locmark")
            labels.append(0)
        texts = texts[:batch_size]
        labels = labels[:batch_size]

        tok = self._get_tokenizer(input_dir, int(cfg["vocab_size"]))

        enc = [tok.encode(t, max_len) for t in texts]
        L = max(len(e) for e in enc)

        # Respect an explicit fixture length hint (non-batch shape (L,)).
        fixture = scenario.get("fixture", None)
        if isinstance(fixture, dict) and "input_ids" in fixture:
            try:
                fx = fixture["input_ids"]
                fx_len = int(fx[-1]) if isinstance(fx, (list, tuple)) else int(fx)
                L = min(L, max(fx_len, 1))
                enc = [e[:L] for e in enc]
            except Exception:
                pass
        L = max(L, 1)

        input_ids = torch.zeros(batch_size, L, dtype=torch.long, device=device)
        attention_mask = torch.zeros(batch_size, L, dtype=torch.long, device=device)
        for i, e in enumerate(enc):
            n = min(len(e), L)
            if n > 0:
                input_ids[i, :n] = torch.tensor(e[:n], dtype=torch.long, device=device)
                attention_mask[i, :n] = 1

        target = torch.tensor(labels, dtype=torch.float32, device=device)

        return {
            # Contract-recognized keys
            "inputs": (input_ids, attention_mask),
            "target": target,
            # Named tensors preserved for fixture contracts
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def build_train_batch(self, scenario, device) -> Dict[str, object]:
        return self._build_batch(scenario, device)

    def build_validation_batch(self, scenario, device) -> Dict[str, object]:
        return self._build_batch(scenario, device)

    # -- steps --------------------------------------------------------------
    def _criterion_for(self, context: Optional[Dict]) -> BCEWithLogitsCriterion:
        cfg = self._merge(context)
        pw = cfg.get("pos_weight", None)
        pw = float(pw) if pw is not None else None
        return BCEWithLogitsCriterion(pos_weight=pw)

    def training_step(
        self, model: nn.Module, batch: Dict[str, object], context: Optional[Dict] = None
    ) -> torch.Tensor:
        input_ids, attention_mask = batch["inputs"]  # type: ignore[misc]
        target = batch["target"]
        logits = model(input_ids, attention_mask)  # type: ignore[arg-type]
        loss = self._criterion_for(context)(logits, target)  # type: ignore[arg-type]
        return loss

    def validation_step(
        self, model: nn.Module, batch: Dict[str, object], context: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask = batch["inputs"]  # type: ignore[misc]
        target = batch["target"]
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                logits = model(input_ids, attention_mask)  # type: ignore[arg-type]
                probs = torch.sigmoid(logits.float())
        finally:
            if was_training:
                model.train()
        return {"probs": probs, "target": target}  # type: ignore[dict-item]

```
        
        ### Step 2: datatype_precision
        **Plan:** ## Plan

This step locks in the Stage 2 precision policy without touching the Stage 1 model family, loss, features, or output interface. I will expose a small, import-safe `PrecisionConfig` + helpers that (a) resolve `DEVICE`/precision at runtime, (b) wrap *only* the forward+loss in `torch.autocast('cuda', fp16)`, (c) hand back a `GradScaler` so backward/step run outside autocast, with `unscale_` before clipping, (d) force-disable TF32 for matmul/cudnn, and (e) provide a single-line precision logger. Fallback is explicit: CPU or unsupported FP16 → pure FP32/no scaler, matching the decision trace's `fallback_policy`.

The physics: this is a sub-2M-parameter word transformer on ~5k training examples — FP16 tensor cores on the A10 help throughput but FP32 is cheap here, so AMP is treated as **optional**; the loss (`BCEWithLogitsLoss`) and the metric are guarded in FP32, and any non-finite loss/grad triggers a graceful FP32 fallback. Model batch elasticity is preserved (dynamic padding, `input_ids` long, `attention_mask` long) — nothing in this stage changes the physical/effective batch size.

Note board:
- baseline_change: added `PrecisionConfig`, runtime AMP/autocast/GradScaler/TF32 helpers and a scalar-loss precision logger instead of leaving precision implicit in the training loop. | purpose: keeps the `disaster_small_transformer_v1` candidate target intact while making the fp16_amp policy selective (autocast only around forward+loss; scaler/unscale outside) and explicitly reversible to FP32. | hardware_keys: precision_policy, precision_model_adaptation, fallback_policy, fp16_amp, use_amp, use_tf32, grad_scaler
- baseline_change: added a non-invasive `prepare_model_for_precision` that asserts FP32 parameter dtype instead of `model.half()`; no precision-required module replacement was added since `precision_model_adaptation=none`. | purpose: prevents accidental whole-pipeline half casting that would break BCE logits reductions and metric accumulation. | hardware_keys: none
- baseline_change: added a CPU-only preflight preference (`force_fp32_on_cpu`) plus a non-finite-loss trigger to downgrade to FP32 at runtime. | purpose: implements the decision trace's stated fallback ("fall back to fp32 or disabled AMP if the selected precision path is unsupported") without changing any other Stage 1/3 decision. | hardware_keys: none
        **Code:**
        ```python
"""
Stage 2: datatype_precision for Disaster Tweets (disaster_small_transformer_v1).

Defines:
  - DEVICE resolution
  - PrecisionConfig (fp16_amp by default, fp32 fallback)
  - selective torch.autocast forward+loss helper
  - GradScaler factory (backward/step run outside autocast)
  - unscale-before-clip helper
  - TF32 disable helper (matmul + cudnn)
  - precision logging (single line, no per-batch noise)
  - PrecisionRuntime bundle for the training loop to consume directly

Does NOT redefine: model family, loss, optimizer, scheduler, batch size, epochs,
data features, validation metric, or submission logic.
No side effects at import time; every helper is called from the training
entrypoint under `if __name__ == "__main__":`.
"""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Policy constants (mirror Pipeline Decision Trace)
# ---------------------------------------------------------------------------
MODEL_FAMILY_PRECISION = "disaster_small_transformer_v1"
PRECISION_POLICY = "fp16_amp"  # from datatype_precision.precision_policy
PRECISION_MODEL_ADAPTATION = "none"  # decision trace: no model replacement
ALLOWED_POLICIES = ("fp32", "disabled", "fp16_amp")
DEFAULT_FALLBACK = "fp32 or disabled AMP if the selected precision path is unsupported"

# BF16 and TF32 are explicitly disallowed in the current normal mode.
_FORBIDDEN_POLICIES = ("bf16", "tf32", "fp8", "nvfp4", "mxfp8")


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class PrecisionConfig:
    """Resolved precision configuration for one training run."""

    device: torch.device
    policy: str  # requested policy
    effective_policy: str  # policy actually used
    use_amp: bool
    amp_dtype: torch.dtype
    use_tf32: bool
    use_scaler: bool
    reason: str = ""
    fallback: str = DEFAULT_FALLBACK
    fallback_triggered: bool = False
    # Runtime diagnostics (updated by PrecisionRuntime / training loop)
    forward_autocast_used: bool = False
    nonfinite_events: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["device"] = str(self.device)
        d["amp_dtype"] = str(self.amp_dtype)
        return d


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------
def resolve_device(preferred: Optional[str] = None) -> torch.device:
    """Resolve the execution device. Scheduler owns assignment; this is a hint."""
    if preferred:
        try:
            dev = torch.device(preferred)
            if dev.type == "cuda" and not torch.cuda.is_available():
                return torch.device("cpu")
            return dev
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# TF32 controls (must be off for the current normal mode)
# ---------------------------------------------------------------------------
def disable_tf32() -> None:
    """Disable TF32 on matmul and cudnn (best-effort, no-op if unsupported)."""
    try:
        torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = False  # type: ignore[attr-defined]
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Autocast + scaler plumbing
# ---------------------------------------------------------------------------
def _cuda_autocast(dtype: torch.dtype):
    """Best-effort CUDA autocast context (fallback to legacy API on old torch)."""
    try:
        return torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(dtype=dtype, enabled=True)


def autocast_context(cfg: PrecisionConfig):
    """Return a context manager for the forward+loss autocast region.

    Only the forward pass and the compatible loss computation should live inside
    this context. Backward, optimizer.step, metric accumulation, and any FP32
    reductions must run outside.
    """
    if cfg.use_amp and cfg.device.type == "cuda" and cfg.effective_policy == "fp16_amp":
        return _cuda_autocast(cfg.amp_dtype)
    return nullcontext()


def build_grad_scaler(cfg: PrecisionConfig):
    """Create a GradScaler when FP16 AMP is active on CUDA, else None."""
    if not (cfg.use_amp and cfg.use_scaler and cfg.device.type == "cuda"):
        return None
    if cfg.effective_policy != "fp16_amp":
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


# ---------------------------------------------------------------------------
# Model preparation (kept FP32 — no half/fp16 cast of parameters)
# ---------------------------------------------------------------------------
def prepare_model_for_precision(model: nn.Module, cfg: PrecisionConfig) -> nn.Module:
    """Validate the Stage 1 model is staying in FP32 parameters/buffers.

    Only precision-safe adaptations are allowed here; we do NOT replace modules
    (`precision_model_adaptation=none` in the decision trace) and we do NOT call
    `model.half()`. If a parameter is found in a forbidden half dtype we restore
    it to FP32, which preserves the Stage 1 model family and output interface.
    """
    for name, p in model.named_parameters():
        if p.dtype in (torch.float16, torch.bfloat16):
            with torch.no_grad():
                p.data = p.data.float()
        # Ensure parameters are actually FP32 going in.
        if p.dtype not in (torch.float32,):
            raise RuntimeError(
                f"parameter {name} has dtype {p.dtype}; expected float32 for fp16_amp policy"
            )
    for name, b in model.named_buffers():
        if b.dtype in (torch.float16, torch.bfloat16):
            model.register_buffer(name, b.float())
    return model


# ---------------------------------------------------------------------------
# Forward + loss under autocast (returns FP32 logits/loss)
# ---------------------------------------------------------------------------
def precision_forward_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    cfg: PrecisionConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run forward + loss inside autocast; return logits/loss in FP32.

    Integer index tensors (`input_ids`, `attention_mask`) are preserved as long.
    Targets are float32 already; they will pass through the loss under autocast
    automatically (BCEWithLogitsLoss is autocast-compatible) and we cast the
    result back to FP32 for downstream math and metric accumulation.
    """
    with autocast_context(cfg):
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, targets)
    # Metrics and downstream reductions live in FP32.
    logits_fp32 = logits.float()
    loss_fp32 = loss.float()
    if cfg.use_amp:
        cfg.forward_autocast_used = True
    return logits_fp32, loss_fp32


def check_finite_for_fallback(loss: torch.Tensor, cfg: PrecisionConfig) -> bool:
    """Return True if loss is finite; flip the fallback flag otherwise."""
    ok = bool(torch.isfinite(loss).all().item())
    if not ok:
        cfg.nonfinite_events += 1
        if cfg.effective_policy != "fp32":
            cfg.fallback_triggered = True
            cfg.effective_policy = "fp32"
            cfg.use_amp = False
            cfg.use_scaler = False
            cfg.reason = cfg.reason + " | runtime fallback to fp32 (non-finite loss)"
    return ok


# ---------------------------------------------------------------------------
# Scaled backward / step / unscale-before-clip
# ---------------------------------------------------------------------------
def scaled_backward(loss: torch.Tensor, scaler) -> None:
    """Backward must run OUTSIDE autocast; scale only for FP16."""
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()


def unscale_and_clip(model: nn.Module, scaler, max_norm: Optional[float]) -> None:
    """Unscale grads (FP16) and then clip; no-op if max_norm is None."""
    if max_norm is None:
        return
    if scaler is not None:
        scaler.unscale_(model)  # must be called before clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)


def scaler_step_and_update(optimizer, scaler) -> bool:
    """Run optimizer step (scaled if FP16). Returns True if a step was applied."""
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
        return True
    optimizer.step()
    return True


# ---------------------------------------------------------------------------
# Runtime bundle (consumed by the training loop)
# ---------------------------------------------------------------------------
class PrecisionRuntime:
    """Bundle of resolved precision config + scaler for the training loop.

    The training loop is expected to:
        runtime = PrecisionRuntime(cfg)
        with autocast via runtime.forward(...)  # or precision_forward_loss
        runtime.backward(loss)
        runtime.unscale_and_clip(model, max_norm)
        runtime.step(optimizer)
    """

    def __init__(self, cfg: PrecisionConfig):
        self.cfg = cfg
        self.scaler = build_grad_scaler(cfg)

    # Pass-through helpers that respect runtime drift (e.g., fallback triggered)
    def forward(self, model, input_ids, attention_mask, targets, criterion):
        return precision_forward_loss(
            model, input_ids, attention_mask, targets, criterion, self.cfg
        )

    def backward(self, loss: torch.Tensor) -> None:
        scaled_backward(loss, self.scaler)

    def unscale_and_clip(self, model: nn.Module, max_norm: Optional[float]) -> None:
        unscale_and_clip(model, self.scaler, max_norm)

    def step(self, optimizer) -> bool:
        return scaler_step_and_update(optimizer, self.scaler)

    def state_dict(self) -> Dict[str, Any]:
        sd: Dict[str, Any] = {
            "cfg": self.cfg.to_dict(),
            "scaler": (self.scaler.state_dict() if self.scaler is not None else None),
        }
        return sd

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        if not isinstance(sd, dict):
            return
        if self.scaler is not None and sd.get("scaler") is not None:
            try:
                self.scaler.load_state_dict(sd["scaler"])
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Policy resolution
# ---------------------------------------------------------------------------
def _can_use_fp16_amp(device: torch.device) -> bool:
    """FP16 AMP on CUDA only; any modern CUDA GPU qualifies (A10 cc8.6 included)."""
    if device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        # Touch device to make sure it's healthy; capability check is informational.
        torch.cuda.get_device_capability(device)
        return True
    except Exception:
        return False


def resolve_precision(
    preferred_device: Optional[str] = None,
    requested_policy: Optional[str] = None,
    force_fp32: bool = False,
    context: Optional[Dict[str, Any]] = None,
) -> PrecisionConfig:
    """Resolve precision for the run. Pure function; call from the training entrypoint.

    Order of precedence:
      1. explicit `requested_policy` argument
      2. `context['precision_policy']` (from Pipeline Decision Trace)
      3. module constant `PRECISION_POLICY`
    Any unsupported/forbidden policy downgrades to `fp32`.
    """
    device = resolve_device(preferred_device)

    policy = (
        requested_policy or (context or {}).get("precision_policy") or PRECISION_POLICY
    )
    if isinstance(policy, str) and policy.lower() in _FORBIDDEN_POLICIES:
        policy = "fp32"
    if policy not in ALLOWED_POLICIES:
        policy = "fp32"

    cpu_only = device.type != "cuda"

    if cpu_only or force_fp32 or policy in ("fp32", "disabled"):
        cfg = PrecisionConfig(
            device=device,
            policy=policy,
            effective_policy="fp32",
            use_amp=False,
            amp_dtype=torch.float32,
            use_tf32=False,
            use_scaler=False,
            reason=(
                "CPU or explicit fp32: AMP/scaling disabled (CPU preflight uses FP32)."
                if cpu_only or force_fp32
                else "Requested policy disabled by caller."
            ),
            fallback=DEFAULT_FALLBACK,
            fallback_triggered=(policy not in ("fp32", "disabled")),
        )
    else:
        if not _can_use_fp16_amp(device):
            # Should be unreachable given `cpu_only` check above, but keep as
            # a defensive fallback so we never silently enable AMP on a broken
            # CUDA path.
            cfg = PrecisionConfig(
                device=device,
                policy="fp16_amp",
                effective_policy="fp32",
                use_amp=False,
                amp_dtype=torch.float32,
                use_tf32=False,
                use_scaler=False,
                reason="FP16 AMP unsupported on this device; falling back to FP32.",
                fallback=DEFAULT_FALLBACK,
                fallback_triggered=True,
            )
        else:
            cfg = PrecisionConfig(
                device=device,
                policy="fp16_amp",
                effective_policy="fp16_amp",
                use_amp=True,
                amp_dtype=torch.float16,
                use_tf32=False,
                use_scaler=True,
                reason=(
                    "Selective FP16 AMP: autocast around forward+loss only; "
                    "backward/optimizer/metrics in FP32; GradScaler with unscale "
                    "before clipping. TF32 disabled."
                ),
                fallback=DEFAULT_FALLBACK,
                fallback_triggered=False,
            )

    # Always enforce TF32 = off (normal mode forbids it).
    disable_tf32()
    return cfg


# ---------------------------------------------------------------------------
# Precision logging (single line; no batch-level noise)
# ---------------------------------------------------------------------------
def format_precision_banner(cfg: PrecisionConfig) -> str:
    return (
        f"[precision] device={cfg.device} policy={cfg.policy} "
        f"effective={cfg.effective_policy} amp={cfg.use_amp} "
        f"amp_dtype={cfg.amp_dtype} scaler={cfg.use_scaler} tf32={cfg.use_tf32} "
        f"fallback_triggered={cfg.fallback_triggered} reason={cfg.reason}"
    )


def log_precision(cfg: PrecisionConfig, logger=None) -> str:
    """Print (or log) one precision banner. Safe to call once per run."""
    msg = format_precision_banner(cfg)
    if logger is not None:
        try:
            logger.info(msg)
            return msg
        except Exception:
            pass
    print(msg)
    return msg

```
        
        ### Step 3: training_evaluation
        **Plan:** ## Plan

**WHAT makes this task unique:** F1 on disaster tweets is threshold-sensitive on a small (6090) labeled set with class imbalance and noisy social-media text; the from-scratch small transformer defined in Stage 1 has ~1-2M params so overfitting *and* underfitting are both live risks, and the reported metric must be produced by the *same* thresholding rule on validation and test.

**WHY this training recipe fits:** I will train with AdamW at a from-scratch-friendly LR (encoder 2e-4, head 1e-3) + weight decay 0.01, a 10% linear-warmup-then-linear-decay schedule sized to observed `steps_per_epoch`, grad-accum=1 (physical bs 32 → effective 32), early-stopping on validation BCE loss with best-checkpoint restore, and a post-hoc F1 threshold search on the validation probs that is then applied verbatim to test probs — so validation and test share identical binarization. Precision is consumed as-is from Stage 2 (`PrecisionRuntime` → selective FP16 AMP with GradScaler; FP32 fallback on non-finite loss; TF32 stays disabled).

**HOW I will validate:** If val F1 after threshold search materially exceeds the majority-class F1 (~0.57) and the train/val loss gap stays bounded across epochs, the Stage-1 architecture hypothesis holds; a diverging or plateauing val F1 will be recorded and the checkpoint with minimum val loss is used — no threshold tuning on the test split, no re-splitting, and `Final Validation Score` is computed on the hold-out split only.

**Memory / batch-elasticity:** static `PHYSICAL_BATCH_SIZE=32` with dynamic padding per batch; `QUALITY_SAFE_PHYSICAL_BATCH_SIZES=[16,32,64]`, `BATCH_LR_SCALING_POLICY="sqrt"`. Configurable dims (vocab_size, max_len, d_model, nhead, num_layers, dim_feedforward) untouched. Scheduler owns launch, CUDA assignment, thread allocation, memory limits.

**Note board:**
- baseline_change: implemented complete training/validation/test pipeline: AdamW with encoder/head param groups (2e-4 / 1e-3), warmup-then-linear-decay LambdaLR keyed to real steps_per_epoch, grad-accum, best-checkpoint restore, validation F1 threshold search reused verbatim on test. | purpose: honors Stage 1 target (small from-scratch word transformer) and the Pipeline Decision Trace optimizer/scheduler choice, and guarantees validation↔test consistency for the F1 metric. | hardware_keys: physical_batch_size, effective_batch_size, steps_per_epoch, warmup_fraction, weight_decay
- baseline_change: consumed Stage 2 `PrecisionRuntime` (autocast only around forward+loss, GradScaler step outside autocast, unscale-before-clip, non-finite → FP32 fallback) and `TrainingDiagnostics` + `script_scheduler_context` hooks. | purpose: preserves the fp16_amp precision policy without whole-pipeline casting, satisfies the diagnostics/scheduler contract, and makes the run resumable without changing model family or output interface. | hardware_keys: use_amp, use_scaler, precision_policy, safe_point, global_step
- baseline_change: added runtime/epoch logging (elapsed seconds, peak VRAM MB, resolved batch size), `MLEVOLVE_EPOCH_METRIC` parseable line per epoch, and submission at `./submission/submission.csv` with `id,target`. | purpose: gives the scheduler real evidence for future graph optimization while keeping the final line `Final Validation Score: {score}` for the parser. | hardware_keys: peak_vram_mb, resolved_batch_size, elapsed_seconds
        **Code:**
        ```python
"""
Stage 3: training_evaluation for Disaster Tweets (disaster_small_transformer_v1).

Consumes (assumed already defined by prior steps in the same file):
    clean_text, build_model_input, prepare_texts
    SimpleWordTokenizer, build_tokenizer_on_train, TweetDataset, collate_batch
    DisasterTransformer, get_criterion, compute_pos_weight
    find_best_threshold, logits_to_probs
    load_dataframes, stratified_split
    INPUT_DIR, SEED, VAL_SIZE, MAX_SEQ_LEN, MAX_VOCAB, MIN_FREQ
    resolve_device, resolve_precision, PrecisionRuntime, prepare_model_for_precision,
    log_precision, autocast_context, check_finite_for_fallback
    CandidateAdapter

Adds:
    - Optimizer / scheduler / accumulation / early-stopping training loop
    - Validation F1 via threshold search on hold-out split
    - Test inference using the SAME threshold + tokenization pipeline
    - submission.csv writer, TrainingDiagnostics + scheduler safe_point hooks
    - Final line: print(f'Final Validation Score: {score}')
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Optional integrations (safe fallbacks so the file runs standalone)
# ---------------------------------------------------------------------------
try:
    from utils.training_diagnostics import TrainingDiagnostics  # type: ignore
except Exception:  # pragma: no cover

    class TrainingDiagnostics:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def after_update(self):
            pass

        def report(self, **kwargs):
            pass

        def state_dict(self):
            return {}

        def load_state_dict(self, sd):
            pass


try:
    from localml_scheduler.execution.script_context import script_scheduler_context  # type: ignore
except Exception:  # pragma: no cover

    def script_scheduler_context():  # type: ignore
        return None


try:
    from localml_scheduler.domain import SafePointType  # type: ignore
except Exception:  # pragma: no cover

    class SafePointType:  # type: ignore
        BEFORE_TRAIN = "BEFORE_TRAIN"
        STEP = "STEP"
        EPOCH = "EPOCH"


# ---------------------------------------------------------------------------
# Stage 3 hyperparameters (owned here)
# ---------------------------------------------------------------------------
MODEL_FAMILY = "disaster_small_transformer_v1"

TRAIN_SEED = 42
PHYSICAL_BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 1
EFFECTIVE_BATCH_SIZE = PHYSICAL_BATCH_SIZE * GRAD_ACCUM_STEPS
PLANNED_EPOCHS = 20
ENCODER_LR = 2e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.10
MAX_GRAD_NORM = 1.0

NUM_WORKERS = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True

EARLY_STOP_PATIENCE = 5
CHECKPOINT_PATH = os.environ.get("MLEVOLVE_CKPT_PATH", "./working/best_model.pt")
SUBMISSION_PATH = "./submission/submission.csv"

# Scheduler envelope: only these physical batch sizes are quality-safe.
QUALITY_SAFE_PHYSICAL_BATCH_SIZES: List[int] = [16, 32, 64]
BATCH_LR_SCALING_POLICY = "sqrt"


# ---------------------------------------------------------------------------
# Optimizer / scheduler factories
# ---------------------------------------------------------------------------
def build_param_groups(
    model: nn.Module, encoder_lr: float, head_lr: float, weight_decay: float
) -> List[Dict[str, Any]]:
    """Split params into encoder-side and classification-head groups."""
    head_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("head."):
            head_params.append(p)
        else:
            other_params.append(p)
    return [
        {"params": other_params, "lr": encoder_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
    ]


def build_optimizer(
    model: nn.Module,
    encoder_lr: float = ENCODER_LR,
    head_lr: float = HEAD_LR,
    weight_decay: float = WEIGHT_DECAY,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        build_param_groups(model, encoder_lr, head_lr, weight_decay)
    )


def make_warmup_linear_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then linear decay to 0."""

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        denom = max(1, total_steps - warmup_steps)
        return max(0.0, float(total_steps - step) / float(denom))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def build_loaders(
    tr_ds,
    va_ds,
    te_ds,
    physical_batch: int,
    device: torch.device,
    num_workers: int = NUM_WORKERS,
):
    pin = bool(PIN_MEMORY and device.type == "cuda")
    pw = bool(PERSISTENT_WORKERS and num_workers > 0)
    tr_loader = DataLoader(
        tr_ds,
        batch_size=physical_batch,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=pw,
        drop_last=False,
    )
    eval_bs = max(physical_batch * 2, physical_batch)
    va_loader = DataLoader(
        va_ds,
        batch_size=eval_bs,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=pw,
        drop_last=False,
    )
    te_loader = DataLoader(
        te_ds,
        batch_size=eval_bs,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=pw,
        drop_last=False,
    )
    return tr_loader, va_loader, te_loader


# ---------------------------------------------------------------------------
# Train / eval step helpers
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    runtime,
    scheduler,
    device: torch.device,
    accum_steps: int,
    max_grad_norm: float,
    diagnostics,
    global_step_holder: List[int],
    context,
    epoch: int,
    steps_per_epoch: int,
    state_factory,
) -> float:
    """One full pass. Runs diagnostics.after_update() only on real optimizer
    update attempts, and scheduler safe_point(STEP, ...) at the same cadence."""
    model.train()
    total, n_seen = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    n_micro = len(loader)

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)

        logits, loss = runtime.forward(
            model, input_ids, attention_mask, targets, criterion
        )
        if not check_finite_for_fallback(loss, runtime.cfg):
            # Skip the update but keep moving.
            optimizer.zero_grad(set_to_none=True)
            continue

        scaled_loss = loss / float(accum_steps)
        runtime.backward(scaled_loss)

        is_last = (step + 1) == n_micro
        if (step + 1) % accum_steps == 0 or is_last:
            runtime.unscale_and_clip(model, max_grad_norm)
            runtime.step(optimizer)
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step_holder[0] += 1
            # Diagnostics after an actual optimizer-update attempt.
            try:
                diagnostics.after_update()
            except Exception:
                pass
            if context is not None:
                context.control_hook.safe_point(
                    SafePointType.STEP,
                    epoch=epoch,
                    global_step=global_step_holder[0],
                    steps_per_epoch=steps_per_epoch,
                    state_factory=state_factory,
                )

        total += float(loss.item())
        n_seen += 1

    return total / max(n_seen, 1)


@torch.no_grad()
def predict_probs(
    model: nn.Module,
    loader: DataLoader,
    runtime,
    device: torch.device,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Inference with identical forward path used at training time (AMP-aware),
    probs exported in FP32. Returns (probs, targets) with targets=None for test."""
    model.eval()
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    has_labels = True
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with autocast_context(runtime.cfg):
            logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits.float()).detach().cpu().numpy()
        all_probs.append(probs)
        if "label" in batch:
            all_labels.append(batch["label"].numpy())
        else:
            has_labels = False
    probs_cat = np.concatenate(all_probs, axis=0).astype(np.float64).ravel()
    labels_cat = (
        np.concatenate(all_labels, axis=0).astype(np.int64).ravel()
        if (has_labels and all_labels)
        else None
    )
    return probs_cat, labels_cat


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    runtime,
    device: torch.device,
) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        with autocast_context(runtime.cfg):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits.float(), targets.float())
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# State factory for scheduler checkpoints
# ---------------------------------------------------------------------------
def make_state_factory(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    scheduler,
    diagnostics,
    global_step_holder: List[int],
    step_in_epoch_holder: List[int],
    sampler_state_holder: List[Any],
):
    def state_factory() -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "diagnostics": (
                diagnostics.state_dict() if diagnostics is not None else None
            ),
            "global_step": int(global_step_holder[0]),
            "step_in_epoch": int(step_in_epoch_holder[0]),
            "sampler_state": sampler_state_holder[0] if sampler_state_holder else None,
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

    return state_factory


def restore_from_state(
    state: Optional[Dict[str, Any]],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    scheduler,
    diagnostics,
) -> int:
    """Restore model/optimizer/scheduler/scaler/RNG; returns resumed global_step."""
    if not state:
        return 0
    try:
        model.load_state_dict(state["model"])
    except Exception:
        pass
    try:
        optimizer.load_state_dict(state["optimizer"])
    except Exception:
        pass
    if scheduler is not None and state.get("scheduler"):
        try:
            scheduler.load_state_dict(state["scheduler"])
        except Exception:
            pass
    if scaler is not None and state.get("scaler"):
        try:
            scaler.load_state_dict(state["scaler"])
        except Exception:
            pass
    if diagnostics is not None and state.get("diagnostics"):
        try:
            diagnostics.load_state_dict(state["diagnostics"])
        except Exception:
            pass
    try:
        if "rng_python" in state:
            random.setstate(state["rng_python"])
        if "rng_numpy" in state:
            np.random.set_state(state["rng_numpy"])
        if "rng_torch" in state:
            torch.set_rng_state(state["rng_torch"])
        if "rng_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng_cuda"])
    except Exception:
        pass
    return int(state.get("global_step", 0))


# ---------------------------------------------------------------------------
# Peak-VRAM / timing helpers
# ---------------------------------------------------------------------------
def _reset_peak_vram() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def _peak_vram_mb() -> float:
    if torch.cuda.is_available():
        try:
            return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        except Exception:
            return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def _run_training() -> float:
    # --- scheduler context (must be obtained before any heavyweight alloc) ---
    context = None
    try:
        context = script_scheduler_context()
    except Exception:
        context = None

    device = resolve_device(None)
    print(f"[device] resolved={device}")

    # --- data -------------------------------------------------------------
    train_df, test_df = load_dataframes(INPUT_DIR)
    tr_df, va_df = stratified_split(train_df, val_size=VAL_SIZE, seed=SEED)

    tr_texts = prepare_texts(tr_df)
    va_texts = prepare_texts(va_df)
    te_texts = prepare_texts(test_df)

    # Tokenizer fit ONLY on training split (leakage-safe).
    tokenizer = build_tokenizer_on_train(
        tr_texts, max_vocab=MAX_VOCAB, min_freq=MIN_FREQ
    )
    print(f"[tokenizer] vocab_size={tokenizer.vocab_size}")

    # --- model ------------------------------------------------------------
    model = DisasterTransformer(
        vocab_size=tokenizer.vocab_size,
        max_len=MAX_SEQ_LEN,
        d_model=192,
        nhead=4,
        num_layers=3,
        dim_feedforward=384,
        dropout=0.15,
    ).to(device)

    pos_weight = compute_pos_weight(tr_df["target"].values)
    criterion = get_criterion(pos_weight=pos_weight)
    print(f"[loss] BCEWithLogitsLoss pos_weight={pos_weight:.4f}")

    # --- precision (Stage 2 policy) ---------------------------------------
    pcfg = resolve_precision(preferred_device=str(device))
    model = prepare_model_for_precision(model, pcfg)
    runtime = PrecisionRuntime(pcfg)
    log_precision(pcfg)

    # --- datasets / loaders ------------------------------------------------
    tr_ds = TweetDataset(tr_texts, tr_df["target"].values, tokenizer, MAX_SEQ_LEN)
    va_ds = TweetDataset(va_texts, va_df["target"].values, tokenizer, MAX_SEQ_LEN)
    te_ds = TweetDataset(te_texts, None, tokenizer, MAX_SEQ_LEN)

    physical_batch = PHYSICAL_BATCH_SIZE
    tr_loader, va_loader, te_loader = build_loaders(
        tr_ds, va_ds, te_ds, physical_batch, device, NUM_WORKERS
    )
    steps_per_epoch = max(1, len(tr_loader) // GRAD_ACCUM_STEPS)
    total_steps = PLANNED_EPOCHS * steps_per_epoch
    warmup_steps = int(WARMUP_FRACTION * total_steps)
    print(
        f"[data] train={len(tr_ds)} val={len(va_ds)} test={len(te_ds)} "
        f"physical_batch={physical_batch} accum={GRAD_ACCUM_STEPS} "
        f"effective_batch={physical_batch * GRAD_ACCUM_STEPS} "
        f"steps_per_epoch={steps_per_epoch} total_steps={total_steps}"
    )

    # --- optimizer / scheduler --------------------------------------------
    optimizer = build_optimizer(
        model, encoder_lr=ENCODER_LR, head_lr=HEAD_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = make_warmup_linear_scheduler(optimizer, total_steps, warmup_steps)

    # --- diagnostics ------------------------------------------------------
    diag_settings: Dict[str, Any] = {
        "physical_batch_size": int(physical_batch),
        "effective_batch_size": int(physical_batch * GRAD_ACCUM_STEPS),
        "planned_epochs": int(PLANNED_EPOCHS),
        "steps_per_epoch": int(steps_per_epoch),
        "model_family": MODEL_FAMILY,
        "split_seed": int(SEED),
        "precision": pcfg.effective_policy,
        "precision_reason": pcfg.reason,
    }

    # --- checkpoint holders -----------------------------------------------
    global_step_holder = [0]
    step_in_epoch_holder = [0]
    sampler_state_holder: List[Any] = [None]

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, Any]] = None
    epochs_without_improve = 0

    os.makedirs(os.path.dirname(CHECKPOINT_PATH) or "./working", exist_ok=True)

    # ---------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------
    print(f"[train] starting training for {PLANNED_EPOCHS} epochs")
    _reset_peak_vram()
    t_start = time.time()

    # Diagnostics must be entered AFTER model/optimizer exist.
    with TrainingDiagnostics(
        model.select if hasattr(model, "select") else model,  # fallback harmless
        optimizer,
        scaler=getattr(runtime, "scaler", None),
        settings=diag_settings,
    ) as diagnostics:
        # Attempt restore from scheduler context
        restored_step = 0
        if context is not None:
            try:
                payload = context.load_resume_checkpoint()
                state = (
                    (payload or {}).get("state") if isinstance(payload, dict) else None
                )
                if state:
                    restored_step = restore_from_state(
                        state, model, optimizer, runtime.scaler, scheduler, diagnostics
                    )
                    global_step_holder[0] = restored_step
                    print(f"[resume] resumed from global_step={restored_step}")
            except Exception as e:  # do not suppress; but keep going from scratch
                print(f"[resume] no usable checkpoint: {e}")

        try:
            if context is not None:
                # Ensure max_epochs is set on the job object where supported.
                try:
                    context.job.max_epochs = PLANNED_EPOCHS
                    if (
                        hasattr(context.job, "config")
                        and context.job.config is not None
                    ):
                        context.job.config.max_epochs = PLANNED_EPOCHS
                except Exception:
                    pass
                try:
                    context.store.save_job(context.job)
                except Exception:
                    pass

            state_factory = make_state_factory(
                model,
                optimizer,
                getattr(runtime, "scaler", None),
                scheduler,
                diagnostics,
                global_step_holder,
                step_in_epoch_holder,
                sampler_state_holder,
            )

            if context is not None:
                context.control_hook.safe_point(
                    SafePointType.BEFORE_TRAIN,
                    epoch=0,
                    global_step=global_step_holder[0],
                    steps_per_epoch=steps_per_epoch,
                    state_factory=state_factory,
                )

            start_epoch = 1
            for epoch in range(start_epoch, PLANNED_EPOCHS + 1):
                t_epoch = time.time()
                train_loss = train_one_epoch(
                    model=model,
                    loader=tr_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    runtime=runtime,
                    scheduler=scheduler,
                    device=device,
                    accum_steps=GRAD_ACCUM_STEPS,
                    max_grad_norm=MAX_GRAD_NORM,
                    diagnostics=diagnostics,
                    global_step_holder=global_step_holder,
                    context=context,
                    epoch=epoch,
                    steps_per_epoch=steps_per_epoch,
                    state_factory=state_factory,
                )

                val_loss = evaluate_loss(model, va_loader, criterion, runtime, device)

                # Val F1 with threshold search (used for reporting/monitoring only;
                # checkpoint selection uses val loss to reduce threshold-peeking).
                val_probs, val_labels = predict_probs(model, va_loader, runtime, device)
                best_t, val_f1 = find_best_threshold(
                    val_probs.astype(np.float64),
                    val_labels.astype(np.int64),
                    lo=0.15,
                    hi=0.85,
                    step=0.01,
                )

                cur_lr = optimizer.param_groups[0]["lr"]
                elapsed_epoch = time.time() - t_epoch
                peak_mb = _peak_vram_mb()

                print(
                    f"Epoch {epoch}/{PLANNED_EPOCHS} "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"val_f1={val_f1:.4f} thr={best_t:.2f} "
                    f"lr={cur_lr:.2e} time={elapsed_epoch:.1f}s peak_vram={peak_mb:.0f}MB"
                )
                print(
                    "MLEVOLVE_EPOCH_METRIC "
                    + json.dumps(
                        {
                            "epoch": int(epoch),
                            "metric": float(val_f1),
                            "metric_name": "f1",
                        }
                    )
                )

                try:
                    diagnostics.report(epoch=epoch)
                except Exception:
                    pass

                # Best-checkpoint selection on val loss.
                improved = val_loss < (best_val_loss - 1e-6)
                if improved:
                    best_val_loss = val_loss
                    epochs_without_improve = 0
                    best_state = {
                        "model": {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        },
                        "optimizer": {
                            k: (
                                v.detach().cpu().clone()
                                if isinstance(v, torch.Tensor)
                                else v
                            )
                            for k, v in optimizer.state_dict().items()
                        },
                        "scheduler": (
                            scheduler.state_dict() if scheduler is not None else None
                        ),
                        "scaler": (
                            runtime.scaler.state_dict()
                            if runtime.scaler is not None
                            else None
                        ),
                        "global_step": int(global_step_holder[0]),
                        "epoch": int(epoch),
                        "val_loss": float(val_loss),
                        "val_f1": float(val_f1),
                        "threshold": float(best_t),
                    }
                    try:
                        torch.save(best_state, CHECKPOINT_PATH)
                    except Exception:
                        pass
                else:
                    epochs_without_improve += 1

                # Scheduler EPOCH safe point.
                step_in_epoch_holder[0] = 0
                if context is not None:
                    context.control_hook.safe_point(
                        SafePointType.EPOCH,
                        epoch=epoch,
                        global_step=global_step_holder[0],
                        steps_per_epoch=steps_per_epoch,
                        state_factory=state_factory,
                    )

                if epochs_without_improve >= EARLY_STOP_PATIENCE:
                    print(
                        f"[early_stop] patience={EARLY_STOP_PATIENCE} reached at epoch {epoch}"
                    )
                    break

        finally:
            pass  # ensure diagnostics exits cleanly even on exception paths

    total_time = time.time() - t_start

    # --- restore best checkpoint ------------------------------------------
    if best_state is not None:
        try:
            model.load_state_dict(best_state["model"])
            print(
                f"[restore] best model epoch={best_state.get('epoch')} "
                f"val_loss={best_state.get('val_loss'):.4f}"
            )
        except Exception as e:
            print(f"[restore] warning: could not restore best model: {e}")

    # --- final validation F1 (identical pipeline to test) -----------------
    val_probs, val_labels = predict_probs(model, va_loader, runtime, device)
    best_thr, val_f1 = find_best_threshold(
        val_probs.astype(np.float64),
        val_labels.astype(np.int64),
        lo=0.15,
        hi=0.85,
        step=0.01,
    )
    score = float(val_f1)
    print(
        f"[final] validation F1={score:.4f} at threshold={best_thr:.2f} "
        f"total_time={total_time:.1f}s peak_vram={_peak_vram_mb():.0f}MB "
        f"physical_batch={physical_batch} "
        f"effective_batch={physical_batch * GRAD_ACCUM_STEPS}"
    )

    # --- test inference with the SAME threshold ---------------------------
    te_probs, _ = predict_probs(model, te_loader, runtime, device)
    te_preds = (te_probs >= best_thr).astype(int)

    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission = pd.DataFrame({"id": test_df["id"].values, "target": te_preds})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(
        f"[submission] wrote {SUBMISSION_PATH} rows={len(submission)} "
        f"pos_rate={te_preds.mean():.4f}"
    )

    return score


if __name__ == "__main__":
    final_score = _run_training()
    print(f"Final Validation Score: {final_score}")

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
        