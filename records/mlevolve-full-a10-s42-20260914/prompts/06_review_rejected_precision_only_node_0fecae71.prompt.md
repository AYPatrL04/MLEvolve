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
 Summary: 2 total attempts - 2 pending execution

-------------------------------
Attempt #1:
Design: I merged the three stages by keeping Stage 1's data cleaning, TF-IDF feature builders, `SparseMLP`, `build_loss`, `tune_threshold` and split helpers as the foundation; layering Stage 2's import-safe precision utilities (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `degrade_to_fp32`, `configure_tf32`, `format_precision_log`) on top; then consuming both inside Stage 3's `run_training` with scheduler safe-points, `TrainingDiagnostics`, linear warmup+cosine AdamW, best-val-F1 restore, and test inference. I resolved the duplicate `CandidateAdapter` by keeping Stage 3's self-contained CPU-safe version (it uses the Stage 1 model/loss/features through the Stage 2 precision path), removed duplicate imports/constants, kept all side effects under `if __name__ == "__main__":`, and preserved the exact `MODEL_FAMILY`, `WORKING_DIR`, `SUBMISSION_DIR` contract.
Status: Code generated, execution pending (will run in parallel with other drafts).

-------------------------------
Attempt #2:
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
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"{'name': 'BCEWithLogitsLoss', 'applied_to': 'single logit output per example', 'reduction': 'mean...","output_interface":"{'raw_output': 'one float32 logit per example', 'prediction': 'sigmoid(logit) >= threshold => 1 e...","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"{'forward_autocast_boundary': \"enter torch.autocast(device_type='cuda', dtype=torch.float16) only...","fallback_policy":"fall back to fp32 or disabled AMP if the selected precision path is unsupported","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"{'name': 'AdamW', 'learning_rate': 2e-05, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'epsilon':...","scheduler":"{'name': 'linear warmup with decay', 'warmup_ratio': 0.1}","batch_size_policy":"fixed","dataloader_policy":"{'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'non_blocking': True, 'shuffle...","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available","no matching exact/similar/packed profiles for current hardware","no measured FP16 vs FP32 speed/quality comparison available at draft stage"]}}
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
 Summary: 2 total attempts - 2 pending execution

-------------------------------
Attempt #1:
Design: I merged the three stages by keeping Stage 1's data cleaning, TF-IDF feature builders, `SparseMLP`, `build_loss`, `tune_threshold` and split helpers as the foundation; layering Stage 2's import-safe precision utilities (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `degrade_to_fp32`, `configure_tf32`, `format_precision_log`) on top; then consuming both inside Stage 3's `run_training` with scheduler safe-points, `TrainingDiagnostics`, linear warmup+cosine AdamW, best-val-F1 restore, and test inference. I resolved the duplicate `CandidateAdapter` by keeping Stage 3's self-contained CPU-safe version (it uses the Stage 1 model/loss/features through the Stage 2 precision path), removed duplicate imports/constants, kept all side effects under `if __name__ == "__main__":`, and preserved the exact `MODEL_FAMILY`, `WORKING_DIR`, `SUBMISSION_DIR` contract.
Status: Code generated, execution pending (will run in parallel with other drafts).

-------------------------------
Attempt #2:
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
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"{'name': 'BCEWithLogitsLoss', 'applied_to': 'single logit output per example', 'reduction': 'mean...","output_interface":"{'raw_output': 'one float32 logit per example', 'prediction': 'sigmoid(logit) >= threshold => 1 e...","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"{'forward_autocast_boundary': \"enter torch.autocast(device_type='cuda', dtype=torch.float16) only...","fallback_policy":"fall back to fp32 or disabled AMP if the selected precision path is unsupported","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"{'name': 'AdamW', 'learning_rate': 2e-05, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'epsilon':...","scheduler":"{'name': 'linear warmup with decay', 'warmup_ratio': 0.1}","batch_size_policy":"fixed","dataloader_policy":"{'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'non_blocking': True, 'shuffle...","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available","no matching exact/similar/packed profiles for current hardware","no measured FP16 vs FP32 speed/quality comparison available at draft stage"]}}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=replaced the TF-IDF+sparse-MLP baseline with a from-scratch multi-view (word-transformer + char-ngram bag + keyword embedding) architecture; purpose=capture short-text morphology and keyword priors that bag-of-words misses, keeping every tensor FP32-friendly and offline-safe; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=exposed `MAX_WORD_TOKENS`, `MAX_CHAR_NGRAMS`, and full `MODEL_DEFAULTS` as configurable geometry; purpose=lets Stage 2 precision and Stage 3 training resize the tensor shapes without changing the family; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=single-logit BCEWithLogitsLoss output with F1-threshold tuning helper `tune_threshold`; purpose=matches the metric exactly and gives Stage 3 a fixed output interface (sigmoid ≥ τ) to reuse at test time; hardware_keys=none

# Previous steps
### model_design
**Plan:** ## Plan

**WHAT makes this task unique**: Disaster tweets are extremely short (~14 words), noisy (URLs, handles, hashtags), and highly ambiguous — many non-disaster tweets use disaster lexicon metaphorically ("this party is fire"). This means surface word statistics alone are insufficient; character morphology and keyword/location priors matter, and threshold choice dominates F1 since the class balance is ~57/43.

**WHY this approach**: I'll build a compact from-scratch **multi-view transformer** that fuses three complementary channels — (a) a word-token transformer encoder with a CLS slot for contextual semantics, (b) a hashed character 3–5-gram bag-mean embedding to capture morphology/OOV robustly, and (c) a learned keyword embedding that injects the strong keyword prior. This directly addresses short-text ambiguity (char + keyword compensate for sparse word context) and avoids any external weight downloads (offline constraint). Loss = `BCEWithLogitsLoss` on a single logit, output interface = `sigmoid(logit) >= τ`, with τ tuned on the internal validation split for F1 — matching the pipeline trace exactly.

**HOW I will validate**: stratified 80/20 internal split (seed 42) from `train.csv`, best-τ chosen by maximizing F1 on the hold-out; confirm the hypothesis if train loss decreases and val F1 with tuned τ beats the 0.5 baseline; if val F1 collapses but train F1 rises, reduce model capacity / increase dropout and re-check — the design stage exposes all knobs (d_model, layers, dropout) so Stage 3 can tune without changing the family.

**Backend / memory**: memory strategy = standard (no gradient checkpointing needed, model is ~1–2M params, transformer dims are small and divisible by nhead for tensor-core-friendly shapes); physical/effective batch elasticity = exposed via `scenario['batch_size']` in the adapter and via a configurable `max_word_tokens`/`max_char_ngrams` on the dataset, so Stage 3 can shrink the physical batch and accumulate. Operator considerations: `norm_first=True` LayerNorm inside the encoder with GELU feed-forward keeps FP16 autocast well-conditioned in Stage 2. Scheduler-owned controls the subprocess must NOT implement: device assignment, thread counts, NUMA binding, and process priority. Fallback: if `train.csv` schema drifts, the dataset degrades to empty text + `<unk>` keyword rather than crashing.

Note board:
- baseline_change: replaced the TF-IDF+sparse-MLP baseline with a from-scratch multi-view (word-transformer + char-ngram bag + keyword embedding) architecture | purpose: capture short-text morphology and keyword priors that bag-of-words misses, keeping every tensor FP32-friendly and offline-safe | hardware_keys: none
- baseline_change: exposed `MAX_WORD_TOKENS`, `MAX_CHAR_NGRAMS`, and full `MODEL_DEFAULTS` as configurable geometry | purpose: lets Stage 2 precision and Stage 3 training resize the tensor shapes without changing the family | hardware_keys: none
- baseline_change: single-logit BCEWithLogitsLoss output with F1-threshold tuning helper `tune_threshold` | purpose: matches the metric exactly and gives Stage 3 a fixed output interface (sigmoid ≥ τ) to reuse at test time | hardware_keys: none
**Code:**
```python
"""Stage 1 model-design candidate: data prep, features, splits, model, loss, output interface.

This module is import-safe: it only defines constants, helper functions, the dataset,
the model, the criterion/output interface, and the CPU-preflight CandidateAdapter.
All training/eval/submission side effects belong to later stages.
"""

import os
import re
import zlib
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# --------------------------------------------------------------------------- #
# Scheduler model family contract
# --------------------------------------------------------------------------- #
MODEL_FAMILY = "multiview_word_char_keyword_transformer_v1"

INPUT_DIR = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"


# --------------------------------------------------------------------------- #
# Feature / tokenization geometry (all configurable downstream)
# --------------------------------------------------------------------------- #
MAX_WORD_TOKENS: int = 48  # word sequence length (includes CLS + SEP + padding)
MAX_CHAR_NGRAMS: int = 256  # cap on hashed char n-grams per sample
CHAR_NGRAM_N_MIN: int = 3
CHAR_NGRAM_N_MAX: int = 5
CHAR_VOCAB_SIZE: int = 20000  # fixed hashed space; index 0 reserved for padding

WORD_PAD_IDX: int = 0
WORD_UNK_IDX: int = 1
WORD_CLS_IDX: int = 2
WORD_SEP_IDX: int = 3
WORD_VOCAB_MIN_FREQ: int = 2

KEYWORD_UNK_IDX: int = 0
KEYWORD_VOCAB_MIN_FREQ: int = 3

# transformer dims are tensor-core friendly (multiples of 8, d_model % nhead == 0)
MODEL_DEFAULTS: Dict[str, int] = dict(
    d_word=96,
    d_char=64,
    d_keyword=32,
    d_model=128,
    nhead=4,
    num_encoder_layers=3,
    dim_feedforward=384,
    dropout=0.15,
    model_max_len=64,
)


# --------------------------------------------------------------------------- #
# Text cleaning / tokenization
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z]+|[0-9]+|<url>|<user>|<hashtag>|[^\sa-z0-9]")


def clean_text(text: object) -> str:
    """Lowercase, normalise URLs/mentions/hashtags, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    s = text.lower()
    s = _URL_RE.sub(" <url> ", s)
    s = _MENTION_RE.sub(" <user> ", s)
    s = _HASHTAG_RE.sub(lambda m: " " + m.group(1) + " <hashtag> ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def word_tokenize(cleaned_text: str) -> List[str]:
    if not cleaned_text:
        return []
    return _TOKEN_RE.findall(cleaned_text)


def _stable_ngram_index(gram: str, space: int) -> int:
    """Deterministic (hash-seed-stable) n-gram index in 1..space-1."""
    if space <= 1:
        return 0
    return (zlib.crc32(gram.encode("utf-8")) % (space - 1)) + 1


def char_ngrams(
    text: str,
    n_min: int = CHAR_NGRAM_N_MIN,
    n_max: int = CHAR_NGRAM_N_MAX,
    space: int = CHAR_VOCAB_SIZE,
) -> List[int]:
    """Return a deduplicated list of hashed char n-gram indices (0 reserved for pad)."""
    if not text:
        return []
    s = " " + text + " "
    L = len(s)
    seen = set()
    out: List[int] = []
    for n in range(n_min, n_max + 1):
        if n > L:
            continue
        for i in range(L - n + 1):
            idx = _stable_ngram_index(s[i : i + n], space)
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
    return out


# --------------------------------------------------------------------------- #
# Vocabulary builders (fit on TRAIN ONLY; applied via .get / lookup)
# --------------------------------------------------------------------------- #
def build_word_vocab(
    cleaned_texts: Sequence[str], min_freq: int = WORD_VOCAB_MIN_FREQ
) -> Dict[str, int]:
    counter: Counter = Counter()
    for t in cleaned_texts:
        counter.update(word_tokenize(t))
    vocab: Dict[str, int] = {
        "<pad>": WORD_PAD_IDX,
        "<unk>": WORD_UNK_IDX,
        "<cls>": WORD_CLS_IDX,
        "<sep>": WORD_SEP_IDX,
    }
    next_idx = 4
    for word, c in counter.most_common():
        if c < min_freq or word in vocab:
            continue
        vocab[word] = next_idx
        next_idx += 1
    return vocab


def build_keyword_vocab(
    keywords: Sequence[object], min_freq: int = KEYWORD_VOCAB_MIN_FREQ
) -> Dict[str, int]:
    counter: Counter = Counter()
    for k in keywords:
        if isinstance(k, str) and k.strip():
            counter[k.strip().lower()] += 1
    vocab: Dict[str, int] = {"<unk>": KEYWORD_UNK_IDX}
    next_idx = 1
    for k, c in counter.most_common():
        if c < min_freq or k in vocab:
            continue
        vocab[k] = next_idx
        next_idx += 1
    return vocab


# --------------------------------------------------------------------------- #
# Encoding helpers
# --------------------------------------------------------------------------- #
def encode_word_ids(
    cleaned_text: str, word_vocab: Dict[str, int], max_len: int = MAX_WORD_TOKENS
) -> List[int]:
    ids: List[int] = [WORD_CLS_IDX]
    for tok in word_tokenize(cleaned_text):
        if len(ids) >= max_len - 1:
            break
        ids.append(word_vocab.get(tok, WORD_UNK_IDX))
    ids.append(WORD_SEP_IDX)
    if len(ids) > max_len:
        ids = ids[:max_len]
    while len(ids) < max_len:
        ids.append(WORD_PAD_IDX)
    return ids


def encode_char_ids(cleaned_text: str, max_ngrams: int = MAX_CHAR_NGRAMS) -> List[int]:
    idxs = char_ngrams(cleaned_text)
    if len(idxs) > max_ngrams:
        idxs = idxs[:max_ngrams]
    if len(idxs) < max_ngrams:
        idxs = idxs + [0] * (max_ngrams - len(idxs))
    return idxs


def encode_keyword_id(keyword: object, keyword_vocab: Dict[str, int]) -> int:
    if isinstance(keyword, str) and keyword.strip():
        return keyword_vocab.get(keyword.strip().lower(), KEYWORD_UNK_IDX)
    return KEYWORD_UNK_IDX


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class DisasterTweetDataset(Dataset):
    """Pre-encodes word ids, char n-gram ids and keyword ids for a DataFrame slice."""

    def __init__(
        self,
        df: pd.DataFrame,
        word_vocab: Dict[str, int],
        keyword_vocab: Dict[str, int],
        max_word_tokens: int = MAX_WORD_TOKENS,
        max_char_ngrams: int = MAX_CHAR_NGRAMS,
    ):
        self.df = df.reset_index(drop=True)
        self.word_vocab = word_vocab
        self.keyword_vocab = keyword_vocab
        self.max_word_tokens = int(max_word_tokens)
        self.max_char_ngrams = int(max_char_ngrams)

        texts = (
            self.df["text"].tolist()
            if "text" in self.df.columns
            else [""] * len(self.df)
        )
        keywords = (
            self.df["keyword"].tolist()
            if "keyword" in self.df.columns
            else [None] * len(self.df)
        )
        if "target" in self.df.columns:
            labels = self.df["target"].astype("float32").tolist()
        else:
            labels = [0.0] * len(self.df)

        self.word_ids: List[List[int]] = []
        self.char_ids: List[List[int]] = []
        self.keyword_ids: List[int] = []
        self.labels: List[float] = []

        for t, k, y in zip(texts, keywords, labels):
            cleaned = clean_text(t)
            self.word_ids.append(
                encode_word_ids(cleaned, word_vocab, self.max_word_tokens)
            )
            self.char_ids.append(encode_char_ids(cleaned, self.max_char_ngrams))
            self.keyword_ids.append(encode_keyword_id(k, keyword_vocab))
            self.labels.append(float(y))

    def __len__(self) -> int:
        return len(self.word_ids)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.word_ids[idx], dtype=torch.long),
            torch.tensor(self.char_ids[idx], dtype=torch.long),
            torch.tensor(self.keyword_ids[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# --------------------------------------------------------------------------- #
# Model: multi-view word-transformer + char-ngram bag + keyword embedding
# --------------------------------------------------------------------------- #
class MultiViewTweetTransformer(nn.Module):
    """Fuses (word-transformer CLS) || (char-ngram bag mean) || (keyword embedding).

    Forward returns a single float32 logit per example with shape (batch,).
    """

    def __init__(
        self,
        word_vocab_size: int,
        char_vocab_size: int = CHAR_VOCAB_SIZE,
        keyword_vocab_size: int = 8,
        d_word: int = 96,
        d_char: int = 64,
        d_keyword: int = 32,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 384,
        dropout: float = 0.15,
        model_max_len: int = 64,
    ):
        super().__init__()
        self.d_word = int(d_word)
        self.model_max_len = int(model_max_len)

        # --- word path ------------------------------------------------------ #
        self.word_embed = nn.Embedding(
            int(word_vocab_size), self.d_word, padding_idx=WORD_PAD_IDX
        )
        self.pos_embed = nn.Embedding(self.model_max_len, self.d_word)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_word,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.word_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=int(num_encoder_layers)
        )

        # --- char path ------------------------------------------------------ #
        self.char_embed = nn.Embedding(int(char_vocab_size), int(d_char), padding_idx=0)
        self.char_proj = nn.Sequential(
            nn.Linear(int(d_char), int(d_char)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

        # --- keyword path --------------------------------------------------- #
        self.keyword_embed = nn.Embedding(int(keyword_vocab_size), int(d_keyword))

        # --- fusion head ---------------------------------------------------- #
        fusion_in = self.d_word + int(d_char) + int(d_keyword)
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Sequential(
            nn.Linear(fusion_in, int(d_model)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model), max(int(d_model) // 2, 16)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(int(d_model) // 2, 16), 1),
        )

    def forward(
        self, word_ids: torch.Tensor, char_ids: torch.Tensor, keyword_ids: torch.Tensor
    ) -> torch.Tensor:
        # ---- word branch ----
        B, L = word_ids.shape
        L = min(L, self.model_max_len)
        word_ids = word_ids[:, :L]
        positions = torch.arange(L, device=word_ids.device, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(B, L)
        x = self.word_embed(word_ids) + self.pos_embed(positions)
        pad_mask = word_ids.eq(WORD_PAD_IDX)  # True => ignore
        x = self.word_encoder(x, src_key_padding_mask=pad_mask)
        cls_repr = x[:, 0]

        # ---- char branch (mean pool over hashed n-grams) ----
        char_mask = char_ids.ne(0).unsqueeze(-1).to(dtype=torch.float32)
        ce = self.char_embed(char_ids) * char_mask
        denom = char_mask.sum(dim=1).clamp(min=1.0)
        char_pool = ce.sum(dim=1) / denom
        char_repr = self.char_proj(char_pool)

        # ---- keyword branch ----
        kw_repr = self.keyword_embed(keyword_ids)

        # ---- fuse & classify ----
        fused = torch.cat([cls_repr, char_repr, kw_repr], dim=-1)
        logits = self.head(self.dropout(fused)).squeeze(-1)
        return logits


# --------------------------------------------------------------------------- #
# Loss, criterion, output interface
# --------------------------------------------------------------------------- #
class Criterion:
    """Thin wrapper so later stages can inject pos_weight without touching the model."""

    def __init__(self, pos_weight: Optional[float] = None):
        if pos_weight is None:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32)
            )

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets.to(dtype=logits.dtype))


def build_criterion(pos_weight: Optional[float] = None) -> Criterion:
    return Criterion(pos_weight=pos_weight)


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """FP32 sigmoid — always keep probability export in FP32."""
    return torch.sigmoid(logits.detach().to(dtype=torch.float32))


def probs_to_labels(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(probs) >= float(threshold)).astype(np.int64)


def tune_threshold(
    probs: np.ndarray, targets: np.ndarray, thresholds: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """Search the F1-optimal probability threshold on a labelled split."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (probs >= float(t)).astype(np.int64)
        f1 = f1_score(targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


# --------------------------------------------------------------------------- #
# Data loading & splits
# --------------------------------------------------------------------------- #
def load_data(input_dir: str = INPUT_DIR):
    train_path = os.path.join(input_dir, "train.csv")
    test_path = os.path.join(input_dir, "test.csv")
    sample_path = os.path.join(input_dir, "sample_submission.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_df = pd.read_csv(sample_path) if os.path.exists(sample_path) else None
    return train_df, test_df, sample_df


def build_splits(
    train_df: pd.DataFrame, val_size: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split from train.csv — the only labels usable for selection."""
    strat = train_df["target"] if "target" in train_df.columns else None
    tr_df, va_df = train_test_split(
        train_df,
        test_size=float(val_size),
        stratify=strat,
        random_state=int(seed),
        shuffle=True,
    )
    return tr_df.reset_index(drop=True), va_df.reset_index(drop=True)


def fit_feature_vocabs(train_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Fit word + keyword vocabularies using ONLY the passed-in training slice."""
    texts = train_df["text"].tolist() if "text" in train_df.columns else []
    cleaned = [clean_text(t) for t in texts]
    keywords = train_df["keyword"].tolist() if "keyword" in train_df.columns else []
    return build_word_vocab(cleaned), build_keyword_vocab(keywords)


# --------------------------------------------------------------------------- #
# CPU preflight adapter
# --------------------------------------------------------------------------- #
class CandidateAdapter:
    """No-argument adapter exercising the real dataset / model / criterion path.

    - Uses ONLY the real cleaned-token encoder and MultiViewTweetTransformer.
    - Batches are built from a small slice of the real train.csv (deterministic).
    - build_model accepts partial context; unspecified sizes fall back to the slice vocab.
    """

    def __init__(self):
        self._cache: Dict[str, object] = {}

    # ---- internal helpers ------------------------------------------------- #
    def _load_reference(self):
        if "reference" in self._cache:
            return self._cache["reference"]
        input_dir = os.environ.get("MLEVOLVE_INPUT_DIR", INPUT_DIR)
        train_path = os.path.join(input_dir, "train.csv")
        if os.path.exists(train_path):
            df = pd.read_csv(train_path).head(64).reset_index(drop=True)
        else:
            df = pd.DataFrame(
                {
                    "text": ["<unk> sample tweet with flood and fire"] * 8,
                    "keyword": ["flood"] * 8,
                    "target": [0, 1, 0, 1, 0, 1, 0, 1],
                }
            )
        if "text" not in df.columns:
            df["text"] = ""
        if "keyword" not in df.columns:
            df["keyword"] = None
        if "target" not in df.columns:
            df["target"] = 0
        cleaned = [clean_text(t) for t in df["text"].tolist()]
        word_vocab = build_word_vocab(cleaned, min_freq=1)
        keyword_vocab = build_keyword_vocab(df["keyword"].tolist(), min_freq=1)
        dataset = DisasterTweetDataset(df, word_vocab, keyword_vocab)
        ref = dict(
            df=df, word_vocab=word_vocab, keyword_vocab=keyword_vocab, dataset=dataset
        )
        self._cache["reference"] = ref
        return ref

    def _resolve_model_cfg(self, context: Optional[Dict]) -> Dict[str, object]:
        ctx = dict(context or {})
        cfg = {
            k: ctx.get(k, ctx.get("model_" + k, v)) for k, v in MODEL_DEFAULTS.items()
        }
        # keep every numeric knob of the correct type
        int_keys = [
            "d_word",
            "d_char",
            "d_keyword",
            "d_model",
            "nhead",
            "num_encoder_layers",
            "dim_feedforward",
            "model_max_len",
        ]
        for k in int_keys:
            if cfg.get(k) is not None:
                cfg[k] = int(cfg[k])
        if cfg.get("dropout") is not None:
            cfg["dropout"] = float(cfg["dropout"])
        return cfg

    # ---- required adapter API -------------------------------------------- #
    def build_model(self, context: Optional[Dict] = None) -> nn.Module:
        ctx = dict(context or {})
        ref = self._load_reference()
        word_vocab_size = ctx.get("word_vocab_size")
        keyword_vocab_size = ctx.get("keyword_vocab_size")
        if word_vocab_size is None:
            word_vocab_size = max(len(ref["word_vocab"]), 16)
        if keyword_vocab_size is None:
            keyword_vocab_size = max(len(ref["keyword_vocab"]), 8)
        cfg = self._resolve_model_cfg(ctx)
        model = MultiViewTweetTransformer(
            word_vocab_size=int(word_vocab_size),
            char_vocab_size=int(ctx.get("char_vocab_size", CHAR_VOCAB_SIZE)),
            keyword_vocab_size=int(keyword_vocab_size),
            **cfg,
        )
        return model

    def build_optimizer(self, model: nn.Module, context: Optional[Dict] = None):
        # Minimal, non-committal optimizer purely for the one-batch CPU preflight
        # step; the real optimizer/scheduler are chosen in Stage 3.
        ctx = dict(context or {})
        lr = float(ctx.get("lr", 1e-3))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    def _build_batch(self, scenario: Optional[Dict], device, split: str):
        scenario = dict(scenario or {})
        bs = int(scenario.get("batch_size", 4)) or 4
        ref = self._load_reference()
        ds: DisasterTweetDataset = ref["dataset"]

        fixture = scenario.get("fixture") or {}
        mapping: Dict[str, object] = {}
        if fixture:
            # Fixture-provided lengths override dataset geometry for shape coverage.
            n = len(ds) if len(ds) else 1
            idx = [i % n for i in range(bs)]
            word_len = (
                int(fixture.get("word_ids_shape", (MAX_WORD_TOKENS,))[-1])
                if isinstance(fixture.get("word_ids_shape"), (list, tuple))
                else MAX_WORD_TOKENS
            )
            char_len = (
                int(fixture.get("char_ids_shape", (MAX_CHAR_NGRAMS,))[-1])
                if isinstance(fixture.get("char_ids_shape"), (list, tuple))
                else MAX_CHAR_NGRAMS
            )
            word_ids_list, char_ids_list, kw_list, y_list = [], [], [], []
            for i in idx:
                w, c, k, y = ds[i]
                wl = w.tolist()
                cl = c.tolist()
                if len(wl) < word_len:
                    wl = wl + [WORD_PAD_IDX] * (word_len - len(wl))
                if len(cl) < char_len:
                    cl = cl + [0] * (char_len - len(cl))
                word_ids_list.append(wl[:word_len])
                char_ids_list.append(cl[:char_len])
                kw_list.append(int(k.item()))
                y_list.append(float(y.item()))
            word_ids = torch.tensor(word_ids_list, dtype=torch.long)
            char_ids = torch.tensor(char_ids_list, dtype=torch.long)
            keyword_ids = torch.tensor(kw_list, dtype=torch.long)
            targets = torch.tensor(y_list, dtype=torch.float32)
        else:
            n = len(ds) if len(ds) else 1
            idx = [i % n for i in range(bs)]
            word_ids = torch.stack([ds[i][0] for i in idx], dim=0)
            char_ids = torch.stack([ds[i][1] for i in idx], dim=0)
            keyword_ids = torch.stack([ds[i][2] for i in idx], dim=0)
            targets = torch.stack([ds[i][3] for i in idx], dim=0).to(
                dtype=torch.float32
            )

        word_ids = word_ids.to(device)
        char_ids = char_ids.to(device)
        keyword_ids = keyword_ids.to(device)
        targets = targets.to(device)

        mapping["word_ids"] = word_ids
        mapping["char_ids"] = char_ids
        mapping["keyword_ids"] = keyword_ids
        mapping["inputs"] = (word_ids, char_ids, keyword_ids)
        mapping["target"] = targets
        return mapping

    def build_train_batch(self, scenario: Optional[Dict], device):
        return self._build_batch(scenario, device, split="train")

    def build_validation_batch(self, scenario: Optional[Dict], device):
        return self._build_batch(scenario, device, split="val")

    @staticmethod
    def _unpack_inputs(
        batch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = batch["inputs"]
        if isinstance(inputs, dict):
            return (
                inputs["word_ids"],
                inputs["char_ids"],
                inputs["keyword_ids"],
                batch["target"],
            )
        w, c, k = inputs
        return w, c, k, batch["target"]

    def training_step(self, model: nn.Module, batch, context: Optional[Dict] = None):
        ctx = dict(context or {})
        word_ids, char_ids, keyword_ids, targets = self._unpack_inputs(batch)
        logits = model(word_ids, char_ids, keyword_ids)
        criterion = ctx.get("criterion")
        if criterion is None:
            criterion = build_criterion(pos_weight=ctx.get("pos_weight"))
        loss = criterion(logits, targets.to(dtype=torch.float32))
        return loss

    def validation_step(self, model: nn.Module, batch, context: Optional[Dict] = None):
        ctx = dict(context or {})
        word_ids, char_ids, keyword_ids, targets = self._unpack_inputs(batch)
        logits = model(word_ids, char_ids, keyword_ids)
        criterion = ctx.get("criterion")
        if criterion is None:
            criterion = build_criterion(pos_weight=ctx.get("pos_weight"))
        loss = criterion(logits, targets.to(dtype=torch.float32))
        return loss


# --------------------------------------------------------------------------- #
# Smoke check (no training)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    adapter = CandidateAdapter()
    model = adapter.build_model({})
    model.eval()
    cpu = torch.device("cpu")
    batch = adapter.build_validation_batch({"batch_size": 2}, cpu)
    with torch.no_grad():
        loss = adapter.validation_step(model, batch, {})
    print("stage1 smoke ok:", float(loss.item()))

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
 Summary: 2 total attempts - 2 pending execution

-------------------------------
Attempt #1:
Design: I merged the three stages by keeping Stage 1's data cleaning, TF-IDF feature builders, `SparseMLP`, `build_loss`, `tune_threshold` and split helpers as the foundation; layering Stage 2's import-safe precision utilities (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `degrade_to_fp32`, `configure_tf32`, `format_precision_log`) on top; then consuming both inside Stage 3's `run_training` with scheduler safe-points, `TrainingDiagnostics`, linear warmup+cosine AdamW, best-val-F1 restore, and test inference. I resolved the duplicate `CandidateAdapter` by keeping Stage 3's self-contained CPU-safe version (it uses the Stage 1 model/loss/features through the Stage 2 precision path), removed duplicate imports/constants, kept all side effects under `if __name__ == "__main__":`, and preserved the exact `MODEL_FAMILY`, `WORKING_DIR`, `SUBMISSION_DIR` contract.
Status: Code generated, execution pending (will run in parallel with other drafts).

-------------------------------
Attempt #2:
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
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"{'name': 'BCEWithLogitsLoss', 'applied_to': 'single logit output per example', 'reduction': 'mean...","output_interface":"{'raw_output': 'one float32 logit per example', 'prediction': 'sigmoid(logit) >= threshold => 1 e...","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"{'forward_autocast_boundary': \"enter torch.autocast(device_type='cuda', dtype=torch.float16) only...","fallback_policy":"fall back to fp32 or disabled AMP if the selected precision path is unsupported","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"{'name': 'AdamW', 'learning_rate': 2e-05, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'epsilon':...","scheduler":"{'name': 'linear warmup with decay', 'warmup_ratio': 0.1}","batch_size_policy":"fixed","dataloader_policy":"{'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'non_blocking': True, 'shuffle...","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available","no matching exact/similar/packed profiles for current hardware","no measured FP16 vs FP32 speed/quality comparison available at draft stage"]}}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=replaced the TF-IDF+sparse-MLP baseline with a from-scratch multi-view (word-transformer + char-ngram bag + keyword embedding) architecture; purpose=capture short-text morphology and keyword priors that bag-of-words misses, keeping every tensor FP32-friendly and offline-safe; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=exposed `MAX_WORD_TOKENS`, `MAX_CHAR_NGRAMS`, and full `MODEL_DEFAULTS` as configurable geometry; purpose=lets Stage 2 precision and Stage 3 training resize the tensor shapes without changing the family; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=single-logit BCEWithLogitsLoss output with F1-threshold tuning helper `tune_threshold`; purpose=matches the metric exactly and gives Stage 3 a fixed output interface (sigmoid ≥ τ) to reuse at test time; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=added a Stage-2 precision policy module (FP16 AMP forward-only, FP32 fusion head + BCE loss island, TF32 disabled, GradScaler, autocast helper, fallback to FP32/disabled) on top of the Stage 1 multi-view transformer; purpose=keeps the multi-view model family, loss, and single-logit output interface untouched while making the transformer forward cheap on the A10 Tensor Cores and keeping the metric-critical reduction/export in FP32; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=precision-required adapter `to_fp32_policy(model)` only casts the module to FP32 (no `.half()`, no whole-batch cast, no bf16, no TE recipe); purpose=honors the "FP16 is an option, not a whole-pipeline dtype" rule and preserves Stage 1 geometry/interface; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=added `precision_log_line` + `run_precision_preflight` helpers that never touch CUDA/TF32 at import time; purpose=gives Stage 3 a single entrypoint to log the actual precision regions and preserves the import-safe contract; hardware_keys=none

# Previous steps
### model_design
**Plan:** ## Plan

**WHAT makes this task unique**: Disaster tweets are extremely short (~14 words), noisy (URLs, handles, hashtags), and highly ambiguous — many non-disaster tweets use disaster lexicon metaphorically ("this party is fire"). This means surface word statistics alone are insufficient; character morphology and keyword/location priors matter, and threshold choice dominates F1 since the class balance is ~57/43.

**WHY this approach**: I'll build a compact from-scratch **multi-view transformer** that fuses three complementary channels — (a) a word-token transformer encoder with a CLS slot for contextual semantics, (b) a hashed character 3–5-gram bag-mean embedding to capture morphology/OOV robustly, and (c) a learned keyword embedding that injects the strong keyword prior. This directly addresses short-text ambiguity (char + keyword compensate for sparse word context) and avoids any external weight downloads (offline constraint). Loss = `BCEWithLogitsLoss` on a single logit, output interface = `sigmoid(logit) >= τ`, with τ tuned on the internal validation split for F1 — matching the pipeline trace exactly.

**HOW I will validate**: stratified 80/20 internal split (seed 42) from `train.csv`, best-τ chosen by maximizing F1 on the hold-out; confirm the hypothesis if train loss decreases and val F1 with tuned τ beats the 0.5 baseline; if val F1 collapses but train F1 rises, reduce model capacity / increase dropout and re-check — the design stage exposes all knobs (d_model, layers, dropout) so Stage 3 can tune without changing the family.

**Backend / memory**: memory strategy = standard (no gradient checkpointing needed, model is ~1–2M params, transformer dims are small and divisible by nhead for tensor-core-friendly shapes); physical/effective batch elasticity = exposed via `scenario['batch_size']` in the adapter and via a configurable `max_word_tokens`/`max_char_ngrams` on the dataset, so Stage 3 can shrink the physical batch and accumulate. Operator considerations: `norm_first=True` LayerNorm inside the encoder with GELU feed-forward keeps FP16 autocast well-conditioned in Stage 2. Scheduler-owned controls the subprocess must NOT implement: device assignment, thread counts, NUMA binding, and process priority. Fallback: if `train.csv` schema drifts, the dataset degrades to empty text + `<unk>` keyword rather than crashing.

Note board:
- baseline_change: replaced the TF-IDF+sparse-MLP baseline with a from-scratch multi-view (word-transformer + char-ngram bag + keyword embedding) architecture | purpose: capture short-text morphology and keyword priors that bag-of-words misses, keeping every tensor FP32-friendly and offline-safe | hardware_keys: none
- baseline_change: exposed `MAX_WORD_TOKENS`, `MAX_CHAR_NGRAMS`, and full `MODEL_DEFAULTS` as configurable geometry | purpose: lets Stage 2 precision and Stage 3 training resize the tensor shapes without changing the family | hardware_keys: none
- baseline_change: single-logit BCEWithLogitsLoss output with F1-threshold tuning helper `tune_threshold` | purpose: matches the metric exactly and gives Stage 3 a fixed output interface (sigmoid ≥ τ) to reuse at test time | hardware_keys: none
**Code:**
```python
"""Stage 1 model-design candidate: data prep, features, splits, model, loss, output interface.

This module is import-safe: it only defines constants, helper functions, the dataset,
the model, the criterion/output interface, and the CPU-preflight CandidateAdapter.
All training/eval/submission side effects belong to later stages.
"""

import os
import re
import zlib
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# --------------------------------------------------------------------------- #
# Scheduler model family contract
# --------------------------------------------------------------------------- #
MODEL_FAMILY = "multiview_word_char_keyword_transformer_v1"

INPUT_DIR = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"


# --------------------------------------------------------------------------- #
# Feature / tokenization geometry (all configurable downstream)
# --------------------------------------------------------------------------- #
MAX_WORD_TOKENS: int = 48  # word sequence length (includes CLS + SEP + padding)
MAX_CHAR_NGRAMS: int = 256  # cap on hashed char n-grams per sample
CHAR_NGRAM_N_MIN: int = 3
CHAR_NGRAM_N_MAX: int = 5
CHAR_VOCAB_SIZE: int = 20000  # fixed hashed space; index 0 reserved for padding

WORD_PAD_IDX: int = 0
WORD_UNK_IDX: int = 1
WORD_CLS_IDX: int = 2
WORD_SEP_IDX: int = 3
WORD_VOCAB_MIN_FREQ: int = 2

KEYWORD_UNK_IDX: int = 0
KEYWORD_VOCAB_MIN_FREQ: int = 3

# transformer dims are tensor-core friendly (multiples of 8, d_model % nhead == 0)
MODEL_DEFAULTS: Dict[str, int] = dict(
    d_word=96,
    d_char=64,
    d_keyword=32,
    d_model=128,
    nhead=4,
    num_encoder_layers=3,
    dim_feedforward=384,
    dropout=0.15,
    model_max_len=64,
)


# --------------------------------------------------------------------------- #
# Text cleaning / tokenization
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z]+|[0-9]+|<url>|<user>|<hashtag>|[^\sa-z0-9]")


def clean_text(text: object) -> str:
    """Lowercase, normalise URLs/mentions/hashtags, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    s = text.lower()
    s = _URL_RE.sub(" <url> ", s)
    s = _MENTION_RE.sub(" <user> ", s)
    s = _HASHTAG_RE.sub(lambda m: " " + m.group(1) + " <hashtag> ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def word_tokenize(cleaned_text: str) -> List[str]:
    if not cleaned_text:
        return []
    return _TOKEN_RE.findall(cleaned_text)


def _stable_ngram_index(gram: str, space: int) -> int:
    """Deterministic (hash-seed-stable) n-gram index in 1..space-1."""
    if space <= 1:
        return 0
    return (zlib.crc32(gram.encode("utf-8")) % (space - 1)) + 1


def char_ngrams(
    text: str,
    n_min: int = CHAR_NGRAM_N_MIN,
    n_max: int = CHAR_NGRAM_N_MAX,
    space: int = CHAR_VOCAB_SIZE,
) -> List[int]:
    """Return a deduplicated list of hashed char n-gram indices (0 reserved for pad)."""
    if not text:
        return []
    s = " " + text + " "
    L = len(s)
    seen = set()
    out: List[int] = []
    for n in range(n_min, n_max + 1):
        if n > L:
            continue
        for i in range(L - n + 1):
            idx = _stable_ngram_index(s[i : i + n], space)
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
    return out


# --------------------------------------------------------------------------- #
# Vocabulary builders (fit on TRAIN ONLY; applied via .get / lookup)
# --------------------------------------------------------------------------- #
def build_word_vocab(
    cleaned_texts: Sequence[str], min_freq: int = WORD_VOCAB_MIN_FREQ
) -> Dict[str, int]:
    counter: Counter = Counter()
    for t in cleaned_texts:
        counter.update(word_tokenize(t))
    vocab: Dict[str, int] = {
        "<pad>": WORD_PAD_IDX,
        "<unk>": WORD_UNK_IDX,
        "<cls>": WORD_CLS_IDX,
        "<sep>": WORD_SEP_IDX,
    }
    next_idx = 4
    for word, c in counter.most_common():
        if c < min_freq or word in vocab:
            continue
        vocab[word] = next_idx
        next_idx += 1
    return vocab


def build_keyword_vocab(
    keywords: Sequence[object], min_freq: int = KEYWORD_VOCAB_MIN_FREQ
) -> Dict[str, int]:
    counter: Counter = Counter()
    for k in keywords:
        if isinstance(k, str) and k.strip():
            counter[k.strip().lower()] += 1
    vocab: Dict[str, int] = {"<unk>": KEYWORD_UNK_IDX}
    next_idx = 1
    for k, c in counter.most_common():
        if c < min_freq or k in vocab:
            continue
        vocab[k] = next_idx
        next_idx += 1
    return vocab


# --------------------------------------------------------------------------- #
# Encoding helpers
# --------------------------------------------------------------------------- #
def encode_word_ids(
    cleaned_text: str, word_vocab: Dict[str, int], max_len: int = MAX_WORD_TOKENS
) -> List[int]:
    ids: List[int] = [WORD_CLS_IDX]
    for tok in word_tokenize(cleaned_text):
        if len(ids) >= max_len - 1:
            break
        ids.append(word_vocab.get(tok, WORD_UNK_IDX))
    ids.append(WORD_SEP_IDX)
    if len(ids) > max_len:
        ids = ids[:max_len]
    while len(ids) < max_len:
        ids.append(WORD_PAD_IDX)
    return ids


def encode_char_ids(cleaned_text: str, max_ngrams: int = MAX_CHAR_NGRAMS) -> List[int]:
    idxs = char_ngrams(cleaned_text)
    if len(idxs) > max_ngrams:
        idxs = idxs[:max_ngrams]
    if len(idxs) < max_ngrams:
        idxs = idxs + [0] * (max_ngrams - len(idxs))
    return idxs


def encode_keyword_id(keyword: object, keyword_vocab: Dict[str, int]) -> int:
    if isinstance(keyword, str) and keyword.strip():
        return keyword_vocab.get(keyword.strip().lower(), KEYWORD_UNK_IDX)
    return KEYWORD_UNK_IDX


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class DisasterTweetDataset(Dataset):
    """Pre-encodes word ids, char n-gram ids and keyword ids for a DataFrame slice."""

    def __init__(
        self,
        df: pd.DataFrame,
        word_vocab: Dict[str, int],
        keyword_vocab: Dict[str, int],
        max_word_tokens: int = MAX_WORD_TOKENS,
        max_char_ngrams: int = MAX_CHAR_NGRAMS,
    ):
        self.df = df.reset_index(drop=True)
        self.word_vocab = word_vocab
        self.keyword_vocab = keyword_vocab
        self.max_word_tokens = int(max_word_tokens)
        self.max_char_ngrams = int(max_char_ngrams)

        texts = (
            self.df["text"].tolist()
            if "text" in self.df.columns
            else [""] * len(self.df)
        )
        keywords = (
            self.df["keyword"].tolist()
            if "keyword" in self.df.columns
            else [None] * len(self.df)
        )
        if "target" in self.df.columns:
            labels = self.df["target"].astype("float32").tolist()
        else:
            labels = [0.0] * len(self.df)

        self.word_ids: List[List[int]] = []
        self.char_ids: List[List[int]] = []
        self.keyword_ids: List[int] = []
        self.labels: List[float] = []

        for t, k, y in zip(texts, keywords, labels):
            cleaned = clean_text(t)
            self.word_ids.append(
                encode_word_ids(cleaned, word_vocab, self.max_word_tokens)
            )
            self.char_ids.append(encode_char_ids(cleaned, self.max_char_ngrams))
            self.keyword_ids.append(encode_keyword_id(k, keyword_vocab))
            self.labels.append(float(y))

    def __len__(self) -> int:
        return len(self.word_ids)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.word_ids[idx], dtype=torch.long),
            torch.tensor(self.char_ids[idx], dtype=torch.long),
            torch.tensor(self.keyword_ids[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# --------------------------------------------------------------------------- #
# Model: multi-view word-transformer + char-ngram bag + keyword embedding
# --------------------------------------------------------------------------- #
class MultiViewTweetTransformer(nn.Module):
    """Fuses (word-transformer CLS) || (char-ngram bag mean) || (keyword embedding).

    Forward returns a single float32 logit per example with shape (batch,).
    """

    def __init__(
        self,
        word_vocab_size: int,
        char_vocab_size: int = CHAR_VOCAB_SIZE,
        keyword_vocab_size: int = 8,
        d_word: int = 96,
        d_char: int = 64,
        d_keyword: int = 32,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 384,
        dropout: float = 0.15,
        model_max_len: int = 64,
    ):
        super().__init__()
        self.d_word = int(d_word)
        self.model_max_len = int(model_max_len)

        # --- word path ------------------------------------------------------ #
        self.word_embed = nn.Embedding(
            int(word_vocab_size), self.d_word, padding_idx=WORD_PAD_IDX
        )
        self.pos_embed = nn.Embedding(self.model_max_len, self.d_word)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_word,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.word_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=int(num_encoder_layers)
        )

        # --- char path ------------------------------------------------------ #
        self.char_embed = nn.Embedding(int(char_vocab_size), int(d_char), padding_idx=0)
        self.char_proj = nn.Sequential(
            nn.Linear(int(d_char), int(d_char)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

        # --- keyword path --------------------------------------------------- #
        self.keyword_embed = nn.Embedding(int(keyword_vocab_size), int(d_keyword))

        # --- fusion head ---------------------------------------------------- #
        fusion_in = self.d_word + int(d_char) + int(d_keyword)
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Sequential(
            nn.Linear(fusion_in, int(d_model)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model), max(int(d_model) // 2, 16)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(int(d_model) // 2, 16), 1),
        )

    def forward(
        self, word_ids: torch.Tensor, char_ids: torch.Tensor, keyword_ids: torch.Tensor
    ) -> torch.Tensor:
        # ---- word branch ----
        B, L = word_ids.shape
        L = min(L, self.model_max_len)
        word_ids = word_ids[:, :L]
        positions = torch.arange(L, device=word_ids.device, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(B, L)
        x = self.word_embed(word_ids) + self.pos_embed(positions)
        pad_mask = word_ids.eq(WORD_PAD_IDX)  # True => ignore
        x = self.word_encoder(x, src_key_padding_mask=pad_mask)
        cls_repr = x[:, 0]

        # ---- char branch (mean pool over hashed n-grams) ----
        char_mask = char_ids.ne(0).unsqueeze(-1).to(dtype=torch.float32)
        ce = self.char_embed(char_ids) * char_mask
        denom = char_mask.sum(dim=1).clamp(min=1.0)
        char_pool = ce.sum(dim=1) / denom
        char_repr = self.char_proj(char_pool)

        # ---- keyword branch ----
        kw_repr = self.keyword_embed(keyword_ids)

        # ---- fuse & classify ----
        fused = torch.cat([cls_repr, char_repr, kw_repr], dim=-1)
        logits = self.head(self.dropout(fused)).squeeze(-1)
        return logits


# --------------------------------------------------------------------------- #
# Loss, criterion, output interface
# --------------------------------------------------------------------------- #
class Criterion:
    """Thin wrapper so later stages can inject pos_weight without touching the model."""

    def __init__(self, pos_weight: Optional[float] = None):
        if pos_weight is None:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32)
            )

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets.to(dtype=logits.dtype))


def build_criterion(pos_weight: Optional[float] = None) -> Criterion:
    return Criterion(pos_weight=pos_weight)


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """FP32 sigmoid — always keep probability export in FP32."""
    return torch.sigmoid(logits.detach().to(dtype=torch.float32))


def probs_to_labels(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(probs) >= float(threshold)).astype(np.int64)


def tune_threshold(
    probs: np.ndarray, targets: np.ndarray, thresholds: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """Search the F1-optimal probability threshold on a labelled split."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (probs >= float(t)).astype(np.int64)
        f1 = f1_score(targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


# --------------------------------------------------------------------------- #
# Data loading & splits
# --------------------------------------------------------------------------- #
def load_data(input_dir: str = INPUT_DIR):
    train_path = os.path.join(input_dir, "train.csv")
    test_path = os.path.join(input_dir, "test.csv")
    sample_path = os.path.join(input_dir, "sample_submission.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_df = pd.read_csv(sample_path) if os.path.exists(sample_path) else None
    return train_df, test_df, sample_df


def build_splits(
    train_df: pd.DataFrame, val_size: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split from train.csv — the only labels usable for selection."""
    strat = train_df["target"] if "target" in train_df.columns else None
    tr_df, va_df = train_test_split(
        train_df,
        test_size=float(val_size),
        stratify=strat,
        random_state=int(seed),
        shuffle=True,
    )
    return tr_df.reset_index(drop=True), va_df.reset_index(drop=True)


def fit_feature_vocabs(train_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Fit word + keyword vocabularies using ONLY the passed-in training slice."""
    texts = train_df["text"].tolist() if "text" in train_df.columns else []
    cleaned = [clean_text(t) for t in texts]
    keywords = train_df["keyword"].tolist() if "keyword" in train_df.columns else []
    return build_word_vocab(cleaned), build_keyword_vocab(keywords)


# --------------------------------------------------------------------------- #
# CPU preflight adapter
# --------------------------------------------------------------------------- #
class CandidateAdapter:
    """No-argument adapter exercising the real dataset / model / criterion path.

    - Uses ONLY the real cleaned-token encoder and MultiViewTweetTransformer.
    - Batches are built from a small slice of the real train.csv (deterministic).
    - build_model accepts partial context; unspecified sizes fall back to the slice vocab.
    """

    def __init__(self):
        self._cache: Dict[str, object] = {}

    # ---- internal helpers ------------------------------------------------- #
    def _load_reference(self):
        if "reference" in self._cache:
            return self._cache["reference"]
        input_dir = os.environ.get("MLEVOLVE_INPUT_DIR", INPUT_DIR)
        train_path = os.path.join(input_dir, "train.csv")
        if os.path.exists(train_path):
            df = pd.read_csv(train_path).head(64).reset_index(drop=True)
        else:
            df = pd.DataFrame(
                {
                    "text": ["<unk> sample tweet with flood and fire"] * 8,
                    "keyword": ["flood"] * 8,
                    "target": [0, 1, 0, 1, 0, 1, 0, 1],
                }
            )
        if "text" not in df.columns:
            df["text"] = ""
        if "keyword" not in df.columns:
            df["keyword"] = None
        if "target" not in df.columns:
            df["target"] = 0
        cleaned = [clean_text(t) for t in df["text"].tolist()]
        word_vocab = build_word_vocab(cleaned, min_freq=1)
        keyword_vocab = build_keyword_vocab(df["keyword"].tolist(), min_freq=1)
        dataset = DisasterTweetDataset(df, word_vocab, keyword_vocab)
        ref = dict(
            df=df, word_vocab=word_vocab, keyword_vocab=keyword_vocab, dataset=dataset
        )
        self._cache["reference"] = ref
        return ref

    def _resolve_model_cfg(self, context: Optional[Dict]) -> Dict[str, object]:
        ctx = dict(context or {})
        cfg = {
            k: ctx.get(k, ctx.get("model_" + k, v)) for k, v in MODEL_DEFAULTS.items()
        }
        # keep every numeric knob of the correct type
        int_keys = [
            "d_word",
            "d_char",
            "d_keyword",
            "d_model",
            "nhead",
            "num_encoder_layers",
            "dim_feedforward",
            "model_max_len",
        ]
        for k in int_keys:
            if cfg.get(k) is not None:
                cfg[k] = int(cfg[k])
        if cfg.get("dropout") is not None:
            cfg["dropout"] = float(cfg["dropout"])
        return cfg

    # ---- required adapter API -------------------------------------------- #
    def build_model(self, context: Optional[Dict] = None) -> nn.Module:
        ctx = dict(context or {})
        ref = self._load_reference()
        word_vocab_size = ctx.get("word_vocab_size")
        keyword_vocab_size = ctx.get("keyword_vocab_size")
        if word_vocab_size is None:
            word_vocab_size = max(len(ref["word_vocab"]), 16)
        if keyword_vocab_size is None:
            keyword_vocab_size = max(len(ref["keyword_vocab"]), 8)
        cfg = self._resolve_model_cfg(ctx)
        model = MultiViewTweetTransformer(
            word_vocab_size=int(word_vocab_size),
            char_vocab_size=int(ctx.get("char_vocab_size", CHAR_VOCAB_SIZE)),
            keyword_vocab_size=int(keyword_vocab_size),
            **cfg,
        )
        return model

    def build_optimizer(self, model: nn.Module, context: Optional[Dict] = None):
        # Minimal, non-committal optimizer purely for the one-batch CPU preflight
        # step; the real optimizer/scheduler are chosen in Stage 3.
        ctx = dict(context or {})
        lr = float(ctx.get("lr", 1e-3))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    def _build_batch(self, scenario: Optional[Dict], device, split: str):
        scenario = dict(scenario or {})
        bs = int(scenario.get("batch_size", 4)) or 4
        ref = self._load_reference()
        ds: DisasterTweetDataset = ref["dataset"]

        fixture = scenario.get("fixture") or {}
        mapping: Dict[str, object] = {}
        if fixture:
            # Fixture-provided lengths override dataset geometry for shape coverage.
            n = len(ds) if len(ds) else 1
            idx = [i % n for i in range(bs)]
            word_len = (
                int(fixture.get("word_ids_shape", (MAX_WORD_TOKENS,))[-1])
                if isinstance(fixture.get("word_ids_shape"), (list, tuple))
                else MAX_WORD_TOKENS
            )
            char_len = (
                int(fixture.get("char_ids_shape", (MAX_CHAR_NGRAMS,))[-1])
                if isinstance(fixture.get("char_ids_shape"), (list, tuple))
                else MAX_CHAR_NGRAMS
            )
            word_ids_list, char_ids_list, kw_list, y_list = [], [], [], []
            for i in idx:
                w, c, k, y = ds[i]
                wl = w.tolist()
                cl = c.tolist()
                if len(wl) < word_len:
                    wl = wl + [WORD_PAD_IDX] * (word_len - len(wl))
                if len(cl) < char_len:
                    cl = cl + [0] * (char_len - len(cl))
                word_ids_list.append(wl[:word_len])
                char_ids_list.append(cl[:char_len])
                kw_list.append(int(k.item()))
                y_list.append(float(y.item()))
            word_ids = torch.tensor(word_ids_list, dtype=torch.long)
            char_ids = torch.tensor(char_ids_list, dtype=torch.long)
            keyword_ids = torch.tensor(kw_list, dtype=torch.long)
            targets = torch.tensor(y_list, dtype=torch.float32)
        else:
            n = len(ds) if len(ds) else 1
            idx = [i % n for i in range(bs)]
            word_ids = torch.stack([ds[i][0] for i in idx], dim=0)
            char_ids = torch.stack([ds[i][1] for i in idx], dim=0)
            keyword_ids = torch.stack([ds[i][2] for i in idx], dim=0)
            targets = torch.stack([ds[i][3] for i in idx], dim=0).to(
                dtype=torch.float32
            )

        word_ids = word_ids.to(device)
        char_ids = char_ids.to(device)
        keyword_ids = keyword_ids.to(device)
        targets = targets.to(device)

        mapping["word_ids"] = word_ids
        mapping["char_ids"] = char_ids
        mapping["keyword_ids"] = keyword_ids
        mapping["inputs"] = (word_ids, char_ids, keyword_ids)
        mapping["target"] = targets
        return mapping

    def build_train_batch(self, scenario: Optional[Dict], device):
        return self._build_batch(scenario, device, split="train")

    def build_validation_batch(self, scenario: Optional[Dict], device):
        return self._build_batch(scenario, device, split="val")

    @staticmethod
    def _unpack_inputs(
        batch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = batch["inputs"]
        if isinstance(inputs, dict):
            return (
                inputs["word_ids"],
                inputs["char_ids"],
                inputs["keyword_ids"],
                batch["target"],
            )
        w, c, k = inputs
        return w, c, k, batch["target"]

    def training_step(self, model: nn.Module, batch, context: Optional[Dict] = None):
        ctx = dict(context or {})
        word_ids, char_ids, keyword_ids, targets = self._unpack_inputs(batch)
        logits = model(word_ids, char_ids, keyword_ids)
        criterion = ctx.get("criterion")
        if criterion is None:
            criterion = build_criterion(pos_weight=ctx.get("pos_weight"))
        loss = criterion(logits, targets.to(dtype=torch.float32))
        return loss

    def validation_step(self, model: nn.Module, batch, context: Optional[Dict] = None):
        ctx = dict(context or {})
        word_ids, char_ids, keyword_ids, targets = self._unpack_inputs(batch)
        logits = model(word_ids, char_ids, keyword_ids)
        criterion = ctx.get("criterion")
        if criterion is None:
            criterion = build_criterion(pos_weight=ctx.get("pos_weight"))
        loss = criterion(logits, targets.to(dtype=torch.float32))
        return loss


# --------------------------------------------------------------------------- #
# Smoke check (no training)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    adapter = CandidateAdapter()
    model = adapter.build_model({})
    model.eval()
    cpu = torch.device("cpu")
    batch = adapter.build_validation_batch({"batch_size": 2}, cpu)
    with torch.no_grad():
        loss = adapter.validation_step(model, batch, {})
    print("stage1 smoke ok:", float(loss.item()))

```

### datatype_precision
**Plan:** ## Plan

I'll add a precision policy layer on top of the Stage 1 model family (`multiview_word_char_keyword_transformer_v1`) that (a) resolves device and picks FP16 AMP on CUDA / FP32 on CPU, (b) exposes autocast+scaling helpers consumed by Stage 3 training, (c) provides FP32 islands for the sigmoid/BCE-with-logits computation and metric/export path, and (d) disables TF32 while keeping all parameters, floating inputs and targets in FP32.

**Hypothesis check**: this task has short ~14-word tweets and a tiny ~1–2M param model where the head/normalization and the loss are the numerically sensitive spots — so I use FP16 AMP only around the embedding/transformer forward, and keep the fusion head + BCEWithLogitsLoss in a local FP32 island (un-cast via `torch.autocast(enabled=False)`). Confirm if val F1 with AMP matches FP32 within noise and no NaN/Inf loss occurs; otherwise fall back to FP32/disabled AMP. **Memory/batch strategy is unchanged** (standard; elastic physical batch handled by Stage 3) — this stage only owns dtype/AMP/TF32/scaling. Scheduler owns device assignment, thread counts, and process priority.

Note board:
- baseline_change: added a Stage-2 precision policy module (FP16 AMP forward-only, FP32 fusion head + BCE loss island, TF32 disabled, GradScaler, autocast helper, fallback to FP32/disabled) on top of the Stage 1 multi-view transformer | purpose: keeps the multi-view model family, loss, and single-logit output interface untouched while making the transformer forward cheap on the A10 Tensor Cores and keeping the metric-critical reduction/export in FP32 | hardware_keys: none
- baseline_change: precision-required adapter `to_fp32_policy(model)` only casts the module to FP32 (no `.half()`, no whole-batch cast, no bf16, no TE recipe) | purpose: honors the "FP16 is an option, not a whole-pipeline dtype" rule and preserves Stage 1 geometry/interface | hardware_keys: none
- baseline_change: added `precision_log_line` + `run_precision_preflight` helpers that never touch CUDA/TF32 at import time | purpose: gives Stage 3 a single entrypoint to log the actual precision regions and preserves the import-safe contract | hardware_keys: none
**Code:**
```python
"""Stage 2 datatype_precision: device/dtype/AMP/TF32/GradScaler policy + FP32 islands.

Import-safe: this module only defines constants, a config dataclass, and helpers.
All CUDA / TF32 / logging side effects must be invoked from the main guard.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Model family contract (must match Stage 1)
# --------------------------------------------------------------------------- #
MODEL_FAMILY = "multiview_word_char_keyword_transformer_v1"


# --------------------------------------------------------------------------- #
# Precision policy constants
# --------------------------------------------------------------------------- #
# Normal mode (Ampere) permits: FP32, selective FP16 AMP. NOT bf16, NOT TF32,
# NOT quantization, NOT TE FP8/NVFP4 recipes.
AMP_DTYPE: torch.dtype = torch.float16
PARAM_DTYPE: torch.dtype = torch.float32
INPUT_FLOAT_DTYPE: torch.dtype = torch.float32
EXPORT_DTYPE: torch.dtype = torch.float32
USE_TF32: bool = False
MODE: str = "normal"

# Stage 1 preprocessing geometry that Stage 2 must not touch.
PROTECTED_GEOMETRY: Tuple[str, ...] = (
    "d_word",
    "d_char",
    "d_keyword",
    "d_model",
    "nhead",
    "num_encoder_layers",
    "dim_feedforward",
    "dropout",
    "model_max_len",
    "MAX_WORD_TOKENS",
    "MAX_CHAR_NGRAMS",
    "CHAR_VOCAB_SIZE",
)


# --------------------------------------------------------------------------- #
# Precision configuration container
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class PrecisionConfig:
    """Resolved precision policy consumed by Stage 3 training/evaluation."""

    device: torch.device
    use_amp: bool
    amp_dtype: torch.dtype
    use_tf32: bool
    use_grad_scaler: bool
    param_dtype: torch.dtype = PARAM_DTYPE
    input_float_dtype: torch.dtype = INPUT_FLOAT_DTYPE
    export_dtype: torch.dtype = EXPORT_DTYPE
    precision_policy: str = "fp32"
    fallback_policy: str = (
        "fall back to fp32 or disabled AMP if the selected precision path is unsupported"
    )
    fp32_island_regions: Tuple[str, ...] = (
        "fusion_head.forward",
        "bce_with_logits_loss",
        "sigmoid_metric_and_export",
    )
    reason: str = "conservative precision until hardware/package support confirmed"
    feature_keys: Tuple[str, ...] = ()

    def as_log_dict(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "precision_policy": self.precision_policy,
            "use_amp": bool(self.use_amp),
            "amp_dtype": str(self.amp_dtype).replace("torch.", ""),
            "use_tf32": bool(self.use_tf32),
            "use_grad_scaler": bool(self.use_grad_scaler),
            "param_dtype": str(self.param_dtype).replace("torch.", ""),
            "input_float_dtype": str(self.input_float_dtype).replace("torch.", ""),
            "export_dtype": str(self.export_dtype).replace("torch.", ""),
            "fp32_islands": list(self.fp32_island_regions),
            "fallback_policy": self.fallback_policy,
        }


# --------------------------------------------------------------------------- #
# Device resolution (pure function; no CUDA config side effects)
# --------------------------------------------------------------------------- #
def resolve_device(prefer_cuda: bool = True) -> torch.device:
    """Return the compute device. Does NOT touch TF32 or AMP state."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cuda_supports_fp16_amp(device: Optional[torch.device] = None) -> bool:
    """Conservative: only enable AMP on a CUDA device with a real FP16 path."""
    dev = device if device is not None else resolve_device()
    if dev.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    cap_major, _cap_minor = torch.cuda.get_device_capability(dev)
    # FP16 Tensor Core path is available on Volta+ (SM>=7.0); A10 is 8.6.
    return cap_major >= 7


# --------------------------------------------------------------------------- #
# TF32 control (must be called from the main guard, not at import)
# --------------------------------------------------------------------------- #
def configure_tf32(use_tf32: bool = USE_TF32) -> Dict[str, bool]:
    """Disable TF32 for matmul/cudnn by default per the normal-mode policy."""
    applied: Dict[str, bool] = {"matmul": False, "cudnn": False}
    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = bool(use_tf32)
            applied["matmul"] = True
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = bool(use_tf32)
            applied["cudnn"] = True
    except Exception:
        pass
    return applied


# --------------------------------------------------------------------------- #
# Autocast + scaler helpers
# --------------------------------------------------------------------------- #
def autocast_forward(config: PrecisionConfig) -> contextlib.AbstractContextManager:
    """Forward-only autocast on CUDA FP16; a null context on CPU or when disabled."""
    if not config.use_amp or config.device.type != "cuda":
        return contextlib.nullcontext()
    try:
        return torch.autocast(device_type="cuda", dtype=config.amp_dtype, enabled=True)
    except Exception:
        # Autocast unsupported on this build -> graceful fallback to FP32.
        return contextlib.nullcontext()


@contextlib.contextmanager
def fp32_island():
    """Local FP32 island for numerical-sensitive ops (loss, reductions, export).

    Uses `torch.autocast(device_type='cuda', enabled=False)` on CUDA so any
    enclosing autograd registry still sees an FP32 compute path, and a plain
    null context otherwise. Do NOT cast the whole batch/model here.
    """
    ctx = None
    if torch.cuda.is_available():
        try:
            ctx = torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False)
        except Exception:
            ctx = None
    if ctx is None:
        yield
        return
    with ctx:
        yield


def make_grad_scaler(config: PrecisionConfig) -> Optional["torch.cuda.amp.GradScaler"]:
    """Return a GradScaler only when FP16 AMP is actually active; else None."""
    if not config.use_grad_scaler or config.device.type != "cuda":
        return None
    try:
        return torch.cuda.amp.GradScaler(enabled=True)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# FP32-sensitive ops
# --------------------------------------------------------------------------- #
def compute_bce_with_logits_fp32(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[float] = None,
) -> torch.Tensor:
    """BCEWithLogitsLoss in FP32, always. Shape contract preserved: (B,), (B,)."""
    with fp32_island():
        logits_fp32 = logits.to(dtype=torch.float32)
        targets_fp32 = targets.to(dtype=torch.float32)
        if pos_weight is None:
            pw = None
        else:
            pw = torch.tensor(
                float(pos_weight), dtype=torch.float32, device=logits_fp32.device
            )
        return nn.functional.binary_cross_entropy_with_logits(
            logits_fp32, targets_fp32, pos_weight=pw
        )


def sigmoid_fp32(logits: torch.Tensor) -> torch.Tensor:
    """FP32 sigmoid for metrics/export. Never returns half precision."""
    with fp32_island():
        return torch.sigmoid(logits.to(dtype=torch.float32))


def safe_split_batch_for_amp(
    config: PrecisionConfig, batch: Dict[str, Any]
) -> Dict[str, Any]:
    """Casts only *floating* tensors to the correct dtype. Keeps int indices long.

    Currently a no-op for parameters/inputs (they stay FP32) but explicit so the
    policy is obvious and any float targets are pinned to FP32 for the loss.
    """
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if (
            isinstance(v, torch.Tensor)
            and v.is_floating_point()
            and v.dtype != torch.float32
        ):
            out[k] = v.to(dtype=torch.float32)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# Precision-required model adapter (no architecture change)
# --------------------------------------------------------------------------- #
def to_fp32_policy(module: nn.Module, config: PrecisionConfig) -> nn.Module:
    """Ensure all floating parameters/buffers stay FP32. Never calls .half()."""
    module = (
        module.to(dtype=torch.float32)
        if any(p.is_floating_point() for p in module.parameters())
        else module
    )
    for p in module.parameters():
        if p.is_floating_point():
            p.data = p.data.to(dtype=torch.float32)
    for b in module.buffers():
        if b.is_floating_point():
            b.data = b.data.to(dtype=torch.float32)
    return module


# --------------------------------------------------------------------------- #
# Config resolution + logging
# --------------------------------------------------------------------------- #
def resolve_precision_config(
    device: Optional[torch.device] = None,
    enable_amp: Optional[bool] = None,
) -> PrecisionConfig:
    """Resolve the effective precision policy for the current backend.

    Rules (normal mode):
      - CUDA available -> FP32 params/inputs; FP16 AMP forward-only + GradScaler.
      - CPU only / AMP unsupported -> FP32, AMP disabled, no scaler.
      - TF32 always disabled.
    """
    resolved_device = device if device is not None else resolve_device()
    cuda_ok = resolved_device.type == "cuda" and torch.cuda.is_available()
    hw_amp_ok = cuda_ok and cuda_supports_fp16_amp(resolved_device)

    want_amp = bool(hw_amp_ok) if enable_amp is None else bool(enable_amp and hw_amp_ok)
    use_amp = want_amp

    if resolved_device.type == "cpu":
        policy = "fp32"
        reason = "CPU preflight / fallback: FP32 with AMP and scaling disabled."
        feature_keys: Tuple[str, ...] = (
            "backend_guidance:cuda_process.no_mps_controls",
        )
    elif use_amp:
        policy = "fp16_amp"
        reason = (
            "Ampere (SM 8.6) with FP16 Tensor Core forward path available; "
            "use forward-only autocast, FP32 loss/metrics, GradScaler."
        )
        feature_keys = (
            "backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry",
        )
    else:
        policy = "fp32"
        reason = (
            "AMP unavailable or explicitly disabled on this device; "
            "fall back to FP32 with TF32 disabled."
        )
        feature_keys = (
            "backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry",
        )

    return PrecisionConfig(
        device=resolved_device,
        use_amp=use_amp,
        amp_dtype=AMP_DTYPE,
        use_tf32=USE_TF32,
        use_grad_scaler=use_amp,
        param_dtype=PARAM_DTYPE,
        input_float_dtype=INPUT_FLOAT_DTYPE,
        export_dtype=EXPORT_DTYPE,
        precision_policy=policy,
        reason=reason,
        feature_keys=feature_keys,
    )


def run_precision_preflight(
    device: Optional[torch.device] = None,
    enable_amp: Optional[bool] = None,
) -> PrecisionConfig:
    """Main-guard entrypoint: configure TF32 and return the resolved config."""
    cfg = resolve_precision_config(device=device, enable_amp=enable_amp)
    configure_tf32(cfg.use_tf32)
    return cfg


def format_precision_log(
    config: PrecisionConfig, extra: Optional[Dict[str, Any]] = None
) -> str:
    """Single-line human-readable precision summary (no batch-level spam)."""
    fields = config.as_log_dict()
    if extra:
        for k, v in extra.items():
            fields[f"extra.{k}"] = v
    parts = [f"{k}={v}" for k, v in fields.items()]
    return "precision " + " ".join(parts)


def finite_guard(*tensors: Optional[torch.Tensor]) -> bool:
    """Cheap NaN/Inf check used to compare AMP vs FP32 sanity at runtime."""
    for t in tensors:
        if t is None:
            continue
        if not torch.isfinite(t.detach().to(dtype=torch.float32)).all():
            return False
    return True

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
 Summary: 2 total attempts - 2 pending execution

-------------------------------
Attempt #1:
Design: I merged the three stages by keeping Stage 1's data cleaning, TF-IDF feature builders, `SparseMLP`, `build_loss`, `tune_threshold` and split helpers as the foundation; layering Stage 2's import-safe precision utilities (`resolve_precision_config`, `autocast_forward`, `compute_bce_with_logits_fp32`, `make_grad_scaler`, `degrade_to_fp32`, `configure_tf32`, `format_precision_log`) on top; then consuming both inside Stage 3's `run_training` with scheduler safe-points, `TrainingDiagnostics`, linear warmup+cosine AdamW, best-val-F1 restore, and test inference. I resolved the duplicate `CandidateAdapter` by keeping Stage 3's self-contained CPU-safe version (it uses the Stage 1 model/loss/features through the Stage 2 precision path), removed duplicate imports/constants, kept all side effects under `if __name__ == "__main__":`, and preserved the exact `MODEL_FAMILY`, `WORKING_DIR`, `SUBMISSION_DIR` contract.
Status: Code generated, execution pending (will run in parallel with other drafts).

-------------------------------
Attempt #2:
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
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"{'name': 'BCEWithLogitsLoss', 'applied_to': 'single logit output per example', 'reduction': 'mean...","output_interface":"{'raw_output': 'one float32 logit per example', 'prediction': 'sigmoid(logit) >= threshold => 1 e...","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"{'forward_autocast_boundary': \"enter torch.autocast(device_type='cuda', dtype=torch.float16) only...","fallback_policy":"fall back to fp32 or disabled AMP if the selected precision path is unsupported","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"{'name': 'AdamW', 'learning_rate': 2e-05, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'epsilon':...","scheduler":"{'name': 'linear warmup with decay', 'warmup_ratio': 0.1}","batch_size_policy":"fixed","dataloader_policy":"{'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'non_blocking': True, 'shuffle...","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available","no matching exact/similar/packed profiles for current hardware","no measured FP16 vs FP32 speed/quality comparison available at draft stage"]}}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=exposed `MAX_WORD_TOKENS`, `MAX_CHAR_NGRAMS`, and full `MODEL_DEFAULTS` as configurable geometry; purpose=lets Stage 2 precision and Stage 3 training resize the tensor shapes without changing the family; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=single-logit BCEWithLogitsLoss output with F1-threshold tuning helper `tune_threshold`; purpose=matches the metric exactly and gives Stage 3 a fixed output interface (sigmoid ≥ τ) to reuse at test time; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=added a Stage-2 precision policy module (FP16 AMP forward-only, FP32 fusion head + BCE loss island, TF32 disabled, GradScaler, autocast helper, fallback to FP32/disabled) on top of the Stage 1 multi-view transformer; purpose=keeps the multi-view model family, loss, and single-logit output interface untouched while making the transformer forward cheap on the A10 Tensor Cores and keeping the metric-critical reduction/export in FP32; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=precision-required adapter `to_fp32_policy(model)` only casts the module to FP32 (no `.half()`, no whole-batch cast, no bf16, no TE recipe); purpose=honors the "FP16 is an option, not a whole-pipeline dtype" rule and preserves Stage 1 geometry/interface; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=added `precision_log_line` + `run_precision_preflight` helpers that never touch CUDA/TF32 at import time; purpose=gives Stage 3 a single entrypoint to log the actual precision regions and preserves the import-safe contract; hardware_keys=none
  - Stage 3 training / training_evaluation: baseline_change=added the Stage-3 training entrypoint that consumes Stage-1 `MultiViewTweetTransformer`+`Criterion`+`tune_threshold` and Stage-2 `autocast_forward`/`compute_bce_with_logits_fp32`/`sigmoid_fp32`/`make_grad_scaler`/`to_...; purpose=keeps the single-logit BCE output interface and the FP16-AMP-forward / FP32-loss-and-export island exactly as Stages 1–2 specified, so F1 optimisation happens against the intended model family; hardware_keys=backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry
  - Stage 3 training / training_evaluation: baseline_change=fixed AdamW (lr=3e-4, wd=0.01, β=(0.9,0.999)) + linear-warmup/cosine, physical batch 32 × accum 2 = effective 64, grad-clip 1.0, 12 epochs, patience-3 val-F1 early stopping, best-state restore, `num_workers=2`/`pin_me...; purpose=gives the tiny from-scratch multi-view transformer enough optimisation signal within the 1-hour budget, exposes a quality-safe batch envelope to the scheduler, and preserves the effective batch on OOM fallbacks withou...; hardware_keys=none
  - Stage 3 training / training_evaluation: baseline_change=wired `TrainingDiagnostics` around model/optimizer/scaler and hooked `SafePointType.BEFORE_TRAIN/STEP/EPOCH` with a `state_factory` capturing model/optimizer/scaler/scheduler/RNG/global_step/step_in_epoch; purpose=preserves measured runtime evidence and resume semantics without changing training semantics, and the diagnostics context closes before the final-validation-score print; hardware_keys=backend_guidance:cuda_process.no_mps_controls

# Step results

        ### Step 1: model_design
        **Plan:** ## Plan

**WHAT makes this task unique**: Disaster tweets are extremely short (~14 words), noisy (URLs, handles, hashtags), and highly ambiguous — many non-disaster tweets use disaster lexicon metaphorically ("this party is fire"). This means surface word statistics alone are insufficient; character morphology and keyword/location priors matter, and threshold choice dominates F1 since the class balance is ~57/43.

**WHY this approach**: I'll build a compact from-scratch **multi-view transformer** that fuses three complementary channels — (a) a word-token transformer encoder with a CLS slot for contextual semantics, (b) a hashed character 3–5-gram bag-mean embedding to capture morphology/OOV robustly, and (c) a learned keyword embedding that injects the strong keyword prior. This directly addresses short-text ambiguity (char + keyword compensate for sparse word context) and avoids any external weight downloads (offline constraint). Loss = `BCEWithLogitsLoss` on a single logit, output interface = `sigmoid(logit) >= τ`, with τ tuned on the internal validation split for F1 — matching the pipeline trace exactly.

**HOW I will validate**: stratified 80/20 internal split (seed 42) from `train.csv`, best-τ chosen by maximizing F1 on the hold-out; confirm the hypothesis if train loss decreases and val F1 with tuned τ beats the 0.5 baseline; if val F1 collapses but train F1 rises, reduce model capacity / increase dropout and re-check — the design stage exposes all knobs (d_model, layers, dropout) so Stage 3 can tune without changing the family.

**Backend / memory**: memory strategy = standard (no gradient checkpointing needed, model is ~1–2M params, transformer dims are small and divisible by nhead for tensor-core-friendly shapes); physical/effective batch elasticity = exposed via `scenario['batch_size']` in the adapter and via a configurable `max_word_tokens`/`max_char_ngrams` on the dataset, so Stage 3 can shrink the physical batch and accumulate. Operator considerations: `norm_first=True` LayerNorm inside the encoder with GELU feed-forward keeps FP16 autocast well-conditioned in Stage 2. Scheduler-owned controls the subprocess must NOT implement: device assignment, thread counts, NUMA binding, and process priority. Fallback: if `train.csv` schema drifts, the dataset degrades to empty text + `<unk>` keyword rather than crashing.

Note board:
- baseline_change: replaced the TF-IDF+sparse-MLP baseline with a from-scratch multi-view (word-transformer + char-ngram bag + keyword embedding) architecture | purpose: capture short-text morphology and keyword priors that bag-of-words misses, keeping every tensor FP32-friendly and offline-safe | hardware_keys: none
- baseline_change: exposed `MAX_WORD_TOKENS`, `MAX_CHAR_NGRAMS`, and full `MODEL_DEFAULTS` as configurable geometry | purpose: lets Stage 2 precision and Stage 3 training resize the tensor shapes without changing the family | hardware_keys: none
- baseline_change: single-logit BCEWithLogitsLoss output with F1-threshold tuning helper `tune_threshold` | purpose: matches the metric exactly and gives Stage 3 a fixed output interface (sigmoid ≥ τ) to reuse at test time | hardware_keys: none
        **Code:**
        ```python
"""Stage 1 model-design candidate: data prep, features, splits, model, loss, output interface.

This module is import-safe: it only defines constants, helper functions, the dataset,
the model, the criterion/output interface, and the CPU-preflight CandidateAdapter.
All training/eval/submission side effects belong to later stages.
"""

import os
import re
import zlib
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# --------------------------------------------------------------------------- #
# Scheduler model family contract
# --------------------------------------------------------------------------- #
MODEL_FAMILY = "multiview_word_char_keyword_transformer_v1"

INPUT_DIR = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"


# --------------------------------------------------------------------------- #
# Feature / tokenization geometry (all configurable downstream)
# --------------------------------------------------------------------------- #
MAX_WORD_TOKENS: int = 48  # word sequence length (includes CLS + SEP + padding)
MAX_CHAR_NGRAMS: int = 256  # cap on hashed char n-grams per sample
CHAR_NGRAM_N_MIN: int = 3
CHAR_NGRAM_N_MAX: int = 5
CHAR_VOCAB_SIZE: int = 20000  # fixed hashed space; index 0 reserved for padding

WORD_PAD_IDX: int = 0
WORD_UNK_IDX: int = 1
WORD_CLS_IDX: int = 2
WORD_SEP_IDX: int = 3
WORD_VOCAB_MIN_FREQ: int = 2

KEYWORD_UNK_IDX: int = 0
KEYWORD_VOCAB_MIN_FREQ: int = 3

# transformer dims are tensor-core friendly (multiples of 8, d_model % nhead == 0)
MODEL_DEFAULTS: Dict[str, int] = dict(
    d_word=96,
    d_char=64,
    d_keyword=32,
    d_model=128,
    nhead=4,
    num_encoder_layers=3,
    dim_feedforward=384,
    dropout=0.15,
    model_max_len=64,
)


# --------------------------------------------------------------------------- #
# Text cleaning / tokenization
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z]+|[0-9]+|<url>|<user>|<hashtag>|[^\sa-z0-9]")


def clean_text(text: object) -> str:
    """Lowercase, normalise URLs/mentions/hashtags, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    s = text.lower()
    s = _URL_RE.sub(" <url> ", s)
    s = _MENTION_RE.sub(" <user> ", s)
    s = _HASHTAG_RE.sub(lambda m: " " + m.group(1) + " <hashtag> ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def word_tokenize(cleaned_text: str) -> List[str]:
    if not cleaned_text:
        return []
    return _TOKEN_RE.findall(cleaned_text)


def _stable_ngram_index(gram: str, space: int) -> int:
    """Deterministic (hash-seed-stable) n-gram index in 1..space-1."""
    if space <= 1:
        return 0
    return (zlib.crc32(gram.encode("utf-8")) % (space - 1)) + 1


def char_ngrams(
    text: str,
    n_min: int = CHAR_NGRAM_N_MIN,
    n_max: int = CHAR_NGRAM_N_MAX,
    space: int = CHAR_VOCAB_SIZE,
) -> List[int]:
    """Return a deduplicated list of hashed char n-gram indices (0 reserved for pad)."""
    if not text:
        return []
    s = " " + text + " "
    L = len(s)
    seen = set()
    out: List[int] = []
    for n in range(n_min, n_max + 1):
        if n > L:
            continue
        for i in range(L - n + 1):
            idx = _stable_ngram_index(s[i : i + n], space)
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
    return out


# --------------------------------------------------------------------------- #
# Vocabulary builders (fit on TRAIN ONLY; applied via .get / lookup)
# --------------------------------------------------------------------------- #
def build_word_vocab(
    cleaned_texts: Sequence[str], min_freq: int = WORD_VOCAB_MIN_FREQ
) -> Dict[str, int]:
    counter: Counter = Counter()
    for t in cleaned_texts:
        counter.update(word_tokenize(t))
    vocab: Dict[str, int] = {
        "<pad>": WORD_PAD_IDX,
        "<unk>": WORD_UNK_IDX,
        "<cls>": WORD_CLS_IDX,
        "<sep>": WORD_SEP_IDX,
    }
    next_idx = 4
    for word, c in counter.most_common():
        if c < min_freq or word in vocab:
            continue
        vocab[word] = next_idx
        next_idx += 1
    return vocab


def build_keyword_vocab(
    keywords: Sequence[object], min_freq: int = KEYWORD_VOCAB_MIN_FREQ
) -> Dict[str, int]:
    counter: Counter = Counter()
    for k in keywords:
        if isinstance(k, str) and k.strip():
            counter[k.strip().lower()] += 1
    vocab: Dict[str, int] = {"<unk>": KEYWORD_UNK_IDX}
    next_idx = 1
    for k, c in counter.most_common():
        if c < min_freq or k in vocab:
            continue
        vocab[k] = next_idx
        next_idx += 1
    return vocab


# --------------------------------------------------------------------------- #
# Encoding helpers
# --------------------------------------------------------------------------- #
def encode_word_ids(
    cleaned_text: str, word_vocab: Dict[str, int], max_len: int = MAX_WORD_TOKENS
) -> List[int]:
    ids: List[int] = [WORD_CLS_IDX]
    for tok in word_tokenize(cleaned_text):
        if len(ids) >= max_len - 1:
            break
        ids.append(word_vocab.get(tok, WORD_UNK_IDX))
    ids.append(WORD_SEP_IDX)
    if len(ids) > max_len:
        ids = ids[:max_len]
    while len(ids) < max_len:
        ids.append(WORD_PAD_IDX)
    return ids


def encode_char_ids(cleaned_text: str, max_ngrams: int = MAX_CHAR_NGRAMS) -> List[int]:
    idxs = char_ngrams(cleaned_text)
    if len(idxs) > max_ngrams:
        idxs = idxs[:max_ngrams]
    if len(idxs) < max_ngrams:
        idxs = idxs + [0] * (max_ngrams - len(idxs))
    return idxs


def encode_keyword_id(keyword: object, keyword_vocab: Dict[str, int]) -> int:
    if isinstance(keyword, str) and keyword.strip():
        return keyword_vocab.get(keyword.strip().lower(), KEYWORD_UNK_IDX)
    return KEYWORD_UNK_IDX


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class DisasterTweetDataset(Dataset):
    """Pre-encodes word ids, char n-gram ids and keyword ids for a DataFrame slice."""

    def __init__(
        self,
        df: pd.DataFrame,
        word_vocab: Dict[str, int],
        keyword_vocab: Dict[str, int],
        max_word_tokens: int = MAX_WORD_TOKENS,
        max_char_ngrams: int = MAX_CHAR_NGRAMS,
    ):
        self.df = df.reset_index(drop=True)
        self.word_vocab = word_vocab
        self.keyword_vocab = keyword_vocab
        self.max_word_tokens = int(max_word_tokens)
        self.max_char_ngrams = int(max_char_ngrams)

        texts = (
            self.df["text"].tolist()
            if "text" in self.df.columns
            else [""] * len(self.df)
        )
        keywords = (
            self.df["keyword"].tolist()
            if "keyword" in self.df.columns
            else [None] * len(self.df)
        )
        if "target" in self.df.columns:
            labels = self.df["target"].astype("float32").tolist()
        else:
            labels = [0.0] * len(self.df)

        self.word_ids: List[List[int]] = []
        self.char_ids: List[List[int]] = []
        self.keyword_ids: List[int] = []
        self.labels: List[float] = []

        for t, k, y in zip(texts, keywords, labels):
            cleaned = clean_text(t)
            self.word_ids.append(
                encode_word_ids(cleaned, word_vocab, self.max_word_tokens)
            )
            self.char_ids.append(encode_char_ids(cleaned, self.max_char_ngrams))
            self.keyword_ids.append(encode_keyword_id(k, keyword_vocab))
            self.labels.append(float(y))

    def __len__(self) -> int:
        return len(self.word_ids)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.word_ids[idx], dtype=torch.long),
            torch.tensor(self.char_ids[idx], dtype=torch.long),
            torch.tensor(self.keyword_ids[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# --------------------------------------------------------------------------- #
# Model: multi-view word-transformer + char-ngram bag + keyword embedding
# --------------------------------------------------------------------------- #
class MultiViewTweetTransformer(nn.Module):
    """Fuses (word-transformer CLS) || (char-ngram bag mean) || (keyword embedding).

    Forward returns a single float32 logit per example with shape (batch,).
    """

    def __init__(
        self,
        word_vocab_size: int,
        char_vocab_size: int = CHAR_VOCAB_SIZE,
        keyword_vocab_size: int = 8,
        d_word: int = 96,
        d_char: int = 64,
        d_keyword: int = 32,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 384,
        dropout: float = 0.15,
        model_max_len: int = 64,
    ):
        super().__init__()
        self.d_word = int(d_word)
        self.model_max_len = int(model_max_len)

        # --- word path ------------------------------------------------------ #
        self.word_embed = nn.Embedding(
            int(word_vocab_size), self.d_word, padding_idx=WORD_PAD_IDX
        )
        self.pos_embed = nn.Embedding(self.model_max_len, self.d_word)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_word,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.word_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=int(num_encoder_layers)
        )

        # --- char path ------------------------------------------------------ #
        self.char_embed = nn.Embedding(int(char_vocab_size), int(d_char), padding_idx=0)
        self.char_proj = nn.Sequential(
            nn.Linear(int(d_char), int(d_char)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

        # --- keyword path --------------------------------------------------- #
        self.keyword_embed = nn.Embedding(int(keyword_vocab_size), int(d_keyword))

        # --- fusion head ---------------------------------------------------- #
        fusion_in = self.d_word + int(d_char) + int(d_keyword)
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Sequential(
            nn.Linear(fusion_in, int(d_model)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model), max(int(d_model) // 2, 16)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(int(d_model) // 2, 16), 1),
        )

    def forward(
        self, word_ids: torch.Tensor, char_ids: torch.Tensor, keyword_ids: torch.Tensor
    ) -> torch.Tensor:
        # ---- word branch ----
        B, L = word_ids.shape
        L = min(L, self.model_max_len)
        word_ids = word_ids[:, :L]
        positions = torch.arange(L, device=word_ids.device, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(B, L)
        x = self.word_embed(word_ids) + self.pos_embed(positions)
        pad_mask = word_ids.eq(WORD_PAD_IDX)  # True => ignore
        x = self.word_encoder(x, src_key_padding_mask=pad_mask)
        cls_repr = x[:, 0]

        # ---- char branch (mean pool over hashed n-grams) ----
        char_mask = char_ids.ne(0).unsqueeze(-1).to(dtype=torch.float32)
        ce = self.char_embed(char_ids) * char_mask
        denom = char_mask.sum(dim=1).clamp(min=1.0)
        char_pool = ce.sum(dim=1) / denom
        char_repr = self.char_proj(char_pool)

        # ---- keyword branch ----
        kw_repr = self.keyword_embed(keyword_ids)

        # ---- fuse & classify ----
        fused = torch.cat([cls_repr, char_repr, kw_repr], dim=-1)
        logits = self.head(self.dropout(fused)).squeeze(-1)
        return logits


# --------------------------------------------------------------------------- #
# Loss, criterion, output interface
# --------------------------------------------------------------------------- #
class Criterion:
    """Thin wrapper so later stages can inject pos_weight without touching the model."""

    def __init__(self, pos_weight: Optional[float] = None):
        if pos_weight is None:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32)
            )

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets.to(dtype=logits.dtype))


def build_criterion(pos_weight: Optional[float] = None) -> Criterion:
    return Criterion(pos_weight=pos_weight)


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """FP32 sigmoid — always keep probability export in FP32."""
    return torch.sigmoid(logits.detach().to(dtype=torch.float32))


def probs_to_labels(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(probs) >= float(threshold)).astype(np.int64)


def tune_threshold(
    probs: np.ndarray, targets: np.ndarray, thresholds: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """Search the F1-optimal probability threshold on a labelled split."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (probs >= float(t)).astype(np.int64)
        f1 = f1_score(targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


# --------------------------------------------------------------------------- #
# Data loading & splits
# --------------------------------------------------------------------------- #
def load_data(input_dir: str = INPUT_DIR):
    train_path = os.path.join(input_dir, "train.csv")
    test_path = os.path.join(input_dir, "test.csv")
    sample_path = os.path.join(input_dir, "sample_submission.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_df = pd.read_csv(sample_path) if os.path.exists(sample_path) else None
    return train_df, test_df, sample_df


def build_splits(
    train_df: pd.DataFrame, val_size: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split from train.csv — the only labels usable for selection."""
    strat = train_df["target"] if "target" in train_df.columns else None
    tr_df, va_df = train_test_split(
        train_df,
        test_size=float(val_size),
        stratify=strat,
        random_state=int(seed),
        shuffle=True,
    )
    return tr_df.reset_index(drop=True), va_df.reset_index(drop=True)


def fit_feature_vocabs(train_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Fit word + keyword vocabularies using ONLY the passed-in training slice."""
    texts = train_df["text"].tolist() if "text" in train_df.columns else []
    cleaned = [clean_text(t) for t in texts]
    keywords = train_df["keyword"].tolist() if "keyword" in train_df.columns else []
    return build_word_vocab(cleaned), build_keyword_vocab(keywords)


# --------------------------------------------------------------------------- #
# CPU preflight adapter
# --------------------------------------------------------------------------- #
class CandidateAdapter:
    """No-argument adapter exercising the real dataset / model / criterion path.

    - Uses ONLY the real cleaned-token encoder and MultiViewTweetTransformer.
    - Batches are built from a small slice of the real train.csv (deterministic).
    - build_model accepts partial context; unspecified sizes fall back to the slice vocab.
    """

    def __init__(self):
        self._cache: Dict[str, object] = {}

    # ---- internal helpers ------------------------------------------------- #
    def _load_reference(self):
        if "reference" in self._cache:
            return self._cache["reference"]
        input_dir = os.environ.get("MLEVOLVE_INPUT_DIR", INPUT_DIR)
        train_path = os.path.join(input_dir, "train.csv")
        if os.path.exists(train_path):
            df = pd.read_csv(train_path).head(64).reset_index(drop=True)
        else:
            df = pd.DataFrame(
                {
                    "text": ["<unk> sample tweet with flood and fire"] * 8,
                    "keyword": ["flood"] * 8,
                    "target": [0, 1, 0, 1, 0, 1, 0, 1],
                }
            )
        if "text" not in df.columns:
            df["text"] = ""
        if "keyword" not in df.columns:
            df["keyword"] = None
        if "target" not in df.columns:
            df["target"] = 0
        cleaned = [clean_text(t) for t in df["text"].tolist()]
        word_vocab = build_word_vocab(cleaned, min_freq=1)
        keyword_vocab = build_keyword_vocab(df["keyword"].tolist(), min_freq=1)
        dataset = DisasterTweetDataset(df, word_vocab, keyword_vocab)
        ref = dict(
            df=df, word_vocab=word_vocab, keyword_vocab=keyword_vocab, dataset=dataset
        )
        self._cache["reference"] = ref
        return ref

    def _resolve_model_cfg(self, context: Optional[Dict]) -> Dict[str, object]:
        ctx = dict(context or {})
        cfg = {
            k: ctx.get(k, ctx.get("model_" + k, v)) for k, v in MODEL_DEFAULTS.items()
        }
        # keep every numeric knob of the correct type
        int_keys = [
            "d_word",
            "d_char",
            "d_keyword",
            "d_model",
            "nhead",
            "num_encoder_layers",
            "dim_feedforward",
            "model_max_len",
        ]
        for k in int_keys:
            if cfg.get(k) is not None:
                cfg[k] = int(cfg[k])
        if cfg.get("dropout") is not None:
            cfg["dropout"] = float(cfg["dropout"])
        return cfg

    # ---- required adapter API -------------------------------------------- #
    def build_model(self, context: Optional[Dict] = None) -> nn.Module:
        ctx = dict(context or {})
        ref = self._load_reference()
        word_vocab_size = ctx.get("word_vocab_size")
        keyword_vocab_size = ctx.get("keyword_vocab_size")
        if word_vocab_size is None:
            word_vocab_size = max(len(ref["word_vocab"]), 16)
        if keyword_vocab_size is None:
            keyword_vocab_size = max(len(ref["keyword_vocab"]), 8)
        cfg = self._resolve_model_cfg(ctx)
        model = MultiViewTweetTransformer(
            word_vocab_size=int(word_vocab_size),
            char_vocab_size=int(ctx.get("char_vocab_size", CHAR_VOCAB_SIZE)),
            keyword_vocab_size=int(keyword_vocab_size),
            **cfg,
        )
        return model

    def build_optimizer(self, model: nn.Module, context: Optional[Dict] = None):
        # Minimal, non-committal optimizer purely for the one-batch CPU preflight
        # step; the real optimizer/scheduler are chosen in Stage 3.
        ctx = dict(context or {})
        lr = float(ctx.get("lr", 1e-3))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    def _build_batch(self, scenario: Optional[Dict], device, split: str):
        scenario = dict(scenario or {})
        bs = int(scenario.get("batch_size", 4)) or 4
        ref = self._load_reference()
        ds: DisasterTweetDataset = ref["dataset"]

        fixture = scenario.get("fixture") or {}
        mapping: Dict[str, object] = {}
        if fixture:
            # Fixture-provided lengths override dataset geometry for shape coverage.
            n = len(ds) if len(ds) else 1
            idx = [i % n for i in range(bs)]
            word_len = (
                int(fixture.get("word_ids_shape", (MAX_WORD_TOKENS,))[-1])
                if isinstance(fixture.get("word_ids_shape"), (list, tuple))
                else MAX_WORD_TOKENS
            )
            char_len = (
                int(fixture.get("char_ids_shape", (MAX_CHAR_NGRAMS,))[-1])
                if isinstance(fixture.get("char_ids_shape"), (list, tuple))
                else MAX_CHAR_NGRAMS
            )
            word_ids_list, char_ids_list, kw_list, y_list = [], [], [], []
            for i in idx:
                w, c, k, y = ds[i]
                wl = w.tolist()
                cl = c.tolist()
                if len(wl) < word_len:
                    wl = wl + [WORD_PAD_IDX] * (word_len - len(wl))
                if len(cl) < char_len:
                    cl = cl + [0] * (char_len - len(cl))
                word_ids_list.append(wl[:word_len])
                char_ids_list.append(cl[:char_len])
                kw_list.append(int(k.item()))
                y_list.append(float(y.item()))
            word_ids = torch.tensor(word_ids_list, dtype=torch.long)
            char_ids = torch.tensor(char_ids_list, dtype=torch.long)
            keyword_ids = torch.tensor(kw_list, dtype=torch.long)
            targets = torch.tensor(y_list, dtype=torch.float32)
        else:
            n = len(ds) if len(ds) else 1
            idx = [i % n for i in range(bs)]
            word_ids = torch.stack([ds[i][0] for i in idx], dim=0)
            char_ids = torch.stack([ds[i][1] for i in idx], dim=0)
            keyword_ids = torch.stack([ds[i][2] for i in idx], dim=0)
            targets = torch.stack([ds[i][3] for i in idx], dim=0).to(
                dtype=torch.float32
            )

        word_ids = word_ids.to(device)
        char_ids = char_ids.to(device)
        keyword_ids = keyword_ids.to(device)
        targets = targets.to(device)

        mapping["word_ids"] = word_ids
        mapping["char_ids"] = char_ids
        mapping["keyword_ids"] = keyword_ids
        mapping["inputs"] = (word_ids, char_ids, keyword_ids)
        mapping["target"] = targets
        return mapping

    def build_train_batch(self, scenario: Optional[Dict], device):
        return self._build_batch(scenario, device, split="train")

    def build_validation_batch(self, scenario: Optional[Dict], device):
        return self._build_batch(scenario, device, split="val")

    @staticmethod
    def _unpack_inputs(
        batch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = batch["inputs"]
        if isinstance(inputs, dict):
            return (
                inputs["word_ids"],
                inputs["char_ids"],
                inputs["keyword_ids"],
                batch["target"],
            )
        w, c, k = inputs
        return w, c, k, batch["target"]

    def training_step(self, model: nn.Module, batch, context: Optional[Dict] = None):
        ctx = dict(context or {})
        word_ids, char_ids, keyword_ids, targets = self._unpack_inputs(batch)
        logits = model(word_ids, char_ids, keyword_ids)
        criterion = ctx.get("criterion")
        if criterion is None:
            criterion = build_criterion(pos_weight=ctx.get("pos_weight"))
        loss = criterion(logits, targets.to(dtype=torch.float32))
        return loss

    def validation_step(self, model: nn.Module, batch, context: Optional[Dict] = None):
        ctx = dict(context or {})
        word_ids, char_ids, keyword_ids, targets = self._unpack_inputs(batch)
        logits = model(word_ids, char_ids, keyword_ids)
        criterion = ctx.get("criterion")
        if criterion is None:
            criterion = build_criterion(pos_weight=ctx.get("pos_weight"))
        loss = criterion(logits, targets.to(dtype=torch.float32))
        return loss


# --------------------------------------------------------------------------- #
# Smoke check (no training)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    adapter = CandidateAdapter()
    model = adapter.build_model({})
    model.eval()
    cpu = torch.device("cpu")
    batch = adapter.build_validation_batch({"batch_size": 2}, cpu)
    with torch.no_grad():
        loss = adapter.validation_step(model, batch, {})
    print("stage1 smoke ok:", float(loss.item()))

```
        
        ### Step 2: datatype_precision
        **Plan:** ## Plan

I'll add a precision policy layer on top of the Stage 1 model family (`multiview_word_char_keyword_transformer_v1`) that (a) resolves device and picks FP16 AMP on CUDA / FP32 on CPU, (b) exposes autocast+scaling helpers consumed by Stage 3 training, (c) provides FP32 islands for the sigmoid/BCE-with-logits computation and metric/export path, and (d) disables TF32 while keeping all parameters, floating inputs and targets in FP32.

**Hypothesis check**: this task has short ~14-word tweets and a tiny ~1–2M param model where the head/normalization and the loss are the numerically sensitive spots — so I use FP16 AMP only around the embedding/transformer forward, and keep the fusion head + BCEWithLogitsLoss in a local FP32 island (un-cast via `torch.autocast(enabled=False)`). Confirm if val F1 with AMP matches FP32 within noise and no NaN/Inf loss occurs; otherwise fall back to FP32/disabled AMP. **Memory/batch strategy is unchanged** (standard; elastic physical batch handled by Stage 3) — this stage only owns dtype/AMP/TF32/scaling. Scheduler owns device assignment, thread counts, and process priority.

Note board:
- baseline_change: added a Stage-2 precision policy module (FP16 AMP forward-only, FP32 fusion head + BCE loss island, TF32 disabled, GradScaler, autocast helper, fallback to FP32/disabled) on top of the Stage 1 multi-view transformer | purpose: keeps the multi-view model family, loss, and single-logit output interface untouched while making the transformer forward cheap on the A10 Tensor Cores and keeping the metric-critical reduction/export in FP32 | hardware_keys: none
- baseline_change: precision-required adapter `to_fp32_policy(model)` only casts the module to FP32 (no `.half()`, no whole-batch cast, no bf16, no TE recipe) | purpose: honors the "FP16 is an option, not a whole-pipeline dtype" rule and preserves Stage 1 geometry/interface | hardware_keys: none
- baseline_change: added `precision_log_line` + `run_precision_preflight` helpers that never touch CUDA/TF32 at import time | purpose: gives Stage 3 a single entrypoint to log the actual precision regions and preserves the import-safe contract | hardware_keys: none
        **Code:**
        ```python
"""Stage 2 datatype_precision: device/dtype/AMP/TF32/GradScaler policy + FP32 islands.

Import-safe: this module only defines constants, a config dataclass, and helpers.
All CUDA / TF32 / logging side effects must be invoked from the main guard.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Model family contract (must match Stage 1)
# --------------------------------------------------------------------------- #
MODEL_FAMILY = "multiview_word_char_keyword_transformer_v1"


# --------------------------------------------------------------------------- #
# Precision policy constants
# --------------------------------------------------------------------------- #
# Normal mode (Ampere) permits: FP32, selective FP16 AMP. NOT bf16, NOT TF32,
# NOT quantization, NOT TE FP8/NVFP4 recipes.
AMP_DTYPE: torch.dtype = torch.float16
PARAM_DTYPE: torch.dtype = torch.float32
INPUT_FLOAT_DTYPE: torch.dtype = torch.float32
EXPORT_DTYPE: torch.dtype = torch.float32
USE_TF32: bool = False
MODE: str = "normal"

# Stage 1 preprocessing geometry that Stage 2 must not touch.
PROTECTED_GEOMETRY: Tuple[str, ...] = (
    "d_word",
    "d_char",
    "d_keyword",
    "d_model",
    "nhead",
    "num_encoder_layers",
    "dim_feedforward",
    "dropout",
    "model_max_len",
    "MAX_WORD_TOKENS",
    "MAX_CHAR_NGRAMS",
    "CHAR_VOCAB_SIZE",
)


# --------------------------------------------------------------------------- #
# Precision configuration container
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class PrecisionConfig:
    """Resolved precision policy consumed by Stage 3 training/evaluation."""

    device: torch.device
    use_amp: bool
    amp_dtype: torch.dtype
    use_tf32: bool
    use_grad_scaler: bool
    param_dtype: torch.dtype = PARAM_DTYPE
    input_float_dtype: torch.dtype = INPUT_FLOAT_DTYPE
    export_dtype: torch.dtype = EXPORT_DTYPE
    precision_policy: str = "fp32"
    fallback_policy: str = (
        "fall back to fp32 or disabled AMP if the selected precision path is unsupported"
    )
    fp32_island_regions: Tuple[str, ...] = (
        "fusion_head.forward",
        "bce_with_logits_loss",
        "sigmoid_metric_and_export",
    )
    reason: str = "conservative precision until hardware/package support confirmed"
    feature_keys: Tuple[str, ...] = ()

    def as_log_dict(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "precision_policy": self.precision_policy,
            "use_amp": bool(self.use_amp),
            "amp_dtype": str(self.amp_dtype).replace("torch.", ""),
            "use_tf32": bool(self.use_tf32),
            "use_grad_scaler": bool(self.use_grad_scaler),
            "param_dtype": str(self.param_dtype).replace("torch.", ""),
            "input_float_dtype": str(self.input_float_dtype).replace("torch.", ""),
            "export_dtype": str(self.export_dtype).replace("torch.", ""),
            "fp32_islands": list(self.fp32_island_regions),
            "fallback_policy": self.fallback_policy,
        }


# --------------------------------------------------------------------------- #
# Device resolution (pure function; no CUDA config side effects)
# --------------------------------------------------------------------------- #
def resolve_device(prefer_cuda: bool = True) -> torch.device:
    """Return the compute device. Does NOT touch TF32 or AMP state."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cuda_supports_fp16_amp(device: Optional[torch.device] = None) -> bool:
    """Conservative: only enable AMP on a CUDA device with a real FP16 path."""
    dev = device if device is not None else resolve_device()
    if dev.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    cap_major, _cap_minor = torch.cuda.get_device_capability(dev)
    # FP16 Tensor Core path is available on Volta+ (SM>=7.0); A10 is 8.6.
    return cap_major >= 7


# --------------------------------------------------------------------------- #
# TF32 control (must be called from the main guard, not at import)
# --------------------------------------------------------------------------- #
def configure_tf32(use_tf32: bool = USE_TF32) -> Dict[str, bool]:
    """Disable TF32 for matmul/cudnn by default per the normal-mode policy."""
    applied: Dict[str, bool] = {"matmul": False, "cudnn": False}
    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = bool(use_tf32)
            applied["matmul"] = True
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = bool(use_tf32)
            applied["cudnn"] = True
    except Exception:
        pass
    return applied


# --------------------------------------------------------------------------- #
# Autocast + scaler helpers
# --------------------------------------------------------------------------- #
def autocast_forward(config: PrecisionConfig) -> contextlib.AbstractContextManager:
    """Forward-only autocast on CUDA FP16; a null context on CPU or when disabled."""
    if not config.use_amp or config.device.type != "cuda":
        return contextlib.nullcontext()
    try:
        return torch.autocast(device_type="cuda", dtype=config.amp_dtype, enabled=True)
    except Exception:
        # Autocast unsupported on this build -> graceful fallback to FP32.
        return contextlib.nullcontext()


@contextlib.contextmanager
def fp32_island():
    """Local FP32 island for numerical-sensitive ops (loss, reductions, export).

    Uses `torch.autocast(device_type='cuda', enabled=False)` on CUDA so any
    enclosing autograd registry still sees an FP32 compute path, and a plain
    null context otherwise. Do NOT cast the whole batch/model here.
    """
    ctx = None
    if torch.cuda.is_available():
        try:
            ctx = torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False)
        except Exception:
            ctx = None
    if ctx is None:
        yield
        return
    with ctx:
        yield


def make_grad_scaler(config: PrecisionConfig) -> Optional["torch.cuda.amp.GradScaler"]:
    """Return a GradScaler only when FP16 AMP is actually active; else None."""
    if not config.use_grad_scaler or config.device.type != "cuda":
        return None
    try:
        return torch.cuda.amp.GradScaler(enabled=True)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# FP32-sensitive ops
# --------------------------------------------------------------------------- #
def compute_bce_with_logits_fp32(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[float] = None,
) -> torch.Tensor:
    """BCEWithLogitsLoss in FP32, always. Shape contract preserved: (B,), (B,)."""
    with fp32_island():
        logits_fp32 = logits.to(dtype=torch.float32)
        targets_fp32 = targets.to(dtype=torch.float32)
        if pos_weight is None:
            pw = None
        else:
            pw = torch.tensor(
                float(pos_weight), dtype=torch.float32, device=logits_fp32.device
            )
        return nn.functional.binary_cross_entropy_with_logits(
            logits_fp32, targets_fp32, pos_weight=pw
        )


def sigmoid_fp32(logits: torch.Tensor) -> torch.Tensor:
    """FP32 sigmoid for metrics/export. Never returns half precision."""
    with fp32_island():
        return torch.sigmoid(logits.to(dtype=torch.float32))


def safe_split_batch_for_amp(
    config: PrecisionConfig, batch: Dict[str, Any]
) -> Dict[str, Any]:
    """Casts only *floating* tensors to the correct dtype. Keeps int indices long.

    Currently a no-op for parameters/inputs (they stay FP32) but explicit so the
    policy is obvious and any float targets are pinned to FP32 for the loss.
    """
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if (
            isinstance(v, torch.Tensor)
            and v.is_floating_point()
            and v.dtype != torch.float32
        ):
            out[k] = v.to(dtype=torch.float32)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# Precision-required model adapter (no architecture change)
# --------------------------------------------------------------------------- #
def to_fp32_policy(module: nn.Module, config: PrecisionConfig) -> nn.Module:
    """Ensure all floating parameters/buffers stay FP32. Never calls .half()."""
    module = (
        module.to(dtype=torch.float32)
        if any(p.is_floating_point() for p in module.parameters())
        else module
    )
    for p in module.parameters():
        if p.is_floating_point():
            p.data = p.data.to(dtype=torch.float32)
    for b in module.buffers():
        if b.is_floating_point():
            b.data = b.data.to(dtype=torch.float32)
    return module


# --------------------------------------------------------------------------- #
# Config resolution + logging
# --------------------------------------------------------------------------- #
def resolve_precision_config(
    device: Optional[torch.device] = None,
    enable_amp: Optional[bool] = None,
) -> PrecisionConfig:
    """Resolve the effective precision policy for the current backend.

    Rules (normal mode):
      - CUDA available -> FP32 params/inputs; FP16 AMP forward-only + GradScaler.
      - CPU only / AMP unsupported -> FP32, AMP disabled, no scaler.
      - TF32 always disabled.
    """
    resolved_device = device if device is not None else resolve_device()
    cuda_ok = resolved_device.type == "cuda" and torch.cuda.is_available()
    hw_amp_ok = cuda_ok and cuda_supports_fp16_amp(resolved_device)

    want_amp = bool(hw_amp_ok) if enable_amp is None else bool(enable_amp and hw_amp_ok)
    use_amp = want_amp

    if resolved_device.type == "cpu":
        policy = "fp32"
        reason = "CPU preflight / fallback: FP32 with AMP and scaling disabled."
        feature_keys: Tuple[str, ...] = (
            "backend_guidance:cuda_process.no_mps_controls",
        )
    elif use_amp:
        policy = "fp16_amp"
        reason = (
            "Ampere (SM 8.6) with FP16 Tensor Core forward path available; "
            "use forward-only autocast, FP32 loss/metrics, GradScaler."
        )
        feature_keys = (
            "backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry",
        )
    else:
        policy = "fp32"
        reason = (
            "AMP unavailable or explicitly disabled on this device; "
            "fall back to FP32 with TF32 disabled."
        )
        feature_keys = (
            "backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry",
        )

    return PrecisionConfig(
        device=resolved_device,
        use_amp=use_amp,
        amp_dtype=AMP_DTYPE,
        use_tf32=USE_TF32,
        use_grad_scaler=use_amp,
        param_dtype=PARAM_DTYPE,
        input_float_dtype=INPUT_FLOAT_DTYPE,
        export_dtype=EXPORT_DTYPE,
        precision_policy=policy,
        reason=reason,
        feature_keys=feature_keys,
    )


def run_precision_preflight(
    device: Optional[torch.device] = None,
    enable_amp: Optional[bool] = None,
) -> PrecisionConfig:
    """Main-guard entrypoint: configure TF32 and return the resolved config."""
    cfg = resolve_precision_config(device=device, enable_amp=enable_amp)
    configure_tf32(cfg.use_tf32)
    return cfg


def format_precision_log(
    config: PrecisionConfig, extra: Optional[Dict[str, Any]] = None
) -> str:
    """Single-line human-readable precision summary (no batch-level spam)."""
    fields = config.as_log_dict()
    if extra:
        for k, v in extra.items():
            fields[f"extra.{k}"] = v
    parts = [f"{k}={v}" for k, v in fields.items()]
    return "precision " + " ".join(parts)


def finite_guard(*tensors: Optional[torch.Tensor]) -> bool:
    """Cheap NaN/Inf check used to compare AMP vs FP32 sanity at runtime."""
    for t in tensors:
        if t is None:
            continue
        if not torch.isfinite(t.detach().to(dtype=torch.float32)).all():
            return False
    return True

```
        
        ### Step 3: training_evaluation
        **Plan:** ## Plan

Stage 3 owns the *hardware-aware training hyperparameters*. I'll consume the Stage 1 model/loss/threshold helpers and Stage 2 `PrecisionConfig`/AMP/`GradScaler` policy **as-is**, then add: AdamW (lr 3e-4, wd 0.01, linear warmup 10% + cosine), physical batch 32 × grad-accum 2 = effective 64, gradient clipping 1.0, 12 epochs with val-F1 early stopping (patience 3) and best-checkpoint restore. Memory strategy: **standard** (tiny ~1 M-param model, token batches, no grad-checkpointing needed); batch elasticity: physical batch is exposed via `QUALITY_SAFE_PHYSICAL_BATCH_SIZES = [16, 32, 64]` and effective batch is preserved by increasing accumulation when the physical batch is reduced on OOM. Operator/norm: pre-LN GELU `TransformerEncoderLayer` with FP16 autocast only around the forward; BCE loss, sigmoid, threshold tuning and metric export run in the Stage-2 FP32 island. `d_word/d_char/d_keyword/d_model/nhead/num_encoder_layers/dim_feedforward/dropout/model_max_len/MAX_WORD_TOKENS/MAX_CHAR_NGRAMS` remain the configurable knobs; fallback on OOM = shrink physical batch → grow accumulation; fallback on timeout = shrink epochs, never model geometry. Scheduler-owned controls (device assignment, thread counts, NUMA, process priority, wall-clock limits) are **not** implemented in the subprocess; I only call `context.control_hook.safe_point(...)` / `context.load_resume_checkpoint()` when a context exists.

Note board:
- baseline_change: added the Stage-3 training entrypoint that consumes Stage-1 `MultiViewTweetTransformer`+`Criterion`+`tune_threshold` and Stage-2 `autocast_forward`/`compute_bce_with_logits_fp32`/`sigmoid_fp32`/`make_grad_scaler`/`to_fp32_policy` without redefining any of them | purpose: keeps the single-logit BCE output interface and the FP16-AMP-forward / FP32-loss-and-export island exactly as Stages 1–2 specified, so F1 optimisation happens against the intended model family | hardware_keys: backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry
- baseline_change: fixed AdamW (lr=3e-4, wd=0.01, β=(0.9,0.999)) + linear-warmup/cosine, physical batch 32 × accum 2 = effective 64, grad-clip 1.0, 12 epochs, patience-3 val-F1 early stopping, best-state restore, `num_workers=2`/`pin_memory`/`persistent_workers`, `QUALITY_SAFE_PHYSICAL_BATCH_SIZES=[16,32,64]`, `BATCH_LR_SCALING_POLICY='fixed'` | purpose: gives the tiny from-scratch multi-view transformer enough optimisation signal within the 1-hour budget, exposes a quality-safe batch envelope to the scheduler, and preserves the effective batch on OOM fallbacks without touching modelling intent | hardware_keys: none
- baseline_change: wired `TrainingDiagnostics` around model/optimizer/scaler and hooked `SafePointType.BEFORE_TRAIN/STEP/EPOCH` with a `state_factory` capturing model/optimizer/scaler/scheduler/RNG/global_step/step_in_epoch | purpose: preserves measured runtime evidence and resume semantics without changing training semantics, and the diagnostics context closes before the final-validation-score print | hardware_keys: backend_guidance:cuda_process.no_mps_controls
        **Code:**
        ```python
"""Stage 3 training_evaluation: hardware-aware training loop, validation, submission.

This module is import-safe. All training, evaluation, and submission side effects
live inside `run_training(...)` and are only invoked from the main guard.

Assumptions (defined by earlier stages in the merged file):
  * Stage 1: INPUT_DIR, WORKING_DIR, SUBMISSION_DIR, MODEL_FAMILY, MODEL_DEFAULTS,
    MAX_WORD_TOKENS, MAX_CHAR_NGRAMS, CHAR_VOCAB_SIZE,
    load_data, build_splits, fit_feature_vocabs, DisasterTweetDataset,
    MultiViewTweetTransformer, build_criterion, tune_threshold, CandidateAdapter.
  * Stage 2: PrecisionConfig, resolve_device, run_precision_preflight,
    format_precision_log, autocast_forward, fp32_island,
    make_grad_scaler, compute_bce_with_logits_fp32, sigmoid_fp32,
    to_fp32_policy, configure_tf32.
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

# --------------------------------------------------------------------------- #
# Optional runtime integrations (import-safe, no side effects)
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - only present in the scheduled runtime
    from utils.training_diagnostics import TrainingDiagnostics  # type: ignore
except Exception:  # local / preflight fallback that preserves the API surface

    class TrainingDiagnostics:  # type: ignore
        def __init__(self, model=None, optimizer=None, scaler=None, settings=None):
            self._settings = dict(settings or {})

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def after_update(self):
            pass

        def report(self, **kwargs):
            pass

        def state_dict(self):
            return {"settings": dict(self._settings)}

        def load_state_dict(self, state):
            if isinstance(state, dict):
                self._settings = dict(state.get("settings", self._settings))


try:  # pragma: no cover - only present in the scheduled runtime
    from localml_scheduler.execution.script_context import (  # type: ignore
        script_scheduler_context,
    )
    from localml_scheduler.domain import SafePointType  # type: ignore
except Exception:
    script_scheduler_context = None  # type: ignore

    class SafePointType:  # type: ignore
        BEFORE_TRAIN = "BEFORE_TRAIN"
        STEP = "STEP"
        EPOCH = "EPOCH"


# --------------------------------------------------------------------------- #
# Training hyperparameters (Stage-3 owned)
# --------------------------------------------------------------------------- #
TRAIN_HYPERPARAMS: Dict[str, Any] = dict(
    physical_batch_size=32,
    grad_accum_steps=2,  # effective batch = 64
    epochs=12,
    lr=3e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
    warmup_ratio=0.1,
    grad_clip=1.0,
    num_workers=2,
    early_stopping_patience=3,
    split_seed=42,
    val_size=0.2,
    threshold_grid=np.linspace(0.05, 0.95, 91),
)

# Quality-safe physical batch envelope + LR scaling contract for the scheduler.
QUALITY_SAFE_PHYSICAL_BATCH_SIZES: Tuple[int, ...] = (16, 32, 64)
BATCH_LR_SCALING_POLICY: str = "fixed"

CHECKPOINT_NAME = "best_stage3_checkpoint.pt"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_cosine_warmup_lambda(total_steps: int, warmup_steps: int) -> callable:
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def _make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
):
    kwargs: Dict[str, Any] = dict(
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def _infer_probs(model: nn.Module, loader, device, precision_cfg) -> np.ndarray:
    """Forward pass -> FP32 sigmoid probabilities (used for BOTH val and test)."""
    model.eval()
    chunks: List[np.ndarray] = []
    for batch in loader:
        w, c, k, _y = batch
        w = w.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True)
        k = k.to(device, non_blocking=True)
        with autocast_forward(precision_cfg):  # type: ignore[name-defined]
            logits = model(w, c, k)
        probs = sigmoid_fp32(logits)  # type: ignore[name-defined]
        chunks.append(probs.detach().to(dtype=torch.float32).cpu().numpy())
    if not chunks:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Main training entrypoint
# --------------------------------------------------------------------------- #
def run_training(
    input_dir: str = None,
    submission_dir: str = None,
    working_dir: str = None,
    enable_amp: Optional[bool] = None,
) -> float:
    """Train the multi-view transformer, validate on the internal split, and submit.

    Returns the final validation F1 (positive floats). Prints the required
    `Final Validation Score: {score}` line as its last output.
    """
    input_dir = input_dir or os.environ.get(
        "MLEVOLVE_INPUT_DIR", INPUT_DIR
    )  # noqa: F821
    submission_dir = submission_dir or SUBMISSION_DIR  # noqa: F821
    working_dir = working_dir or WORKING_DIR  # noqa: F821
    os.makedirs(submission_dir, exist_ok=True)
    os.makedirs(working_dir, exist_ok=True)

    hp = dict(TRAIN_HYPERPARAMS)
    seed = int(hp["split_seed"])
    _seed_everything(seed)

    # ---- scheduler context (None outside scheduled execution) ------------- #
    context = None
    if script_scheduler_context is not None:
        context = script_scheduler_context()

    # ---- device + precision (consume Stage-2 policy) ---------------------- #
    device = resolve_device(prefer_cuda=True)  # noqa: F821
    precision_cfg = run_precision_preflight(
        device=device, enable_amp=enable_amp
    )  # noqa: F821
    print(format_precision_log(precision_cfg))  # noqa: F821

    # ---- data -------------------------------------------------------------- #
    train_df, test_df, _sample_df = load_data(input_dir)  # noqa: F821
    tr_df, va_df = build_splits(
        train_df, val_size=float(hp["val_size"]), seed=seed
    )  # noqa: F821

    # vocabularies fit ONLY on the training slice (no leakage)
    word_vocab, keyword_vocab = fit_feature_vocabs(tr_df)  # noqa: F821

    train_ds = DisasterTweetDataset(tr_df, word_vocab, keyword_vocab)  # noqa: F821
    val_ds = DisasterTweetDataset(va_df, word_vocab, keyword_vocab)  # noqa: F821
    test_ds = DisasterTweetDataset(test_df, word_vocab, keyword_vocab)  # noqa: F821

    # ---- model ------------------------------------------------------------- #
    model = MultiViewTweetTransformer(  # noqa: F821
        word_vocab_size=max(len(word_vocab), 16),
        char_vocab_size=CHAR_VOCAB_SIZE,  # noqa: F821
        keyword_vocab_size=max(len(keyword_vocab), 8),
        **MODEL_DEFAULTS,  # noqa: F821
    )
    model = to_fp32_policy(model, precision_cfg)  # noqa: F821
    model = model.to(device)

    # ---- class-imbalance weight (from TRAIN slice only) -------------------- #
    y_tr = tr_df["target"].astype("float32").to_numpy()
    n_pos = float((y_tr >= 0.5).sum())
    n_neg = float((y_tr < 0.5).sum())
    pos_weight = (n_neg / max(n_pos, 1.0)) if n_pos > 0 else 1.0
    criterion = build_criterion(pos_weight=float(pos_weight))  # noqa: F821

    # ---- optimizer + scheduler -------------------------------------------- #
    physical_batch_size = int(hp["physical_batch_size"])
    accum_steps = max(int(hp["grad_accum_steps"]), 1)
    effective_batch_size = physical_batch_size * accum_steps

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hp["lr"]) * (1.0 if BATCH_LR_SCALING_POLICY == "fixed" else 1.0),
        weight_decay=float(hp["weight_decay"]),
        betas=tuple(hp["betas"]),
        eps=float(hp["eps"]),
    )

    train_loader = _make_loader(
        train_ds, physical_batch_size, True, int(hp["num_workers"]), device
    )
    val_loader = _make_loader(
        val_ds, physical_batch_size * 2, False, int(hp["num_workers"]), device
    )
    test_loader = _make_loader(
        test_ds, physical_batch_size * 2, False, int(hp["num_workers"]), device
    )

    steps_per_epoch = max(1, len(train_loader) // accum_steps)
    total_steps = steps_per_epoch * int(hp["epochs"])
    warmup_steps = int(float(hp["warmup_ratio"]) * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _build_cosine_warmup_lambda(total_steps, warmup_steps)
    )
    scaler = make_grad_scaler(precision_cfg)  # noqa: F821

    diagnostics_settings = {
        "physical_batch_size": physical_batch_size,
        "effective_batch_size": effective_batch_size,
        "planned_epochs": int(hp["epochs"]),
        "steps_per_epoch": steps_per_epoch,
        "model_family": MODEL_FAMILY,  # noqa: F821
        "split_seed": seed,
        "precision": precision_cfg.precision_policy,
        "precision_reason": precision_cfg.reason,
    }

    # ---- scheduler-driven resume ------------------------------------------ #
    state_holder: Dict[str, Any] = {"global_step": 0, "epoch": 0, "step_in_epoch": 0}
    resume_state: Optional[Dict[str, Any]] = None
    if context is not None:
        try:
            ckpt = context.load_resume_checkpoint()
        except Exception:
            ckpt = None
        if ckpt is not None and isinstance(ckpt, dict):
            resume_state = ckpt.get("state")

    best_val_f1: float = -1.0
    best_threshold: float = 0.5
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch: int = -1
    start_epoch: int = 1
    global_step: int = 0

    if resume_state is not None:
        try:
            if resume_state.get("model") is not None:
                model.load_state_dict(resume_state["model"])
            if resume_state.get("optimizer") is not None:
                optimizer.load_state_dict(resume_state["optimizer"])
            if scaler is not None and resume_state.get("scaler") is not None:
                scaler.load_state_dict(resume_state["scaler"])
            if resume_state.get("scheduler") is not None:
                scheduler.load_state_dict(resume_state["scheduler"])
            if resume_state.get("rng_torch") is not None:
                torch.set_rng_state(resume_state["rng_torch"])
            if resume_state.get("rng_numpy") is not None:
                np.random.set_state(resume_state["rng_numpy"])
            if resume_state.get("rng_python") is not None:
                random.setstate(resume_state["rng_python"])
            global_step = int(resume_state.get("global_step", 0))
            start_epoch = int(resume_state.get("epoch", 0)) + 1
            best_val_f1 = float(resume_state.get("best_val_f1", best_val_f1))
            best_threshold = float(resume_state.get("best_threshold", best_threshold))
            if resume_state.get("best_state") is not None:
                best_state = resume_state["best_state"]
            if resume_state.get("best_epoch") is not None:
                best_epoch = int(resume_state["best_epoch"])
            print(f"resumed training at epoch {start_epoch} step {global_step}")
        except Exception as exc:  # never crash on a partial checkpoint
            print(f"resume failed ({exc}); starting fresh")

    # ---- scheduler wiring -------------------------------------------------- #
    if context is not None:
        try:
            context.job.max_epochs = int(hp["epochs"])
            if hasattr(context.job, "config") and context.job.config is not None:
                context.job.config.max_epochs = int(hp["epochs"])
            if hasattr(context, "store") and context.store is not None:
                context.store.save_job(context.job)
        except Exception:
            pass

    timing_start = time.time()

    with TrainingDiagnostics(
        model, optimizer, scaler=scaler, settings=diagnostics_settings
    ) as diagnostics:

        def state_factory():
            return {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "scheduler": scheduler.state_dict(),
                "global_step": int(state_holder["global_step"]),
                "epoch": int(state_holder["epoch"]),
                "step_in_epoch": int(state_holder["step_in_epoch"]),
                "rng_torch": torch.get_rng_state(),
                "rng_numpy": np.random.get_state(),
                "rng_python": random.getstate(),
                "rng_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
                "best_val_f1": float(best_val_f1),
                "best_threshold": float(best_threshold),
                "best_epoch": int(best_epoch),
                "best_state": best_state,
                "diagnostics": diagnostics.state_dict(),
            }

        # Restore diagnostics state if present
        if resume_state is not None and resume_state.get("diagnostics") is not None:
            try:
                diagnostics.load_state_dict(resume_state["diagnostics"])
            except Exception:
                pass

        if context is not None:
            context.control_hook.safe_point(
                SafePointType.BEFORE_TRAIN,
                epoch=start_epoch - 1,
                global_step=global_step,
                steps_per_epoch=steps_per_epoch,
                state_factory=state_factory,
            )

        epochs_no_improve = 0

        for epoch in range(start_epoch, int(hp["epochs"]) + 1):
            state_holder["epoch"] = epoch
            model.train()
            running_loss = 0.0
            micro_count = 0
            optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(train_loader):
                w, c, k, y = batch
                w = w.to(device, non_blocking=True)
                c = c.to(device, non_blocking=True)
                k = k.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).to(dtype=torch.float32)

                with autocast_forward(precision_cfg):  # noqa: F821
                    logits = model(w, c, k)
                loss = compute_bce_with_logits_fp32(  # noqa: F821
                    logits, y, pos_weight=float(pos_weight)
                )
                scaled_loss = loss / float(accum_steps)

                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                running_loss += float(loss.detach().item())
                micro_count += 1

                is_last_micro = (batch_idx + 1) == len(train_loader)
                is_step_boundary = (batch_idx + 1) % accum_steps == 0

                if is_step_boundary or is_last_micro:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(hp["grad_clip"])
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(hp["grad_clip"])
                        )
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                    diagnostics.after_update()
                    global_step += 1
                    state_holder["global_step"] = global_step
                    state_holder["step_in_epoch"] = batch_idx + 1

                    if context is not None:
                        context.control_hook.safe_point(
                            SafePointType.STEP,
                            epoch=epoch,
                            global_step=global_step,
                            steps_per_epoch=steps_per_epoch,
                            state_factory=state_factory,
                        )

            mean_train_loss = running_loss / max(micro_count, 1)

            # ---- validation (identical forward path to test inference) ---- #
            val_probs = _infer_probs(model, val_loader, device, precision_cfg)
            val_targets = va_df["target"].astype("int64").to_numpy()
            best_t, val_f1 = tune_threshold(  # noqa: F821
                val_probs, val_targets, thresholds=hp["threshold_grid"]
            )

            print(
                f"epoch {epoch} train_loss={mean_train_loss:.4f} "
                f"val_f1={val_f1:.4f} thr={best_t:.3f} "
                f"elapsed={time.time() - timing_start:.1f}s"
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

            diagnostics.report(epoch=epoch)

            if val_f1 > best_val_f1:
                best_val_f1 = float(val_f1)
                best_threshold = float(best_t)
                best_epoch = int(epoch)
                best_state = {
                    key: val.detach().cpu().clone()
                    for key, val in model.state_dict().items()
                }
                epochs_no_improve = 0
                # persist best checkpoint (trusted local file)
                try:
                    torch.save(
                        {
                            "model": best_state,
                            "threshold": best_threshold,
                            "val_f1": best_val_f1,
                            "epoch": best_epoch,
                        },
                        os.path.join(working_dir, CHECKPOINT_NAME),
                    )
                except Exception:
                    pass
            else:
                epochs_no_improve += 1

            if context is not None:
                context.control_hook.safe_point(
                    SafePointType.EPOCH,
                    epoch=epoch,
                    global_step=global_step,
                    steps_per_epoch=steps_per_epoch,
                    state_factory=state_factory,
                )

            if epochs_no_improve >= int(hp["early_stopping_patience"]):
                print(f"early stopping at epoch {epoch} (no val-F1 improvement)")
                break

        # ---- restore best checkpoint before final evaluation -------------- #
        if best_state is not None:
            try:
                model.load_state_dict(best_state)
            except Exception as exc:
                print(f"best-state restore failed ({exc}); using current weights")
        model.eval()

        # Final validation with the SAME forward path used for test inference.
        final_val_probs = _infer_probs(model, val_loader, device, precision_cfg)
        final_val_targets = va_df["target"].astype("int64").to_numpy()
        final_threshold, final_val_f1 = tune_threshold(  # noqa: F821
            final_val_probs, final_val_targets, thresholds=hp["threshold_grid"]
        )

        # ---- test inference (identical pipeline as validation) ------------ #
        test_probs = _infer_probs(model, test_loader, device, precision_cfg)
        test_preds = (test_probs >= float(final_threshold)).astype(np.int64)

        test_ids = (
            test_df["id"].to_numpy()
            if "id" in test_df.columns
            else np.arange(len(test_df))
        )
        sub_df = pd.DataFrame({"id": test_ids, "target": test_preds})
        sub_path = os.path.join(submission_dir, "submission.csv")
        sub_df.to_csv(sub_path, index=False)

        peak_vram_mb = (
            float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
            if device.type == "cuda"
            else 0.0
        )
        print(
            f"runtime_log elapsed_seconds={time.time() - timing_start:.2f} "
            f"peak_vram_mb={peak_vram_mb:.1f} "
            f"resolved_batch_size={physical_batch_size} "
            f"effective_batch_size={effective_batch_size} "
            f"precision={precision_cfg.precision_policy} "
            f"best_epoch={best_epoch} threshold={final_threshold:.3f}"
        )
        print(
            f"submission written: {sub_path} rows={len(sub_df)} "
            f"pos_rate={float(test_preds.mean()):.4f}"
        )

    score = float(final_val_f1)
    print(f"Final Validation Score: {score}")
    return score


# --------------------------------------------------------------------------- #
# Main guard
# --------------------------------------------------------------------------- #
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
        