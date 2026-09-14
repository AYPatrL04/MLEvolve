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
 Summary: 2 total attempts - 2 executed, 2 successful (best: 0.7573)

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
Required hardware-aware step order: model-design -> datatype/quantization -> training (model_design -> datatype_precision -> training_evaluation). Use the trace as the source of truth, apply hardware evidence only when compatible, and use recorded fallbacks when evidence is missing.

# Pipeline Decision Trace
```json
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"{'criterion': 'BCEWithLogitsLoss', 'reduction': 'mean', 'fp32_sensitive': True, 'notes': 'compute...","output_interface":"match the competition metric and required submission format","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"none","fallback_policy":"fall back to fp32 or disabled AMP if the selected precision path is unsupported","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"{'name': 'AdamW', 'params': {'lr': 2e-05, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-...","scheduler":"{'name': 'linear_warmup_decay', 'warmup_ratio': 0.1, 'epochs': 3}","batch_size_policy":"fixed","dataloader_policy":"safe defaults; enable workers and pinned memory only when compatible","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available","HWDB recommended_patterns are conditional candidates, not proof of correctness, speed or package...","dataset_decomposition limitations: no hardware-specific acceleration and no direct inference bene..."]}}
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
 Summary: 2 total attempts - 2 executed, 2 successful (best: 0.7573)

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
Required hardware-aware step order: model-design -> datatype/quantization -> training (model_design -> datatype_precision -> training_evaluation). Use the trace as the source of truth, apply hardware evidence only when compatible, and use recorded fallbacks when evidence is missing.

# Pipeline Decision Trace
```json
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"{'criterion': 'BCEWithLogitsLoss', 'reduction': 'mean', 'fp32_sensitive': True, 'notes': 'compute...","output_interface":"match the competition metric and required submission format","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"none","fallback_policy":"fall back to fp32 or disabled AMP if the selected precision path is unsupported","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"{'name': 'AdamW', 'params': {'lr': 2e-05, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-...","scheduler":"{'name': 'linear_warmup_decay', 'warmup_ratio': 0.1, 'epochs': 3}","batch_size_policy":"fixed","dataloader_policy":"safe defaults; enable workers and pinned memory only when compatible","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available","HWDB recommended_patterns are conditional candidates, not proof of correctness, speed or package...","dataset_decomposition limitations: no hardware-specific acceleration and no direct inference bene..."]}}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=replaced single-view word TF-IDF MLP / transformer with a multi-view fusion net (word+char TF-IDF ⊕ token-embedding pooling ⊕ keyword embedding); purpose=capture both lexical and keyword-conditioned semantic signal for higher F1; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=added char_wb n-grams + explicit keyword vocabulary fitted only on train; purpose=robustness to tweet noise/typos and strong keyword priors; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=exposed FP32 logit interface + CandidateAdapter exercising the real model/loss/features; purpose=keep Stage-2 precision and Stage-3 training aligned to this candidate; hardware_keys=none

# Previous steps
### model_design
**Plan:** ## Plan

**What makes this task unique:** Disaster tweets are short, noisy (URLs, mentions, hashtags, typos), and critically rely on *keyword-conditioned* semantics — e.g. "ablaze"/"earthquake" almost always signal real disasters, while "damage"/"body bags" are frequently metaphorical. So the discriminative signal lives in *both* surface lexical patterns (where sparse TF-IDF dominates) and keyword-relative meaning.

**Why this approach:** The two prior attempts used a single-view word TF-IDF MLP (0.7573) and a from-scratch hybrid transformer (0.7341). Both fail to combine (a) sparse lexical generalization that won best score, with (b) learned compositional/semantic signal. I will build a **multi-view fusion network**: a word+char TF-IDF sparse branch (proven strong) fused with a trainable token-embedding mean-pool branch and a keyword-embedding branch. Char n-grams add robustness to tweet noise/typos; the keyword embedding explicitly lets the head condition text evidence on the keyword identity — a genuine architectural change, not a hyperparameter tweak.

**How I'll validate:** Internal stratified 20% hold-out (seed 42). Confirmation = fused model beats the single-view sparse-only baseline on hold-out F1; re-evaluation trigger = if the learned embedding branch adds no val-F1 over the TF-IDF-only branch, later stages should down-weight it (keep fusion architecture but rely on sparse head).

Memory strategy: fit vectorizers/tokenizer/keyword-vocab **only on the train split** and reuse for val/test. Batch elasticity: model consumes a fixed sparse dim; batch size stays a Stage-3 concern (exposed via `scenario['batch_size']`). Loss is FP32-sensitive `BCEWithLogitsLoss` (output logits, F1 threshold tuned later). Scheduler-owned controls (device, workers, launch) are **not** implemented here.

Note board:
- baseline_change: replaced single-view word TF-IDF MLP / transformer with a multi-view fusion net (word+char TF-IDF ⊕ token-embedding pooling ⊕ keyword embedding) | purpose: capture both lexical and keyword-conditioned semantic signal for higher F1 | hardware_keys: none
- baseline_change: added char_wb n-grams + explicit keyword vocabulary fitted only on train | purpose: robustness to tweet noise/typos and strong keyword priors | hardware_keys: none
- baseline_change: exposed FP32 logit interface + CandidateAdapter exercising the real model/loss/features | purpose: keep Stage-2 precision and Stage-3 training aligned to this candidate | hardware_keys: none
**Code:**
```python
"""
Stage 1 - model_design
Disaster Tweets (binary F1).

This module is import-safe: it only defines constants, data/feature helpers,
the torch model, the criterion factory, split helper, dataset class, and the
CPU-preflight CandidateAdapter. All training / inference side effects belong
to the final main guard added by later stages.

No optimizer/scheduler/precision/batch-size policy is finalized here; only
adapter-required defaults are provided so the CPU preflight can run.
"""

import os
import re
import html
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Contract constants
# --------------------------------------------------------------------------- #
MODEL_FAMILY = "multiview_tfidf_embed_mlp_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
INPUT_DIR = "./input"

SEED = 42
VAL_SIZE = 0.20

# Feature / model geometry (configurable)
MAX_WORD_FEATURES = 20000
MAX_CHAR_FEATURES = 20000
WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (2, 5)
SEQ_LEN = 48
EMB_DIM = 96
HIDDEN_DIM = 128
DROPOUT = 0.30
KEYWORD_EMB_DIM = 16

PAD_ID = 0
UNK_ID = 1
NO_KEYWORD = "__nokeyword__"

# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_NUM_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def clean_text(text: object) -> str:
    """Normalize a tweet: unescape HTML, lowercase, mask URLs/mentions/numbers."""
    if not isinstance(text, str):
        return ""
    t = html.unescape(text).lower()
    t = _URL_RE.sub(" urltoken ", t)
    t = _MENTION_RE.sub(" mentiontoken ", t)
    t = t.replace("#", " ")
    t = _NUM_RE.sub(" numtoken ", t)
    t = _NON_ALNUM_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def clean_keyword(value: object) -> str:
    if not isinstance(value, str) or value.strip() == "":
        return NO_KEYWORD
    return value.replace("%20", " ").strip().lower()


def clean_location(value: object) -> str:
    if not isinstance(value, str) or value.strip() == "":
        return ""
    t = html.unescape(value).lower()
    t = _NON_ALNUM_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def load_and_clean_data(input_dir: str = INPUT_DIR) -> pd.DataFrame:
    """Load train.csv and add cleaned text/keyword/location columns."""
    df = pd.read_csv(os.path.join(input_dir, "train.csv"))
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    df["clean_keyword"] = df.get("keyword", pd.Series([None] * len(df))).apply(
        clean_keyword
    )
    df["clean_location"] = df.get("location", pd.Series([None] * len(df))).apply(
        clean_location
    )
    # Model text: keyword prior + tweet semantics + location tokens.
    df["model_text"] = (
        df["clean_keyword"].replace(NO_KEYWORD, "")
        + " "
        + df["clean_text"]
        + " "
        + df["clean_location"]
    ).str.strip()
    if "target" in df.columns:
        df["target"] = df["target"].astype(np.float32)
    return df


def make_splits(
    df: pd.DataFrame, val_size: float = VAL_SIZE, seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/validation split. Fitted only on the train split later."""
    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=seed, stratify=df["target"].values
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Feature artifacts (fit on train split only)
# --------------------------------------------------------------------------- #
@dataclass
class FeatureArtifacts:
    word_vec: TfidfVectorizer
    char_vec: TfidfVectorizer
    token_to_id: Dict[str, int]
    keyword_to_id: Dict[str, int]
    n_keywords: int
    sparse_dim: int
    vocab_size: int
    seq_len: int = SEQ_LEN


_TOKEN_SPLIT_RE = re.compile(r"\s+")


def tokenize(text: str) -> List[str]:
    return [tok for tok in _TOKEN_SPLIT_RE.split(text) if tok]


def fit_feature_artifacts(train_df: pd.DataFrame) -> FeatureArtifacts:
    """Fit word/char TF-IDF, token vocab and keyword vocab on TRAIN ONLY."""
    texts = train_df["model_text"].fillna("").tolist()

    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=WORD_NGRAM_RANGE,
        min_df=2,
        max_features=MAX_WORD_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=CHAR_NGRAM_RANGE,
        min_df=3,
        max_features=MAX_CHAR_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    word_vec.fit(texts)
    char_vec.fit(texts)

    # token vocabulary
    token_to_id: Dict[str, int] = {}
    for text in texts:
        for tok in tokenize(text):
            if tok not in token_to_id:
                token_to_id[tok] = len(token_to_id) + 2  # 0=pad, 1=unk
    vocab_size = max(len(token_to_id) + 2, 2)

    # keyword vocabulary
    keywords = train_df["clean_keyword"].fillna(NO_KEYWORD).tolist()
    keyword_to_id: Dict[str, int] = {}
    for kw in keywords:
        if kw not in keyword_to_id:
            keyword_to_id[kw] = len(keyword_to_id)
    if NO_KEYWORD not in keyword_to_id:
        keyword_to_id[NO_KEYWORD] = len(keyword_to_id)
    n_keywords = max(len(keyword_to_id), 1)

    sparse_dim = len(word_vec.get_feature_names_out()) + len(
        char_vec.get_feature_names_out()
    )

    return FeatureArtifacts(
        word_vec=word_vec,
        char_vec=char_vec,
        token_to_id=token_to_id,
        keyword_to_id=keyword_to_id,
        n_keywords=n_keywords,
        sparse_dim=int(sparse_dim),
        vocab_size=int(vocab_size),
        seq_len=SEQ_LEN,
    )


def transform_sparse(df: pd.DataFrame, art: FeatureArtifacts) -> sparse.csr_matrix:
    texts = df["model_text"].fillna("").tolist()
    w = art.word_vec.transform(texts)
    c = art.char_vec.transform(texts)
    return sparse.hstack([w, c]).tocsr().astype(np.float32)


def encode_tokens(
    df: pd.DataFrame, art: FeatureArtifacts, seq_len: Optional[int] = None
) -> torch.Tensor:
    seq_len = seq_len or art.seq_len
    out = np.zeros((len(df), seq_len), dtype=np.int64)
    for i, text in enumerate(df["model_text"].fillna("").tolist()):
        ids = [art.token_to_id.get(tok, UNK_ID) for tok in tokenize(text)][:seq_len]
        if ids:
            out[i, : len(ids)] = ids
    return torch.from_numpy(out)


def encode_keywords(df: pd.DataFrame, art: FeatureArtifacts) -> torch.Tensor:
    ids = [
        art.keyword_to_id.get(kw, art.keyword_to_id[NO_KEYWORD])
        for kw in df["clean_keyword"].fillna(NO_KEYWORD).tolist()
    ]
    return torch.tensor(ids, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Model: multi-view fusion
# --------------------------------------------------------------------------- #
class MultiViewDisasterNet(nn.Module):
    """
    Fuses three views of a tweet:
      * sparse word+char TF-IDF vector  -> linear branch (lexical generalization)
      * learned token-embedding mean pool -> compositional/semantic branch
      * keyword embedding -> lets the head condition text on the keyword prior
    Outputs raw logits (batch,) for BCEWithLogitsLoss.
    """

    def __init__(
        self,
        sparse_dim: int,
        vocab_size: int,
        n_keywords: int,
        emb_dim: int = EMB_DIM,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
        keyword_emb_dim: int = KEYWORD_EMB_DIM,
        pad_id: int = PAD_ID,
    ):
        super().__init__()
        self.pad_id = pad_id

        self.sparse_branch = nn.Sequential(
            nn.Linear(sparse_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.token_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.token_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.keyword_embedding = nn.Embedding(n_keywords, keyword_emb_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + keyword_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        sparse_features: torch.Tensor,
        token_ids: torch.Tensor,
        keyword_ids: torch.Tensor,
    ) -> torch.Tensor:
        sparse_features = sparse_features.float()
        token_ids = token_ids.long()
        keyword_ids = keyword_ids.long()

        a = self.sparse_branch(sparse_features)

        emb = self.token_embedding(token_ids)  # (B, L, E)
        mask = (token_ids != self.pad_id).unsqueeze(-1).float()  # (B, L, 1)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        b = self.token_proj(pooled)

        c = self.keyword_embedding(keyword_ids).reshape(keyword_ids.size(0), -1)

        x = torch.cat([a, b, c], dim=1)
        return self.head(x).squeeze(-1)


def build_loss() -> nn.Module:
    """FP32-sensitive criterion used by the training/eval stages."""
    return nn.BCEWithLogitsLoss(reduction="mean")


def build_model_from_artifacts(art: FeatureArtifacts) -> MultiViewDisasterNet:
    return MultiViewDisasterNet(
        sparse_dim=art.sparse_dim,
        vocab_size=art.vocab_size,
        n_keywords=art.n_keywords,
    )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class TextDisasterDataset(torch.utils.data.Dataset):
    """
    Yields (inputs, target) where inputs = (sparse_row, token_ids, keyword_ids).
    Sparse rows are stored as a compact float32 array to allow fast indexing.
    """

    def __init__(
        self,
        sparse_matrix: sparse.csr_matrix,
        token_ids: torch.Tensor,
        keyword_ids: torch.Tensor,
        targets: Optional[np.ndarray] = None,
        dense_sparse: bool = True,
    ):
        self.sparse_matrix = (
            np.ascontiguousarray(sparse_matrix.toarray(), dtype=np.float32)
            if dense_sparse
            else sparse_matrix
        )
        self.token_ids = token_ids.long()
        self.keyword_ids = keyword_ids.long()
        self.targets = (
            np.asarray(targets, dtype=np.float32) if targets is not None else None
        )

    def __len__(self) -> int:
        return self.token_ids.size(0)

    def __getitem__(self, idx: int):
        if isinstance(self.sparse_matrix, np.ndarray):
            sp = torch.from_numpy(self.sparse_matrix[idx])
        else:
            sp = torch.from_numpy(self.sparse_matrix[idx].toarray().ravel())
        item = {
            "inputs": (sp, self.token_ids[idx], self.keyword_ids[idx]),
        }
        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return item


# --------------------------------------------------------------------------- #
# CPU-preflight CandidateAdapter
# --------------------------------------------------------------------------- #
_FIXTURE_NAMES = ("sparse_features", "token_ids", "keyword_ids")


class CandidateAdapter:
    """
    No-argument adapter exercising the real model, features, loss and optimizer.
    Artifacts are fit lazily from the input fixtures and cached per instance.
    """

    def __init__(self):
        self._cache: Dict[str, object] = {}

    # -- helpers ---------------------------------------------------------- #
    def _input_dir(self) -> str:
        return os.environ.get("MLEVOLVE_INPUT_DIR", INPUT_DIR)

    def _ensure_data(self) -> Tuple[pd.DataFrame, FeatureArtifacts]:
        if "df" not in self._cache:
            df = load_and_clean_data(self._input_dir())
            art = fit_feature_artifacts(df)
            self._cache["df"] = df
            self._cache["art"] = art
        return self._cache["df"], self._cache["art"]  # type: ignore

    # -- required API ----------------------------------------------------- #
    def build_model(self, context) -> nn.Module:
        _df, art = self._ensure_data()
        return build_model_from_artifacts(art)

    def build_optimizer(self, model, context):
        context = dict(context or {})
        lr = float(context.get("lr", 1e-3))
        weight_decay = float(context.get("weight_decay", 0.01))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def build_train_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def _build_batch(self, scenario, device):
        scenario = dict(scenario or {})
        batch_size = int(scenario.get("batch_size", 4))
        fixture = scenario.get("fixture") or {}

        # Authoritative fixture path: build declared tensors, keep names.
        if fixture and all(name in fixture for name in _FIXTURE_NAMES):
            built: Dict[str, torch.Tensor] = {}
            for name in _FIXTURE_NAMES:
                spec = fixture[name]
                shape = spec.get("shape") if isinstance(spec, dict) else spec
                if name in ("token_ids", "keyword_ids") or "int" in str(
                    (spec or {}).get("dtype", "") if isinstance(spec, dict) else ""
                ):
                    built[name] = torch.zeros(
                        tuple(shape), dtype=torch.long, device=device
                    )
                else:
                    built[name] = torch.zeros(
                        tuple(shape), dtype=torch.float32, device=device
                    )
            target = torch.zeros(
                built["token_ids"].size(0), dtype=torch.float32, device=device
            )
            out = dict(built)
            out["inputs"] = (
                built["sparse_features"],
                built["token_ids"],
                built["keyword_ids"],
            )
            out["target"] = target
            return out

        # Real-data path.
        df, art = self._ensure_data()
        sub = df.head(batch_size).reset_index(drop=True)

        sp = transform_sparse(sub, art).toarray().astype(np.float32)
        sparse_t = torch.from_numpy(sp).to(device)
        tokens = encode_tokens(sub, art).to(device)
        keywords = encode_keywords(sub, art).to(device)
        target = torch.tensor(
            sub["target"].values.astype(np.float32), dtype=torch.float32, device=device
        )

        return {
            "inputs": (sparse_t, tokens, keywords),
            "target": target,
            "sparse_features": sparse_t,
            "token_ids": tokens,
            "keyword_ids": keywords,
        }

    def _resolve_criterion(self, context) -> nn.Module:
        context = dict(context or {})
        crit = context.get("criterion")
        if crit is None:
            crit = build_loss()
        return crit

    def training_step(self, model, batch, context):
        inputs = batch["inputs"]
        target = batch["target"].float()
        logits = model(*inputs)
        criterion = self._resolve_criterion(context)
        return criterion(logits, target)

    def validation_step(self, model, batch, context):
        inputs = batch["inputs"]
        target = batch["target"].float()
        logits = model(*inputs)
        criterion = self._resolve_criterion(context)
        loss = criterion(logits, target)
        probs = torch.sigmoid(logits.detach())
        return {
            "loss": loss,
            "logits": logits.detach(),
            "probs": probs,
            "target": target,
        }

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
 Summary: 2 total attempts - 2 executed, 2 successful (best: 0.7573)

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
Required hardware-aware step order: model-design -> datatype/quantization -> training (model_design -> datatype_precision -> training_evaluation). Use the trace as the source of truth, apply hardware evidence only when compatible, and use recorded fallbacks when evidence is missing.

# Pipeline Decision Trace
```json
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"{'criterion': 'BCEWithLogitsLoss', 'reduction': 'mean', 'fp32_sensitive': True, 'notes': 'compute...","output_interface":"match the competition metric and required submission format","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"none","fallback_policy":"fall back to fp32 or disabled AMP if the selected precision path is unsupported","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"{'name': 'AdamW', 'params': {'lr': 2e-05, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-...","scheduler":"{'name': 'linear_warmup_decay', 'warmup_ratio': 0.1, 'epochs': 3}","batch_size_policy":"fixed","dataloader_policy":"safe defaults; enable workers and pinned memory only when compatible","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available","HWDB recommended_patterns are conditional candidates, not proof of correctness, speed or package...","dataset_decomposition limitations: no hardware-specific acceleration and no direct inference bene..."]}}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=replaced single-view word TF-IDF MLP / transformer with a multi-view fusion net (word+char TF-IDF ⊕ token-embedding pooling ⊕ keyword embedding); purpose=capture both lexical and keyword-conditioned semantic signal for higher F1; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=added char_wb n-grams + explicit keyword vocabulary fitted only on train; purpose=robustness to tweet noise/typos and strong keyword priors; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=exposed FP32 logit interface + CandidateAdapter exercising the real model/loss/features; purpose=keep Stage-2 precision and Stage-3 training aligned to this candidate; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=## Plan **What's unique here**: The task allows only `fp32`/`disabled`/`fp16_amp` on an Ampere A10 (sm_86).; purpose=Support the Stage 1 candidate target with a compatible precision policy.; hardware_keys=fp16_amp, auto_degrade_on_nonfinite, sparse_dim, vocab_size, n_keywords, emb_dim, hidden_dim, seq_len

# Previous steps
### model_design
**Plan:** ## Plan

**What makes this task unique:** Disaster tweets are short, noisy (URLs, mentions, hashtags, typos), and critically rely on *keyword-conditioned* semantics — e.g. "ablaze"/"earthquake" almost always signal real disasters, while "damage"/"body bags" are frequently metaphorical. So the discriminative signal lives in *both* surface lexical patterns (where sparse TF-IDF dominates) and keyword-relative meaning.

**Why this approach:** The two prior attempts used a single-view word TF-IDF MLP (0.7573) and a from-scratch hybrid transformer (0.7341). Both fail to combine (a) sparse lexical generalization that won best score, with (b) learned compositional/semantic signal. I will build a **multi-view fusion network**: a word+char TF-IDF sparse branch (proven strong) fused with a trainable token-embedding mean-pool branch and a keyword-embedding branch. Char n-grams add robustness to tweet noise/typos; the keyword embedding explicitly lets the head condition text evidence on the keyword identity — a genuine architectural change, not a hyperparameter tweak.

**How I'll validate:** Internal stratified 20% hold-out (seed 42). Confirmation = fused model beats the single-view sparse-only baseline on hold-out F1; re-evaluation trigger = if the learned embedding branch adds no val-F1 over the TF-IDF-only branch, later stages should down-weight it (keep fusion architecture but rely on sparse head).

Memory strategy: fit vectorizers/tokenizer/keyword-vocab **only on the train split** and reuse for val/test. Batch elasticity: model consumes a fixed sparse dim; batch size stays a Stage-3 concern (exposed via `scenario['batch_size']`). Loss is FP32-sensitive `BCEWithLogitsLoss` (output logits, F1 threshold tuned later). Scheduler-owned controls (device, workers, launch) are **not** implemented here.

Note board:
- baseline_change: replaced single-view word TF-IDF MLP / transformer with a multi-view fusion net (word+char TF-IDF ⊕ token-embedding pooling ⊕ keyword embedding) | purpose: capture both lexical and keyword-conditioned semantic signal for higher F1 | hardware_keys: none
- baseline_change: added char_wb n-grams + explicit keyword vocabulary fitted only on train | purpose: robustness to tweet noise/typos and strong keyword priors | hardware_keys: none
- baseline_change: exposed FP32 logit interface + CandidateAdapter exercising the real model/loss/features | purpose: keep Stage-2 precision and Stage-3 training aligned to this candidate | hardware_keys: none
**Code:**
```python
"""
Stage 1 - model_design
Disaster Tweets (binary F1).

This module is import-safe: it only defines constants, data/feature helpers,
the torch model, the criterion factory, split helper, dataset class, and the
CPU-preflight CandidateAdapter. All training / inference side effects belong
to the final main guard added by later stages.

No optimizer/scheduler/precision/batch-size policy is finalized here; only
adapter-required defaults are provided so the CPU preflight can run.
"""

import os
import re
import html
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Contract constants
# --------------------------------------------------------------------------- #
MODEL_FAMILY = "multiview_tfidf_embed_mlp_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
INPUT_DIR = "./input"

SEED = 42
VAL_SIZE = 0.20

# Feature / model geometry (configurable)
MAX_WORD_FEATURES = 20000
MAX_CHAR_FEATURES = 20000
WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (2, 5)
SEQ_LEN = 48
EMB_DIM = 96
HIDDEN_DIM = 128
DROPOUT = 0.30
KEYWORD_EMB_DIM = 16

PAD_ID = 0
UNK_ID = 1
NO_KEYWORD = "__nokeyword__"

# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_NUM_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def clean_text(text: object) -> str:
    """Normalize a tweet: unescape HTML, lowercase, mask URLs/mentions/numbers."""
    if not isinstance(text, str):
        return ""
    t = html.unescape(text).lower()
    t = _URL_RE.sub(" urltoken ", t)
    t = _MENTION_RE.sub(" mentiontoken ", t)
    t = t.replace("#", " ")
    t = _NUM_RE.sub(" numtoken ", t)
    t = _NON_ALNUM_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def clean_keyword(value: object) -> str:
    if not isinstance(value, str) or value.strip() == "":
        return NO_KEYWORD
    return value.replace("%20", " ").strip().lower()


def clean_location(value: object) -> str:
    if not isinstance(value, str) or value.strip() == "":
        return ""
    t = html.unescape(value).lower()
    t = _NON_ALNUM_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def load_and_clean_data(input_dir: str = INPUT_DIR) -> pd.DataFrame:
    """Load train.csv and add cleaned text/keyword/location columns."""
    df = pd.read_csv(os.path.join(input_dir, "train.csv"))
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    df["clean_keyword"] = df.get("keyword", pd.Series([None] * len(df))).apply(
        clean_keyword
    )
    df["clean_location"] = df.get("location", pd.Series([None] * len(df))).apply(
        clean_location
    )
    # Model text: keyword prior + tweet semantics + location tokens.
    df["model_text"] = (
        df["clean_keyword"].replace(NO_KEYWORD, "")
        + " "
        + df["clean_text"]
        + " "
        + df["clean_location"]
    ).str.strip()
    if "target" in df.columns:
        df["target"] = df["target"].astype(np.float32)
    return df


def make_splits(
    df: pd.DataFrame, val_size: float = VAL_SIZE, seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/validation split. Fitted only on the train split later."""
    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=seed, stratify=df["target"].values
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Feature artifacts (fit on train split only)
# --------------------------------------------------------------------------- #
@dataclass
class FeatureArtifacts:
    word_vec: TfidfVectorizer
    char_vec: TfidfVectorizer
    token_to_id: Dict[str, int]
    keyword_to_id: Dict[str, int]
    n_keywords: int
    sparse_dim: int
    vocab_size: int
    seq_len: int = SEQ_LEN


_TOKEN_SPLIT_RE = re.compile(r"\s+")


def tokenize(text: str) -> List[str]:
    return [tok for tok in _TOKEN_SPLIT_RE.split(text) if tok]


def fit_feature_artifacts(train_df: pd.DataFrame) -> FeatureArtifacts:
    """Fit word/char TF-IDF, token vocab and keyword vocab on TRAIN ONLY."""
    texts = train_df["model_text"].fillna("").tolist()

    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=WORD_NGRAM_RANGE,
        min_df=2,
        max_features=MAX_WORD_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=CHAR_NGRAM_RANGE,
        min_df=3,
        max_features=MAX_CHAR_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    word_vec.fit(texts)
    char_vec.fit(texts)

    # token vocabulary
    token_to_id: Dict[str, int] = {}
    for text in texts:
        for tok in tokenize(text):
            if tok not in token_to_id:
                token_to_id[tok] = len(token_to_id) + 2  # 0=pad, 1=unk
    vocab_size = max(len(token_to_id) + 2, 2)

    # keyword vocabulary
    keywords = train_df["clean_keyword"].fillna(NO_KEYWORD).tolist()
    keyword_to_id: Dict[str, int] = {}
    for kw in keywords:
        if kw not in keyword_to_id:
            keyword_to_id[kw] = len(keyword_to_id)
    if NO_KEYWORD not in keyword_to_id:
        keyword_to_id[NO_KEYWORD] = len(keyword_to_id)
    n_keywords = max(len(keyword_to_id), 1)

    sparse_dim = len(word_vec.get_feature_names_out()) + len(
        char_vec.get_feature_names_out()
    )

    return FeatureArtifacts(
        word_vec=word_vec,
        char_vec=char_vec,
        token_to_id=token_to_id,
        keyword_to_id=keyword_to_id,
        n_keywords=n_keywords,
        sparse_dim=int(sparse_dim),
        vocab_size=int(vocab_size),
        seq_len=SEQ_LEN,
    )


def transform_sparse(df: pd.DataFrame, art: FeatureArtifacts) -> sparse.csr_matrix:
    texts = df["model_text"].fillna("").tolist()
    w = art.word_vec.transform(texts)
    c = art.char_vec.transform(texts)
    return sparse.hstack([w, c]).tocsr().astype(np.float32)


def encode_tokens(
    df: pd.DataFrame, art: FeatureArtifacts, seq_len: Optional[int] = None
) -> torch.Tensor:
    seq_len = seq_len or art.seq_len
    out = np.zeros((len(df), seq_len), dtype=np.int64)
    for i, text in enumerate(df["model_text"].fillna("").tolist()):
        ids = [art.token_to_id.get(tok, UNK_ID) for tok in tokenize(text)][:seq_len]
        if ids:
            out[i, : len(ids)] = ids
    return torch.from_numpy(out)


def encode_keywords(df: pd.DataFrame, art: FeatureArtifacts) -> torch.Tensor:
    ids = [
        art.keyword_to_id.get(kw, art.keyword_to_id[NO_KEYWORD])
        for kw in df["clean_keyword"].fillna(NO_KEYWORD).tolist()
    ]
    return torch.tensor(ids, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Model: multi-view fusion
# --------------------------------------------------------------------------- #
class MultiViewDisasterNet(nn.Module):
    """
    Fuses three views of a tweet:
      * sparse word+char TF-IDF vector  -> linear branch (lexical generalization)
      * learned token-embedding mean pool -> compositional/semantic branch
      * keyword embedding -> lets the head condition text on the keyword prior
    Outputs raw logits (batch,) for BCEWithLogitsLoss.
    """

    def __init__(
        self,
        sparse_dim: int,
        vocab_size: int,
        n_keywords: int,
        emb_dim: int = EMB_DIM,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
        keyword_emb_dim: int = KEYWORD_EMB_DIM,
        pad_id: int = PAD_ID,
    ):
        super().__init__()
        self.pad_id = pad_id

        self.sparse_branch = nn.Sequential(
            nn.Linear(sparse_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.token_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.token_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.keyword_embedding = nn.Embedding(n_keywords, keyword_emb_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + keyword_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        sparse_features: torch.Tensor,
        token_ids: torch.Tensor,
        keyword_ids: torch.Tensor,
    ) -> torch.Tensor:
        sparse_features = sparse_features.float()
        token_ids = token_ids.long()
        keyword_ids = keyword_ids.long()

        a = self.sparse_branch(sparse_features)

        emb = self.token_embedding(token_ids)  # (B, L, E)
        mask = (token_ids != self.pad_id).unsqueeze(-1).float()  # (B, L, 1)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        b = self.token_proj(pooled)

        c = self.keyword_embedding(keyword_ids).reshape(keyword_ids.size(0), -1)

        x = torch.cat([a, b, c], dim=1)
        return self.head(x).squeeze(-1)


def build_loss() -> nn.Module:
    """FP32-sensitive criterion used by the training/eval stages."""
    return nn.BCEWithLogitsLoss(reduction="mean")


def build_model_from_artifacts(art: FeatureArtifacts) -> MultiViewDisasterNet:
    return MultiViewDisasterNet(
        sparse_dim=art.sparse_dim,
        vocab_size=art.vocab_size,
        n_keywords=art.n_keywords,
    )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class TextDisasterDataset(torch.utils.data.Dataset):
    """
    Yields (inputs, target) where inputs = (sparse_row, token_ids, keyword_ids).
    Sparse rows are stored as a compact float32 array to allow fast indexing.
    """

    def __init__(
        self,
        sparse_matrix: sparse.csr_matrix,
        token_ids: torch.Tensor,
        keyword_ids: torch.Tensor,
        targets: Optional[np.ndarray] = None,
        dense_sparse: bool = True,
    ):
        self.sparse_matrix = (
            np.ascontiguousarray(sparse_matrix.toarray(), dtype=np.float32)
            if dense_sparse
            else sparse_matrix
        )
        self.token_ids = token_ids.long()
        self.keyword_ids = keyword_ids.long()
        self.targets = (
            np.asarray(targets, dtype=np.float32) if targets is not None else None
        )

    def __len__(self) -> int:
        return self.token_ids.size(0)

    def __getitem__(self, idx: int):
        if isinstance(self.sparse_matrix, np.ndarray):
            sp = torch.from_numpy(self.sparse_matrix[idx])
        else:
            sp = torch.from_numpy(self.sparse_matrix[idx].toarray().ravel())
        item = {
            "inputs": (sp, self.token_ids[idx], self.keyword_ids[idx]),
        }
        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return item


# --------------------------------------------------------------------------- #
# CPU-preflight CandidateAdapter
# --------------------------------------------------------------------------- #
_FIXTURE_NAMES = ("sparse_features", "token_ids", "keyword_ids")


class CandidateAdapter:
    """
    No-argument adapter exercising the real model, features, loss and optimizer.
    Artifacts are fit lazily from the input fixtures and cached per instance.
    """

    def __init__(self):
        self._cache: Dict[str, object] = {}

    # -- helpers ---------------------------------------------------------- #
    def _input_dir(self) -> str:
        return os.environ.get("MLEVOLVE_INPUT_DIR", INPUT_DIR)

    def _ensure_data(self) -> Tuple[pd.DataFrame, FeatureArtifacts]:
        if "df" not in self._cache:
            df = load_and_clean_data(self._input_dir())
            art = fit_feature_artifacts(df)
            self._cache["df"] = df
            self._cache["art"] = art
        return self._cache["df"], self._cache["art"]  # type: ignore

    # -- required API ----------------------------------------------------- #
    def build_model(self, context) -> nn.Module:
        _df, art = self._ensure_data()
        return build_model_from_artifacts(art)

    def build_optimizer(self, model, context):
        context = dict(context or {})
        lr = float(context.get("lr", 1e-3))
        weight_decay = float(context.get("weight_decay", 0.01))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def build_train_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def _build_batch(self, scenario, device):
        scenario = dict(scenario or {})
        batch_size = int(scenario.get("batch_size", 4))
        fixture = scenario.get("fixture") or {}

        # Authoritative fixture path: build declared tensors, keep names.
        if fixture and all(name in fixture for name in _FIXTURE_NAMES):
            built: Dict[str, torch.Tensor] = {}
            for name in _FIXTURE_NAMES:
                spec = fixture[name]
                shape = spec.get("shape") if isinstance(spec, dict) else spec
                if name in ("token_ids", "keyword_ids") or "int" in str(
                    (spec or {}).get("dtype", "") if isinstance(spec, dict) else ""
                ):
                    built[name] = torch.zeros(
                        tuple(shape), dtype=torch.long, device=device
                    )
                else:
                    built[name] = torch.zeros(
                        tuple(shape), dtype=torch.float32, device=device
                    )
            target = torch.zeros(
                built["token_ids"].size(0), dtype=torch.float32, device=device
            )
            out = dict(built)
            out["inputs"] = (
                built["sparse_features"],
                built["token_ids"],
                built["keyword_ids"],
            )
            out["target"] = target
            return out

        # Real-data path.
        df, art = self._ensure_data()
        sub = df.head(batch_size).reset_index(drop=True)

        sp = transform_sparse(sub, art).toarray().astype(np.float32)
        sparse_t = torch.from_numpy(sp).to(device)
        tokens = encode_tokens(sub, art).to(device)
        keywords = encode_keywords(sub, art).to(device)
        target = torch.tensor(
            sub["target"].values.astype(np.float32), dtype=torch.float32, device=device
        )

        return {
            "inputs": (sparse_t, tokens, keywords),
            "target": target,
            "sparse_features": sparse_t,
            "token_ids": tokens,
            "keyword_ids": keywords,
        }

    def _resolve_criterion(self, context) -> nn.Module:
        context = dict(context or {})
        crit = context.get("criterion")
        if crit is None:
            crit = build_loss()
        return crit

    def training_step(self, model, batch, context):
        inputs = batch["inputs"]
        target = batch["target"].float()
        logits = model(*inputs)
        criterion = self._resolve_criterion(context)
        return criterion(logits, target)

    def validation_step(self, model, batch, context):
        inputs = batch["inputs"]
        target = batch["target"].float()
        logits = model(*inputs)
        criterion = self._resolve_criterion(context)
        loss = criterion(logits, target)
        probs = torch.sigmoid(logits.detach())
        return {
            "loss": loss,
            "logits": logits.detach(),
            "probs": probs,
            "target": target,
        }

```

### datatype_precision
**Plan:** ## Plan

**What's unique here**: The task allows only `fp32`/`disabled`/`fp16_amp` on an Ampere A10 (sm_86). The Stage-1 model is a small multi-view MLP with a sparse TF-IDF linear branch, an embedding-pool branch, and an FP32-sensitive `BCEWithLogitsLoss`. Since Stage 1 chose `precision_model_adaptation: none`, I cannot swap layers — I can only wrap eligible forward ops in autocast, keep the loss and metric/export in FP32, and provide a graceful CPU/unsupported fallback.

**Why this design**: FP16 AMP shrinks activation memory and speeds matmul on Ampere Tensor Cores only if the ops are eligible; my sparse branch is a simple `Linear` (eligible), the embedding pooling is cheap, and the head is `Linear` — so the only real gain is in `Linear` GEMMs, which AMP covers. I disable TF32 (not allowed in normal mode) and disable AMP on CPU/unsupported GPUs. `BCEWithLogitsLoss` is FP32-sensitive per the decision trace, so I run it in `autocast(enabled=False)` with explicit FP32 casts of logits/targets.

**How I'll validate**: The training step will log the resolved precision plan (device, use_amp, dtype, use_tf32, scaled) once and the diagnostics helper will report the *actual* autocast dtype per epoch; if val-F1 does not beat the FP32 baseline within tolerance, the fallback flag degrades to `disabled` AMP and we re-run FP32. Confirmation = finite loss under FP16 AMP for the full run + non-regressed val-F1; disconfirmation = NaN/inf loss or a drop in val-F1 → degrade to FP32.

**Memory strategy & batch elasticity**: No change to Stage-3 batch size. AMP only changes dtype of activations (halves memory pressure), leaving batch size/accumulation to Stage 3. Fallback: on OOM or non-finite loss, degrade to FP32 AMP (`auto_degrade_on_nonfinite`) without touching the optimizer, LR, epochs, or dataloaders.

**Operator & normalization**: Autocast wraps model forward only (matmul → fp16). Loss is a FP32 island. Backward and `scaler.step`/`scaler.update` run outside autocast; gradient unscale via `scaler.unscale_(optimizer)` before clipping. Normalization layers (none in Stage-1 model) would be kept in FP32 automatically by autocast; embeddings stay FP32 (autocast does not cast `nn.Embedding` outputs). BatchNorm/softmax would be opaque but we have none — this is a safe workload for fp16 AMP.

**Configurable model dims**: unchanged from Stage 1 (`sparse_dim`, `vocab_size`, `n_keywords`, `emb_dim`, `hidden_dim`, `seq_len`) — precision stage does not touch them.

**Scheduler-owned controls (NOT implemented here)**: device assignment, thread/worker allocation, exclusive-vs-shared VRAM packing, subprocess launch, and NCCL/collective setup all belong to the scheduler per the backend contract (`cuda_process`, `subprocess_job_v1`). This module only resolves *tensor* dtype/precision inside a single process.

**Note board:**
- baseline_change: added a precision-plan utility layer (`resolve_precision_config`, `sentence_forward_no_grad`, FP32 loss island, `make_grad_scaler`, `configure_tf32`, `degrade_to_fp32`) with no model surgery | purpose: consume the pipeline `datatype_precision.precision_policy=fp16_amp` on sm_86 while preserving Stage-1 model family, loss, features and output interface | hardware_keys: compute_capability=8.6, vram_mib=22589, precision_mode=normal, allowed_policies=fp32/disabled/fp16_amp
- baseline_change: explicit TF32 disable helper invoked only from the main guard | purpose: normal-mode rule forbids TF32, but capability alone does not authorize it; disabling keeps numerics identical to the FP32 reference | hardware_keys: tf32_policy=disable
- baseline_change: fallback path (`auto_degrade_on_nonfinite`, `degrade_to_fp32`) tracks a runtime state and forces AMP off on CPU/unsupported/NaN | purpose: keeps run robust without changing loss/model/optimizer; Stage-3 can act on the flag when non-finite loss is detected | hardware_keys: fallback_policy=fall-back-to-fp32-or-disabled
**Code:**
```python
"""
Stage 2 - datatype_precision
Disaster Tweets binary F1.

Precision policy consumed from the pipeline trace:
  datatype_precision.precision_policy = "fp16_amp"
  datatype_precision.precision_model_adaptation = "none"
  datatype_precision.fallback_policy = "fall back to fp32 or disabled AMP if the
                                        selected precision path is unsupported"

Allowed precision policies in normal mode: fp32, disabled, fp16_amp.
BF16, TF32, quantization, and Transformer-Engine FP8/MXFP8/NVFP4 recipes are OUT.

This module is import-safe: only constants, dataclass, and helpers are defined.
It never touches CUDA/TF32, never instantiates models/optimizers/losses, never
reads data, and never mutates global torch flags at import time. All activation
of these helpers happens inside `if __name__ == '__main__':` in the final
training entrypoint.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Precision contract constants (import-safe; no side effects)
# --------------------------------------------------------------------------- #
# Pipeline-decision default. Env can opt out without touching code.
DEFAULT_PRECISION_POLICY: str = "fp16_amp"  # "fp16_amp" | "fp32" | "disabled"
SUPPORTED_PRECISION_POLICIES: Tuple[str, ...] = ("fp16_amp", "fp32", "disabled")
AMP_DTYPE: torch.dtype = torch.float16  # fp16 only in normal mode
FP32_DTYPE: torch.dtype = torch.float32

# Ampere sm_86 supports fp16 tensor cores; we still gate on cuda.is_available().
MIN_AMP_COMPUTE_CAPABILITY: Tuple[int, int] = (7, 0)  # Volta+ has fp16 tensor cores

# Precision-required model adaptation per pipeline trace.
PRECISION_MODEL_ADAPTATION: str = "none"

# Small epsilon used for numerical comparisons only (not a policy knob).
_NONFINITE_EPS: float = float("nan")


# --------------------------------------------------------------------------- #
# Resolved precision configuration
# --------------------------------------------------------------------------- #
@dataclass
class PrecisionConfig:
    """
    Immutable-ish snapshot of the resolved precision plan for one process.

    Attributes
    ----------
    device : torch.device
        Target device selected for training/eval. Assigned by the entrypoint
        (scheduler is authoritative for GPU allocation); we only record it.
    use_amp : bool
        Whether CUDA autocast (fp16) is enabled for eligible forward ops.
    amp_dtype : torch.dtype
        Autocast dtype when ``use_amp`` is True. Always fp16 here.
    use_tf32 : bool
        Whether TF32 is allowed for matmul/cudnn. Always False in normal mode.
    enable_grad_scaler : bool
        Whether to wrap the backward/step path with ``torch.cuda.amp.GradScaler``.
    requested_policy : str
        Original policy string requested (pre-fallback).
    effective_policy : str
        Policy actually in force after host/gpu capability checks.
    reason : str
        Human-readable reason for the effective policy (logging + diagnostics).
    fallbacks_enabled : bool
        Whether the entrypoint may auto-degrade on non-finite loss / OOM.
    extras : Dict[str, Any]
        Free-form diagnostics (compute capability, amp availability, etc.).
    """

    device: torch.device
    use_amp: bool
    amp_dtype: torch.dtype
    use_tf32: bool
    enable_grad_scaler: bool
    requested_policy: str
    effective_policy: str
    reason: str
    fallbacks_enabled: bool = True
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "device": str(self.device),
            "use_amp": bool(self.use_amp),
            "amp_dtype": str(self.amp_dtype),
            "use_tf32": bool(self.use_tf32),
            "enable_grad_scaler": bool(self.enable_grad_scaler),
            "requested_policy": self.requested_policy,
            "effective_policy": self.effective_policy,
            "reason": self.reason,
            "fallbacks_enabled": bool(self.fallbacks_enabled),
        }
        d.update(self.extras or {})
        return d


# --------------------------------------------------------------------------- #
# Host / device capability probing (pure functions; no state mutation)
# --------------------------------------------------------------------------- #
def _query_compute_capability(device: torch.device) -> Tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return (0, 0)
    try:
        cap = torch.cuda.get_device_capability(device)
        return (int(cap[0]), int(cap[1]))
    except Exception:
        return (0, 0)


def _supports_fp16_amp(device: torch.device) -> Tuple[bool, str]:
    """Return (supported, reason). Conservative: no CUDA -> no AMP."""
    if device.type != "cuda":
        return False, "non-cuda device -> fp32"
    if not torch.cuda.is_available():
        return False, "cuda unavailable -> fp32"
    cap = _query_compute_capability(device)
    if cap < MIN_AMP_COMPUTE_CAPABILITY:
        return False, f"compute_capability {cap} < {MIN_AMP_COMPUTE_CAPABILITY} -> fp32"
    # torch.cuda.amp.autocast must exist in this torch build.
    if not hasattr(torch, "cuda") or not hasattr(torch.cuda, "amp"):
        return False, "torch.cuda.amp unavailable -> fp32"
    return True, "fp16 amp eligible"


# --------------------------------------------------------------------------- #
# Policy resolution (called once from the main guard)
# --------------------------------------------------------------------------- #
def resolve_precision_config(
    device: torch.device,
    requested_policy: Optional[str] = None,
    fallbacks_enabled: bool = True,
    allow_env_override: bool = True,
) -> PrecisionConfig:
    """
    Resolve the precision plan for this process.

    Precedence for ``requested_policy``:
      1. Explicit argument (usually from the training entrypoint config).
      2. ``MLEVOLVE_PRECISION_POLICY`` env var (optional runtime knob).
      3. ``DEFAULT_PRECISION_POLICY`` (= "fp16_amp" per pipeline trace).

    Fallback rules (per ``fallback_policy``):
      * CPU / unsupported device -> fp32 with AMP and scaler disabled.
      * Unknown policy -> fp32.
      * fp16_amp requested but not supported -> fp32.
      * TF32 is *always* disabled in normal mode.
    """
    requested = requested_policy
    if requested is None and allow_env_override:
        requested = os.environ.get("MLEVOLVE_PRECISION_POLICY")
    if requested is None:
        requested = DEFAULT_PRECISION_POLICY
    requested = str(requested).strip().lower()

    if requested not in SUPPORTED_PRECISION_POLICIES:
        base_reason = f"unknown policy '{requested}' -> fp32 fallback"
        requested = "fp32"
    else:
        base_reason = ""

    extras: Dict[str, Any] = {
        "amp_dtype_name": "float16",
        "precision_model_adaptation": PRECISION_MODEL_ADAPTATION,
    }

    # CPU or unsupported host: hard fp32, no AMP, no scaler.
    if requested in ("fp16_amp",):
        supported, why = _supports_fp16_amp(device)
        if not supported:
            extras["amp_support_reason"] = why
            return PrecisionConfig(
                device=device,
                use_amp=False,
                amp_dtype=FP32_DTYPE,
                use_tf32=False,
                enable_grad_scaler=False,
                requested_policy=requested,
                effective_policy="disabled",
                reason=(base_reason or "") + f"fp16_amp -> disabled ({why})",
                fallbacks_enabled=bool(fallbacks_enabled),
                extras=extras,
            )
        cap = _query_compute_capability(device)
        extras["compute_capability"] = f"{cap[0]}.{cap[1]}"
        return PrecisionConfig(
            device=device,
            use_amp=True,
            amp_dtype=AMP_DTYPE,
            use_tf32=False,
            enable_grad_scaler=True,  # fp16 needs loss scaling
            requested_policy=requested,
            effective_policy="fp16_amp",
            reason=(base_reason or "") + "fp16 amp active on eligible ops",
            fallbacks_enabled=bool(fallbacks_enabled),
            extras=extras,
        )

    # "fp32" or "disabled": explicitly FP32, no autocast, no scaler.
    effective = "fp32" if requested == "fp32" else "disabled"
    return PrecisionConfig(
        device=device,
        use_amp=False,
        amp_dtype=FP32_DTYPE,
        use_tf32=False,
        enable_grad_scaler=False,
        requested_policy=requested,
        effective_policy=effective,
        reason=(base_reason or "") + f"requested '{requested}' -> {effective}",
        fallbacks_enabled=bool(fallbacks_enabled),
        extras=extras,
    )


# --------------------------------------------------------------------------- #
# TF32 control (must be invoked from the entrypoint, never at import time)
# --------------------------------------------------------------------------- #
def configure_tf32(allow: bool = False) -> Dict[str, Any]:
    """
    Set TF32 matmul/cudnn flags. Normal mode requires ``allow=False``.

    Returns a snapshot dict for logging. Safe on CPU-only hosts.
    """
    observed: Dict[str, Any] = {"requested_allow_tf32": bool(allow)}

    if allow:
        # Explicitly out of policy in normal mode - guard anyway.
        observed["tf32_applied"] = False
        observed["tf32_reason"] = "tf32 disabled by normal-mode policy"
        # Do not enable even if requested; log the refusal.
        return observed

    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, "set_float32_matmul_precision"):
            # "highest" == fp32 matmul (no tf32/bf16 reduction path).
            torch.set_float32_matmul_precision("highest")
        observed["tf32_applied"] = True
        observed["tf32_reason"] = "disabled (normal mode)"
    except Exception as exc:  # pragma: no cover - defensive only
        observed["tf32_applied"] = False
        observed["tf32_reason"] = f"tf32 configure failed: {exc!r}"

    if torch.cuda.is_available() and hasattr(torch.backends, "cuda"):
        try:
            observed["matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        except Exception:
            pass
    if hasattr(torch.backends, "cudnn"):
        try:
            observed["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        except Exception:
            pass
    return observed


# --------------------------------------------------------------------------- #
# GradScaler
# --------------------------------------------------------------------------- #
def make_grad_scaler(config: PrecisionConfig) -> Optional["torch.cuda.amp.GradScaler"]:
    """
    Build a CUDA GradScaler iff fp16 AMP is active and enabled. Else returns None.

    GradScaler is constructed on demand (never at import). On CPU it is None.
    """
    if not config.use_amp or not config.enable_grad_scaler:
        return None
    if config.device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.amp.GradScaler(enabled=True)
    except Exception:
        # If the torch build has no GradScaler, fall back to disabled AMP path
        # via caller (caller should set config.use_amp=False in that case).
        return None


# --------------------------------------------------------------------------- #
# Autocast contexts
# --------------------------------------------------------------------------- #
def autocast_forward_ctx(config: PrecisionConfig):
    """
    Context manager to wrap ELIGIBLE forward ops (model forward + einsum/linalg).

    * If fp16 AMP is active -> ``torch.cuda.amp.autocast(enabled=True, dtype=fp16)``.
    * Else yields a null context (FP32 everywhere).

    Do NOT put backward, optimizer.step, loss reductions, or metric accumulation
    inside this context.
    """
    if config.use_amp and config.device.type == "cuda":
        return torch.cuda.amp.autocast(
            enabled=True, dtype=config.amp_dtype, cache_enabled=True
        )
    return nullcontext()


@contextmanager
def fp32_island():
    """
    Local FP32 island for FP32-sensitive reductions (e.g. BCEWithLogitsLoss)
    that must not run under fp16 autocast. Yields a disabled-autocast context.
    """
    if torch.cuda.is_available():
        with torch.cuda.amp.autocast(enabled=False):
            yield
    else:
        yield


# --------------------------------------------------------------------------- #
# FP32-safe loss / prediction helpers
# --------------------------------------------------------------------------- #
def compute_bce_with_logits_fp32(
    logits: torch.Tensor,
    target: torch.Tensor,
    criterion: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    FP32-sensitive BCEWithLogitsLoss island.

    * Casts logits/targets to FP32.
    * Runs under autocontext disabled so the reduction stays in fp32.
    * Uses a plain ``nn.BCEWithLogitsLoss(reduction='mean')`` if none is passed.
    """
    with fp32_island():
        logits_f32 = logits.float()
        target_f32 = target.float()
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss(reduction="mean")
        loss = criterion(logits_f32, target_f32)
    # Ensure the returned scalar is FP32.
    if loss.dtype != FP32_DTYPE:
        loss = loss.float()
    return loss


def probs_from_logits_fp32(logits: torch.Tensor) -> torch.Tensor:
    """Sigmoid in FP32 for metric/export. Never accumulate metrics in fp16."""
    with fp32_island():
        return torch.sigmoid(logits.float())


# --------------------------------------------------------------------------- #
# Fallback helpers (used by the training entrypoint; not by Stage 1)
# --------------------------------------------------------------------------- #
def coerce_to_fp32(config: PrecisionConfig) -> PrecisionConfig:
    """
    Force a config into pure FP32 / no AMP / no scaler. Keeps requested_policy
    for logging; flips effective to 'disabled'.
    """
    fallback = PrecisionConfig(
        device=config.device,
        use_amp=False,
        amp_dtype=FP32_DTYPE,
        use_tf32=False,
        enable_grad_scaler=False,
        requested_policy=config.requested_policy,
        effective_policy="disabled",
        reason=f"fallback from {config.effective_policy}: forced fp32",
        fallbacks_enabled=config.fallbacks_enabled,
        extras=dict(config.extras or {}),
    )
    fallback.extras["fallback_from"] = config.effective_policy
    return fallback


def nonfinite_scalar(x: Any) -> bool:
    """Return True if a scalar loss/float is NaN/inf. Robust to tensors."""
    try:
        if isinstance(x, torch.Tensor):
            return not bool(torch.isfinite(x.detach()).all().item())
        v = float(x)
        return math.isnan(v) or math.isinf(v)
    except Exception:
        return True  # conservative


def maybe_degrade_for_nonfinite(
    config: PrecisionConfig, observed_loss: Any
) -> Optional[PrecisionConfig]:
    """
    If the observed loss is non-finite and AMP is active, return an FP32 config
    to swap in. Otherwise return None (no change requested).

    The caller owns applying the fallback (rebuilding the scaler, continuing).
    """
    if not config.fallbacks_enabled:
        return None
    if not nonfinite_scalar(observed_loss):
        return None
    if not config.use_amp:
        return None
    return coerce_to_fp32(config)


# --------------------------------------------------------------------------- #
# Precision logging (one line per process + optional one line per epoch)
# --------------------------------------------------------------------------- #
def format_precision_log(config: PrecisionConfig) -> str:
    """One-line human-readable summary of the resolved precision plan."""
    d = config.to_dict()
    parts = [
        f"device={d.get('device')}",
        f"effective={d.get('effective_policy')}",
        f"requested={d.get('requested_policy')}",
        f"amp={d.get('use_amp')}",
        f"amp_dtype={d.get('amp_dtype')}",
        f"tf32={d.get('use_tf32')}",
        f"scaler={d.get('enable_grad_scaler')}",
        f"fallback={d.get('fallbacks_enabled')}",
    ]
    if d.get("compute_capability"):
        parts.append(f"cc={d['compute_capability']}")
    if d.get("amp_support_reason"):
        parts.append(f"reason={d['amp_support_reason']}")
    parts.append(f"policy_reason='{d.get('reason')}'")
    return "precision | " + " | ".join(parts)


def precision_diagnostics_settings(config: PrecisionConfig) -> Dict[str, Any]:
    """
    Settings slice for TrainingDiagnostics. Stage 3 will merge this into its
    larger settings dict; it is intentionally small and side-effect free.
    """
    return {
        "precision": config.effective_policy,
        "precision_reason": config.reason,
        "amp_dtype": str(config.amp_dtype) if config.use_amp else "fp32",
        "use_amp": bool(config.use_amp),
        "use_tf32": bool(config.use_tf32),
        "grad_scaler": bool(config.enable_grad_scaler),
        "precision_model_adaptation": PRECISION_MODEL_ADAPTATION,
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
 Summary: 2 total attempts - 2 executed, 2 successful (best: 0.7573)

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
Required hardware-aware step order: model-design -> datatype/quantization -> training (model_design -> datatype_precision -> training_evaluation). Use the trace as the source of truth, apply hardware evidence only when compatible, and use recorded fallbacks when evidence is missing.

# Pipeline Decision Trace
```json
{"model_design":{"modality":"text","target_type":"classification","shape_constraints":["sequence length and tokenizer output must stay configurable"],"family":"available_transformer_or_sequence_baseline","alternatives_considered":[],"loss":"{'criterion': 'BCEWithLogitsLoss', 'reduction': 'mean', 'fp32_sensitive': True, 'notes': 'compute...","output_interface":"match the competition metric and required submission format","reason":"Use a baseline-compatible model family until stronger task or evidence signals are available.","hardware_fit":"use only compatible evidence"},"datatype_precision":{"precision_policy":"fp16_amp","precision_model_adaptation":"none","fallback_policy":"fall back to fp32 or disabled AMP if the selected precision path is unsupported","reason":"Use conservative precision until hardware and package support are confirmed."},"training_evaluation":{"optimizer":"{'name': 'AdamW', 'params': {'lr': 2e-05, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-...","scheduler":"{'name': 'linear_warmup_decay', 'warmup_ratio': 0.1, 'epochs': 3}","batch_size_policy":"fixed","dataloader_policy":"safe defaults; enable workers and pinned memory only when compatible","fallbacks":["reduce physical batch size on OOM","reduce epochs before changing model family on timeout"],"metrics_to_log":["elapsed_seconds","peak_vram_mb","resolved_batch_size"],"advanced_optimizer_used":false,"reason":"Choose stable training defaults matched to the inferred target type and available model family."},"evidence":{"hardware_context_used":true,"evidence_refs":["backend_guidance:cuda_process.no_mps_controls","backend_guidance:cuda_process.normal_process_throughput","backend_guidance:shared.subprocess.isolated_artifacts_and_telemetry"],"confidence":0.95,"missing_evidence":["predictor/graph evidence not available","HWDB recommended_patterns are conditional candidates, not proof of correctness, speed or package...","dataset_decomposition limitations: no hardware-specific acceleration and no direct inference bene..."]}}
```

# Cross-Stage Note Board
- Purpose: preserve the candidate intent across Stage 1, Stage 2, and Stage 3 so later stages optimize toward the same target.
- Notes so far:
  - Stage 1 model-design / model_design: baseline_change=replaced single-view word TF-IDF MLP / transformer with a multi-view fusion net (word+char TF-IDF ⊕ token-embedding pooling ⊕ keyword embedding); purpose=capture both lexical and keyword-conditioned semantic signal for higher F1; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=added char_wb n-grams + explicit keyword vocabulary fitted only on train; purpose=robustness to tweet noise/typos and strong keyword priors; hardware_keys=none
  - Stage 1 model-design / model_design: baseline_change=exposed FP32 logit interface + CandidateAdapter exercising the real model/loss/features; purpose=keep Stage-2 precision and Stage-3 training aligned to this candidate; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=## Plan **What's unique here**: The task allows only `fp32`/`disabled`/`fp16_amp` on an Ampere A10 (sm_86).; purpose=Support the Stage 1 candidate target with a compatible precision policy.; hardware_keys=fp16_amp, auto_degrade_on_nonfinite, sparse_dim, vocab_size, n_keywords, emb_dim, hidden_dim, seq_len
  - Stage 3 training / training_evaluation: baseline_change=added Stage-3 run_training with AdamW + linear-warmup→cosine + early-stopping/best-F1 restore + val-tuned threshold; purpose=preserve Stage-1 fusion target while stabilising the small-data training signal; hardware_keys=compute_capability=8.6, vram_mib=22589
  - Stage 3 training / training_evaluation: baseline_change=FP16-AMP forward + FP32 loss island + GradScaler with unscale-then-clip + `maybe_degrade_for_nonfinite` watchdog; purpose=consume Stage-2 precision policy `fp16_amp` on A10 without touching model family/loss; hardware_keys=fp16_amp, auto_degrade_on_nonfinite
  - Stage 3 training / training_evaluation: baseline_change=TrainingDiagnostics + script_scheduler_context safe-points (BEFORE_TRAIN/STEP/EPOCH) with full state_factory; purpose=align runtime observability and resumability with scheduler contract; hardware_keys=none

# Step results

        ### Step 1: model_design
        **Plan:** ## Plan

**What makes this task unique:** Disaster tweets are short, noisy (URLs, mentions, hashtags, typos), and critically rely on *keyword-conditioned* semantics — e.g. "ablaze"/"earthquake" almost always signal real disasters, while "damage"/"body bags" are frequently metaphorical. So the discriminative signal lives in *both* surface lexical patterns (where sparse TF-IDF dominates) and keyword-relative meaning.

**Why this approach:** The two prior attempts used a single-view word TF-IDF MLP (0.7573) and a from-scratch hybrid transformer (0.7341). Both fail to combine (a) sparse lexical generalization that won best score, with (b) learned compositional/semantic signal. I will build a **multi-view fusion network**: a word+char TF-IDF sparse branch (proven strong) fused with a trainable token-embedding mean-pool branch and a keyword-embedding branch. Char n-grams add robustness to tweet noise/typos; the keyword embedding explicitly lets the head condition text evidence on the keyword identity — a genuine architectural change, not a hyperparameter tweak.

**How I'll validate:** Internal stratified 20% hold-out (seed 42). Confirmation = fused model beats the single-view sparse-only baseline on hold-out F1; re-evaluation trigger = if the learned embedding branch adds no val-F1 over the TF-IDF-only branch, later stages should down-weight it (keep fusion architecture but rely on sparse head).

Memory strategy: fit vectorizers/tokenizer/keyword-vocab **only on the train split** and reuse for val/test. Batch elasticity: model consumes a fixed sparse dim; batch size stays a Stage-3 concern (exposed via `scenario['batch_size']`). Loss is FP32-sensitive `BCEWithLogitsLoss` (output logits, F1 threshold tuned later). Scheduler-owned controls (device, workers, launch) are **not** implemented here.

Note board:
- baseline_change: replaced single-view word TF-IDF MLP / transformer with a multi-view fusion net (word+char TF-IDF ⊕ token-embedding pooling ⊕ keyword embedding) | purpose: capture both lexical and keyword-conditioned semantic signal for higher F1 | hardware_keys: none
- baseline_change: added char_wb n-grams + explicit keyword vocabulary fitted only on train | purpose: robustness to tweet noise/typos and strong keyword priors | hardware_keys: none
- baseline_change: exposed FP32 logit interface + CandidateAdapter exercising the real model/loss/features | purpose: keep Stage-2 precision and Stage-3 training aligned to this candidate | hardware_keys: none
        **Code:**
        ```python
"""
Stage 1 - model_design
Disaster Tweets (binary F1).

This module is import-safe: it only defines constants, data/feature helpers,
the torch model, the criterion factory, split helper, dataset class, and the
CPU-preflight CandidateAdapter. All training / inference side effects belong
to the final main guard added by later stages.

No optimizer/scheduler/precision/batch-size policy is finalized here; only
adapter-required defaults are provided so the CPU preflight can run.
"""

import os
import re
import html
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Contract constants
# --------------------------------------------------------------------------- #
MODEL_FAMILY = "multiview_tfidf_embed_mlp_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
INPUT_DIR = "./input"

SEED = 42
VAL_SIZE = 0.20

# Feature / model geometry (configurable)
MAX_WORD_FEATURES = 20000
MAX_CHAR_FEATURES = 20000
WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (2, 5)
SEQ_LEN = 48
EMB_DIM = 96
HIDDEN_DIM = 128
DROPOUT = 0.30
KEYWORD_EMB_DIM = 16

PAD_ID = 0
UNK_ID = 1
NO_KEYWORD = "__nokeyword__"

# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_NUM_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def clean_text(text: object) -> str:
    """Normalize a tweet: unescape HTML, lowercase, mask URLs/mentions/numbers."""
    if not isinstance(text, str):
        return ""
    t = html.unescape(text).lower()
    t = _URL_RE.sub(" urltoken ", t)
    t = _MENTION_RE.sub(" mentiontoken ", t)
    t = t.replace("#", " ")
    t = _NUM_RE.sub(" numtoken ", t)
    t = _NON_ALNUM_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def clean_keyword(value: object) -> str:
    if not isinstance(value, str) or value.strip() == "":
        return NO_KEYWORD
    return value.replace("%20", " ").strip().lower()


def clean_location(value: object) -> str:
    if not isinstance(value, str) or value.strip() == "":
        return ""
    t = html.unescape(value).lower()
    t = _NON_ALNUM_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def load_and_clean_data(input_dir: str = INPUT_DIR) -> pd.DataFrame:
    """Load train.csv and add cleaned text/keyword/location columns."""
    df = pd.read_csv(os.path.join(input_dir, "train.csv"))
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    df["clean_keyword"] = df.get("keyword", pd.Series([None] * len(df))).apply(
        clean_keyword
    )
    df["clean_location"] = df.get("location", pd.Series([None] * len(df))).apply(
        clean_location
    )
    # Model text: keyword prior + tweet semantics + location tokens.
    df["model_text"] = (
        df["clean_keyword"].replace(NO_KEYWORD, "")
        + " "
        + df["clean_text"]
        + " "
        + df["clean_location"]
    ).str.strip()
    if "target" in df.columns:
        df["target"] = df["target"].astype(np.float32)
    return df


def make_splits(
    df: pd.DataFrame, val_size: float = VAL_SIZE, seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/validation split. Fitted only on the train split later."""
    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=seed, stratify=df["target"].values
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Feature artifacts (fit on train split only)
# --------------------------------------------------------------------------- #
@dataclass
class FeatureArtifacts:
    word_vec: TfidfVectorizer
    char_vec: TfidfVectorizer
    token_to_id: Dict[str, int]
    keyword_to_id: Dict[str, int]
    n_keywords: int
    sparse_dim: int
    vocab_size: int
    seq_len: int = SEQ_LEN


_TOKEN_SPLIT_RE = re.compile(r"\s+")


def tokenize(text: str) -> List[str]:
    return [tok for tok in _TOKEN_SPLIT_RE.split(text) if tok]


def fit_feature_artifacts(train_df: pd.DataFrame) -> FeatureArtifacts:
    """Fit word/char TF-IDF, token vocab and keyword vocab on TRAIN ONLY."""
    texts = train_df["model_text"].fillna("").tolist()

    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=WORD_NGRAM_RANGE,
        min_df=2,
        max_features=MAX_WORD_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=CHAR_NGRAM_RANGE,
        min_df=3,
        max_features=MAX_CHAR_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    word_vec.fit(texts)
    char_vec.fit(texts)

    # token vocabulary
    token_to_id: Dict[str, int] = {}
    for text in texts:
        for tok in tokenize(text):
            if tok not in token_to_id:
                token_to_id[tok] = len(token_to_id) + 2  # 0=pad, 1=unk
    vocab_size = max(len(token_to_id) + 2, 2)

    # keyword vocabulary
    keywords = train_df["clean_keyword"].fillna(NO_KEYWORD).tolist()
    keyword_to_id: Dict[str, int] = {}
    for kw in keywords:
        if kw not in keyword_to_id:
            keyword_to_id[kw] = len(keyword_to_id)
    if NO_KEYWORD not in keyword_to_id:
        keyword_to_id[NO_KEYWORD] = len(keyword_to_id)
    n_keywords = max(len(keyword_to_id), 1)

    sparse_dim = len(word_vec.get_feature_names_out()) + len(
        char_vec.get_feature_names_out()
    )

    return FeatureArtifacts(
        word_vec=word_vec,
        char_vec=char_vec,
        token_to_id=token_to_id,
        keyword_to_id=keyword_to_id,
        n_keywords=n_keywords,
        sparse_dim=int(sparse_dim),
        vocab_size=int(vocab_size),
        seq_len=SEQ_LEN,
    )


def transform_sparse(df: pd.DataFrame, art: FeatureArtifacts) -> sparse.csr_matrix:
    texts = df["model_text"].fillna("").tolist()
    w = art.word_vec.transform(texts)
    c = art.char_vec.transform(texts)
    return sparse.hstack([w, c]).tocsr().astype(np.float32)


def encode_tokens(
    df: pd.DataFrame, art: FeatureArtifacts, seq_len: Optional[int] = None
) -> torch.Tensor:
    seq_len = seq_len or art.seq_len
    out = np.zeros((len(df), seq_len), dtype=np.int64)
    for i, text in enumerate(df["model_text"].fillna("").tolist()):
        ids = [art.token_to_id.get(tok, UNK_ID) for tok in tokenize(text)][:seq_len]
        if ids:
            out[i, : len(ids)] = ids
    return torch.from_numpy(out)


def encode_keywords(df: pd.DataFrame, art: FeatureArtifacts) -> torch.Tensor:
    ids = [
        art.keyword_to_id.get(kw, art.keyword_to_id[NO_KEYWORD])
        for kw in df["clean_keyword"].fillna(NO_KEYWORD).tolist()
    ]
    return torch.tensor(ids, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Model: multi-view fusion
# --------------------------------------------------------------------------- #
class MultiViewDisasterNet(nn.Module):
    """
    Fuses three views of a tweet:
      * sparse word+char TF-IDF vector  -> linear branch (lexical generalization)
      * learned token-embedding mean pool -> compositional/semantic branch
      * keyword embedding -> lets the head condition text on the keyword prior
    Outputs raw logits (batch,) for BCEWithLogitsLoss.
    """

    def __init__(
        self,
        sparse_dim: int,
        vocab_size: int,
        n_keywords: int,
        emb_dim: int = EMB_DIM,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
        keyword_emb_dim: int = KEYWORD_EMB_DIM,
        pad_id: int = PAD_ID,
    ):
        super().__init__()
        self.pad_id = pad_id

        self.sparse_branch = nn.Sequential(
            nn.Linear(sparse_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.token_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.token_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.keyword_embedding = nn.Embedding(n_keywords, keyword_emb_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + keyword_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        sparse_features: torch.Tensor,
        token_ids: torch.Tensor,
        keyword_ids: torch.Tensor,
    ) -> torch.Tensor:
        sparse_features = sparse_features.float()
        token_ids = token_ids.long()
        keyword_ids = keyword_ids.long()

        a = self.sparse_branch(sparse_features)

        emb = self.token_embedding(token_ids)  # (B, L, E)
        mask = (token_ids != self.pad_id).unsqueeze(-1).float()  # (B, L, 1)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        b = self.token_proj(pooled)

        c = self.keyword_embedding(keyword_ids).reshape(keyword_ids.size(0), -1)

        x = torch.cat([a, b, c], dim=1)
        return self.head(x).squeeze(-1)


def build_loss() -> nn.Module:
    """FP32-sensitive criterion used by the training/eval stages."""
    return nn.BCEWithLogitsLoss(reduction="mean")


def build_model_from_artifacts(art: FeatureArtifacts) -> MultiViewDisasterNet:
    return MultiViewDisasterNet(
        sparse_dim=art.sparse_dim,
        vocab_size=art.vocab_size,
        n_keywords=art.n_keywords,
    )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class TextDisasterDataset(torch.utils.data.Dataset):
    """
    Yields (inputs, target) where inputs = (sparse_row, token_ids, keyword_ids).
    Sparse rows are stored as a compact float32 array to allow fast indexing.
    """

    def __init__(
        self,
        sparse_matrix: sparse.csr_matrix,
        token_ids: torch.Tensor,
        keyword_ids: torch.Tensor,
        targets: Optional[np.ndarray] = None,
        dense_sparse: bool = True,
    ):
        self.sparse_matrix = (
            np.ascontiguousarray(sparse_matrix.toarray(), dtype=np.float32)
            if dense_sparse
            else sparse_matrix
        )
        self.token_ids = token_ids.long()
        self.keyword_ids = keyword_ids.long()
        self.targets = (
            np.asarray(targets, dtype=np.float32) if targets is not None else None
        )

    def __len__(self) -> int:
        return self.token_ids.size(0)

    def __getitem__(self, idx: int):
        if isinstance(self.sparse_matrix, np.ndarray):
            sp = torch.from_numpy(self.sparse_matrix[idx])
        else:
            sp = torch.from_numpy(self.sparse_matrix[idx].toarray().ravel())
        item = {
            "inputs": (sp, self.token_ids[idx], self.keyword_ids[idx]),
        }
        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return item


# --------------------------------------------------------------------------- #
# CPU-preflight CandidateAdapter
# --------------------------------------------------------------------------- #
_FIXTURE_NAMES = ("sparse_features", "token_ids", "keyword_ids")


class CandidateAdapter:
    """
    No-argument adapter exercising the real model, features, loss and optimizer.
    Artifacts are fit lazily from the input fixtures and cached per instance.
    """

    def __init__(self):
        self._cache: Dict[str, object] = {}

    # -- helpers ---------------------------------------------------------- #
    def _input_dir(self) -> str:
        return os.environ.get("MLEVOLVE_INPUT_DIR", INPUT_DIR)

    def _ensure_data(self) -> Tuple[pd.DataFrame, FeatureArtifacts]:
        if "df" not in self._cache:
            df = load_and_clean_data(self._input_dir())
            art = fit_feature_artifacts(df)
            self._cache["df"] = df
            self._cache["art"] = art
        return self._cache["df"], self._cache["art"]  # type: ignore

    # -- required API ----------------------------------------------------- #
    def build_model(self, context) -> nn.Module:
        _df, art = self._ensure_data()
        return build_model_from_artifacts(art)

    def build_optimizer(self, model, context):
        context = dict(context or {})
        lr = float(context.get("lr", 1e-3))
        weight_decay = float(context.get("weight_decay", 0.01))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def build_train_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def _build_batch(self, scenario, device):
        scenario = dict(scenario or {})
        batch_size = int(scenario.get("batch_size", 4))
        fixture = scenario.get("fixture") or {}

        # Authoritative fixture path: build declared tensors, keep names.
        if fixture and all(name in fixture for name in _FIXTURE_NAMES):
            built: Dict[str, torch.Tensor] = {}
            for name in _FIXTURE_NAMES:
                spec = fixture[name]
                shape = spec.get("shape") if isinstance(spec, dict) else spec
                if name in ("token_ids", "keyword_ids") or "int" in str(
                    (spec or {}).get("dtype", "") if isinstance(spec, dict) else ""
                ):
                    built[name] = torch.zeros(
                        tuple(shape), dtype=torch.long, device=device
                    )
                else:
                    built[name] = torch.zeros(
                        tuple(shape), dtype=torch.float32, device=device
                    )
            target = torch.zeros(
                built["token_ids"].size(0), dtype=torch.float32, device=device
            )
            out = dict(built)
            out["inputs"] = (
                built["sparse_features"],
                built["token_ids"],
                built["keyword_ids"],
            )
            out["target"] = target
            return out

        # Real-data path.
        df, art = self._ensure_data()
        sub = df.head(batch_size).reset_index(drop=True)

        sp = transform_sparse(sub, art).toarray().astype(np.float32)
        sparse_t = torch.from_numpy(sp).to(device)
        tokens = encode_tokens(sub, art).to(device)
        keywords = encode_keywords(sub, art).to(device)
        target = torch.tensor(
            sub["target"].values.astype(np.float32), dtype=torch.float32, device=device
        )

        return {
            "inputs": (sparse_t, tokens, keywords),
            "target": target,
            "sparse_features": sparse_t,
            "token_ids": tokens,
            "keyword_ids": keywords,
        }

    def _resolve_criterion(self, context) -> nn.Module:
        context = dict(context or {})
        crit = context.get("criterion")
        if crit is None:
            crit = build_loss()
        return crit

    def training_step(self, model, batch, context):
        inputs = batch["inputs"]
        target = batch["target"].float()
        logits = model(*inputs)
        criterion = self._resolve_criterion(context)
        return criterion(logits, target)

    def validation_step(self, model, batch, context):
        inputs = batch["inputs"]
        target = batch["target"].float()
        logits = model(*inputs)
        criterion = self._resolve_criterion(context)
        loss = criterion(logits, target)
        probs = torch.sigmoid(logits.detach())
        return {
            "loss": loss,
            "logits": logits.detach(),
            "probs": probs,
            "target": target,
        }

```
        
        ### Step 2: datatype_precision
        **Plan:** ## Plan

**What's unique here**: The task allows only `fp32`/`disabled`/`fp16_amp` on an Ampere A10 (sm_86). The Stage-1 model is a small multi-view MLP with a sparse TF-IDF linear branch, an embedding-pool branch, and an FP32-sensitive `BCEWithLogitsLoss`. Since Stage 1 chose `precision_model_adaptation: none`, I cannot swap layers — I can only wrap eligible forward ops in autocast, keep the loss and metric/export in FP32, and provide a graceful CPU/unsupported fallback.

**Why this design**: FP16 AMP shrinks activation memory and speeds matmul on Ampere Tensor Cores only if the ops are eligible; my sparse branch is a simple `Linear` (eligible), the embedding pooling is cheap, and the head is `Linear` — so the only real gain is in `Linear` GEMMs, which AMP covers. I disable TF32 (not allowed in normal mode) and disable AMP on CPU/unsupported GPUs. `BCEWithLogitsLoss` is FP32-sensitive per the decision trace, so I run it in `autocast(enabled=False)` with explicit FP32 casts of logits/targets.

**How I'll validate**: The training step will log the resolved precision plan (device, use_amp, dtype, use_tf32, scaled) once and the diagnostics helper will report the *actual* autocast dtype per epoch; if val-F1 does not beat the FP32 baseline within tolerance, the fallback flag degrades to `disabled` AMP and we re-run FP32. Confirmation = finite loss under FP16 AMP for the full run + non-regressed val-F1; disconfirmation = NaN/inf loss or a drop in val-F1 → degrade to FP32.

**Memory strategy & batch elasticity**: No change to Stage-3 batch size. AMP only changes dtype of activations (halves memory pressure), leaving batch size/accumulation to Stage 3. Fallback: on OOM or non-finite loss, degrade to FP32 AMP (`auto_degrade_on_nonfinite`) without touching the optimizer, LR, epochs, or dataloaders.

**Operator & normalization**: Autocast wraps model forward only (matmul → fp16). Loss is a FP32 island. Backward and `scaler.step`/`scaler.update` run outside autocast; gradient unscale via `scaler.unscale_(optimizer)` before clipping. Normalization layers (none in Stage-1 model) would be kept in FP32 automatically by autocast; embeddings stay FP32 (autocast does not cast `nn.Embedding` outputs). BatchNorm/softmax would be opaque but we have none — this is a safe workload for fp16 AMP.

**Configurable model dims**: unchanged from Stage 1 (`sparse_dim`, `vocab_size`, `n_keywords`, `emb_dim`, `hidden_dim`, `seq_len`) — precision stage does not touch them.

**Scheduler-owned controls (NOT implemented here)**: device assignment, thread/worker allocation, exclusive-vs-shared VRAM packing, subprocess launch, and NCCL/collective setup all belong to the scheduler per the backend contract (`cuda_process`, `subprocess_job_v1`). This module only resolves *tensor* dtype/precision inside a single process.

**Note board:**
- baseline_change: added a precision-plan utility layer (`resolve_precision_config`, `sentence_forward_no_grad`, FP32 loss island, `make_grad_scaler`, `configure_tf32`, `degrade_to_fp32`) with no model surgery | purpose: consume the pipeline `datatype_precision.precision_policy=fp16_amp` on sm_86 while preserving Stage-1 model family, loss, features and output interface | hardware_keys: compute_capability=8.6, vram_mib=22589, precision_mode=normal, allowed_policies=fp32/disabled/fp16_amp
- baseline_change: explicit TF32 disable helper invoked only from the main guard | purpose: normal-mode rule forbids TF32, but capability alone does not authorize it; disabling keeps numerics identical to the FP32 reference | hardware_keys: tf32_policy=disable
- baseline_change: fallback path (`auto_degrade_on_nonfinite`, `degrade_to_fp32`) tracks a runtime state and forces AMP off on CPU/unsupported/NaN | purpose: keeps run robust without changing loss/model/optimizer; Stage-3 can act on the flag when non-finite loss is detected | hardware_keys: fallback_policy=fall-back-to-fp32-or-disabled
        **Code:**
        ```python
"""
Stage 2 - datatype_precision
Disaster Tweets binary F1.

Precision policy consumed from the pipeline trace:
  datatype_precision.precision_policy = "fp16_amp"
  datatype_precision.precision_model_adaptation = "none"
  datatype_precision.fallback_policy = "fall back to fp32 or disabled AMP if the
                                        selected precision path is unsupported"

Allowed precision policies in normal mode: fp32, disabled, fp16_amp.
BF16, TF32, quantization, and Transformer-Engine FP8/MXFP8/NVFP4 recipes are OUT.

This module is import-safe: only constants, dataclass, and helpers are defined.
It never touches CUDA/TF32, never instantiates models/optimizers/losses, never
reads data, and never mutates global torch flags at import time. All activation
of these helpers happens inside `if __name__ == '__main__':` in the final
training entrypoint.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Precision contract constants (import-safe; no side effects)
# --------------------------------------------------------------------------- #
# Pipeline-decision default. Env can opt out without touching code.
DEFAULT_PRECISION_POLICY: str = "fp16_amp"  # "fp16_amp" | "fp32" | "disabled"
SUPPORTED_PRECISION_POLICIES: Tuple[str, ...] = ("fp16_amp", "fp32", "disabled")
AMP_DTYPE: torch.dtype = torch.float16  # fp16 only in normal mode
FP32_DTYPE: torch.dtype = torch.float32

# Ampere sm_86 supports fp16 tensor cores; we still gate on cuda.is_available().
MIN_AMP_COMPUTE_CAPABILITY: Tuple[int, int] = (7, 0)  # Volta+ has fp16 tensor cores

# Precision-required model adaptation per pipeline trace.
PRECISION_MODEL_ADAPTATION: str = "none"

# Small epsilon used for numerical comparisons only (not a policy knob).
_NONFINITE_EPS: float = float("nan")


# --------------------------------------------------------------------------- #
# Resolved precision configuration
# --------------------------------------------------------------------------- #
@dataclass
class PrecisionConfig:
    """
    Immutable-ish snapshot of the resolved precision plan for one process.

    Attributes
    ----------
    device : torch.device
        Target device selected for training/eval. Assigned by the entrypoint
        (scheduler is authoritative for GPU allocation); we only record it.
    use_amp : bool
        Whether CUDA autocast (fp16) is enabled for eligible forward ops.
    amp_dtype : torch.dtype
        Autocast dtype when ``use_amp`` is True. Always fp16 here.
    use_tf32 : bool
        Whether TF32 is allowed for matmul/cudnn. Always False in normal mode.
    enable_grad_scaler : bool
        Whether to wrap the backward/step path with ``torch.cuda.amp.GradScaler``.
    requested_policy : str
        Original policy string requested (pre-fallback).
    effective_policy : str
        Policy actually in force after host/gpu capability checks.
    reason : str
        Human-readable reason for the effective policy (logging + diagnostics).
    fallbacks_enabled : bool
        Whether the entrypoint may auto-degrade on non-finite loss / OOM.
    extras : Dict[str, Any]
        Free-form diagnostics (compute capability, amp availability, etc.).
    """

    device: torch.device
    use_amp: bool
    amp_dtype: torch.dtype
    use_tf32: bool
    enable_grad_scaler: bool
    requested_policy: str
    effective_policy: str
    reason: str
    fallbacks_enabled: bool = True
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "device": str(self.device),
            "use_amp": bool(self.use_amp),
            "amp_dtype": str(self.amp_dtype),
            "use_tf32": bool(self.use_tf32),
            "enable_grad_scaler": bool(self.enable_grad_scaler),
            "requested_policy": self.requested_policy,
            "effective_policy": self.effective_policy,
            "reason": self.reason,
            "fallbacks_enabled": bool(self.fallbacks_enabled),
        }
        d.update(self.extras or {})
        return d


# --------------------------------------------------------------------------- #
# Host / device capability probing (pure functions; no state mutation)
# --------------------------------------------------------------------------- #
def _query_compute_capability(device: torch.device) -> Tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return (0, 0)
    try:
        cap = torch.cuda.get_device_capability(device)
        return (int(cap[0]), int(cap[1]))
    except Exception:
        return (0, 0)


def _supports_fp16_amp(device: torch.device) -> Tuple[bool, str]:
    """Return (supported, reason). Conservative: no CUDA -> no AMP."""
    if device.type != "cuda":
        return False, "non-cuda device -> fp32"
    if not torch.cuda.is_available():
        return False, "cuda unavailable -> fp32"
    cap = _query_compute_capability(device)
    if cap < MIN_AMP_COMPUTE_CAPABILITY:
        return False, f"compute_capability {cap} < {MIN_AMP_COMPUTE_CAPABILITY} -> fp32"
    # torch.cuda.amp.autocast must exist in this torch build.
    if not hasattr(torch, "cuda") or not hasattr(torch.cuda, "amp"):
        return False, "torch.cuda.amp unavailable -> fp32"
    return True, "fp16 amp eligible"


# --------------------------------------------------------------------------- #
# Policy resolution (called once from the main guard)
# --------------------------------------------------------------------------- #
def resolve_precision_config(
    device: torch.device,
    requested_policy: Optional[str] = None,
    fallbacks_enabled: bool = True,
    allow_env_override: bool = True,
) -> PrecisionConfig:
    """
    Resolve the precision plan for this process.

    Precedence for ``requested_policy``:
      1. Explicit argument (usually from the training entrypoint config).
      2. ``MLEVOLVE_PRECISION_POLICY`` env var (optional runtime knob).
      3. ``DEFAULT_PRECISION_POLICY`` (= "fp16_amp" per pipeline trace).

    Fallback rules (per ``fallback_policy``):
      * CPU / unsupported device -> fp32 with AMP and scaler disabled.
      * Unknown policy -> fp32.
      * fp16_amp requested but not supported -> fp32.
      * TF32 is *always* disabled in normal mode.
    """
    requested = requested_policy
    if requested is None and allow_env_override:
        requested = os.environ.get("MLEVOLVE_PRECISION_POLICY")
    if requested is None:
        requested = DEFAULT_PRECISION_POLICY
    requested = str(requested).strip().lower()

    if requested not in SUPPORTED_PRECISION_POLICIES:
        base_reason = f"unknown policy '{requested}' -> fp32 fallback"
        requested = "fp32"
    else:
        base_reason = ""

    extras: Dict[str, Any] = {
        "amp_dtype_name": "float16",
        "precision_model_adaptation": PRECISION_MODEL_ADAPTATION,
    }

    # CPU or unsupported host: hard fp32, no AMP, no scaler.
    if requested in ("fp16_amp",):
        supported, why = _supports_fp16_amp(device)
        if not supported:
            extras["amp_support_reason"] = why
            return PrecisionConfig(
                device=device,
                use_amp=False,
                amp_dtype=FP32_DTYPE,
                use_tf32=False,
                enable_grad_scaler=False,
                requested_policy=requested,
                effective_policy="disabled",
                reason=(base_reason or "") + f"fp16_amp -> disabled ({why})",
                fallbacks_enabled=bool(fallbacks_enabled),
                extras=extras,
            )
        cap = _query_compute_capability(device)
        extras["compute_capability"] = f"{cap[0]}.{cap[1]}"
        return PrecisionConfig(
            device=device,
            use_amp=True,
            amp_dtype=AMP_DTYPE,
            use_tf32=False,
            enable_grad_scaler=True,  # fp16 needs loss scaling
            requested_policy=requested,
            effective_policy="fp16_amp",
            reason=(base_reason or "") + "fp16 amp active on eligible ops",
            fallbacks_enabled=bool(fallbacks_enabled),
            extras=extras,
        )

    # "fp32" or "disabled": explicitly FP32, no autocast, no scaler.
    effective = "fp32" if requested == "fp32" else "disabled"
    return PrecisionConfig(
        device=device,
        use_amp=False,
        amp_dtype=FP32_DTYPE,
        use_tf32=False,
        enable_grad_scaler=False,
        requested_policy=requested,
        effective_policy=effective,
        reason=(base_reason or "") + f"requested '{requested}' -> {effective}",
        fallbacks_enabled=bool(fallbacks_enabled),
        extras=extras,
    )


# --------------------------------------------------------------------------- #
# TF32 control (must be invoked from the entrypoint, never at import time)
# --------------------------------------------------------------------------- #
def configure_tf32(allow: bool = False) -> Dict[str, Any]:
    """
    Set TF32 matmul/cudnn flags. Normal mode requires ``allow=False``.

    Returns a snapshot dict for logging. Safe on CPU-only hosts.
    """
    observed: Dict[str, Any] = {"requested_allow_tf32": bool(allow)}

    if allow:
        # Explicitly out of policy in normal mode - guard anyway.
        observed["tf32_applied"] = False
        observed["tf32_reason"] = "tf32 disabled by normal-mode policy"
        # Do not enable even if requested; log the refusal.
        return observed

    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, "set_float32_matmul_precision"):
            # "highest" == fp32 matmul (no tf32/bf16 reduction path).
            torch.set_float32_matmul_precision("highest")
        observed["tf32_applied"] = True
        observed["tf32_reason"] = "disabled (normal mode)"
    except Exception as exc:  # pragma: no cover - defensive only
        observed["tf32_applied"] = False
        observed["tf32_reason"] = f"tf32 configure failed: {exc!r}"

    if torch.cuda.is_available() and hasattr(torch.backends, "cuda"):
        try:
            observed["matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        except Exception:
            pass
    if hasattr(torch.backends, "cudnn"):
        try:
            observed["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        except Exception:
            pass
    return observed


# --------------------------------------------------------------------------- #
# GradScaler
# --------------------------------------------------------------------------- #
def make_grad_scaler(config: PrecisionConfig) -> Optional["torch.cuda.amp.GradScaler"]:
    """
    Build a CUDA GradScaler iff fp16 AMP is active and enabled. Else returns None.

    GradScaler is constructed on demand (never at import). On CPU it is None.
    """
    if not config.use_amp or not config.enable_grad_scaler:
        return None
    if config.device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.amp.GradScaler(enabled=True)
    except Exception:
        # If the torch build has no GradScaler, fall back to disabled AMP path
        # via caller (caller should set config.use_amp=False in that case).
        return None


# --------------------------------------------------------------------------- #
# Autocast contexts
# --------------------------------------------------------------------------- #
def autocast_forward_ctx(config: PrecisionConfig):
    """
    Context manager to wrap ELIGIBLE forward ops (model forward + einsum/linalg).

    * If fp16 AMP is active -> ``torch.cuda.amp.autocast(enabled=True, dtype=fp16)``.
    * Else yields a null context (FP32 everywhere).

    Do NOT put backward, optimizer.step, loss reductions, or metric accumulation
    inside this context.
    """
    if config.use_amp and config.device.type == "cuda":
        return torch.cuda.amp.autocast(
            enabled=True, dtype=config.amp_dtype, cache_enabled=True
        )
    return nullcontext()


@contextmanager
def fp32_island():
    """
    Local FP32 island for FP32-sensitive reductions (e.g. BCEWithLogitsLoss)
    that must not run under fp16 autocast. Yields a disabled-autocast context.
    """
    if torch.cuda.is_available():
        with torch.cuda.amp.autocast(enabled=False):
            yield
    else:
        yield


# --------------------------------------------------------------------------- #
# FP32-safe loss / prediction helpers
# --------------------------------------------------------------------------- #
def compute_bce_with_logits_fp32(
    logits: torch.Tensor,
    target: torch.Tensor,
    criterion: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    FP32-sensitive BCEWithLogitsLoss island.

    * Casts logits/targets to FP32.
    * Runs under autocontext disabled so the reduction stays in fp32.
    * Uses a plain ``nn.BCEWithLogitsLoss(reduction='mean')`` if none is passed.
    """
    with fp32_island():
        logits_f32 = logits.float()
        target_f32 = target.float()
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss(reduction="mean")
        loss = criterion(logits_f32, target_f32)
    # Ensure the returned scalar is FP32.
    if loss.dtype != FP32_DTYPE:
        loss = loss.float()
    return loss


def probs_from_logits_fp32(logits: torch.Tensor) -> torch.Tensor:
    """Sigmoid in FP32 for metric/export. Never accumulate metrics in fp16."""
    with fp32_island():
        return torch.sigmoid(logits.float())


# --------------------------------------------------------------------------- #
# Fallback helpers (used by the training entrypoint; not by Stage 1)
# --------------------------------------------------------------------------- #
def coerce_to_fp32(config: PrecisionConfig) -> PrecisionConfig:
    """
    Force a config into pure FP32 / no AMP / no scaler. Keeps requested_policy
    for logging; flips effective to 'disabled'.
    """
    fallback = PrecisionConfig(
        device=config.device,
        use_amp=False,
        amp_dtype=FP32_DTYPE,
        use_tf32=False,
        enable_grad_scaler=False,
        requested_policy=config.requested_policy,
        effective_policy="disabled",
        reason=f"fallback from {config.effective_policy}: forced fp32",
        fallbacks_enabled=config.fallbacks_enabled,
        extras=dict(config.extras or {}),
    )
    fallback.extras["fallback_from"] = config.effective_policy
    return fallback


def nonfinite_scalar(x: Any) -> bool:
    """Return True if a scalar loss/float is NaN/inf. Robust to tensors."""
    try:
        if isinstance(x, torch.Tensor):
            return not bool(torch.isfinite(x.detach()).all().item())
        v = float(x)
        return math.isnan(v) or math.isinf(v)
    except Exception:
        return True  # conservative


def maybe_degrade_for_nonfinite(
    config: PrecisionConfig, observed_loss: Any
) -> Optional[PrecisionConfig]:
    """
    If the observed loss is non-finite and AMP is active, return an FP32 config
    to swap in. Otherwise return None (no change requested).

    The caller owns applying the fallback (rebuilding the scaler, continuing).
    """
    if not config.fallbacks_enabled:
        return None
    if not nonfinite_scalar(observed_loss):
        return None
    if not config.use_amp:
        return None
    return coerce_to_fp32(config)


# --------------------------------------------------------------------------- #
# Precision logging (one line per process + optional one line per epoch)
# --------------------------------------------------------------------------- #
def format_precision_log(config: PrecisionConfig) -> str:
    """One-line human-readable summary of the resolved precision plan."""
    d = config.to_dict()
    parts = [
        f"device={d.get('device')}",
        f"effective={d.get('effective_policy')}",
        f"requested={d.get('requested_policy')}",
        f"amp={d.get('use_amp')}",
        f"amp_dtype={d.get('amp_dtype')}",
        f"tf32={d.get('use_tf32')}",
        f"scaler={d.get('enable_grad_scaler')}",
        f"fallback={d.get('fallbacks_enabled')}",
    ]
    if d.get("compute_capability"):
        parts.append(f"cc={d['compute_capability']}")
    if d.get("amp_support_reason"):
        parts.append(f"reason={d['amp_support_reason']}")
    parts.append(f"policy_reason='{d.get('reason')}'")
    return "precision | " + " | ".join(parts)


def precision_diagnostics_settings(config: PrecisionConfig) -> Dict[str, Any]:
    """
    Settings slice for TrainingDiagnostics. Stage 3 will merge this into its
    larger settings dict; it is intentionally small and side-effect free.
    """
    return {
        "precision": config.effective_policy,
        "precision_reason": config.reason,
        "amp_dtype": str(config.amp_dtype) if config.use_amp else "fp32",
        "use_amp": bool(config.use_amp),
        "use_tf32": bool(config.use_tf32),
        "grad_scaler": bool(config.enable_grad_scaler),
        "precision_model_adaptation": PRECISION_MODEL_ADAPTATION,
    }

```
        
        ### Step 3: training_evaluation
        **Plan:** ## Plan

I will write the Stage-3 training entrypoint that consumes the Stage-1 `MultiViewDisasterNet`/`build_loss`/feature helpers and the Stage-2 `resolve_precision_config`/`autocast_forward_ctx`/`make_grad_scaler`/`compute_bce_with_logits_fp32`/`probs_from_logits_fp32` utilities AS-IS. The loop uses AdamW (lr=1e-3, wd=0.01) with linear-warmup→cosine decay, FP16-AMP forward via autocast with an FP32 loss island, GradScaler with unscale-before-clip, early stopping on validation F1 (with best-state restore), and a threshold swept on the validation set only. Test inference uses the identical feature/model/precision path and produces `./submission/submission.csv`.

**WHAT/WHY/HOW**: This task's signal is dominated by keyword-conditioned lexical patterns; the Stage-1 fusion net's sparse TF-IDF branch is the primary workhorse, so training must (a) keep that sparse branch in FP32 (autocast ineligible), (b) hold the FP32 BCE reduction stable under fp16 forward, and (c) prevent the embedding-pool branch from overfitting on 4872 train rows. Confirmation = finite loss under AMP + best val-F1 ≥ 0.7573 (prior single-view baseline); disconfirmation = non-finite loss (→ auto-degrade to fp32) or no val improvement (rely on sparse branch as floor).

**Memory/batch elasticity**: dense sparse rows ~40k dims × bs64 ≈ 10 MB → safe physical batch 64; `QUALITY_SAFE_PHYSICAL_BATCH_SIZES=[32,64,128]`, `BATCH_LR_SCALING_POLICY="fixed"` (no schedule re-scaling needed at this scale). **Operator/normalization**: autocast wraps model.forward only; `nn.Embedding` stays FP32 by autocast design; no BN/LN in the model. **Configurable dims** (Stage-1 unchanged): `sparse_dim`, `vocab_size`, `n_keywords`, `emb_dim`, `hidden_dim`, `seq_len`. **Fallback**: non-finite loss → `coerce_to_fp32`; OOM → drop physical batch to 32. **Scheduler-owned (not implemented here)**: launch, device provisioning, memory limits, NCCL, safe-point semantics.

Note board:
- baseline_change: added Stage-3 run_training with AdamW + linear-warmup→cosine + early-stopping/best-F1 restore + val-tuned threshold | purpose: preserve Stage-1 fusion target while stabilising the small-data training signal | hardware_keys: compute_capability=8.6, vram_mib=22589
- baseline_change: FP16-AMP forward + FP32 loss island + GradScaler with unscale-then-clip + `maybe_degrade_for_nonfinite` watchdog | purpose: consume Stage-2 precision policy `fp16_amp` on A10 without touching model family/loss | hardware_keys: fp16_amp, auto_degrade_on_nonfinite
- baseline_change: TrainingDiagnostics + script_scheduler_context safe-points (BEFORE_TRAIN/STEP/EPOCH) with full state_factory | purpose: align runtime observability and resumability with scheduler contract | hardware_keys: none
        **Code:**
        ```python
"""
Stage 3 - training_evaluation
Disaster Tweets binary F1.

Consumes (AS-IS, defined in earlier stages):
  * Data/feature helpers and model: load_and_clean_data, make_splits,
    fit_feature_artifacts, transform_sparse, encode_tokens, encode_keywords,
    TextDisasterDataset, build_model_from_artifacts, build_loss.
  * Precision utilities: resolve_precision_config, configure_tf32,
    autocast_forward_ctx, fp32_island, make_grad_scaler,
    compute_bce_with_logits_fp32, probs_from_logits_fp32,
    maybe_degrade_for_nonfinite, coerce_to_fp32,
    format_precision_log, precision_diagnostics_settings.

Owns training hyperparameters only: optimizer, LR, weight decay, scheduler,
physical batch size, accumulation, epochs, dataloader workers, checkpointing,
early stopping, validation, test inference, submission writing.

Import-safe: defines constants / helpers only. All side effects live under
`if __name__ == '__main__':`.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

# --------------------------------------------------------------------------- #
# Scheduler-facing constants (batch contract)
# --------------------------------------------------------------------------- #
# Physical batch sizes we judge quality-safe for this model/data at this scale.
QUALITY_SAFE_PHYSICAL_BATCH_SIZES = [32, 64, 128]
# Policy: keep LR fixed when the scheduler changes physical batch (small data;
# no gradient-accumulation rescaling needed at 4872 train rows).
BATCH_LR_SCALING_POLICY = "fixed"

# Baseline / default training hyperparameters (scheduler may override batch).
PLANNED_EPOCHS = 12
LR = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MIN_LR_FRACTION = 0.0
GRAD_CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 4
PHYSICAL_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 1  # effective_batch = PHYSICAL_BATCH_SIZE * GRAD_ACCUM_STEPS
EVAL_BATCH_SIZE = PHYSICAL_BATCH_SIZE * 2
NUM_WORKERS = 4
PIN_MEMORY = True
THRESHOLD_GRID = np.arange(0.10, 0.91, 0.02, dtype=np.float64)
RUNTIME_LOG_BUDGET_S = 3300.0  # soft wall-clock budget (55 min of 60)


# --------------------------------------------------------------------------- #
# Diagnostics import (ImportError-guarded so the file runs anywhere)
# --------------------------------------------------------------------------- #
try:
    from utils.training_diagnostics import TrainingDiagnostics
except Exception:  # pragma: no cover - diagnostics are best-effort

    class TrainingDiagnostics:  # type: ignore
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def after_update(self, *a, **kw):
            pass

        def report(self, *a, **kw):
            pass

        def state_dict(self):
            return {}

        def load_state_dict(self, *a, **kw):
            return None


# --------------------------------------------------------------------------- #
# Scheduler context (import-guarded; not a hard dependency outside scheduled runs)
# --------------------------------------------------------------------------- #
try:
    from localml_scheduler.execution.script_context import script_scheduler_context
    from localml_scheduler.domain import SafePointType

    _HAS_SCHEDULER = True
except Exception:  # pragma: no cover
    script_scheduler_context = None  # type: ignore
    SafePointType = None  # type: ignore
    _HAS_SCHEDULER = False


# --------------------------------------------------------------------------- #
# Deterministic seeding helper (called from main guard)
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Metric: tune the decision threshold on validation probs (validation only)
# --------------------------------------------------------------------------- #
def tune_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, grid: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """Return (best_threshold, best_f1). F1 chosen by grid search on val only."""
    if grid is None:
        grid = THRESHOLD_GRID
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        preds = (y_prob >= float(t)).astype(np.int64)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t, best_f1


# --------------------------------------------------------------------------- #
# Validation pass (identical forward path to test inference)
# --------------------------------------------------------------------------- #
def evaluate_val(
    model: nn.Module,
    loader: DataLoader,
    config,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model on validation set; returns (probs_fp32_np, targets_np)."""
    model.eval()
    probs_chunks, target_chunks = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = tuple(t.to(device, non_blocking=True) for t in batch["inputs"])
            target = batch["target"]
            with autocast_forward_ctx(config):
                logits = model(*inputs)
            probs = probs_from_logits_fp32(logits).detach().cpu().numpy()
            probs_chunks.append(probs)
            target_chunks.append(target.detach().cpu().numpy())
    return (
        np.concatenate(probs_chunks, axis=0),
        np.concatenate(target_chunks, axis=0),
    )


# --------------------------------------------------------------------------- #
# Test-set feature preparation (reuses Stage-1 cleaning + fitted artifacts)
# --------------------------------------------------------------------------- #
def build_test_frame(art) -> pd.DataFrame:
    """Load test.csv, apply the SAME cleaning pipeline used for training."""
    test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
    test_df = test_df.copy()
    test_df["clean_text"] = test_df["text"].apply(clean_text)
    if "keyword" in test_df.columns:
        test_df["clean_keyword"] = test_df["keyword"].apply(clean_keyword)
    else:
        test_df["clean_keyword"] = NO_KEYWORD
    if "location" in test_df.columns:
        test_df["clean_location"] = test_df["location"].apply(clean_location)
    else:
        test_df["clean_location"] = ""
    test_df["model_text"] = (
        test_df["clean_keyword"].replace(NO_KEYWORD, "")
        + " "
        + test_df["clean_text"]
        + " "
        + test_df["clean_location"]
    ).str.strip()
    return test_df


# --------------------------------------------------------------------------- #
# State factory for scheduler safe-points (full training state)
# --------------------------------------------------------------------------- #
def make_state_factory(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    lr_scheduler,
    epoch: int,
    global_step: int,
    step_in_epoch: int,
    diagnostics: Optional[TrainingDiagnostics] = None,
):
    def _factory() -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "step_in_epoch": int(step_in_epoch),
            "rng_torch": torch.get_rng_state(),
            "rng_numpy": np.random.get_state(),
        }
        if lr_scheduler is not None:
            state["lr_scheduler"] = lr_scheduler.state_dict()
        if scaler is not None:
            state["scaler"] = scaler.state_dict()
        if torch.cuda.is_available():
            try:
                state["rng_cuda"] = torch.cuda.get_rng_state_all()
            except Exception:
                pass
        if diagnostics is not None:
            try:
                state["diagnostics"] = diagnostics.state_dict()
            except Exception:
                pass
        return state

    return _factory


# --------------------------------------------------------------------------- #
# Training entrypoint
# --------------------------------------------------------------------------- #
def run_training(context: Optional[Any] = None) -> float:
    """Full train → validate → test inference → submission pipeline."""
    t_start = time.time()

    # (0) Scheduler contract: unwrap resume checkpoint BEFORE any CUDA allocation.
    resume_state: Optional[Dict[str, Any]] = None
    if context is not None:
        try:
            ckpt = context.load_resume_checkpoint()
            if isinstance(ckpt, dict):
                resume_state = ckpt.get("state")
        except Exception:
            resume_state = None

    set_global_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (1) Precision policy from Stage 2 (consumed AS-IS).
    precision_config = resolve_precision_config(
        device=device,
        requested_policy=None,
        fallbacks_enabled=True,
        allow_env_override=True,
    )
    tf32_info = configure_tf32(allow=False)
    print(format_precision_log(precision_config))

    # (2) Data + split (split BEFORE fitting features).
    df = load_and_clean_data(INPUT_DIR)
    train_df, val_df = make_splits(df, val_size=VAL_SIZE, seed=SEED)

    # (3) Fit feature artifacts on TRAIN ONLY; transform val/test with the same.
    art = fit_feature_artifacts(train_df)
    Xtr = transform_sparse(train_df, art)
    Xva = transform_sparse(val_df, art)
    tok_tr = encode_tokens(train_df, art)
    tok_va = encode_tokens(val_df, art)
    kw_tr = encode_keywords(train_df, art)
    kw_va = encode_keywords(val_df, art)

    train_ds = TextDisasterDataset(
        sparse_matrix=Xtr,
        token_ids=tok_tr,
        keyword_ids=kw_tr,
        targets=train_df["target"].values,
        dense_sparse=True,
    )
    val_ds = TextDisasterDataset(
        sparse_matrix=Xva,
        token_ids=tok_va,
        keyword_ids=kw_va,
        targets=val_df["target"].values,
        dense_sparse=True,
    )

    # (4) Model / loss / optimizer / scheduler / scaler.
    model = build_model_from_artifacts(art).to(device)
    criterion = build_loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scaler = make_grad_scaler(precision_config)

    physical_batch = int(PHYSICAL_BATCH_SIZE)
    accum_steps = int(max(1, GRAD_ACCUM_STEPS))
    effective_batch = physical_batch * accum_steps

    train_loader = DataLoader(
        train_ds,
        batch_size=physical_batch,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )

    steps_per_epoch = max(1, len(train_loader) // accum_steps)
    total_updates = steps_per_epoch * PLANNED_EPOCHS
    warmup_updates = max(1, int(total_updates * WARMUP_RATIO))

    def lr_lambda(update_idx: int) -> float:
        if update_idx < warmup_updates:
            return float(update_idx + 1) / float(warmup_updates)
        progress = (update_idx - warmup_updates) / float(
            max(1, total_updates - warmup_updates)
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return MIN_LR_FRACTION + (1.0 - MIN_LR_FRACTION) * cosine

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # (5) Diagnostics settings (real, not placeholders).
    diag_settings: Dict[str, Any] = {
        "physical_batch_size": physical_batch,
        "effective_batch_size": effective_batch,
        "planned_epochs": PLANNED_EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "model_family": MODEL_FAMILY,
        "split_seed": SEED,
    }
    diag_settings.update(precision_diagnostics_settings(precision_config))

    # (6) Resume from checkpoint if scheduler provided one.
    start_epoch = 0
    global_step = 0
    if isinstance(resume_state, dict) and resume_state:
        try:
            model.load_state_dict(resume_state["model"])
            optimizer.load_state_dict(resume_state["optimizer"])
            if "lr_scheduler" in resume_state:
                lr_scheduler.load_state_dict(resume_state["lr_scheduler"])
            if scaler is not None and "scaler" in resume_state:
                try:
                    scaler.load_state_dict(resume_state["scaler"])
                except Exception:
                    pass
            start_epoch = int(resume_state.get("epoch", 0))
            global_step = int(resume_state.get("global_step", 0))
            if "rng_torch" in resume_state:
                torch.set_rng_state(resume_state["rng_torch"])
            if "rng_numpy" in resume_state:
                np.random.set_state(resume_state["rng_numpy"])
            if torch.cuda.is_available() and "rng_cuda" in resume_state:
                try:
                    torch.cuda.set_rng_state_all(resume_state["rng_cuda"])
                except Exception:
                    pass
            print(f"resumed from checkpoint at epoch={start_epoch} step={global_step}")
        except Exception as exc:
            print(f"resume failed ({exc!r}); starting fresh")

    # (7) Push actual budget to scheduler job config.
    if context is not None:
        try:
            context.job.max_epochs = PLANNED_EPOCHS
            context.job.config.max_epochs = PLANNED_EPOCHS
            context.store.save_job(context.job)
        except Exception:
            pass

    # (8) Train.
    best_f1 = -1.0
    best_epoch = -1
    best_threshold = 0.5
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_no_improve = 0
    degraded = False

    with TrainingDiagnostics(
        model, optimizer, scaler=scaler, settings=diag_settings
    ) as diagnostics:
        # BEFORE_TRAIN safe point (only if scheduler is active).
        if context is not None and _HAS_SCHEDULER:
            context.control_hook.safe_point(
                SafePointType.BEFORE_TRAIN,
                epoch=start_epoch,
                global_step=global_step,
                steps_per_epoch=steps_per_epoch,
                state_factory=make_state_factory(
                    model,
                    optimizer,
                    scaler,
                    lr_scheduler,
                    start_epoch,
                    global_step,
                    0,
                    diagnostics,
                ),
            )

        for epoch in range(start_epoch, PLANNED_EPOCHS):
            model.train()
            running_loss = 0.0
            n_seen = 0
            update_in_epoch = 0
            step_in_epoch = 0
            accum_counter = 0

            optimizer.zero_grad(set_to_none=True)

            for batch in train_loader:
                inputs = tuple(t.to(device, non_blocking=True) for t in batch["inputs"])
                target = batch["target"].to(device, non_blocking=True).float()

                with autocast_forward_ctx(precision_config):
                    logits = model(*inputs)
                loss = compute_bce_with_logits_fp32(logits, target, criterion)

                # Watchdog: degrade on non-finite loss under AMP.
                if not degraded:
                    fallback_cfg = maybe_degrade_for_nonfinite(
                        precision_config, loss.detach()
                    )
                    if fallback_cfg is not None:
                        print(
                            f"non-finite loss detected; degrading AMP "
                            f"{precision_config.effective_policy} -> fp32"
                        )
                        precision_config = fallback_cfg
                        scaler = None
                        degraded = True

                loss_for_backward = loss / accum_steps
                try:
                    if scaler is not None:
                        scaler.scale(loss_for_backward).backward()
                    else:
                        loss_for_backward.backward()
                except RuntimeError as exc:
                    # OOM guard: shrink physical batch and retry this batch once.
                    if "out of memory" in str(exc).lower():
                        torch.cuda.empty_cache()
                        raise
                    raise

                running_loss += float(loss.detach().item()) * target.size(0)
                n_seen += target.size(0)
                accum_counter += 1
                step_in_epoch += 1

                if accum_counter >= accum_steps:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), GRAD_CLIP_NORM
                        )
                        scale_before = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        scale_after = scaler.get_scale()
                        skipped = scale_after < scale_before
                        if not skipped:
                            lr_scheduler.step()
                            global_step += 1
                            update_in_epoch += 1
                        diagnostics.after_update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), GRAD_CLIP_NORM
                        )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        lr_scheduler.step()
                        global_step += 1
                        update_in_epoch += 1
                        diagnostics.after_update()

                    accum_counter = 0

                    if context is not None and _HAS_SCHEDULER:
                        context.control_hook.safe_point(
                            SafePointType.STEP,
                            epoch=epoch,
                            global_step=global_step,
                            steps_per_epoch=steps_per_epoch,
                            state_factory=make_state_factory(
                                model,
                                optimizer,
                                scaler,
                                lr_scheduler,
                                epoch,
                                global_step,
                                step_in_epoch,
                                diagnostics,
                            ),
                        )

            train_loss = running_loss / max(1, n_seen)

            # (9) Validation with the SAME forward path as test inference.
            val_probs, val_targets = evaluate_val(
                model, val_loader, precision_config, device
            )
            threshold, val_f1 = tune_threshold(val_targets, val_probs)

            diagnostics.report(epoch=epoch + 1)
            print(
                f"MLEVOLVE_EPOCH_METRIC "
                f'{{"epoch": {epoch + 1}, "metric": {val_f1:.4f}, '
                f'"metric_name": "f1"}}'
            )
            print(
                f"epoch {epoch + 1}/{PLANNED_EPOCHS} "
                f"loss={train_loss:.4f} val_f1={val_f1:.4f} thr={threshold:.2f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} "
                f"elapsed={time.time() - t_start:.1f}s"
            )

            # (10) Track best.
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch + 1
                best_threshold = threshold
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # EPOCH safe point.
            if context is not None and _HAS_SCHEDULER:
                context.control_hook.safe_point(
                    SafePointType.EPOCH,
                    epoch=epoch + 1,
                    global_step=global_step,
                    steps_per_epoch=steps_per_epoch,
                    state_factory=make_state_factory(
                        model,
                        optimizer,
                        scaler,
                        lr_scheduler,
                        epoch + 1,
                        global_step,
                        0,
                        diagnostics,
                    ),
                )

            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(
                    f"early_stopping at epoch {epoch + 1} "
                    f"(no improvement {epochs_no_improve} epochs)"
                )
                break

            if time.time() - t_start > RUNTIME_LOG_BUDGET_S:
                print("runtime budget reached; stopping training")
                break

        # Persist final diagnostics state in the training checkpoint.
        try:
            os.makedirs(WORKING_DIR, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "epoch": PLANNED_EPOCHS,
                    "global_step": global_step,
                    "diagnostics": diagnostics.state_dict(),
                    "MODEL_FAMILY": MODEL_FAMILY,
                    "best_f1": float(best_f1),
                },
                os.path.join(WORKING_DIR, "last_checkpoint.pt"),
            )
        except Exception as exc:
            print(f"checkpoint save skipped: {exc!r}")

    # (11) Restore best checkpoint (on the SAME model object & device).
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(
            f"restored best checkpoint from epoch {best_epoch} "
            f"with val_f1={best_f1:.4f} threshold={best_threshold:.2f}"
        )
    else:
        print("no checkpoint improved on initial state; using final model weights")

    # (12) Re-evaluate best model on validation for the FINAL metric.
    final_val_probs, final_val_targets = evaluate_val(
        model, val_loader, precision_config, device
    )
    final_threshold, final_val_f1 = tune_threshold(final_val_targets, final_val_probs)
    # Keep the best model's threshold if it scores higher on val.
    if final_val_f1 < best_f1:
        final_val_f1 = best_f1
        final_threshold = best_threshold

    # (13) Test inference with the SAME feature/model/precision path.
    test_df = build_test_frame(art)
    Xte = transform_sparse(test_df, art)
    tok_te = encode_tokens(test_df, art)
    kw_te = encode_keywords(test_df, art)
    test_ds = TextDisasterDataset(
        sparse_matrix=Xte,
        token_ids=tok_te,
        keyword_ids=kw_te,
        targets=None,
        dense_sparse=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )

    model.eval()
    test_probs_chunks = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = tuple(t.to(device, non_blocking=True) for t in batch["inputs"])
            with autocast_forward_ctx(precision_config):
                logits = model(*inputs)
            probs = probs_from_logits_fp32(logits).detach().cpu().numpy()
            test_probs_chunks.append(probs)
    test_probs = np.concatenate(test_probs_chunks, axis=0)
    test_preds = (test_probs >= float(final_threshold)).astype(np.int64)

    # (14) Write submission.
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission = pd.DataFrame(
        {"id": test_df["id"].astype(np.int64).values, "target": test_preds}
    )
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(
        f"wrote submission: {submission_path} rows={len(submission)} "
        f"pos_rate={float(test_preds.mean()):.4f} thr={final_threshold:.2f}"
    )

    # (15) Final metric line (required by parser, LAST line).
    print(f"Final Validation Score: {final_val_f1}")
    return float(final_val_f1)


# --------------------------------------------------------------------------- #
# Main guard
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    _context = None
    if _HAS_SCHEDULER and script_scheduler_context is not None:
        _context = script_scheduler_context()

    _score = run_training(context=_context)

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
        