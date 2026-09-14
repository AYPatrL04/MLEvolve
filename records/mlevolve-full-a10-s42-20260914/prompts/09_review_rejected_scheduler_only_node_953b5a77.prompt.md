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
Design: 
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
    "loss": "binary cross-entropy with logits (BCEWithLogitsLoss)",
    "output_interface": "per-example probability via sigmoid at threshold 0.5 for F1",
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
    "optimizer": "AdamW",
    "scheduler": "linear warmup + linear decay",
    "batch_size_policy": "fixed",
    "dataloader_policy": "safe defaults; enable workers and pinned memory only when compatible",
    "fallbacks": [
      "if AMP unstable/slow, run FP32",
      "if transformer unavailable, TF-IDF + linear model",
      "if loss/gradients non-finite, disable autocast locally or globally"
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
Design: 
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
    "loss": "binary cross-entropy with logits (BCEWithLogitsLoss)",
    "output_interface": "per-example probability via sigmoid at threshold 0.5 for F1",
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
    "optimizer": "AdamW",
    "scheduler": "linear warmup + linear decay",
    "batch_size_policy": "fixed",
    "dataloader_policy": "safe defaults; enable workers and pinned memory only when compatible",
    "fallbacks": [
      "if AMP unstable/slow, run FP32",
      "if transformer unavailable, TF-IDF + linear model",
      "if loss/gradients non-finite, disable autocast locally or globally"
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
  - Stage 1 model-design / model_design: baseline_change=## Plan **WHAT makes this unique**: Disaster tweets are extremely short (~15 words) with high informality (slang, URLs, @-mentions, hashtags), strong class ambiguity (word "fire" can be disaster or metaphor), and two...; purpose=Define the Stage 1 candidate target for later precision and training stages.; hardware_keys=lstm_hidden, word_emb_dim, char_cnn_filters

# Previous steps
### model_design
**Plan:** ## Plan

**WHAT makes this unique**: Disaster tweets are extremely short (~15 words) with high informality (slang, URLs, @-mentions, hashtags), strong class ambiguity (word "fire" can be disaster or metaphor), and two sparse metadata fields (`keyword`, `location`) that prior attempts under-used. Models that only look at TF-IDF n-grams (attempt #1) or replace text with a from-scratch transformer (attempt #2) miss the sub-word structure and the meta evidence that disambiguate disaster vs. metaphor.

**WHY this approach**: I'll build a *three-view neural text classifier* — (a) a **word-level BiLSTM with additive attention pooling** to capture short-range argument structure and important token salience, (b) a **char-level multi-kernel CNN** to capture hashtags, misspellings, and morphological cues that word n-grams miss, and (c) a **meta view** that embeds `keyword` categorically and encodes `location` as a small bag-of-words + explicit presence flags, plus 16 hand-crafted numeric features (URL/@/# presence, caps ratio, punctuation, length). All three views are concatenated into an MLP head with BCEWithLogitsLoss. This is a fundamentally different hypothesis from the memory (TF-IDF+SparseMLP / hybrid transformer): it is an *inductive-bias-matched recurrent + convolutional + tabular fusion*, not a bag-of-ngrams or attention-only model.

**HOW to validate**: I'll fit vocab/stats only on training rows, split with the same 80/20 stratified seed-42 protocol, and evaluate hold-out F1 at the tuned threshold. Confirmation: char+word+meta views each contribute measurable F1 over the word-only ablation and exceed 0.7573. Falsification: if F1 sits near the word-only baseline, drop the char/meta views for Stage 3.

**Backend / elasticity / fallback**: single-process CUDA subprocess (no MPS); batch elasticity comes from Stage 3 (fixed batch policy per trace); the model keeps `lstm_hidden`, `word_emb_dim`, `char_cnn_filters`, sequence lengths and dropout configurable; fallback is FP32 CPU path in the preflight adapter. Scheduler owns device placement, thread allocation, and job launch. Loss is BCEWithLogitsLoss (trace). Output interface: per-example probability via sigmoid → tuned-threshold binary prediction.

**Note board:**
- baseline_change: replaced TF-IDF+MLP / from-scratch transformer with a **multi-view BiLSTM-attention + char-CNN + keyword/location-embedding fusion** classifier. | purpose: match the short, noisy, sub-word-rich structure of tweets and actually exploit `keyword`/`location` metadata, which the prior attempts ignored. | hardware_keys: none
- baseline_change: added explicit hand-crafted numeric features (URL/@/# presence, caps ratio, punctuation counts, length, unique-word ratio) and a location bag-of-words view. | purpose: give the model cheap disambiguators for metaphor vs. real disaster without inflating parameter count. | hardware_keys: none
- baseline_change: exposed `CandidateAdapter` that exercises the *real* model + loss on real train rows during CPU preflight, and confirmed BCEWithLogitsLoss + sigmoid@tuned-threshold output interface matches the F1 metric. | purpose: satisfy the CPU-preflight contract and keep Stage 2/3 aligned with Stage 1 family. | hardware_keys: none
**Code:**
```python
"""Stage 1 model-design: Disaster Tweets multi-view text classifier.

Defines data cleaning, feature/vocab construction, PyTorch model architecture,
loss, output interface, split helpers, and a CPU-safe CandidateAdapter.
No training/evaluation/submission side effects happen at import time.
"""

import os
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants / contract
# ---------------------------------------------------------------------------

MODEL_FAMILY = "multiview_text_cnn_bilstm_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
INPUT_DIR = "./input"

# Vocabulary / geometry (configurable, low-risk defaults for tweet length)
WORD_VOCAB_SIZE = 15000
MAX_WORD_LEN = 40
MAX_CHAR_LEN = 160

# Model dimensions
WORD_EMB_DIM = 128
CHAR_EMB_DIM = 48
LSTM_HIDDEN = 128
CHAR_CNN_FILTERS = 64
KEYWORD_EMB_DIM = 32
LOC_BOW_DIM = 40  # top-N location tokens used as BOW
NUMERIC_DIM = 16  # explicit hand-crafted numeric features
HIDDEN_DIM = 256
DROPOUT = 0.3

# Split
SPLIT_SEED = 42
VAL_SIZE = 0.2


# ---------------------------------------------------------------------------
# Char vocabulary (fixed, ASCII-only, import-safe)
# ---------------------------------------------------------------------------

_ASCII_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?'\"#@:/_-()"
# 0 = pad, 1 = unk
CHAR2IDX = {c: i + 2 for i, c in enumerate(_ASCII_CHARS)}
CHAR_VOCAB_SIZE = len(_ASCII_CHARS) + 2


# ---------------------------------------------------------------------------
# Text cleaning / tokenization
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NUM_RE = re.compile(r"\d+")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_NON_PRINT_RE = re.compile(r"[^\x20-\x7E]")
_WS_RE = re.compile(r"\s+")


def clean_text(s):
    """Normalize one tweet string into a canonical lowercased token stream."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _URL_RE.sub(" url ", s)
    s = _MENTION_RE.sub(" user ", s)
    s = _HASHTAG_RE.sub(r" hashtag_\1 ", s)
    s = _NUM_RE.sub(" num ", s)
    s = _NON_PRINT_RE.sub(" ", s)
    s = s.lower()
    s = _WS_RE.sub(" ", s).strip()
    return s


def tokenize(s):
    """Return word tokens from a (cleaned) tweet string."""
    return _WORD_RE.findall(s)


def numeric_features(text, location, keyword):
    """16 hand-crafted numeric features for the meta view."""
    t = str(text) if text is not None else ""
    loc = str(location) if location is not None else ""
    kw = str(keyword) if keyword is not None else ""

    n_chars = len(t)
    words = t.split()
    n_words = len(words)
    has_url = 1.0 if _URL_RE.search(t) else 0.0
    has_mention = 1.0 if _MENTION_RE.search(t) else 0.0
    has_hashtag = 1.0 if _HASHTAG_RE.search(t) else 0.0
    has_num = 1.0 if _NUM_RE.search(t) else 0.0
    caps = sum(1 for c in t if c.isupper())
    caps_ratio = caps / max(n_chars, 1)
    excl = t.count("!")
    quest = t.count("?")
    has_loc = 1.0 if loc.strip() else 0.0
    has_kw = 1.0 if kw.strip() else 0.0
    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
    unique_ratio = (
        (len(set(w.lower() for w in words)) / max(n_words, 1)) if words else 0.0
    )
    all_upper = 1.0 if (n_chars > 3 and t.isupper()) else 0.0

    return np.array(
        [
            min(n_chars, 100) / 100.0,
            min(n_words, 20) / 20.0,
            has_url,
            has_mention,
            has_hashtag,
            has_num,
            caps_ratio,
            min(excl, 5) / 5.0,
            min(quest, 5) / 5.0,
            has_loc,
            has_kw,
            avg_word_len / 8.0,
            unique_ratio,
            all_upper,
            min(n_chars, 50) / 50.0,
            min(n_words, 10) / 10.0,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Vocabulary / feature-state construction (fit on TRAIN ONLY)
# ---------------------------------------------------------------------------


def build_vocabs(df, word_vocab_size=WORD_VOCAB_SIZE, loc_topk=LOC_BOW_DIM):
    """Fit word vocab, keyword vocab and location BOW vocab on `df`."""
    word_counts = Counter()
    for t in df["text"].fillna("").astype(str):
        word_counts.update(tokenize(clean_text(t)))

    # keep place 0 for pad, 1 for unk
    reserve = 2
    most_common = word_counts.most_common(max(word_vocab_size - reserve, 1))
    word2idx = {w: i + reserve for i, (w, _) in enumerate(most_common)}

    keyword_values = df["keyword"].fillna("__missing__").astype(str).unique().tolist()
    # index 0 = missing keyword
    kw2idx = {k: i + 1 for i, k in enumerate(sorted(keyword_values))}

    loc_counter = Counter()
    for l in df["location"].fillna("").astype(str):
        for tok in tokenize(l.lower()):
            if len(tok) >= 3:
                loc_counter[tok] += 1
    # index 0 reserved for "location present" flag
    top_loc = [t for t, _ in loc_counter.most_common(max(loc_topk - 1, 1))]
    loc2idx = {t: i + 1 for i, t in enumerate(top_loc)}

    return {
        "word2idx": word2idx,
        "kw2idx": kw2idx,
        "loc2idx": loc2idx,
    }


def encode_text(df, state, max_word_len=MAX_WORD_LEN, max_char_len=MAX_CHAR_LEN):
    """Encode a DataFrame into the five model input tensors (as numpy ints/floats)."""
    word2idx = state["word2idx"]
    kw2idx = state["kw2idx"]
    loc2idx = state["loc2idx"]

    n = len(df)
    word_arr = np.zeros((n, max_word_len), dtype=np.int64)
    char_arr = np.zeros((n, max_char_len), dtype=np.int64)
    kw_arr = np.zeros((n,), dtype=np.int64)
    loc_arr = np.zeros((n, len(loc2idx) + 1), dtype=np.float32)
    num_arr = np.zeros((n, NUMERIC_DIM), dtype=np.float32)

    texts = df["text"].fillna("").astype(str).tolist()
    keywords = df["keyword"].fillna("__missing__").astype(str).tolist()
    locations = df["location"].fillna("").astype(str).tolist()

    for i in range(n):
        cleaned = clean_text(texts[i])
        toks = tokenize(cleaned)[:max_word_len]
        for j, tok in enumerate(toks):
            word_arr[i, j] = word2idx.get(tok, 1)

        chars = list(texts[i].lower())[:max_char_len]
        for j, ch in enumerate(chars):
            char_arr[i, j] = CHAR2IDX.get(ch, 1)

        kw_arr[i] = kw2idx.get(keywords[i], 0)

        loc_str = locations[i].lower()
        if loc_str.strip():
            loc_arr[i, 0] = 1.0
        for tok in tokenize(loc_str):
            if tok in loc2idx:
                loc_arr[i, loc2idx[tok]] = 1.0

        num_arr[i] = numeric_features(texts[i], locations[i], keywords[i])

    return {
        "word_ids": word_arr,
        "char_ids": char_arr,
        "kw_ids": kw_arr,
        "loc_vec": loc_arr,
        "num_feats": num_arr,
    }


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------


def make_split(df, val_size=VAL_SIZE, seed=SPLIT_SEED):
    """Stratified train/val indices. Uses labels of TRAIN only, no leakage."""
    idx = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_size,
        stratify=df["target"].values,
        random_state=seed,
    )
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MultiViewTextClassifier(nn.Module):
    """Three-view text classifier: word BiLSTM-attention + char CNN + meta MLP."""

    def __init__(
        self,
        word_vocab_size=WORD_VOCAB_SIZE,
        char_vocab_size=CHAR_VOCAB_SIZE,
        num_keywords=1000,
        num_loc_tokens=LOC_BOW_DIM + 1,
        numeric_dim=NUMERIC_DIM,
        word_emb_dim=WORD_EMB_DIM,
        char_emb_dim=CHAR_EMB_DIM,
        lstm_hidden=LSTM_HIDDEN,
        char_cnn_filters=CHAR_CNN_FILTERS,
        kw_emb_dim=KEYWORD_EMB_DIM,
        loc_dim=32,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()

        # --- word view ---
        self.word_emb = nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            word_emb_dim, lstm_hidden, batch_first=True, bidirectional=True
        )
        self.word_att = nn.Linear(lstm_hidden * 2, 1)

        # --- char view ---
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.char_convs = nn.ModuleList(
            [
                nn.Conv1d(char_emb_dim, char_cnn_filters, k, padding=k // 2)
                for k in (2, 3, 4, 5)
            ]
        )

        # --- meta view ---
        self.kw_emb = nn.Embedding(num_keywords + 1, kw_emb_dim, padding_idx=None)
        self.loc_proj = nn.Linear(num_loc_tokens, loc_dim)
        self.num_proj = nn.Linear(numeric_dim, 32)

        # --- head ---
        in_dim = lstm_hidden * 2 + char_cnn_filters * 4 + kw_emb_dim + loc_dim + 32
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(p.data)
            elif "bias" in name and p.dim() == 1:
                nn.init.zeros_(p.data)
            elif p.dim() >= 2:
                nn.init.xavier_uniform_(p.data)

    def forward(self, word_ids, char_ids, kw_ids, loc_vec, num_feats):
        # ---- word view ----
        w = self.word_emb(word_ids)  # B, L, E
        h, _ = self.lstm(w)  # B, L, 2H
        att_logits = self.word_att(h).squeeze(-1)  # B, L
        # mask padding positions (assume pad idx 0)
        pad_mask = word_ids != 0
        neg_inf = torch.finfo(att_logits.dtype).min
        att_logits = att_logits.masked_fill(~pad_mask, neg_inf)
        att = torch.softmax(att_logits, dim=1).unsqueeze(-1)  # B, L, 1
        word_feat = (h * att).sum(dim=1)  # B, 2H

        # ---- char view ----
        c = self.char_emb(char_ids).transpose(1, 2)  # B, Ec, Lc
        char_feats = []
        for conv in self.char_convs:
            x = F.relu(conv(c))
            x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
            char_feats.append(x)
        char_feat = torch.cat(char_feats, dim=1)  # B, 4*F

        # ---- meta view ----
        kw_feat = self.kw_emb(kw_ids)  # B, Dkw
        loc_feat = F.relu(self.loc_proj(loc_vec))  # B, Dloc
        num_feat = F.relu(self.num_proj(num_feats))  # B, 32
        meta_feat = torch.cat([kw_feat, loc_feat, num_feat], dim=1)

        z = torch.cat([word_feat, char_feat, meta_feat], dim=1)
        logits = self.head(z).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# Loss / criterion
# ---------------------------------------------------------------------------


def build_loss():
    """BCEWithLogitsLoss as directed by the pipeline decision trace."""
    return nn.BCEWithLogitsLoss()


def predict_proba(model, encoded, device, batch_size=64):
    """Real-model forward pass producing float32 probabilities (no grad)."""
    model.eval()
    n = len(encoded["word_ids"])
    probs = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            wi = torch.tensor(
                encoded["word_ids"][start:end], dtype=torch.long, device=device
            )
            ci = torch.tensor(
                encoded["char_ids"][start:end], dtype=torch.long, device=device
            )
            ki = torch.tensor(
                encoded["kw_ids"][start:end], dtype=torch.long, device=device
            )
            lv = torch.tensor(
                encoded["loc_vec"][start:end], dtype=torch.float32, device=device
            )
            nf = torch.tensor(
                encoded["num_feats"][start:end], dtype=torch.float32, device=device
            )
            logits = model(wi, ci, ki, lv, nf)
            probs[start:end] = torch.sigmoid(logits.float()).cpu().numpy()
    return probs


# ---------------------------------------------------------------------------
# CPU preflight adapter (real model + real loss on real data)
# ---------------------------------------------------------------------------


class CandidateAdapter:
    """No-arg adapter exercising the real model/loss/inputs on CPU."""

    _SUBSET_FOR_PREFLIGHT = 256

    def __init__(self):
        self._state = None
        self._encoded = None
        self._targets = None

    # -------- internal: build the same feature state the pipeline uses ----
    def _ensure(self):
        if self._state is not None:
            return
        input_dir = os.environ.get("MLEVOLVE_INPUT_DIR", INPUT_DIR)
        train_path = os.path.join(input_dir, "train.csv")
        df = pd.read_csv(train_path)
        state = build_vocabs(df)
        n = min(self._SUBSET_FOR_PREFLIGHT, len(df))
        small = df.iloc[:n].reset_index(drop=True)
        self._state = state
        self._encoded = encode_text(small, state)
        self._targets = small["target"].astype(np.float32).values

    # -------- construction --------
    def build_model(self, context=None):
        self._ensure()
        num_keywords = len(self._state["kw2idx"])
        num_loc_tokens = len(self._state["loc2idx"]) + 1
        model = MultiViewTextClassifier(
            word_vocab_size=WORD_VOCAB_SIZE,
            char_vocab_size=CHAR_VOCAB_SIZE,
            num_keywords=num_keywords,
            num_loc_tokens=num_loc_tokens,
            numeric_dim=NUMERIC_DIM,
            word_emb_dim=WORD_EMB_DIM,
            char_emb_dim=CHAR_EMB_DIM,
            lstm_hidden=LSTM_HIDDEN,
            char_cnn_filters=CHAR_CNN_FILTERS,
            kw_emb_dim=KEYWORD_EMB_DIM,
            loc_dim=32,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
        )
        return model

    def build_optimizer(self, model, context=None):
        # Only a placeholder for the preflight; Stage 3 owns the real optimizer.
        return torch.optim.AdamW(model.parameters(), lr=1e-3)

    def _make_batch(self, scenario, device):
        self._ensure()
        bs = int(scenario.get("batch_size", 8)) if scenario else 8
        bs = max(1, min(bs, len(self._targets)))
        idx = np.arange(bs)
        enc = self._encoded
        inputs = (
            torch.tensor(enc["word_ids"][idx], dtype=torch.long, device=device),
            torch.tensor(enc["char_ids"][idx], dtype=torch.long, device=device),
            torch.tensor(enc["kw_ids"][idx], dtype=torch.long, device=device),
            torch.tensor(enc["loc_vec"][idx], dtype=torch.float32, device=device),
            torch.tensor(enc["num_feats"][idx], dtype=torch.float32, device=device),
        )
        target = torch.tensor(self._targets[idx], dtype=torch.float32, device=device)
        return {"inputs": inputs, "target": target}

    def build_train_batch(self, scenario, device):
        return self._make_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._make_batch(scenario, device)

    def training_step(self, model, batch, context=None):
        inputs = batch["inputs"]
        target = batch["target"]
        logits = model(*inputs)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        return loss

    def validation_step(self, model, batch, context=None):
        inputs = batch["inputs"]
        target = batch["target"]
        logits = model(*inputs)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        return {"loss": loss, "probs": probs, "target": target}


if __name__ == "__main__":
    # Stage 1 owns only definitions; training is driven by Stage 3.
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
Design: 
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
    "loss": "binary cross-entropy with logits (BCEWithLogitsLoss)",
    "output_interface": "per-example probability via sigmoid at threshold 0.5 for F1",
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
    "optimizer": "AdamW",
    "scheduler": "linear warmup + linear decay",
    "batch_size_policy": "fixed",
    "dataloader_policy": "safe defaults; enable workers and pinned memory only when compatible",
    "fallbacks": [
      "if AMP unstable/slow, run FP32",
      "if transformer unavailable, TF-IDF + linear model",
      "if loss/gradients non-finite, disable autocast locally or globally"
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
  - Stage 1 model-design / model_design: baseline_change=## Plan **WHAT makes this unique**: Disaster tweets are extremely short (~15 words) with high informality (slang, URLs, @-mentions, hashtags), strong class ambiguity (word "fire" can be disaster or metaphor), and two...; purpose=Define the Stage 1 candidate target for later precision and training stages.; hardware_keys=lstm_hidden, word_emb_dim, char_cnn_filters
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=Added a selective FP16 AMP utility layer (`PrecisionConfig`, `resolve_precision_config`, `autocast_forward`, `compute_bce_loss`, `make_grad_scaler`, `precision_state_dict`/`load_precision_state_dict`, `describe_precis...; purpose=implement the trace `datatype_precision.precision_policy="fp16_amp"` while keeping FP32 parameters, integer inputs, and sigmoid→tuned-threshold output for F1 intact, and preserve the Stage 1 multi-view target.; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=Added explicit `disable_tf32()` helper (invoked only at run-time by Stage 3) and a runtime fallback path (`degrade_to_fp32`) that flips AMP off if loss/grads are non-finite or CUDA is unavailable.; purpose=match the trace fallback policy "fall back to fp32 or disabled AMP if unsupported/unstable" and the normal-mode rule forbidding TF32 on Ampere.; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=AMP/scaling region is scoped to forward+loss only; backward/step remain outside autocast with `GradScaler.unscale_` before clipping, and metrics/export stay FP32.; purpose=prevent the FP16 instability that memory attempts #1/#2 risked, matching the precision-required "high-precision islands" guidance.; hardware_keys=none

# Previous steps
### model_design
**Plan:** ## Plan

**WHAT makes this unique**: Disaster tweets are extremely short (~15 words) with high informality (slang, URLs, @-mentions, hashtags), strong class ambiguity (word "fire" can be disaster or metaphor), and two sparse metadata fields (`keyword`, `location`) that prior attempts under-used. Models that only look at TF-IDF n-grams (attempt #1) or replace text with a from-scratch transformer (attempt #2) miss the sub-word structure and the meta evidence that disambiguate disaster vs. metaphor.

**WHY this approach**: I'll build a *three-view neural text classifier* — (a) a **word-level BiLSTM with additive attention pooling** to capture short-range argument structure and important token salience, (b) a **char-level multi-kernel CNN** to capture hashtags, misspellings, and morphological cues that word n-grams miss, and (c) a **meta view** that embeds `keyword` categorically and encodes `location` as a small bag-of-words + explicit presence flags, plus 16 hand-crafted numeric features (URL/@/# presence, caps ratio, punctuation, length). All three views are concatenated into an MLP head with BCEWithLogitsLoss. This is a fundamentally different hypothesis from the memory (TF-IDF+SparseMLP / hybrid transformer): it is an *inductive-bias-matched recurrent + convolutional + tabular fusion*, not a bag-of-ngrams or attention-only model.

**HOW to validate**: I'll fit vocab/stats only on training rows, split with the same 80/20 stratified seed-42 protocol, and evaluate hold-out F1 at the tuned threshold. Confirmation: char+word+meta views each contribute measurable F1 over the word-only ablation and exceed 0.7573. Falsification: if F1 sits near the word-only baseline, drop the char/meta views for Stage 3.

**Backend / elasticity / fallback**: single-process CUDA subprocess (no MPS); batch elasticity comes from Stage 3 (fixed batch policy per trace); the model keeps `lstm_hidden`, `word_emb_dim`, `char_cnn_filters`, sequence lengths and dropout configurable; fallback is FP32 CPU path in the preflight adapter. Scheduler owns device placement, thread allocation, and job launch. Loss is BCEWithLogitsLoss (trace). Output interface: per-example probability via sigmoid → tuned-threshold binary prediction.

**Note board:**
- baseline_change: replaced TF-IDF+MLP / from-scratch transformer with a **multi-view BiLSTM-attention + char-CNN + keyword/location-embedding fusion** classifier. | purpose: match the short, noisy, sub-word-rich structure of tweets and actually exploit `keyword`/`location` metadata, which the prior attempts ignored. | hardware_keys: none
- baseline_change: added explicit hand-crafted numeric features (URL/@/# presence, caps ratio, punctuation counts, length, unique-word ratio) and a location bag-of-words view. | purpose: give the model cheap disambiguators for metaphor vs. real disaster without inflating parameter count. | hardware_keys: none
- baseline_change: exposed `CandidateAdapter` that exercises the *real* model + loss on real train rows during CPU preflight, and confirmed BCEWithLogitsLoss + sigmoid@tuned-threshold output interface matches the F1 metric. | purpose: satisfy the CPU-preflight contract and keep Stage 2/3 aligned with Stage 1 family. | hardware_keys: none
**Code:**
```python
"""Stage 1 model-design: Disaster Tweets multi-view text classifier.

Defines data cleaning, feature/vocab construction, PyTorch model architecture,
loss, output interface, split helpers, and a CPU-safe CandidateAdapter.
No training/evaluation/submission side effects happen at import time.
"""

import os
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants / contract
# ---------------------------------------------------------------------------

MODEL_FAMILY = "multiview_text_cnn_bilstm_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
INPUT_DIR = "./input"

# Vocabulary / geometry (configurable, low-risk defaults for tweet length)
WORD_VOCAB_SIZE = 15000
MAX_WORD_LEN = 40
MAX_CHAR_LEN = 160

# Model dimensions
WORD_EMB_DIM = 128
CHAR_EMB_DIM = 48
LSTM_HIDDEN = 128
CHAR_CNN_FILTERS = 64
KEYWORD_EMB_DIM = 32
LOC_BOW_DIM = 40  # top-N location tokens used as BOW
NUMERIC_DIM = 16  # explicit hand-crafted numeric features
HIDDEN_DIM = 256
DROPOUT = 0.3

# Split
SPLIT_SEED = 42
VAL_SIZE = 0.2


# ---------------------------------------------------------------------------
# Char vocabulary (fixed, ASCII-only, import-safe)
# ---------------------------------------------------------------------------

_ASCII_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?'\"#@:/_-()"
# 0 = pad, 1 = unk
CHAR2IDX = {c: i + 2 for i, c in enumerate(_ASCII_CHARS)}
CHAR_VOCAB_SIZE = len(_ASCII_CHARS) + 2


# ---------------------------------------------------------------------------
# Text cleaning / tokenization
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NUM_RE = re.compile(r"\d+")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_NON_PRINT_RE = re.compile(r"[^\x20-\x7E]")
_WS_RE = re.compile(r"\s+")


def clean_text(s):
    """Normalize one tweet string into a canonical lowercased token stream."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _URL_RE.sub(" url ", s)
    s = _MENTION_RE.sub(" user ", s)
    s = _HASHTAG_RE.sub(r" hashtag_\1 ", s)
    s = _NUM_RE.sub(" num ", s)
    s = _NON_PRINT_RE.sub(" ", s)
    s = s.lower()
    s = _WS_RE.sub(" ", s).strip()
    return s


def tokenize(s):
    """Return word tokens from a (cleaned) tweet string."""
    return _WORD_RE.findall(s)


def numeric_features(text, location, keyword):
    """16 hand-crafted numeric features for the meta view."""
    t = str(text) if text is not None else ""
    loc = str(location) if location is not None else ""
    kw = str(keyword) if keyword is not None else ""

    n_chars = len(t)
    words = t.split()
    n_words = len(words)
    has_url = 1.0 if _URL_RE.search(t) else 0.0
    has_mention = 1.0 if _MENTION_RE.search(t) else 0.0
    has_hashtag = 1.0 if _HASHTAG_RE.search(t) else 0.0
    has_num = 1.0 if _NUM_RE.search(t) else 0.0
    caps = sum(1 for c in t if c.isupper())
    caps_ratio = caps / max(n_chars, 1)
    excl = t.count("!")
    quest = t.count("?")
    has_loc = 1.0 if loc.strip() else 0.0
    has_kw = 1.0 if kw.strip() else 0.0
    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
    unique_ratio = (
        (len(set(w.lower() for w in words)) / max(n_words, 1)) if words else 0.0
    )
    all_upper = 1.0 if (n_chars > 3 and t.isupper()) else 0.0

    return np.array(
        [
            min(n_chars, 100) / 100.0,
            min(n_words, 20) / 20.0,
            has_url,
            has_mention,
            has_hashtag,
            has_num,
            caps_ratio,
            min(excl, 5) / 5.0,
            min(quest, 5) / 5.0,
            has_loc,
            has_kw,
            avg_word_len / 8.0,
            unique_ratio,
            all_upper,
            min(n_chars, 50) / 50.0,
            min(n_words, 10) / 10.0,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Vocabulary / feature-state construction (fit on TRAIN ONLY)
# ---------------------------------------------------------------------------


def build_vocabs(df, word_vocab_size=WORD_VOCAB_SIZE, loc_topk=LOC_BOW_DIM):
    """Fit word vocab, keyword vocab and location BOW vocab on `df`."""
    word_counts = Counter()
    for t in df["text"].fillna("").astype(str):
        word_counts.update(tokenize(clean_text(t)))

    # keep place 0 for pad, 1 for unk
    reserve = 2
    most_common = word_counts.most_common(max(word_vocab_size - reserve, 1))
    word2idx = {w: i + reserve for i, (w, _) in enumerate(most_common)}

    keyword_values = df["keyword"].fillna("__missing__").astype(str).unique().tolist()
    # index 0 = missing keyword
    kw2idx = {k: i + 1 for i, k in enumerate(sorted(keyword_values))}

    loc_counter = Counter()
    for l in df["location"].fillna("").astype(str):
        for tok in tokenize(l.lower()):
            if len(tok) >= 3:
                loc_counter[tok] += 1
    # index 0 reserved for "location present" flag
    top_loc = [t for t, _ in loc_counter.most_common(max(loc_topk - 1, 1))]
    loc2idx = {t: i + 1 for i, t in enumerate(top_loc)}

    return {
        "word2idx": word2idx,
        "kw2idx": kw2idx,
        "loc2idx": loc2idx,
    }


def encode_text(df, state, max_word_len=MAX_WORD_LEN, max_char_len=MAX_CHAR_LEN):
    """Encode a DataFrame into the five model input tensors (as numpy ints/floats)."""
    word2idx = state["word2idx"]
    kw2idx = state["kw2idx"]
    loc2idx = state["loc2idx"]

    n = len(df)
    word_arr = np.zeros((n, max_word_len), dtype=np.int64)
    char_arr = np.zeros((n, max_char_len), dtype=np.int64)
    kw_arr = np.zeros((n,), dtype=np.int64)
    loc_arr = np.zeros((n, len(loc2idx) + 1), dtype=np.float32)
    num_arr = np.zeros((n, NUMERIC_DIM), dtype=np.float32)

    texts = df["text"].fillna("").astype(str).tolist()
    keywords = df["keyword"].fillna("__missing__").astype(str).tolist()
    locations = df["location"].fillna("").astype(str).tolist()

    for i in range(n):
        cleaned = clean_text(texts[i])
        toks = tokenize(cleaned)[:max_word_len]
        for j, tok in enumerate(toks):
            word_arr[i, j] = word2idx.get(tok, 1)

        chars = list(texts[i].lower())[:max_char_len]
        for j, ch in enumerate(chars):
            char_arr[i, j] = CHAR2IDX.get(ch, 1)

        kw_arr[i] = kw2idx.get(keywords[i], 0)

        loc_str = locations[i].lower()
        if loc_str.strip():
            loc_arr[i, 0] = 1.0
        for tok in tokenize(loc_str):
            if tok in loc2idx:
                loc_arr[i, loc2idx[tok]] = 1.0

        num_arr[i] = numeric_features(texts[i], locations[i], keywords[i])

    return {
        "word_ids": word_arr,
        "char_ids": char_arr,
        "kw_ids": kw_arr,
        "loc_vec": loc_arr,
        "num_feats": num_arr,
    }


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------


def make_split(df, val_size=VAL_SIZE, seed=SPLIT_SEED):
    """Stratified train/val indices. Uses labels of TRAIN only, no leakage."""
    idx = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_size,
        stratify=df["target"].values,
        random_state=seed,
    )
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MultiViewTextClassifier(nn.Module):
    """Three-view text classifier: word BiLSTM-attention + char CNN + meta MLP."""

    def __init__(
        self,
        word_vocab_size=WORD_VOCAB_SIZE,
        char_vocab_size=CHAR_VOCAB_SIZE,
        num_keywords=1000,
        num_loc_tokens=LOC_BOW_DIM + 1,
        numeric_dim=NUMERIC_DIM,
        word_emb_dim=WORD_EMB_DIM,
        char_emb_dim=CHAR_EMB_DIM,
        lstm_hidden=LSTM_HIDDEN,
        char_cnn_filters=CHAR_CNN_FILTERS,
        kw_emb_dim=KEYWORD_EMB_DIM,
        loc_dim=32,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()

        # --- word view ---
        self.word_emb = nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            word_emb_dim, lstm_hidden, batch_first=True, bidirectional=True
        )
        self.word_att = nn.Linear(lstm_hidden * 2, 1)

        # --- char view ---
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.char_convs = nn.ModuleList(
            [
                nn.Conv1d(char_emb_dim, char_cnn_filters, k, padding=k // 2)
                for k in (2, 3, 4, 5)
            ]
        )

        # --- meta view ---
        self.kw_emb = nn.Embedding(num_keywords + 1, kw_emb_dim, padding_idx=None)
        self.loc_proj = nn.Linear(num_loc_tokens, loc_dim)
        self.num_proj = nn.Linear(numeric_dim, 32)

        # --- head ---
        in_dim = lstm_hidden * 2 + char_cnn_filters * 4 + kw_emb_dim + loc_dim + 32
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(p.data)
            elif "bias" in name and p.dim() == 1:
                nn.init.zeros_(p.data)
            elif p.dim() >= 2:
                nn.init.xavier_uniform_(p.data)

    def forward(self, word_ids, char_ids, kw_ids, loc_vec, num_feats):
        # ---- word view ----
        w = self.word_emb(word_ids)  # B, L, E
        h, _ = self.lstm(w)  # B, L, 2H
        att_logits = self.word_att(h).squeeze(-1)  # B, L
        # mask padding positions (assume pad idx 0)
        pad_mask = word_ids != 0
        neg_inf = torch.finfo(att_logits.dtype).min
        att_logits = att_logits.masked_fill(~pad_mask, neg_inf)
        att = torch.softmax(att_logits, dim=1).unsqueeze(-1)  # B, L, 1
        word_feat = (h * att).sum(dim=1)  # B, 2H

        # ---- char view ----
        c = self.char_emb(char_ids).transpose(1, 2)  # B, Ec, Lc
        char_feats = []
        for conv in self.char_convs:
            x = F.relu(conv(c))
            x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
            char_feats.append(x)
        char_feat = torch.cat(char_feats, dim=1)  # B, 4*F

        # ---- meta view ----
        kw_feat = self.kw_emb(kw_ids)  # B, Dkw
        loc_feat = F.relu(self.loc_proj(loc_vec))  # B, Dloc
        num_feat = F.relu(self.num_proj(num_feats))  # B, 32
        meta_feat = torch.cat([kw_feat, loc_feat, num_feat], dim=1)

        z = torch.cat([word_feat, char_feat, meta_feat], dim=1)
        logits = self.head(z).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# Loss / criterion
# ---------------------------------------------------------------------------


def build_loss():
    """BCEWithLogitsLoss as directed by the pipeline decision trace."""
    return nn.BCEWithLogitsLoss()


def predict_proba(model, encoded, device, batch_size=64):
    """Real-model forward pass producing float32 probabilities (no grad)."""
    model.eval()
    n = len(encoded["word_ids"])
    probs = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            wi = torch.tensor(
                encoded["word_ids"][start:end], dtype=torch.long, device=device
            )
            ci = torch.tensor(
                encoded["char_ids"][start:end], dtype=torch.long, device=device
            )
            ki = torch.tensor(
                encoded["kw_ids"][start:end], dtype=torch.long, device=device
            )
            lv = torch.tensor(
                encoded["loc_vec"][start:end], dtype=torch.float32, device=device
            )
            nf = torch.tensor(
                encoded["num_feats"][start:end], dtype=torch.float32, device=device
            )
            logits = model(wi, ci, ki, lv, nf)
            probs[start:end] = torch.sigmoid(logits.float()).cpu().numpy()
    return probs


# ---------------------------------------------------------------------------
# CPU preflight adapter (real model + real loss on real data)
# ---------------------------------------------------------------------------


class CandidateAdapter:
    """No-arg adapter exercising the real model/loss/inputs on CPU."""

    _SUBSET_FOR_PREFLIGHT = 256

    def __init__(self):
        self._state = None
        self._encoded = None
        self._targets = None

    # -------- internal: build the same feature state the pipeline uses ----
    def _ensure(self):
        if self._state is not None:
            return
        input_dir = os.environ.get("MLEVOLVE_INPUT_DIR", INPUT_DIR)
        train_path = os.path.join(input_dir, "train.csv")
        df = pd.read_csv(train_path)
        state = build_vocabs(df)
        n = min(self._SUBSET_FOR_PREFLIGHT, len(df))
        small = df.iloc[:n].reset_index(drop=True)
        self._state = state
        self._encoded = encode_text(small, state)
        self._targets = small["target"].astype(np.float32).values

    # -------- construction --------
    def build_model(self, context=None):
        self._ensure()
        num_keywords = len(self._state["kw2idx"])
        num_loc_tokens = len(self._state["loc2idx"]) + 1
        model = MultiViewTextClassifier(
            word_vocab_size=WORD_VOCAB_SIZE,
            char_vocab_size=CHAR_VOCAB_SIZE,
            num_keywords=num_keywords,
            num_loc_tokens=num_loc_tokens,
            numeric_dim=NUMERIC_DIM,
            word_emb_dim=WORD_EMB_DIM,
            char_emb_dim=CHAR_EMB_DIM,
            lstm_hidden=LSTM_HIDDEN,
            char_cnn_filters=CHAR_CNN_FILTERS,
            kw_emb_dim=KEYWORD_EMB_DIM,
            loc_dim=32,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
        )
        return model

    def build_optimizer(self, model, context=None):
        # Only a placeholder for the preflight; Stage 3 owns the real optimizer.
        return torch.optim.AdamW(model.parameters(), lr=1e-3)

    def _make_batch(self, scenario, device):
        self._ensure()
        bs = int(scenario.get("batch_size", 8)) if scenario else 8
        bs = max(1, min(bs, len(self._targets)))
        idx = np.arange(bs)
        enc = self._encoded
        inputs = (
            torch.tensor(enc["word_ids"][idx], dtype=torch.long, device=device),
            torch.tensor(enc["char_ids"][idx], dtype=torch.long, device=device),
            torch.tensor(enc["kw_ids"][idx], dtype=torch.long, device=device),
            torch.tensor(enc["loc_vec"][idx], dtype=torch.float32, device=device),
            torch.tensor(enc["num_feats"][idx], dtype=torch.float32, device=device),
        )
        target = torch.tensor(self._targets[idx], dtype=torch.float32, device=device)
        return {"inputs": inputs, "target": target}

    def build_train_batch(self, scenario, device):
        return self._make_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._make_batch(scenario, device)

    def training_step(self, model, batch, context=None):
        inputs = batch["inputs"]
        target = batch["target"]
        logits = model(*inputs)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        return loss

    def validation_step(self, model, batch, context=None):
        inputs = batch["inputs"]
        target = batch["target"]
        logits = model(*inputs)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        return {"loss": loss, "probs": probs, "target": target}


if __name__ == "__main__":
    # Stage 1 owns only definitions; training is driven by Stage 3.
    pass

```

### datatype_precision
**Plan:** ## Plan

I will implement the Stage 2 `datatype_precision` layer for the Stage 1 `multiview_text_cnn_bilstm_v1` classifier, honoring the trace's `precision_policy="fp16_amp"` and conservative fallback. The policy is a **selective FP16 AMP** design: FP32 parameters/inputs/targets remain FP32, autocast is applied only around the eligible forward pass, the BCEWithLogitsLoss is computed inside autocast (it is autocast-safe), backward and optimizer step happen outside autocast, and a `GradScaler` (only when CUDA+FP16) unscales before clipping. TF32 is explicitly disabled, and a CPU-only preflight uses pure FP32 with AMP/scaling off. Precision-required adaptation is `none` — I only add thin, optional wrappers (a `precision_forward`/`compute_loss` helper) that preserve the model family, loss, features and output interface.

## Note board:
- baseline_change: Added a selective FP16 AMP utility layer (`PrecisionConfig`, `resolve_precision_config`, `autocast_forward`, `compute_bce_loss`, `make_grad_scaler`, `precision_state_dict`/`load_precision_state_dict`, `describe_precision`) that wraps the Stage 1 model's forward and loss without altering architecture, features, or output interface. | purpose: implement the trace `datatype_precision.precision_policy="fp16_amp"` while keeping FP32 parameters, integer inputs, and sigmoid→tuned-threshold output for F1 intact, and preserve the Stage 1 multi-view target. | hardware_keys: none
- baseline_change: Added explicit `disable_tf32()` helper (invoked only at run-time by Stage 3) and a runtime fallback path (`degrade_to_fp32`) that flips AMP off if loss/grads are non-finite or CUDA is unavailable. | purpose: match the trace fallback policy "fall back to fp32 or disabled AMP if unsupported/unstable" and the normal-mode rule forbidding TF32 on Ampere. | hardware_keys: none
- baseline_change: AMP/scaling region is scoped to forward+loss only; backward/step remain outside autocast with `GradScaler.unscale_` before clipping, and metrics/export stay FP32. | purpose: prevent the FP16 instability that memory attempts #1/#2 risked, matching the precision-required "high-precision islands" guidance. | hardware_keys: none
**Code:**
```python
"""Stage 2 datatype/precision utilities for the Disaster Tweets classifier.

Defines device/dtype policy, AMP/autocast region helpers, GradScaler factory,
loss computation, TF32 disabling, fallback, and precision logging.

Import-safe: NO data access, NO CUDA configuration, NO logging setup at import.
All side effects must be triggered explicitly from the Stage 3 training entry.
Preserves Stage 1 model family (multiview_text_cnn_bilstm_v1), loss
(BCEWithLogitsLoss), data features, and output interface (sigmoid probability).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Policy constants (from pipeline decision trace / hardware context)
# ---------------------------------------------------------------------------

# Allowed: "fp32" | "fp16_amp"  (bf16 / TF32 / TE recipes are NOT allowed in
# the current normal mode for compute capability 8.6.)
DEFAULT_PRECISION_POLICY = "fp16_amp"

# AMP autocast dtype candidate (only fp16 is permitted in normal mode).
AMP_DTYPE = torch.float16

# TF32 must be disabled in normal mode.
USE_TF32 = False

# Number of consecutive non-finite steps before a hard fallback to FP32.
_NONFINITE_PATIENCE = 3


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class PrecisionConfig:
    """Resolved precision policy for the current run.

    Attributes
    ----------
    policy          : "fp32" or "fp16_amp" (the requested policy).
    resolved_policy : the policy actually in effect after fallbacks.
    device          : torch.device the model/batches live on.
    amp_dtype       : autocast dtype when AMP is enabled, else None.
    use_amp         : whether CUDA autocast + GradScaler should be used.
    use_tf32        : always False in normal mode (recorded for logging).
    fallback_events : list of human-readable fallback reasons, in order.
    """

    policy: str = DEFAULT_PRECISION_POLICY
    resolved_policy: str = "fp32"
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    amp_dtype: Optional[torch.dtype] = None
    use_amp: bool = False
    use_tf32: bool = False
    fallback_events: list = field(default_factory=list)

    @property
    def is_amp(self) -> bool:
        return bool(self.use_amp and self.amp_dtype is not None)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available device without mutating global state."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# TF32 disabling (must be called at run-time, not import time)
# ---------------------------------------------------------------------------


def disable_tf32() -> None:
    """Explicitly disable TF32 matmul/cudnn paths (normal mode requirement)."""
    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
    except Exception:
        # Non-fatal: fall through, the policy is still recorded.
        pass


# ---------------------------------------------------------------------------
# Config resolution + fallback
# ---------------------------------------------------------------------------


def resolve_precision_config(
    policy: Optional[str] = None,
    device: Optional[torch.device] = None,
    prefer_cuda: bool = True,
) -> PrecisionConfig:
    """Resolve the effective precision policy from an optional requested policy.

    Applies the trace fallback rule: fall back to fp32 / disabled AMP when the
    selected precision path is unsupported on the current device.
    """
    requested = (policy or DEFAULT_PRECISION_POLICY).lower()
    dev = device if device is not None else resolve_device(prefer_cuda=prefer_cuda)

    cfg = PrecisionConfig(policy=requested, device=dev, use_tf32=False)
    cfg.fallback_events = []

    if requested not in ("fp32", "fp16_amp"):
        cfg.fallback_events.append(f"unknown_policy:{requested}->fp32")
        requested = "fp32"
        cfg.policy = "fp32"

    if requested == "fp16_amp":
        if dev.type == "cuda" and torch.cuda.is_available():
            cfg.resolved_policy = "fp16_amp"
            cfg.amp_dtype = AMP_DTYPE
            cfg.use_amp = True
        else:
            cfg.resolved_policy = "fp32"
            cfg.amp_dtype = None
            cfg.use_amp = False
            cfg.fallback_events.append("no_cuda->fp32")
    else:
        cfg.resolved_policy = "fp32"
        cfg.amp_dtype = None
        cfg.use_amp = False

    return cfg


def degrade_to_fp32(
    cfg: PrecisionConfig,
    reason: str = "nonfinite_or_unstable",
    device: Optional[torch.device] = None,
) -> PrecisionConfig:
    """Return a new PrecisionConfig switched to FP32 (idempotent fallback)."""
    new_cfg = PrecisionConfig(
        policy="fp32",
        resolved_policy="fp32",
        device=cfg.device if device is None else device,
        amp_dtype=None,
        use_amp=False,
        use_tf32=False,
    )
    new_cfg.fallback_events = list(cfg.fallback_events) + [f"degrade:{reason}"]
    return new_cfg


# ---------------------------------------------------------------------------
# Autocast / forward helpers
# ---------------------------------------------------------------------------


class autocast_forward:
    """Context manager wrapping ONLY the eligible forward ops in CUDA autocast.

    Usage inside training_step / validation_step:

        with autocast_forward(cfg):
            logits = model(*inputs)
            loss = compute_bce_loss(logits, targets)   # autocast-safe

    Backward and optimizer step MUST happen outside this region.
    """

    __slots__ = ("_cfg", "_enabled")

    def __init__(self, cfg: PrecisionConfig, enabled: Optional[bool] = None):
        self._cfg = cfg
        if enabled is None:
            self._enabled = bool(cfg is not None and cfg.is_amp)
        else:
            self._enabled = bool(enabled)

    def __enter__(self):
        if self._enabled:
            self._ctx = torch.autocast(
                device_type="cuda", dtype=self._cfg.amp_dtype, enabled=True
            )
            self._ctx.__enter__()
        else:
            self._ctx = None
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._ctx is not None:
            return self._ctx.__exit__(exc_type, exc, tb)
        return False


# ---------------------------------------------------------------------------
# Loss (autocast-safe, FP32-stabilized)
# ---------------------------------------------------------------------------


def compute_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BCEWithLogitsLoss computed in FP32 for numerical stability.

    Under autocast, fp16 logits can overflow during sigmoid/BCE. We upcast
    logits and targets to float32 so the loss reduction happens in a
    "higher-precision island". Gradients remain connected to the model.
    """
    logits32 = logits.float()
    targets32 = targets.float()
    pw = None
    if pos_weight is not None:
        pw = pos_weight.float().to(logits32.device)
    return F.binary_cross_entropy_with_logits(
        logits32, targets32, pos_weight=pw, reduction="mean"
    )


def compute_bce_loss_module(
    criterion: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute loss through the Stage-1 real criterion but in FP32-safe form.

    Falls back to `compute_bce_loss` when the criterion does not accept a
    positional (logits, targets) pair directly (e.g. pos_weight defaults).
    """
    logits32 = logits.float()
    targets32 = targets.float()
    try:
        return criterion(logits32, targets32)
    except TypeError:
        return compute_bce_loss(logits32, targets32)


# ---------------------------------------------------------------------------
# GradScaler factory (FP16 only)
# ---------------------------------------------------------------------------


def make_grad_scaler(cfg: PrecisionConfig) -> Optional["torch.cuda.amp.GradScaler"]:
    """Create a GradScaler iff AMP is active with FP16; else None.

    Returns None on CPU or FP32 to keep the training loop branch-free.
    """
    if cfg is None or not cfg.is_amp:
        return None
    if cfg.device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.amp.GradScaler(enabled=True)
    except Exception:
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Finite-value guard for fallback triggers
# ---------------------------------------------------------------------------


def is_finite_scalar(value: Any) -> bool:
    """True iff the (possibly tensor) scalar value is finite."""
    try:
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value.detach()).all().item())
        return bool(torch.isfinite(torch.tensor(float(value))).item())
    except Exception:
        return False


def update_fallback_on_nonfinite(
    cfg: PrecisionConfig,
    loss_value: float,
    consecutive_counter: int,
    patience: int = _NONFINITE_PATIENCE,
) -> Tuple[PrecisionConfig, int]:
    """Track non-finite losses and trigger FP32 downgrade after patience.

    Returns (possibly-new-cfg, updated-consecutive-counter).
    Stage 3 should call this right after computing the loss and before
    the backward/scaler step.
    """
    if is_finite_scalar(loss_value):
        return cfg, 0
    counter = int(consecutive_counter) + 1
    if counter >= patience and cfg.is_amp:
        return degrade_to_fp32(cfg, reason=f"nonfinite_x{counter}"), 0
    return cfg, counter


# ---------------------------------------------------------------------------
# Precision state (checkpoint-safe)
# ---------------------------------------------------------------------------


def precision_state_dict(
    cfg: PrecisionConfig,
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
) -> Dict[str, Any]:
    """Serialize precision choices so resume restores the same policy."""
    state: Dict[str, Any] = {
        "policy": cfg.policy,
        "resolved_policy": cfg.resolved_policy,
        "use_amp": bool(cfg.use_amp),
        "use_tf32": bool(cfg.use_tf32),
        "amp_dtype": str(cfg.amp_dtype) if cfg.amp_dtype is not None else None,
        "fallback_events": list(cfg.fallback_events),
    }
    if scaler is not None:
        try:
            state["scaler"] = scaler.state_dict()
        except Exception:
            pass
    return state


def load_precision_state_dict(
    cfg: PrecisionConfig,
    state: Optional[Dict[str, Any]],
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
) -> PrecisionConfig:
    """Restore an earlier PrecisionConfig (best-effort, never raises)."""
    if not state:
        return cfg
    resolved = state.get("resolved_policy", cfg.resolved_policy)
    if (
        resolved == "fp16_amp"
        and cfg.device.type == "cuda"
        and torch.cuda.is_available()
    ):
        cfg.resolved_policy = "fp16_amp"
        cfg.amp_dtype = AMP_DTYPE
        cfg.use_amp = True
    else:
        cfg.resolved_policy = "fp32"
        cfg.amp_dtype = None
        cfg.use_amp = False
    if isinstance(state.get("fallback_events"), list):
        cfg.fallback_events = list(state["fallback_events"])
    if scaler is not None and "scaler" in state:
        try:
            scaler.load_state_dict(state["scaler"])
        except Exception:
            pass
    return cfg


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def describe_precision(cfg: PrecisionConfig) -> str:
    """Single-line human-readable precision summary for the epoch log."""
    dtype_str = str(cfg.amp_dtype).replace("torch.", "") if cfg.amp_dtype else "fp32"
    events = ";".join(cfg.fallback_events) if cfg.fallback_events else "none"
    return (
        f"precision=policy:{cfg.policy} resolved:{cfg.resolved_policy} "
        f"dtype:{dtype_str} use_amp:{bool(cfg.use_amp)} "
        f"tf32:{bool(cfg.use_tf32)} device:{cfg.device.type} "
        f"fallbacks:[{events}]"
    )


def log_precision(cfg: PrecisionConfig, print_fn=print) -> str:
    """Log the precision summary once at run start (no batch-level noise)."""
    line = describe_precision(cfg)
    try:
        print_fn(f"[precision] {line}")
    except Exception:
        pass
    return line


__all__ = [
    "DEFAULT_PRECISION_POLICY",
    "AMP_DTYPE",
    "USE_TF32",
    "PrecisionConfig",
    "resolve_device",
    "disable_tf32",
    "resolve_precision_config",
    "degrade_to_fp32",
    "autocast_forward",
    "compute_bce_loss",
    "compute_bce_loss_module",
    "make_grad_scaler",
    "is_finite_scalar",
    "update_fallback_on_nonfinite",
    "precision_state_dict",
    "load_precision_state_dict",
    "describe_precision",
    "log_precision",
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
Design: 
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
    "loss": "binary cross-entropy with logits (BCEWithLogitsLoss)",
    "output_interface": "per-example probability via sigmoid at threshold 0.5 for F1",
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
    "optimizer": "AdamW",
    "scheduler": "linear warmup + linear decay",
    "batch_size_policy": "fixed",
    "dataloader_policy": "safe defaults; enable workers and pinned memory only when compatible",
    "fallbacks": [
      "if AMP unstable/slow, run FP32",
      "if transformer unavailable, TF-IDF + linear model",
      "if loss/gradients non-finite, disable autocast locally or globally"
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
  - Stage 1 model-design / model_design: baseline_change=## Plan **WHAT makes this unique**: Disaster tweets are extremely short (~15 words) with high informality (slang, URLs, @-mentions, hashtags), strong class ambiguity (word "fire" can be disaster or metaphor), and two...; purpose=Define the Stage 1 candidate target for later precision and training stages.; hardware_keys=lstm_hidden, word_emb_dim, char_cnn_filters
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=Added a selective FP16 AMP utility layer (`PrecisionConfig`, `resolve_precision_config`, `autocast_forward`, `compute_bce_loss`, `make_grad_scaler`, `precision_state_dict`/`load_precision_state_dict`, `describe_precis...; purpose=implement the trace `datatype_precision.precision_policy="fp16_amp"` while keeping FP32 parameters, integer inputs, and sigmoid→tuned-threshold output for F1 intact, and preserve the Stage 1 multi-view target.; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=Added explicit `disable_tf32()` helper (invoked only at run-time by Stage 3) and a runtime fallback path (`degrade_to_fp32`) that flips AMP off if loss/grads are non-finite or CUDA is unavailable.; purpose=match the trace fallback policy "fall back to fp32 or disabled AMP if unsupported/unstable" and the normal-mode rule forbidding TF32 on Ampere.; hardware_keys=none
  - Stage 2 datatype/quantization / datatype_precision: baseline_change=AMP/scaling region is scoped to forward+loss only; backward/step remain outside autocast with `GradScaler.unscale_` before clipping, and metrics/export stay FP32.; purpose=prevent the FP16 instability that memory attempts #1/#2 risked, matching the precision-required "high-precision islands" guidance.; hardware_keys=none
  - Stage 3 training / training_evaluation: baseline_change=## Plan **WHAT/WHY/HOW recap**: This step trains the Stage‑1 multi-view classifier (word BiLSTM+attention, char CNN, keyword/location/meta MLP) with the Stage‑2 selective FP16 AMP policy, using AdamW + linear warmup/l...; purpose=Train and evaluate the Stage 1 candidate using the selected precision policy.; hardware_keys=resolve_precision_config, autocast_forward, compute_bce_loss, make_grad_scaler, disable_tf32, update_fallback_on_nonfinite, pin_memory, non_blocking

# Step results

        ### Step 1: model_design
        **Plan:** ## Plan

**WHAT makes this unique**: Disaster tweets are extremely short (~15 words) with high informality (slang, URLs, @-mentions, hashtags), strong class ambiguity (word "fire" can be disaster or metaphor), and two sparse metadata fields (`keyword`, `location`) that prior attempts under-used. Models that only look at TF-IDF n-grams (attempt #1) or replace text with a from-scratch transformer (attempt #2) miss the sub-word structure and the meta evidence that disambiguate disaster vs. metaphor.

**WHY this approach**: I'll build a *three-view neural text classifier* — (a) a **word-level BiLSTM with additive attention pooling** to capture short-range argument structure and important token salience, (b) a **char-level multi-kernel CNN** to capture hashtags, misspellings, and morphological cues that word n-grams miss, and (c) a **meta view** that embeds `keyword` categorically and encodes `location` as a small bag-of-words + explicit presence flags, plus 16 hand-crafted numeric features (URL/@/# presence, caps ratio, punctuation, length). All three views are concatenated into an MLP head with BCEWithLogitsLoss. This is a fundamentally different hypothesis from the memory (TF-IDF+SparseMLP / hybrid transformer): it is an *inductive-bias-matched recurrent + convolutional + tabular fusion*, not a bag-of-ngrams or attention-only model.

**HOW to validate**: I'll fit vocab/stats only on training rows, split with the same 80/20 stratified seed-42 protocol, and evaluate hold-out F1 at the tuned threshold. Confirmation: char+word+meta views each contribute measurable F1 over the word-only ablation and exceed 0.7573. Falsification: if F1 sits near the word-only baseline, drop the char/meta views for Stage 3.

**Backend / elasticity / fallback**: single-process CUDA subprocess (no MPS); batch elasticity comes from Stage 3 (fixed batch policy per trace); the model keeps `lstm_hidden`, `word_emb_dim`, `char_cnn_filters`, sequence lengths and dropout configurable; fallback is FP32 CPU path in the preflight adapter. Scheduler owns device placement, thread allocation, and job launch. Loss is BCEWithLogitsLoss (trace). Output interface: per-example probability via sigmoid → tuned-threshold binary prediction.

**Note board:**
- baseline_change: replaced TF-IDF+MLP / from-scratch transformer with a **multi-view BiLSTM-attention + char-CNN + keyword/location-embedding fusion** classifier. | purpose: match the short, noisy, sub-word-rich structure of tweets and actually exploit `keyword`/`location` metadata, which the prior attempts ignored. | hardware_keys: none
- baseline_change: added explicit hand-crafted numeric features (URL/@/# presence, caps ratio, punctuation counts, length, unique-word ratio) and a location bag-of-words view. | purpose: give the model cheap disambiguators for metaphor vs. real disaster without inflating parameter count. | hardware_keys: none
- baseline_change: exposed `CandidateAdapter` that exercises the *real* model + loss on real train rows during CPU preflight, and confirmed BCEWithLogitsLoss + sigmoid@tuned-threshold output interface matches the F1 metric. | purpose: satisfy the CPU-preflight contract and keep Stage 2/3 aligned with Stage 1 family. | hardware_keys: none
        **Code:**
        ```python
"""Stage 1 model-design: Disaster Tweets multi-view text classifier.

Defines data cleaning, feature/vocab construction, PyTorch model architecture,
loss, output interface, split helpers, and a CPU-safe CandidateAdapter.
No training/evaluation/submission side effects happen at import time.
"""

import os
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants / contract
# ---------------------------------------------------------------------------

MODEL_FAMILY = "multiview_text_cnn_bilstm_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
INPUT_DIR = "./input"

# Vocabulary / geometry (configurable, low-risk defaults for tweet length)
WORD_VOCAB_SIZE = 15000
MAX_WORD_LEN = 40
MAX_CHAR_LEN = 160

# Model dimensions
WORD_EMB_DIM = 128
CHAR_EMB_DIM = 48
LSTM_HIDDEN = 128
CHAR_CNN_FILTERS = 64
KEYWORD_EMB_DIM = 32
LOC_BOW_DIM = 40  # top-N location tokens used as BOW
NUMERIC_DIM = 16  # explicit hand-crafted numeric features
HIDDEN_DIM = 256
DROPOUT = 0.3

# Split
SPLIT_SEED = 42
VAL_SIZE = 0.2


# ---------------------------------------------------------------------------
# Char vocabulary (fixed, ASCII-only, import-safe)
# ---------------------------------------------------------------------------

_ASCII_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?'\"#@:/_-()"
# 0 = pad, 1 = unk
CHAR2IDX = {c: i + 2 for i, c in enumerate(_ASCII_CHARS)}
CHAR_VOCAB_SIZE = len(_ASCII_CHARS) + 2


# ---------------------------------------------------------------------------
# Text cleaning / tokenization
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NUM_RE = re.compile(r"\d+")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_NON_PRINT_RE = re.compile(r"[^\x20-\x7E]")
_WS_RE = re.compile(r"\s+")


def clean_text(s):
    """Normalize one tweet string into a canonical lowercased token stream."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _URL_RE.sub(" url ", s)
    s = _MENTION_RE.sub(" user ", s)
    s = _HASHTAG_RE.sub(r" hashtag_\1 ", s)
    s = _NUM_RE.sub(" num ", s)
    s = _NON_PRINT_RE.sub(" ", s)
    s = s.lower()
    s = _WS_RE.sub(" ", s).strip()
    return s


def tokenize(s):
    """Return word tokens from a (cleaned) tweet string."""
    return _WORD_RE.findall(s)


def numeric_features(text, location, keyword):
    """16 hand-crafted numeric features for the meta view."""
    t = str(text) if text is not None else ""
    loc = str(location) if location is not None else ""
    kw = str(keyword) if keyword is not None else ""

    n_chars = len(t)
    words = t.split()
    n_words = len(words)
    has_url = 1.0 if _URL_RE.search(t) else 0.0
    has_mention = 1.0 if _MENTION_RE.search(t) else 0.0
    has_hashtag = 1.0 if _HASHTAG_RE.search(t) else 0.0
    has_num = 1.0 if _NUM_RE.search(t) else 0.0
    caps = sum(1 for c in t if c.isupper())
    caps_ratio = caps / max(n_chars, 1)
    excl = t.count("!")
    quest = t.count("?")
    has_loc = 1.0 if loc.strip() else 0.0
    has_kw = 1.0 if kw.strip() else 0.0
    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
    unique_ratio = (
        (len(set(w.lower() for w in words)) / max(n_words, 1)) if words else 0.0
    )
    all_upper = 1.0 if (n_chars > 3 and t.isupper()) else 0.0

    return np.array(
        [
            min(n_chars, 100) / 100.0,
            min(n_words, 20) / 20.0,
            has_url,
            has_mention,
            has_hashtag,
            has_num,
            caps_ratio,
            min(excl, 5) / 5.0,
            min(quest, 5) / 5.0,
            has_loc,
            has_kw,
            avg_word_len / 8.0,
            unique_ratio,
            all_upper,
            min(n_chars, 50) / 50.0,
            min(n_words, 10) / 10.0,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Vocabulary / feature-state construction (fit on TRAIN ONLY)
# ---------------------------------------------------------------------------


def build_vocabs(df, word_vocab_size=WORD_VOCAB_SIZE, loc_topk=LOC_BOW_DIM):
    """Fit word vocab, keyword vocab and location BOW vocab on `df`."""
    word_counts = Counter()
    for t in df["text"].fillna("").astype(str):
        word_counts.update(tokenize(clean_text(t)))

    # keep place 0 for pad, 1 for unk
    reserve = 2
    most_common = word_counts.most_common(max(word_vocab_size - reserve, 1))
    word2idx = {w: i + reserve for i, (w, _) in enumerate(most_common)}

    keyword_values = df["keyword"].fillna("__missing__").astype(str).unique().tolist()
    # index 0 = missing keyword
    kw2idx = {k: i + 1 for i, k in enumerate(sorted(keyword_values))}

    loc_counter = Counter()
    for l in df["location"].fillna("").astype(str):
        for tok in tokenize(l.lower()):
            if len(tok) >= 3:
                loc_counter[tok] += 1
    # index 0 reserved for "location present" flag
    top_loc = [t for t, _ in loc_counter.most_common(max(loc_topk - 1, 1))]
    loc2idx = {t: i + 1 for i, t in enumerate(top_loc)}

    return {
        "word2idx": word2idx,
        "kw2idx": kw2idx,
        "loc2idx": loc2idx,
    }


def encode_text(df, state, max_word_len=MAX_WORD_LEN, max_char_len=MAX_CHAR_LEN):
    """Encode a DataFrame into the five model input tensors (as numpy ints/floats)."""
    word2idx = state["word2idx"]
    kw2idx = state["kw2idx"]
    loc2idx = state["loc2idx"]

    n = len(df)
    word_arr = np.zeros((n, max_word_len), dtype=np.int64)
    char_arr = np.zeros((n, max_char_len), dtype=np.int64)
    kw_arr = np.zeros((n,), dtype=np.int64)
    loc_arr = np.zeros((n, len(loc2idx) + 1), dtype=np.float32)
    num_arr = np.zeros((n, NUMERIC_DIM), dtype=np.float32)

    texts = df["text"].fillna("").astype(str).tolist()
    keywords = df["keyword"].fillna("__missing__").astype(str).tolist()
    locations = df["location"].fillna("").astype(str).tolist()

    for i in range(n):
        cleaned = clean_text(texts[i])
        toks = tokenize(cleaned)[:max_word_len]
        for j, tok in enumerate(toks):
            word_arr[i, j] = word2idx.get(tok, 1)

        chars = list(texts[i].lower())[:max_char_len]
        for j, ch in enumerate(chars):
            char_arr[i, j] = CHAR2IDX.get(ch, 1)

        kw_arr[i] = kw2idx.get(keywords[i], 0)

        loc_str = locations[i].lower()
        if loc_str.strip():
            loc_arr[i, 0] = 1.0
        for tok in tokenize(loc_str):
            if tok in loc2idx:
                loc_arr[i, loc2idx[tok]] = 1.0

        num_arr[i] = numeric_features(texts[i], locations[i], keywords[i])

    return {
        "word_ids": word_arr,
        "char_ids": char_arr,
        "kw_ids": kw_arr,
        "loc_vec": loc_arr,
        "num_feats": num_arr,
    }


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------


def make_split(df, val_size=VAL_SIZE, seed=SPLIT_SEED):
    """Stratified train/val indices. Uses labels of TRAIN only, no leakage."""
    idx = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_size,
        stratify=df["target"].values,
        random_state=seed,
    )
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MultiViewTextClassifier(nn.Module):
    """Three-view text classifier: word BiLSTM-attention + char CNN + meta MLP."""

    def __init__(
        self,
        word_vocab_size=WORD_VOCAB_SIZE,
        char_vocab_size=CHAR_VOCAB_SIZE,
        num_keywords=1000,
        num_loc_tokens=LOC_BOW_DIM + 1,
        numeric_dim=NUMERIC_DIM,
        word_emb_dim=WORD_EMB_DIM,
        char_emb_dim=CHAR_EMB_DIM,
        lstm_hidden=LSTM_HIDDEN,
        char_cnn_filters=CHAR_CNN_FILTERS,
        kw_emb_dim=KEYWORD_EMB_DIM,
        loc_dim=32,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()

        # --- word view ---
        self.word_emb = nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            word_emb_dim, lstm_hidden, batch_first=True, bidirectional=True
        )
        self.word_att = nn.Linear(lstm_hidden * 2, 1)

        # --- char view ---
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.char_convs = nn.ModuleList(
            [
                nn.Conv1d(char_emb_dim, char_cnn_filters, k, padding=k // 2)
                for k in (2, 3, 4, 5)
            ]
        )

        # --- meta view ---
        self.kw_emb = nn.Embedding(num_keywords + 1, kw_emb_dim, padding_idx=None)
        self.loc_proj = nn.Linear(num_loc_tokens, loc_dim)
        self.num_proj = nn.Linear(numeric_dim, 32)

        # --- head ---
        in_dim = lstm_hidden * 2 + char_cnn_filters * 4 + kw_emb_dim + loc_dim + 32
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(p.data)
            elif "bias" in name and p.dim() == 1:
                nn.init.zeros_(p.data)
            elif p.dim() >= 2:
                nn.init.xavier_uniform_(p.data)

    def forward(self, word_ids, char_ids, kw_ids, loc_vec, num_feats):
        # ---- word view ----
        w = self.word_emb(word_ids)  # B, L, E
        h, _ = self.lstm(w)  # B, L, 2H
        att_logits = self.word_att(h).squeeze(-1)  # B, L
        # mask padding positions (assume pad idx 0)
        pad_mask = word_ids != 0
        neg_inf = torch.finfo(att_logits.dtype).min
        att_logits = att_logits.masked_fill(~pad_mask, neg_inf)
        att = torch.softmax(att_logits, dim=1).unsqueeze(-1)  # B, L, 1
        word_feat = (h * att).sum(dim=1)  # B, 2H

        # ---- char view ----
        c = self.char_emb(char_ids).transpose(1, 2)  # B, Ec, Lc
        char_feats = []
        for conv in self.char_convs:
            x = F.relu(conv(c))
            x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
            char_feats.append(x)
        char_feat = torch.cat(char_feats, dim=1)  # B, 4*F

        # ---- meta view ----
        kw_feat = self.kw_emb(kw_ids)  # B, Dkw
        loc_feat = F.relu(self.loc_proj(loc_vec))  # B, Dloc
        num_feat = F.relu(self.num_proj(num_feats))  # B, 32
        meta_feat = torch.cat([kw_feat, loc_feat, num_feat], dim=1)

        z = torch.cat([word_feat, char_feat, meta_feat], dim=1)
        logits = self.head(z).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# Loss / criterion
# ---------------------------------------------------------------------------


def build_loss():
    """BCEWithLogitsLoss as directed by the pipeline decision trace."""
    return nn.BCEWithLogitsLoss()


def predict_proba(model, encoded, device, batch_size=64):
    """Real-model forward pass producing float32 probabilities (no grad)."""
    model.eval()
    n = len(encoded["word_ids"])
    probs = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            wi = torch.tensor(
                encoded["word_ids"][start:end], dtype=torch.long, device=device
            )
            ci = torch.tensor(
                encoded["char_ids"][start:end], dtype=torch.long, device=device
            )
            ki = torch.tensor(
                encoded["kw_ids"][start:end], dtype=torch.long, device=device
            )
            lv = torch.tensor(
                encoded["loc_vec"][start:end], dtype=torch.float32, device=device
            )
            nf = torch.tensor(
                encoded["num_feats"][start:end], dtype=torch.float32, device=device
            )
            logits = model(wi, ci, ki, lv, nf)
            probs[start:end] = torch.sigmoid(logits.float()).cpu().numpy()
    return probs


# ---------------------------------------------------------------------------
# CPU preflight adapter (real model + real loss on real data)
# ---------------------------------------------------------------------------


class CandidateAdapter:
    """No-arg adapter exercising the real model/loss/inputs on CPU."""

    _SUBSET_FOR_PREFLIGHT = 256

    def __init__(self):
        self._state = None
        self._encoded = None
        self._targets = None

    # -------- internal: build the same feature state the pipeline uses ----
    def _ensure(self):
        if self._state is not None:
            return
        input_dir = os.environ.get("MLEVOLVE_INPUT_DIR", INPUT_DIR)
        train_path = os.path.join(input_dir, "train.csv")
        df = pd.read_csv(train_path)
        state = build_vocabs(df)
        n = min(self._SUBSET_FOR_PREFLIGHT, len(df))
        small = df.iloc[:n].reset_index(drop=True)
        self._state = state
        self._encoded = encode_text(small, state)
        self._targets = small["target"].astype(np.float32).values

    # -------- construction --------
    def build_model(self, context=None):
        self._ensure()
        num_keywords = len(self._state["kw2idx"])
        num_loc_tokens = len(self._state["loc2idx"]) + 1
        model = MultiViewTextClassifier(
            word_vocab_size=WORD_VOCAB_SIZE,
            char_vocab_size=CHAR_VOCAB_SIZE,
            num_keywords=num_keywords,
            num_loc_tokens=num_loc_tokens,
            numeric_dim=NUMERIC_DIM,
            word_emb_dim=WORD_EMB_DIM,
            char_emb_dim=CHAR_EMB_DIM,
            lstm_hidden=LSTM_HIDDEN,
            char_cnn_filters=CHAR_CNN_FILTERS,
            kw_emb_dim=KEYWORD_EMB_DIM,
            loc_dim=32,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
        )
        return model

    def build_optimizer(self, model, context=None):
        # Only a placeholder for the preflight; Stage 3 owns the real optimizer.
        return torch.optim.AdamW(model.parameters(), lr=1e-3)

    def _make_batch(self, scenario, device):
        self._ensure()
        bs = int(scenario.get("batch_size", 8)) if scenario else 8
        bs = max(1, min(bs, len(self._targets)))
        idx = np.arange(bs)
        enc = self._encoded
        inputs = (
            torch.tensor(enc["word_ids"][idx], dtype=torch.long, device=device),
            torch.tensor(enc["char_ids"][idx], dtype=torch.long, device=device),
            torch.tensor(enc["kw_ids"][idx], dtype=torch.long, device=device),
            torch.tensor(enc["loc_vec"][idx], dtype=torch.float32, device=device),
            torch.tensor(enc["num_feats"][idx], dtype=torch.float32, device=device),
        )
        target = torch.tensor(self._targets[idx], dtype=torch.float32, device=device)
        return {"inputs": inputs, "target": target}

    def build_train_batch(self, scenario, device):
        return self._make_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._make_batch(scenario, device)

    def training_step(self, model, batch, context=None):
        inputs = batch["inputs"]
        target = batch["target"]
        logits = model(*inputs)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        return loss

    def validation_step(self, model, batch, context=None):
        inputs = batch["inputs"]
        target = batch["target"]
        logits = model(*inputs)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        return {"loss": loss, "probs": probs, "target": target}


if __name__ == "__main__":
    # Stage 1 owns only definitions; training is driven by Stage 3.
    pass

```
        
        ### Step 2: datatype_precision
        **Plan:** ## Plan

I will implement the Stage 2 `datatype_precision` layer for the Stage 1 `multiview_text_cnn_bilstm_v1` classifier, honoring the trace's `precision_policy="fp16_amp"` and conservative fallback. The policy is a **selective FP16 AMP** design: FP32 parameters/inputs/targets remain FP32, autocast is applied only around the eligible forward pass, the BCEWithLogitsLoss is computed inside autocast (it is autocast-safe), backward and optimizer step happen outside autocast, and a `GradScaler` (only when CUDA+FP16) unscales before clipping. TF32 is explicitly disabled, and a CPU-only preflight uses pure FP32 with AMP/scaling off. Precision-required adaptation is `none` — I only add thin, optional wrappers (a `precision_forward`/`compute_loss` helper) that preserve the model family, loss, features and output interface.

## Note board:
- baseline_change: Added a selective FP16 AMP utility layer (`PrecisionConfig`, `resolve_precision_config`, `autocast_forward`, `compute_bce_loss`, `make_grad_scaler`, `precision_state_dict`/`load_precision_state_dict`, `describe_precision`) that wraps the Stage 1 model's forward and loss without altering architecture, features, or output interface. | purpose: implement the trace `datatype_precision.precision_policy="fp16_amp"` while keeping FP32 parameters, integer inputs, and sigmoid→tuned-threshold output for F1 intact, and preserve the Stage 1 multi-view target. | hardware_keys: none
- baseline_change: Added explicit `disable_tf32()` helper (invoked only at run-time by Stage 3) and a runtime fallback path (`degrade_to_fp32`) that flips AMP off if loss/grads are non-finite or CUDA is unavailable. | purpose: match the trace fallback policy "fall back to fp32 or disabled AMP if unsupported/unstable" and the normal-mode rule forbidding TF32 on Ampere. | hardware_keys: none
- baseline_change: AMP/scaling region is scoped to forward+loss only; backward/step remain outside autocast with `GradScaler.unscale_` before clipping, and metrics/export stay FP32. | purpose: prevent the FP16 instability that memory attempts #1/#2 risked, matching the precision-required "high-precision islands" guidance. | hardware_keys: none
        **Code:**
        ```python
"""Stage 2 datatype/precision utilities for the Disaster Tweets classifier.

Defines device/dtype policy, AMP/autocast region helpers, GradScaler factory,
loss computation, TF32 disabling, fallback, and precision logging.

Import-safe: NO data access, NO CUDA configuration, NO logging setup at import.
All side effects must be triggered explicitly from the Stage 3 training entry.
Preserves Stage 1 model family (multiview_text_cnn_bilstm_v1), loss
(BCEWithLogitsLoss), data features, and output interface (sigmoid probability).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Policy constants (from pipeline decision trace / hardware context)
# ---------------------------------------------------------------------------

# Allowed: "fp32" | "fp16_amp"  (bf16 / TF32 / TE recipes are NOT allowed in
# the current normal mode for compute capability 8.6.)
DEFAULT_PRECISION_POLICY = "fp16_amp"

# AMP autocast dtype candidate (only fp16 is permitted in normal mode).
AMP_DTYPE = torch.float16

# TF32 must be disabled in normal mode.
USE_TF32 = False

# Number of consecutive non-finite steps before a hard fallback to FP32.
_NONFINITE_PATIENCE = 3


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class PrecisionConfig:
    """Resolved precision policy for the current run.

    Attributes
    ----------
    policy          : "fp32" or "fp16_amp" (the requested policy).
    resolved_policy : the policy actually in effect after fallbacks.
    device          : torch.device the model/batches live on.
    amp_dtype       : autocast dtype when AMP is enabled, else None.
    use_amp         : whether CUDA autocast + GradScaler should be used.
    use_tf32        : always False in normal mode (recorded for logging).
    fallback_events : list of human-readable fallback reasons, in order.
    """

    policy: str = DEFAULT_PRECISION_POLICY
    resolved_policy: str = "fp32"
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    amp_dtype: Optional[torch.dtype] = None
    use_amp: bool = False
    use_tf32: bool = False
    fallback_events: list = field(default_factory=list)

    @property
    def is_amp(self) -> bool:
        return bool(self.use_amp and self.amp_dtype is not None)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available device without mutating global state."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# TF32 disabling (must be called at run-time, not import time)
# ---------------------------------------------------------------------------


def disable_tf32() -> None:
    """Explicitly disable TF32 matmul/cudnn paths (normal mode requirement)."""
    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
    except Exception:
        # Non-fatal: fall through, the policy is still recorded.
        pass


# ---------------------------------------------------------------------------
# Config resolution + fallback
# ---------------------------------------------------------------------------


def resolve_precision_config(
    policy: Optional[str] = None,
    device: Optional[torch.device] = None,
    prefer_cuda: bool = True,
) -> PrecisionConfig:
    """Resolve the effective precision policy from an optional requested policy.

    Applies the trace fallback rule: fall back to fp32 / disabled AMP when the
    selected precision path is unsupported on the current device.
    """
    requested = (policy or DEFAULT_PRECISION_POLICY).lower()
    dev = device if device is not None else resolve_device(prefer_cuda=prefer_cuda)

    cfg = PrecisionConfig(policy=requested, device=dev, use_tf32=False)
    cfg.fallback_events = []

    if requested not in ("fp32", "fp16_amp"):
        cfg.fallback_events.append(f"unknown_policy:{requested}->fp32")
        requested = "fp32"
        cfg.policy = "fp32"

    if requested == "fp16_amp":
        if dev.type == "cuda" and torch.cuda.is_available():
            cfg.resolved_policy = "fp16_amp"
            cfg.amp_dtype = AMP_DTYPE
            cfg.use_amp = True
        else:
            cfg.resolved_policy = "fp32"
            cfg.amp_dtype = None
            cfg.use_amp = False
            cfg.fallback_events.append("no_cuda->fp32")
    else:
        cfg.resolved_policy = "fp32"
        cfg.amp_dtype = None
        cfg.use_amp = False

    return cfg


def degrade_to_fp32(
    cfg: PrecisionConfig,
    reason: str = "nonfinite_or_unstable",
    device: Optional[torch.device] = None,
) -> PrecisionConfig:
    """Return a new PrecisionConfig switched to FP32 (idempotent fallback)."""
    new_cfg = PrecisionConfig(
        policy="fp32",
        resolved_policy="fp32",
        device=cfg.device if device is None else device,
        amp_dtype=None,
        use_amp=False,
        use_tf32=False,
    )
    new_cfg.fallback_events = list(cfg.fallback_events) + [f"degrade:{reason}"]
    return new_cfg


# ---------------------------------------------------------------------------
# Autocast / forward helpers
# ---------------------------------------------------------------------------


class autocast_forward:
    """Context manager wrapping ONLY the eligible forward ops in CUDA autocast.

    Usage inside training_step / validation_step:

        with autocast_forward(cfg):
            logits = model(*inputs)
            loss = compute_bce_loss(logits, targets)   # autocast-safe

    Backward and optimizer step MUST happen outside this region.
    """

    __slots__ = ("_cfg", "_enabled")

    def __init__(self, cfg: PrecisionConfig, enabled: Optional[bool] = None):
        self._cfg = cfg
        if enabled is None:
            self._enabled = bool(cfg is not None and cfg.is_amp)
        else:
            self._enabled = bool(enabled)

    def __enter__(self):
        if self._enabled:
            self._ctx = torch.autocast(
                device_type="cuda", dtype=self._cfg.amp_dtype, enabled=True
            )
            self._ctx.__enter__()
        else:
            self._ctx = None
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._ctx is not None:
            return self._ctx.__exit__(exc_type, exc, tb)
        return False


# ---------------------------------------------------------------------------
# Loss (autocast-safe, FP32-stabilized)
# ---------------------------------------------------------------------------


def compute_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BCEWithLogitsLoss computed in FP32 for numerical stability.

    Under autocast, fp16 logits can overflow during sigmoid/BCE. We upcast
    logits and targets to float32 so the loss reduction happens in a
    "higher-precision island". Gradients remain connected to the model.
    """
    logits32 = logits.float()
    targets32 = targets.float()
    pw = None
    if pos_weight is not None:
        pw = pos_weight.float().to(logits32.device)
    return F.binary_cross_entropy_with_logits(
        logits32, targets32, pos_weight=pw, reduction="mean"
    )


def compute_bce_loss_module(
    criterion: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute loss through the Stage-1 real criterion but in FP32-safe form.

    Falls back to `compute_bce_loss` when the criterion does not accept a
    positional (logits, targets) pair directly (e.g. pos_weight defaults).
    """
    logits32 = logits.float()
    targets32 = targets.float()
    try:
        return criterion(logits32, targets32)
    except TypeError:
        return compute_bce_loss(logits32, targets32)


# ---------------------------------------------------------------------------
# GradScaler factory (FP16 only)
# ---------------------------------------------------------------------------


def make_grad_scaler(cfg: PrecisionConfig) -> Optional["torch.cuda.amp.GradScaler"]:
    """Create a GradScaler iff AMP is active with FP16; else None.

    Returns None on CPU or FP32 to keep the training loop branch-free.
    """
    if cfg is None or not cfg.is_amp:
        return None
    if cfg.device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.amp.GradScaler(enabled=True)
    except Exception:
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Finite-value guard for fallback triggers
# ---------------------------------------------------------------------------


def is_finite_scalar(value: Any) -> bool:
    """True iff the (possibly tensor) scalar value is finite."""
    try:
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value.detach()).all().item())
        return bool(torch.isfinite(torch.tensor(float(value))).item())
    except Exception:
        return False


def update_fallback_on_nonfinite(
    cfg: PrecisionConfig,
    loss_value: float,
    consecutive_counter: int,
    patience: int = _NONFINITE_PATIENCE,
) -> Tuple[PrecisionConfig, int]:
    """Track non-finite losses and trigger FP32 downgrade after patience.

    Returns (possibly-new-cfg, updated-consecutive-counter).
    Stage 3 should call this right after computing the loss and before
    the backward/scaler step.
    """
    if is_finite_scalar(loss_value):
        return cfg, 0
    counter = int(consecutive_counter) + 1
    if counter >= patience and cfg.is_amp:
        return degrade_to_fp32(cfg, reason=f"nonfinite_x{counter}"), 0
    return cfg, counter


# ---------------------------------------------------------------------------
# Precision state (checkpoint-safe)
# ---------------------------------------------------------------------------


def precision_state_dict(
    cfg: PrecisionConfig,
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
) -> Dict[str, Any]:
    """Serialize precision choices so resume restores the same policy."""
    state: Dict[str, Any] = {
        "policy": cfg.policy,
        "resolved_policy": cfg.resolved_policy,
        "use_amp": bool(cfg.use_amp),
        "use_tf32": bool(cfg.use_tf32),
        "amp_dtype": str(cfg.amp_dtype) if cfg.amp_dtype is not None else None,
        "fallback_events": list(cfg.fallback_events),
    }
    if scaler is not None:
        try:
            state["scaler"] = scaler.state_dict()
        except Exception:
            pass
    return state


def load_precision_state_dict(
    cfg: PrecisionConfig,
    state: Optional[Dict[str, Any]],
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
) -> PrecisionConfig:
    """Restore an earlier PrecisionConfig (best-effort, never raises)."""
    if not state:
        return cfg
    resolved = state.get("resolved_policy", cfg.resolved_policy)
    if (
        resolved == "fp16_amp"
        and cfg.device.type == "cuda"
        and torch.cuda.is_available()
    ):
        cfg.resolved_policy = "fp16_amp"
        cfg.amp_dtype = AMP_DTYPE
        cfg.use_amp = True
    else:
        cfg.resolved_policy = "fp32"
        cfg.amp_dtype = None
        cfg.use_amp = False
    if isinstance(state.get("fallback_events"), list):
        cfg.fallback_events = list(state["fallback_events"])
    if scaler is not None and "scaler" in state:
        try:
            scaler.load_state_dict(state["scaler"])
        except Exception:
            pass
    return cfg


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def describe_precision(cfg: PrecisionConfig) -> str:
    """Single-line human-readable precision summary for the epoch log."""
    dtype_str = str(cfg.amp_dtype).replace("torch.", "") if cfg.amp_dtype else "fp32"
    events = ";".join(cfg.fallback_events) if cfg.fallback_events else "none"
    return (
        f"precision=policy:{cfg.policy} resolved:{cfg.resolved_policy} "
        f"dtype:{dtype_str} use_amp:{bool(cfg.use_amp)} "
        f"tf32:{bool(cfg.use_tf32)} device:{cfg.device.type} "
        f"fallbacks:[{events}]"
    )


def log_precision(cfg: PrecisionConfig, print_fn=print) -> str:
    """Log the precision summary once at run start (no batch-level noise)."""
    line = describe_precision(cfg)
    try:
        print_fn(f"[precision] {line}")
    except Exception:
        pass
    return line


__all__ = [
    "DEFAULT_PRECISION_POLICY",
    "AMP_DTYPE",
    "USE_TF32",
    "PrecisionConfig",
    "resolve_device",
    "disable_tf32",
    "resolve_precision_config",
    "degrade_to_fp32",
    "autocast_forward",
    "compute_bce_loss",
    "compute_bce_loss_module",
    "make_grad_scaler",
    "is_finite_scalar",
    "update_fallback_on_nonfinite",
    "precision_state_dict",
    "load_precision_state_dict",
    "describe_precision",
    "log_precision",
]

```
        
        ### Step 3: training_evaluation
        **Plan:** ## Plan

**WHAT/WHY/HOW recap**: This step trains the Stage‑1 multi-view classifier (word BiLSTM+attention, char CNN, keyword/location/meta MLP) with the Stage‑2 selective FP16 AMP policy, using AdamW + linear warmup/linear decay and early stopping on validation F1 with best-checkpoint restore. I will consume Stage 2 utilities (`resolve_precision_config`, `autocast_forward`, `compute_bce_loss`, `make_grad_scaler`, `disable_tf32`, `update_fallback_on_nonfinite`) verbatim, run dataloaders with pinned host memory and 2 workers only on CUDA, and pick the F1 threshold on the internal validation split (identical logic used for test inference). Confirmation: val F1 exceeds 0.7573 at a threshold different from 0.5 and test submission has exactly 1523 binary predictions; falsification: if AMP degrades val F1 vs. the FP32 fallback or loss goes non-finite, the fallback triggers a mid-run downgrade to FP32, and val F1 tracks (not diverges from) the final test prediction distribution.

**Backend strategy**: single CUDA subprocess; scheduler owns device assignment, thread allocation, and memory limits. Batch elasticity is driven by `QUALITY_SAFE_PHYSICAL_BATCH_SIZES = [32, 64, 128]` with `BATCH_LR_SCALING_POLICY = "sqrt"`; effective batch = physical × grad_accum, epochs fixed to data exposure. Memory strategy: sparse batched tensor transfers, `pin_memory` + `non_blocking` only on CUDA, keep model/optimizer FP32, autocast scoped to the forward + loss island, GradScaler only when FP16 AMP resolves. Backend controls I do NOT implement: process signals, CPU/GPU assignment, memory cgroups, any sibling-job launch.

**Note board:**
- baseline_change: added `run_training` that consumes Stage 2 precision utilities end‑to‑end (autocast over forward+loss, GradScaler unscale before clip, mid‑run fallback via `update_fallback_on_nonfinite`), with AdamW + linear warmup/decay, threshold-tuned F1, and best-checkpoint restore. | purpose: implement the trace `training_evaluation` contract (AdamW, linear warmup+linear decay, fixed physical batch policy, safe dataloader defaults, AMP→FP32 fallback) while preserving Stage 1 target (multi‑view classifier, BCEWithLogitsLoss, sigmoid→tuned threshold). | hardware_keys: none
- baseline_change: hooked scheduler integration with exact `SafePointType.STEP/BEFORE_TRAIN/EPOCH` signature, `state_factory` carrying model/optimizer/GradScaler/LR-scheduler/global_step/epoch/best metrics, and `TrainingDiagnostics` wrapping the loop with `after_update()` per optimizer attempt and `report()` per epoch. | purpose: satisfy the scheduler/diagnostics contract without altering task semantics; keep hooks inactive when `script_scheduler_context` returns None. | hardware_keys: none
- baseline_change: redefined a self-contained `CandidateAdapter` that reuses the real `MultiViewTextClassifier`, real `build_loss()` criterion, real `compute_bce_loss` precision path, and honors `scenario['batch_size']` with a proper `{'inputs': (...), 'target': ...}` mapping. | purpose: pass the CPU preflight with real model+loss, no mocks, and remain import-safe under `if __name__ == '__main__':`. | hardware_keys: none
        **Code:**
        ```python
"""Stage 3 training/evaluation for the multi-view Disaster Tweets classifier.

Consumes:
- Stage 1: MultiViewTextClassifier, build_vocabs, encode_text, make_split,
           build_loss, predict_proba, WORD_VOCAB_SIZE, CHAR_VOCAB_SIZE,
           NUMERIC_DIM, INPUT_DIR constants.
- Stage 2: resolve_precision_config, autocast_forward, compute_bce_loss,
           compute_bce_loss_module, make_grad_scaler, disable_tf32,
           degrade_to_fp32, log_precision, precision_state_dict,
           load_precision_state_dict, update_fallback_on_nonfinite.

Import-safe: only definitions below. All side effects are under __main__.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Scheduler / diagnostics imports (guarded – must not fail on CPU preflight)
# ---------------------------------------------------------------------------

try:  # pragma: no cover
    from localml_scheduler.execution.script_context import script_scheduler_context
    from localml_scheduler.domain import SafePointType
except Exception:  # pragma: no cover
    script_scheduler_context = None
    SafePointType = None

try:  # pragma: no cover
    from utils.training_diagnostics import TrainingDiagnostics
except Exception:  # pragma: no cover
    TrainingDiagnostics = None


# ---------------------------------------------------------------------------
# Contract constants (already defined at Stage 1: MODEL_FAMILY, WORKING_DIR,
# SUBMISSION_DIR, INPUT_DIR). We re-assert them here so Stage 3 is self-contained.
# ---------------------------------------------------------------------------

MODEL_FAMILY = "multiview_text_cnn_bilstm_v1"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
INPUT_DIR = "./input"

# ---------------------------------------------------------------------------
# Training hyperparameters (Stage 3 owns these)
# ---------------------------------------------------------------------------

SEED = 42
SPLIT_SEED = 42

# Batch envelope (scheduler may pick inside this set only)
QUALITY_SAFE_PHYSICAL_BATCH_SIZES = [32, 64, 128]
BATCH_LR_SCALING_POLICY = "sqrt"  # one of: fixed | linear | sqrt

PHYSICAL_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 1
EFFECTIVE_BATCH_SIZE = PHYSICAL_BATCH_SIZE * GRAD_ACCUM_STEPS

PLANNED_EPOCHS = 15
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
GRAD_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 4

NUM_WORKERS_GPU = 2
NUM_WORKERS_CPU = 0
PIN_MEMORY_GPU = True
PERSISTENT_WORKERS = False  # avoid worker-teardown edge cases on short runs

# Threshold grid for F1 tuning on validation split (same logic used for test)
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)

# ---------------------------------------------------------------------------
# Utility: reproducible seeding
# ---------------------------------------------------------------------------


def set_all_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset wrapper over the Stage-1 encoded tensors
# ---------------------------------------------------------------------------


class EncodedDataset(Dataset):
    """Torch Dataset wrapping Stage 1 encoded dict (word/char/kw/loc/num)."""

    def __init__(
        self, encoded: Dict[str, np.ndarray], targets: Optional[np.ndarray] = None
    ):
        self.word_ids = torch.tensor(encoded["word_ids"], dtype=torch.long)
        self.char_ids = torch.tensor(encoded["char_ids"], dtype=torch.long)
        self.kw_ids = torch.tensor(encoded["kw_ids"], dtype=torch.long)
        self.loc_vec = torch.tensor(encoded["loc_vec"], dtype=torch.float32)
        self.num_feats = torch.tensor(encoded["num_feats"], dtype=torch.float32)
        if targets is not None:
            self.targets = torch.tensor(
                np.asarray(targets, dtype=np.float32), dtype=torch.float32
            )
        else:
            self.targets = None

    def __len__(self) -> int:
        return int(self.word_ids.shape[0])

    def __getitem__(self, idx: int):
        item = (
            self.word_ids[idx],
            self.char_ids[idx],
            self.kw_ids[idx],
            self.loc_vec[idx],
            self.num_feats[idx],
        )
        if self.targets is not None:
            return item + (self.targets[idx],)
        return item


# ---------------------------------------------------------------------------
# Threshold tuning / metric helpers (identical val and test logic)
# ---------------------------------------------------------------------------


def tune_threshold(probs: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """Sweep a threshold grid and return (best_threshold, best_f1)."""
    t = np.asarray(targets).astype(int)
    best_thr = 0.5
    best_f1 = 0.0
    for thr in THRESHOLD_GRID:
        pred = (probs >= thr).astype(int)
        f1 = f1_score(t, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr, best_f1


# ---------------------------------------------------------------------------
# Inference (shared by validation and test; identical preprocessing path)
# ---------------------------------------------------------------------------


def infer_probs(
    model: nn.Module, loader: DataLoader, cfg, device: torch.device
) -> np.ndarray:
    """Run model forward under Stage-2 autocast policy, return FP32 probabilities."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in loader:
            inputs = tuple(
                b.to(device, non_blocking=True) if torch.is_tensor(b) else b
                for b in batch[:5]
            )
            with autocast_forward(cfg):
                logits = model(*inputs)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            chunks.append(probs)
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# Training step (single optimizer update; keeps AMP region as Stage 2 policy)
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler,
    cfg,
    criterion,
    device: torch.device,
    epoch: int,
    global_step: int,
    steps_per_epoch: int,
    state_factory,
    diagnostics,
    ctx,
) -> Tuple[float, int, Any, Any, Optional[Any]]:
    """One training epoch. Returns (avg_loss, new_global_step, cfg, scaler, lr_scheduler)."""
    model.train()
    running = 0.0
    batches = 0
    nonfinite_counter = 0
    accum_counter = 0

    for batch in loader:
        inputs = tuple(
            b.to(device, non_blocking=True) if torch.is_tensor(b) else b
            for b in batch[:5]
        )
        targets = batch[5].to(device, non_blocking=True)

        with autocast_forward(cfg):
            logits = model(*inputs)
            loss = compute_bce_loss_module(criterion, logits, targets)

        # Non-finite loss guard → mid-run precision fallback
        cfg, nonfinite_counter = update_fallback_on_nonfinite(
            cfg, loss, nonfinite_counter
        )
        if not cfg.is_amp and scaler is not None:
            # AMP was degraded; drop the scaler entirely.
            scaler = None

        if scaler is not None and cfg.is_amp:
            scaler.scale(loss).backward()
            # Accumulate micro-steps before stepping (currently 1 so step every batch)
            accum_counter += 1
            if accum_counter % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                diagnostics.after_update()
                lr_scheduler.step()
                global_step += 1
                if ctx is not None and SafePointType is not None:
                    ctx.control_hook.safe_point(
                        SafePointType.STEP,
                        epoch=epoch,
                        global_step=global_step,
                        steps_per_epoch=steps_per_epoch,
                        state_factory=state_factory,
                    )
        else:
            loss.backward()
            accum_counter += 1
            if accum_counter % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                diagnostics.after_update()
                lr_scheduler.step()
                global_step += 1
                if ctx is not None and SafePointType is not None:
                    ctx.control_hook.safe_point(
                        SafePointType.STEP,
                        epoch=epoch,
                        global_step=global_step,
                        steps_per_epoch=steps_per_epoch,
                        state_factory=state_factory,
                    )

        running += float(loss.detach().item())
        batches += 1

    avg_loss = running / max(batches, 1)
    return avg_loss, global_step, cfg, scaler, lr_scheduler


# ---------------------------------------------------------------------------
# CPU preflight adapter (real model, real criterion, real data)
# ---------------------------------------------------------------------------


class CandidateAdapter:
    """No-arg adapter that exercises the real model/loss/inputs on CPU."""

    _SUBSET_FOR_PREFLIGHT = 256

    def __init__(self):
        self._state: Optional[Dict[str, Any]] = None
        self._encoded: Optional[Dict[str, np.ndarray]] = None
        self._targets: Optional[np.ndarray] = None
        self._criterion: Optional[nn.Module] = None

    # -- lazy data/vocab construction (train.csv only) --
    def _ensure(self) -> None:
        if self._state is not None:
            return
        input_dir = os.environ.get("MLEVOLVE_INPUT_DIR", INPUT_DIR)
        df = pd.read_csv(os.path.join(input_dir, "train.csv"))
        state = build_vocabs(df)
        n = min(self._SUBSET_FOR_PREFLIGHT, len(df))
        small = df.iloc[:n].reset_index(drop=True)
        self._state = state
        self._encoded = encode_text(small, state)
        self._targets = small["target"].astype(np.float32).values
        self._criterion = build_loss()

    def _build_real_model(self, num_keywords: int, num_loc_tokens: int) -> nn.Module:
        return MultiViewTextClassifier(
            word_vocab_size=WORD_VOCAB_SIZE,
            char_vocab_size=CHAR_VOCAB_SIZE,
            num_keywords=num_keywords,
            num_loc_tokens=num_loc_tokens,
            numeric_dim=NUMERIC_DIM,
        )

    # -- contract methods --
    def build_model(self, context=None) -> nn.Module:
        self._ensure()
        assert self._state is not None
        num_keywords = len(self._state["kw2idx"])
        num_loc_tokens = len(self._state["loc2idx"]) + 1
        model = self._build_real_model(num_keywords, num_loc_tokens)
        return model

    def build_optimizer(self, model: nn.Module, context=None) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

    def _resolve_criterion(self, context=None) -> nn.Module:
        crit = None
        if context is not None:
            try:
                crit = (
                    context.get("criterion")
                    if hasattr(context, "get")
                    else getattr(context, "criterion", None)
                )
            except Exception:
                crit = None
        if crit is not None:
            return crit
        if self._criterion is None:
            self._criterion = build_loss()
        return self._criterion

    def _make_batch(self, scenario, device: torch.device):
        self._ensure()
        assert self._encoded is not None and self._targets is not None
        bs = int((scenario or {}).get("batch_size", 8))
        bs = max(1, min(bs, len(self._targets)))
        idx = np.arange(bs)
        enc = self._encoded
        inputs = (
            torch.tensor(enc["word_ids"][idx], dtype=torch.long, device=device),
            torch.tensor(enc["char_ids"][idx], dtype=torch.long, device=device),
            torch.tensor(enc["kw_ids"][idx], dtype=torch.long, device=device),
            torch.tensor(enc["loc_vec"][idx], dtype=torch.float32, device=device),
            torch.tensor(enc["num_feats"][idx], dtype=torch.float32, device=device),
        )
        target = torch.tensor(self._targets[idx], dtype=torch.float32, device=device)
        return {"inputs": inputs, "target": target}

    def build_train_batch(self, scenario, device):
        return self._make_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._make_batch(scenario, device)

    def training_step(self, model: nn.Module, batch, context=None) -> torch.Tensor:
        inputs = batch["inputs"]
        target = batch["target"]
        logits = model(*inputs)
        criterion = self._resolve_criterion(context)
        return compute_bce_loss_module(criterion, logits, target)

    def validation_step(self, model: nn.Module, batch, context=None):
        inputs = batch["inputs"]
        target = batch["target"]
        logits = model(*inputs)
        criterion = self._resolve_criterion(context)
        loss = compute_bce_loss_module(criterion, logits, target)
        probs = torch.sigmoid(logits.float())
        return {"loss": loss, "probs": probs, "target": target}


# ---------------------------------------------------------------------------
# Helpers for LR schedule / diagnostics wrappers
# ---------------------------------------------------------------------------


def _make_lr_lambda(total_steps: int, warmup_steps: int):
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(min(int(warmup_steps), total_steps), 0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(total_steps - step, 0)
        denom = max(total_steps - warmup_steps, 1)
        return float(remaining) / float(denom)

    return lr_lambda


class _NoopDiagnostics:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def after_update(self):
        pass

    def report(self, **kwargs):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, *args, **kwargs):
        pass


@contextlib.contextmanager
def _diagnostics_ctx(model, optimizer, scaler, settings):
    if TrainingDiagnostics is not None:
        try:
            d = TrainingDiagnostics(model, optimizer, scaler=scaler, settings=settings)
        except TypeError:
            d = TrainingDiagnostics(model, optimizer, scaler)
        with d as diag:
            yield diag
    else:
        yield _NoopDiagnostics()


# ---------------------------------------------------------------------------
# Main training entrypoint
# ---------------------------------------------------------------------------


def run_training() -> float:
    # --- Scheduler context (before CUDA allocations) ---
    ctx = script_scheduler_context() if script_scheduler_context is not None else None

    set_all_seeds(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Precision policy from Stage 2 (do not invent new dtypes) ---
    cfg = resolve_precision_config(policy="fp16_amp", device=device)
    disable_tf32()
    log_precision(cfg)
    scaler = make_grad_scaler(cfg)

    # --- Data ---
    input_dir = os.environ.get("MLEVOLVE_INPUT_DIR", INPUT_DIR)
    train_path = os.path.join(input_dir, "train.csv")
    test_path = os.path.join(input_dir, "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_idx, val_idx = make_split(train_df)
    fit_df = train_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    # Fit vocab/stats ONLY on training split (no leakage).
    state = build_vocabs(fit_df)
    enc_train = encode_text(fit_df, state)
    enc_val = encode_text(val_df, state)
    enc_test = encode_text(test_df, state)

    y_train = fit_df["target"].astype(np.float32).values
    y_val = val_df["target"].astype(np.float32).values

    # --- Model ---
    num_keywords = len(state["kw2idx"])
    num_loc_tokens = len(state["loc2idx"]) + 1
    model = MultiViewTextClassifier(
        word_vocab_size=WORD_VOCAB_SIZE,
        char_vocab_size=CHAR_VOCAB_SIZE,
        num_keywords=num_keywords,
        num_loc_tokens=num_loc_tokens,
        numeric_dim=NUMERIC_DIM,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = build_loss()

    # --- DataLoaders ---
    train_ds = EncodedDataset(enc_train, y_train)
    val_ds = EncodedDataset(enc_val, y_val)
    test_ds = EncodedDataset(enc_test, None)

    on_cuda = device.type == "cuda"
    num_workers = NUM_WORKERS_GPU if on_cuda else NUM_WORKERS_CPU
    pin_memory = bool(PIN_MEMORY_GPU and on_cuda)

    common_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=PERSISTENT_WORKERS and num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = PLANNED_EPOCHS * steps_per_epoch
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_make_lr_lambda(total_steps, warmup_steps)
    )

    # --- Runtime state & resume ---
    current_epoch = [0]
    global_step = 0
    best_val_f1 = 0.0
    best_threshold = 0.5
    best_state: Optional[Dict[str, torch.Tensor]] = None
    start_epoch = 1

    if ctx is not None:
        try:
            resume = ctx.load_resume_checkpoint()
        except Exception:
            resume = None
        if resume and isinstance(resume, dict) and resume.get("state"):
            st = resume["state"]
            try:
                if "model" in st and st["model"] is not None:
                    model.load_state_dict(st["model"])
                if "optimizer" in st and st["optimizer"] is not None:
                    optimizer.load_state_dict(st["optimizer"])
                if "scheduler" in st and st["scheduler"] is not None:
                    lr_scheduler.load_state_dict(st["scheduler"])
                if scaler is not None and "scaler" in st and st["scaler"] is not None:
                    try:
                        scaler.load_state_dict(st["scaler"])
                    except Exception:
                        pass
                global_step = int(st.get("global_step", 0))
                current_epoch[0] = int(st.get("epoch", 0))
                start_epoch = current_epoch[0] + 1
                best_val_f1 = float(st.get("best_val_f1", 0.0))
                best_threshold = float(st.get("best_threshold", 0.5))
            except Exception:
                # Resume is best-effort; fall back to fresh start on any error.
                start_epoch = 1
                global_step = 0
                current_epoch[0] = 0
                best_val_f1 = 0.0
                best_threshold = 0.5

        # Notify the scheduler of our actual epoch budget / job config.
        try:
            if hasattr(ctx, "job"):
                if hasattr(ctx.job, "max_epochs"):
                    ctx.job.max_epochs = PLANNED_EPOCHS
                if hasattr(ctx.job, "config") and hasattr(ctx.job.config, "max_epochs"):
                    ctx.job.config.max_epochs = PLANNED_EPOCHS
            if hasattr(ctx, "store") and hasattr(ctx.store, "save_job"):
                ctx.store.save_job(ctx.job)
        except Exception:
            pass

    def state_factory():
        st: Dict[str, Any] = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": (
                lr_scheduler.state_dict() if lr_scheduler is not None else None
            ),
            "global_step": global_step,
            "epoch": current_epoch[0],
            "best_val_f1": best_val_f1,
            "best_threshold": best_threshold,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                st["cuda_rng"] = torch.cuda.get_rng_state_all()
            except Exception:
                pass
        if scaler is not None:
            try:
                st["scaler"] = scaler.state_dict()
            except Exception:
                pass
        return st

    # --- BEFORE_TRAIN safe point ---
    if ctx is not None and SafePointType is not None:
        ctx.control_hook.safe_point(
            SafePointType.BEFORE_TRAIN,
            epoch=current_epoch[0],
            global_step=global_step,
            steps_per_epoch=steps_per_epoch,
            state_factory=state_factory,
        )

    # --- Diagnostics settings ---
    diag_settings = {
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "planned_epochs": PLANNED_EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "model_family": MODEL_FAMILY,
        "split_seed": SPLIT_SEED,
        "precision": cfg.resolved_policy,
        "precision_reason": "trace_policy:fp16_amp_selective_forward_only",
    }

    run_start = time.time()
    best_epoch = start_epoch - 1
    epochs_no_improve = 0
    last_epoch_ran = start_epoch - 1

    with _diagnostics_ctx(
        model, optimizer, scaler=scaler, settings=diag_settings
    ) as diagnostics:
        for epoch in range(start_epoch, PLANNED_EPOCHS + 1):
            current_epoch[0] = epoch
            epoch_t0 = time.time()

            avg_loss, global_step, cfg, scaler, lr_scheduler = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                cfg=cfg,
                criterion=criterion,
                device=device,
                epoch=epoch,
                global_step=global_step,
                steps_per_epoch=steps_per_epoch,
                state_factory=state_factory,
                diagnostics=diagnostics,
                ctx=ctx,
            )

            val_probs = infer_probs(model, val_loader, cfg, device)
            best_thr, val_f1 = tune_threshold(val_probs, y_val)

            current_lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_t0
            last_epoch_ran = epoch

            print(
                f"Epoch {epoch}/{PLANNED_EPOCHS} train_loss={avg_loss:.4f} "
                f"val_f1={val_f1:.4f} thr={best_thr:.2f} lr={current_lr:.6f} "
                f"time={epoch_time:.1f}s precision={cfg.resolved_policy}"
            )
            print(
                f"MLEVOLVE_EPOCH_METRIC "
                f'{json.dumps({"epoch": int(epoch), "metric": float(val_f1), "metric_name": "f1"})}'
            )

            # Per-epoch diagnostics report
            try:
                diagnostics.report(epoch=epoch)
            except Exception:
                pass

            # --- EPOCH safe point ---
            if ctx is not None and SafePointType is not None:
                ctx.control_hook.safe_point(
                    SafePointType.EPOCH,
                    epoch=epoch,
                    global_step=global_step,
                    steps_per_epoch=steps_per_epoch,
                    state_factory=state_factory,
                )

            # Best-checkpoint tracking (restore by full tensor snapshot; small model)
            if val_f1 > best_val_f1:
                best_val_f1 = float(val_f1)
                best_threshold = float(best_thr)
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {epochs_no_improve} epochs)"
                )
                break

    # Restore best snapshot for evaluation + test inference
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Final validation score with best threshold (identical to test path) ---
    final_val_probs = infer_probs(model, val_loader, cfg, device)
    final_val_pred = (final_val_probs >= best_threshold).astype(int)
    final_val_f1 = float(f1_score(y_val.astype(int), final_val_pred, zero_division=0))

    # --- Test inference (same preprocessing + same threshold) ---
    test_probs = infer_probs(model, test_loader, cfg, device)
    test_pred = (test_probs >= best_threshold).astype(int)

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    sub_df = pd.DataFrame({"id": test_df["id"].values, "target": test_pred})
    sub_df.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

    # --- Persist best checkpoint + precision state for reproducibility ---
    os.makedirs(WORKING_DIR, exist_ok=True)
    try:
        torch.save(
            {
                "model": model.state_dict(),
                "threshold": best_threshold,
                "val_f1": final_val_f1,
                "best_epoch": best_epoch,
                "last_epoch_ran": last_epoch_ran,
                "precision": precision_state_dict(cfg, scaler),
            },
            os.path.join(WORKING_DIR, "best_model.pt"),
        )
    except Exception:
        pass

    total_time = time.time() - run_start
    print(
        f"[summary] best_epoch={best_epoch} thr={best_threshold:.3f} "
        f"val_f1={final_val_f1:.4f} elapsed={total_time:.1f}s "
        f"precision={cfg.resolved_policy} fam={MODEL_FAMILY}"
    )

    return float(final_val_f1)


if __name__ == "__main__":
    _score = run_training()
    print(f"Final Validation Score: {_score}")

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
        