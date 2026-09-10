# HWDB precision prompt audit

Date: 2026-09-10. Baseline: MLEvolve `93371dd`.

Scope: RTX 5090, A10, and the four A100 PCIe/SXM4 40/80 GB records. Dataset changes are limited to those six Hardware nodes and their twelve AMP/FP16 edges. Other GPU records and generic Feature nodes are unchanged. Aggressive mode's precision allowlist remains unchanged.

## Mode contract

| Mode | Candidate floating-point computation |
| --- | --- |
| conservative | IEEE FP32 only; disable AMP, GradScaler and TF32 matmul/convolution |
| normal | FP32 plus optional, selective FP16 AMP; disable BF16, TF32 and quantized paths |

Integer indices and categorical targets keep their task-required dtype. Floating regression targets must not accidentally remain NumPy/Pandas float64. The agent LLM's own inference precision is independent of candidate precision. `disabled` remains the existing serialized alias for disabled AMP with FP32 computation.

These definitions supersede the FP32/TF32 conservative behavior described in the historical September 8 record. Shared policy code applies them throughout the candidate pipeline, not just to the six edited hardware records.

## Findings and changes

1. **Wrong hardware selection was possible.** Substring lookup could match A100 to A10. Exact normalized matches now take precedence, partial matches require word boundaries, and ambiguous variants return no match rather than guessed capabilities.
2. **The mode names did not match the requested experiment.** Conservative allowed TF32; normal allowed BF16 and TF32 and preferred BF16 on Ampere. The common allowlists, JSON decision schema, prompt instructions and explicit-code validation now agree on the stricter modes.
3. **Permitted formats were being promoted to recommendations.** Both static feature filtering and agent filtering could overwrite an edge's `recommended=false`. They now preserve that rejection. Hardware support remains eligibility, not evidence of task-specific benefit.
4. **Advice leaked across stages.** Hardware-node recommendation text was precision-filtered only in Stage 2. Positive advice is now checked in every stage, including all-stage queries; individual negative restrictions remain available in the full stage guidance.
5. **Instructions encouraged one global dtype.** Stage 2 listed every format, and generic AMP examples favored BF16. Target-specific AMP/FP16 guidance now distinguishes parameters, floating inputs/targets, forward operations, sensitive loss/reductions, backward/scaling, validation and export. FP32 remains a valid normal-mode choice.
6. **Recommendations lacked conditions.** Targeted packing, tiling, channels-last and accumulation advice is now conditional on task semantics, memory or measured benefit. Hardware-only optimization must preserve the modeling and data-exposure budget. Generic prompt rules no longer call curated advice empirical evidence or suggest reducing resolution/epochs as an ordinary hardware fallback.
7. **Prompt shortening could remove a qualification.** Stage feature guidance keeps complete conditions and recommendations together in one budgeted block. An oversized block is omitted whole. Raw and filtered contexts are preserved for diagnosis. Retrieval and total prompt limits still apply; this does not guarantee every relevant record is shown.
8. **Database ingestion could lose GPU-specific advice.** Relationship descriptions, patterns and examples now survive the JSON-to-Neo4j loader and public-result conversion. The checked-in Cypher mirror was regenerated from the JSON.

## Selective FP16 contract

Keep model parameters and floating input storage in FP32. Let autocast choose eligible forward operations. For a fragile loss/reduction, disable autocast locally and cast its inputs to FP32. Backward and optimizer updates run outside autocast; FP16 uses GradScaler, with unscale before clipping. Validation and test forwards may share a validated autocast path, while metric accumulation and exports use FP32. CPU preflight disables AMP/scaling. Do not globally call `model.half()` or cast all batch fields to half.

This follows the operation-level model in the [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html). It does not imply that every loss needs a manual FP32 region or that FP16 is faster for every workload. Retain a change only after checking finite losses/gradients, task quality and measured time against FP32 under matched conditions.

## Evidence and logging

The workflow has three internal owners: `model_design`, `datatype_precision`, and `training_evaluation`. They participate in outer draft/improve/debug workflows. Stage-aware review and repair happen before CPU preflight; preflight runs before scheduler submission/GPU probing and can trigger a targeted repair/recheck. It is not solely a final bug label after all iterations.

Existing records already contain useful counterexamples:

- `records/2026-08-31_petfinder_a100_agent_a10_comparison.md` describes an A10 backward failure with `Found dtype Double but expected Float`, attributed in that run record to float64 regression targets.
- `records/2026-08-28_petfinder_sonnet5090_liveprofile_result.json` records `policy=disabled`, `use_amp=False`, but `use_tf32=True`; disabled AMP did not mean strict FP32.
- `records/2026-09-08_conservative_precision_context_analysis.md` documents a previously fixed false FP16 classification from dormant helper code and warns that prior runs do not establish an FP16 ablation effect.

These observations establish failure mechanisms, not that a particular HWDB phrase caused them. No new end-to-end agent comparison or target-GPU benchmark was run in this audit.

Existing logs include pipeline SQLite `prompt_snapshots`, review histories, checker reports and admission summaries. New `hardware_prompt_audit` entries on each SearchNode preserve raw retrieval, filtered context, rendered guidance, precision policy and prompt SHA-256 when hardware context is attached.

Each preflight attempt now also writes:

- `working/preflight/<node-id>/candidate_attempt_<n>.py`
- `working/preflight/<node-id>/feedback_attempt_<n>.json`

Feedback joins the candidate hash, checker outcome/issues, stage decision/note board, guidance snapshots, evidence references and review history. Earlier candidate/feedback files survive a repair attempt. Missing snapshots in legacy nodes remain missing, not reconstructed evidence. Attribution starts as `unassigned`; checker infrastructure errors are distinct from candidate defects.

For a failed candidate, inspect the exact code and issue first, then compare raw guidance with the filtered/rendered prompt. Correct raw advice distorted by filtering suggests a filter issue; wrong applicable raw advice suggests an HWDB issue; correct visible advice ignored by code suggests generation or integration. Confirm that hypothesis with a controlled replay before editing shared knowledge. CPU PASS cannot prove CUDA AMP correctness, numerical quality or throughput; retain the target-GPU canary and benchmark.

## Verification

311 focused policy, target-GPU guidance, filtering, prompt, loader, journal, review/repair, configuration and preflight integration tests pass locally; two live-worker tests were excluded from that passing run for the platform failure below. JSON parses and its Cypher mirror matches regeneration. `git diff --check` passes. The six-node/twelve-edge data scope was checked against the baseline. Tests ran in the workspace `.venv`; no target-GPU benchmark was performed.

Two existing live CPU-worker integration tests cannot complete on this Mac: the pinned checker fails in `preexec_fn` while setting `RLIMIT_AS`; a direct subprocess reproduction returns `ValueError: current limit exceeds maximum limit`. They must be rerun in the supported deployment environment. The checker submodule and its resource limits were not changed. No live Neo4j/Qdrant store was reloaded; deploy the revised source and re-ingest the hardware schema before comparing database-backed runs.
