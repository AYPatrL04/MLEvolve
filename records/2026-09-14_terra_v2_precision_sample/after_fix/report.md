The v2 graph retrieval defect and precision-safety gaps found in the fresh Terra audit are fixed. The original Terra prompt, response, code and failure evidence remain unchanged; `after_fix/draft.py` is a local correction, not a second Terra generation.

- Canonical records retain required empty fields and literal evidence references through graph and client response cleanup. All five newly ingested V100 records pass projection validation, and graph evidence now reaches the unified prompt. The rebuilt prompt contains 17 selected records.
- Normal/aggressive precision validation rejects recognizable whole-model and parameter-state FP16/BF16 casts, including aliases, model loading, default dtypes, and layer iteration. Existing GPU allowlists and the specialized Transformer Engine checks remain in place.
- Shared generation guidance applies AMP safety without requiring cold start or HWDB enablement. Hardware-only changes must preserve task design, effective batch, resolution and training budget.
- `select_validated_precision` chooses FP32 when matching measured quality/speed evidence or an explicit tolerance is absent. A comparison must match the model, data, split, initialization, preprocessing, training budget, effective batch, metric, hardware, software and backend. Worse-than-tolerance quality or no epoch-time gain rejects the optimization. The helper validates supplied comparison data; it does not authenticate external evidence documents or establish future model quality.
- New AMP scripts must record the quality decision and include recognizable finite-loss/gradient checks. Runtime diagnostics reject a contradictory autocast choice, missing FP16 loss scaling, and non-FP32 parameters in ordinary AMP. The parameter check runs once rather than scanning all parameters on every forward pass. Original diagnostics without the new decision retain their observation behavior.
- Generation now requests compatible RMSE calculation and best-checkpoint restoration. The corrected candidate fixes the RMSE calls, restores the best finite checkpoint, and implements a bounded numerical fallback that restores model/optimizer/RNG state before retrying the same batch in FP32. It never resets diagnostics counters or adds epochs.

Verification:

- [Broad regression check](pytest.txt): **389 passed in 26.48 seconds**, covering knowledge migration/retrieval, precision, preflight order, review/repair, component independence, scheduler hooks, training diagnostics, lesson publication and legacy loading.
- Source and artifact syntax checks passed. Whitespace checks exclude the exact archived prompt and unified patch, whose original blank-line whitespace is preserved.
- [Final targeted check](final_targeted_pytest.txt): **47 passed in 6.30 seconds** after tightening finite-check recognition and malformed-decision handling.
- [Corrected sample result](result.json): preflight **INCONCLUSIVE / admitted**, with the CUDA branch still requiring GPU validation. CPU construction, input contracts, training and validation checks passed.
- Controlled CPU autocast NaN injection recovered to one finite FP32 optimizer update, exactly matching the reference parameters after restoring RNG/state. [Fallback evidence](numerical_fallback.json).
- A complete two-epoch synthetic CPU workflow produced four valid predictions and restored its best checkpoint. It selected **FP32**, with quality status **unverified**, because no matched performance/quality comparison was supplied. The synthetic RMSE is an interface/workflow result, not evidence of real PetFinder quality.
- [Combined Gantt and metric-node PNG](comparison.png) shows both the original sample and local correction. Their timings are different scopes: the correction does not include another LLM call. They are not an epoch-speed comparison.

No archived experiment run was inspected, no live database was modified, and no GPU job was started. There is no fixed knowledge-token cap; exact deployment sizing remains unavailable. The graph source update is included with its regenerated Cypher mirror; publishing newly curated summaries to a live database remains an ingestion operation, separate from this code push.

To reproduce the bounded verification from the repository root:

```bash
PYTHONPATH="$PWD/nn-model-preflight-checker/src:$PWD" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 110s .venv/bin/python records/2026-09-14_terra_v2_precision_sample/after_fix/verify.py
PYTHONPATH="$PWD" .venv/bin/python records/2026-09-14_terra_v2_precision_sample/after_fix/plot.py
```

The original failing trace is frozen under its recorded source hashes. Replaying its exact prompt requires that original code version; the corrected verification above uses the current code. Source hashes for this correction are in [manifest.json](manifest.json), and the candidate edit is reviewable as [candidate.patch](candidate.patch).
