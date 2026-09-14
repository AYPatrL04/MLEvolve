Fresh Terra generation using the current v2 knowledge path did not produce the feared whole-model FP16 conversion. It produced a reasonable AMP implementation, but this sample does not establish that accuracy is preserved. The audit also exposed a current graph-record retrieval defect and a generated-code compatibility failure.

- Model: `gpt-5.6-terra`, one fresh subagent response, with no inherited conversation or access to old runs.
- Task: synthetic PetFinder-style image plus metadata regression; V100 32GB; normal precision mode; two epochs maximum; accuracy-first objective explicitly supplied.
- Source: the current `schema/hardware_knowledge_graph.json`, SHA-256 `b159abbc416e2b3fc764b04a26bd8e19192ac6dab1ce9ce32ce3979f58d1b537`. Code commit and hashes are in [manifest.json](manifest.json).
- Storage isolation: production graph ingestion, projection, client, filtering, and draft assembly ran against an in-memory storage adapter populated with the current V100 relationships. No live database was modified or queried. Qdrant code records, learned profiles, retrieved history, and scheduler execution were disabled. This is a current-corpus prompt/output audit, not a deployed-database integration test.
- Evidence: [exact application prompt](prompt.json), [readable prompt](prompt.txt), [untouched Terra response](response.txt), [brief design rationale](plan.txt), [extracted code](draft.py), and [combined Gantt/metric-node PNG](audit.png). The rationale is the model's public explanation of its choices, not a transcript of private reasoning.

The prompt contained 14 records covering architecture, precision, training, evaluation, and runtime. It included FP16 autocast, loss scaling, FP32 optimizer-state/sensitive-operation safeguards, an FP32 fallback, and metric-first architecture advice. There were 11 unique summary strings: two summaries repeated across records with different restriction sets. No records were cut for length. The rendered prompt was 20,721 characters; exact deployment token/window sizing was unavailable, and no guaranteed fit is claimed.

| Question | Fresh sample evidence | Assessment |
|---|---|---|
| Did FP16 advice force every layer/state to FP16? | `draft.py:267-305` uses autocast and GradScaler. Model construction uses `.to(device)` without a dtype cast. CPU inspection found all parameters, gradients, and floating AdamW state in FP32. | Not observed in this sample. |
| Did the architecture follow task reasoning? | The rationale selects a compact separable CNN, a tabular MLP, and a fusion head because of the two modalities and small fixture. Code implements both modalities, with 42,857 parameters. | Explanation and architecture agree; no measured quality/speed claim. |
| Were sensitive operations handled? | The model uses LayerNorm and MSE; loss runs inside autocast, predictions are converted to float before the external RMSE calculation. | These are recognizable framework AMP patterns. CUDA execution was not tested. |
| Is fallback real? | CPU/no-CUDA and `PRECISION=fp32` disable AMP. | Yes for availability/manual selection. No explicit automatic FP32 retry for instability. |
| Is quality preservation enforced? | There is no paired FP32 comparison, quality tolerance, or precision acceptance gate. The final checkpoint is the last epoch rather than an explicitly selected best epoch. | Accuracy-first remains an intent, not a demonstrated acceptance condition. |
| Is the output runnable on this worker? | Preflight rejects `mean_squared_error(..., squared=False)` at lines 309 and 324; the installed scikit-learn 1.9.0 raises `TypeError`. | No. The untouched draft needs a metric API repair. |

PyTorch documents operation-specific autocast, with MSE and LayerNorm among the FP32 operations, and cautions that FP16 is not suitable for every model. This supports the implementation-pattern assessment, not a claim about this sample's numerical accuracy or target execution. [PyTorch AMP reference](https://docs.pytorch.org/docs/2.14/amp.html#cuda-op-specific-behavior).

The fresh database-path audit found a material defect before generation. Five v2 records created by production V100 graph ingestion passed `validate_record`. The graph's `_sanitize_public_payload` then removed required empty fields such as `applies_when`, `fallbacks`, and sometimes `restrictions`; all five projected records failed validation as incomplete. Therefore the direct v2 graph records contributed no usable records. The current static curated-source projection still supplied the guidance seen by Terra. This means the successful AMP interpretation cannot be used to claim that the new database retrieval path works end to end. [Reproduction evidence](graph_projection_audit.json); production locations: `localml_scheduler/hardware_knowledge/store.py:483-499` and `knowledge/records.py:48-55`.

There is also a correction to the earlier source-only audit: the explicit `DO NOT manually convert to .half()` draft instruction is inside the pretrained cold-start branch (`agents/draft_agent.py:205-222`). It was absent from this fresh non-cold-start prompt. Other mixed-precision safeguards were present, and Terra followed them. [Prompt audit](prompt_audit.json).

Verification used the untouched response and required one MLEvolve generation entrypoint invocation, with no extraction retry, selector, decision, staged generation, merge, review, repair, or summarizer call. `generation_strategy` was `single_pass`, the pipeline decision was null, and stage notes were empty. The transport read the saved Terra response; internal orchestrator inference counts and usage are not exposed, so this is not a provider-level single-inference measurement. Prompt-ready-to-response-saved turnaround was 94.49 seconds, including dispatch time; CPU preflight took 16.57 seconds and the additional dtype/update smoke took 1.01 seconds.

The checker passed construction, data contract, abstract forward, CPU training, and validation. Abstract checks used the meta fallback; torch.export did not succeed because its invocation omitted the tabular argument. The integration gate correctly returned `FAIL` for the RMSE API incompatibility, while retaining `GPU003` as an inconclusive CUDA-branch advisory. A separate bounded CPU check confirmed finite loss/gradients, actual parameter updates, and FP32 state. No training entrypoint, GPU canary, full training, real validation-quality comparison, or live scheduler job ran. CPU host versions were PyTorch 2.11.0+cu130 and scikit-learn 1.9.0; target PyTorch 2.5.1/CUDA 12.4 were explicit sample assumptions. [Checks](checks.json), [preflight result](preflight.json).

The practical follow-up is to preserve the canonical record schema through graph cleanup, make the AMP/whole-model-cast distinction an unconditional validation rule, and define how a precision change earns acceptance under the task-quality requirement. This audit leaves production code and the generated sample unchanged.

To repeat local checks without another model call, from the repository root:

```bash
PYTHONPATH="$PWD/nn-model-preflight-checker/src:$PWD" CUDA_VISIBLE_DEVICES="" .venv/bin/python records/2026-09-14_terra_v2_precision_sample/audit_sample.py finish
PYTHONPATH="$PWD/nn-model-preflight-checker/src:$PWD" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 75s .venv/bin/python records/2026-09-14_terra_v2_precision_sample/verify_sample.py
PYTHONPATH="$PWD" .venv/bin/python records/2026-09-14_terra_v2_precision_sample/plot_sample.py
```

The frozen prompt must match exactly during replay. Do not run `prepare` over this archived sample after changing the source; use a new output directory for a new generation. Full current-source V100 inputs are preserved in [source_snapshot.json](source_snapshot.json); selected records and references are in [knowledge_records.json](knowledge_records.json).
