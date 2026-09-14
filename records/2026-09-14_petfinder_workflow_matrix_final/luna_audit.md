# Final Luna medium matrix audit

The final four-cell audit passes every recorded assertion: `normal_hwdb_off`
14/14, `normal_hwdb_on` 13/13, `conservative_hwdb_off` 14/14, and
`conservative_hwdb_on` 15/15. Each cell used one initial real MLEvolve draft
generation, actual CPU preflight, a deterministic injected candidate failure,
real feedback parsing, stage-owned debug repair, and repaired CPU preflight.
All construction, data-contract, training, and validation checks passed in all
four cells. Overall status is `INCONCLUSIVE` with admission allowed and
`gpu_check_required=true`; no GPU execution, scheduler submission, or training
entrypoint execution occurred, so the audit makes no GPU performance claim or
measured model-quality claim.

The generated scripts satisfy the PetFinder contract: complete import-safe
programs, six required `CandidateAdapter` methods, image `[B,3,256,256]`,
tabular `[B,12]`, float32 targets, real forward/loss paths, main guards,
validation RMSE, submission generation, and final score formatting. Normal and
conservative cells provide explicit FP32 policy evidence and pass their final
precision assertions. HWDB-on cells retrieve conditional RTX 5090 hardware and
design records with evidence/conditions; HWDB-off cells make zero optional
hardware retrievals. Combined artifacts include persisted prompts, responses,
candidate revisions, CPU reports, call records, and the required schedule/metric
PNG.

Two historical failures are retained as evidence of enforcement. The first
`normal_hwdb_on` draft omitted `target` when the fixture listed only image and
tabular shapes, producing grouped `DAT002` failures across data-contract,
abstract-forward, CPU-training, and validation checks; continuation repair added
the target contract and passed. The first `conservative_hwdb_on` repair fixed
validation/test arrays but left three FP64 preprocessing sites; the improved
validator projected all three sites in one issue, and continuation 2 changed the
accumulators and image conversion to float32, after which the final cell passed.

The feedback and repair responses are concise and precise for their roles: the
feedback names the exact failing method, diagnostic/site, stage, and admission
consequence; repairs are scoped SEARCH/REPLACE patches preserving interfaces.
The draft responses are longer because a complete runnable script is required.

The MLEvolve transport call count is exactly one initial generation per cell.
Claude CLI envelopes also report auxiliary Haiku usage alongside Sonnet; this is
service-level usage and must not be described as the entire CLI making one model
request. No selector, summary, merge, or staged-decision calls ran.
