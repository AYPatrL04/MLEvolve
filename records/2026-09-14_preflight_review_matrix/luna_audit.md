# Final Luna medium preflight-before-review audit

All four completed controller cells pass 17/17 assertions:

- `normal_hwdb_off`: 17/17
- `normal_hwdb_on`: 17/17
- `conservative_hwdb_off`: 17/17
- `conservative_hwdb_on`: 17/17 (retry 1)

Each completed cell used one successful initial MLEvolve draft generation,
real controller ordering, CPU preflight, an injected `training_step` failure,
real stage-owned repair, a fresh preflight admission, and static review only
after the repaired candidate had an exact fresh preflight hash. The controller
skipped review for the initially rejected candidate, then reviewed the repaired
candidate with CPU preflight evidence. Final review was approved in all cells;
final preflight was `INCONCLUSIVE` but admitted, with GPU confirmation required.
No GPU execution, scheduler submission, full training entrypoint, or result-parser
lifecycle was tested in this audit.

The off-cell review responses are grounded and concise. Normal review confirms
train-only preprocessing, original-unit RMSE, real inference, and the adapter
contract, while treating the unexecuted CUDA branch as an advisory GPU003 warning.
Conservative review confirms FP32/no-AMP enforcement and correctly treats
`(1,1,3)` image-stat reshapes as channel broadcasting. The off-cell warnings
are GPU003 and, for conservative mode, BAT001; they are advisory and not
blockers.

The normal HWDB-on review likewise reports train-only preprocessing, real
forward-pass inference, and a non-blocking best-checkpoint warning plus GPU003.
Its prompt contains the conditional RTX 5090 architecture, precision, and
training records with evidence references. HWDB-off cells perform zero optional
retrievals; HWDB-on cells retrieve the conditional records without duplication.

The conservative HWDB-on retry repair responses are valid scoped patches. One
changes all image-stat accumulators and array conversion from `float64` to
`float32`; the other removes the injected loss-path exception. The final review
confirms FP32 policy and identifies only the best-checkpoint and unverified GPU
branch as warnings.

The HWDB-on cells also report optional best-checkpoint/validation-tracking
warnings; these remain non-blocking review observations. The initial conservative
HWDB-on transport attempt timed out at 240 seconds
before producing a candidate. It remains preserved as a failed earlier attempt;
the successful retry used the existing matrix cell workflow and one completed
draft generation. Across cells, total draft transport attempts are therefore
1/1/1/2 in the order normal-off, normal-on, conservative-off,
conservative-on; each completed attempt used one draft call, and the final `2`
includes the failed timeout plus retry.

Claude CLI envelopes may include auxiliary Haiku usage alongside Sonnet. The
one-call assertion applies to the MLEvolve initial-generation transport, not to
the entire CLI service envelope. No selector, summary, merge, or staged-decision
calls ran.
