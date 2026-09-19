# HWKG merge audit and seed-42 experiment

The starting fork was b4352eac03aa55835622031a4dfbfd7451b68475.
The latest fetched upstream was 1bf8d2f27bce5b289cb29caec736722b2efd3d8a,
including the post-merge fix 1955d75. The merge retains strict conservative
IEEE FP32 and normal FP32/selective FP16; BF16 and TF32 are not re-enabled.

## Confirmed integration defects

- Generic response sanitizers removed required empty fields and literal evidence
  URLs from design-knowledge-v2 records. Upstream repaired two paths; the scheduler
  client had the same missing exemption and is repaired here too.
- The v2 prompt projection exempted every record marked `hard` from precision
  filtering. External advice can no longer override the configured dtype policy
  by declaring that strength.
- Exact hardware matching missed CUDA's `NVIDIA GeForce RTX 5090` against the
  catalog's `GeForce RTX 5090`. Vendor-prefix normalization now matches it while
  retaining the A10/A100 distinction and ambiguous-SKU rejection.
- The old GPU bootstrap checked out the main repository without updating its
  submodules. The new CPU preparation explicitly initializes the pinned submodules.
- The old ablation loop could extend indefinitely until it found one valid node.
  This study uses exactly ten primary slots per cell; a zero-yield cell remains
  a valid experimental outcome, not a reason to manufacture extra attempts.

## Structure assessment

All three snapshots have 112 nodes and 1197 edges. The useful v2 change is a
canonical prompt record carrying applicability, restrictions, fallbacks, source
identity/hash, evidence and verification status together, rather than graph size.
That is a more auditable representation. It does not itself demonstrate fewer
bugs, better task quality, or faster execution.

`audit.json` and the named text/JSON files contain offline projections for A10,
A100 SXM4 80GB and RTX 5090 using the same repaired filter. These are retrieval
fixtures, not prompts from new executed candidates. For A10 the revised fixture
contains 28/29 records and 12961/13322 characters for conservative/normal. The
upstream fixture has 26 records in each mode and 11985/11995 characters. More
text is not evidence of better guidance; actual agent prompts and execution
outcomes must decide whether the added content helps.

Upstream's new quality gate chooses FP32 when matched quality/speed evidence is
missing. Normal therefore permits FP16 but does not guarantee its use. The
experiment must report observed autocast and parameter dtypes, not infer them
from the mode name. No synthetic quality measurement or extra unbudgeted
comparison training is authorized by a hardware recommendation.

## New experiment

- Prefix: `hwkg-merge-s42-20260919`.
- Public task: Disaster Tweets, retaining the preceding content-ablation scope.
- Seed 42; latest-upstream/original vs revised HWKG content; conservative vs normal.
- Four cells, ten primary attempts each. Filters, agent, v2 projection, preflight,
  scheduler and execution stack are held fixed. This isolates content effects;
  it cannot independently identify the causal effect of the filter.
- DeepSeek agents run in a CPU-only Nautilus coordinator. No held-out data mount.
- The local dispatcher uses existing kube credentials; none are copied into the
  pods. The Mac must remain available to submit the next worker. If it is offline,
  the CPU queue waits without reserving a GPU.
- An A10 worker is requested only for the initial hardware probe or an admitted
  candidate. It has no API Secret or service-account token. Its real scheduler
  still uses branch-profile admission, without a fixed parallel-job cap.
- GPU telemetry starts at container startup. A ten-minute mean below 40% stops
  the worker, preserves telemetry, and stops the study for inspection. Each worker
  also has a 75-minute Kubernetes deadline and no automatic retries. Short job
  durations alone are not a substitute for genuine utilization.
- Worker logs and results persist on the experiment PVC; the dispatcher archives
  evidence locally before acknowledging release. Job deletion is confirmed before
  the CPU search advances. Prompt generation and review never reserve a GPU.
- First-valid wall time includes queue/provisioning delays. Worker execution time
  and GPU occupancy must be reported separately from those delays.

Local core validation passed 255 tests, with further targeted merge and launcher
tests passing. Two full preflight subprocess tests fail on this macOS environment
at the subprocess resource-limit hook; the Linux CPU preparation reruns those
tests before allowing any GPU worker. Deployment outcomes belong in the separate
run records, not this static audit.
