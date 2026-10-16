# HWKD: hardware knowledge and context injection

Optional hardware context for original InternScience/MLEvolve, based on commit
`9c5c8a3b23f0361708b59a401452dddc00f97189`. The implementation is on the
`hwkd-impl` branch. The component's runtime policy is v8.19.

## Contents

- `graph.json`: hardware specifications, feature edges, advice patterns and sources.
- `injection.env.example`: minimal activation and configuration.
- Python modules: hardware verification, graph lookup, staged prompt injection,
  source/evidence scope checks and advice audit. These are runtime dependencies.

Historical tests, release notes, task fixtures and workstation calibration JSON
files are not bundled. They remain available in prior Git commits. Calibration
readers remain optional runtime functionality; no measurement is performed during
context generation and no host's timing records are shipped by default.

## Enable

From this checkout's root, choose the graph profile matching the actual visible
GPU, then load the example configuration:

```bash
export MLEVOLVE_HWKG_HARDWARE_ID=nvidia.ampere.a10.spec
source hwkg/injection.env.example
```

For A100, select its actual variant, for example
`nvidia.ampere.a100_sxm4_80gb.spec` or `nvidia.ampere.a100_pcie_40gb.spec`.
Other profile IDs are the `hw:` node IDs in `graph.json`, without that prefix.
Keep the task, data, model/API settings and budgets equal to your baseline.
Unset `MLEVOLVE_HWKG_GRAPH_PATH` to disable this component.

Graph SHA256:
`ef5850d7ab5af271413141ade5777878fbf8dcee0750e7eba406c924e4c6bec1`.

## Device and runtime scope

The graph contains 75 profiles: 67 NVIDIA and 8 AMD. The selected profile must
match the single visible GPU's identity, architecture and per-GPU memory. CUDA
uses `sm_*`; ROCm uses its HIP/`gfx` identity. Multi-GPU platform entries describe
one visible member; the component does not launch distributed training.

Advice intersects graph edges with installed runtime capabilities. Native BF16
requires the relevant hardware/runtime checks; FP16 requires AMP and GradScaler;
fused AdamW requires its API and backend-kernel eligibility. Unsupported choices
are omitted. Source syntax, advice delivery and successful kernel execution are
separate evidence levels.

The configuration ships no default GPU ID or workstation speed calibration.
To use a separately obtained calibration, explicitly set
`MLEVOLVE_HWKG_CALIBRATION_PATH` or
`MLEVOLVE_HWKG_OPTIMIZER_CALIBRATION_PATH` after loading the configuration.
Records must be owned regular files and match the exact GPU, backend, framework
and workload. A measurement from one GPU is not transferred to another.

## Injection boundaries

Existing agent/planner/coder hooks supply optional context at native generation
calls. Draft advice provides actual device facts and eligible precision/optimizer
options; parent advice uses stage, code and available execution evidence. Merge
adds no new intervention; debug tuning requires a hardware-related failure.
Advice is recorded under `MLEVOLVE_HWKG_AUDIT_DIR`.

The component does not rewrite generated candidates or modify native execution,
scoring, rewards, search selection, API parameters, SDK/parser or retry counts.
It makes no extra provider calls. Suggestions use the task's ordinary training
budget, without extra benchmark or calibration passes.

## Evidence

The completed local 30+30 training comparison used frozen v8.18 on RTX5080.
It delivered 5080 capability advice but activated no numerical microcalibration.
Catalogue-wide v8.19 routing was added afterward. Cross-profile fixture checks
cover routing and eligibility; equivalent task performance on every physical
GPU has not been measured. The separate multi-GPU drafting study measured code
choice distributions using mocked hardware, without executing generated code.
