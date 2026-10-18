# HWKG: in-process hardware knowledge

The `paper` branch is a standalone Python library for hardware-aware coding
agents. It retains the hardware graph and advice policies from MLEvolve's
`hwkd-impl` branch; the search engine, agents, provider clients, web assets and
training dependency bundles are excluded.

The agent imports `hwkg` in its own process. The graph is loaded, checksum-checked
and indexed once per path/digest, then shared in memory (up to four cached graphs).
There is no database server, HTTP/RPC layer, worker process or extra LLM call.
The package has no mandatory third-party dependencies. GPU advice uses the
agent environment's existing PyTorch installation; graph queries need only Python.

## Install locally

Python 3.10+ and the build tools `setuptools>=61` and `wheel` are required.
For development, from this checkout:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

For a local wheelhouse (no public PyPI publication or package server needed):

```bash
python -m pip wheel --no-deps --no-build-isolation --wheel-dir ../wheelhouse .
python -m pip install --no-index --find-links ../wheelhouse hwkg==0.1.0
```

Once this branch is committed and available at your repository URL, a consuming
agent repository can pin it as a Git submodule and install the same library:

```bash
git submodule add -b paper <repository-url> external/hwkg
python -m pip install --no-deps --no-build-isolation ./external/hwkg
```

Commit the consuming repository's `.gitmodules` and gitlink to pin the library
revision. Subsequent checkouts use `git submodule update --init --recursive`.

## Use inside the agent process

Create one instance per selected device/profile and reuse it across candidates:

```python
from hwkg import HardwareKnowledge

knowledge = HardwareKnowledge("nvidia.ampere.a10.spec")

# Catalogue queries do not import PyTorch or probe the GPU.
profile = knowledge.hardware
feature_edges = knowledge.feature_edges
feature_definitions = knowledge.features
profile_ids = knowledge.hardware_ids

# At the agent's ordinary prompt construction boundary:
advice = knowledge.guidance(
    "draft",
    task_description="Train an image classifier within the task budget.",
    phase="plan_and_code",
)
if advice:
    prompt += "\n\n# Hardware knowledge (advisory)\n" + advice
```

Select the actual visible GPU profile, not the example A10 unless that is your
device. The first guidance call verifies exactly one visible GPU's name,
architecture and per-device memory against the graph, and snapshots runtime
capabilities. Missing PyTorch, a mismatched device or an ambiguous unsupported
profile fails explicitly. CUDA and ROCm devices are supported.

The snapshot is reused for later calls. Call `knowledge.refresh_runtime()` after
changing device visibility, installed packages or backend flags. Free memory is
a snapshot, not a live allocation limit; recheck it at execution. Refresh failure
invalidates the old snapshot. Graph contents remain fixed for the instance;
custom graphs require both `graph_path` and their pinned `graph_sha256`.

For later candidates, pass `candidate_code`, `phase`, `proposed_plan`,
`failure_text`, `parent_id`, `parent_execution_succeeded` and `blocked_profiles`
as available. A successful parent or source flag alone does not prove a speedup.
The default v8.19 policy preserves task/metric/budget boundaries, uses bounded
context, skips unrelated debug tuning and adds no merge intervention.

## Optional compatibility and evidence

Existing `hwkg.injection.hardware_guidance` and `hwkg.routing` imports remain
available for MLEvolve integrations. Their environment-driven activation stays
opt-in; `hwkg/injection.env.example` selects the graph from a source checkout.
Modern v6/v7/v8 hooks reuse cached library instances. Historical policies remain
available through the compatibility adapter.

The direct API requires no environment variables and writes no audit files by
default. Pass `audit_dir="runs/hwkg-audit"` to record advice provenance. Explicit
`MLEVOLVE_HWKG_CALIBRATION_PATH` and
`MLEVOLVE_HWKG_OPTIMIZER_CALIBRATION_PATH` remain optional; no calibration files
are discovered or read by default, and `torchvision` is only needed when enabling
calibration-backed guidance. Existing ownership, file-size, host, runtime and
workload matching checks remain in place. No calibration runs during guidance.

The unchanged graph contains 75 hardware profiles (67 NVIDIA, 8 AMD), 40 feature
nodes and 1,348 edges. Its SHA-256 is
`ef5850d7ab5af271413141ade5777878fbf8dcee0750e7eba406c924e4c6bec1`.
Graph support, source observations, runtime eligibility and measured task gains
remain distinct evidence levels. The earlier RTX5080 study used v8.18; it does
not establish this refactor's performance or results on every supported GPU.

## Verify

```bash
python -m pytest -q tests
python -m compileall -q hwkg
```

Tests use mocked hardware/runtime facts; physical GPU task quality and training
performance require separate validation. This library retains the original
Apache-2.0 license and attribution in `LICENSE`.
