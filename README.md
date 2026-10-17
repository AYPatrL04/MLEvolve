# HWKD — standalone hardware knowledge and context injection

This branch contains the HWKD component extracted from `hwkd-impl` at
[`597b5b2`](https://github.com/AYPatrL04/MLEvolve/commit/597b5b23e6a2f11593f702903df331f000338e62).
The Python import name remains **`hwkg`**, and the runtime policy remains v8.19.
MLEvolve's agents, execution engine, search, scoring, provider clients, datasets,
models and experiment records are not included. Any agent can integrate the
component through the small interfaces below.

## Contents

| Path | Purpose |
| --- | --- |
| `hwkg/graph.json` | Hardware specifications, feature relationships, patterns and source references: 75 profiles, including 67 NVIDIA and 8 AMD profiles |
| `hwkg/injection.py` | Pinned graph lookup, visible-device verification and advice construction |
| `hwkg/routing.py` | Candidate scope, planner/coder phases, provider-bound prompt injection and audit |
| Other `hwkg/*.py` modules | Device eligibility, staged policies, source/evidence checks, optional calibration readers and trial feedback |
| `hwkg/injection.env.example` | Activation configuration, with no default GPU selection or timing calibration |

All **19 Python runtime modules** are byte-identical to the source commit. The
graph retains all 115 nodes and every outgoing edge of every existing node.
Five pre-existing unreachable edges pointing to the removed generic RTX 4070 ID
were removed; the graph now has 1,343 edges. Both current RTX 4070 memory-variant
profiles retain their existing relationships. The matching graph pin is updated
in the activation configuration. The original `hwkd-impl` branch is unchanged.

Graph SHA256:
`91e52fd2f7299b24c782f63f600ed4e2fa429a7159f9743d912b1dcf922efd3a`.

## Install

```bash
git clone --single-branch --branch hwkd-only https://github.com/AYPatrL04/MLEvolve.git HWKD
cd HWKD
python -m pip install .
```

Install a PyTorch build appropriate for your CUDA or ROCm environment separately.
Importing HWKD or using its disabled path needs only Python 3.10+; enabled
hardware advice requires PyTorch and exactly one visible supported GPU. Standard
PyTorch CUDA APIs are also used by its ROCm backend. This package does not install
a driver or choose a CUDA/ROCm wheel for you.

For optional calibration collection, install compatible `torchvision`, `numpy`
and `pandas` alongside PyTorch. Normal advice generation does not train a model,
run a benchmark, download weights or require those calibration dependencies.

## Select the actual GPU and enable

From this source checkout's root:

```bash
# Select the profile that matches your visible device. Example: A10.
export MLEVOLVE_HWKG_HARDWARE_ID=nvidia.ampere.a10.spec
source hwkg/injection.env.example
```

For example, A100 variants include `nvidia.ampere.a100_sxm4_80gb.spec` and
`nvidia.ampere.a100_pcie_40gb.spec`. List all profiles with:

```python
import json
from pathlib import Path
import hwkg

graph = json.loads(Path(hwkg.__file__).with_name("graph.json").read_text())
for node in graph["nodes"]:
    if node["id"].startswith("hw:"):
        print(node["id"][3:], node["properties"]["name"])
```

For an installed package used outside this checkout, configure its resource path
directly instead of sourcing a relative file:

```python
import hashlib
import os
from pathlib import Path
import hwkg

graph_path = Path(hwkg.__file__).with_name("graph.json")
os.environ.update(
    MLEVOLVE_HWKG_GRAPH_PATH=str(graph_path),
    MLEVOLVE_HWKG_GRAPH_SHA256=hashlib.sha256(graph_path.read_bytes()).hexdigest(),
    MLEVOLVE_HWKG_HARDWARE_ID="nvidia.ampere.a10.spec",  # Match your actual GPU.
    MLEVOLVE_HWKG_EVIDENCE_POLICY="v8",
    MLEVOLVE_HWKG_AUDIT_DIR="./hwkd-audit",
    MLEVOLVE_HWKG_CALIBRATION_PATH="",
    MLEVOLVE_HWKG_OPTIMIZER_CALIBRATION_PATH="",
)
```

Freeze the graph hash in experiment configurations. Device name, architecture
and per-device memory must match the selected profile; conflicting identities
raise an error. Platform profiles apply to one visible GPU, not aggregate system
memory. Unset `MLEVOLVE_HWKG_GRAPH_PATH` to disable injection.

## Use with an existing agent

### Direct context construction

```python
from hwkg.injection import hardware_guidance

task = "Image regression. Fit train.csv only; validation is eval-only."
advice = hardware_guidance(
    "draft", task_description=task, provider_prompt={"user": task}
)
if advice:
    # Add this advisory block to your existing generation prompt.
    print(advice)
```

Pass the actual provider-bound prompt so existing hardware information can be
recognized. For a refinement, supply `candidate_code`, the appropriate `stage`,
`parent_id`, `parent_execution_succeeded` and available `failure_text`.
This API constructs context; your caller owns its placement and provider call.

### Scoped injection and audit

The decorators attach context only within a candidate generation scope. They
accept an existing transport with signature `generate(prompt, cfg, ...)`, where
the prompt is a string or a dictionary containing `system` / `user` / `assistant`.
For dispatch audit, `cfg.agent.code.model` supplies the provider's model label.
The example below wraps a caller-provided transport; HWKD contains no API client.

```python
from types import SimpleNamespace
from uuid import uuid4
from hwkg.routing import hardware_context, hardware_phase, with_hardware_prompt

class MyAgent:
    def __init__(self, task_description, cfg, existing_generate):
        self.task_desc = task_description
        self.cfg = cfg
        self.generate = with_hardware_prompt(existing_generate)

    @hardware_context("draft")
    def draft(self, parent_node=None):
        code = self.generate({"user": self.task_desc}, self.cfg)
        return SimpleNamespace(id=uuid4().hex, code=code)
```

Apply `hardware_context` to candidate entry methods for `draft`, `improve`,
`debug`, `evolution`, `fusion` or `aggregation` as appropriate. Its `agent`
argument exposes `task_desc`; a `parent_node` may expose `id`, `code`, `_term_out`,
`exc_type` and `is_buggy`. A returned candidate exposes `id` and `code`.

For separate planning and coding methods, additionally use
`@hardware_phase("planner")`, `@hardware_phase("coder")` or
`@hardware_phase("plan_and_code")`; use `@hardware_phase("merge")` for assembly.
Decorated planner returns update the proposed-plan context. Calls outside a
candidate scope pass through unchanged. These interfaces do not import any
MLEvolve module.

## Advice lifecycle

| Point | Component behavior |
| --- | --- |
| Task assignment | Caller supplies the task; HWKD does not rewrite its goals, split or budgets |
| Draft / planner / coder | Intersects actual device/runtime capabilities with graph edges; offers eligible precision and optimizer options within its context budget |
| Refinement | Uses the parent source, execution status and available evidence to select an eligible trial; numerical timing claims require an exact matching external calibration |
| Debug | Performance tuning requires relevant hardware-failure evidence |
| Merge | Adds no new hardware trial |
| Execution / validation / submission | Owned entirely by the host agent; HWKD does not execute or rewrite a candidate, score it or change search selection |

Current draft context is at most 340 characters; parent context is at most 750.
Unsupported or redundant advice may be omitted. If an injected provider response
is empty, the component withdraws only its own exact advisory block on the
host's next native retry; it does not create a retry or alter a response.

The graph includes illustrative code and references to optional software. Such
examples do not bundle those libraries or project-specific helpers. In
particular, `utils.muon_optimizer` is not part of this distribution; the legacy
eligibility check also recognizes native `torch.optim.Muon` when available.

## Audit and optional calibration

`MLEVOLVE_HWKG_AUDIT_DIR` receives `prompt-audit.jsonl` for context decisions and
`dispatch-audit.jsonl` for delivery, returns and source observations. Keep these
outside the package. Advice delivery, source adoption, actual execution and
measured benefit are separate records. Audit files may contain task-specific
advice and generated-code metadata; review them before publishing.

No workstation-specific measurements or calibration JSON files are included.
`MLEVOLVE_HWKG_CALIBRATION_PATH` and
`MLEVOLVE_HWKG_OPTIMIZER_CALIBRATION_PATH` can point to separately obtained,
owned regular files. Matching is specific to GPU, backend/framework and workload;
timings are not transferred between GPU models. These variables are empty in
the example configuration.

## License and provenance

Apache-2.0; see [LICENSE](LICENSE). Existing upstream attribution is preserved.
This extraction packages the existing HWKD implementation; it does not include
MLEvolve baseline code or establish new task-performance measurements.
