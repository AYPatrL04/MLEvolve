"""An indexed hardware graph and runtime snapshot owned by the agent process."""

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import threading


GRAPH_SHA256 = "ef5850d7ab5af271413141ade5777878fbf8dcee0750e7eba406c924e4c6bec1"
_graph_lock = threading.Lock()


@lru_cache(maxsize=4)
def _load_graph(path, expected_sha):
    raw = Path(path).read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError("HWKG graph SHA-256 does not match the pinned experiment graph")
    graph = json.loads(raw)
    nodes = {node["id"]: node for node in graph["nodes"]}
    if len(nodes) != len(graph["nodes"]):
        raise ValueError("HWKG graph contains duplicate node IDs")
    features = {key: node for key, node in nodes.items() if key.startswith("feat:")}
    edges = {}
    for edge in graph["edges"]:
        edges.setdefault(edge["from"], {})[edge["to"]] = edge
    return graph, nodes, features, edges


class HardwareKnowledge:
    """Load once; query and generate advice without a database service.

    Graph queries use only the standard library. The first guidance call lazily
    verifies the visible GPU and snapshots the installed PyTorch capabilities.
    Call refresh_runtime() after changing devices, packages or backend settings.
    """

    def __init__(self, hardware_id, *, graph_path=None, graph_sha256=None,
                 audit_dir=None, optimization_mode="normal", policy="v8"):
        if policy not in {"v6", "v7", "v8"}:
            raise ValueError(f"Unknown HWKG library policy: {policy}")
        if optimization_mode not in {"normal", "aggressive"}:
            raise ValueError("Unknown HWKG optimization mode")
        if graph_path is None:
            graph_path = Path(__file__).with_name("graph.json")
            graph_sha256 = graph_sha256 or GRAPH_SHA256
        elif not graph_sha256:
            raise ValueError("A custom HWKG graph requires graph_sha256")
        with _graph_lock:
            self._graph, nodes, self._features, self._edges = _load_graph(
                str(Path(graph_path).resolve()), graph_sha256)
        node_id = hardware_id if hardware_id.startswith("hw:") else f"hw:{hardware_id}"
        if node_id not in nodes or not node_id.startswith("hw:"):
            raise ValueError(f"HWKG hardware node not found: {node_id}")
        self._hardware = nodes[node_id]
        self.hardware_ids = tuple(key for key in nodes if key.startswith("hw:"))
        self.graph_sha256 = graph_sha256
        self.policy = policy
        self.audit_dir = Path(audit_dir) if audit_dir else None
        self.optimization_mode = optimization_mode
        self._snapshot = None
        self._lock = threading.RLock()

    @property
    def hardware(self):
        """Selected catalogue profile; this does not verify a physical device."""
        return deepcopy(self._hardware)

    @property
    def feature_edges(self):
        """Feature permissions and provenance for the selected profile."""
        return deepcopy(self._edges.get(self._hardware["id"], {}))

    @property
    def features(self):
        """Definitions and sources for features linked to the selected profile."""
        return deepcopy({key: self._features[key] for key in self._edges.get(self._hardware["id"], {})
                         if key in self._features})

    def refresh_runtime(self):
        """Replace the device/capability snapshot without reloading the graph."""
        from .injection import _matches_visible_device, _verify_visible_hardware
        from .host_policy import runtime_facts

        with self._lock:
            self._snapshot = None
            hardware = self._hardware
            device, capability, memory_mib = _verify_visible_hardware(
                self._graph, hardware, policy=self.policy)
            runtime = runtime_facts()
            edges = self._edges.get(hardware["id"], {})
            siblings = [node for node in self._graph["nodes"]
                        if node.get("properties", {}).get("gpu_count", 1) == 1
                        and _matches_visible_device(node, device)]
            runtime["hardware_profile_identity_verified"] = (
                len(siblings) == 1 and hardware["properties"].get("gpu_count", 1) == 1)
            runtime["matching_graph_profile_ids"] = [node["id"] for node in siblings]
            if len(siblings) > 1:
                hardware = {**hardware, "properties": {**hardware["properties"],
                    "memory_bandwidth_GBps": None,
                    "vram_type": "unverified driver-identical variant"}}
                sets = [{key for key, edge in self._edges.get(node["id"], {}).items()
                         if edge.get("properties", {}).get("verified") is True
                         and edge["properties"].get("support_level") == "supported"}
                        for node in siblings]
                shared = set.intersection(*sets)
                edges = {key: edge for key, edge in edges.items() if key in shared}
            self._snapshot = device, capability, memory_mib, hardware, edges, runtime

    def guidance(self, stage, task_description="", candidate_code="", **stage_context):
        """Return bounded, advisory prompt text, or None when inapplicable."""
        from .host_policy import guidance_v6
        from .adaptive_policy import guidance_v7
        from .feedback_policy import guidance_v8

        with self._lock:
            if self._snapshot is None:
                self.refresh_runtime()
            device, capability, memory_mib, hardware, edges, snapshot = self._snapshot
        runtime = dict(snapshot)
        if self.policy == "v8":
            runtime["calibration_flags"] = {}
            if (os.environ.get("MLEVOLVE_HWKG_CALIBRATION_PATH")
                    or os.environ.get("MLEVOLVE_HWKG_OPTIMIZER_CALIBRATION_PATH")):
                from .calibration import calibration_flags
                runtime["calibration_flags"] = calibration_flags()
        selected = {"v6": guidance_v6, "v7": guidance_v7, "v8": guidance_v8}[self.policy]
        text, evidence = selected(stage, task_description, candidate_code, device,
            hardware, self._features, edges, runtime, 1600,
            optimization_mode=self.optimization_mode, **stage_context)
        if self.audit_dir:
            record = {"event": "context_built", "stage": stage,
                "graph_sha256": self.graph_sha256, "visible_gpu_name": device.name,
                "visible_compute_capability": capability,
                "visible_memory_mib": round(memory_mib), "chars": len(text or ""),
                "context_sha256": hashlib.sha256((text or "").encode()).hexdigest(),
                "context_text": text,
                "graph_profile_gpu_count": hardware["properties"].get("gpu_count", 1),
                "visible_device_scope": "one GPU; platform topology and aggregate memory not verified",
                **evidence}
            with self._lock:
                self.audit_dir.mkdir(parents=True, exist_ok=True)
                with (self.audit_dir / "prompt-audit.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
        return text
