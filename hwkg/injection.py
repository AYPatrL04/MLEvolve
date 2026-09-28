"""A small, opt-in HWKG prompt adapter for the upstream agents."""

import hashlib
import json
import os
from pathlib import Path


_FEATURE_IDS = ("feat:amp", "feat:fp16", "feat:bf16")
_MAX_CHARS = 3200


def hardware_guidance(stage):
    """Return graph-backed advice, or None when the treatment is disabled."""
    graph_path = os.environ.get("MLEVOLVE_HWKG_GRAPH_PATH")
    if not graph_path:
        return None

    expected_sha = os.environ["MLEVOLVE_HWKG_GRAPH_SHA256"]
    hardware_id = os.environ["MLEVOLVE_HWKG_HARDWARE_ID"]
    raw = Path(graph_path).read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError("HWKG graph SHA-256 does not match the pinned experiment graph")

    graph = json.loads(raw)
    node_id = f"hw:{hardware_id}"
    hardware = next((node for node in graph["nodes"] if node["id"] == node_id), None)
    if hardware is None:
        raise ValueError(f"HWKG hardware node not found: {node_id}")

    props = hardware["properties"]
    lines = [
        f"Graph SHA-256: {actual_sha}; hardware node: {node_id}.",
        f"Device: {props['name']}; compute capability: {', '.join(props['compute_capabilities'])}; "
        f"VRAM: {props['vram_MB']} MB.",
        "These are optional hardware facts, not acceptance rules. Choose the model for the task first; "
        "use a GPU technique only when applicable, and retain a measured FP32 fallback.",
    ]
    selected = []
    edges = {edge["to"]: edge for edge in graph["edges"] if edge["from"] == node_id}
    for feature_id in _FEATURE_IDS:
        edge = edges.get(feature_id)
        if not edge:
            continue
        feature = edge["properties"]
        if feature.get("support_level") != "supported" or not feature.get("verified"):
            continue
        selected.append(feature_id)
        lines.append(f"{feature_id}: {feature.get('description') or feature.get('limitations') or 'supported'}")
        lines.extend(feature.get("recommended_patterns", [])[:2])
        lines.extend(feature.get("avoid_patterns", [])[:1])

    text = "\n".join(lines)
    if len(text) > _MAX_CHARS:
        raise ValueError("HWKG prompt exceeds the fixed character budget")

    audit_dir = os.environ.get("MLEVOLVE_HWKG_AUDIT_DIR")
    if audit_dir:
        path = Path(audit_dir)
        path.mkdir(parents=True, exist_ok=True)
        record = {
            "stage": stage,
            "graph_sha256": actual_sha,
            "hardware_node": node_id,
            "selected_feature_ids": selected,
            "chars": len(text),
            "context_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        with (path / "prompt-audit.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    return text
