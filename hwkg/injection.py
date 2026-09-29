"""A small, opt-in HWKG prompt adapter for the upstream agents."""

import ast
import hashlib
import json
import os
import re
import textwrap
from pathlib import Path


_FEATURE_IDS = ("feat:amp", "feat:fp16", "feat:bf16")
_MAX_CHARS = 3200
_IMAGE_TASK = re.compile(r"\b(?:images?|photos?|pictures?|jpe?gs?|pngs?|vision)\b", re.I)
_LANGUAGE_MODEL_TASK = re.compile(
    r"\b(?:language model(?:ing)?|next[- ]token|autoregressive|token prediction)\b", re.I
)
_LONG_TEXT = re.compile(r"\b(?:documents?|corpus|sequences?|long context)\b", re.I)


def _visible_cuda_device():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("HWKG requires PyTorch to verify the visible GPU") from exc

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("HWKG requires exactly one visible CUDA GPU")
    return torch.cuda.get_device_properties(0)


def _normalized_gpu_name(name):
    return re.sub(r"[^a-z0-9]", "", re.sub(r"^nvidia\s+", "", name.casefold()))


def _matches_visible_device(hardware, device):
    if not hardware["id"].startswith("hw:nvidia."):
        return False
    props = hardware["properties"]
    capability = f"{device.major}.{device.minor}"
    memory_mib = device.total_memory / (1024 * 1024)
    expected_memory = props["vram_MB"]
    return (
        _normalized_gpu_name(device.name) == _normalized_gpu_name(props["name"])
        and capability in props["compute_capabilities"]
        and 0.85 * expected_memory <= memory_mib <= 1.15 * expected_memory
    )


def _verify_visible_hardware(graph, hardware):
    device = _visible_cuda_device()
    capability = f"{device.major}.{device.minor}"
    memory_mib = device.total_memory / (1024 * 1024)
    matches = [node for node in graph["nodes"] if _matches_visible_device(node, device)]
    if len(matches) != 1 or matches[0]["id"] != hardware["id"]:
        raise RuntimeError(
            f"HWKG visible GPU {device.name} ({capability}, {memory_mib:.0f} MiB) "
            f"does not uniquely match pinned graph node {hardware['id']}"
        )
    return device, capability, memory_mib


def _has_hidden_matrix_model(candidate_code):
    if not candidate_code:
        return False
    try:
        tree = ast.parse(candidate_code)
    except SyntaxError:
        return False

    def name(node):
        if isinstance(node, ast.Attribute):
            return node.attr
        return node.id if isinstance(node, ast.Name) else ""

    def constructors(node):
        return [name(item.func) for item in ast.walk(node) if isinstance(item, ast.Call)]

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and any(name(base) == "Module" for base in node.bases):
            layers = constructors(node)
            if layers.count("Linear") >= 2 or any(
                layer in layers for layer in ("TransformerEncoderLayer", "TransformerDecoderLayer")
            ):
                return True
        if isinstance(node, ast.Call) and name(node.func) == "Sequential":
            if constructors(node).count("Linear") >= 2:
                return True
    return False


def _muon_available():
    helper = Path(__file__).resolve().parents[1] / "utils" / "muon_optimizer.py"
    if helper.is_file():
        return True
    try:
        import torch
    except ImportError:
        return False
    return callable(getattr(torch.optim, "Muon", None))


def _exploratory_relevance(feature_id, task_description, candidate_code):
    task_description = str(task_description or "")
    if feature_id == "feat:nvimagecodec_gpu_decode":
        return bool(_IMAGE_TASK.search(task_description)), "no image-task evidence"
    if feature_id == "feat:muon_optimizer":
        if not _has_hidden_matrix_model(candidate_code):
            return False, "no explicit hidden-matrix model in candidate code"
        if not _muon_available():
            return False, "no Muon implementation available in this environment"
    if feature_id == "feat:cut_cross_entropy":
        return bool(_LANGUAGE_MODEL_TASK.search(task_description)), "no language-model task evidence"
    if feature_id == "feat:dataset_decomposition":
        relevant = _LANGUAGE_MODEL_TASK.search(task_description) and _LONG_TEXT.search(task_description)
        return bool(relevant), "no long-sequence language-model task evidence"
    return True, ""


def hardware_guidance(stage, task_description="", candidate_code=""):
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
    device, capability, memory_mib = _verify_visible_hardware(graph, hardware)

    props = hardware["properties"]
    lines = [
        f"Graph SHA-256: {actual_sha}; hardware node: {node_id}.",
        f"Device: {props['name']}; compute capability: {', '.join(props['compute_capabilities'])}; "
        f"VRAM: {props['vram_MB']} MB.",
        f"Runtime-visible GPU memory: {memory_mib:.0f} MiB total (not free); use this for memory planning.",
        "These are optional hardware facts, not acceptance rules. Choose the model for the task first; "
        "use a GPU technique only when applicable, and retain a measured FP32 fallback.",
    ]
    selected = []
    exploratory = []
    background = []
    background_reasons = {}
    seen_lines = set(lines)

    def append_once(line):
        if line and line not in seen_lines:
            lines.append(line)
            seen_lines.add(line)

    edges = {edge["to"]: edge for edge in graph["edges"] if edge["from"] == node_id}
    verified_groups = {}
    for feature_id in _FEATURE_IDS:
        edge = edges.get(feature_id)
        if not edge:
            continue
        feature = edge["properties"]
        if feature.get("support_level") != "supported" or not feature.get("verified"):
            continue
        selected.append(feature_id)
        description = feature.get("description") or feature.get("limitations") or "supported"
        verified_groups.setdefault(description, []).append((feature_id, feature))
    for description, features in verified_groups.items():
        append_once(f"{', '.join(feature_id for feature_id, _ in features)}: {description}")
        for _, feature in features:
            for pattern in feature.get("recommended_patterns", [])[:2]:
                append_once(pattern)
            for pattern in feature.get("avoid_patterns", [])[:1]:
                append_once(pattern)

    feature_nodes = {node["id"]: node for node in graph["nodes"] if node["id"].startswith("feat:")}
    for feature_id, edge in sorted(edges.items()):
        feature = edge["properties"]
        if feature.get("verified") or feature.get("support_level") not in ("supported", "experimental"):
            continue
        if feature.get("support_level") == "supported" and feature.get("recommended"):
            relevant, reason = _exploratory_relevance(feature_id, task_description, candidate_code)
            if relevant:
                exploratory.append(feature_id)
                continue
            background_reasons[feature_id] = reason
        background.append(feature_id)

    if exploratory:
        lines.append(
            "Lower-confidence, unverified possibilities for this GPU only: not defaults or proven "
            "speedups. Check task fit and installed dependencies, then benchmark against a baseline; "
            "graph examples do not imply their helpers exist in this repository."
        )
        for feature_id in exploratory:
            feature = feature_nodes.get(feature_id, {}).get("properties", {})
            summary = feature.get("description") or feature.get("design_summary") or feature_id
            summary = textwrap.shorten(" ".join(summary.split()), width=150, placeholder="...")
            pattern = feature.get("recommended_patterns", [])
            advice = f" Optional trial: {textwrap.shorten(pattern[0], width=130, placeholder='...')}" if pattern else ""
            lines.append(f"{feature_id} [unverified]: {summary}{advice}")
    if background:
        lines.append(
            "Other unverified or experimental GPU-linked entries (background only, not advice): "
            + ", ".join(background) + "."
        )

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
            "visible_gpu_name": device.name,
            "visible_compute_capability": capability,
            "visible_memory_mib": round(memory_mib),
            "selected_feature_ids": selected,
            "exploratory_feature_ids": exploratory,
            "background_unverified_feature_ids": background,
            "background_reasons": background_reasons,
            "chars": len(text),
            "context_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        with (path / "prompt-audit.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    return text
