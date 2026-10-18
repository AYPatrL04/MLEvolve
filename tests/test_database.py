"""Library packaging, graph reuse and advice eligibility without a physical GPU."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hwkg import HardwareKnowledge
from hwkg import database, injection


GRAPH = Path(database.__file__).with_name("graph.json")
HOSTS = [node for node in json.loads(GRAPH.read_bytes())["nodes"]
         if node["id"].startswith("hw:")]
A10 = "nvidia.ampere.a10.spec"


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch):
    for key in os.environ:
        if key.startswith("MLEVOLVE_HWKG_"):
            monkeypatch.delenv(key)
    database._load_graph.cache_clear()
    injection._configured_knowledge.cache_clear()


def mock_runtime(monkeypatch, node=None):
    node = node or next(node for node in HOSTS if node["id"] == "hw:" + A10)
    props = node["properties"]
    capability = props["compute_capabilities"][0]
    rocm = capability.startswith("gfx")
    device = SimpleNamespace(name=props["name"],
        total_memory=props.get("vram_per_gpu_MB", props["vram_MB"]) * 1024**2,
        major=0 if rocm else int(capability.split(".")[0]),
        minor=0 if rocm else int(capability.split(".")[1]))
    if rocm:
        device.gcnArchName = capability + ":sramecc-:xnack-"
    runtime = dict(torch="test-runtime", cuda=None if rocm else "13.0",
        hip="7.2" if rocm else None, backend="rocm" if rocm else "cuda",
        bf16_supported=rocm or device.major >= 8, amp_api=True, grad_scaler_api=True,
        adamw_fused_api=True, adamw_fused_kernel=True,
        free_mib_snapshot=device.total_memory // 1024**2,
        sdpa_api=True, transformer_engine_package=False, torchao_package=False)
    device_probe = Mock(return_value=device)
    runtime_probe = Mock(side_effect=lambda: dict(runtime))
    monkeypatch.setattr(injection, "_visible_cuda_device", device_probe)
    monkeypatch.setattr("hwkg.host_policy.runtime_facts", runtime_probe)
    return device_probe, runtime_probe, runtime


def test_graph_queries_need_no_third_party_imports():
    code = """
import sys
sys.path.insert(0, sys.argv[1])
from hwkg import HardwareKnowledge
knowledge = HardwareKnowledge('nvidia.ampere.a10.spec')
assert len(knowledge.hardware_ids) == 75
assert knowledge.hardware['id'] == 'hw:nvidia.ampere.a10.spec'
assert 'feat:amp' in knowledge.feature_edges
assert knowledge.features['feat:amp']['properties']['source_url']
assert 'torch' not in sys.modules and 'torchvision' not in sys.modules
try:
    knowledge.guidance('draft')
except RuntimeError as error:
    assert 'requires PyTorch' in str(error)
else:
    raise AssertionError('GPU guidance must fail without PyTorch')
"""
    subprocess.run([sys.executable, "-S", "-c", code, str(GRAPH.parents[1])], check=True)


def test_instances_share_one_graph_read_and_queries_cannot_mutate_it(monkeypatch):
    read_bytes = Mock(wraps=Path.read_bytes)
    monkeypatch.setattr(Path, "read_bytes", lambda path: read_bytes(path))
    with ThreadPoolExecutor(max_workers=4) as executor:
        instances = list(executor.map(lambda _: HardwareKnowledge(A10), range(8)))
    assert read_bytes.call_count == 1
    assert all(instance._graph is instances[0]._graph for instance in instances)
    instances[0].hardware["properties"]["name"] = "changed"
    instances[0].feature_edges["feat:amp"]["properties"]["verified"] = False
    instances[0].features["feat:amp"]["properties"]["name"] = "changed"
    assert instances[1].hardware["properties"]["name"] == "NVIDIA A10"
    assert instances[1].feature_edges["feat:amp"]["properties"]["verified"] is True
    assert instances[1].features["feat:amp"]["properties"]["name"] == "amp"


def test_custom_graph_requires_matching_digest(tmp_path):
    path = tmp_path / "graph.json"
    path.write_bytes(GRAPH.read_bytes())
    with pytest.raises(ValueError, match="requires graph_sha256"):
        HardwareKnowledge(A10, graph_path=path)
    with pytest.raises(ValueError, match="SHA-256"):
        HardwareKnowledge(A10, graph_path=path, graph_sha256="0" * 64)
    knowledge = HardwareKnowledge(A10, graph_path=path, graph_sha256=database.GRAPH_SHA256)
    path.unlink()
    assert knowledge.hardware["id"] == "hw:" + A10


def test_duplicate_node_ids_are_rejected(tmp_path):
    graph = json.loads(GRAPH.read_bytes())
    graph["nodes"].append(graph["nodes"][0])
    raw = json.dumps(graph).encode()
    path = tmp_path / "graph.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="duplicate"):
        HardwareKnowledge(A10, graph_path=path, graph_sha256=hashlib.sha256(raw).hexdigest())


def test_unknown_hardware_fails_before_runtime_probe(monkeypatch):
    device_probe, runtime_probe, _ = mock_runtime(monkeypatch)
    with pytest.raises(ValueError, match="hardware node not found"):
        HardwareKnowledge("nvidia.missing")
    device_probe.assert_not_called()
    runtime_probe.assert_not_called()


@pytest.mark.parametrize("node", HOSTS, ids=lambda node: node["id"])
def test_catalogue_guidance_preserves_backend_and_bounds(monkeypatch, node):
    _, _, runtime = mock_runtime(monkeypatch, node)
    knowledge = HardwareKnowledge(node["id"])
    text = knowledge.guidance("draft", "CNN classification; train.csv (4000 rows)")
    assert text and len(text) <= 340 and "v8.19" in text
    assert node["properties"]["name"] in text
    if runtime["hip"]:
        assert "ROCm" in text and "gfx" in text and "sm_" not in text
    else:
        assert "CUDA" in text and "sm_" in text


def test_repeated_guidance_reuses_runtime_and_does_no_file_io(monkeypatch):
    device_probe, runtime_probe, _ = mock_runtime(monkeypatch)
    knowledge = HardwareKnowledge(A10)
    device_probe.assert_not_called()
    runtime_probe.assert_not_called()
    first = knowledge.guidance("draft", "CNN classification")
    no_io = Mock(side_effect=AssertionError("Unexpected file I/O"))
    monkeypatch.setattr(Path, "read_bytes", no_io)
    monkeypatch.setattr(Path, "open", no_io)
    monkeypatch.setattr(os, "open", no_io)
    monkeypatch.setattr("hwkg.calibration.calibration_flags", no_io)
    with ThreadPoolExecutor(max_workers=4) as executor:
        texts = list(executor.map(lambda _: knowledge.guidance("draft", "CNN classification"), range(8)))
    assert texts == [first] * 8
    knowledge.guidance("improve", "CNN classification", "import torch\nmodel = torch.nn.Linear(8, 2)",
                       parent_execution_succeeded=True, phase="coder")
    device_probe.assert_called_once()
    runtime_probe.assert_called_once()
    no_io.assert_not_called()


@pytest.mark.parametrize("field", ["amp_api", "bf16_supported", "adamw_fused_kernel"])
def test_refresh_updates_capability_eligibility(monkeypatch, field):
    device_probe, runtime_probe, runtime = mock_runtime(monkeypatch)
    knowledge = HardwareKnowledge(A10)
    assert "BF16" in knowledge.guidance("draft")
    runtime[field] = False
    knowledge.refresh_runtime()
    text = knowledge.guidance("draft")
    if field in {"amp_api", "bf16_supported"}:
        assert "BF16" not in text
    assert "fused=True" not in text
    assert device_probe.call_count == runtime_probe.call_count == 2


def test_failed_refresh_does_not_keep_stale_hardware(monkeypatch):
    device_probe, _, _ = mock_runtime(monkeypatch)
    knowledge = HardwareKnowledge(A10)
    knowledge.guidance("draft")
    device_probe.return_value = SimpleNamespace(name="Unknown GPU", major=8, minor=6,
                                                total_memory=24 * 1024**3)
    with pytest.raises(RuntimeError, match="does not uniquely match"):
        knowledge.refresh_runtime()
    with pytest.raises(RuntimeError, match="does not uniquely match"):
        knowledge.guidance("draft")


def test_stage_guards_and_optional_audit(monkeypatch, tmp_path):
    mock_runtime(monkeypatch)
    knowledge = HardwareKnowledge(A10, audit_dir=tmp_path)
    text = knowledge.guidance("draft")
    assert knowledge.guidance("improve", phase="merge") is None
    assert knowledge.guidance("debug", failure_text="FileNotFoundError") is None
    assert knowledge.guidance("improve", parent_execution_succeeded=False) is None
    rows = [json.loads(line) for line in (tmp_path / "prompt-audit.jsonl").read_text().splitlines()]
    assert rows[0]["context_text"] == text
    assert rows[0]["graph_sha256"] == database.GRAPH_SHA256
    assert rows[0]["added_provider_calls"] == 0
    assert rows[0]["end_to_end_gain_verified"] is False
    assert all(row["context_text"] is None for row in rows[1:])


def test_calibration_flags_are_only_read_when_configured(monkeypatch):
    mock_runtime(monkeypatch)
    flags = Mock(return_value={})
    monkeypatch.setattr("hwkg.calibration.calibration_flags", flags)
    knowledge = HardwareKnowledge(A10)
    knowledge.guidance("draft")
    flags.assert_not_called()
    monkeypatch.setenv("MLEVOLVE_HWKG_CALIBRATION_PATH", "/explicit/calibration.json")
    knowledge.guidance("draft")
    flags.assert_called_once()


def test_environment_adapter_is_opt_in_and_reuses_library(monkeypatch):
    device_probe, runtime_probe, _ = mock_runtime(monkeypatch)
    assert injection.hardware_guidance("draft") is None
    device_probe.assert_not_called()
    monkeypatch.setenv("MLEVOLVE_HWKG_GRAPH_PATH", str(GRAPH))
    monkeypatch.setenv("MLEVOLVE_HWKG_GRAPH_SHA256", database.GRAPH_SHA256)
    monkeypatch.setenv("MLEVOLVE_HWKG_HARDWARE_ID", A10)
    first = injection.hardware_guidance("draft")
    assert first == injection.hardware_guidance("draft")
    device_probe.assert_called_once()
    runtime_probe.assert_called_once()
    monkeypatch.setenv("MLEVOLVE_HWKG_GRAPH_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        injection.hardware_guidance("draft")
