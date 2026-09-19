from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from deployments.gpu_lease_guard import low_utilization
from deployments.launch_queued_ablation import manifest, ROOT
from deployments.queued_ablation_coordinator import run_worker


def test_only_workers_reserve_gpu_and_workers_have_no_credentials():
    for phase in ("prepare", "coordinator", "worker"):
        job = manifest(phase, "a" * 40, ROOT + "/queue/abcd" if phase == "worker" else None)
        pod = job["spec"]["template"]["spec"]
        container = pod["containers"][0]
        assert ("nvidia.com/gpu" in container["resources"]["requests"]) == (phase == "worker")
        assert pod["automountServiceAccountToken"] is False
        assert not any("heldout" in mount["mountPath"] for mount in container["volumeMounts"])
        if phase == "worker":
            assert job["spec"]["activeDeadlineSeconds"] < 3 * 3600
            assert not any("valueFrom" in value for value in container["env"])
            assert container["command"][-1] == ROOT + "/queue/abcd"
            assert "gpu_lease_guard.py" in container["command"][1]
        assert job["spec"]["backoffLimit"] == 0


def test_guard_trips_on_sustained_low_utilization_not_one_idle_sample():
    assert not low_utilization([(0, 0), (10, 0)], 10)
    assert low_utilization([(i, 1) for i in range(0, 601, 10)], 600)
    assert not low_utilization([(i, 75) for i in range(0, 601, 10)], 600)
    assert not low_utilization([(i, 40) for i in range(0, 601, 10)], 600)


@pytest.mark.parametrize("success", [True, False])
def test_worker_is_archived_and_deleted_on_terminal_outcome(tmp_path, success):
    class Cluster:
        deleted = False
        def request(self, path, method="GET", payload=None, raw=False):
            if method == "DELETE":
                self.deleted = True
            if raw:
                return "retained log"
            return {"status": {"conditions": [{"type": "Complete" if success else "Failed", "status": "True"}]}}
        def pods(self, job):
            return [] if self.deleted else [{"metadata": {"name": "worker"}}]
    cluster = Cluster()
    if success:
        run_worker(cluster, tmp_path / "request", "a" * 40)
    else:
        with pytest.raises(RuntimeError, match="worker failed"):
            run_worker(cluster, tmp_path / "request", "a" * 40)
    assert cluster.deleted
    assert (tmp_path / "request/container.log").read_text() == "retained log"
    assert json.loads((tmp_path / "request/released.json").read_text())["ok"] == success


def test_enqueue_preserves_preflight_and_never_exports_api_key(tmp_path, monkeypatch):
    from deployments import queued_execution
    queue = tmp_path / "queue"
    monkeypatch.setenv("MLEVOLVE_EXECUTION_QUEUE", str(queue))
    cfg = OmegaConf.create({"agent": {"code": {"api_key": "private"}, "feedback": {"api_key": "private"}}})
    interpreter = SimpleNamespace(cfg=cfg, working_dir=tmp_path, metric_maximize=True,
                                  pipeline_logger=SimpleNamespace(db_path=tmp_path / "pipeline.db", run_id="test", mode="hardware_aware"))
    node = SimpleNamespace(preflight_admitted=True, preflight_code_hash="hash")
    def finish(_):
        folder = next(queue.iterdir())
        queued_execution.save(folder / "result.json", {"term_out": ["ok"], "exec_time": 1., "exc_type": None})
        queued_execution.save(folder / "released.json", {"ok": True})
    monkeypatch.setattr(queued_execution.time, "sleep", finish)
    result = queued_execution.enqueue(interpreter, "print('ok')", "node", None, node)
    payload = json.loads(next(queue.glob("*/request.json")).read_text())
    assert payload["node"]["preflight_code_hash"] == "hash"
    assert payload["node"]["preflight_admitted"] is True
    assert payload["cfg"]["agent"]["code"]["api_key"] == ""
    assert payload["cfg"]["agent"]["feedback"]["api_key"] == ""
    assert cfg.agent.code.api_key == "private"
    assert result.term_out == ["ok"]


def test_queued_run_many_retains_preflight_gate(tmp_path, monkeypatch):
    from engine.executor import Interpreter
    from deployments import queued_execution
    monkeypatch.setenv("MLEVOLVE_EXECUTION_QUEUE", str(tmp_path))
    calls = []
    monkeypatch.setattr(queued_execution, "enqueue", lambda *args: calls.append(args) or "queued")
    interpreter = Interpreter(tmp_path, cpu_number=3, max_parallel_run=1)
    interpreter.scheduler_client = object()
    node = SimpleNamespace(preflight_admitted=False)
    results = interpreter.run_many([{"code": "pass", "id": "bad", "node": node}, {"code": "pass", "id": "ok"}])
    assert results["bad"].exc_type == "PreflightRejected"
    assert results["ok"] == "queued"
    assert len(calls) == 1
