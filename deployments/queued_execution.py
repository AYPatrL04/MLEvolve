"""Durable handoff from CPU agent search to a short-lived GPU scheduler worker."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

from omegaconf import OmegaConf

from deployments.run_hwdb_precision_matrix import save


def enqueue(interpreter, code, node_id, working_dir, node):
    from engine.executor import ExecutionResult
    from engine.preflight import node_preflight_metadata

    queue = Path(os.environ["MLEVOLVE_EXECUTION_QUEUE"])
    folder = queue / uuid.uuid4().hex
    folder.mkdir(parents=True)
    cfg = json.loads(json.dumps(OmegaConf.to_container(interpreter.cfg, resolve=True), default=str))
    # Workers never receive the agent credentials.
    for role in ("code", "feedback"):
        cfg["agent"][role]["api_key"] = ""
    metadata = node_preflight_metadata(node)
    metadata.update(model_family=getattr(node, "model_family", None),
                    branch_id=getattr(node, "branch_id", None))
    log = interpreter.pipeline_logger
    save(folder / "request.json", {
        "code": code, "node_id": str(node_id), "node": metadata, "cfg": cfg,
        "working_dir": str(working_dir or interpreter.working_dir),
        "metric_maximize": interpreter.metric_maximize,
        "logger": {"db_path": str(log.db_path), "run_id": log.run_id, "mode": log.mode},
        "env": {key: os.environ[key] for key in ("MLEVOLVE_CONFIG", "MLEVOLVE_HWDB_GRAPH_PATH", "MLEVOLVE_HWDB_GRAPH_SHA256", "MLEVOLVE_HWDB_ARM", "MLEVOLVE_VLLM_CACHE_SALT") if key in os.environ},
    })
    deadline = time.monotonic() + 86400
    while not (folder / "released.json").exists():
        if (queue.parent / "STOP.json").exists():
            raise RuntimeError("GPU coordinator stopped; retained evidence requires inspection")
        if time.monotonic() > deadline:
            save(folder / "cancel.json", {"reason": "GPU handoff exceeded 24 hours"})
            raise TimeoutError("GPU queue wait exceeded 24 hours")
        time.sleep(2)
    released = json.loads((folder / "released.json").read_text())
    if not released["ok"]:
        raise RuntimeError("GPU worker infrastructure failure; see " + str(folder))
    return ExecutionResult(**json.loads((folder / "result.json").read_text()))


def execute(folder):
    from engine.executor import Interpreter
    from localml_scheduler.client import SchedulerClient
    from localml_scheduler.hardware import detect_hardware_profile
    from run import _scheduler_settings_from_cfg
    from utils.pipeline_logging import PipelineActionLogger

    for key in ("MLEVOLVE_EXECUTION_QUEUE", "MLEVOLVE_TARGET_HARDWARE_PROFILE"):
        os.environ.pop(key, None)
    profile = detect_hardware_profile()
    if profile.gpu_name != "NVIDIA A10":
        raise RuntimeError("A10 worker hardware mismatch")
    save(folder / "hardware.json", profile.to_dict())
    request = json.loads((folder / "request.json").read_text())
    os.environ.update(request["env"])
    cfg = OmegaConf.create(request["cfg"])
    client = SchedulerClient(_scheduler_settings_from_cfg(cfg, cfg.scheduler))
    interpreter = Interpreter(request["working_dir"], **OmegaConf.to_container(cfg.exec), cfg=cfg,
                              pipeline_logger=PipelineActionLogger(**request["logger"]))
    interpreter.attach_scheduler(client, cfg.scheduler)
    interpreter.set_metric_direction(request["metric_maximize"])
    service = client.create_service().start(background=True)
    try:
        result = interpreter.run(request["code"], request["node_id"],
                                 node_context=SimpleNamespace(**request["node"]))
        save(folder / "result.json", asdict(result))
    finally:
        interpreter.terminate_all_subprocesses()
        service.stop()


if __name__ == "__main__":
    import sys
    execute(Path(sys.argv[1]))
