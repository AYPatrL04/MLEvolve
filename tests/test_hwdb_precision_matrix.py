import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("matrix", ROOT / "deployments/run_hwdb_precision_matrix.py")
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


@pytest.mark.parametrize("mode", ["conservative", "normal"])
def test_matched_configuration(tmp_path, mode):
    cfg = matrix.make_config(ROOT, tmp_path, tmp_path / "public", "test", mode, 10800, 10)
    assert cfg["agent"]["precision_optimization_mode"] == mode
    assert cfg["agent"]["steps"] == 10
    assert cfg["agent"]["time_limit"] == 10800
    assert cfg["preflight"]["enabled"]
    assert not cfg["preflight"]["fail_open_on_internal_error"]
    assert not cfg["lesson_profiles"]["enabled"]
    assert not cfg["agent"]["use_global_memory"]
    assert cfg["scheduler"]["settings"]["prediction"]["mode"] == "branch_profile"
    assert cfg["scheduler"]["settings"]["gpu_scheduler"]["parallel_job_cap"] is None


def test_twelve_unique_competitions():
    assert len(set(matrix.COMPETITIONS)) == 12


def test_milestone_config_retains_operation_timeout_only(tmp_path):
    from omegaconf import OmegaConf
    from config import Config
    cfg = matrix.make_config(ROOT, tmp_path, tmp_path / "public", "nlp-getting-started", "conservative", 3600, 1, "deepseek-flash", milestone=True)
    parsed = OmegaConf.merge(OmegaConf.structured(Config), cfg)
    assert parsed.agent.time_limit is None
    assert cfg["agent"]["time_limit"] is None
    assert cfg["agent"]["stop_after_valid_nodes"] == 1
    assert cfg["exec"]["timeout"] == 3600
    assert cfg["exp_id"] == "nlp-getting-started"


@pytest.mark.parametrize("competition,training", [("new-york-city-taxi-fare-prediction", "labels.csv"), ("mlsp-2013-birds", "essential_data/labels.csv")])
def test_nonstandard_prepared_layouts(tmp_path, competition, training):
    (tmp_path / "description.md").write_text("task")
    (tmp_path / "test.csv").write_text("id\n1\n")
    assert not matrix.public_data_ready(tmp_path, competition)
    path = tmp_path / training
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("id,label\n1,0\n")
    assert matrix.public_data_ready(tmp_path, competition)


def test_wait_without_overall_deadline():
    process = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    descendants = set()
    try:
        assert matrix.wait_bounded(process, None, descendants) == 0
    finally:
        matrix.stop_group(process, descendants)


def test_timeout_cleans_process():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    descendants = set()
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            matrix.wait_bounded(process, 0.05, descendants)
    finally:
        matrix.stop_group(process, descendants)
    assert process.poll() is not None
