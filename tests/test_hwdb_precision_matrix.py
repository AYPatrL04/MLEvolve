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


def test_timeout_cleans_process():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    descendants = set()
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            matrix.wait_bounded(process, 0.05, descendants)
    finally:
        matrix.stop_group(process, descendants)
    assert process.poll() is not None
