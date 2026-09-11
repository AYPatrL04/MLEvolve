from pathlib import Path
import subprocess

import pytest


HELPER = Path(__file__).resolve().parents[1] / "deployments/git_retry.sh"


@pytest.mark.parametrize("failures,expected_code,attempts", [(0, 0, 1), (1, 0, 2), (3, 56, 3)])
def test_bounded_retries(failures, expected_code, attempts):
    script = '''
source "$1"
failures=$2
attempts=0
timeout() {
  attempts=$((attempts + 1))
  [[ "$1" == --signal=TERM && "$2" == --kill-after=15s && "$3" == 1200s ]] || return 99
  if [[ "$attempts" -le "$failures" ]]; then return 56; fi
  return 0
}
sleep() { :; }
git_retry fetch --depth 1 origin pinned
code=$?
printf '%s %s' "$code" "$attempts"
exit "$code"
'''
    result = subprocess.run(["bash", "-c", script, "test", str(HELPER), str(failures)], text=True, capture_output=True)
    assert result.returncode == expected_code
    assert result.stdout == f"{expected_code} {attempts}"


def test_bootstrap_shell_syntax():
    subprocess.run(["bash", "-n", str(HELPER)], check=True)
    subprocess.run(["bash", "-n", str(HELPER.with_name("bootstrap_hwdb_precision.sh"))], check=True)
