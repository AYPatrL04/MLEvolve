"""Prepare the PetFinder public dataset for the paired A100 Qwen runs."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from deployments.prepare_kaggle_data import LOCAL, ROOT, archive, digest
from deployments.run_hwdb_precision_matrix import save, stop_group, wait_bounded


COMPETITION = "petfinder-pawpularity-score"


def update_status(row: dict) -> None:
    path = ROOT / "status.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    rows = [item for item in rows if item.get("competition") != COMPETITION]
    rows.append(row)
    save(path, rows)


def verify_ready() -> dict | None:
    ready_path = ROOT / COMPETITION / "ready.json"
    archive_path = ROOT / COMPETITION / "public.tar"
    if not ready_path.is_file() or not archive_path.is_file():
        return None
    ready = json.loads(ready_path.read_text())
    if ready.get("competition") != COMPETITION:
        raise RuntimeError("PetFinder ready metadata names a different competition")
    if digest(archive_path) != ready.get("sha256"):
        raise RuntimeError("PetFinder public archive checksum mismatch")
    return ready


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    ready = verify_ready()
    if ready is not None:
        update_status(ready)
        print(json.dumps(ready), flush=True)
        return

    folder = ROOT / COMPETITION
    folder.mkdir(parents=True, exist_ok=True)
    row = {
        "competition": COMPETITION,
        "status": "preparing",
        "started_at": time.time(),
    }
    update_status(row)
    with (folder / "prepare.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlebench.cli",
                "prepare",
                "-c",
                COMPETITION,
                "--data-dir",
                str(LOCAL),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        descendants: set[int] = set()
        try:
            code = wait_bounded(process, 3600, descendants)
        except subprocess.TimeoutExpired:
            code = 124
        finally:
            stop_group(process, descendants)
    if code:
        row.update(status="blocked_preparation", exit_code=code, ended_at=time.time())
        update_status(row)
        raise SystemExit(code)

    prepared = LOCAL / COMPETITION / "prepared"
    private = prepared / "private"
    if private.is_dir():
        archive(private, Path("/heldout") / COMPETITION / "private.tar")
    row.update(
        status="ready",
        sha256=archive(prepared / "public", folder / "public.tar"),
        exit_code=0,
        ended_at=time.time(),
    )
    save(folder / "ready.json", row)
    update_status(row)
    shutil.rmtree(LOCAL / COMPETITION, ignore_errors=True)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
