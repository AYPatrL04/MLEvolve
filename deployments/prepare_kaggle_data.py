"""Remote-only official MLE-bench preparation; never accept competition rules."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, "/runtime/repo")
from deployments.run_hwdb_precision_matrix import COMPETITIONS, save, stop_group, wait_bounded

ROOT = Path("/experiment/datasets")
LOCAL = Path("/runtime/kaggle-data")
EXISTING = {"spooky-author-identification", "jigsaw-toxic-comment-classification-challenge"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archive(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial")
    subprocess.run(["tar", "--blocking-factor=8192", "-C", str(source), "-cf", str(temporary), "."], check=True)
    temporary.replace(destination)
    return digest(destination)


def stage_public(competitions=None):
    rows = json.loads((ROOT / "status.json").read_text())
    for row in rows:
        if competitions is not None and row["competition"] not in competitions:
            continue
        override = ROOT / row["competition"] / "ready.json"
        if override.exists():
            ready = json.loads(override.read_text())
            assert ready["competition"] == row["competition"]
            row.update(ready)
        if row["status"] != "ready":
            continue
        path = ROOT / row["competition"] / "public.tar"
        if digest(path) != row["sha256"]:
            raise RuntimeError("Public dataset archive checksum mismatch")
        target = Path("/datasets") / row["competition"] / "prepared/public"
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path) as bundle:
            bundle.extractall(target, filter="data")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    rows = [{"competition": c, "status": "existing_public" if c in EXISTING else "queued"} for c in COMPETITIONS]
    save(ROOT / "status.json", rows)
    for row in rows:
        if row["status"] != "queued":
            continue
        name = row["competition"]
        row.update(status="preparing", started_at=time.time())
        save(ROOT / "status.json", rows)
        folder = ROOT / name
        folder.mkdir(exist_ok=True)
        print(json.dumps(row), flush=True)
        with (folder / "prepare.log").open("w") as log:
            command = ([sys.executable, "/launcher/prepare_disaster_tweets.py"] if name == "nlp-getting-started"
                       else [sys.executable, "-m", "mlebench.cli", "prepare", "-c", name, "--data-dir", str(LOCAL)])
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            descendants = set()
            try:
                code = wait_bounded(proc, 3600, descendants)
            except subprocess.TimeoutExpired:
                code = 124
            finally:
                stop_group(proc, descendants)
        row.update(exit_code=code, status="blocked_preparation" if code else "publishing")
        if code == 0 and name == "nlp-getting-started":
            row.update(json.loads((folder / "ready.json").read_text()))
        elif code == 0:
            prepared = LOCAL / name / "prepared"
            # Held-out labels are stored in a separate mount absent from agent pods.
            archive(prepared / "private", Path("/heldout") / name / "private.tar")
            row["sha256"] = archive(prepared / "public", folder / "public.tar")
            row["status"] = "ready"
        row["ended_at"] = time.time()
        save(ROOT / "status.json", rows)
        print(json.dumps(row), flush=True)
        # Only this job's scratch download/extraction directory is removed.
        scratch = LOCAL / name
        if scratch.exists():
            shutil.rmtree(scratch)


if __name__ == "__main__":
    stage_public() if "--stage-public" in sys.argv else main()
