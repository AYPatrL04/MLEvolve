import json

import pytest

from deployments import prepare_kaggle_data as data


def test_archive_digest_detects_changes(tmp_path):
    path = tmp_path / "public.tar"
    path.write_bytes(b"original")
    initial = data.digest(path)
    path.write_bytes(b"changed")
    assert data.digest(path) != initial


def test_corrupt_archive_is_rejected_before_extracting(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    folder = tmp_path / "nlp-getting-started"
    folder.mkdir()
    (folder / "public.tar").write_bytes(b"invalid")
    (tmp_path / "status.json").write_text(json.dumps([
        {"competition": folder.name, "status": "ready", "sha256": "wrong"}]))
    with pytest.raises(RuntimeError, match="checksum"):
        data.stage_public()


def test_unverified_data_is_never_staged(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    (tmp_path / "status.json").write_text(json.dumps([
        {"competition": "nlp-getting-started", "status": "blocked_preparation"},
        {"competition": "spooky-author-identification", "status": "existing_public"}]))
    data.stage_public()
