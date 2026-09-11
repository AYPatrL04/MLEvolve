import json

import pytest
import pandas as pd

from deployments import prepare_kaggle_data as data
from deployments.prepare_disaster_tweets import split_frame


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


def test_disaster_split_is_reproducible_and_labels_are_private():
    frame = pd.DataFrame({"id": range(100), "text": ["sample"] * 100, "target": [0, 1] * 50})
    train, test, private = split_frame(frame)
    assert len(train) == 80 and len(test) == 20
    assert "target" not in test and private["id"].equals(test["id"])
    assert set(train["id"]).isdisjoint(test["id"])
    assert train.equals(split_frame(frame)[0])
    assert private["target"].value_counts().to_dict() == {0: 10, 1: 10}
