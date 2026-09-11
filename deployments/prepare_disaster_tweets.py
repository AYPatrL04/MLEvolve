"""Explicit custom holdout for Disaster Tweets, which is absent from MLE-bench."""

import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split


def split_frame(frame):
    assert {"id", "text", "target"} <= set(frame.columns)
    assert frame["id"].is_unique and frame["target"].notna().all()
    assert set(frame["target"].unique()) == {0, 1}
    train, heldout = train_test_split(frame, test_size=0.2, random_state=42, stratify=frame["target"])
    return train, heldout.drop(columns="target"), heldout[["id", "target"]]


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    sys.path.insert(0, "/launcher")
    from prepare_kaggle_data import archive, digest, save

    name = "nlp-getting-started"
    root = Path("/runtime/kaggle-data") / name
    root.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(name, path=str(root), quiet=True)
    bundles = list(root.glob("*.zip"))
    assert len(bundles) == 1
    with zipfile.ZipFile(bundles[0]) as bundle:
        names = [n for n in bundle.namelist() if PurePosixPath(n).name == "train.csv"]
        assert len(names) == 1
        with bundle.open(names[0]) as stream:
            train, test, answers = split_frame(pd.read_csv(stream))
    public, private = root / "prepared/public", root / "prepared/private"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    train.to_csv(public / "train.csv", index=False)
    test.to_csv(public / "test.csv", index=False)
    answers.to_csv(private / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], "target": 0}).to_csv(public / "sample_submission.csv", index=False)
    (public / "description.md").write_text(
        "# Disaster Tweets: Custom Precision Benchmark\n\n"
        "Predict target=1 for tweets about real disasters, target=0 otherwise. "
        "Use text, keyword and location as appropriate. The metric is binary F1. "
        "Write submission.csv with id,target columns and binary predictions for test.csv. "
        "Use only train.csv labels for model selection, with your own internal validation split.\n\n"
        "This is a custom benchmark, not an official MLE-bench competition definition or "
        "Kaggle leaderboard evaluation. The original labeled Kaggle training data was split "
        "80/20 with stratification and seed 42. Both precision modes use this same holdout; "
        "test labels are unavailable to the agents. No leaderboard submission is made.\n")
    provenance = {"definition": "custom_stratified_holdout", "seed": 42, "test_fraction": 0.2,
                  "metric": "binary_f1", "source_zip_sha256": digest(bundles[0]),
                  "train_rows": len(train), "test_rows": len(test)}
    destination = Path("/experiment/datasets") / name
    destination.mkdir(parents=True, exist_ok=True)
    save(destination / "split_provenance.json", provenance)
    archive(private, Path("/heldout") / name / "private.tar")
    checksum = archive(public, destination / "public.tar")
    save(destination / "ready.json", {"competition": name, "status": "ready", "sha256": checksum,
                                       "definition": "custom_stratified_holdout"})
    print(json.dumps({"competition": name, "status": "ready", **provenance}), flush=True)


if __name__ == "__main__":
    main()
