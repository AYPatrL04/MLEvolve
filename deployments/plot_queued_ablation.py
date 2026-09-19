"""Plot retained attempt timing and validation metrics without imputing outcomes."""

import json
from pathlib import Path


def plot(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = Path(root) / "results"
    traces = []
    for path in sorted(results.glob("*/runs/*/logs/ablation/attempts.json")):
        traces.append((path.parents[4].name, json.loads(path.read_text())))
    if not traces:
        return
    starts = [r["started_at"] for _, rows in traces for r in rows]
    if not starts:
        return
    origin = min(starts)
    fig, (timeline, metrics) = plt.subplots(2, 1, figsize=(14, 9))
    colors = ["#267a9f", "#b34c56", "#4b8452", "#8a629b"]
    for index, (label, rows) in enumerate(traces):
        points = []
        for row in rows:
            if row.get("ended_at"):
                timeline.barh(index, (row["ended_at"]-row["started_at"])/3600,
                              left=(row["started_at"]-origin)/3600, color=colors[index % 4], edgecolor="white")
            evidence = row.get("verification") or {}
            if row.get("verified_valid") and evidence.get("metric") is not None:
                points.append((row["attempt"], evidence["metric"]))
        if points:
            metrics.plot(*zip(*points), "o-", label=label, color=colors[index % 4])
    timeline.set_yticks(range(len(traces)), [label for label, _ in traces])
    timeline.set(xlabel="Wall time since first attempt (hours)", title="Seed 42 attempt timeline (includes CPU generation and GPU queue waits)")
    metrics.set(xlabel="Primary attempt", ylabel="Verified validation metric", title="Only verified-valid candidates")
    if metrics.lines:
        metrics.legend()
    fig.tight_layout()
    fig.savefig(Path(root) / "comparison.png", dpi=160)
    plt.close(fig)
