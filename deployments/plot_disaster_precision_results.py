"""Reproduce precision figures and narrowly scoped before/after defect checks."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mlevolve-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "records" / "disaster_precision_figures"
BASE = "93371dd64b8e2b888c1bde7cb9d90d7c03ac4e5d"
CURRENT = "032609a"
COLORS = {"conservative": "#007C91", "normal": "#B94568"}
LABELS = {"conservative": "Conservative: FP32", "normal": "Normal: selective FP16"}


def git_module(revision, path):
    source = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT, text=True)
    name = "_comparison_" + revision[:8] + "_" + Path(path).stem
    module = types.ModuleType(name)
    module.__file__ = f"{revision}:{path}"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def regression_checks():
    current = subprocess.check_output(["git", "rev-parse", CURRENT], cwd=ROOT, text=True).strip()
    rows = []
    for revision in (BASE, current):
        inspector = git_module(revision, "engine/script_introspection.py")
        logger_module = git_module(revision, "utils/pipeline_logging.py")
        batch = inspector.detect_quality_safe_physical_batch_sizes("QUALITY_SAFE_PHYSICAL_BATCH_SIZES: list[int] = [16, 32, 64]")
        lr = inspector.detect_learning_rate_scaling_policy('BATCH_LR_SCALING_POLICY: str = "fixed"')
        with tempfile.TemporaryDirectory(prefix="mlevolve-regression-") as folder:
            logger = logger_module.PipelineActionLogger(Path(folder) / "pipeline.sqlite3", run_id="comparison", mode="test")
            try:
                logger.upsert_job_packet("job", node_id="node", status="PENDING", requires_gpu=True,
                                         script_signature="original", submitted_at="original")
                logger.upsert_job_packet("job", node_id="node", status="COMPLETED", duration_seconds=27)
                packet = logger.latest_job_packet("node")
                gpu = packet["requires_gpu"]
            finally:
                logger.close()
        rows.append({"revision": revision, "checks": [
            {"name": "Typed batch-size declaration", "expected": [16, 32, 64], "observed": batch, "pass": batch == [16, 32, 64]},
            {"name": "Typed learning-rate policy", "expected": "fixed", "observed": lr, "pass": lr == "fixed"},
            {"name": "GPU provenance after partial update", "expected": 1, "observed": gpu, "pass": gpu == 1},
        ]})
    return {"scope": "Three selected regression checks, not a random sample of agent bugs or an end-to-end benchmark.",
            "baseline": "Original captured JustinLinKK hardware-awared checkout; not upstream main or latest remote HEAD.",
            "versions": rows}


def finish(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=200, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def precision_figure(runs):
    modes = ("conservative", "normal")
    seeds = sorted({r["seed"] for r in runs})
    by = {(r["seed"], r["mode"]): r for r in runs}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    fig.subplots_adjust(top=.84, bottom=.17, left=.085, right=.97, hspace=.66, wspace=.29)
    fig.suptitle("Disaster Tweets on A10: two precision strategies", x=.055, ha="left", y=.975, fontsize=19, weight="bold")
    fig.text(.055, .922, "Frozen DeepSeek-produced model | 3 paired seeds | identical split and initial weights within pairs", fontsize=11)
    for i, mode in enumerate(modes):
        fig.text(.055 + i * .37, .884, LABELS[mode], color=COLORS[mode], fontsize=12, weight="bold")

    ax = axes[0, 0]
    for seed, marker in zip(seeds, ("o", "s", "^")):
        ys = [by[seed, m]["best_validation_f1"] for m in modes]
        ax.plot([0, 1], ys, color="#9FA6AC", lw=1.3, zorder=1)
        for i, mode in enumerate(modes):
            ax.scatter(i, ys[i], color=COLORS[mode], s=65, marker=marker, zorder=3)
        ax.annotate(str(seed), (0, ys[0]), xytext=(-14, 0), textcoords="offset points", ha="right", va="center", fontsize=10)
        offset = 9 if seed == 42 else -9 if seed == 43 else 0
        ax.annotate(str(seed), (1, ys[1]), xytext=(15, offset), textcoords="offset points", ha="left", va="center", fontsize=10,
                    arrowprops={"arrowstyle": "-", "color": "#9FA6AC", "lw": .7} if offset else None)
    means = [np.mean([by[s, m]["best_validation_f1"] for s in seeds]) for m in modes]
    ax.set(title=f"A  Validation F1: means {means[0]:.5f} vs {means[1]:.5f}", ylabel="Internal validation F1 (higher is better)",
           xticks=[0, 1], xticklabels=["FP32", "Selective FP16"], xlim=(-.4, 1.4))
    ax.margins(y=.22)
    ax.text(0, -.27, "Zoomed F1 axis; labels identify initialization seeds.", transform=ax.transAxes, fontsize=10, color="#50565A")

    ax = axes[0, 1]
    ratios = [by[s, "conservative"]["fixed_work_benchmark"]["seconds"] / by[s, "normal"]["fixed_work_benchmark"]["seconds"] for s in seeds]
    bars = ax.bar(range(3), ratios, color=COLORS["normal"], width=.55)
    ax.axhline(1, color=COLORS["conservative"], lw=1.5)
    ax.text(2.45, 1.095, "FP32 reference = 1.00", ha="right", fontsize=10, color=COLORS["conservative"])
    ax.bar_label(bars, labels=[f"{r:.3f}x" for r in ratios], label_type="center", color="white", fontsize=12, weight="bold")
    ax.set(title="B  Equal-work throughput", ylabel="FP16 / FP32 throughput (higher is better)",
           xticks=range(3), xticklabels=[f"Seed {s}" for s in seeds], ylim=(0, 1.18))
    ax.text(0, -.27, f"Median: {(1-np.median(ratios))*100:.1f}% lower with FP16; 100 timed updates.", transform=ax.transAxes, fontsize=10, color="#50565A")

    ax = axes[1, 0]
    memory = [np.mean([by[s, m]["peak_allocated_mib"] for s in seeds]) for m in modes]
    bars = ax.bar([0, 1], memory, color=[COLORS[m] for m in modes], width=.55)
    ax.bar_label(bars, labels=[f"{x:.2f} MiB" for x in memory], padding=6, fontsize=11)
    ax.set(title="C  Peak allocated training memory", ylabel="PyTorch allocated memory (MiB; lower is better)",
           xticks=[0, 1], xticklabels=["FP32", "Selective FP16"], ylim=(0, max(memory)*1.25))
    ax.text(0, -.27, f"FP16 uses {(1-memory[1]/memory[0])*100:.1f}% less; not total GPU memory.", transform=ax.transAxes, fontsize=10, color="#50565A")

    ax = axes[1, 1]
    for index, mode in enumerate(modes):
        values = [by[s, mode]["training_wall_seconds"] for s in seeds]
        bars = ax.bar(np.arange(3) + (index-.5)*.34, values, width=.31, color=COLORS[mode])
        ax.bar_label(bars, labels=[f'{len_epochs(by[s, mode])} ep' for s in seeds], padding=5, fontsize=10)
    ax.set(title="D  Training + validation wall time", ylabel="Seconds (lower is better)", xticks=range(3),
           xticklabels=[f"Seed {s}" for s in seeds], ylim=(0, 24))
    ax.text(0, -.27, "Different stopping epochs; not an equal-work speed test.", transform=ax.transAxes, fontsize=10, color="#50565A")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E1E5E7", lw=.65)
        ax.set_axisbelow(True)
        ax.title.set_fontsize(12)
        ax.title.set_weight("bold")
        ax.tick_params(labelsize=10)
    fig.text(.055, .035, "Pilot only: one fixed internal split, not a leaderboard score or a general A10 recommendation.\nNormal mode permits FP32; this experiment explicitly exercises its optional FP16 path.", fontsize=10, color="#50565A")
    finish(fig, "precision-strategies")


def len_epochs(run):
    return run.get("epoch_count", len(run.get("epochs", [])))


def reliability_figure(checks, runs):
    fig = plt.figure(figsize=(12.5, 7.1))
    fig.text(.055, .94, "Reliability: what improved, and what remains unmeasured", fontsize=18, weight="bold")
    fig.text(.055, .885, "Targeted regression fixes and runtime completion are different evidence from agent bug rate.", fontsize=11)
    ax = fig.add_axes([.055, .28, .56, .52])
    ax.set_xlim(0, 5.7)
    ax.set_ylim(-.6, 3.6)
    ax.axis("off")
    ax.text(0, 3.35, "A  Replayed against original source", weight="bold", fontsize=13)
    for x, label in ((3.1, "Original\n93371dd"), (4.65, "Updated\n032609a")):
        ax.text(x+.35, 2.88, label, ha="center", fontsize=11)
    labels = ["Recognize typed\nbatch-size list", "Recognize typed\nlearning-rate policy", "Preserve GPU flag\non partial log update"]
    for i, label in enumerate(labels):
        y = 2.05 - i*.95
        ax.text(0, y+.14, label, fontsize=11, va="center")
        for j, x in enumerate((3.1, 4.65)):
            passed = checks["versions"][j]["checks"][i]["pass"]
            ax.add_patch(Rectangle((x, y-.12), .8, .55, facecolor="#DCEFE8" if passed else "#F5DEE2", edgecolor="none"))
            ax.text(x+.4, y+.15, "PASS" if passed else "FAIL", ha="center", va="center", fontsize=11, weight="bold", color="#166248" if passed else "#922E46")
    old = sum(not c["pass"] for c in checks["versions"][0]["checks"])
    new = sum(not c["pass"] for c in checks["versions"][1]["checks"])
    fig.text(.055, .265, f"Selected checks failing: {old}/3 before -> {new}/3 now", fontsize=12, weight="bold")
    fig.text(.055, .225, "Two defect areas; deliberately selected tests, not an unbiased bug sample.", fontsize=10, color="#50565A")

    ax = fig.add_axes([.65, .31, .3, .47])
    ax.set_xlim(-.5, 1.5)
    ax.set_ylim(-.65, 3.45)
    ax.axis("off")
    ax.text(-.5, 3.35, "B  Current frozen-model runs", fontsize=13, weight="bold")
    ax.text(0, 2.85, "FP32", ha="center", fontsize=11)
    ax.text(1, 2.85, "Selective FP16", ha="center", fontsize=11)
    for i, seed in enumerate((42, 43, 44)):
        for j, mode in enumerate(("conservative", "normal")):
            row = next(r for r in runs if r["seed"] == seed and r["mode"] == mode)
            assert row["runtime"]["status"] == "completed" and row["submission_hash_verified"]
            ax.scatter(j, 2-i*.9, s=700, color=COLORS[mode], marker="s")
            ax.text(j, 2-i*.9, str(seed), color="white", ha="center", va="center", fontsize=11, weight="bold")
    fig.text(.65, .265, "Runtime failures: 0/6", fontsize=12, weight="bold")
    fig.text(.65, .205, "All submissions verified; zero skipped updates.\nOne selected model, not six new agent designs.", fontsize=10, color="#50565A")
    fig.text(.055, .128, "Overall agent bug-rate reduction: NOT MEASURED", fontsize=14, weight="bold")
    fig.text(.055, .075, "No matched original-versus-updated agent search exists for this experiment.\nEarlier review rejections cannot be compared with these selected training runs as a failure-rate decrease.", fontsize=11)
    fig.text(.055, .025, "Baseline: captured JustinLinKK hardware-awared checkout, not upstream main or latest remote HEAD.", fontsize=9, color="#50565A")
    finish(fig, "reliability-evidence")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = json.loads((ROOT / "records/2026-09-12_disaster_precision_results.json").read_text())["runs"]
    checks = regression_checks()
    (OUT / "targeted-regression-comparison.json").write_text(json.dumps(checks, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 10})
    precision_figure(runs)
    reliability_figure(checks, runs)
    print(json.dumps(checks, indent=2))
    print("Figures written to", OUT)


if __name__ == "__main__":
    main()
