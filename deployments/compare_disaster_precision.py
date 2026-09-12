"""Paired precision execution study of a frozen DeepSeek-produced candidate.

This is not a fresh agent-search comparison. Model/data code is checksum-pinned;
one shared training loop owns both modes, without scheduler batch auto-tuning.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from deployments.run_hwdb_precision_matrix import save, stop_group, wait_bounded
from utils.training_diagnostics import TrainingDiagnostics


SOURCE_HASH = "6e309b76170c598bcc57f1a1c5a6c9ecbadd8a5de3ba2ae68798b14505bff0a2"
SOURCE = Path("/experiment/milestone-20260912/results/nlp-getting-started/conservative/runs/20260912_140839_nlp-getting-started_conservative/workspace/working/preflight/99091e8f68044b83bdbe54369f4a1324/candidate_attempt_1.py")
DATA = Path("/datasets/nlp-getting-started/prepared/public")
MODES = ("conservative", "normal")
SEEDS = (42, 43, 44)


def load_candidate(path=SOURCE):
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_HASH:
        raise ValueError("Frozen candidate source checksum mismatch")
    spec = importlib.util.spec_from_file_location("frozen_disaster_candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tensor_hash(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def forward_loss(model, inputs, targets, mode, device):
    # Autocast chooses eligible forward ops; no parameter/embedding-index cast.
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=mode == "normal" and device.type == "cuda"):
        logits = model(inputs)
    expected_dtype = torch.float16 if mode == "normal" and device.type == "cuda" else torch.float32
    if logits.dtype != expected_dtype:
        raise RuntimeError(f"Unexpected forward output dtype: {logits.dtype}")
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), targets.float())
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    return loss


def update(model, optimizer, scaler, diagnostics, inputs, targets, mode, device, grad_clip):
    optimizer.zero_grad(set_to_none=True)
    loss = forward_loss(model, inputs, targets, mode, device)
    previous = diagnostics.completed_updates
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()
    diagnostics.after_update()
    return diagnostics.completed_updates > previous


def make_training_objects(candidate, vocab_size, seed, mode, device):
    candidate.set_seed(seed)
    model = candidate.SmallTransformerClassifier(vocab_size=vocab_size).float()
    initial_hash = tensor_hash(model.state_dict().items())
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate.LR,
                                  weight_decay=candidate.WEIGHT_DECAY, betas=candidate.BETAS,
                                  eps=candidate.EPS, fused=False)
    scaler = torch.amp.GradScaler("cuda", enabled=mode == "normal" and device.type == "cuda", init_scale=1024.0)
    return model, optimizer, scaler, initial_hash


def prepare_data(candidate, data_dir):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    training, validation = candidate.make_splits(train, val_ratio=candidate.VAL_RATIO, seed=42)
    texts = candidate.build_text_series(training).tolist()
    vocab = candidate.build_vocab(texts, max_size=candidate.VOCAB_MAX_SIZE, min_freq=candidate.VOCAB_MIN_FREQ)
    datasets = [candidate.TweetDataset(candidate.build_text_series(frame).tolist(),
                frame["target"].to_numpy(dtype=np.float32) if "target" in frame else None,
                vocab, max_len=candidate.MAX_SEQ_LEN) for frame in (training, validation, test)]
    split_hash = hashlib.sha256(json.dumps({"train": training["id"].tolist(), "validation": validation["id"].tolist(),
                                           "test": test["id"].tolist()}, sort_keys=True).encode()).hexdigest()
    return datasets, max(len(vocab), 100), validation["target"].to_numpy(), test["id"].to_numpy(), split_hash


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    values = []
    # Identical FP32 validation/test inference in both training modes.
    for inputs, _ in loader:
        logits = model(inputs.to(device, non_blocking=True))
        values.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(values)


def fixed_work_benchmark(candidate, dataset, vocab_size, seed, mode, device):
    model, optimizer, scaler, _ = make_training_objects(candidate, vocab_size, seed, mode, device)
    model.train()
    inputs, targets = next(iter(DataLoader(dataset, batch_size=candidate.PHYSICAL_BATCH_SIZE)))
    inputs, targets = inputs.to(device), targets.to(device)
    with TrainingDiagnostics(model, optimizer, scaler=scaler, settings={"phase": "fixed_work_benchmark", "precision": mode}) as diagnostics:
        for _ in range(10):
            update(model, optimizer, scaler, diagnostics, inputs, targets, mode, device, candidate.GRAD_CLIP)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = diagnostics.completed_updates
        started = time.perf_counter()
        for _ in range(100):
            update(model, optimizer, scaler, diagnostics, inputs, targets, mode, device, candidate.GRAD_CLIP)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        completed = diagnostics.completed_updates - before
        if completed != 100:
            raise FloatingPointError("Fixed-work comparison requires 100 completed optimizer updates")
        result = {"warmup_updates": 10, "timed_updates": completed, "seconds": elapsed,
                  "samples_per_second": 100 * len(inputs) / elapsed,
                  "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2}
    del diagnostics, model, optimizer, scaler
    torch.cuda.empty_cache()
    return result


def run_one(root, mode, seed):
    folder = root / f"{seed}-{mode}"
    folder.mkdir(parents=True, exist_ok=False)
    os.environ["MLEVOLVE_INPUT_DIR"] = str(DATA)
    candidate = load_candidate()
    if not torch.cuda.is_available():
        raise RuntimeError("Controlled GPU comparison requires CUDA")
    gpu = torch.cuda.get_device_name(0)
    if not (gpu == "NVIDIA A10" or gpu.startswith("NVIDIA A100")):
        raise RuntimeError("Only A10/A100 are authorized for this comparison")
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    candidate.apply_precision_runtime(candidate.configure_precision({"device": str(device)}))
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 must be disabled in both modes")
    datasets, vocab_size, labels, test_ids, split_hash = prepare_data(candidate, DATA)
    model, optimizer, scaler, initial_hash = make_training_objects(candidate, vocab_size, seed, mode, device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(datasets[0], batch_size=candidate.PHYSICAL_BATCH_SIZE, shuffle=True,
                              generator=generator, num_workers=candidate.NUM_WORKERS, pin_memory=True)
    validation_loader, test_loader = [DataLoader(d, batch_size=candidate.PHYSICAL_BATCH_SIZE * 2,
                                                num_workers=candidate.NUM_WORKERS, pin_memory=True) for d in datasets[1:]]
    total_steps = len(train_loader) * candidate.EPOCHS
    warmup_steps = int(total_steps * candidate.WARMUP_RATIO)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(max((step - warmup_steps) / max(1, total_steps - warmup_steps), 0), 1)
        return candidate.MIN_LR_RATIO + (1 - candidate.MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best, best_threshold, best_state, bad_epochs = -1.0, 0.5, None, 0
    epochs = []
    settings = dict(model_family=candidate.MODEL_FAMILY, split_seed=42, initialization_seed=seed,
                    physical_batch_size=candidate.PHYSICAL_BATCH_SIZE, effective_batch_size=candidate.PHYSICAL_BATCH_SIZE,
                    planned_epochs=candidate.EPOCHS, steps_per_epoch=len(train_loader), precision=mode,
                    precision_reason="paired controlled forward-only FP16 AMP" if mode == "normal" else "paired strict IEEE FP32")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with TrainingDiagnostics(model, optimizer, scaler=scaler, settings=settings) as diagnostics:
        for epoch in range(1, candidate.EPOCHS + 1):
            model.train()
            torch.cuda.synchronize()
            epoch_started = time.perf_counter()
            for inputs, targets in train_loader:
                completed = update(model, optimizer, scaler, diagnostics, inputs.to(device, non_blocking=True),
                                   targets.to(device, non_blocking=True), mode, device, candidate.GRAD_CLIP)
                if completed:
                    scheduler.step()
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - epoch_started
            values = predict(model, validation_loader, device)
            threshold, f1 = candidate.find_best_threshold(labels, values)
            epochs.append({"epoch": epoch, "f1": f1, "threshold": threshold, "train_seconds": train_seconds})
            print(json.dumps({"mode": mode, "seed": seed, **epochs[-1]}), flush=True)
            diagnostics.report(epoch=epoch)
            if f1 > best:
                best, best_threshold, bad_epochs = f1, threshold, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
            if bad_epochs >= candidate.PATIENCE:
                break
    torch.cuda.synchronize()
    training_wall_seconds = time.perf_counter() - started
    peak_allocated_mib = torch.cuda.max_memory_allocated() / 1024**2
    runtime = diagnostics.report(status="completed")
    observed = set(runtime["autocast_dtypes_observed"])
    if runtime["parameter_dtypes"] != ["torch.float32"] or runtime["optimizer_state_dtypes"] != ["torch.float32"]:
        raise RuntimeError("Parameters and optimizer state must remain FP32")
    if mode == "normal" and "torch.float16" not in observed:
        raise RuntimeError("Normal-mode experiment did not observe real FP16 forward execution")
    if not observed <= ({"disabled"} if mode == "conservative" else {"disabled", "torch.float16"}):
        raise RuntimeError("Unexpected autocast dtype")
    if runtime["unaccounted_optimizer_calls"] or runtime["completed_updates"] <= 0:
        raise RuntimeError("Missing or unaccounted optimizer updates")
    model.load_state_dict(best_state)
    submission = pd.DataFrame({"id": test_ids, "target": (predict(model, test_loader, device) >= best_threshold).astype(int)})
    expected = pd.read_csv(DATA / "sample_submission.csv")
    if list(submission.columns) != list(expected.columns) or not submission["id"].equals(expected["id"]) or not submission["target"].isin([0, 1]).all():
        raise RuntimeError("Submission contract mismatch")
    submission.to_csv(folder / "submission.csv", index=False)
    torch.save({"model": best_state, "threshold": best_threshold, "seed": seed, "mode": mode}, folder / "best.pt")
    del diagnostics, model, optimizer, scaler, scheduler
    torch.cuda.empty_cache()
    benchmark = fixed_work_benchmark(candidate, datasets[0], vocab_size, seed, mode, device)
    result = {"status": "complete", "mode": mode, "seed": seed, "gpu": gpu, "source_sha256": SOURCE_HASH,
              "initial_weights_sha256": initial_hash, "split_sha256": split_hash,
              "best_validation_f1": best, "best_threshold": best_threshold, "epochs": epochs,
              "training_wall_seconds": training_wall_seconds, "peak_allocated_mib": peak_allocated_mib,
              "runtime": runtime, "fixed_work_benchmark": benchmark,
              "submission_sha256": hashlib.sha256((folder / "submission.csv").read_bytes()).hexdigest()}
    save(folder / "result.json", result)
    return result


def summarize(results):
    pairs = []
    for seed in SEEDS:
        pair = {r["mode"]: r for r in results if r.get("seed") == seed and r.get("status") == "complete"}
        if set(pair) != set(MODES):
            continue
        a, b = pair["conservative"], pair["normal"]
        if (a["initial_weights_sha256"] != b["initial_weights_sha256"]
                or a["split_sha256"] != b["split_sha256"]
                or a["source_sha256"] != SOURCE_HASH or b["source_sha256"] != SOURCE_HASH
                or a["gpu"] != b["gpu"]):
            raise RuntimeError("Pair does not share initial weights and split")
        pairs.append({"seed": seed, "fp32_f1": a["best_validation_f1"], "fp16_f1": b["best_validation_f1"],
                      "f1_difference": b["best_validation_f1"] - a["best_validation_f1"],
                      "skipped_updates": {m: pair[m]["runtime"]["skipped_updates"] for m in MODES},
                      "fixed_work_speedup": a["fixed_work_benchmark"]["seconds"] / b["fixed_work_benchmark"]["seconds"]})
    output = {"status": "complete" if len(pairs) == len(SEEDS) else "incomplete", "paired_results": pairs,
              "scope": "Frozen agent-produced model; precision execution comparison, not agent-search or held-out leaderboard evaluation"}
    if pairs:
        delta = statistics.mean(p["f1_difference"] for p in pairs)
        speedup = statistics.median(p["fixed_work_speedup"] for p in pairs)
        output.update(mean_f1_difference=delta, median_fixed_work_speedup=speedup,
                      caveat="Three paired seeds on one fixed internal validation split are not a general hardware recommendation; report quality/efficiency tradeoffs separately.")
        if output["status"] == "complete":
            output["stability_review_required"] = any(any(p["skipped_updates"].values()) for p in pairs)
            if not output["stability_review_required"]:
                output["provisional_quality_first_choice"] = "normal_fp16" if delta > 0 or (delta == 0 and speedup > 1.05) else "fp32"
    return output


def cpu_smoke(root):
    candidate = load_candidate()
    datasets, vocab_size, _, _, split_hash = prepare_data(candidate, DATA)
    torch.set_num_threads(2)
    device = torch.device("cpu")
    inputs, targets = next(iter(DataLoader(datasets[0], batch_size=2)))
    model, optimizer, scaler, initial_hash = make_training_objects(candidate, vocab_size, 42, "conservative", device)
    with TrainingDiagnostics(model, optimizer, scaler=scaler) as diagnostics:
        assert update(model, optimizer, scaler, diagnostics, inputs, targets, "conservative", device, candidate.GRAD_CLIP)
        assert diagnostics.completed_updates == 1
    save(root / "candidate-cpu-smoke.json", {"source_sha256": SOURCE_HASH, "initial_weights_sha256": initial_hash,
                                            "split_sha256": split_hash, "cpu_update_completed": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-one", action="store_true")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.cpu_smoke:
        cpu_smoke(args.root)
        return
    if args.run_one:
        if args.mode is None or args.seed not in SEEDS:
            parser.error("--run-one requires --mode and one of the paired --seed values")
        run_one(args.root, args.mode, args.seed)
        return
    args.root.mkdir(parents=True, exist_ok=False)
    rows = [{"seed": seed, "mode": mode, "status": "queued"} for index, seed in enumerate(SEEDS)
            for mode in (MODES if index % 2 == 0 else tuple(reversed(MODES)))]
    save(args.root / "matrix.json", rows)
    results = []
    for row in rows:
        row.update(status="running", started_at=time.time())
        save(args.root / "matrix.json", rows)
        with (args.root / f'{row["seed"]}-{row["mode"]}.log').open("x") as log:
            proc = subprocess.Popen([sys.executable, "-m", "deployments.compare_disaster_precision", "--root", str(args.root),
                                     "--run-one", "--mode", row["mode"], "--seed", str(row["seed"])],
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            descendants = set()
            try:
                code = wait_bounded(proc, 3600, descendants)
            except subprocess.TimeoutExpired:
                code = 124
            finally:
                stop_group(proc, descendants)
        path = args.root / f'{row["seed"]}-{row["mode"]}' / "result.json"
        row.update(status="complete" if code == 0 and path.exists() else "failed", exit_code=code, ended_at=time.time())
        results.append(json.loads(path.read_text()) if row["status"] == "complete" else row.copy())
        save(args.root / "matrix.json", rows)
        save(args.root / "comparison.json", summarize(results))
        print(json.dumps(row), flush=True)
    if any(row["status"] != "complete" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
