import os
import random
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.training_diagnostics import TrainingDiagnostics
from utils.precision_quality import select_validated_precision

METADATA_COLUMNS = [
    "Subject Focus",
    "Eyes",
    "Face",
    "Near",
    "Action",
    "Accessory",
    "Group",
    "Collage",
    "Human",
    "Occlusion",
    "Info",
    "Blur",
]
IMAGE_SIZE = 256
SEED = 2026


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PetDataset(Dataset):
    def __init__(self, frame, image_dir, tabular, transform, has_target=True):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.tabular = torch.as_tensor(tabular, dtype=torch.float32)
        self.transform = transform
        self.has_target = has_target
        self.targets = (
            torch.as_tensor(self.frame["Pawpularity"].to_numpy(), dtype=torch.float32)
            if has_target
            else None
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.image_dir / f"{row['Id']}.jpg") as image:
            image = self.transform(image.convert("RGB"))
        item = {"image": image, "tabular": self.tabular[index], "id": row["Id"]}
        if self.has_target:
            item["target"] = self.targets[index]
        return item


class SeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class ImageTabularRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 24, 3, 2, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            SeparableBlock(24, 40, 2),
            SeparableBlock(40, 64, 2),
            SeparableBlock(64, 96, 2),
            SeparableBlock(96, 128, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.tabular_encoder = nn.Sequential(
            nn.Linear(12, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 32),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(160, 96),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(96, 1),
        )

    def forward(self, image, tabular):
        return self.head(
            torch.cat([self.image_encoder(image), self.tabular_encoder(tabular)], dim=1)
        ).squeeze(1)


class CandidateAdapter:
    def _context(self, context):
        defaults = {"lr": 2e-3, "weight_decay": 1e-4, "criterion": nn.MSELoss()}
        defaults.update(context or {})
        if defaults.get("criterion") is None:
            defaults["criterion"] = nn.MSELoss()
        return defaults

    def build_model(self, context):
        return ImageTabularRegressor()

    def build_optimizer(self, model, context):
        cfg = self._context(context)
        return AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    def _batch(self, scenario, device):
        fixture = (scenario or {}).get("fixture") or {}
        batch_size = int((scenario or {}).get("batch_size", 2))
        image_shape = tuple(fixture.get("image", [3, 256, 256]))
        tabular_shape = tuple(fixture.get("tabular", [12]))
        batch = {
            "image": torch.zeros(
                (batch_size, *image_shape), dtype=torch.float32, device=device
            ),
            "tabular": torch.zeros(
                (batch_size, *tabular_shape), dtype=torch.float32, device=device
            ),
            "target": torch.zeros(batch_size, dtype=torch.float32, device=device),
        }
        for name, shape in fixture.items():
            if name not in batch and name != "target":
                batch[name] = torch.zeros(
                    (batch_size, *tuple(shape)), dtype=torch.float32, device=device
                )
        return batch

    def build_train_batch(self, scenario, device):
        return self._batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._batch(scenario, device)

    def training_step(self, model, batch, context):
        cfg = self._context(context)
        return cfg["criterion"](
            model(batch["image"], batch["tabular"]), batch["target"]
        )

    def validation_step(self, model, batch, context):
        cfg = self._context(context)
        return cfg["criterion"](
            model(batch["image"], batch["tabular"]), batch["target"]
        )


def make_loader(dataset, batch_size, shuffle):
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        generator=generator,
    )


def predict(model, loader, device, amp_enabled):
    model.eval()
    predictions = []
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if amp_enabled
        else nullcontext()
    )
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            tabular = batch["tabular"].to(device, non_blocking=True)
            with autocast:
                output = model(image, tabular)
            predictions.extend(output.float().cpu().numpy())
    return np.clip(np.asarray(predictions), 0.0, 100.0)


def finite_training_step(model, optimizer, criterion, image, tabular, target,
                         device, amp_enabled, scaler, diagnostics):
    # Snapshot only the optional accelerated attempt; the default FP32 path has no copy cost.
    checkpoint = None
    if amp_enabled:
        checkpoint = (deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()),
                      torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
    for attempt in range(2 if amp_enabled else 1):
        optimizer.zero_grad(set_to_none=True)
        with (torch.autocast(device_type=device.type, dtype=torch.float16) if amp_enabled else nullcontext()):
            loss = criterion(model(image, tabular), target)
        finite = bool(torch.isfinite(loss))
        if finite:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        if finite:
            scaler.step(optimizer)
            scaler.update()
            diagnostics.after_update()
            finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
            finite = finite and all(bool(torch.isfinite(v).all()) for state in optimizer.state.values()
                                    for v in state.values() if torch.is_tensor(v) and v.is_floating_point())
            if finite:
                return loss, amp_enabled, scaler
        if not amp_enabled:
            raise FloatingPointError("Non-finite FP32 training step; no invalid state will be exported")
        model.load_state_dict(checkpoint[0])
        optimizer.load_state_dict(checkpoint[1])
        torch.set_rng_state(checkpoint[2])
        if checkpoint[3] is not None:
            torch.cuda.set_rng_state(checkpoint[3], device)
        amp_enabled = False
        scaler = GradScaler(enabled=False)
        diagnostics.scaler = scaler
        diagnostics.settings.update(precision="fp32", precision_reason="Numerical fallback restored finite state")
        diagnostics.settings["precision_quality"] = select_validated_precision("fp32")
    raise RuntimeError("Precision retry exhausted")


def main():
    seed_everything()
    input_dir = Path(os.environ.get("MLEVOLVE_INPUT_DIR", "./input"))
    train_df = pd.read_csv(input_dir / "train.csv")
    test_df = pd.read_csv(input_dir / "test.csv")
    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.25, random_state=SEED, shuffle=True
    )
    train_frame, valid_frame = (
        train_df.iloc[train_idx].copy(),
        train_df.iloc[valid_idx].copy(),
    )
    scaler = StandardScaler().fit(train_frame[METADATA_COLUMNS])
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_set = PetDataset(
        train_frame,
        input_dir / "train",
        scaler.transform(train_frame[METADATA_COLUMNS]),
        train_transform,
    )
    valid_set = PetDataset(
        valid_frame,
        input_dir / "train",
        scaler.transform(valid_frame[METADATA_COLUMNS]),
        eval_transform,
    )
    test_set = PetDataset(
        test_df,
        input_dir / "test",
        scaler.transform(test_df[METADATA_COLUMNS]),
        eval_transform,
        has_target=False,
    )
    batch_size = int(os.environ.get("BATCH_SIZE", 32))
    train_loader, valid_loader = make_loader(train_set, batch_size, True), make_loader(
        valid_set, batch_size, False
    )
    test_loader = make_loader(test_set, batch_size, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision_quality = select_validated_precision(os.environ.get("PRECISION", "fp16_amp"))
    amp_enabled = device.type == "cuda" and precision_quality["precision"] == "fp16_amp"
    model = ImageTabularRegressor().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scaler_amp = GradScaler(enabled=amp_enabled)
    epochs = max(1, min(2, int(os.environ.get("EPOCHS", 2))))
    settings = {
        "physical_batch_size": batch_size,
        "effective_batch_size": batch_size,
        "planned_epochs": epochs,
        "steps_per_epoch": len(train_loader),
        "model_family": "depthwise_cnn_tabular_fusion",
        "split_seed": SEED,
        "precision": "fp16_amp" if amp_enabled else "fp32",
        "precision_reason": precision_quality["reason"],
        "precision_quality": precision_quality,
    }
    best_score, best_checkpoint = float("inf"), None
    patience, bad_epochs = 2, 0
    with TrainingDiagnostics(
        model, optimizer, scaler=scaler_amp if amp_enabled else None, settings=settings
    ) as diagnostics:
        for epoch in range(epochs):
            model.train()
            losses = []
            for batch in train_loader:
                image = batch["image"].to(device, non_blocking=True)
                tabular = batch["tabular"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                loss, amp_enabled, scaler_amp = finite_training_step(
                    model, optimizer, criterion, image, tabular, target,
                    device, amp_enabled, scaler_amp, diagnostics,
                )
                losses.append(loss.detach().item())
            valid_predictions = predict(model, valid_loader, device, amp_enabled)
            valid_score = float(np.sqrt(mean_squared_error(
                valid_frame["Pawpularity"], valid_predictions
            )))
            if not np.isfinite(valid_score):
                raise FloatingPointError("Non-finite validation metric")
            if valid_score < best_score:
                best_score = valid_score
                best_checkpoint = {
                    "model": deepcopy(model.state_dict()),
                    "optimizer": deepcopy(optimizer.state_dict()),
                    "scaler": deepcopy(scaler_amp.state_dict()),
                    "epoch": epoch + 1,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                    "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate(),
                    "diagnostics": diagnostics.state_dict(),
                    "precision_quality": deepcopy(diagnostics.settings["precision_quality"]),
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
            diagnostics.report(epoch=epoch + 1)
            print(
                f"Epoch {epoch + 1}/{epochs} loss={np.mean(losses):.5f} val_rmse={valid_score:.5f}"
            )
            if bad_epochs >= patience:
                break
        if best_checkpoint is None:
            raise RuntimeError("No finite validation checkpoint")
        model.load_state_dict(best_checkpoint["model"])
        Path("./working").mkdir(parents=True, exist_ok=True)
        torch.save(best_checkpoint, "./working/petfinder_checkpoint.pt")
    final_predictions = predict(model, valid_loader, device, amp_enabled)
    score = float(np.sqrt(mean_squared_error(
        valid_frame["Pawpularity"], final_predictions
    )))
    test_predictions = predict(model, test_loader, device, amp_enabled)
    Path("./submission").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Id": test_df["Id"], "Pawpularity": test_predictions}).to_csv(
        "./submission/submission.csv", index=False
    )
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
