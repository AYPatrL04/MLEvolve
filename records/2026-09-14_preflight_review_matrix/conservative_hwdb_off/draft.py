import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

try:
    from utils.training_diagnostics import TrainingDiagnostics
except ImportError:

    class TrainingDiagnostics:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def after_update(self):
            pass

        def report(self, epoch=None):
            pass

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            pass


# ---------------------------------------------------------------------------
# Import-safe constants and definitions
# ---------------------------------------------------------------------------
IMAGE_SIZE = 256
TABULAR_COLS = [
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
TARGET_COL = "Pawpularity"
SEED = 42
EPOCHS = 2
BATCH_SIZE = 4

DATA_ROOT = os.environ.get(
    "MLEVOLVE_INPUT_DIR",
    "/home/justin/MLEvolve/records/2026-09-14_preflight_review_matrix/"
    "conservative_hwdb_off/workspace/input",
)


def compute_image_stats(image_dir, ids, size=IMAGE_SIZE):
    """Compute per-channel mean/std over a set of images (training only)."""
    sum_ = np.zeros(3, dtype=np.float64)
    sumsq = np.zeros(3, dtype=np.float64)
    count = 0
    for img_id in ids:
        path = os.path.join(image_dir, f"{img_id}.jpg")
        img = Image.open(path).convert("RGB").resize((size, size))
        arr = np.asarray(img, dtype=np.float64) / 255.0
        flat = arr.reshape(-1, 3)
        sum_ += flat.sum(axis=0)
        sumsq += (flat**2).sum(axis=0)
        count += flat.shape[0]
    mean = sum_ / max(count, 1)
    var = sumsq / max(count, 1) - mean**2
    std = np.sqrt(np.maximum(var, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


class PetDataset(Dataset):
    """Loads image + tabular + (optional) target from disk. Preprocessing
    parameters (mean/std, scaled tabular array) must be fit on training
    data only, then passed in here for val/test as well."""

    def __init__(
        self, ids, image_dir, tabular_array, targets, mean, std, size=IMAGE_SIZE
    ):
        self.ids = list(ids)
        self.image_dir = image_dir
        self.tabular = tabular_array.astype(np.float32)
        self.targets = targets.astype(np.float32) if targets is not None else None
        self.mean = mean.reshape(1, 1, 3)
        self.std = std.reshape(1, 1, 3)
        self.size = size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        path = os.path.join(self.image_dir, f"{img_id}.jpg")
        img = Image.open(path).convert("RGB").resize((self.size, self.size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)
        image_tensor = torch.from_numpy(arr)
        tabular_tensor = torch.from_numpy(self.tabular[idx])
        if self.targets is not None:
            target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
        else:
            target_tensor = torch.tensor(0.0, dtype=torch.float32)
        return image_tensor, tabular_tensor, target_tensor


class TinyPetNet(nn.Module):
    """Tiny CNN (2 conv layers, 8/16 channels) + global pooling + small
    tabular MLP branch, fused for scalar Pawpularity regression."""

    def __init__(self, tabular_dim=len(TABULAR_COLS)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.tab_fc1 = nn.Linear(tabular_dim, 16)
        self.tab_fc2 = nn.Linear(16, 16)
        self.dropout = nn.Dropout(0.2)

        self.fusion_fc1 = nn.Linear(32, 16)
        self.fusion_fc2 = nn.Linear(16, 1)

    def forward(self, image, tabular):
        x = self.pool(self.relu(self.bn1(self.conv1(image))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.global_pool(x)
        x = x.flatten(1)  # [B, 16]

        t = self.relu(self.tab_fc1(tabular))
        t = self.dropout(t)
        t = self.relu(self.tab_fc2(t))  # [B, 16]

        fused = torch.cat([x, t], dim=1)  # [B, 32]
        f = self.relu(self.fusion_fc1(fused))
        f = self.dropout(f)
        out = self.fusion_fc2(f)  # [B, 1]
        return out


# ---------------------------------------------------------------------------
# CPU preflight adapter contract
# ---------------------------------------------------------------------------
class CandidateAdapter:
    def __init__(self):
        pass

    def _defaults(self):
        return {"lr": 1e-3, "weight_decay": 1e-4, "criterion": None}

    def build_model(self, context):
        context = {**self._defaults(), **(context or {})}
        model = TinyPetNet()
        return model

    def build_optimizer(self, model, context):
        context = {**self._defaults(), **(context or {})}
        return torch.optim.AdamW(
            model.parameters(), lr=context["lr"], weight_decay=context["weight_decay"]
        )

    def _load_real_batch(self, input_dir, batch_size, device):
        try:
            train_csv = os.path.join(input_dir, "train.csv")
            df = pd.read_csv(train_csv)
            df = df.head(batch_size)
            n = len(df)
            if n == 0:
                raise ValueError("empty training fixture")
            images = []
            for _, row in df.iterrows():
                img_path = os.path.join(input_dir, "train", f"{row['Id']}.jpg")
                img = (
                    Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                )
                arr = np.asarray(img, dtype=np.float32) / 255.0
                arr = np.transpose(arr, (2, 0, 1))
                images.append(arr)
            images = np.stack(images, axis=0)
            tabular = df[TABULAR_COLS].values.astype(np.float32)
            if TARGET_COL in df.columns:
                target = df[TARGET_COL].values.astype(np.float32)
            else:
                target = np.random.rand(n).astype(np.float32) * 100.0
            if n < batch_size:
                pad = batch_size - n
                images = np.concatenate(
                    [images, np.repeat(images[-1:], pad, axis=0)], axis=0
                )
                tabular = np.concatenate(
                    [tabular, np.repeat(tabular[-1:], pad, axis=0)], axis=0
                )
                target = np.concatenate(
                    [target, np.repeat(target[-1:], pad, axis=0)], axis=0
                )
            image_t = torch.from_numpy(images).to(device=device, dtype=torch.float32)
            tabular_t = torch.from_numpy(tabular).to(device=device, dtype=torch.float32)
            target_t = torch.from_numpy(target).to(device=device, dtype=torch.float32)
            return image_t, tabular_t, target_t
        except Exception:
            image_t = torch.zeros(
                batch_size,
                3,
                IMAGE_SIZE,
                IMAGE_SIZE,
                dtype=torch.float32,
                device=device,
            )
            tabular_t = torch.zeros(
                batch_size, len(TABULAR_COLS), dtype=torch.float32, device=device
            )
            target_t = (
                torch.rand(batch_size, dtype=torch.float32, device=device) * 100.0
            )
            return image_t, tabular_t, target_t

    def _build_batch(self, scenario, device):
        batch_size = scenario.get("batch_size", 4)
        fixture = scenario.get("fixture") or {}
        if fixture:
            batch = {}
            for name, spec in fixture.items():
                shape = spec.get("shape") if isinstance(spec, dict) else spec
                if shape is None:
                    continue
                shape = list(shape)
                batch[name] = torch.zeros(
                    [batch_size] + shape, dtype=torch.float32, device=device
                )
            if "image" not in batch:
                batch["image"] = torch.zeros(
                    batch_size,
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    dtype=torch.float32,
                    device=device,
                )
            if "tabular" not in batch:
                batch["tabular"] = torch.zeros(
                    batch_size, len(TABULAR_COLS), dtype=torch.float32, device=device
                )
            batch["target"] = (
                torch.rand(batch_size, dtype=torch.float32, device=device) * 100.0
            )
            return batch
        else:
            input_dir = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")
            image, tabular, target = self._load_real_batch(
                input_dir, batch_size, device
            )
            return {"image": image, "tabular": tabular, "target": target}

    def build_train_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def training_step(self, model, batch, context):
        context = {**self._defaults(), **(context or {})}
        criterion = context.get("criterion") or nn.MSELoss()
        model.train()
        image = batch["image"]
        tabular = batch["tabular"]
        target = batch["target"]
        pred = model(image, tabular).squeeze(-1)
        loss = criterion(pred, target)
        return loss

    def validation_step(self, model, batch, context):
        context = {**self._defaults(), **(context or {})}
        criterion = context.get("criterion") or nn.MSELoss()
        model.eval()
        with torch.no_grad():
            image = batch["image"]
            tabular = batch["tabular"]
            target = batch["target"]
            pred = model(image, tabular).squeeze(-1)
            loss = criterion(pred, target)
        return loss


# ---------------------------------------------------------------------------
# Training / inference entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs("./submission", exist_ok=True)
    os.makedirs("./working", exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        try:
            cap_major, _ = torch.cuda.get_device_capability(0)
        except Exception:
            cap_major = 0
        if cap_major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    train_csv_path = os.path.join(DATA_ROOT, "train.csv")
    test_csv_path = os.path.join(DATA_ROOT, "test.csv")
    train_image_dir = os.path.join(DATA_ROOT, "train")
    test_image_dir = os.path.join(DATA_ROOT, "test")

    full_train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Split FIRST, then fit all preprocessing on the training split only.
    n_total = len(full_train_df)
    test_size = (
        0.2 if n_total >= 10 else max(1, int(round(n_total * 0.2))) / max(n_total, 1)
    )
    train_df, val_df = train_test_split(
        full_train_df, test_size=test_size, random_state=SEED
    )

    # Tabular scaler: fit on train split only.
    tab_scaler = StandardScaler()
    train_tab = tab_scaler.fit_transform(
        train_df[TABULAR_COLS].values.astype(np.float32)
    )
    val_tab = tab_scaler.transform(val_df[TABULAR_COLS].values.astype(np.float32))
    test_tab = tab_scaler.transform(test_df[TABULAR_COLS].values.astype(np.float32))

    # Image normalization stats: fit on train split only.
    img_mean, img_std = compute_image_stats(train_image_dir, train_df["Id"].tolist())

    train_targets = train_df[TARGET_COL].values.astype(np.float32)
    val_targets = val_df[TARGET_COL].values.astype(np.float32)

    train_dataset = PetDataset(
        train_df["Id"].tolist(),
        train_image_dir,
        train_tab,
        train_targets,
        img_mean,
        img_std,
    )
    val_dataset = PetDataset(
        val_df["Id"].tolist(), train_image_dir, val_tab, val_targets, img_mean, img_std
    )
    test_dataset = PetDataset(
        test_df["Id"].tolist(), test_image_dir, test_tab, None, img_mean, img_std
    )

    effective_batch = min(BATCH_SIZE, max(len(train_dataset), 1))
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch,
        shuffle=True,
        num_workers=2,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=effective_batch,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    model = TinyPetNet().to(device)
    model = model.float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    steps_per_epoch = max(
        1, (len(train_dataset) + effective_batch - 1) // effective_batch
    )
    settings = {
        "physical_batch_size": effective_batch,
        "effective_batch_size": effective_batch,
        "planned_epochs": EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "model_family": "tiny_cnn_tabular_fusion",
        "split_seed": SEED,
        "precision": "fp32",
        "precision_reason": "conservative_mode_mandates_fp32_no_amp",
    }

    with TrainingDiagnostics(
        model, optimizer, scaler=None, settings=settings
    ) as diagnostics:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            for images, tabular, targets in train_loader:
                images = images.to(device=device, dtype=torch.float32)
                tabular = tabular.to(device=device, dtype=torch.float32)
                targets = targets.to(device=device, dtype=torch.float32)

                optimizer.zero_grad()
                preds = model(images, tabular).squeeze(-1)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()
                diagnostics.after_update()

                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / max(len(train_dataset), 1)
            diagnostics.report(epoch=epoch)
            print(f"Epoch {epoch}: train_mse_loss={epoch_loss:.4f}")

    # Validation inference (real forward pass, after leaving diagnostics context).
    model.eval()
    val_preds_list = []
    val_targets_list = []
    with torch.no_grad():
        for images, tabular, targets in val_loader:
            images = images.to(device=device, dtype=torch.float32)
            tabular = tabular.to(device=device, dtype=torch.float32)
            preds = model(images, tabular).squeeze(-1).cpu().numpy()
            preds = np.clip(preds, 0.0, 100.0)
            val_preds_list.extend(preds.tolist())
            val_targets_list.extend(targets.numpy().tolist())

    val_preds_arr = np.array(val_preds_list, dtype=np.float64)
    val_targets_arr = np.array(val_targets_list, dtype=np.float64)
    rmse = float(np.sqrt(mean_squared_error(val_targets_arr, val_preds_arr)))

    # Test inference: identical processing pipeline as validation.
    test_preds_list = []
    with torch.no_grad():
        for images, tabular, _ in test_loader:
            images = images.to(device=device, dtype=torch.float32)
            tabular = tabular.to(device=device, dtype=torch.float32)
            preds = model(images, tabular).squeeze(-1).cpu().numpy()
            preds = np.clip(preds, 0.0, 100.0)
            test_preds_list.extend(preds.tolist())

    submission = pd.DataFrame(
        {
            "Id": test_df["Id"].values,
            "Pawpularity": np.array(test_preds_list, dtype=np.float64),
        }
    )
    submission.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {rmse}")
