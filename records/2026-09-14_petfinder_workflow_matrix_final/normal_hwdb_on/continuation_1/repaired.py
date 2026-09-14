import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

try:
    from utils.training_diagnostics import TrainingDiagnostics
except ImportError:

    class TrainingDiagnostics:
        """CPU-safe no-op fallback if the real diagnostics helper is unavailable."""

        def __init__(self, model, optimizer, scaler=None, settings=None):
            self.model = model
            self.optimizer = optimizer
            self.scaler = scaler
            self.settings = settings or {}

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
# Import-safe constants and configuration
# ---------------------------------------------------------------------------
SEED = 42
IMG_SIZE = 256
METADATA_COLS = [
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
INPUT_DIR = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Model: tiny CNN (2 conv layers: 8, 16 channels) + global pooling + small
# tabular MLP branch, fused into a bounded regression head.
# ---------------------------------------------------------------------------
class TinyPetCNN(nn.Module):
    def __init__(self, tab_dim=12):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.tab_fc1 = nn.Linear(tab_dim, 16)
        self.tab_fc2 = nn.Linear(16, 16)
        self.dropout = nn.Dropout(0.2)

        self.head1 = nn.Linear(16 + 16, 16)
        self.head2 = nn.Linear(16, 1)

    def forward(self, image, tabular):
        x = self.relu(self.bn1(self.conv1(image)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.global_pool(x).flatten(1)  # [B, 16]

        t = self.relu(self.tab_fc1(tabular))
        t = self.dropout(t)
        t = self.relu(self.tab_fc2(t))  # [B, 16]

        f = torch.cat([x, t], dim=1)
        f = self.relu(self.head1(f))
        out = torch.sigmoid(self.head2(f)).squeeze(1)  # bounded in [0, 1]
        return out


# ---------------------------------------------------------------------------
# Dataset: applies train-fit image normalization + tabular scaling.
# ---------------------------------------------------------------------------
class PetDataset(Dataset):
    def __init__(self, df, image_dir, img_mean, img_std, tab_scaler, has_target=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.img_mean = img_mean.astype(np.float32)
        self.img_std = img_std.astype(np.float32)
        self.tab_scaler = tab_scaler
        self.has_target = has_target

    def __len__(self):
        return len(self.df)

    def _load_image(self, img_id):
        path = os.path.join(self.image_dir, f"{img_id}.jpg")
        img = (
            Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
        arr = (arr - self.img_mean) / self.img_std
        arr = np.transpose(arr, (2, 0, 1)).copy()  # CHW
        return torch.from_numpy(arr.astype(np.float32))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["Id"]
        image = self._load_image(img_id)

        tab_raw = row[METADATA_COLS].values.astype(np.float32).reshape(1, -1)
        tab = self.tab_scaler.transform(tab_raw).astype(np.float32).reshape(-1)
        tab = torch.from_numpy(tab)

        if self.has_target:
            target = torch.tensor(float(row["Pawpularity"]), dtype=torch.float32)
            return image, tab, target
        return image, tab, str(img_id)


# ---------------------------------------------------------------------------
# CPU preflight adapter contract
# ---------------------------------------------------------------------------
class CandidateAdapter:
    def _defaults(self):
        return {
            "tab_dim": len(METADATA_COLS),
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "criterion": None,
        }

    def build_model(self, context):
        ctx = {**self._defaults(), **(context or {})}
        model = TinyPetCNN(tab_dim=ctx["tab_dim"])
        return model

    def build_optimizer(self, model, context):
        ctx = {**self._defaults(), **(context or {})}
        return torch.optim.Adam(
            model.parameters(), lr=ctx["lr"], weight_decay=ctx["weight_decay"]
        )

    def _build_batch(self, scenario, device):
        scenario = scenario or {}
        batch_size = int(scenario.get("batch_size", 4))
        fixture = scenario.get("fixture") or {}
        gen = torch.Generator().manual_seed(0)

        if fixture:
            batch = {}
            for name, shape in fixture.items():
                full_shape = (batch_size,) + tuple(shape)
                if name == "target":
                    batch[name] = (
                        torch.rand(full_shape, generator=gen, dtype=torch.float32)
                        * 100.0
                    ).to(device)
                elif name == "tabular":
                    batch[name] = (
                        (
                            torch.rand(full_shape, generator=gen, dtype=torch.float32)
                            > 0.5
                        )
                        .float()
                        .to(device)
                    )
                else:
                    batch[name] = torch.rand(
                        full_shape, generator=gen, dtype=torch.float32
                    ).to(device)
            if "target" not in batch:
                batch["target"] = (
                    torch.rand((batch_size,), generator=gen, dtype=torch.float32)
                    * 100.0
                ).to(device)
            return batch

        image = torch.rand(
            (batch_size, 3, IMG_SIZE, IMG_SIZE), generator=gen, dtype=torch.float32
        ).to(device)
        tabular = (
            (
                torch.rand(
                    (batch_size, len(METADATA_COLS)), generator=gen, dtype=torch.float32
                )
                > 0.5
            )
            .float()
            .to(device)
        )
        target = (
            torch.rand((batch_size,), generator=gen, dtype=torch.float32) * 100.0
        ).to(device)
        return {"image": image, "tabular": tabular, "target": target}

    def build_train_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def training_step(self, model, batch, context):
        ctx = {**self._defaults(), **(context or {})}
        criterion = ctx.get("criterion") or nn.MSELoss()
        model.train()
        pred = model(batch["image"], batch["tabular"]) * 100.0
        target = batch["target"].float()
        loss = criterion(pred.float(), target)
        return loss

    def validation_step(self, model, batch, context):
        ctx = {**self._defaults(), **(context or {})}
        criterion = ctx.get("criterion") or nn.MSELoss()
        model.eval()
        with torch.no_grad():
            pred = model(batch["image"], batch["tabular"]) * 100.0
            target = batch["target"].float()
            loss = criterion(pred.float(), target)
        return loss


# ---------------------------------------------------------------------------
# Training / evaluation / submission (side effects only under main guard)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    os.makedirs(WORKING_DIR, exist_ok=True)

    train_csv = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    test_csv = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
    train_image_dir = os.path.join(INPUT_DIR, "train")
    test_image_dir = os.path.join(INPUT_DIR, "test")

    # Single train/validation split (fixed seed for reproducibility).
    tr_df, val_df = train_test_split(train_csv, test_size=0.25, random_state=SEED)
    tr_df = tr_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Fit tabular scaler ONLY on training rows.
    tab_scaler = StandardScaler()
    tab_scaler.fit(tr_df[METADATA_COLS].values.astype(np.float32))

    # Fit per-channel image normalization stats ONLY on training images.
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    n_pixels = 0
    for img_id in tr_df["Id"]:
        path = os.path.join(train_image_dir, f"{img_id}.jpg")
        img = (
            Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        )
        arr = np.asarray(img, dtype=np.float64) / 255.0
        pixel_sum += arr.sum(axis=(0, 1))
        pixel_sq_sum += (arr**2).sum(axis=(0, 1))
        n_pixels += arr.shape[0] * arr.shape[1]
    img_mean = (pixel_sum / n_pixels).astype(np.float32)
    img_var = (pixel_sq_sum / n_pixels) - (img_mean.astype(np.float64) ** 2)
    img_std = np.sqrt(np.maximum(img_var, 1e-6)).astype(np.float32)
    img_std = np.where(img_std < 1e-3, 1.0, img_std).astype(np.float32)

    # Device: CPU-safe by default; opportunistically use CUDA if present
    # (target deployment is RTX 5090, audit only exercises CPU adapter checks).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision_name = "fp32"
    precision_reason = "FP32 default per precision policy; no accelerated path required for this tiny model"
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        precision_name = "tf32"
        precision_reason = (
            "TF32 matmul enabled on CUDA while keeping loss/RMSE reductions in FP32"
        )

    train_dataset = PetDataset(
        tr_df, train_image_dir, img_mean, img_std, tab_scaler, has_target=True
    )
    val_dataset = PetDataset(
        val_df, train_image_dir, img_mean, img_std, tab_scaler, has_target=True
    )
    test_dataset = PetDataset(
        test_csv, test_image_dir, img_mean, img_std, tab_scaler, has_target=False
    )

    batch_size = 4
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model = TinyPetCNN(tab_dim=len(METADATA_COLS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    epochs = 2
    steps_per_epoch = max(1, len(train_loader))
    settings = {
        "physical_batch_size": batch_size,
        "effective_batch_size": batch_size,
        "planned_epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "model_family": "tiny_cnn_tabular_fusion",
        "split_seed": SEED,
        "precision": precision_name,
        "precision_reason": precision_reason,
    }

    with TrainingDiagnostics(
        model, optimizer, scaler=None, settings=settings
    ) as diagnostics:
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            n_batches = 0
            for image, tab, target in train_loader:
                image = image.to(device)
                tab = tab.to(device)
                target = target.to(device).float()

                optimizer.zero_grad()
                pred = model(image, tab) * 100.0
                loss = criterion(pred.float(), target)
                loss.backward()
                optimizer.step()
                diagnostics.after_update()

                running_loss += loss.item()
                n_batches += 1
            avg_train_loss = running_loss / max(1, n_batches)
            diagnostics.report(epoch=epoch)
            print(f"Epoch {epoch}: train_mse={avg_train_loss:.4f}")

    # Validation inference (real model forward pass).
    model.eval()
    val_preds_list = []
    val_targets_list = []
    with torch.no_grad():
        for image, tab, target in val_loader:
            image = image.to(device)
            tab = tab.to(device)
            pred = model(image, tab) * 100.0
            val_preds_list.append(pred.cpu().numpy())
            val_targets_list.append(target.numpy())
    val_preds = np.clip(np.concatenate(val_preds_list).astype(np.float64), 0.0, 100.0)
    val_targets = np.concatenate(val_targets_list).astype(np.float64)
    rmse = float(np.sqrt(mean_squared_error(val_targets, val_preds)))

    # Test inference (identical processing/model path as validation).
    model.eval()
    test_ids = []
    test_preds_list = []
    with torch.no_grad():
        for image, tab, ids in test_loader:
            image = image.to(device)
            tab = tab.to(device)
            pred = model(image, tab) * 100.0
            test_preds_list.append(pred.cpu().numpy())
            test_ids.extend(list(ids))
    test_preds = np.clip(np.concatenate(test_preds_list).astype(np.float64), 0.0, 100.0)

    submission = pd.DataFrame({"Id": test_ids, "Pawpularity": test_preds})
    submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

    print(f"Final Validation Score: {rmse}")
