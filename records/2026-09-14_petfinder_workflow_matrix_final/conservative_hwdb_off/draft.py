import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Import-safe constants / config (read-only, no side effects)
# ----------------------------------------------------------------------------
SEED = 42
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
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 2
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Root input directory. Honors MLEVOLVE_INPUT_DIR override for isolated CPU checks.
INPUT_DIR = os.environ.get(
    "MLEVOLVE_INPUT_DIR",
    "/home/justin/MLEvolve/records/2026-09-14_petfinder_workflow_matrix_final/"
    "conservative_hwdb_off/workspace/input",
)
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"

# Conservative precision: FP32 everywhere. TF32 only if confirmed Ampere-or-newer CUDA.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_USE_TF32 = False
if torch.cuda.is_available():
    try:
        major, _ = torch.cuda.get_device_capability(0)
        _USE_TF32 = major >= 8
    except Exception:
        _USE_TF32 = False
torch.backends.cuda.matmul.allow_tf32 = _USE_TF32
torch.backends.cudnn.allow_tf32 = _USE_TF32

# Runtime diagnostics (fallback no-op if utils module unavailable)
try:
    from utils.training_diagnostics import TrainingDiagnostics
except Exception:

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


# ----------------------------------------------------------------------------
# Model definition (real model used both in training and CandidateAdapter)
# ----------------------------------------------------------------------------
class PetfinderNet(nn.Module):
    """Tiny CNN (2 conv layers, 8/16 channels) + global pooling + small tabular MLP."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.tab_fc1 = nn.Linear(12, 16)
        self.tab_fc2 = nn.Linear(16, 8)

        self.fusion_fc1 = nn.Linear(16 + 8, 16)
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(16, 1)

    def forward(self, image, tabular):
        x = F.relu(self.conv1(image))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.gap(x).flatten(1)  # [B, 16]

        t = F.relu(self.tab_fc1(tabular))
        t = F.relu(self.tab_fc2(t))  # [B, 8]

        fused = torch.cat([x, t], dim=1)
        fused = F.relu(self.fusion_fc1(fused))
        fused = self.dropout(fused)
        pred = self.out(fused).squeeze(-1)
        return pred


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class PetfinderDataset(Dataset):
    def __init__(self, ids, tabular_arr, targets, image_dir, augment=False):
        self.ids = list(ids)
        self.tabular = tabular_arr.astype(np.float32)
        self.targets = (
            None if targets is None else np.asarray(targets, dtype=np.float32)
        )
        self.image_dir = image_dir
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def _load_image(self, img_id):
        path = os.path.join(self.image_dir, f"{img_id}.jpg")
        try:
            img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(img, dtype=np.float32) / 255.0
        except Exception:
            arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        if self.augment and np.random.rand() < 0.5:
            arr = np.ascontiguousarray(arr[:, ::-1, :])
        arr = np.transpose(arr, (2, 0, 1))
        return arr

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        arr = self._load_image(img_id)
        tab = self.tabular[idx]
        image_t = torch.from_numpy(arr.copy()).float()
        tab_t = torch.from_numpy(tab.copy()).float()
        if self.targets is not None:
            target_t = torch.tensor(self.targets[idx], dtype=torch.float32)
            return image_t, tab_t, target_t
        return image_t, tab_t, img_id


# ----------------------------------------------------------------------------
# CPU Preflight Adapter Contract
# ----------------------------------------------------------------------------
class CandidateAdapter:
    def _default_context(self):
        return {
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "criterion": None,
        }

    def build_model(self, context):
        context = {**self._default_context(), **(context or {})}
        return PetfinderNet()

    def build_optimizer(self, model, context):
        context = {**self._default_context(), **(context or {})}
        lr = context.get("lr", LR)
        wd = context.get("weight_decay", WEIGHT_DECAY)
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    def _build_batch(self, scenario, device):
        scenario = scenario or {}
        batch_size = int(scenario.get("batch_size", BATCH_SIZE))
        fixture = scenario.get("fixture") or {}

        tensors = {}
        if fixture:
            for name, shape in fixture.items():
                shape = tuple(shape) if shape else tuple()
                if name == "target":
                    if shape:
                        tensors[name] = (
                            torch.rand(
                                (batch_size,) + shape,
                                dtype=torch.float32,
                                device=device,
                            )
                            * 100.0
                        )
                    else:
                        tensors[name] = (
                            torch.rand(batch_size, dtype=torch.float32, device=device)
                            * 100.0
                        )
                else:
                    tensors[name] = torch.randn(
                        (batch_size,) + shape, dtype=torch.float32, device=device
                    )

        if "image" not in tensors:
            tensors["image"] = torch.randn(
                batch_size, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32, device=device
            )
        if "tabular" not in tensors:
            tensors["tabular"] = torch.randn(
                batch_size, 12, dtype=torch.float32, device=device
            )
        if "target" not in tensors:
            tensors["target"] = (
                torch.rand(batch_size, dtype=torch.float32, device=device) * 100.0
            )

        return tensors

    def build_train_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def training_step(self, model, batch, context):
        context = {**self._default_context(), **(context or {})}
        criterion = context.get("criterion") or nn.MSELoss()
        image = batch["image"]
        tabular = batch["tabular"]
        target = batch["target"].float().view(-1)
        pred = model(image, tabular).view(-1)
        loss = criterion(pred, target)
        return loss

    def validation_step(self, model, batch, context):
        context = {**self._default_context(), **(context or {})}
        criterion = context.get("criterion") or nn.MSELoss()
        image = batch["image"]
        tabular = batch["tabular"]
        target = batch["target"].float().view(-1)
        with torch.no_grad():
            pred = model(image, tabular).view(-1)
            loss = criterion(pred, target)
        return loss


# ----------------------------------------------------------------------------
# Main training / inference entrypoint
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    os.makedirs(WORKING_DIR, exist_ok=True)

    train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

    # ---- Split FIRST, then fit all preprocessing on the training split only ----
    n_val = max(1, int(round(0.2 * len(train_df))))
    tr_df, val_df = train_test_split(train_df, test_size=n_val, random_state=SEED)
    tr_df = tr_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    tab_scaler = StandardScaler()
    tr_tab = tab_scaler.fit_transform(tr_df[METADATA_COLS].values.astype(np.float32))
    val_tab = tab_scaler.transform(val_df[METADATA_COLS].values.astype(np.float32))
    test_tab = tab_scaler.transform(test_df[METADATA_COLS].values.astype(np.float32))

    target_mean = float(tr_df["Pawpularity"].mean())
    target_std = float(tr_df["Pawpularity"].std())
    if target_std < 1e-6:
        target_std = 1.0

    tr_target_norm = (
        tr_df["Pawpularity"].values.astype(np.float32) - target_mean
    ) / target_std
    val_target_raw = val_df["Pawpularity"].values.astype(np.float32)

    train_ds = PetfinderDataset(
        tr_df["Id"].values, tr_tab, tr_target_norm, TRAIN_IMG_DIR, augment=True
    )
    val_ds = PetfinderDataset(
        val_df["Id"].values, val_tab, val_target_raw, TRAIN_IMG_DIR, augment=False
    )
    test_ds = PetfinderDataset(
        test_df["Id"].values, test_tab, None, TEST_IMG_DIR, augment=False
    )

    eff_batch = min(BATCH_SIZE, max(1, len(train_ds)))
    train_loader = DataLoader(
        train_ds, batch_size=eff_batch, shuffle=True, num_workers=2, drop_last=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(BATCH_SIZE, max(1, len(val_ds))),
        shuffle=False,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=min(BATCH_SIZE, max(1, len(test_ds))),
        shuffle=False,
        num_workers=2,
    )

    model = PetfinderNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    diagnostics_settings = {
        "physical_batch_size": eff_batch,
        "effective_batch_size": eff_batch,
        "planned_epochs": EPOCHS,
        "steps_per_epoch": len(train_loader),
        "model_family": "tiny_cnn_tabular_fusion",
        "split_seed": SEED,
        "precision": {
            "choice": "fp32",
            "reason": "conservative precision mandated; TF32 only if confirmed Ampere+",
        },
    }

    with TrainingDiagnostics(
        model, optimizer, scaler=None, settings=diagnostics_settings
    ) as diagnostics:
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            n_batches = 0
            for image, tabular, target in train_loader:
                image = image.to(DEVICE)
                tabular = tabular.to(DEVICE)
                target = target.to(DEVICE).float().view(-1)

                optimizer.zero_grad()
                pred = model(image, tabular).view(-1)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                diagnostics.after_update()

                running_loss += loss.item()
                n_batches += 1

            avg_loss = running_loss / max(1, n_batches)
            diagnostics.report(epoch=epoch + 1)
            print(f"Epoch {epoch + 1}/{EPOCHS} - train_loss(norm_mse): {avg_loss:.4f}")

    # ---- Validation inference (identical pipeline as test) ----
    model.eval()
    val_preds = []
    with torch.no_grad():
        for image, tabular, _ in val_loader:
            image = image.to(DEVICE)
            tabular = tabular.to(DEVICE)
            pred_norm = model(image, tabular).view(-1).cpu().numpy()
            pred_orig = pred_norm * target_std + target_mean
            val_preds.append(pred_orig)
    val_preds = np.concatenate(val_preds) if val_preds else np.array([])
    val_preds = np.clip(val_preds, 0.0, 100.0)

    rmse = float(np.sqrt(np.mean((val_preds - val_target_raw) ** 2)))

    # ---- Test inference (identical processing logic as validation) ----
    test_preds = []
    test_ids_ordered = []
    with torch.no_grad():
        for image, tabular, ids_batch in test_loader:
            image = image.to(DEVICE)
            tabular = tabular.to(DEVICE)
            pred_norm = model(image, tabular).view(-1).cpu().numpy()
            pred_orig = pred_norm * target_std + target_mean
            test_preds.append(pred_orig)
            test_ids_ordered.extend(list(ids_batch))
    test_preds = np.concatenate(test_preds) if test_preds else np.array([])
    test_preds = np.clip(test_preds, 0.0, 100.0)

    submission = pd.DataFrame({"Id": test_ids_ordered, "Pawpularity": test_preds})
    submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

    # NOTE: This is synthetic audit evidence (tiny dataset, capped architecture,
    # single split, <=2 epochs). Results do not reflect leaderboard quality.
    print(f"Final Validation Score: {rmse}")
