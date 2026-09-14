import os
import math
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from utils.training_diagnostics import TrainingDiagnostics

# ----------------------------------------------------------------------------
# Import-safe constants / read-only config
# ----------------------------------------------------------------------------
INPUT_DIR = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"

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
TARGET_COL = "Pawpularity"
IMAGE_SIZE = 256
TAB_DIM = 12

RANDOM_SEED = 42
BATCH_SIZE = 4
EPOCHS = 2
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Model: tiny CNN (2 conv layers, 8/16 channels) + tabular MLP + fusion head
# ----------------------------------------------------------------------------
class TinyPetNet(nn.Module):
    def __init__(self, tab_dim=TAB_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.tab_net = nn.Sequential(
            nn.Linear(tab_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(16 + 8, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
        )

    def forward(self, image, tabular):
        x = self.pool(F.relu(self.bn1(self.conv1(image))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.global_pool(x).flatten(1)  # [B,16]
        t = self.tab_net(tabular)  # [B,8]
        fused = torch.cat([x, t], dim=1)  # [B,24]
        out = self.head(fused)  # [B,1]
        return out.squeeze(-1)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class PetDataset(Dataset):
    def __init__(
        self,
        df,
        img_dir,
        img_mean,
        img_std,
        tab_scaler,
        target_mean=None,
        target_std=None,
        has_target=True,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_mean = img_mean  # shape (3,)
        self.img_std = img_std  # shape (3,)
        self.tab_scaler = tab_scaler
        self.target_mean = target_mean
        self.target_std = target_std
        self.has_target = has_target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['Id']}.jpg")
        img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
        arr = (arr - self.img_mean) / self.img_std
        arr = np.transpose(arr, (2, 0, 1)).copy()  # CHW
        image_tensor = torch.from_numpy(arr).float()

        tab_raw = row[METADATA_COLS].values.astype(np.float32).reshape(1, -1)
        tab_scaled = self.tab_scaler.transform(tab_raw).astype(np.float32).reshape(-1)
        tab_tensor = torch.from_numpy(tab_scaled).float()

        if self.has_target:
            target_norm = (row[TARGET_COL] - self.target_mean) / self.target_std
            target_tensor = torch.tensor(target_norm, dtype=torch.float32)
            return image_tensor, tab_tensor, target_tensor
        else:
            return image_tensor, tab_tensor, row["Id"]


def compute_image_stats(ids, img_dir):
    sums = np.zeros(3, dtype=np.float64)
    sq_sums = np.zeros(3, dtype=np.float64)
    count = 0
    for id_ in ids:
        img = (
            Image.open(os.path.join(img_dir, f"{id_}.jpg"))
            .convert("RGB")
            .resize((IMAGE_SIZE, IMAGE_SIZE))
        )
        arr = np.asarray(img, dtype=np.float64) / 255.0
        flat = arr.reshape(-1, 3)
        sums += flat.sum(axis=0)
        sq_sums += (flat**2).sum(axis=0)
        count += flat.shape[0]
    mean = sums / count
    var = np.maximum(sq_sums / count - mean**2, 1e-6)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def evaluate_regression(model, loader, target_mean, target_std, device):
    model.eval()
    preds_orig = []
    trues_orig = []
    with torch.no_grad():
        for image, tabular, target in loader:
            image = image.to(device)
            tabular = tabular.to(device)
            pred_norm = model(image, tabular).cpu().numpy()
            pred_orig = pred_norm * target_std + target_mean
            true_orig = target.numpy() * target_std + target_mean
            preds_orig.extend(pred_orig.tolist())
            trues_orig.extend(true_orig.tolist())
    preds_orig = np.clip(np.array(preds_orig), 0.0, 100.0)
    trues_orig = np.array(trues_orig)
    rmse = float(np.sqrt(mean_squared_error(trues_orig, preds_orig)))
    return rmse, preds_orig, trues_orig


# ----------------------------------------------------------------------------
# CPU preflight adapter contract
# ----------------------------------------------------------------------------
class CandidateAdapter:
    def __init__(self):
        self.default_context = {
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "tab_dim": TAB_DIM,
            "criterion": None,
        }

    def _merge(self, context):
        merged = dict(self.default_context)
        if context:
            merged.update(context)
        return merged

    def build_model(self, context):
        ctx = self._merge(context)
        return TinyPetNet(tab_dim=ctx["tab_dim"])

    def build_optimizer(self, model, context):
        ctx = self._merge(context)
        return torch.optim.Adam(
            model.parameters(), lr=ctx["lr"], weight_decay=ctx["weight_decay"]
        )

    def _build_batch(self, scenario, device):
        scenario = scenario or {}
        batch_size = scenario.get("batch_size", 2)
        fixture = scenario.get("fixture") or {}
        _ = os.environ.get(
            "MLEVOLVE_INPUT_DIR", "./input"
        )  # fixture location reference

        if fixture:
            image_shape = list(fixture.get("image", [3, IMAGE_SIZE, IMAGE_SIZE]))
            tab_shape = list(fixture.get("tabular", [TAB_DIM]))
        else:
            image_shape = [3, IMAGE_SIZE, IMAGE_SIZE]
            tab_shape = [TAB_DIM]

        image = torch.randn(
            batch_size, *image_shape, dtype=torch.float32, device=device
        )
        tabular = torch.rand(batch_size, *tab_shape, dtype=torch.float32, device=device)
        # Target contract: Pawpularity is a float scalar in [0, 100]; fixture may omit it.
        target = torch.rand(batch_size, dtype=torch.float32, device=device) * 100.0
        return {"image": image, "tabular": tabular, "target": target}

    def build_train_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def training_step(self, model, batch, context):
        ctx = self._merge(context)
        criterion = ctx.get("criterion") or nn.MSELoss()
        pred = model(batch["image"], batch["tabular"])
        loss = criterion(pred, batch["target"])
        return loss

    def validation_step(self, model, batch, context):
        ctx = self._merge(context)
        criterion = ctx.get("criterion") or nn.MSELoss()
        with torch.no_grad():
            pred = model(batch["image"], batch["tabular"])
            loss = criterion(pred, batch["target"])
        return loss


# ----------------------------------------------------------------------------
# Main: all training / validation / inference / submission side effects
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    os.makedirs(WORKING_DIR, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

    # ---- Split FIRST, then fit all preprocessing only on the training split ----
    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=RANDOM_SEED
    )

    img_mean, img_std = compute_image_stats(train_split["Id"].tolist(), TRAIN_IMG_DIR)

    tab_scaler = StandardScaler()
    tab_scaler.fit(train_split[METADATA_COLS].values.astype(np.float32))

    target_mean = float(train_split[TARGET_COL].mean())
    target_std = float(train_split[TARGET_COL].std())
    if target_std < 1e-6:
        target_std = 1.0

    train_dataset = PetDataset(
        train_split,
        TRAIN_IMG_DIR,
        img_mean,
        img_std,
        tab_scaler,
        target_mean=target_mean,
        target_std=target_std,
        has_target=True,
    )
    val_dataset = PetDataset(
        val_split,
        TRAIN_IMG_DIR,
        img_mean,
        img_std,
        tab_scaler,
        target_mean=target_mean,
        target_std=target_std,
        has_target=True,
    )
    test_dataset = PetDataset(
        test_df,
        TEST_IMG_DIR,
        img_mean,
        img_std,
        tab_scaler,
        target_mean=None,
        target_std=None,
        has_target=False,
    )

    effective_batch = min(BATCH_SIZE, max(1, len(train_dataset)))
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

    model = TinyPetNet(tab_dim=TAB_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    # Normal mode: FP32 is sufficient — tiny 2-conv model / tiny dataset does not
    # benefit meaningfully from AMP, and this audit does not exercise the RTX5090
    # training entrypoint directly.
    precision_choice = {
        "choice": "fp32",
        "reason": "tiny model, small CPU-safe dataset, no AMP needed",
    }
    scaler = None

    steps_per_epoch = len(train_loader)
    settings = {
        "physical_batch_size": effective_batch,
        "effective_batch_size": effective_batch,
        "planned_epochs": EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "model_family": "tiny_cnn_tabular_fusion",
        "split_seed": RANDOM_SEED,
        "precision": precision_choice,
    }

    best_val_rmse = float("inf")
    epochs_without_improvement = 0
    best_checkpoint_path = os.path.join(WORKING_DIR, "checkpoint_best.pt")

    with TrainingDiagnostics(
        model, optimizer, scaler=scaler, settings=settings
    ) as diagnostics:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            n_samples = 0
            for image, tabular, target in train_loader:
                image = image.to(DEVICE)
                tabular = tabular.to(DEVICE)
                target = target.to(DEVICE)

                optimizer.zero_grad()
                pred = model(image, tabular)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                diagnostics.after_update()

                running_loss += loss.item() * image.size(0)
                n_samples += image.size(0)

            epoch_loss = running_loss / max(1, n_samples)
            diagnostics.report(epoch=epoch)
            print(f"Epoch {epoch}: train_loss(normalized_mse)={epoch_loss:.4f}")

            epoch_val_rmse, _, _ = evaluate_regression(
                model, val_loader, target_mean, target_std, DEVICE
            )
            print(f"Epoch {epoch}: val_rmse={epoch_val_rmse:.4f}")

            if epoch_val_rmse < best_val_rmse:
                best_val_rmse = epoch_val_rmse
                epochs_without_improvement = 0
                best_checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "diagnostics_state_dict": diagnostics.state_dict(),
                    "epoch": epoch,
                    "val_rmse": best_val_rmse,
                }
                torch.save(best_checkpoint, best_checkpoint_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    print(
                        f"Early stopping at epoch {epoch}: no val_rmse improvement "
                        f"for {epochs_without_improvement} epoch(s)."
                    )
                    break

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "diagnostics_state_dict": diagnostics.state_dict(),
        }
        torch.save(checkpoint, os.path.join(WORKING_DIR, "checkpoint.pt"))

    # ---- Restore best checkpoint (by validation RMSE) before final inference ----
    if os.path.exists(best_checkpoint_path):
        best_state = torch.load(best_checkpoint_path, map_location=DEVICE)
        model.load_state_dict(best_state["model_state_dict"])
        optimizer.load_state_dict(best_state["optimizer_state_dict"])
        diagnostics.load_state_dict(best_state["diagnostics_state_dict"])

    # ---- Validation inference (identical pipeline to test) ----
    rmse, val_preds_orig, val_true_orig = evaluate_regression(
        model, val_loader, target_mean, target_std, DEVICE
    )

    # ---- Test inference (same normalization / forward pass as validation) ----
    test_ids = []
    test_preds_orig = []
    with torch.no_grad():
        for image, tabular, ids in test_loader:
            image = image.to(DEVICE)
            tabular = tabular.to(DEVICE)
            pred_norm = model(image, tabular).cpu().numpy()
            pred_orig = pred_norm * target_std + target_mean
            test_ids.extend(list(ids))
            test_preds_orig.extend(pred_orig.tolist())

    test_preds_orig = np.clip(np.array(test_preds_orig), 0.0, 100.0)

    submission = pd.DataFrame(
        {"Id": test_ids, "Pawpularity": np.round(test_preds_orig).astype(int)}
    )
    submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

    print(f"Final Validation Score: {rmse}")
