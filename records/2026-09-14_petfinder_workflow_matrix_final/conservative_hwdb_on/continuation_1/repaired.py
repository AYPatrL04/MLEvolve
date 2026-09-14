import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

try:
    from utils.training_diagnostics import TrainingDiagnostics
except Exception:

    class TrainingDiagnostics:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def after_update(self):
            pass

        def report(self, epoch=None):
            pass

        def state_dict(self):
            return {}

        def load_state_dict(self, *args, **kwargs):
            pass


# ----------------------------------------------------------------------------
# Import-safe constants / read-only config
# NOTE: This is synthetic audit evidence (~24 train / 4 test images). Results
# here are a functional smoke test, not a claim of leaderboard-quality skill.
# ----------------------------------------------------------------------------
INPUT_DIR = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")
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
SEED = 42
EPOCHS = 2
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_precision():
    """Conservative precision: strict FP32 everywhere; TF32 matmul only on
    confirmed Ampere-or-newer CUDA hardware (e.g. the RTX 5090 target)."""
    if torch.cuda.is_available():
        try:
            cap = torch.cuda.get_device_capability(0)
        except Exception:
            cap = (0, 0)
        if cap[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return "tf32"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return "fp32"


class ImageBranch(nn.Module):
    """At most two conv layers (8 -> 16 channels) + global pooling."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, 8)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(4, 16)
        self.pool2 = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.gn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.act(self.gn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.global_pool(x)
        return x.flatten(1)  # [B, 16]


class TabularBranch(nn.Module):
    def __init__(self, in_features=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)  # [B, 8]


class PetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_branch = ImageBranch()
        self.tabular_branch = TabularBranch(len(METADATA_COLS))
        self.fusion = nn.Sequential(
            nn.Linear(16 + 8, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, image, tabular):
        image = image.float()
        tabular = tabular.float()
        img_feat = self.image_branch(image)
        tab_feat = self.tabular_branch(tabular)
        fused = torch.cat([img_feat, tab_feat], dim=1)
        out = self.fusion(fused)
        return out  # [B, 1]


class PetDataset(Dataset):
    def __init__(
        self,
        df,
        input_dir,
        split,
        scaler,
        img_mean,
        img_std,
        augment=False,
        has_target=True,
    ):
        self.df = df.reset_index(drop=True)
        self.input_dir = input_dir
        self.split = split
        self.scaler = scaler
        self.img_mean = img_mean
        self.img_std = img_std
        self.augment = augment
        self.has_target = has_target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["Id"]
        img_path = os.path.join(self.input_dir, self.split, f"{img_id}.jpg")
        img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.augment and np.random.rand() < 0.5:
            arr = arr[:, ::-1, :].copy()
        arr = (arr - self.img_mean) / self.img_std
        arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)
        image_tensor = torch.from_numpy(arr)

        tab_values = row[METADATA_COLS].values.astype(np.float32).reshape(1, -1)
        tab_scaled = self.scaler.transform(tab_values).astype(np.float32).reshape(-1)
        tabular_tensor = torch.from_numpy(tab_scaled)

        sample = {"image": image_tensor, "tabular": tabular_tensor, "id": str(img_id)}
        if self.has_target:
            sample["target"] = torch.tensor(
                float(row["Pawpularity"]), dtype=torch.float32
            )
        return sample


def compute_image_stats(df, input_dir, split="train"):
    """Fit per-channel mean/std ONLY on training images."""
    sums = np.zeros(3, dtype=np.float64)
    sq_sums = np.zeros(3, dtype=np.float64)
    n_pixels = 0
    for img_id in df["Id"]:
        path = os.path.join(input_dir, split, f"{img_id}.jpg")
        img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img, dtype=np.float64) / 255.0
        sums += arr.sum(axis=(0, 1))
        sq_sums += (arr**2).sum(axis=(0, 1))
        n_pixels += arr.shape[0] * arr.shape[1]
    mean = sums / n_pixels
    var = sq_sums / n_pixels - mean**2
    std = np.sqrt(np.clip(var, 1e-6, None))
    return mean.astype(np.float32), std.astype(np.float32)


class CandidateAdapter:
    """CPU-safe preflight adapter exercising the real PetNet model, loss,
    and optimizer used by the training pipeline above."""

    def _defaults(self):
        return {"lr": 1e-3, "weight_decay": 1e-4, "device": "cpu", "criterion": None}

    def build_model(self, context):
        context = {**self._defaults(), **(context or {})}
        model = PetNet()
        model.to(context["device"])
        return model

    def build_optimizer(self, model, context):
        context = {**self._defaults(), **(context or {})}
        return torch.optim.AdamW(
            model.parameters(), lr=context["lr"], weight_decay=context["weight_decay"]
        )

    def _build_batch(self, scenario, device):
        scenario = scenario or {}
        batch_size = int(scenario.get("batch_size", 4))
        fixture = scenario.get("fixture") or {}

        def _shape_for(name, default_shape):
            spec = fixture.get(name)
            if spec is None:
                return default_shape
            if isinstance(spec, dict) and "shape" in spec:
                return tuple(spec["shape"])
            if isinstance(spec, (list, tuple)):
                return tuple(spec)
            return default_shape

        image_shape = _shape_for("image", (3, IMG_SIZE, IMG_SIZE))
        tabular_shape = _shape_for("tabular", (len(METADATA_COLS),))

        image = torch.rand(batch_size, *image_shape, dtype=torch.float32, device=device)
        tabular = torch.randint(
            0, 2, (batch_size, *tabular_shape), device=device
        ).float()
        target = torch.rand(batch_size, dtype=torch.float32, device=device) * 100.0
        return {"image": image, "tabular": tabular, "target": target}

    def build_train_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def training_step(self, model, batch, context):
        context = {**self._defaults(), **(context or {})}
        criterion = context.get("criterion") or nn.SmoothL1Loss()
        model.train()
        pred = model(batch["image"], batch["tabular"]).squeeze(-1)
        loss = criterion(pred, batch["target"])
        return loss

    def validation_step(self, model, batch, context):
        context = {**self._defaults(), **(context or {})}
        criterion = context.get("criterion") or nn.SmoothL1Loss()
        model.eval()
        with torch.no_grad():
            pred = model(batch["image"], batch["tabular"]).squeeze(-1)
            loss = criterion(pred, batch["target"])
        return loss


if __name__ == "__main__":
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    precision_mode = configure_precision()

    train_csv = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    test_csv = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

    train_df, val_df = train_test_split(train_csv, test_size=0.2, random_state=SEED)

    # Fit preprocessing ONLY on training split (no leakage).
    scaler = StandardScaler()
    scaler.fit(train_df[METADATA_COLS].values.astype(np.float32))
    img_mean, img_std = compute_image_stats(train_df, INPUT_DIR, split="train")

    train_dataset = PetDataset(
        train_df,
        INPUT_DIR,
        "train",
        scaler,
        img_mean,
        img_std,
        augment=True,
        has_target=True,
    )
    val_dataset = PetDataset(
        val_df,
        INPUT_DIR,
        "train",
        scaler,
        img_mean,
        img_std,
        augment=False,
        has_target=True,
    )
    test_dataset = PetDataset(
        test_csv,
        INPUT_DIR,
        "test",
        scaler,
        img_mean,
        img_std,
        augment=False,
        has_target=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    model = PetNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()  # kept FP32 as required

    settings = {
        "physical_batch_size": BATCH_SIZE,
        "effective_batch_size": BATCH_SIZE,
        "planned_epochs": EPOCHS,
        "steps_per_epoch": len(train_loader),
        "model_family": "tiny_cnn_tabular_fusion",
        "split_seed": SEED,
        "precision_choice": precision_mode,
        "precision_reason": "Conservative FP32 training; TF32 matmul only if confirmed Ampere+ CUDA",
    }

    with TrainingDiagnostics(
        model, optimizer, scaler=None, settings=settings
    ) as diagnostics:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            n_seen = 0
            for batch in train_loader:
                image = batch["image"].to(DEVICE, dtype=torch.float32)
                tabular = batch["tabular"].to(DEVICE, dtype=torch.float32)
                target = batch["target"].to(DEVICE, dtype=torch.float32)

                optimizer.zero_grad()
                pred = model(image, tabular).squeeze(-1)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                diagnostics.after_update()

                running_loss += loss.item() * image.size(0)
                n_seen += image.size(0)

            train_loss = running_loss / max(n_seen, 1)
            diagnostics.report(epoch=epoch)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

    # ---- Validation inference (real forward pass) ----
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(DEVICE, dtype=torch.float32)
            tabular = batch["tabular"].to(DEVICE, dtype=torch.float32)
            target = batch["target"].to(DEVICE, dtype=torch.float32)
            pred = model(image, tabular).squeeze(-1)
            val_preds.append(pred.cpu().numpy())
            val_targets.append(target.cpu().numpy())
    val_preds = np.concatenate(val_preds).astype(np.float32)
    val_targets = np.concatenate(val_targets).astype(np.float32)
    rmse = float(np.sqrt(mean_squared_error(val_targets, val_preds)))

    # ---- Test inference (identical processing path as validation) ----
    test_preds, test_ids = [], []
    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(DEVICE, dtype=torch.float32)
            tabular = batch["tabular"].to(DEVICE, dtype=torch.float32)
            pred = model(image, tabular).squeeze(-1)
            test_preds.append(pred.cpu().numpy())
            test_ids.extend(batch["id"])
    test_preds = np.concatenate(test_preds).astype(np.float32)
    test_preds = np.clip(test_preds, 0.0, 100.0)

    submission = pd.DataFrame({"Id": test_ids, "Pawpularity": test_preds})
    submission.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {rmse}")
