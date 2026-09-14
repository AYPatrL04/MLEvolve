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

try:
    from utils.training_diagnostics import TrainingDiagnostics
except ImportError:

    class TrainingDiagnostics:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def after_update(self):
            pass

        def report(self, epoch=None):
            pass

        def state_dict(self):
            return {}

        def load_state_dict(self, *args, **kwargs):
            pass


IMG_SIZE = 256
TAB_COLS = [
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


class TinyPetNet(nn.Module):
    """Tiny CNN (2 conv layers: 8 -> 16 channels) + small tabular branch, fused."""

    def __init__(self, num_tab_features=len(TAB_COLS)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.tab_fc1 = nn.Linear(num_tab_features, 16)
        self.tab_fc2 = nn.Linear(16, 16)

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, image, tabular):
        x = F.relu(self.conv1(image))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = self.global_pool(x).flatten(1)  # [B, 16]

        t = F.relu(self.tab_fc1(tabular))
        t = F.relu(self.tab_fc2(t))  # [B, 16]

        fused = torch.cat([x, t], dim=1)  # [B, 32]
        fused = self.dropout(fused)
        out = F.relu(self.fc1(fused))
        out = self.fc2(out)
        return out


class PetfinderImageDataset(Dataset):
    """Loads image + tabular features; preprocessing stats must be fit on train only."""

    def __init__(
        self,
        df,
        img_dir,
        tab_cols,
        img_mean,
        img_std,
        tab_mean,
        tab_std,
        has_target=True,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tab_cols = tab_cols
        self.img_mean = img_mean
        self.img_std = img_std
        self.tab_mean = tab_mean
        self.tab_std = tab_std
        self.has_target = has_target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['Id']}.jpg")
        img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - self.img_mean) / self.img_std
        arr = np.transpose(arr, (2, 0, 1)).copy()
        image_tensor = torch.from_numpy(arr).float()

        tab = row[self.tab_cols].to_numpy().astype(np.float32)
        tab = (tab - self.tab_mean) / self.tab_std
        tab_tensor = torch.from_numpy(tab).float()

        if self.has_target:
            target = torch.tensor(float(row["Pawpularity"]), dtype=torch.float32)
            return image_tensor, tab_tensor, target
        else:
            return image_tensor, tab_tensor, str(row["Id"])


def compute_image_stats(df, img_dir):
    imgs = []
    for id_ in df["Id"]:
        img = (
            Image.open(os.path.join(img_dir, f"{id_}.jpg"))
            .convert("RGB")
            .resize((IMG_SIZE, IMG_SIZE))
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0
        imgs.append(arr)
    imgs = np.stack(imgs, axis=0)
    mean = imgs.mean(axis=(0, 1, 2)).astype(np.float32)
    std = (imgs.std(axis=(0, 1, 2)) + 1e-6).astype(np.float32)
    return mean, std


class CandidateAdapter:
    """CPU-safe adapter reusing the real TinyPetNet model, loss, and optimizer."""

    def __init__(self):
        self.default_context = {"device": "cpu", "lr": 1e-3, "criterion": None}

    def _merged(self, context):
        merged = dict(self.default_context)
        if context:
            merged.update(context)
        return merged

    def build_model(self, context):
        context = self._merged(context)
        model = TinyPetNet(num_tab_features=len(TAB_COLS))
        return model.to(context.get("device", "cpu"))

    def build_optimizer(self, model, context):
        context = self._merged(context)
        lr = context.get("lr", 1e-3)
        return torch.optim.Adam(model.parameters(), lr=lr)

    def _build_batch(self, scenario, device):
        scenario = scenario or {}
        batch_size = int(scenario.get("batch_size", 4))
        fixture = scenario.get("fixture") or {}
        device = device or "cpu"

        image_shape = tuple(fixture.get("image", (3, IMG_SIZE, IMG_SIZE)))
        tab_shape = tuple(fixture.get("tabular", (len(TAB_COLS),)))

        image = torch.rand(batch_size, *image_shape, dtype=torch.float32, device=device)
        tabular = torch.randint(0, 2, (batch_size, *tab_shape), device=device).float()
        target = torch.rand(batch_size, dtype=torch.float32, device=device) * 100.0
        return {"image": image, "tabular": tabular, "target": target}

    def build_train_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def build_validation_batch(self, scenario, device):
        return self._build_batch(scenario, device)

    def training_step(self, model, batch, context):
        context = self._merged(context)
        criterion = context.get("criterion") or nn.MSELoss()
        image = batch["image"]
        tabular = batch["tabular"]
        target = batch["target"].float()
        pred = model(image, tabular).squeeze(-1)
        loss = criterion(pred.float(), target)
        return loss

    def validation_step(self, model, batch, context):
        context = self._merged(context)
        criterion = context.get("criterion") or nn.MSELoss()
        image = batch["image"]
        tabular = batch["tabular"]
        target = batch["target"].float()
        with torch.no_grad():
            pred = model(image, tabular).squeeze(-1)
            loss = criterion(pred.float(), target)
        return loss


if __name__ == "__main__":
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    INPUT_DIR = os.environ.get("MLEVOLVE_INPUT_DIR", "./input")
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")
    SUB_DIR = "./submission"
    os.makedirs(SUB_DIR, exist_ok=True)
    os.makedirs("./working", exist_ok=True)

    # Note: this is synthetic audit evidence (tiny dataset); results are not leaderboard-quality claims.
    train_df_full = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    train_df, val_df = train_test_split(
        train_df_full, test_size=0.25, random_state=RANDOM_SEED
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Fit all preprocessing statistics on TRAIN split only.
    img_mean, img_std = compute_image_stats(train_df, TRAIN_IMG_DIR)
    tab_mean = train_df[TAB_COLS].mean().to_numpy().astype(np.float32)
    tab_std = train_df[TAB_COLS].std().to_numpy().astype(np.float32)
    tab_std[tab_std < 1e-6] = 1.0

    train_ds = PetfinderImageDataset(
        train_df,
        TRAIN_IMG_DIR,
        TAB_COLS,
        img_mean,
        img_std,
        tab_mean,
        tab_std,
        has_target=True,
    )
    val_ds = PetfinderImageDataset(
        val_df,
        TRAIN_IMG_DIR,
        TAB_COLS,
        img_mean,
        img_std,
        tab_mean,
        tab_std,
        has_target=True,
    )
    test_ds = PetfinderImageDataset(
        test_df,
        TEST_IMG_DIR,
        TAB_COLS,
        img_mean,
        img_std,
        tab_mean,
        tab_std,
        has_target=False,
    )

    physical_batch_size = 4
    train_loader = DataLoader(
        train_ds, batch_size=physical_batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=physical_batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_ds, batch_size=physical_batch_size, shuffle=False, num_workers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyPetNet(num_tab_features=len(TAB_COLS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    planned_epochs = 2
    steps_per_epoch = max(1, math.ceil(len(train_ds) / physical_batch_size))

    precision_choice = "fp32"
    precision_reason = (
        "Tiny CNN+tabular model on a small synthetic dataset; FP32 is numerically "
        "safe and no accelerated AMP training path is available/needed at this scale."
    )

    settings = {
        "physical_batch_size": physical_batch_size,
        "effective_batch_size": physical_batch_size,
        "planned_epochs": planned_epochs,
        "steps_per_epoch": steps_per_epoch,
        "model_family": "tiny_cnn_tabular_petfinder",
        "split_seed": RANDOM_SEED,
        "precision": precision_choice,
        "precision_reason": precision_reason,
    }

    with TrainingDiagnostics(
        model, optimizer, scaler=None, settings=settings
    ) as diagnostics:
        for epoch in range(1, planned_epochs + 1):
            model.train()
            running_loss = 0.0
            n_batches = 0
            for image, tabular, target in train_loader:
                image = image.to(device)
                tabular = tabular.to(device)
                target = target.to(device).float()

                optimizer.zero_grad()
                pred = model(image, tabular).squeeze(-1)
                loss = criterion(pred.float(), target)
                loss.backward()
                optimizer.step()
                diagnostics.after_update()

                running_loss += loss.item()
                n_batches += 1

            epoch_loss = running_loss / max(1, n_batches)
            diagnostics.report(epoch=epoch)
            print(f"Epoch {epoch}/{planned_epochs} - train_loss(MSE): {epoch_loss:.4f}")

    # Validation inference (identical processing pipeline as test).
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for image, tabular, target in val_loader:
            image = image.to(device)
            tabular = tabular.to(device)
            pred = model(image, tabular).squeeze(-1)
            val_preds.append(pred.cpu().numpy())
            val_targets.append(target.numpy())
    val_preds = np.clip(np.concatenate(val_preds).astype(np.float64), 0.0, 100.0)
    val_targets = np.concatenate(val_targets).astype(np.float64)
    rmse = float(np.sqrt(np.mean((val_preds - val_targets) ** 2)))

    # Test inference (same model, same preprocessing/clipping logic as validation).
    test_preds = []
    test_ids = []
    with torch.no_grad():
        for image, tabular, ids in test_loader:
            image = image.to(device)
            tabular = tabular.to(device)
            pred = model(image, tabular).squeeze(-1)
            test_preds.append(pred.cpu().numpy())
            test_ids.extend(list(ids))
    test_preds = np.clip(np.concatenate(test_preds).astype(np.float64), 0.0, 100.0)

    submission = pd.DataFrame({"Id": test_ids, "Pawpularity": test_preds})
    submission.to_csv(os.path.join(SUB_DIR, "submission.csv"), index=False)

    print(f"Final Validation Score: {rmse}")
