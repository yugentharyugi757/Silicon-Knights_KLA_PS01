import os
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

GT_DIR = "train/GT"
NOISY_DIR = "train/NoisyLR"

CHECKPOINT_DIR = "checkpoints"
RESULT_DIR = "results"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Dataset split
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Training
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

# Training crop
LR_PATCH_SIZE = 64
SCALE = 2

# Model
NUM_FEATURES = 64
NUM_BLOCKS = 10

# Workers
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# DATASET
# ============================================================

class KLADataset(Dataset):

    def __init__(
        self,
        filenames,
        training=False
    ):

        self.filenames = filenames
        self.training = training

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):

        filename = self.filenames[index]

        noisy_path = os.path.join(
            NOISY_DIR,
            filename
        )

        gt_path = os.path.join(
            GT_DIR,
            filename
        )

        # ----------------------------------------------------
        # Load numpy arrays
        # ----------------------------------------------------

        noisy = np.load(noisy_path).astype(
            np.float32
        )

        gt = np.load(gt_path).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Training crop
        # ----------------------------------------------------

        if self.training:

            h, w = noisy.shape

            ps = LR_PATCH_SIZE

            top = np.random.randint(
                0,
                h - ps + 1
            )

            left = np.random.randint(
                0,
                w - ps + 1
            )

            noisy = noisy[
                top:top + ps,
                left:left + ps
            ]

            gt_top = top * SCALE
            gt_left = left * SCALE

            gt = gt[
                gt_top:gt_top + ps * SCALE,
                gt_left:gt_left + ps * SCALE
            ]

            # ------------------------------------------------
            # Random horizontal flip
            # ------------------------------------------------

            if np.random.random() < 0.5:

                noisy = np.flip(
                    noisy,
                    axis=1
                ).copy()

                gt = np.flip(
                    gt,
                    axis=1
                ).copy()

            # ------------------------------------------------
            # Random vertical flip
            # ------------------------------------------------

            if np.random.random() < 0.5:

                noisy = np.flip(
                    noisy,
                    axis=0
                ).copy()

                gt = np.flip(
                    gt,
                    axis=0
                ).copy()

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        noisy = torch.from_numpy(
            noisy
        ).unsqueeze(0)

        gt = torch.from_numpy(
            gt
        ).unsqueeze(0)

        return noisy, gt, filename


# ============================================================
# DATASET SPLIT
# ============================================================

all_files = sorted([
    f
    for f in os.listdir(NOISY_DIR)
    if f.endswith(".npy")
])


# Make sure every noisy image has GT
all_files = [
    f for f in all_files
    if os.path.exists(
        os.path.join(GT_DIR, f)
    )
]


print("=" * 70)
print("KLA DATASET")
print("=" * 70)

print(
    f"Total paired images: {len(all_files)}"
)


random.Random(SEED).shuffle(all_files)

n = len(all_files)

train_end = int(
    TRAIN_RATIO * n
)

val_end = int(
    (TRAIN_RATIO + VAL_RATIO) * n
)

train_files = all_files[:train_end]

val_files = all_files[
    train_end:val_end
]

test_files = all_files[
    val_end:
]


print(
    f"Training   : {len(train_files)}"
)

print(
    f"Validation : {len(val_files)}"
)

print(
    f"Testing    : {len(test_files)}"
)

print("=" * 70)


# ============================================================
# DATALOADERS
# ============================================================

train_dataset = KLADataset(
    train_files,
    training=True
)

val_dataset = KLADataset(
    val_files,
    training=False
)

test_dataset = KLADataset(
    test_files,
    training=False
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# DnCNN RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(
            channels
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(
            channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

    def forward(self, x):

        residual = x

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        out = out + residual

        out = self.relu(out)

        return out


# ============================================================
# DnCNN ×2 RESTORATION MODEL
# ============================================================

class DnCNNRestorer(nn.Module):

    def __init__(
        self,
        num_features=64,
        num_blocks=10
    ):

        super().__init__()

        # ----------------------------------------------------
        # Initial feature extraction
        # ----------------------------------------------------

        self.head = nn.Sequential(

            nn.Conv2d(
                1,
                num_features,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            )
        )

        # ----------------------------------------------------
        # Residual feature extraction
        # ----------------------------------------------------

        blocks = []

        for _ in range(num_blocks):

            blocks.append(
                ResidualBlock(
                    num_features
                )
            )

        self.body = nn.Sequential(
            *blocks
        )

        # ----------------------------------------------------
        # Feature reconstruction
        # ----------------------------------------------------

        self.reconstruction = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # PixelShuffle ×2
        # ----------------------------------------------------

        self.upsample = nn.Sequential(

            nn.Conv2d(
                num_features,
                num_features * 4,
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(2),

            nn.ReLU(
                inplace=True
            )
        )

        # ----------------------------------------------------
        # Output layer
        # ----------------------------------------------------

        self.output = nn.Conv2d(
            num_features,
            1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        # LR features
        features = self.head(x)

        # Deep residual features
        body = self.body(features)

        # Global residual connection
        body = body + features

        body = self.reconstruction(body)

        # Upsampling
        out = self.upsample(body)

        # Final reconstruction
        out = self.output(out)

        return out


# ============================================================
# MODEL
# ============================================================

model = DnCNNRestorer(
    num_features=NUM_FEATURES,
    num_blocks=NUM_BLOCKS
).to(DEVICE)


print("\nMODEL")
print("=" * 70)

print(model)

num_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"\nTrainable parameters: "
    f"{num_parameters:,}"
)

print(
    f"Device: {DEVICE}"
)

print("=" * 70)


# ============================================================
# LOSS
# ============================================================

criterion = nn.MSELoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    prediction,
    target
):

    prediction = np.clip(
        prediction,
        0.0,
        1.0
    )

    target = np.clip(
        target,
        0.0,
        1.0
    )

    mse = np.mean(
        (prediction - target) ** 2
    )

    psnr = peak_signal_noise_ratio(
        target,
        prediction,
        data_range=1.0
    )

    ssim = structural_similarity(
        target,
        prediction,
        data_range=1.0
    )

    return mse, psnr, ssim


# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0

    psnr_values = []
    ssim_values = []

    with torch.no_grad():

        for noisy, gt, _ in val_loader:

            noisy = noisy.to(
                DEVICE,
                non_blocking=True
            )

            gt = gt.to(
                DEVICE,
                non_blocking=True
            )

            prediction = model(
                noisy
            )

            loss = criterion(
                prediction,
                gt
            )

            total_loss += loss.item()

            pred_np = (
                prediction
                .squeeze()
                .cpu()
                .numpy()
            )

            gt_np = (
                gt
                .squeeze()
                .cpu()
                .numpy()
            )

            _, psnr, ssim = calculate_metrics(
                pred_np,
                gt_np
            )

            psnr_values.append(psnr)
            ssim_values.append(ssim)

    avg_loss = (
        total_loss /
        len(val_loader)
    )

    avg_psnr = np.mean(
        psnr_values
    )

    avg_ssim = np.mean(
        ssim_values
    )

    return (
        avg_loss,
        avg_psnr,
        avg_ssim
    )


# ============================================================
# TRAINING
# ============================================================

train_losses = []
val_losses = []
val_psnr_history = []
val_ssim_history = []


best_val_psnr = -float("inf")

patience = 10
epochs_without_improvement = 0


print("\n")
print("=" * 70)
print("TRAINING STARTED")
print("=" * 70)


for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{EPOCHS}"
    )

    for noisy, gt, _ in progress:

        noisy = noisy.to(
            DEVICE,
            non_blocking=True
        )

        gt = gt.to(
            DEVICE,
            non_blocking=True
        )

        # ------------------------------------------------
        # Forward
        # ------------------------------------------------

        prediction = model(
            noisy
        )

        # ------------------------------------------------
        # Loss
        # ------------------------------------------------

        loss = criterion(
            prediction,
            gt
        )

        # ------------------------------------------------
        # Backprop
        # ------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.6f}"
        )

    avg_train_loss = (
        running_loss /
        len(train_loader)
    )

    # ----------------------------------------------------
    # Validation
    # ----------------------------------------------------

    (
        val_loss,
        val_psnr,
        val_ssim
    ) = validate()

    scheduler.step(
        val_loss
    )

    train_losses.append(
        avg_train_loss
    )

    val_losses.append(
        val_loss
    )

    val_psnr_history.append(
        val_psnr
    )

    val_ssim_history.append(
        val_ssim
    )

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"\nEpoch {epoch:03d} | "
        f"Train Loss: {avg_train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"Val PSNR: {val_psnr:.4f} dB | "
        f"Val SSIM: {val_ssim:.6f} | "
        f"LR: {current_lr:.2e}"
    )

    # ----------------------------------------------------
    # Save best model
    # ----------------------------------------------------

    if val_psnr > best_val_psnr:

        best_val_psnr = val_psnr

        epochs_without_improvement = 0

        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "val_psnr":
                val_psnr,

            "val_ssim":
                val_ssim,

            "config": {

                "num_features":
                    NUM_FEATURES,

                "num_blocks":
                    NUM_BLOCKS,

                "scale":
                    SCALE
            }
        }

        torch.save(
            checkpoint,
            os.path.join(
                CHECKPOINT_DIR,
                "best_dncnn.pth"
            )
        )

        print(
            f"✓ Best model saved "
            f"(PSNR={val_psnr:.4f} dB)"
        )

    else:

        epochs_without_improvement += 1

    # ----------------------------------------------------
    # Early stopping
    # ----------------------------------------------------

    if epochs_without_improvement >= patience:

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# TRAINING CURVES
# ============================================================

epochs_done = range(
    1,
    len(train_losses) + 1
)


plt.figure(figsize=(8, 5))

plt.plot(
    epochs_done,
    train_losses,
    label="Training Loss"
)

plt.plot(
    epochs_done,
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "loss_curve.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# PSNR CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_done,
    val_psnr_history,
    label="Validation PSNR"
)

plt.axhline(
    24.6331,
    linestyle="--",
    label="Bilateral Baseline"
)

plt.xlabel("Epoch")

plt.ylabel("PSNR (dB)")

plt.title(
    "Validation PSNR"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "psnr_curve.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# SSIM CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_done,
    val_ssim_history,
    label="Validation SSIM"
)

plt.axhline(
    0.611477,
    linestyle="--",
    label="Bilateral Baseline"
)

plt.xlabel("Epoch")

plt.ylabel("SSIM")

plt.title(
    "Validation SSIM"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "ssim_curve.png"
    ),
    dpi=150
)

plt.close()


print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best validation PSNR: "
    f"{best_val_psnr:.4f} dB"
)

print(
    "Best model:"
)

print(
    os.path.join(
        CHECKPOINT_DIR,
        "best_dncnn.pth"
    )
)

print("=" * 70)
