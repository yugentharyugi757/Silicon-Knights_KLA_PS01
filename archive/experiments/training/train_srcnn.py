import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.super_resolution.srcnn import SRCNN


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
SCALE_FACTOR = 2
NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0

GT_DIR = PROJECT_ROOT / "train" / "GT"
NOISY_DIR = PROJECT_ROOT / "train" / "NoisyLR"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_srcnn_x2.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Keep the repository's existing split behavior exactly as used by
# the Noise-Aware DnCNN training/evaluation scripts.


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATASET
# ============================================================


def build_pairs():
    gt_files = sorted([f for f in os.listdir(GT_DIR) if f.endswith(".npy")])
    noisy_files = sorted([f for f in os.listdir(NOISY_DIR) if f.endswith(".npy")])
    noisy_set = set(noisy_files)

    pairs = []
    for filename in gt_files:
        if filename in noisy_set:
            noisy_path = os.path.join(NOISY_DIR, filename)
            gt_path = os.path.join(GT_DIR, filename)
            pairs.append((noisy_path, gt_path))

    return pairs


class SRDataset(Dataset):
    def __init__(self, pairs, scale_factor=2):
        self.pairs = pairs
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, gt_path = self.pairs[idx]

        gt = np.load(gt_path).astype(np.float32)
        gt = np.nan_to_num(gt, nan=0.0, posinf=1.0, neginf=0.0)
        gt = np.clip(gt, 0.0, 1.0)

        if gt.ndim != 2:
            raise ValueError(f"Expected 2D grayscale GT image, got shape {gt.shape} for {gt_path}")

        hr = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)
        h, w = hr.shape[-2:]
        output_size = (max(1, h // self.scale_factor), max(1, w // self.scale_factor))

        lr = F.interpolate(
            hr,
            size=output_size,
            mode="bicubic",
            align_corners=False,
        )

        input_tensor = F.interpolate(
            lr,
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return input_tensor.squeeze(0), hr.squeeze(0)


# ============================================================
# MODEL / TRAINING
# ============================================================


def load_split():
    pairs = build_pairs()
    print(f"Total paired images: {len(pairs)}")

    random.Random(SEED).shuffle(pairs)
    total = len(pairs)
    train_end = int(0.80 * total)
    val_end = int(0.90 * total)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    print(f"Training : {len(train_pairs)}")
    print(f"Validation: {len(val_pairs)}")
    print(f"Testing  : {len(test_pairs)}")

    return train_pairs, val_pairs, test_pairs



def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("=" * 70)
    print("SRCNN SUPER-RESOLUTION TRAINING")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_pairs, val_pairs, _ = load_split()

    train_dataset = SRDataset(train_pairs, scale_factor=SCALE_FACTOR)
    val_dataset = SRDataset(val_pairs, scale_factor=SCALE_FACTOR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = SRCNN(
        input_channels=1,
        output_channels=1,
        first_features=64,
        first_kernel_size=9,
        second_features=32,
        second_kernel_size=1,
        third_kernel_size=5,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        avg_train_loss = train_loss / len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        avg_val_loss = val_loss / len(val_dataset)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                    "scale_factor": SCALE_FACTOR,
                    "model_config": {
                        "input_channels": 1,
                        "output_channels": 1,
                        "first_features": 64,
                        "first_kernel_size": 9,
                        "second_features": 32,
                        "second_kernel_size": 1,
                        "third_kernel_size": 5,
                    },
                },
                CHECKPOINT_PATH,
            )

        print(f"Epoch {epoch + 1:02d}/{NUM_EPOCHS:02d} | train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f} | best_val_loss={best_val_loss:.6f}")

    print("\nTraining complete.")
    print(f"Best validation epoch: {best_epoch}")
    print(f"Best validation loss : {best_val_loss:.6f}")
    print(f"Checkpoint saved to  : {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
