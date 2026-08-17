import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.noise_aware_dncnn import NoiseAwareDnCNN


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

GT_DIR = "train/GT"
NOISY_DIR = "train/NoisyLR"

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_noise_aware_final.pth"
)

RESULTS_DIR = "results"
TRAINING_HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "final_training_history.npy"
)

NUM_FEATURES = 64
NUM_BLOCKS = 10

BATCH_SIZE = 16
NUM_EPOCHS = 50

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

NUM_WORKERS = 0

ALPHA = 0.8
BETA = 0.2

# Empirical variance model obtained from your dataset
VAR_A = 0.01846589
VAR_B = 0.00843552
VAR_C = -0.00033869


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# NOISE MAP
# ============================================================

def compute_noise_sigma(image):

    """
    Compute estimated signal-dependent noise standard deviation.

    Variance model:

        sigma²(x) =
            0.01846589*x²
            + 0.00843552*x
            - 0.00033869

    x is assumed to be normalized to [0,1].
    """

    variance = (
        VAR_A * image ** 2
        + VAR_B * image
        + VAR_C
    )

    variance = np.maximum(variance, 0.0)
    sigma = np.sqrt(variance)

    return sigma.astype(np.float32)


# ============================================================
# DATASET
# ============================================================

class KLADataset(Dataset):

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, gt_path = self.pairs[idx]

        noisy = np.load(noisy_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)

        noisy = np.nan_to_num(
            noisy,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        gt = np.nan_to_num(
            gt,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        gt = np.clip(gt, 0.0, 1.0)

        noisy_for_noise_estimation = np.clip(
            noisy,
            0.0,
            1.0
        )

        sigma = compute_noise_sigma(
            noisy_for_noise_estimation
        )

        noisy_tensor = torch.from_numpy(
            noisy
        ).unsqueeze(0)

        sigma_tensor = torch.from_numpy(
            sigma
        ).unsqueeze(0)

        gt_tensor = torch.from_numpy(
            gt
        ).unsqueeze(0)

        input_tensor = torch.cat(
            [
                noisy_tensor,
                sigma_tensor
            ],
            dim=0
        )

        return input_tensor, gt_tensor


# ============================================================
# BUILD PAIRS
# ============================================================

def build_pairs():

    gt_files = sorted(
        [
            f for f in os.listdir(GT_DIR)
            if f.endswith(".npy")
        ]
    )

    noisy_files = sorted(
        [
            f for f in os.listdir(NOISY_DIR)
            if f.endswith(".npy")
        ]
    )

    noisy_set = set(noisy_files)
    pairs = []

    for filename in gt_files:
        if filename in noisy_set:
            noisy_path = os.path.join(
                NOISY_DIR,
                filename
            )

            gt_path = os.path.join(
                GT_DIR,
                filename
            )

            pairs.append(
                (
                    noisy_path,
                    gt_path
                )
            )

    return pairs


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("KLA NOISE-AWARE DnCNN FINAL TRAINING")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pairs = build_pairs()
    print(f"Total paired images: {len(pairs)}")

    if len(pairs) != 3200:
        raise ValueError(
            f"Expected 3200 paired images, but found {len(pairs)}"
        )

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

    train_dataset = KLADataset(train_pairs)
    val_dataset = KLADataset(val_pairs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    model = NoiseAwareDnCNN(
        num_features=NUM_FEATURES,
        num_blocks=NUM_BLOCKS
    ).to(DEVICE)

    sample_input, sample_target = next(iter(train_loader))
    sample_input = sample_input.to(DEVICE)
    sample_target = sample_target.to(DEVICE)

    if sample_input.shape[1:] != (2, 128, 128):
        raise ValueError(
            f"Model input shape mismatch: expected [B,2,H,W] with H=W=128, got {sample_input.shape}"
        )

   # GT is the 2x higher-resolution target.
# Input:  [B, 2, 128, 128]
# Target: [B, 1, 256, 256]
    if sample_target.shape[1:] != (1, 256, 256):
        raise ValueError(
            f"Target shape mismatch: expected [B,1,256,256], "
            f"got {sample_target.shape}"
        )

    model.eval()
    with torch.no_grad():
        dummy_output = model(sample_input)

    print(f"Safety check input shape: {sample_input.shape}")
    print(f"Safety check target shape: {sample_target.shape}")
    print(f"Safety check output shape: {dummy_output.shape}")

    # Model must output the 2x upscaled image.
    if dummy_output.shape[1:] != (1, 256, 256):
        raise ValueError(
            f"Model output shape mismatch: expected [B,1,256,256], "
            f"got {dummy_output.shape}"
        )

    model.train()

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    start_time = time.time()

    history = {
        "train_loss": [],
        "val_loss": []
    }

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            mse = mse_loss(outputs, targets)
            l1 = l1_loss(outputs, targets)
            loss = ALPHA * mse + BETA * l1

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                outputs = model(inputs)
                mse = mse_loss(outputs, targets)
                l1 = l1_loss(outputs, targets)
                loss = ALPHA * mse + BETA * l1

                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_dataset)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "config": {
                    "num_features": NUM_FEATURES,
                    "num_blocks": NUM_BLOCKS,
                    "input_channels": 2,
                    "upscale": 2,
                    "variance_model": {
                        "a": VAR_A,
                        "b": VAR_B,
                        "c": VAR_C
                    },
                    "loss_type": "0.8*MSE + 0.2*L1",
                    "alpha": ALPHA,
                    "beta": BETA
                }
            }

            torch.save(
                checkpoint,
                CHECKPOINT_PATH
            )

        print(
            f"Epoch [{epoch + 1:03d}/{NUM_EPOCHS}] "
            f"Train loss: {train_loss:.6f} | "
            f"Validation loss: {val_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )

    np.save(
        TRAINING_HISTORY_PATH,
        history
    )

    total_training_time = time.time() - start_time

    print()
    print("=" * 70)
    print("FINAL TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.8f}")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")
    print(f"Total training time: {total_training_time:.2f} seconds")


if __name__ == "__main__":
    main()
