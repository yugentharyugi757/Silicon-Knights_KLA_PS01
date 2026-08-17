import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
    "best_noise_aware_dncnn.pth"
)

NUM_FEATURES = 64
NUM_BLOCKS = 10

BATCH_SIZE = 16
NUM_EPOCHS = 50

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

NUM_WORKERS = 0

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

print("=" * 70)
print("KLA NOISE-AWARE DnCNN TRAINING")
print("=" * 70)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


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

    # Variance cannot be negative
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

        # ----------------------------------------------------
        # Safety normalization
        # ----------------------------------------------------

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

        # GT is normalized
        gt = np.clip(gt, 0.0, 1.0)

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # The noise model is signal-dependent.
        #
        # We don't know clean x at inference.
        #
        # Therefore use a clipped estimate of the observed
        # noisy image to construct the noise-level map.
        # ----------------------------------------------------

        noisy_for_noise_estimation = np.clip(
            noisy,
            0.0,
            1.0
        )

        sigma = compute_noise_sigma(
            noisy_for_noise_estimation
        )

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        noisy_tensor = torch.from_numpy(
            noisy
        ).unsqueeze(0)

        sigma_tensor = torch.from_numpy(
            sigma
        ).unsqueeze(0)

        gt_tensor = torch.from_numpy(
            gt
        ).unsqueeze(0)

        # ----------------------------------------------------
        # Concatenate:
        #
        # [NoisyLR, NoiseSigma]
        #
        # Shape:
        # [2, 128, 128]
        # ----------------------------------------------------

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
# TRAIN / VALIDATION SPLIT
# ============================================================

pairs = build_pairs()

print(f"Total pairs: {len(pairs)}")

random.Random(SEED).shuffle(pairs)

total = len(pairs)

train_end = int(
    0.80 * total
)

val_end = int(
    0.90 * total
)

train_pairs = pairs[:train_end]

val_pairs = pairs[
    train_end:val_end
]

test_pairs = pairs[
    val_end:
]

print(f"Training : {len(train_pairs)}")
print(f"Validation: {len(val_pairs)}")
print(f"Testing  : {len(test_pairs)}")


# ============================================================
# DATA LOADERS
# ============================================================

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


# ============================================================
# MODEL
# ============================================================

model = NoiseAwareDnCNN(
    num_features=NUM_FEATURES,
    num_blocks=NUM_BLOCKS
).to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = nn.MSELoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
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
# CHECKPOINT DIRECTORY
# ============================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")

history = {
    "train_loss": [],
    "val_loss": []
}


print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(NUM_EPOCHS):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(
        train_loader
    ):

        inputs = inputs.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(inputs)

        loss = criterion(
            outputs,
            targets
        )

        loss.backward()

        # Prevent occasional exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        train_loss += (
            loss.item()
            * inputs.size(0)
        )

    train_loss /= len(train_dataset)


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for inputs, targets in val_loader:

            inputs = inputs.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(inputs)

            loss = criterion(
                outputs,
                targets
            )

            val_loss += (
                loss.item()
                * inputs.size(0)
            )

    val_loss /= len(val_dataset)


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history["train_loss"].append(
        train_loss
    )

    history["val_loss"].append(
        val_loss
    )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        checkpoint = {

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "epoch":
                epoch + 1,

            "best_val_loss":
                best_val_loss,

            "config": {

                "num_features":
                    NUM_FEATURES,

                "num_blocks":
                    NUM_BLOCKS,

                "input_channels":
                    2,

                "upscale":
                    2,

                "variance_model": {

                    "a": VAR_A,

                    "b": VAR_B,

                    "c": VAR_C

                }

            }

        }

        torch.save(
            checkpoint,
            CHECKPOINT_PATH
        )

        saved = "*"
    else:

        saved = ""


    # ========================================================
    # PRINT
    # ========================================================

    print(
        f"Epoch [{epoch+1:03d}/{NUM_EPOCHS}] "
        f"Train: {train_loss:.6f} | "
        f"Val: {val_loss:.6f} | "
        f"LR: {current_lr:.2e} "
        f"{saved}"
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

np.save(
    "noise_aware_training_history.npy",
    history
)


print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best validation loss: "
    f"{best_val_loss:.8f}"
)

print(
    f"Checkpoint: {CHECKPOINT_PATH}"
)