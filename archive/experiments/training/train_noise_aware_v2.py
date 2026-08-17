import ast
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

from models.noise_aware_dncnn_v2 import NoiseAwareDnCNNV2


# ============================================================
# REUSE V1 DATASET / CONFIG LOGIC WITHOUT EXECUTING TRAINING
# ============================================================

train_noise_aware_path = PROJECT_ROOT / "train_noise_aware.py"
with open(train_noise_aware_path, "r", encoding="utf-8") as f:
    source = f.read()

module = ast.parse(source)

allowed_names = {
    "SEED",
    "GT_DIR",
    "NOISY_DIR",
    "CHECKPOINT_DIR",
    "NUM_FEATURES",
    "NUM_BLOCKS",
    "BATCH_SIZE",
    "NUM_EPOCHS",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "NUM_WORKERS",
    "VAR_A",
    "VAR_B",
    "VAR_C",
    "compute_noise_sigma",
    "KLADataset",
    "build_pairs",
}

selected_nodes = []
for node in module.body:
    if isinstance(node, ast.Assign):
        names = [
            target.id
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
        if any(name in allowed_names for name in names):
            selected_nodes.append(node)
    elif isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name) and node.target.id in allowed_names:
            selected_nodes.append(node)
    elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        if node.name in allowed_names:
            selected_nodes.append(node)

namespace = {
    "os": os,
    "random": random,
    "np": np,
    "torch": torch,
    "nn": nn,
    "Dataset": Dataset,
    "DataLoader": DataLoader,
}
exec(compile(ast.Module(body=selected_nodes, type_ignores=[]), str(train_noise_aware_path), "exec"), namespace)

SEED = namespace["SEED"]
GT_DIR = namespace["GT_DIR"]
NOISY_DIR = namespace["NOISY_DIR"]
CHECKPOINT_DIR = namespace["CHECKPOINT_DIR"]
NUM_FEATURES = namespace["NUM_FEATURES"]
NUM_BLOCKS = namespace["NUM_BLOCKS"]
BATCH_SIZE = namespace["BATCH_SIZE"]
NUM_EPOCHS = namespace["NUM_EPOCHS"]
LEARNING_RATE = namespace["LEARNING_RATE"]
WEIGHT_DECAY = namespace["WEIGHT_DECAY"]
NUM_WORKERS = namespace["NUM_WORKERS"]
VAR_A = namespace["VAR_A"]
VAR_B = namespace["VAR_B"]
VAR_C = namespace["VAR_C"]
compute_noise_sigma = namespace["compute_noise_sigma"]
KLADataset = namespace["KLADataset"]
build_pairs = namespace["build_pairs"]

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_noise_aware_v2.pth"
)


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


def main():
    print("=" * 70)
    print("KLA NOISE-AWARE DnCNN V2 TRAINING")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: CPU")

    pairs = build_pairs()
    print(f"Total paired images: {len(pairs)}")

    random.Random(SEED).shuffle(pairs)

    total = len(pairs)
    train_end = int(0.80 * total)
    val_end = int(0.90 * total)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    print(f"Train count: {len(train_pairs)}")
    print(f"Validation count: {len(val_pairs)}")
    print(f"Test count: {len(test_pairs)}")
    print(f"Features: {NUM_FEATURES}")
    print(f"Residual blocks: {NUM_BLOCKS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    criterion = nn.MSELoss()
    print(f"Loss function: {criterion.__class__.__name__}")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")

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

    model = NoiseAwareDnCNNV2(
        num_features=64,
        num_blocks=10
    ).to(DEVICE)

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

    best_val_loss = float("inf")
    best_epoch = 0
    start_time = time.time()

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
            loss = criterion(outputs, targets)
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
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_dataset)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1:03d}/{NUM_EPOCHS}] "
            f"Train loss: {train_loss:.6f} | "
            f"Validation loss: {val_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            compatibility_model = NoiseAwareDnCNNV2(
                num_features=64,
                num_blocks=10
            )
            compatibility_model.load_state_dict(model.state_dict())

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
                        "c": VAR_C,
                    },
                },
            }

            torch.save(checkpoint, CHECKPOINT_PATH)
            print(f"Checkpoint saved: {CHECKPOINT_PATH}")

    total_training_time = time.time() - start_time

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.8f}")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")
    print(f"Total training time: {total_training_time:.2f} seconds")


if __name__ == "__main__":
    main()
