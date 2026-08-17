import os
import csv
import time
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from models.noise_aware_dncnn import NoiseAwareDnCNN


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

GT_DIR = "train/GT"
NOISY_DIR = "train/NoisyLR"

CHECKPOINT_PATH = "checkpoints/best_noise_aware_dncnn.pth"

RESULTS_DIR = "results"
OUTPUT_DIR = "outputs/evaluation_noise_aware"

RESTORED_DIR = os.path.join(
    OUTPUT_DIR, "restored"
)

COMPARISON_DIR = os.path.join(
    OUTPUT_DIR, "comparisons"
)

CSV_PATH = os.path.join(
    RESULTS_DIR,
    "noise_aware_test_results.csv"
)

SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "noise_aware_test_summary.txt"
)

NUM_FEATURES = 64
NUM_BLOCKS = 10

BATCH_SIZE = 1
NUM_WORKERS = 0

# Same model obtained from your noise analysis
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
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# NOISE SIGMA MAP
# ============================================================

def compute_noise_sigma(image):

    variance = (
        VAR_A * image ** 2
        + VAR_B * image
        + VAR_C
    )

    variance = np.maximum(
        variance,
        0.0
    )

    sigma = np.sqrt(variance)

    return sigma.astype(np.float32)


# ============================================================
# DATASET
# ============================================================

class KLATestDataset(Dataset):

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        noisy_path, gt_path = self.pairs[idx]

        noisy = np.load(
            noisy_path
        ).astype(np.float32)

        gt = np.load(
            gt_path
        ).astype(np.float32)

        # Same safety handling as training
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
        gt = np.clip(
            gt,
            0.0,
            1.0
        )

        # IMPORTANT:
        # Same procedure as training.
        noisy_for_sigma = np.clip(
            noisy,
            0.0,
            1.0
        )

        sigma = compute_noise_sigma(
            noisy_for_sigma
        )

        # Channel 0 = noisy LR
        noisy_tensor = torch.from_numpy(
            noisy
        ).unsqueeze(0)

        # Channel 1 = sigma map
        sigma_tensor = torch.from_numpy(
            sigma
        ).unsqueeze(0)

        # GT
        gt_tensor = torch.from_numpy(
            gt
        ).unsqueeze(0)

        # [NoisyLR, Sigma]
        input_tensor = torch.cat(
            [
                noisy_tensor,
                sigma_tensor
            ],
            dim=0
        )

        return (
            input_tensor,
            gt_tensor,
            os.path.basename(noisy_path)
        )


# ============================================================
# BUILD DATASET PAIRS
# ============================================================

def build_pairs():

    gt_files = sorted([
        f
        for f in os.listdir(GT_DIR)
        if f.endswith(".npy")
    ])

    noisy_files = sorted([
        f
        for f in os.listdir(NOISY_DIR)
        if f.endswith(".npy")
    ])

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
# LOAD CHECKPOINT
# ============================================================

def load_model():

    model = NoiseAwareDnCNN(
        num_features=NUM_FEATURES,
        num_blocks=NUM_BLOCKS
    ).to(DEVICE)

    print()
    print("Loading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("Checkpoint loaded successfully.")

    print(
        f"Checkpoint epoch : "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Best val loss    : "
        f"{checkpoint.get('best_val_loss', 'unknown')}"
    )

    model.eval()

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESTORED_DIR,
        exist_ok=True
    )

    os.makedirs(
        COMPARISON_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print("KLA NOISE-AWARE DnCNN TEST EVALUATION")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # BUILD EXACT SAME SPLIT AS TRAINING
    # --------------------------------------------------------

    pairs = build_pairs()

    print(
        f"Total paired images : "
        f"{len(pairs)}"
    )

    random.Random(SEED).shuffle(
        pairs
    )

    total = len(pairs)

    train_end = int(
        0.80 * total
    )

    val_end = int(
        0.90 * total
    )

    test_pairs = pairs[
        val_end:
    ]

    print(
        f"Testing             : "
        f"{len(test_pairs)}"
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    test_dataset = KLATestDataset(
        test_pairs
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        f"Features            : "
        f"{NUM_FEATURES}"
    )

    print(
        f"Residual blocks     : "
        f"{NUM_BLOCKS}"
    )

    model = load_model()

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    results = []

    inference_times = []

    print()
    print("=" * 70)
    print("STARTING EVALUATION")
    print("=" * 70)

    with torch.no_grad():

        for i, (
            inputs,
            targets,
            filenames
        ) in enumerate(test_loader):

            inputs = inputs.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            outputs = model(inputs)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter()
                - start
            )

            inference_ms = (
                elapsed
                * 1000
                / inputs.size(0)
            )

            inference_times.append(
                inference_ms
            )

            # ------------------------------------------------
            # Convert to NumPy
            # ------------------------------------------------

            restored = (
                outputs
                .cpu()
                .numpy()[0, 0]
            )

            gt = (
                targets
                .cpu()
                .numpy()[0, 0]
            )

            # Only channel 0 is the actual noisy image.
            noisy = (
                inputs
                .cpu()
                .numpy()[0, 0]
            )

            # Keep evaluation within valid image range
            restored = np.clip(
                restored,
                0.0,
                1.0
            )

            gt = np.clip(
                gt,
                0.0,
                1.0
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            mse = np.mean(
                (restored - gt) ** 2
            )

            mae = np.mean(
                np.abs(restored - gt)
            )

            psnr = peak_signal_noise_ratio(
                gt,
                restored,
                data_range=1.0
            )

            ssim = structural_similarity(
                gt,
                restored,
                data_range=1.0
            )

            filename = filenames[0]

            results.append({
                "filename": filename,
                "mse": float(mse),
                "mae": float(mae),
                "psnr_db": float(psnr),
                "ssim": float(ssim),
                "inference_ms": float(
                    inference_ms
                )
            })

            # ------------------------------------------------
            # Save restored output
            # ------------------------------------------------

            np.save(
                os.path.join(
                    RESTORED_DIR,
                    filename
                ),
                restored.astype(
                    np.float32
                )
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                (i + 1) % 50 == 0
                or
                (i + 1) == len(test_loader)
            ):

                print(
                    f"Processed "
                    f"{i + 1}/"
                    f"{len(test_loader)}"
                )

    # ========================================================
    # AGGREGATE RESULTS
    # ========================================================

    mean_mse = np.mean([
        x["mse"]
        for x in results
    ])

    mean_mae = np.mean([
        x["mae"]
        for x in results
    ])

    mean_psnr = np.mean([
        x["psnr_db"]
        for x in results
    ])

    mean_ssim = np.mean([
        x["ssim"]
        for x in results
    ])

    mean_time = np.mean(
        inference_times
    )

    median_time = np.median(
        inference_times
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "mse",
                "mae",
                "psnr_db",
                "ssim",
                "inference_ms"
            ]
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = f"""
======================================================================
NOISE-AWARE DnCNN TEST RESULTS
======================================================================
Images evaluated : {len(results)}
Mean MSE         : {mean_mse:.8f}
Mean MAE         : {mean_mae:.8f}
Mean PSNR        : {mean_psnr:.4f} dB
Mean SSIM        : {mean_ssim:.6f}
Mean inference   : {mean_time:.2f} ms/image
Median inference : {median_time:.2f} ms/image
======================================================================
"""

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(summary)

    print(summary)

    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"CSV results : "
        f"{CSV_PATH}"
    )

    print(
        f"Summary     : "
        f"{SUMMARY_PATH}"
    )

    print(
        f"Restored    : "
        f"{RESTORED_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()