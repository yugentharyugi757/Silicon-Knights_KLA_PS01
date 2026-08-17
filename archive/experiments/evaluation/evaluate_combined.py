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
OUTPUT_DIR = "outputs/combined_evaluation"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Gaussian noise strengths
GAUSSIAN_SIGMAS = [
    0.010,
    0.030,
    0.050
]

# Speckle noise strengths
SPECKLE_SIGMAS = [
    0.050,
    0.100,
    0.200
]

NUM_FEATURES = 64
NUM_BLOCKS = 10

BATCH_SIZE = 1
NUM_WORKERS = 0

# Same noise model used during training
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
print("KLA COMBINED DEGRADATION EVALUATION")
print("=" * 70)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
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

    sigma = np.sqrt(
        variance
    )

    return sigma.astype(
        np.float32
    )


# ============================================================
# BUILD PAIRED DATA
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

            pairs.append(
                (
                    os.path.join(
                        NOISY_DIR,
                        filename
                    ),
                    os.path.join(
                        GT_DIR,
                        filename
                    ),
                    filename
                )
            )

    return pairs


# ============================================================
# LOAD DATA
# ============================================================

pairs = build_pairs()

print(f"Total paired images : {len(pairs)}")


# Same deterministic 80/10/10 split
random.Random(SEED).shuffle(pairs)

total = len(pairs)

train_end = int(
    0.80 * total
)

val_end = int(
    0.90 * total
)

test_pairs = pairs[val_end:]

print(
    f"Testing             : {len(test_pairs)}"
)


# ============================================================
# DATASET
# ============================================================

class CombinedDataset(Dataset):

    def __init__(
        self,
        pairs
    ):
        self.pairs = pairs

    def __len__(self):

        return len(self.pairs)

    def __getitem__(self, idx):

        noisy_path, gt_path, filename = (
            self.pairs[idx]
        )

        noisy = np.load(
            noisy_path
        ).astype(np.float32)

        gt = np.load(
            gt_path
        ).astype(np.float32)

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

        gt = np.clip(
            gt,
            0.0,
            1.0
        )

        return (
            noisy,
            gt,
            filename
        )


dataset = CombinedDataset(
    test_pairs
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print()
print("Loading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

config = checkpoint.get(
    "config",
    {}
)

num_features = config.get(
    "num_features",
    NUM_FEATURES
)

num_blocks = config.get(
    "num_blocks",
    NUM_BLOCKS
)

model = NoiseAwareDnCNN(
    num_features=num_features,
    num_blocks=num_blocks
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

print("Checkpoint loaded successfully.")

print(
    f"Checkpoint epoch : "
    f"{checkpoint.get('epoch', 'N/A')}"
)

print(
    f"Best val loss    : "
    f"{checkpoint.get('best_val_loss', 'N/A')}"
)

print(
    f"Features         : {num_features}"
)

print(
    f"Residual blocks  : {num_blocks}"
)


# ============================================================
# METRICS
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

    mae = np.mean(
        np.abs(
            prediction - target
        )
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

    return (
        mse,
        mae,
        psnr,
        ssim
    )


# ============================================================
# COMBINED EVALUATION
# ============================================================

all_results = []

print()
print("=" * 70)
print("STARTING COMBINED DEGRADATION EVALUATION")
print("=" * 70)


for gaussian_sigma in GAUSSIAN_SIGMAS:

    for speckle_sigma in SPECKLE_SIGMAS:

        print()
        print("=" * 70)

        print(
            f"GAUSSIAN σ = {gaussian_sigma:.3f} | "
            f"SPECKLE σ = {speckle_sigma:.3f}"
        )

        print("=" * 70)

        mse_values = []
        mae_values = []
        psnr_values = []
        ssim_values = []
        inference_times = []

        visual_saved = False

        with torch.no_grad():

            for index, (
                noisy,
                gt,
                filenames
            ) in enumerate(loader):

                noisy_np = (
                    noisy.squeeze(0).numpy()
                )

                gt_np = (
                    gt.squeeze(0).numpy()
                )

                # ------------------------------------------------
                # Existing KLA NoisyLR
                # Shape = 128 × 128
                # ------------------------------------------------

                degraded = noisy_np.copy()

                # ------------------------------------------------
                # Gaussian noise
                # ------------------------------------------------

                gaussian_noise = np.random.normal(
                    loc=0.0,
                    scale=gaussian_sigma,
                    size=degraded.shape
                ).astype(
                    np.float32
                )

                degraded = (
                    degraded
                    + gaussian_noise
                )

                # ------------------------------------------------
                # Speckle noise
                # Multiplicative model
                # ------------------------------------------------

                speckle_noise = np.random.normal(
                    loc=0.0,
                    scale=speckle_sigma,
                    size=degraded.shape
                ).astype(
                    np.float32
                )

                degraded = (
                    degraded
                    * (1.0 + speckle_noise)
                )

                # IMPORTANT:
                # Do NOT clip the degraded input.
                # KLA degraded images may exceed [0,1].
                degraded = np.nan_to_num(
                    degraded,
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0
                )

                # ------------------------------------------------
                # Noise-aware sigma map
                # Same procedure as training
                # ------------------------------------------------

                sigma_map = compute_noise_sigma(
                    np.clip(
                        degraded,
                        0.0,
                        1.0
                    )
                )

                # ------------------------------------------------
                # Build 2-channel input
                #
                # Channel 0 = degraded LR
                # Channel 1 = sigma map
                #
                # Shape:
                # [1, 2, 128, 128]
                # ------------------------------------------------

                input_np = np.stack(
                    [
                        degraded,
                        sigma_map
                    ],
                    axis=0
                )

                input_tensor = torch.from_numpy(
                    input_np
                ).unsqueeze(0)

                input_tensor = input_tensor.to(
                    DEVICE,
                    non_blocking=True
                )

                gt_tensor = torch.from_numpy(
                    gt_np
                ).unsqueeze(0).to(
                    DEVICE,
                    non_blocking=True
                )

                # ------------------------------------------------
                # Inference timing
                # ------------------------------------------------

                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()

                start = time.perf_counter()

                prediction = model(
                    input_tensor
                )

                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()

                end = time.perf_counter()

                inference_time = (
                    end - start
                )

                # ------------------------------------------------
                # Convert output
                # ------------------------------------------------

                prediction_np = (
                    prediction
                    .squeeze()
                    .cpu()
                    .numpy()
                )

                gt_np_eval = (
                    gt_tensor
                    .squeeze()
                    .cpu()
                    .numpy()
                )

                # ------------------------------------------------
                # Metrics
                # ------------------------------------------------

                mse, mae, psnr, ssim = (
                    calculate_metrics(
                        prediction_np,
                        gt_np_eval
                    )
                )

                mse_values.append(mse)
                mae_values.append(mae)
                psnr_values.append(psnr)
                ssim_values.append(ssim)
                inference_times.append(
                    inference_time * 1000.0
                )

                # ------------------------------------------------
                # Progress
                # ------------------------------------------------

                if (
                    (index + 1) % 50 == 0
                    or index + 1 == len(dataset)
                ):

                    print(
                        f"Processed "
                        f"{index + 1}/"
                        f"{len(dataset)}"
                    )

        # --------------------------------------------------------
        # Aggregate
        # --------------------------------------------------------

        mean_mse = np.mean(
            mse_values
        )

        mean_mae = np.mean(
            mae_values
        )

        mean_psnr = np.mean(
            psnr_values
        )

        mean_ssim = np.mean(
            ssim_values
        )

        mean_time = np.mean(
            inference_times
        )

        median_time = np.median(
            inference_times
        )

        print()

        print(
            f"Gaussian σ = "
            f"{gaussian_sigma:.3f}"
        )

        print(
            f"Speckle σ = "
            f"{speckle_sigma:.3f}"
        )

        print(
            f"Mean MSE         : "
            f"{mean_mse:.8f}"
        )

        print(
            f"Mean MAE         : "
            f"{mean_mae:.8f}"
        )

        print(
            f"Mean PSNR        : "
            f"{mean_psnr:.4f} dB"
        )

        print(
            f"Mean SSIM        : "
            f"{mean_ssim:.6f}"
        )

        print(
            f"Mean inference   : "
            f"{mean_time:.2f} ms/image"
        )

        print(
            f"Median inference : "
            f"{median_time:.2f} ms/image"
        )

        all_results.append({
            "gaussian_sigma": gaussian_sigma,
            "speckle_sigma": speckle_sigma,
            "mse": mean_mse,
            "mae": mean_mae,
            "psnr_db": mean_psnr,
            "ssim": mean_ssim,
            "mean_inference_ms": mean_time,
            "median_inference_ms": median_time
        })


# ============================================================
# SAVE CSV
# ============================================================

csv_path = os.path.join(
    RESULTS_DIR,
    "combined_degradation_results.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "gaussian_sigma",
            "speckle_sigma",
            "mse",
            "mae",
            "psnr_db",
            "ssim",
            "mean_inference_ms",
            "median_inference_ms"
        ]
    )

    writer.writeheader()

    writer.writerows(
        all_results
    )


# ============================================================
# SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULTS_DIR,
    "combined_degradation_summary.txt"
)

with open(
    summary_path,
    "w"
) as f:

    f.write(
        "KLA COMBINED DEGRADATION EVALUATION\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Device: {DEVICE}\n"
    )

    if DEVICE.type == "cuda":

        f.write(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}\n"
        )

    f.write(
        f"Test images: {len(test_pairs)}\n\n"
    )

    for result in all_results:

        f.write(
            f"Gaussian sigma: "
            f"{result['gaussian_sigma']:.3f}\n"
        )

        f.write(
            f"Speckle sigma: "
            f"{result['speckle_sigma']:.3f}\n"
        )

        f.write(
            f"MSE: "
            f"{result['mse']:.8f}\n"
        )

        f.write(
            f"MAE: "
            f"{result['mae']:.8f}\n"
        )

        f.write(
            f"PSNR: "
            f"{result['psnr_db']:.4f} dB\n"
        )

        f.write(
            f"SSIM: "
            f"{result['ssim']:.6f}\n"
        )

        f.write(
            f"Mean inference: "
            f"{result['mean_inference_ms']:.2f} ms/image\n"
        )

        f.write(
            f"Median inference: "
            f"{result['median_inference_ms']:.2f} ms/image\n"
        )

        f.write(
            "-" * 70 + "\n"
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("COMBINED DEGRADATION EVALUATION COMPLETE")
print("=" * 70)

print(
    f"CSV     : {csv_path}"
)

print(
    f"Summary : {summary_path}"
)

print("=" * 70)