import os
import csv
import time
import random

import numpy as np
import torch
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from models.noise_aware_dncnn import NoiseAwareDnCNN


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

GT_DIR = "train/GT"

CHECKPOINT_PATH = (
    "checkpoints/best_noise_aware_dncnn.pth"
)

OUTPUT_DIR = "outputs/speckle_evaluation"
RESULTS_DIR = "results"

NUM_FEATURES = 64
NUM_BLOCKS = 10

# Test only the same 10% test split used previously
TEST_RATIO = 0.10

# Synthetic speckle stress levels.
# These are NOT official KLA parameters.
SPECKLE_LEVELS = [
    0.05,
    0.10,
    0.20,
]

SEED = 42

# Empirical signal-dependent noise model
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

print("=" * 70)
print("KLA SPECKLE ROBUSTNESS EVALUATION")
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

    sigma = np.sqrt(variance)

    return sigma.astype(np.float32)


# ============================================================
# SPECKLE GENERATION
# ============================================================

def add_speckle(image, strength):

    """
    Synthetic multiplicative speckle stress test.

    IMPORTANT:
    This is an evaluation proxy only.
    The KLA problem statement does not define
    an exact mathematical speckle distribution.
    """

    noise = np.random.normal(
        loc=0.0,
        scale=strength,
        size=image.shape
    ).astype(np.float32)

    degraded = image * (
        1.0 + noise
    )

    # Intentionally DO NOT clip here.
    #
    # The challenge states that degraded
    # intensity values may exceed the GT range.

    return degraded.astype(np.float32)


# ============================================================
# BUILD TEST SET
# ============================================================

def build_test_files():

    files = sorted([
        f
        for f in os.listdir(GT_DIR)
        if f.endswith(".npy")
    ])

    total = len(files)

    test_count = int(
        total * TEST_RATIO
    )

    # Same deterministic ordering
    # used throughout our evaluation.
    test_files = files[
        total - test_count:
    ]

    return test_files


# ============================================================
# LOAD MODEL
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

    model.eval()

    print(
        "Checkpoint loaded successfully."
    )

    print(
        f"Checkpoint epoch : "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Best val loss    : "
        f"{checkpoint.get('best_val_loss', 'unknown')}"
    )

    return model


# ============================================================
# SAVE IMAGE
# ============================================================

def save_npy(path, image):

    np.save(
        path,
        image.astype(np.float32)
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_level(
    model,
    test_files,
    speckle_strength
):

    level_name = (
        f"sigma_{speckle_strength:.3f}"
        .replace(".", "_")
    )

    restored_dir = os.path.join(
        OUTPUT_DIR,
        level_name,
        "restored"
    )

    degraded_dir = os.path.join(
        OUTPUT_DIR,
        level_name,
        "degraded"
    )

    os.makedirs(
        restored_dir,
        exist_ok=True
    )

    os.makedirs(
        degraded_dir,
        exist_ok=True
    )

    mse_values = []
    mae_values = []
    psnr_values = []
    ssim_values = []
    inference_times = []

    print()
    print("=" * 70)

    print(
        f"SPECKLE STRENGTH = "
        f"{speckle_strength:.3f}"
    )

    print("=" * 70)

    for index, filename in enumerate(
        test_files,
        start=1
    ):

        gt_path = os.path.join(
            GT_DIR,
            filename
        )

        gt = np.load(
            gt_path
        ).astype(np.float32)

        # ----------------------------------------------------
        # Safety handling
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Generate synthetic speckle
        # ----------------------------------------------------

        speckled = add_speckle(
            gt,
            speckle_strength
        )

        # ----------------------------------------------------
        # Downsample to LR
        # ----------------------------------------------------

        speckled_tensor = torch.from_numpy(
            speckled
        ).unsqueeze(0).unsqueeze(0)

        noisy_lr_tensor = F.interpolate(
            speckled_tensor,
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        )

        noisy_lr = (
            noisy_lr_tensor
            .squeeze()
            .numpy()
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # Sigma map
        #
        # Same inference procedure as our trained model.
        # ----------------------------------------------------

        noisy_for_sigma = np.clip(
            noisy_lr,
            0.0,
            1.0
        )

        sigma_map = compute_noise_sigma(
            noisy_for_sigma
        )

        # ----------------------------------------------------
        # Build two-channel model input
        #
        # [NoisyLR, Sigma]
        # ----------------------------------------------------

        noisy_tensor = torch.from_numpy(
            noisy_lr
        ).unsqueeze(0)

        sigma_tensor = torch.from_numpy(
            sigma_map
        ).unsqueeze(0)

        input_tensor = torch.cat(
            [
                noisy_tensor,
                sigma_tensor
            ],
            dim=0
        )

        input_tensor = (
            input_tensor
            .unsqueeze(0)
            .to(DEVICE)
        )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            output = model(
                input_tensor
            )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        inference_times.append(
            elapsed
        )

        # ----------------------------------------------------
        # Convert output
        # ----------------------------------------------------

        restored = (
            output
            .squeeze()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        restored = np.clip(
            restored,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        error = restored - gt

        mse = float(
            np.mean(error ** 2)
        )

        mae = float(
            np.mean(np.abs(error))
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

        mse_values.append(mse)
        mae_values.append(mae)
        psnr_values.append(psnr)
        ssim_values.append(ssim)

        # ----------------------------------------------------
        # Save outputs
        # ----------------------------------------------------

        save_npy(
            os.path.join(
                degraded_dir,
                filename
            ),
            noisy_lr
        )

        save_npy(
            os.path.join(
                restored_dir,
                filename
            ),
            restored
        )

        if (
            index % 50 == 0
            or index == len(test_files)
        ):

            print(
                f"Processed "
                f"{index}/{len(test_files)}"
            )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    result = {
        "speckle_strength":
            speckle_strength,

        "images":
            len(test_files),

        "mse":
            float(np.mean(mse_values)),

        "mae":
            float(np.mean(mae_values)),

        "psnr":
            float(np.mean(psnr_values)),

        "ssim":
            float(np.mean(ssim_values)),

        "mean_inference_ms":
            float(np.mean(inference_times)),

        "median_inference_ms":
            float(np.median(inference_times)),
    }

    print()
    print(
        f"Speckle σ = "
        f"{speckle_strength:.3f}"
    )

    print(
        f"Mean MSE         : "
        f"{result['mse']:.8f}"
    )

    print(
        f"Mean MAE         : "
        f"{result['mae']:.8f}"
    )

    print(
        f"Mean PSNR        : "
        f"{result['psnr']:.4f} dB"
    )

    print(
        f"Mean SSIM        : "
        f"{result['ssim']:.6f}"
    )

    print(
        f"Mean inference   : "
        f"{result['mean_inference_ms']:.2f} ms/image"
    )

    print(
        f"Median inference : "
        f"{result['median_inference_ms']:.2f} ms/image"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    test_files = build_test_files()

    print(
        f"Total GT images : "
        f"{len(os.listdir(GT_DIR))}"
    )

    print(
        f"Testing         : "
        f"{len(test_files)}"
    )

    model = load_model()

    all_results = []

    for strength in SPECKLE_LEVELS:

        result = evaluate_level(
            model,
            test_files,
            strength
        )

        all_results.append(
            result
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        RESULTS_DIR,
        "speckle_summary.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "speckle_strength",
                "images",
                "mse",
                "mae",
                "psnr",
                "ssim",
                "mean_inference_ms",
                "median_inference_ms",
            ]
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SPECKLE EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Summary : {csv_path}"
    )

    print(
        f"Outputs : {OUTPUT_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()