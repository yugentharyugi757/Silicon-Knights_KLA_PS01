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

CHECKPOINT_PATH = "checkpoints/best_noise_aware_dncnn.pth"

OUTPUT_ROOT = "outputs/gaussian_evaluation"
RESULTS_DIR = "results"

NUM_FEATURES = 64
NUM_BLOCKS = 10

# Gaussian noise standard deviations to test
GAUSSIAN_SIGMAS = [0.01, 0.03, 0.05]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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
# DATASET PAIRS
# ============================================================

def build_test_pairs():

    gt_files = sorted([
        f
        for f in os.listdir(GT_DIR)
        if f.endswith(".npy")
    ])

    pairs = [
        os.path.join(GT_DIR, f)
        for f in gt_files
    ]

    # Same split strategy as the existing evaluation
    random.Random(SEED).shuffle(pairs)

    total = len(pairs)

    val_end = int(0.90 * total)

    test_pairs = pairs[val_end:]

    return test_pairs


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

    print("Checkpoint loaded successfully.")

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
# LOAD GT
# ============================================================

def load_gt(path):

    gt = np.load(path).astype(np.float32)

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

    return gt


# ============================================================
# CREATE 128x128 LOW-RES IMAGE
# ============================================================

def downsample_gt(gt):

    tensor = torch.from_numpy(gt).float()

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    lr = F.interpolate(
        tensor,
        size=(128, 128),
        mode="bicubic",
        align_corners=False
    )

    lr = lr.squeeze().numpy()

    return lr.astype(np.float32)


# ============================================================
# ADD GAUSSIAN NOISE
# ============================================================

def add_gaussian_noise(image, sigma):

    noise = np.random.normal(
        loc=0.0,
        scale=sigma,
        size=image.shape
    ).astype(np.float32)

    noisy = image + noise

    # IMPORTANT:
    # Keep raw Gaussian noise before clipping for analysis,
    # but model input must be numerically safe.
    noisy = np.clip(
        noisy,
        0.0,
        1.0
    )

    return noisy.astype(np.float32)


# ============================================================
# CREATE SIGMA MAP
# ============================================================

def create_sigma_map(
    height,
    width,
    sigma
):

    sigma_map = np.full(
        (height, width),
        sigma,
        dtype=np.float32
    )

    return sigma_map


# ============================================================
# SAVE COMPARISON
# ============================================================

def save_comparison(
    gt,
    noisy,
    restored,
    path
):

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(
        gt,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[0].set_title("Ground Truth")

    axes[1].imshow(
        noisy,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[1].set_title("Gaussian + LR")

    axes[2].imshow(
        restored,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[2].set_title("Noise-Aware DnCNN")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# EVALUATE ONE SIGMA
# ============================================================

def evaluate_sigma(
    model,
    test_paths,
    sigma
):

    sigma_name = f"sigma_{sigma:.2f}".replace(
        ".",
        ""
    )

    output_dir = os.path.join(
        OUTPUT_ROOT,
        sigma_name
    )

    restored_dir = os.path.join(
        output_dir,
        "restored"
    )

    comparison_dir = os.path.join(
        output_dir,
        "comparisons"
    )

    os.makedirs(
        restored_dir,
        exist_ok=True
    )

    os.makedirs(
        comparison_dir,
        exist_ok=True
    )

    results = []

    inference_times = []

    print()
    print("=" * 70)
    print(
        f"GAUSSIAN SIGMA = {sigma:.3f}"
    )
    print("=" * 70)

    with torch.no_grad():

        for i, gt_path in enumerate(test_paths):

            filename = os.path.basename(
                gt_path
            )

            # ------------------------------------------------
            # GT
            # ------------------------------------------------

            gt = load_gt(
                gt_path
            )

            # ------------------------------------------------
            # 256 -> 128
            # ------------------------------------------------

            lr = downsample_gt(
                gt
            )

            # ------------------------------------------------
            # Add Gaussian noise
            # ------------------------------------------------

            noisy = add_gaussian_noise(
                lr,
                sigma
            )

            # ------------------------------------------------
            # Sigma channel
            # ------------------------------------------------

            sigma_map = create_sigma_map(
                128,
                128,
                sigma
            )

            # ------------------------------------------------
            # Model input
            #
            # Channel 0 = noisy LR
            # Channel 1 = sigma map
            # ------------------------------------------------

            noisy_tensor = torch.from_numpy(
                noisy
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

            input_tensor = input_tensor.unsqueeze(
                0
            ).to(
                DEVICE,
                non_blocking=True
            )

            # ------------------------------------------------
            # Inference timing
            # ------------------------------------------------

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            output = model(
                input_tensor
            )

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter()
                - start
            )

            inference_ms = (
                elapsed * 1000
            )

            inference_times.append(
                inference_ms
            )

            # ------------------------------------------------
            # Convert output
            # ------------------------------------------------

            restored = (
                output
                .cpu()
                .numpy()[0, 0]
            )

            restored = np.clip(
                restored,
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
                np.abs(
                    restored - gt
                )
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

            results.append({
                "filename": filename,
                "sigma": sigma,
                "mse": float(mse),
                "mae": float(mae),
                "psnr_db": float(psnr),
                "ssim": float(ssim),
                "inference_ms": float(
                    inference_ms
                )
            })

            # ------------------------------------------------
            # Save restored image
            # ------------------------------------------------

            np.save(
                os.path.join(
                    restored_dir,
                    filename
                ),
                restored.astype(
                    np.float32
                )
            )

            # ------------------------------------------------
            # Save selected visual comparisons
            # ------------------------------------------------

            if i < 10:

                comparison_path = os.path.join(
                    comparison_dir,
                    filename.replace(
                        ".npy",
                        ".png"
                    )
                )

                save_comparison(
                    gt,
                    noisy,
                    restored,
                    comparison_path
                )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                (i + 1) % 50 == 0
                or
                (i + 1) == len(test_paths)
            ):

                print(
                    f"Processed "
                    f"{i + 1}/"
                    f"{len(test_paths)}"
                )

    # ========================================================
    # AGGREGATE
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

    csv_path = os.path.join(
        RESULTS_DIR,
        f"gaussian_sigma_{sigma:.2f}.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "sigma",
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
    # PRINT RESULTS
    # ========================================================

    print()
    print(
        f"Gaussian σ = {sigma:.3f}"
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

    return {
        "sigma": sigma,
        "mse": mean_mse,
        "mae": mean_mae,
        "psnr": mean_psnr,
        "ssim": mean_ssim,
        "mean_time": mean_time,
        "median_time": median_time
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "KLA GAUSSIAN DEGRADATION EVALUATION"
    )
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Test split
    # --------------------------------------------------------

    test_paths = build_test_pairs()

    print(
        f"Total GT images : "
        f"{len(test_paths)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Run all Gaussian levels
    # --------------------------------------------------------

    all_results = []

    for sigma in GAUSSIAN_SIGMAS:

        result = evaluate_sigma(
            model,
            test_paths,
            sigma
        )

        all_results.append(
            result
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    summary_path = os.path.join(
        RESULTS_DIR,
        "gaussian_summary.csv"
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sigma",
                "mse",
                "mae",
                "psnr",
                "ssim",
                "mean_time",
                "median_time"
            ]
        )

        writer.writeheader()

        for result in all_results:

            writer.writerow({
                "sigma": result["sigma"],
                "mse": result["mse"],
                "mae": result["mae"],
                "psnr": result["psnr"],
                "ssim": result["ssim"],
                "mean_time": result["mean_time"],
                "median_time": result["median_time"]
            })

    print()
    print("=" * 70)
    print(
        "GAUSSIAN EVALUATION COMPLETE"
    )
    print("=" * 70)

    print(
        f"Summary : {summary_path}"
    )

    print(
        f"Outputs : {OUTPUT_ROOT}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()