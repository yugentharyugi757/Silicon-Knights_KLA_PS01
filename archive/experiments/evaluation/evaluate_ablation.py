import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import csv
import os
import random
import time

import lpips
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset

from models.noise_aware_dncnn import NoiseAwareDnCNN


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

GT_DIR = PROJECT_ROOT / "train" / "GT"
NOISY_DIR = PROJECT_ROOT / "train" / "NoisyLR"
BASELINE_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_dncnn.pth"
NOISE_AWARE_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_noise_aware_dncnn.pth"

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ablation"
BASELINE_DIR = OUTPUT_DIR / "baseline_dncnn"
NOISE_AWARE_DIR = OUTPUT_DIR / "noise_aware_dncnn"
COMPARISON_DIR = OUTPUT_DIR / "comparisons"

CSV_PATH = RESULTS_DIR / "ablation_results.csv"
SUMMARY_PATH = RESULTS_DIR / "ablation_summary.txt"

NUM_FEATURES = 64
NUM_BLOCKS = 10
BATCH_SIZE = 1
NUM_WORKERS = 0

# Same model obtained from noise analysis
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# NOISE SIGMA MAP
# ============================================================

def compute_noise_sigma(image):
    variance = VAR_A * image ** 2 + VAR_B * image + VAR_C
    variance = np.maximum(variance, 0.0)
    sigma = np.sqrt(variance)
    return sigma.astype(np.float32)


# ============================================================
# RESIDUAL BLOCK (shared by both models)
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

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

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
# BASELINE DnCNN RESTORER
# ============================================================

class DnCNNRestorer(nn.Module):

    def __init__(
        self,
        num_features=64,
        num_blocks=10
    ):

        super().__init__()

        # Initial feature extraction
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

        # Residual feature extraction
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

        # Feature reconstruction
        self.reconstruction = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            padding=1
        )

        # PixelShuffle ×2
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

        # Output layer
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
# DATASET
# ============================================================

class KLATestDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, gt_path = self.pairs[idx]

        noisy = np.load(noisy_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)

        noisy = np.nan_to_num(noisy, nan=0.0, posinf=1.0, neginf=0.0)
        gt = np.nan_to_num(gt, nan=0.0, posinf=1.0, neginf=0.0)

        gt = np.clip(gt, 0.0, 1.0)

        noisy_for_sigma = np.clip(noisy, 0.0, 1.0)
        sigma = compute_noise_sigma(noisy_for_sigma)

        noisy_tensor = torch.from_numpy(noisy).unsqueeze(0)
        sigma_tensor = torch.from_numpy(sigma).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt).unsqueeze(0)

        return noisy_tensor, sigma_tensor, gt_tensor, os.path.basename(noisy_path)


# ============================================================
# BUILD DATASET PAIRS
# ============================================================

def build_pairs():
    gt_files = sorted([
        f for f in os.listdir(GT_DIR) if f.endswith(".npy")
    ])

    noisy_files = sorted([
        f for f in os.listdir(NOISY_DIR) if f.endswith(".npy")
    ])

    noisy_set = set(noisy_files)
    pairs = []

    for filename in gt_files:
        if filename in noisy_set:
            noisy_path = os.path.join(NOISY_DIR, filename)
            gt_path = os.path.join(GT_DIR, filename)
            pairs.append((noisy_path, gt_path))

    return pairs


# ============================================================
# LOAD CHECKPOINTS
# ============================================================

def load_baseline_model():
    model = DnCNNRestorer(
        num_features=NUM_FEATURES,
        num_blocks=NUM_BLOCKS,
    ).to(DEVICE)

    checkpoint = torch.load(
        BASELINE_CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint


def load_noise_aware_model():
    model = NoiseAwareDnCNN(
        num_features=NUM_FEATURES,
        num_blocks=NUM_BLOCKS,
    ).to(DEVICE)

    checkpoint = torch.load(
        NOISE_AWARE_CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint


# ============================================================
# LPIPS HELPERS
# ============================================================

def grayscale_to_lpips(image):
    """
    Input should be either [H, W] or [1, H, W].
    LPIPS expects [N, 3, H, W] with the grayscale content replicated.
    """
    if image.dim() == 2:
        image = image.unsqueeze(0)

    if image.dim() == 3 and image.shape[0] == 1:
        image = image.unsqueeze(0)
        image = image.repeat(1, 3, 1, 1)
    elif image.dim() == 3 and image.shape[0] != 1:
        image = image.unsqueeze(0)
        image = image.repeat(1, 3, 1, 1)
    elif image.dim() == 4:
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
    else:
        raise ValueError(f"Unexpected image shape for LPIPS conversion: {image.shape}")

    image = image.float()
    image = image * 2.0 - 1.0
    return image


# ============================================================
# INFERENCE TIMING
# ============================================================

def measure_pure_model_inference_time(model, inputs, device):
    """Measure only the model forward-pass time for a single batch."""
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    restored = model(inputs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return restored, elapsed


# ============================================================
# SAVE UTILITIES
# ============================================================

def save_grayscale_png(image_array, output_path):
    image = np.clip(image_array, 0.0, 1.0)
    image = (image * 255.0).astype(np.uint8)
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[:, :, 0]
    if image.ndim == 2:
        image_pil = Image.fromarray(image, mode="L")
        image_pil.save(output_path)
    else:
        raise ValueError(f"Unsupported grayscale image shape for PNG save: {image.shape}")


def save_4panel_comparison_png(degraded, baseline_restored, noise_aware_restored, gt, output_path):
    """Save a 4-panel comparison: degraded, baseline, noise-aware, ground truth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degraded_vis = np.asarray(degraded, dtype=np.float32)
    baseline_vis = np.asarray(baseline_restored, dtype=np.float32)
    noise_aware_vis = np.asarray(noise_aware_restored, dtype=np.float32)
    gt_vis = np.asarray(gt, dtype=np.float32)

    degraded_vis = np.clip(degraded_vis, 0.0, 1.0)
    baseline_vis = np.clip(baseline_vis, 0.0, 1.0)
    noise_aware_vis = np.clip(noise_aware_vis, 0.0, 1.0)
    gt_vis = np.clip(gt_vis, 0.0, 1.0)

    if degraded_vis.ndim == 3:
        degraded_vis = degraded_vis[0]
    if baseline_vis.ndim == 3:
        baseline_vis = baseline_vis[0]
    if noise_aware_vis.ndim == 3:
        noise_aware_vis = noise_aware_vis[0]
    if gt_vis.ndim == 3:
        gt_vis = gt_vis[0]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(degraded_vis, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Degraded Input")
    axes[0].axis("off")

    axes[1].imshow(baseline_vis, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Baseline DnCNN")
    axes[1].axis("off")

    axes[2].imshow(noise_aware_vis, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Noise-Aware DnCNN")
    axes[2].axis("off")

    axes[3].imshow(gt_vis, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Ground Truth")
    axes[3].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(BASELINE_DIR, exist_ok=True)
    os.makedirs(NOISE_AWARE_DIR, exist_ok=True)
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    print("=" * 70)
    print("KLA ABLATION STUDY")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Use the exact same split logic as evaluate_final.py
    pairs = build_pairs()
    print(f"Total paired images : {len(pairs)}")

    random.Random(SEED).shuffle(pairs)
    total = len(pairs)
    train_end = int(0.80 * total)
    val_end = int(0.90 * total)
    test_pairs = pairs[val_end:]

    print(f"Testing             : {len(test_pairs)}")

    test_dataset = KLATestDataset(test_pairs)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # Load both models
    print("\nLoading baseline DnCNN...")
    baseline_model, baseline_checkpoint = load_baseline_model()
    print(f"Baseline checkpoint: {str(BASELINE_CHECKPOINT_PATH)}")

    print("Loading Noise-Aware DnCNN...")
    noise_aware_model, noise_aware_checkpoint = load_noise_aware_model()
    print(f"Noise-Aware checkpoint: {str(NOISE_AWARE_CHECKPOINT_PATH)}")

    print(f"Features            : {NUM_FEATURES}")
    print(f"Residual blocks     : {NUM_BLOCKS}")

    print("\nLoading LPIPS model...")
    lpips_model = lpips.LPIPS(net="alex").to(DEVICE)
    lpips_model.eval()
    print("LPIPS network: AlexNet")

    results = []
    baseline_inference_times = []
    noise_aware_inference_times = []

    print("\n" + "=" * 70)
    print("STARTING ABLATION STUDY EVALUATION")
    print("=" * 70)

    with torch.no_grad():
        # Warm-up pass for both models
        warmup_noisy, warmup_sigma, _, _ = next(iter(test_loader))
        warmup_noisy = warmup_noisy.to(DEVICE, non_blocking=True)
        warmup_sigma = warmup_sigma.to(DEVICE, non_blocking=True)

        _, _ = measure_pure_model_inference_time(baseline_model, warmup_noisy, DEVICE)
        warmup_noise_aware_input = torch.cat([warmup_noisy, warmup_sigma], dim=1)
        _, _ = measure_pure_model_inference_time(noise_aware_model, warmup_noise_aware_input, DEVICE)

        # Main evaluation loop
        for i, (noisy, sigma, targets, filenames) in enumerate(test_loader):
            noisy = noisy.to(DEVICE, non_blocking=True)
            sigma = sigma.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            # Baseline DnCNN inference (only noisy input)
            baseline_outputs, baseline_elapsed = measure_pure_model_inference_time(
                baseline_model, noisy, DEVICE
            )
            baseline_inference_ms = baseline_elapsed * 1000.0

            # Noise-Aware DnCNN inference (noisy + sigma input)
            noise_aware_input = torch.cat([noisy, sigma], dim=1)
            noise_aware_outputs, noise_aware_elapsed = measure_pure_model_inference_time(
                noise_aware_model, noise_aware_input, DEVICE
            )
            noise_aware_inference_ms = noise_aware_elapsed * 1000.0

            baseline_inference_times.append(baseline_inference_ms)
            noise_aware_inference_times.append(noise_aware_inference_ms)

            # Extract arrays
            baseline_raw = baseline_outputs[0, 0].detach().cpu().numpy().copy()
            noise_aware_raw = noise_aware_outputs[0, 0].detach().cpu().numpy().copy()
            gt = targets[0, 0].detach().cpu().numpy()
            degraded_input = noisy[0, 0].detach().cpu().numpy()

            # Clip for metrics
            baseline_for_metrics = np.clip(baseline_raw, 0.0, 1.0)
            noise_aware_for_metrics = np.clip(noise_aware_raw, 0.0, 1.0)
            gt = np.clip(gt, 0.0, 1.0)

            # Calculate metrics for baseline
            baseline_mse = np.mean((baseline_for_metrics - gt) ** 2)
            baseline_mae = np.mean(np.abs(baseline_for_metrics - gt))
            baseline_psnr = peak_signal_noise_ratio(gt, baseline_for_metrics, data_range=1.0)
            baseline_ssim = structural_similarity(gt, baseline_for_metrics, data_range=1.0)

            baseline_tensor = torch.from_numpy(baseline_for_metrics).float()
            gt_tensor = torch.from_numpy(gt).float()
            baseline_lpips_input = grayscale_to_lpips(baseline_tensor).to(DEVICE)
            gt_lpips_input = grayscale_to_lpips(gt_tensor).to(DEVICE)
            baseline_lpips = lpips_model(baseline_lpips_input, gt_lpips_input).item()

            # Calculate metrics for noise-aware
            noise_aware_mse = np.mean((noise_aware_for_metrics - gt) ** 2)
            noise_aware_mae = np.mean(np.abs(noise_aware_for_metrics - gt))
            noise_aware_psnr = peak_signal_noise_ratio(gt, noise_aware_for_metrics, data_range=1.0)
            noise_aware_ssim = structural_similarity(gt, noise_aware_for_metrics, data_range=1.0)

            noise_aware_tensor = torch.from_numpy(noise_aware_for_metrics).float()
            noise_aware_lpips_input = grayscale_to_lpips(noise_aware_tensor).to(DEVICE)
            noise_aware_lpips = lpips_model(noise_aware_lpips_input, gt_lpips_input).item()

            filename = filenames[0]
            base_name = Path(filename).stem

            results.append({
                "filename": filename,
                "baseline_mse": float(baseline_mse),
                "baseline_mae": float(baseline_mae),
                "baseline_psnr": float(baseline_psnr),
                "baseline_ssim": float(baseline_ssim),
                "baseline_lpips": float(baseline_lpips),
                "baseline_inference_ms": float(baseline_inference_ms),
                "noise_aware_mse": float(noise_aware_mse),
                "noise_aware_mae": float(noise_aware_mae),
                "noise_aware_psnr": float(noise_aware_psnr),
                "noise_aware_ssim": float(noise_aware_ssim),
                "noise_aware_lpips": float(noise_aware_lpips),
                "noise_aware_inference_ms": float(noise_aware_inference_ms),
            })

            # Save restored outputs
            np.save(BASELINE_DIR / f"{base_name}.npy", baseline_raw.astype(np.float32))
            save_grayscale_png(baseline_for_metrics, BASELINE_DIR / f"{base_name}.png")

            np.save(NOISE_AWARE_DIR / f"{base_name}.npy", noise_aware_raw.astype(np.float32))
            save_grayscale_png(noise_aware_for_metrics, NOISE_AWARE_DIR / f"{base_name}.png")

            # Save 4-panel comparison
            save_4panel_comparison_png(
                degraded_input,
                baseline_for_metrics,
                noise_aware_for_metrics,
                gt,
                COMPARISON_DIR / f"{base_name}_comparison.png",
            )

            if (i + 1) % 50 == 0 or (i + 1) == len(test_loader):
                print(f"Processed {i + 1}/{len(test_loader)}")

    # Calculate aggregated metrics
    baseline_mean_mse = np.mean([x["baseline_mse"] for x in results])
    baseline_mean_mae = np.mean([x["baseline_mae"] for x in results])
    baseline_mean_psnr = np.mean([x["baseline_psnr"] for x in results])
    baseline_mean_ssim = np.mean([x["baseline_ssim"] for x in results])
    baseline_mean_lpips = np.mean([x["baseline_lpips"] for x in results])
    baseline_mean_inference = np.mean(baseline_inference_times)
    baseline_median_inference = np.median(baseline_inference_times)

    noise_aware_mean_mse = np.mean([x["noise_aware_mse"] for x in results])
    noise_aware_mean_mae = np.mean([x["noise_aware_mae"] for x in results])
    noise_aware_mean_psnr = np.mean([x["noise_aware_psnr"] for x in results])
    noise_aware_mean_ssim = np.mean([x["noise_aware_ssim"] for x in results])
    noise_aware_mean_lpips = np.mean([x["noise_aware_lpips"] for x in results])
    noise_aware_mean_inference = np.mean(noise_aware_inference_times)
    noise_aware_median_inference = np.median(noise_aware_inference_times)

    # Calculate improvements
    psnr_improvement = noise_aware_mean_psnr - baseline_mean_psnr
    ssim_improvement = noise_aware_mean_ssim - baseline_mean_ssim
    lpips_improvement = baseline_mean_lpips - noise_aware_mean_lpips  # Lower is better
    mse_improvement = baseline_mean_mse - noise_aware_mean_mse  # Lower is better
    inference_time_diff = baseline_mean_inference - noise_aware_mean_inference

    # Save CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "baseline_mse",
                "baseline_mae",
                "baseline_psnr",
                "baseline_ssim",
                "baseline_lpips",
                "baseline_inference_ms",
                "noise_aware_mse",
                "noise_aware_mae",
                "noise_aware_psnr",
                "noise_aware_ssim",
                "noise_aware_lpips",
                "noise_aware_inference_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    # Save text summary
    summary = (
        "KLA ABLATION STUDY\n"
        "======================================================================\n"
        f"Images evaluated : {len(results)}\n"
        f"Device           : {DEVICE}\n"
        f"GPU              : {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'}\n"
        "\n"
        "BASELINE DnCNN\n"
        "======================================================================\n"
        f"Checkpoint path  : {str(BASELINE_CHECKPOINT_PATH)}\n"
        f"Mean MSE         : {baseline_mean_mse:.8f}\n"
        f"Mean MAE         : {baseline_mean_mae:.8f}\n"
        f"Mean PSNR        : {baseline_mean_psnr:.4f} dB\n"
        f"Mean SSIM        : {baseline_mean_ssim:.6f}\n"
        f"Mean LPIPS       : {baseline_mean_lpips:.6f}\n"
        f"Pure model inference time (mean) : {baseline_mean_inference:.2f} ms/image\n"
        f"Pure model inference time (median) : {baseline_median_inference:.2f} ms/image\n"
        "\n"
        "NOISE-AWARE DnCNN\n"
        "======================================================================\n"
        f"Checkpoint path  : {str(NOISE_AWARE_CHECKPOINT_PATH)}\n"
        f"Checkpoint epoch : {noise_aware_checkpoint.get('epoch', 'unknown')}\n"
        f"Best val loss    : {noise_aware_checkpoint.get('best_val_loss', 'unknown')}\n"
        f"Mean MSE         : {noise_aware_mean_mse:.8f}\n"
        f"Mean MAE         : {noise_aware_mean_mae:.8f}\n"
        f"Mean PSNR        : {noise_aware_mean_psnr:.4f} dB\n"
        f"Mean SSIM        : {noise_aware_mean_ssim:.6f}\n"
        f"Mean LPIPS       : {noise_aware_mean_lpips:.6f}\n"
        f"Pure model inference time (mean) : {noise_aware_mean_inference:.2f} ms/image\n"
        f"Pure model inference time (median) : {noise_aware_median_inference:.2f} ms/image\n"
        "\n"
        "IMPROVEMENT (Noise-Aware vs Baseline)\n"
        "======================================================================\n"
        f"PSNR difference  : {psnr_improvement:+.4f} dB\n"
        f"SSIM difference  : {ssim_improvement:+.6f}\n"
        f"LPIPS difference : {lpips_improvement:+.6f} (lower is better)\n"
        f"MSE difference   : {mse_improvement:+.8f} (lower is better)\n"
        f"Inference time (ms/image) : {inference_time_diff:+.2f}\n"
        "======================================================================\n"
    )

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary)

    # Print results
    print("\n" + "=" * 70)
    print("KLA ABLATION STUDY")
    print("=" * 70)

    print("\nBASELINE DnCNN")
    print(f"Checkpoint path  : {str(BASELINE_CHECKPOINT_PATH)}")
    print(f"Mean MSE         : {baseline_mean_mse:.8f}")
    print(f"Mean MAE         : {baseline_mean_mae:.8f}")
    print(f"Mean PSNR        : {baseline_mean_psnr:.4f} dB")
    print(f"Mean SSIM        : {baseline_mean_ssim:.6f}")
    print(f"Mean LPIPS       : {baseline_mean_lpips:.6f}")
    print(f"Pure model inference time (mean) : {baseline_mean_inference:.2f} ms/image")
    print(f"Pure model inference time (median) : {baseline_median_inference:.2f} ms/image")

    print("\nNOISE-AWARE DnCNN")
    print(f"Checkpoint path  : {str(NOISE_AWARE_CHECKPOINT_PATH)}")
    print(f"Checkpoint epoch : {noise_aware_checkpoint.get('epoch', 'unknown')}")
    print(f"Best val loss    : {noise_aware_checkpoint.get('best_val_loss', 'unknown')}")
    print(f"Mean MSE         : {noise_aware_mean_mse:.8f}")
    print(f"Mean MAE         : {noise_aware_mean_mae:.8f}")
    print(f"Mean PSNR        : {noise_aware_mean_psnr:.4f} dB")
    print(f"Mean SSIM        : {noise_aware_mean_ssim:.6f}")
    print(f"Mean LPIPS       : {noise_aware_mean_lpips:.6f}")
    print(f"Pure model inference time (mean) : {noise_aware_mean_inference:.2f} ms/image")
    print(f"Pure model inference time (median) : {noise_aware_median_inference:.2f} ms/image")

    print("\nIMPROVEMENT (Noise-Aware vs Baseline)")
    print("=" * 70)
    print(f"PSNR difference  : {psnr_improvement:+.4f} dB")
    print(f"SSIM difference  : {ssim_improvement:+.6f}")
    print(f"LPIPS difference : {lpips_improvement:+.6f} (lower is better)")
    print(f"MSE difference   : {mse_improvement:+.8f} (lower is better)")
    print(f"Inference time (ms/image) : {inference_time_diff:+.2f}")

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    print(f"CSV results      : {CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Summary          : {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Baseline outputs : {BASELINE_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Noise-Aware outputs : {NOISE_AWARE_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Comparisons      : {COMPARISON_DIR.relative_to(PROJECT_ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
