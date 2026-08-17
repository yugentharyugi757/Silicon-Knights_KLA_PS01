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
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset

from models.noise_aware_dncnn import NoiseAwareDnCNN


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

GT_DIR = PROJECT_ROOT / "train" / "GT"
NOISY_DIR = PROJECT_ROOT / "train" / "NoisyLR"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_noise_aware_dncnn.pth"

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "super_resolution_evaluation"
RESTORED_DIR = OUTPUT_DIR / "restored"
COMPARISON_DIR = OUTPUT_DIR / "comparisons"
CSV_PATH = RESULTS_DIR / "super_resolution_summary.csv"
SUMMARY_PATH = RESULTS_DIR / "super_resolution_summary.txt"

NUM_FEATURES = 64
NUM_BLOCKS = 10
BATCH_SIZE = 1
NUM_WORKERS = 0

# Same empirical noise model used by the project training pipeline.
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
# DATASET
# ============================================================

class KLATestDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, gt_path = self.pairs[idx]

        degraded = np.load(noisy_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)

        degraded = np.nan_to_num(degraded, nan=0.0, posinf=1.0, neginf=0.0)
        gt = np.nan_to_num(gt, nan=0.0, posinf=1.0, neginf=0.0)

        degraded = np.clip(degraded, 0.0, 1.0)
        gt = np.clip(gt, 0.0, 1.0)

        if degraded.shape != (128, 128):
            raise ValueError(f"Expected degraded input shape (128,128), got {degraded.shape} for {noisy_path}")

        if gt.shape != (256, 256):
            raise ValueError(f"Expected GT shape (256,256), got {gt.shape} for {gt_path}")

        # For pure super-resolution evaluation there is no synthetic Gaussian or speckle corruption.
        # Use a zero-valued noise map to represent the absence of additional noise in the model's
        # expected two-channel input layout: [degraded_lr, sigma_map].
        sigma = np.zeros_like(degraded, dtype=np.float32)

        degraded_tensor = torch.from_numpy(degraded).unsqueeze(0)
        sigma_tensor = torch.from_numpy(sigma).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt).unsqueeze(0)

        input_tensor = torch.cat([degraded_tensor, sigma_tensor], dim=0)
        return input_tensor, gt_tensor, os.path.basename(noisy_path)


# ============================================================
# BUILD DATASET PAIRS
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


# ============================================================
# LOAD CHECKPOINT
# ============================================================


def load_model():
    model = NoiseAwareDnCNN(num_features=NUM_FEATURES, num_blocks=NUM_BLOCKS).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

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
# SAVE IMAGE HELPERS
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



def save_comparison_png(degraded, restored, gt, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degraded_vis = np.asarray(degraded, dtype=np.float32)
    restored_vis = np.asarray(restored, dtype=np.float32)
    gt_vis = np.asarray(gt, dtype=np.float32)

    degraded_vis = np.clip(degraded_vis, 0.0, 1.0)
    restored_vis = np.clip(restored_vis, 0.0, 1.0)
    gt_vis = np.clip(gt_vis, 0.0, 1.0)

    if degraded_vis.ndim == 3:
        degraded_vis = degraded_vis[0]
    if restored_vis.ndim == 3:
        restored_vis = restored_vis[0]
    if gt_vis.ndim == 3:
        gt_vis = gt_vis[0]

    if degraded_vis.shape != restored_vis.shape:
        degraded_vis = resize(
            degraded_vis,
            restored_vis.shape,
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(degraded_vis, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("DEGRADED INPUT")
    axes[0].axis("off")

    axes[1].imshow(restored_vis, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("RESTORED OUTPUT")
    axes[1].axis("off")

    axes[2].imshow(gt_vis, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("GROUND TRUTH")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RESTORED_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    print("=" * 70)
    print("KLA SUPER-RESOLUTION EVALUATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pairs = build_pairs()
    print(f"Total paired images : {len(pairs)}")

    random.Random(SEED).shuffle(pairs)
    total = len(pairs)
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

    model, checkpoint = load_model()

    print("Loading checkpoint...")
    print("Checkpoint loaded successfully.")

    print("\n" + "=" * 70)
    print("STARTING SUPER-RESOLUTION EVALUATION")
    print("=" * 70)

    lpips_model = lpips.LPIPS(net="alex").to(DEVICE)
    lpips_model.eval()

    results = []
    inference_times = []

    with torch.no_grad():
        for i, (inputs, targets, filenames) in enumerate(test_loader):
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            outputs = model(inputs)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            inference_ms = elapsed * 1000.0
            inference_times.append(inference_ms)

            restored_raw = outputs[0, 0].detach().cpu().numpy().copy()
            gt = targets[0, 0].detach().cpu().numpy()

            restored_for_metrics = np.clip(restored_raw, 0.0, 1.0)
            gt = np.clip(gt, 0.0, 1.0)

            mse = np.mean((restored_for_metrics - gt) ** 2)
            mae = np.mean(np.abs(restored_for_metrics - gt))
            psnr = peak_signal_noise_ratio(gt, restored_for_metrics, data_range=1.0)
            ssim = structural_similarity(gt, restored_for_metrics, data_range=1.0)

            restored_tensor = torch.from_numpy(restored_for_metrics).float()
            gt_tensor = torch.from_numpy(gt).float()

            restored_lpips = grayscale_to_lpips(restored_tensor).to(DEVICE)
            gt_lpips = grayscale_to_lpips(gt_tensor).to(DEVICE)

            lpips_score = lpips_model(restored_lpips, gt_lpips).item()

            filename = filenames[0]
            base_name = Path(filename).stem

            results.append({
                "filename": filename,
                "mse": float(mse),
                "mae": float(mae),
                "psnr": float(psnr),
                "ssim": float(ssim),
                "lpips": float(lpips_score),
                "inference_ms": float(inference_ms),
            })

            np.save(RESTORED_DIR / f"{base_name}.npy", restored_raw.astype(np.float32))
            save_grayscale_png(restored_for_metrics, RESTORED_DIR / f"{base_name}.png")

            degraded_input = inputs[0, 0].detach().cpu().numpy()
            save_comparison_png(
                degraded_input,
                restored_for_metrics,
                gt,
                COMPARISON_DIR / f"{base_name}_comparison.png",
            )

            if (i + 1) % 50 == 0 or (i + 1) == len(test_loader):
                print(f"Processed {i + 1}/{len(test_loader)}")

    mean_mse = np.mean([x["mse"] for x in results])
    mean_mae = np.mean([x["mae"] for x in results])
    mean_psnr = np.mean([x["psnr"] for x in results])
    mean_ssim = np.mean([x["ssim"] for x in results])
    mean_lpips = np.mean([x["lpips"] for x in results])
    mean_inference = np.mean(inference_times)
    median_inference = np.median(inference_times)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "mse", "mae", "psnr", "ssim", "lpips", "inference_ms"],
        )
        writer.writeheader()
        writer.writerows(results)

    summary = (
        "KLA SUPER-RESOLUTION EVALUATION\n"
        "======================================================================\n"
        f"Images evaluated : {len(results)}\n"
        f"Mean MSE         : {mean_mse:.8f}\n"
        f"Mean MAE         : {mean_mae:.8f}\n"
        f"Mean PSNR        : {mean_psnr:.4f} dB\n"
        f"Mean SSIM        : {mean_ssim:.6f}\n"
        f"Mean LPIPS       : {mean_lpips:.6f}\n"
        f"Mean inference   : {mean_inference:.2f} ms/image\n"
        f"Median inference : {median_inference:.2f} ms/image\n"
        f"Device           : {DEVICE}\n"
        f"GPU              : {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'}\n"
        f"Checkpoint path  : {CHECKPOINT_PATH}\n"
        f"Checkpoint epoch : {checkpoint.get('epoch', 'unknown')}\n"
        f"Best val loss    : {checkpoint.get('best_val_loss', 'unknown')}\n"
        "======================================================================\n"
    )

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary)

    print("\n" + "=" * 70)
    print("SUPER-RESOLUTION RESULTS")
    print("=" * 70)
    print(f"Images evaluated : {len(results)}")
    print(f"Mean MSE         : {mean_mse:.8f}")
    print(f"Mean MAE         : {mean_mae:.8f}")
    print(f"Mean PSNR        : {mean_psnr:.4f} dB")
    print(f"Mean SSIM        : {mean_ssim:.6f}")
    print(f"Mean LPIPS       : {mean_lpips:.6f}")
    print(f"Mean inference   : {mean_inference:.2f} ms/image")
    print(f"Median inference : {median_inference:.2f} ms/image")
    print("=" * 70)

    print("\n======================================================================")
    print("SUPER-RESOLUTION EVALUATION COMPLETE")
    print("======================================================================")
    print(f"CSV results      : {CSV_PATH}")
    print(f"Summary          : {SUMMARY_PATH}")
    print(f"Restored PNG/NPY : {RESTORED_DIR}")
    print(f"Comparisons      : {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
