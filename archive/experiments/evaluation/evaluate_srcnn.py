import csv
import os
import random
import sys
import time
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.super_resolution.srcnn import SRCNN


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
SCALE_FACTOR = 2
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_srcnn_x2.pth"

GT_DIR = PROJECT_ROOT / "train" / "GT"
NOISY_DIR = PROJECT_ROOT / "train" / "NoisyLR"

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "srcnn_evaluation"
RESTORED_DIR = OUTPUT_DIR / "restored"
COMPARISON_DIR = OUTPUT_DIR / "comparisons"

BICUBIC_CSV_PATH = RESULTS_DIR / "bicubic_x2_summary.csv"
BICUBIC_SUMMARY_PATH = RESULTS_DIR / "bicubic_x2_summary.txt"
SRCNN_CSV_PATH = RESULTS_DIR / "srcnn_x2_summary.csv"
SRCNN_SUMMARY_PATH = RESULTS_DIR / "srcnn_x2_summary.txt"

BATCH_SIZE = 1
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        if gt.shape != (256, 256):
            raise ValueError(f"Expected GT shape (256,256), got {gt.shape} for {gt_path}")

        hr = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)
        lr_size = (max(1, 256 // self.scale_factor), max(1, 256 // self.scale_factor))

        lr = F.interpolate(
            hr,
            size=lr_size,
            mode="bicubic",
            align_corners=False,
        )

        upscaled_lr = F.interpolate(
            lr,
            size=(256, 256),
            mode="bicubic",
            align_corners=False,
        )

        input_tensor = upscaled_lr.squeeze(0)
        target_tensor = hr.squeeze(0)
        return input_tensor, target_tensor, os.path.basename(gt_path)


# ============================================================
# MODEL / CHECKPOINT
# ============================================================


def load_model():
    model = SRCNN(
        input_channels=1,
        output_channels=1,
        first_features=64,
        first_kernel_size=9,
        second_features=32,
        second_kernel_size=1,
        third_kernel_size=5,
    ).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


# ============================================================
# METRICS
# ============================================================


def grayscale_to_lpips(image):
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
    image = image.float() * 2.0 - 1.0
    return image


# ============================================================
# SAVE OUTPUTS
# ============================================================


def save_grayscale_png(image_array, output_path):
    image = np.clip(np.asarray(image_array, dtype=np.float32), 0.0, 1.0)
    image = (image * 255.0).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[:, :, 0]
    Image.fromarray(image, mode="L").save(output_path)



def save_comparison_png(bicubic, srcnn, gt, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bicubic_vis = np.clip(np.asarray(bicubic, dtype=np.float32), 0.0, 1.0)
    srcnn_vis = np.clip(np.asarray(srcnn, dtype=np.float32), 0.0, 1.0)
    gt_vis = np.clip(np.asarray(gt, dtype=np.float32), 0.0, 1.0)

    if bicubic_vis.ndim == 3:
        bicubic_vis = bicubic_vis[0]
    if srcnn_vis.ndim == 3:
        srcnn_vis = srcnn_vis[0]
    if gt_vis.ndim == 3:
        gt_vis = gt_vis[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(bicubic_vis, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("BICUBIC")
    axes[0].axis("off")

    axes[1].imshow(srcnn_vis, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("SRCNN OUTPUT")
    axes[1].axis("off")

    axes[2].imshow(gt_vis, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("GROUND TRUTH")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# BICUBIC BASELINE
# ============================================================


def compute_bicubic_metrics(gt_image):
    lr = gt_image.astype(np.float32)
    lr = np.clip(lr, 0.0, 1.0)

    lr_tensor = torch.from_numpy(lr).float().unsqueeze(0).unsqueeze(0)
    lr_size = (max(1, 256 // SCALE_FACTOR), max(1, 256 // SCALE_FACTOR))
    lr_small = F.interpolate(lr_tensor, size=lr_size, mode="bicubic", align_corners=False)
    bicubic = F.interpolate(lr_small, size=(256, 256), mode="bicubic", align_corners=False)
    bicubic_np = bicubic.squeeze(0).squeeze(0).cpu().numpy()
    bicubic_np = np.clip(bicubic_np, 0.0, 1.0)
    gt_np = np.clip(gt_image.astype(np.float32), 0.0, 1.0)

    mse = float(np.mean((bicubic_np - gt_np) ** 2))
    mae = float(np.mean(np.abs(bicubic_np - gt_np)))
    psnr = float(peak_signal_noise_ratio(gt_np, bicubic_np, data_range=1.0))
    ssim = float(structural_similarity(gt_np, bicubic_np, data_range=1.0))
    return mse, mae, psnr, ssim


# ============================================================
# EVALUATION MAIN
# ============================================================


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RESTORED_DIR, exist_ok=True)
    os.makedirs(COMPARISON_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("SRCNN ×2 SUPER-RESOLUTION BENCHMARK")
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

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT_PATH}")

    model, checkpoint = load_model()
    lpips_model = lpips.LPIPS(net="alex").to(DEVICE)
    lpips_model.eval()

    test_loader = DataLoader(
        SRDataset(test_pairs, scale_factor=SCALE_FACTOR),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    srcnn_results = []
    bicubic_results = []
    srcnn_inference_times = []

    with torch.no_grad():
        for i, (inputs, targets, filename) in enumerate(test_loader):
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = model(inputs)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            srcnn_inference_times.append(elapsed * 1000.0)

            predicted = outputs[0, 0].detach().cpu().numpy().copy()
            target_np = targets[0, 0].detach().cpu().numpy()
            predicted_clip = np.clip(predicted, 0.0, 1.0)
            target_clip = np.clip(target_np, 0.0, 1.0)

            mse = float(np.mean((predicted_clip - target_clip) ** 2))
            mae = float(np.mean(np.abs(predicted_clip - target_clip)))
            psnr = float(peak_signal_noise_ratio(target_clip, predicted_clip, data_range=1.0))
            ssim = float(structural_similarity(target_clip, predicted_clip, data_range=1.0))

            restored_tensor = torch.from_numpy(predicted_clip).float()
            gt_tensor = torch.from_numpy(target_clip).float()
            restored_lpips = grayscale_to_lpips(restored_tensor).to(DEVICE)
            gt_lpips = grayscale_to_lpips(gt_tensor).to(DEVICE)
            lpips_score = float(lpips_model(restored_lpips, gt_lpips).item())

            srcnn_results.append(
                {
                    "image_name": filename,
                    "mse": mse,
                    "mae": mae,
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lpips_score,
                    "inference_time_ms": float(elapsed * 1000.0),
                }
            )

            base_name = Path(filename).stem
            np.save(RESTORED_DIR / f"{base_name}.npy", predicted.astype(np.float32))
            save_grayscale_png(predicted_clip, RESTORED_DIR / f"{base_name}.png")

            # Bicubic baseline on the same GT image using the LR created from the GT.
            lr = torch.from_numpy(target_clip).float().unsqueeze(0).unsqueeze(0)
            lr_small = F.interpolate(lr, size=(128, 128), mode="bicubic", align_corners=False)
            bicubic_up = F.interpolate(lr_small, size=(256, 256), mode="bicubic", align_corners=False)
            bicubic_np = bicubic_up.squeeze(0).squeeze(0).numpy()
            bicubic_np = np.clip(bicubic_np, 0.0, 1.0)

            bic_mse = float(np.mean((bicubic_np - target_clip) ** 2))
            bic_mae = float(np.mean(np.abs(bicubic_np - target_clip)))
            bic_psnr = float(peak_signal_noise_ratio(target_clip, bicubic_np, data_range=1.0))
            bic_ssim = float(structural_similarity(target_clip, bicubic_np, data_range=1.0))

            bicubic_lpips = grayscale_to_lpips(torch.from_numpy(bicubic_np).float()).to(DEVICE)
            gt_lpips_bic = grayscale_to_lpips(gt_tensor).to(DEVICE)
            bic_lpips_score = float(lpips_model(bicubic_lpips, gt_lpips_bic).item())

            bicubic_results.append(
                {
                    "image_name": filename,
                    "mse": bic_mse,
                    "mae": bic_mae,
                    "psnr": bic_psnr,
                    "ssim": bic_ssim,
                    "lpips": bic_lpips_score,
                }
            )

            if i < 10 or (i + 1) % 50 == 0 or (i + 1) == len(test_loader):
                save_comparison_png(
                    bicubic_np,
                    predicted_clip,
                    target_clip,
                    COMPARISON_DIR / f"{base_name}_comparison.png",
                )

            if (i + 1) % 50 == 0 or (i + 1) == len(test_loader):
                print(f"Processed {i + 1}/{len(test_loader)}")

    srcnn_mse = float(np.mean([x["mse"] for x in srcnn_results]))
    srcnn_mae = float(np.mean([x["mae"] for x in srcnn_results]))
    srcnn_psnr = float(np.mean([x["psnr"] for x in srcnn_results]))
    srcnn_ssim = float(np.mean([x["ssim"] for x in srcnn_results]))
    srcnn_lpips = float(np.mean([x["lpips"] for x in srcnn_results]))
    srcnn_inference_mean = float(np.mean(srcnn_inference_times))
    srcnn_inference_median = float(np.median(srcnn_inference_times))

    bic_mse = float(np.mean([x["mse"] for x in bicubic_results]))
    bic_mae = float(np.mean([x["mae"] for x in bicubic_results]))
    bic_psnr = float(np.mean([x["psnr"] for x in bicubic_results]))
    bic_ssim = float(np.mean([x["ssim"] for x in bicubic_results]))
    bic_lpips = float(np.mean([x["lpips"] for x in bicubic_results]))

    with open(SRCNN_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "mse", "mae", "psnr", "ssim", "lpips", "inference_time_ms"])
        writer.writeheader()
        writer.writerows(srcnn_results)

    with open(BICUBIC_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "mse", "mae", "psnr", "ssim", "lpips"])
        writer.writeheader()
        writer.writerows(bicubic_results)

    srcnn_summary = (
        "SRCNN ×2 SUPER-RESOLUTION BENCHMARK\n"
        "======================================================================\n"
        f"Images evaluated : {len(srcnn_results)}\n"
        f"Mean MSE         : {srcnn_mse:.8f}\n"
        f"Mean MAE         : {srcnn_mae:.8f}\n"
        f"Mean PSNR        : {srcnn_psnr:.4f} dB\n"
        f"Mean SSIM        : {srcnn_ssim:.6f}\n"
        f"Mean LPIPS       : {srcnn_lpips:.6f}\n"
        f"Mean inference   : {srcnn_inference_mean:.2f} ms/image\n"
        f"Median inference : {srcnn_inference_median:.2f} ms/image\n"
        f"Device           : {DEVICE}\n"
        f"GPU              : {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'}\n"
        f"Checkpoint       : {CHECKPOINT_PATH}\n"
        f"Scale factor     : {SCALE_FACTOR}x\n"
        "======================================================================\n"
    )

    with open(SRCNN_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(srcnn_summary)

    bic_summary = (
        "BICUBIC ×2 SUPER-RESOLUTION BASELINE\n"
        "======================================================================\n"
        f"Images evaluated : {len(bicubic_results)}\n"
        f"Mean MSE         : {bic_mse:.8f}\n"
        f"Mean MAE         : {bic_mae:.8f}\n"
        f"Mean PSNR        : {bic_psnr:.4f} dB\n"
        f"Mean SSIM        : {bic_ssim:.6f}\n"
        f"Mean LPIPS       : {bic_lpips:.6f}\n"
        f"Device           : {DEVICE}\n"
        f"GPU              : {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'}\n"
        "======================================================================\n"
    )
    with open(BICUBIC_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(bic_summary)

    psnr_improvement = srcnn_psnr - bic_psnr
    ssim_improvement = srcnn_ssim - bic_ssim
    lpips_improvement = bic_lpips - srcnn_lpips

    print("\n" + "=" * 70)
    print("SRCNN ×2 SUPER-RESOLUTION BENCHMARK")
    print("=" * 70)
    print(f"Images evaluated : {len(srcnn_results)}")
    print(f"Bicubic PSNR      : {bic_psnr:.4f} dB")
    print(f"SRCNN PSNR        : {srcnn_psnr:.4f} dB")
    print(f"Bicubic SSIM      : {bic_ssim:.6f}")
    print(f"SRCNN SSIM        : {srcnn_ssim:.6f}")
    print(f"Bicubic LPIPS     : {bic_lpips:.6f}")
    print(f"SRCNN LPIPS       : {srcnn_lpips:.6f}")
    print(f"SRCNN inference   : {srcnn_inference_mean:.2f} ms/image")
    print("=" * 70)
    print(f"PSNR improvement  : {psnr_improvement:.4f} dB")
    print(f"SSIM improvement  : {ssim_improvement:.6f}")
    print(f"LPIPS improvement : {lpips_improvement:.6f}")
    print("=" * 70)

    print(f"SRCNN CSV         : {SRCNN_CSV_PATH}")
    print(f"SRCNN summary     : {SRCNN_SUMMARY_PATH}")
    print(f"Bicubic CSV       : {BICUBIC_CSV_PATH}")
    print(f"Bicubic summary   : {BICUBIC_SUMMARY_PATH}")
    print(f"Restored outputs  : {RESTORED_DIR}")
    print(f"Comparisons       : {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
