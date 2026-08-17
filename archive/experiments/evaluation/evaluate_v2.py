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
from torch.utils.data import DataLoader, Dataset

from models.noise_aware_dncnn_v2 import NoiseAwareDnCNNV2


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

GT_DIR = PROJECT_ROOT / "train" / "GT"
NOISY_DIR = PROJECT_ROOT / "train" / "NoisyLR"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_noise_aware_v2.pth"

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v2_evaluation"
RESTORED_DIR = OUTPUT_DIR / "restored"
COMPARISON_DIR = OUTPUT_DIR / "comparisons"
CSV_PATH = RESULTS_DIR / "v2_test_results.csv"
SUMMARY_PATH = RESULTS_DIR / "v2_test_summary.txt"

NUM_FEATURES = 64
NUM_BLOCKS = 10
BATCH_SIZE = 1
NUM_WORKERS = 0

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

        input_tensor = torch.cat([noisy_tensor, sigma_tensor], dim=0)
        return input_tensor, gt_tensor, os.path.basename(noisy_path)


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
# LOAD CHECKPOINT
# ============================================================

def verify_checkpoint_matches_v2(model, checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model_keys = list(model.state_dict().keys())
    checkpoint_keys = list(state_dict.keys())

    if model_keys != checkpoint_keys:
        mismatch = {
            "missing_in_checkpoint": [k for k in model_keys if k not in checkpoint_keys],
            "extra_in_checkpoint": [k for k in checkpoint_keys if k not in model_keys],
        }
        raise ValueError(
            "Checkpoint architecture does not match NoiseAwareDnCNNV2 exactly. "
            f"Mismatch details: {mismatch}"
        )

    for key, value in model.state_dict().items():
        if key not in state_dict:
            raise ValueError(f"Missing layer in checkpoint: {key}")
        if tuple(value.shape) != tuple(state_dict[key].shape):
            raise ValueError(
                f"Shape mismatch for {key}: model={tuple(value.shape)}, "
                f"checkpoint={tuple(state_dict[key].shape)}"
            )


def load_model():
    model = NoiseAwareDnCNNV2(
        num_features=NUM_FEATURES,
        num_blocks=NUM_BLOCKS,
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
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
    verify_checkpoint_matches_v2(model, checkpoint)
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
# MAIN
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
        from skimage.transform import resize
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


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RESTORED_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    print("=" * 70)
    print("KLA NOISE-AWARE DnCNN V2 BENCHMARK")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

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

    model, checkpoint = load_model()

    print(f"Features            : {NUM_FEATURES}")
    print(f"Residual blocks     : {NUM_BLOCKS}")

    print("\nLoading LPIPS model...")
    lpips_model = lpips.LPIPS(net="alex").to(DEVICE)
    lpips_model.eval()
    print("LPIPS network: AlexNet")

    results = []
    inference_times = []

    print("\n" + "=" * 70)
    print("STARTING V2 EVALUATION")
    print("=" * 70)

    with torch.no_grad():
        warmup_inputs, _, _ = next(iter(test_loader))
        warmup_inputs = warmup_inputs.to(DEVICE, non_blocking=True)
        _, _ = measure_pure_model_inference_time(model, warmup_inputs, DEVICE)

        for i, (inputs, targets, filenames) in enumerate(test_loader):
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            outputs, elapsed = measure_pure_model_inference_time(model, inputs, DEVICE)
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
        "KLA NOISE-AWARE DnCNN V2 BENCHMARK\n"
        "======================================================================\n"
        f"Images evaluated : {len(results)}\n"
        f"Mean MSE         : {mean_mse:.8f}\n"
        f"Mean MAE         : {mean_mae:.8f}\n"
        f"Mean PSNR        : {mean_psnr:.4f} dB\n"
        f"Mean SSIM        : {mean_ssim:.6f}\n"
        f"Mean LPIPS       : {mean_lpips:.6f}\n"
        f"Pure model inference time (mean) : {mean_inference:.2f} ms/image\n"
        f"Pure model inference time (median) : {median_inference:.2f} ms/image\n"
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
    print("V2 BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Images evaluated : {len(results)}")
    print(f"Mean MSE         : {mean_mse:.8f}")
    print(f"Mean MAE         : {mean_mae:.8f}")
    print(f"Mean PSNR        : {mean_psnr:.4f} dB")
    print(f"Mean SSIM        : {mean_ssim:.6f}")
    print(f"Mean LPIPS       : {mean_lpips:.6f}")
    print(f"Pure model inference time (mean) : {mean_inference:.2f} ms/image")
    print(f"Pure model inference time (median) : {median_inference:.2f} ms/image")
    print(f"Device           : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint path  : {CHECKPOINT_PATH}")
    print(f"Checkpoint epoch : {checkpoint.get('epoch', 'unknown')}")
    print(f"Best val loss    : {checkpoint.get('best_val_loss', 'unknown')}")
    print("=" * 70)

    terminal_report = (
        "\n======================================================================\n"
        "KLA NOISE-AWARE DnCNN V2 BENCHMARK\n"
        "======================================================================\n"
        f"Device: {DEVICE}\n"
        f"GPU: {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'}\n"
        f"Test images: {len(results)}\n"
        f"MSE: {mean_mse:.8f}\n"
        f"MAE: {mean_mae:.8f}\n"
        f"PSNR: {mean_psnr:.4f} dB\n"
        f"SSIM: {mean_ssim:.6f}\n"
        f"LPIPS: {mean_lpips:.6f}\n"
        f"Pure model inference time (mean): {mean_inference:.2f} ms/image\n"
        f"Pure model inference time (median): {median_inference:.2f} ms/image\n"
        "======================================================================\n"
    )
    print(terminal_report)

    print(f"CSV results : {CSV_PATH}")
    print(f"Summary     : {SUMMARY_PATH}")
    print(f"Restored NPY     : {RESTORED_DIR}")
    print(f"Restored PNG     : {RESTORED_DIR}")
    print(f"Comparisons      : {COMPARISON_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
