import os
import time
import csv

import numpy as np
import torch
import lpips

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from models.noise_aware_dncnn import NoiseAwareDnCNN


# ============================================================
# CONFIG
# ============================================================

GT_DIR = r"train\GT"
NOISY_DIR = r"train\NoisyLR"

CHECKPOINT = r"checkpoints\best_noise_aware_dncnn.pth"

OUTPUT_DIR = r"outputs\lpips_evaluation"
RESULTS_DIR = r"results"

NUM_FEATURES = 64
NUM_BLOCKS = 10
VAR_A = 0.01846589
VAR_B = 0.00843552
VAR_C = -0.00033869

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("KLA NOISE-AWARE DnCNN LPIPS EVALUATION")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# MODEL
# ============================================================

print("\nLoading checkpoint...")

model = NoiseAwareDnCNN(
    num_features=NUM_FEATURES,
    num_blocks=NUM_BLOCKS
).to(device)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)


# Handle checkpoint formats
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

print("Checkpoint loaded successfully.")

if isinstance(checkpoint, dict):

    if "epoch" in checkpoint:
        print("Checkpoint epoch :", checkpoint["epoch"])

    if "best_val_loss" in checkpoint:
        print(
            "Best val loss    :",
            checkpoint["best_val_loss"]
        )


# ============================================================
# LPIPS
# ============================================================

print("\nLoading LPIPS model...")

lpips_model = lpips.LPIPS(
    net="alex"
).to(device)

lpips_model.eval()

print("LPIPS network: AlexNet")


# ============================================================
# DATASET
# ============================================================

gt_files = sorted([
    f for f in os.listdir(GT_DIR)
    if f.endswith(".npy")
])

noisy_files = sorted([
    f for f in os.listdir(NOISY_DIR)
    if f.endswith(".npy")
])


pairs = [
    f for f in noisy_files
    if f in gt_files
]


print("\nTotal paired images:", len(pairs))


# ============================================================
# HELPER
# ============================================================

def load_npy(path):

    image = np.load(path).astype(np.float32)

    return torch.from_numpy(image)


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


def grayscale_to_lpips(image):
    """
    Input:
        [1, H, W]

    LPIPS:
        [1, 3, H, W]

    Replicate grayscale channel.
    """

    image = image.unsqueeze(0)

    image = image.repeat(
        1, 3, 1, 1
    )

    # [0,1] -> [-1,1]
    image = image * 2.0 - 1.0

    return image


# ============================================================
# EVALUATION
# ============================================================

results = []

total_mse = 0.0
total_psnr = 0.0
total_ssim = 0.0
total_lpips = 0.0
total_inference = 0.0


print("\n" + "=" * 70)
print("STARTING LPIPS EVALUATION")
print("=" * 70)


for index, filename in enumerate(pairs):

    noisy_path = os.path.join(
        NOISY_DIR,
        filename
    )

    gt_path = os.path.join(
        GT_DIR,
        filename
    )


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    noisy = load_npy(noisy_path)
    gt = load_npy(gt_path)


    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if noisy.shape != (128, 128):
        raise ValueError(
            f"Unexpected input shape for "
            f"{filename}: {noisy.shape}"
        )

    if gt.shape != (256, 256):
        raise ValueError(
            f"Unexpected GT shape for "
            f"{filename}: {gt.shape}"
        )


    # --------------------------------------------------------
    # Model input
    # --------------------------------------------------------

    # Convert noisy image to NumPy for sigma estimation
    noisy_for_sigma = np.clip(
        noisy.numpy(),
        0.0,
        1.0
    )

    # Estimate per-pixel noise sigma
    sigma = compute_noise_sigma(
        noisy_for_sigma
    )

    # Channel 0: noisy LR image
    noisy_tensor = noisy.unsqueeze(0)

    # Channel 1: estimated noise sigma map
    sigma_tensor = torch.from_numpy(
        sigma
    ).unsqueeze(0)

    # Combine into [2, 128, 128]
    input_tensor = torch.cat(
        [
            noisy_tensor,
            sigma_tensor
        ],
        dim=0
    )

    # Add batch dimension -> [1, 2, 128, 128]
    noisy_input = input_tensor.unsqueeze(0).to(device)

    print("Model input shape:", noisy_input.shape)


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        restored = model(noisy_input)

    print("Model output shape:", restored.shape)

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_time = (
        time.perf_counter() - start
    ) * 1000.0


    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    restored = restored.squeeze(
        0
    ).squeeze(
        0
    )

    restored = torch.clamp(
        restored,
        0.0,
        1.0
    )


    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    restored_np = (
        restored
        .detach()
        .cpu()
        .numpy()
    )

    gt_np = gt.numpy()


    # --------------------------------------------------------
    # Verify output
    # --------------------------------------------------------

    if restored_np.shape != gt_np.shape:

        raise ValueError(
            f"Output/GT mismatch for "
            f"{filename}: "
            f"{restored_np.shape} vs "
            f"{gt_np.shape}"
        )


    # --------------------------------------------------------
    # MSE
    # --------------------------------------------------------

    mse = np.mean(
        (restored_np - gt_np) ** 2
    )


    # --------------------------------------------------------
    # PSNR
    # --------------------------------------------------------

    psnr = peak_signal_noise_ratio(
        gt_np,
        restored_np,
        data_range=1.0
    )


    # --------------------------------------------------------
    # SSIM
    # --------------------------------------------------------

    ssim = structural_similarity(
        gt_np,
        restored_np,
        data_range=1.0
    )


    # --------------------------------------------------------
    # LPIPS
    # --------------------------------------------------------

    restored_tensor = torch.from_numpy(
        restored_np
    ).float()

    gt_tensor = torch.from_numpy(
        gt_np
    ).float()


    restored_lpips = grayscale_to_lpips(
        restored_tensor
    ).to(device)

    gt_lpips = grayscale_to_lpips(
        gt_tensor
    ).to(device)


    with torch.no_grad():

        lpips_score = lpips_model(
            restored_lpips,
            gt_lpips
        ).item()


    # --------------------------------------------------------
    # Accumulate
    # --------------------------------------------------------

    total_mse += mse
    total_psnr += psnr
    total_ssim += ssim
    total_lpips += lpips_score
    total_inference += inference_time


    results.append({
        "filename": filename,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips_score,
        "inference_ms": inference_time
    })


    # --------------------------------------------------------
    # Save restored image
    # --------------------------------------------------------

    np.save(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        restored_np
    )


    if (index + 1) % 50 == 0:

        print(
            f"Processed "
            f"{index + 1}/{len(pairs)}"
        )


# ============================================================
# SUMMARY
# ============================================================

count = len(results)

mean_mse = total_mse / count
mean_psnr = total_psnr / count
mean_ssim = total_ssim / count
mean_lpips = total_lpips / count
mean_inference = total_inference / count


inference_times = [
    x["inference_ms"]
    for x in results
]

median_inference = np.median(
    inference_times
)


print("\n" + "=" * 70)
print("NOISE-AWARE DnCNN LPIPS TEST RESULTS")
print("=" * 70)

print(
    f"Images evaluated : {count}"
)

print(
    f"Mean MSE         : "
    f"{mean_mse:.8f}"
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
    f"Mean LPIPS       : "
    f"{mean_lpips:.6f}"
)

print(
    f"Mean inference   : "
    f"{mean_inference:.2f} ms/image"
)

print(
    f"Median inference : "
    f"{median_inference:.2f} ms/image"
)

print("=" * 70)


# ============================================================
# SAVE CSV
# ============================================================

csv_path = os.path.join(
    RESULTS_DIR,
    "noise_aware_lpips_results.csv"
)


with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "mse",
            "psnr",
            "ssim",
            "lpips",
            "inference_ms"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULTS_DIR,
    "noise_aware_lpips_summary.txt"
)


with open(
    summary_path,
    "w"
) as f:

    f.write(
        "KLA NOISE-AWARE DnCNN LPIPS EVALUATION\n"
    )

    f.write("=" * 70 + "\n")

    f.write(
        f"Images evaluated : {count}\n"
    )

    f.write(
        f"Mean MSE         : {mean_mse:.8f}\n"
    )

    f.write(
        f"Mean PSNR        : {mean_psnr:.4f} dB\n"
    )

    f.write(
        f"Mean SSIM        : {mean_ssim:.6f}\n"
    )

    f.write(
        f"Mean LPIPS       : {mean_lpips:.6f}\n"
    )

    f.write(
        f"Mean inference   : "
        f"{mean_inference:.2f} ms/image\n"
    )

    f.write(
        f"Median inference : "
        f"{median_inference:.2f} ms/image\n"
    )

    f.write("=" * 70 + "\n")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("LPIPS EVALUATION COMPLETE")
print("=" * 70)

print(
    "CSV results :",
    csv_path
)

print(
    "Summary     :",
    summary_path
)

print(
    "Restored    :",
    OUTPUT_DIR
)

print("=" * 70)