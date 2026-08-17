import os
import cv2
import numpy as np

from tqdm import tqdm
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


# ============================================================
# CONFIG
# ============================================================

GT_DIR = "train/GT"
NOISY_DIR = "train/NoisyLR"

OUTPUT_DIR = "outputs/traditional"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):
    return np.load(path).astype(np.float32)


# ============================================================
# METRICS
# ============================================================

def evaluate(prediction, gt):

    prediction = np.clip(prediction, 0.0, 1.0)
    gt = np.clip(gt, 0.0, 1.0)

    mse = np.mean((prediction - gt) ** 2)

    psnr = peak_signal_noise_ratio(
        gt,
        prediction,
        data_range=1.0
    )

    ssim = structural_similarity(
        gt,
        prediction,
        data_range=1.0
    )

    return mse, psnr, ssim


# ============================================================
# FILTERS
# ============================================================

def bicubic(noisy):

    return cv2.resize(
        noisy,
        (256, 256),
        interpolation=cv2.INTER_CUBIC
    )


def gaussian_filter(noisy):

    return cv2.GaussianBlur(
        noisy,
        (5, 5),
        sigmaX=1.0
    )


def median_filter(noisy):

    return cv2.medianBlur(
        noisy.astype(np.float32),
        5
    )


def bilateral_filter(noisy):

    return cv2.bilateralFilter(
        noisy.astype(np.float32),
        d=5,
        sigmaColor=0.1,
        sigmaSpace=5
    )


# ============================================================
# EXPERIMENT
# ============================================================

filters = {

    "Bicubic": lambda x: x,

    "Gaussian": gaussian_filter,

    "Median": median_filter,

    "Bilateral": bilateral_filter,
}


results = {}


for name in filters:

    results[name] = {
        "mse": [],
        "psnr": [],
        "ssim": []
    }


# ============================================================
# DATASET
# ============================================================

files = sorted([
    f for f in os.listdir(NOISY_DIR)
    if f.endswith(".npy")
])


print("=" * 70)
print("KLA TRADITIONAL IMAGE PROCESSING BASELINE")
print("=" * 70)

print(f"Images: {len(files)}")


# ============================================================
# LOOP
# ============================================================

for filename in tqdm(files):

    noisy_path = os.path.join(
        NOISY_DIR,
        filename
    )

    gt_path = os.path.join(
        GT_DIR,
        filename
    )

    if not os.path.exists(gt_path):
        continue

    noisy = load_image(noisy_path)
    gt = load_image(gt_path)

    for name, filter_function in filters.items():

        # --------------------------------------------
        # Apply filter
        # --------------------------------------------

        filtered = filter_function(noisy)

        # --------------------------------------------
        # Upsample
        # --------------------------------------------

        prediction = bicubic(filtered)

        # --------------------------------------------
        # Evaluate
        # --------------------------------------------

        mse, psnr, ssim = evaluate(
            prediction,
            gt
        )

        results[name]["mse"].append(mse)
        results[name]["psnr"].append(psnr)
        results[name]["ssim"].append(ssim)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(
    f"{'Method':<15}"
    f"{'MSE':>15}"
    f"{'PSNR (dB)':>15}"
    f"{'SSIM':>15}"
)

print("-" * 70)


for name in results:

    mse = np.mean(results[name]["mse"])
    psnr = np.mean(results[name]["psnr"])
    ssim = np.mean(results[name]["ssim"])

    print(
        f"{name:<15}"
        f"{mse:>15.8f}"
        f"{psnr:>15.4f}"
        f"{ssim:>15.6f}"
    )


print("=" * 70)