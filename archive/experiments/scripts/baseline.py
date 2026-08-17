import os
import numpy as np
import cv2
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# ============================================================
# CONFIGURATION
# ============================================================

GT_DIR = "train/GT"
NOISY_DIR = "train/NoisyLR"

OUTPUT_DIR = "outputs/bicubic"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD NPY IMAGE
# ============================================================

def load_npy(path):
    image = np.load(path)

    # Convert to float32
    image = image.astype(np.float32)

    return image


# ============================================================
# NORMALIZE IMAGE
# ============================================================

def normalize_for_evaluation(image):
    """
    GT images are already approximately [0, 1].

    NoisyLR can contain values below 0 and above 1.
    For image reconstruction output we clip it to [0, 1].
    """

    image = np.clip(image, 0.0, 1.0)

    return image


# ============================================================
# BICUBIC UPSAMPLING
# ============================================================

def bicubic_upscale(image):
    """
    128x128 -> 256x256
    """

    output = cv2.resize(
        image,
        (256, 256),
        interpolation=cv2.INTER_CUBIC
    )

    return output.astype(np.float32)


# ============================================================
# MAIN
# ============================================================

def main():

    noisy_files = sorted([
        f for f in os.listdir(NOISY_DIR)
        if f.endswith(".npy")
    ])

    print("=" * 60)
    print("KLA BICUBIC BASELINE")
    print("=" * 60)

    print(f"NoisyLR images: {len(noisy_files)}")

    psnr_values = []
    ssim_values = []
    mse_values = []

    for filename in tqdm(noisy_files):

        noisy_path = os.path.join(NOISY_DIR, filename)
        gt_path = os.path.join(GT_DIR, filename)

        # ----------------------------------------------------
        # Check pair
        # ----------------------------------------------------

        if not os.path.exists(gt_path):
            print(f"Missing GT: {filename}")
            continue

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        noisy = load_npy(noisy_path)
        gt = load_npy(gt_path)

        # ----------------------------------------------------
        # Upsample
        # ----------------------------------------------------

        prediction = bicubic_upscale(noisy)

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        prediction = normalize_for_evaluation(prediction)
        gt = normalize_for_evaluation(gt)

        # ----------------------------------------------------
        # MSE
        # ----------------------------------------------------

        mse = np.mean((prediction - gt) ** 2)

        # ----------------------------------------------------
        # PSNR
        # ----------------------------------------------------

        psnr = peak_signal_noise_ratio(
            gt,
            prediction,
            data_range=1.0
        )

        # ----------------------------------------------------
        # SSIM
        # ----------------------------------------------------

        ssim = structural_similarity(
            gt,
            prediction,
            data_range=1.0
        )

        mse_values.append(mse)
        psnr_values.append(psnr)
        ssim_values.append(ssim)

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("BICUBIC BASELINE RESULTS")
    print("=" * 60)

    print(f"Images evaluated : {len(psnr_values)}")
    print(f"Mean MSE         : {np.mean(mse_values):.8f}")
    print(f"Mean PSNR        : {np.mean(psnr_values):.4f} dB")
    print(f"Mean SSIM        : {np.mean(ssim_values):.6f}")

    print("=" * 60)

    # Save metrics
    np.save(
        os.path.join(OUTPUT_DIR, "bicubic_psnr.npy"),
        np.array(psnr_values)
    )

    np.save(
        os.path.join(OUTPUT_DIR, "bicubic_ssim.npy"),
        np.array(ssim_values)
    )

    np.save(
        os.path.join(OUTPUT_DIR, "bicubic_mse.npy"),
        np.array(mse_values)
    )


if __name__ == "__main__":
    main()