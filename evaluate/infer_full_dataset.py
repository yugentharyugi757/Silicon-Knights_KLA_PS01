import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import time

import numpy as np
import torch
from PIL import Image

from models.noise_aware_dncnn import NoiseAwareDnCNN


# ============================================================
# CONFIGURATION
# ============================================================

GT_DIR = PROJECT_ROOT / "train" / "GT"
NOISY_DIR = PROJECT_ROOT / "train" / "NoisyLR"
CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints" / "best_noise_aware_dncnn.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "full_dataset_inference"
RESTORED_DIR = OUTPUT_DIR / "restored"
COMPARISON_DIR = OUTPUT_DIR / "comparisons"

NUM_FEATURES = 64
NUM_BLOCKS = 10


# Same noise model used by the final evaluation script
VAR_A = 0.01846589
VAR_B = 0.00843552
VAR_C = -0.00033869


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
# LOAD MODEL
# ============================================================

def load_model():
    model = NoiseAwareDnCNN(
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

    return model, checkpoint


# ============================================================
# SAVE RESTORED PNG
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
        raise ValueError(
            f"Unsupported grayscale image shape for PNG save: {image.shape}"
        )


# ============================================================
# SAVE COMPARISON IMAGE
# ============================================================

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

    # Match degraded input size to restored output size
    # when the model performs super-resolution.
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

    axes[0].imshow(
        degraded_vis,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[0].set_title("DEGRADED INPUT")
    axes[0].axis("off")

    axes[1].imshow(
        restored_vis,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("RESTORED OUTPUT")
    axes[1].axis("off")

    axes[2].imshow(
        gt_vis,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("GROUND TRUTH")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# BUILD ALL PAIRED FILES
# ============================================================

def build_pairs():
    gt_files = sorted(
        [
            f
            for f in os.listdir(GT_DIR)
            if f.endswith(".npy")
        ]
    )

    noisy_files = sorted(
        [
            f
            for f in os.listdir(NOISY_DIR)
            if f.endswith(".npy")
        ]
    )

    noisy_set = set(noisy_files)

    pairs = []

    for filename in gt_files:
        if filename in noisy_set:
            noisy_path = os.path.join(
                NOISY_DIR,
                filename,
            )

            gt_path = os.path.join(
                GT_DIR,
                filename,
            )

            pairs.append(
                (
                    noisy_path,
                    gt_path,
                )
            )

    return pairs


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(model, noisy_path, gt_path):
    noisy = np.load(noisy_path).astype(np.float32)
    gt = np.load(gt_path).astype(np.float32)

    noisy = np.nan_to_num(
        noisy,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    gt = np.nan_to_num(
        gt,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    gt = np.clip(gt, 0.0, 1.0)

    # Same sigma-map calculation as final evaluation
    noisy_for_sigma = np.clip(
        noisy,
        0.0,
        1.0,
    )

    sigma = compute_noise_sigma(
        noisy_for_sigma
    )

    noisy_tensor = (
        torch.from_numpy(noisy)
        .unsqueeze(0)
    )

    sigma_tensor = (
        torch.from_numpy(sigma)
        .unsqueeze(0)
    )

    input_tensor = torch.cat(
        [
            noisy_tensor,
            sigma_tensor,
        ],
        dim=0,
    )

    input_tensor = (
        input_tensor
        .unsqueeze(0)
        .to(
            DEVICE,
            non_blocking=True,
        )
    )

    with torch.no_grad():
        output = model(input_tensor)

    restored_raw = (
        output[0, 0]
        .detach()
        .cpu()
        .numpy()
        .copy()
    )

    restored_for_png = np.clip(
        restored_raw,
        0.0,
        1.0,
    )

    return (
        noisy,
        restored_raw,
        restored_for_png,
        gt,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("KLA FULL DATASET INFERENCE")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # VERIFY REQUIRED PATHS
    # --------------------------------------------------------

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    if not NOISY_DIR.exists():
        raise FileNotFoundError(
            f"Noisy dataset directory not found:\n{NOISY_DIR}"
        )

    if not GT_DIR.exists():
        raise FileNotFoundError(
            f"Ground-truth directory not found:\n{GT_DIR}"
        )

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORIES
    # --------------------------------------------------------

    RESTORED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    COMPARISON_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # BUILD ALL PAIRS
    # --------------------------------------------------------

    pairs = build_pairs()

    print(
        f"Total paired images : {len(pairs)}"
    )

    if len(pairs) == 0:
        raise RuntimeError(
            "No paired images were found."
        )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading final model...")

    model, checkpoint = load_model()

    print(
        f"Features            : {NUM_FEATURES}"
    )

    print(
        f"Residual blocks     : {NUM_BLOCKS}"
    )

    if isinstance(checkpoint, dict):
        print(
            f"Checkpoint epoch    : "
            f"{checkpoint.get('epoch', 'unknown')}"
        )

        print(
            f"Best validation loss: "
            f"{checkpoint.get('best_val_loss', 'unknown')}"
        )

    print(
        f"Checkpoint          : "
        f"{CHECKPOINT_PATH}"
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STARTING FULL DATASET INFERENCE")
    print("=" * 70)

    start_time = time.perf_counter()

    successful = 0
    failed = []

    total = len(pairs)

    for index, (noisy_path, gt_path) in enumerate(
        pairs,
        start=1,
    ):

        filename = os.path.basename(
            noisy_path
        )

        base_name = Path(
            filename
        ).stem

        try:

            (
                degraded,
                restored_raw,
                restored_for_png,
                gt,
            ) = process_image(
                model,
                noisy_path,
                gt_path,
            )

            # ------------------------------------------------
            # SAVE NPY
            # ------------------------------------------------

            np.save(
                RESTORED_DIR
                / f"{base_name}.npy",
                restored_raw.astype(
                    np.float32
                ),
            )

            # ------------------------------------------------
            # SAVE PNG
            # ------------------------------------------------

            save_grayscale_png(
                restored_for_png,
                RESTORED_DIR
                / f"{base_name}.png",
            )

            # ------------------------------------------------
            # SAVE COMPARISON
            # ------------------------------------------------

            save_comparison_png(
                degraded,
                restored_for_png,
                gt,
                COMPARISON_DIR
                / f"{base_name}_comparison.png",
            )

            successful += 1

        except Exception as error:

            failed.append(
                (
                    filename,
                    str(error),
                )
            )

            print(
                f"\nERROR: {filename}"
            )

            print(
                f"       {error}"
            )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            index % 100 == 0
            or index == total
        ):

            elapsed = (
                time.perf_counter()
                - start_time
            )

            avg_time = (
                elapsed / index
            )

            remaining = (
                avg_time
                * (total - index)
            )

            print(
                f"Processed {index}/{total} "
                f"| Successful: {successful} "
                f"| Failed: {len(failed)} "
                f"| Avg: {avg_time:.3f} s/image "
                f"| ETA: {remaining / 60:.1f} min"
            )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    total_time = (
        time.perf_counter()
        - start_time
    )

    print("\n" + "=" * 70)
    print("FULL DATASET INFERENCE COMPLETE")
    print("=" * 70)

    print(
        f"Total input images      : {total}"
    )

    print(
        f"Successfully restored   : {successful}"
    )

    print(
        f"Failed                  : {len(failed)}"
    )

    print(
        f"Total inference time    : "
        f"{total_time:.2f} seconds"
    )

    if successful > 0:
        print(
            f"Average time/image      : "
            f"{total_time / successful:.3f} seconds"
        )

    print(
        f"\nRestored NPY            : "
        f"{RESTORED_DIR}"
    )

    print(
        f"Restored PNG            : "
        f"{RESTORED_DIR}"
    )

    print(
        f"Comparisons             : "
        f"{COMPARISON_DIR}"
    )

    # --------------------------------------------------------
    # FAILED FILES
    # --------------------------------------------------------

    if failed:

        print("\n" + "=" * 70)
        print("FAILED FILES")
        print("=" * 70)

        for filename, error in failed:
            print(
                f"{filename} -> {error}"
            )

    else:

        print(
            "\nAll images were processed successfully."
        )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()