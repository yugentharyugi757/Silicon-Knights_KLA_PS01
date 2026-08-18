import sys
from pathlib import Path

import numpy as np
import torch

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.noise_aware_dncnn import NoiseAwareDnCNN


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_noise_aware_dncnn.pth"
)

NUM_FEATURES = 64
NUM_BLOCKS = 10

# Same noise model used during final evaluation/inference
VAR_A = 0.01846589
VAR_B = 0.00843552
VAR_C = -0.00033869


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# NOISE SIGMA MAP
# ============================================================

def compute_noise_sigma(image):
    """
    Compute the signal-dependent noise sigma map.

    This is the same noise model used by the
    final trained model.
    """

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
# LOAD MODEL
# ============================================================

def load_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    model = NoiseAwareDnCNN(
        num_features=NUM_FEATURES,
        num_blocks=NUM_BLOCKS
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
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

    return model


# ============================================================
# VALIDATE INPUT
# ============================================================

def prepare_input(array, filename):

    array = np.asarray(
        array,
        dtype=np.float32
    )

    # Remove unnecessary single channel dimension
    if array.ndim == 3:

        if array.shape[0] == 1:
            array = array[0]

        elif array.shape[-1] == 1:
            array = array[:, :, 0]

        else:
            raise ValueError(
                f"{filename}: expected grayscale array, "
                f"got shape {array.shape}"
            )

    if array.ndim != 2:
        raise ValueError(
            f"{filename}: expected shape (H, W), "
            f"got {array.shape}"
        )

    # Remove invalid values
    array = np.nan_to_num(
        array,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    # Input images are expected in [0, 1]
    array = np.clip(
        array,
        0.0,
        1.0
    )

    return array


# ============================================================
# RESTORE ONE IMAGE
# ============================================================

def restore_image(model, noisy):

    # Same preprocessing used in final inference
    noisy_for_sigma = np.clip(
        noisy,
        0.0,
        1.0
    )

    sigma = compute_noise_sigma(
        noisy_for_sigma
    )

    # Image channel
    noisy_tensor = (
        torch.from_numpy(noisy)
        .unsqueeze(0)
    )

    # Noise sigma channel
    sigma_tensor = (
        torch.from_numpy(sigma)
        .unsqueeze(0)
    )

    # 2-channel input:
    # channel 0 = noisy image
    # channel 1 = noise sigma
    input_tensor = torch.cat(
        [
            noisy_tensor,
            sigma_tensor
        ],
        dim=0
    )

    # Add batch dimension
    input_tensor = (
        input_tensor
        .unsqueeze(0)
        .to(
            DEVICE,
            non_blocking=True
        )
    )

    # Inference
    with torch.no_grad():

        output = model(
            input_tensor
        )

    # Model output:
    # [batch, channel, H*2, W*2]
    restored = (
        output[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    # Remove NaN / Inf
    restored = np.nan_to_num(
        restored,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    # Required output range
    restored = np.clip(
        restored,
        0.0,
        1.0
    )

    return restored.astype(
        np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # ARGUMENT CHECK
    # --------------------------------------------------------

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            "python run.py <input-dir> <output-dir>"
        )

        sys.exit(1)

    input_dir = Path(
        sys.argv[1]
    )

    output_dir = Path(
        sys.argv[2]
    )

    # --------------------------------------------------------
    # VALIDATE INPUT DIRECTORY
    # --------------------------------------------------------

    if not input_dir.exists():

        raise FileNotFoundError(
            f"Input directory not found:\n{input_dir}"
        )

    if not input_dir.is_dir():

        raise NotADirectoryError(
            f"Input path is not a directory:\n{input_dir}"
        )

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # FIND INPUT FILES
    # --------------------------------------------------------

    input_files = sorted(
        input_dir.glob("*.npy")
    )

    if not input_files:

        raise RuntimeError(
            f"No .npy files found in:\n{input_dir}"
        )

    print("=" * 70)
    print("KLA IMAGE RESTORATION")
    print("=" * 70)

    print(f"Device       : {DEVICE}")

    if DEVICE.type == "cuda":

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Input files  : {len(input_files)}"
    )

    print(
        f"Input folder : {input_dir}"
    )

    print(
        f"Output folder: {output_dir}"
    )

    # --------------------------------------------------------
    # LOAD MODEL ONCE
    # --------------------------------------------------------

    print("\nLoading model...")

    model = load_model()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # PROCESS FILES
    # --------------------------------------------------------

    successful = 0
    failed = []

    for index, input_path in enumerate(
        input_files,
        start=1
    ):

        output_path = (
            output_dir
            / input_path.name
        )

        try:

            # Load input
            noisy = np.load(
                input_path
            )

            # Validate / prepare
            noisy = prepare_input(
                noisy,
                input_path.name
            )

            # Restore
            restored = restore_image(
                model,
                noisy
            )

            # Final safety checks
            if not np.isfinite(
                restored
            ).all():

                raise ValueError(
                    "Output contains NaN or Inf"
                )

            if restored.min() < 0.0:

                raise ValueError(
                    "Output contains values below 0"
                )

            if restored.max() > 1.0:

                raise ValueError(
                    "Output contains values above 1"
                )

            # Save using EXACT same filename
            np.save(
                output_path,
                restored
            )

            successful += 1

            print(
                f"[{index}/{len(input_files)}] "
                f"{input_path.name} "
                f"-> "
                f"{restored.shape}"
            )

        except Exception as error:

            failed.append(
                (
                    input_path.name,
                    str(error)
                )
            )

            print(
                f"[ERROR] "
                f"{input_path.name}: "
                f"{error}"
            )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)

    print(
        f"Total input files : {len(input_files)}"
    )

    print(
        f"Successful        : {successful}"
    )

    print(
        f"Failed            : {len(failed)}"
    )

    print(
        f"Output directory  : {output_dir}"
    )

    if failed:

        print("\nFailed files:")

        for filename, error in failed:

            print(
                f"  {filename}: {error}"
            )

        sys.exit(1)

    print(
        "\nAll files processed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()