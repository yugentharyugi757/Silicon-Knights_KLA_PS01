import os
import random
import time
import csv

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

GT_DIR = "train/GT"
NOISY_DIR = "train/NoisyLR"

CHECKPOINT_PATH = "checkpoints/best_dncnn.pth"

OUTPUT_DIR = "outputs/evaluation"
RESTORED_DIR = os.path.join(OUTPUT_DIR, "restored")
COMPARISON_DIR = os.path.join(OUTPUT_DIR, "comparisons")

RESULT_DIR = "results"

os.makedirs(RESTORED_DIR, exist_ok=True)
os.makedirs(COMPARISON_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

BATCH_SIZE = 1
NUM_WORKERS = 0

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# DATASET
# ============================================================

class KLATestDataset(Dataset):

    def __init__(self, filenames):

        self.filenames = filenames

    def __len__(self):

        return len(self.filenames)

    def __getitem__(self, index):

        filename = self.filenames[index]

        noisy_path = os.path.join(
            NOISY_DIR,
            filename
        )

        gt_path = os.path.join(
            GT_DIR,
            filename
        )

        noisy = np.load(
            noisy_path
        ).astype(np.float32)

        gt = np.load(
            gt_path
        ).astype(np.float32)

        noisy = torch.from_numpy(
            noisy
        ).unsqueeze(0)

        gt = torch.from_numpy(
            gt
        ).unsqueeze(0)

        return noisy, gt, filename


# ============================================================
# RECREATE EXACT SAME TRAIN / VAL / TEST SPLIT
# ============================================================

all_files = sorted([
    f
    for f in os.listdir(NOISY_DIR)
    if f.endswith(".npy")
])

all_files = [
    f
    for f in all_files
    if os.path.exists(
        os.path.join(GT_DIR, f)
    )
]

random.Random(SEED).shuffle(all_files)

n = len(all_files)

train_end = int(
    TRAIN_RATIO * n
)

val_end = int(
    (TRAIN_RATIO + VAL_RATIO) * n
)

train_files = all_files[:train_end]

val_files = all_files[
    train_end:val_end
]

test_files = all_files[
    val_end:
]


print("=" * 70)
print("KLA TEST DATASET")
print("=" * 70)

print(f"Total paired images : {len(all_files)}")
print(f"Training            : {len(train_files)}")
print(f"Validation          : {len(val_files)}")
print(f"Testing             : {len(test_files)}")

print("=" * 70)


# ============================================================
# TEST DATALOADER
# ============================================================

test_dataset = KLATestDataset(
    test_files
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# RESIDUAL BLOCK
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

        self.bn1 = nn.BatchNorm2d(
            channels
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(
            channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

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
# DnCNN RESTORER
# ============================================================

class DnCNNRestorer(nn.Module):

    def __init__(
        self,
        num_features=64,
        num_blocks=10
    ):

        super().__init__()

        # Feature extraction

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

        # Residual blocks

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

        # Reconstruction

        self.reconstruction = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            padding=1
        )

        # ×2 upsampling

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

        # Output

        self.output = nn.Conv2d(
            num_features,
            1,
            kernel_size=3,
            padding=1
        )


    def forward(self, x):

        features = self.head(x)

        body = self.body(features)

        body = body + features

        body = self.reconstruction(body)

        out = self.upsample(body)

        out = self.output(out)

        return out


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

config = checkpoint.get(
    "config",
    {}
)

num_features = config.get(
    "num_features",
    64
)

num_blocks = config.get(
    "num_blocks",
    10
)

model = DnCNNRestorer(
    num_features=num_features,
    num_blocks=num_blocks
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


print("\nMODEL")
print("=" * 70)

print(f"Device       : {DEVICE}")
print(f"Features     : {num_features}")
print(f"Residual     : {num_blocks}")
print(
    f"Best epoch   : "
    f"{checkpoint.get('epoch', 'N/A')}"
)

print(
    f"Val PSNR     : "
    f"{checkpoint.get('val_psnr', 0):.4f} dB"
)

print(
    f"Val SSIM     : "
    f"{checkpoint.get('val_ssim', 0):.6f}"
)

print("=" * 70)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    prediction,
    target
):

    prediction = np.clip(
        prediction,
        0.0,
        1.0
    )

    target = np.clip(
        target,
        0.0,
        1.0
    )

    mse = np.mean(
        (prediction - target) ** 2
    )

    psnr = peak_signal_noise_ratio(
        target,
        prediction,
        data_range=1.0
    )

    ssim = structural_similarity(
        target,
        prediction,
        data_range=1.0
    )

    return mse, psnr, ssim


# ============================================================
# TESTING
# ============================================================

print("\n")
print("=" * 70)
print("TESTING DnCNN")
print("=" * 70)

mse_values = []
psnr_values = []
ssim_values = []
inference_times = []

per_image_results = []

# Store a few visual examples

visual_examples = []


with torch.no_grad():

    for index, (
        noisy,
        gt,
        filename
    ) in enumerate(test_loader):

        noisy = noisy.to(
            DEVICE,
            non_blocking=True
        )

        gt = gt.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Inference timing
        # ----------------------------------------------------

        if DEVICE.type == "cuda":

            torch.cuda.synchronize()

        start_time = time.perf_counter()

        prediction = model(
            noisy
        )

        restored = prediction

        print("\n===== SHAPE DEBUG =====")
        print("Noisy input :", noisy.shape)
        print("Model output:", restored.shape)
        print("GT target   :", gt.shape)
        print("=======================\n")

        if DEVICE.type == "cuda":

            torch.cuda.synchronize()

        end_time = time.perf_counter()

        inference_time = (
            end_time - start_time
        )

        # ----------------------------------------------------
        # Convert to NumPy
        # ----------------------------------------------------

        prediction_np = (
            prediction
            .squeeze()
            .cpu()
            .numpy()
        )

        gt_np = (
            gt
            .squeeze()
            .cpu()
            .numpy()
        )

        noisy_np = (
            noisy
            .squeeze()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        mse, psnr, ssim = calculate_metrics(
            prediction_np,
            gt_np
        )

        mse_values.append(mse)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        inference_times.append(
            inference_time
        )

        fname = filename[0]

        per_image_results.append({

            "filename": fname,
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim,
            "inference_time_sec":
                inference_time
        })

        # ----------------------------------------------------
        # Save restored image
        # ----------------------------------------------------

        restored = np.clip(
            prediction_np,
            0.0,
            1.0
        )

        restored_uint8 = (
            restored * 255
        ).astype(np.uint8)

        from PIL import Image

        Image.fromarray(
            restored_uint8
        ).save(
            os.path.join(
                RESTORED_DIR,
                fname.replace(
                    ".npy",
                    ".png"
                )
            )
        )

        # ----------------------------------------------------
        # Select visual examples
        # ----------------------------------------------------

        if index in [0, 1, 2, 3, 4]:

            visual_examples.append({

                "filename": fname,

                "noisy": noisy_np,

                "prediction": restored,

                "gt": np.clip(
                    gt_np,
                    0.0,
                    1.0
                ),

                "psnr": psnr,

                "ssim": ssim
            })

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (index + 1) % 50 == 0
            or
            (index + 1) == len(test_loader)
        ):

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(test_loader)}"
            )


# ============================================================
# FINAL METRICS
# ============================================================

mean_mse = np.mean(
    mse_values
)

mean_psnr = np.mean(
    psnr_values
)

mean_ssim = np.mean(
    ssim_values
)

mean_time = np.mean(
    inference_times
)

median_time = np.median(
    inference_times
)


print("\n")
print("=" * 70)
print("DnCNN TEST RESULTS")
print("=" * 70)

print(
    f"Images evaluated : "
    f"{len(test_files)}"
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
    f"Mean inference   : "
    f"{mean_time * 1000:.2f} ms/image"
)

print(
    f"Median inference : "
    f"{median_time * 1000:.2f} ms/image"
)

print("=" * 70)


# ============================================================
# SAVE PER-IMAGE CSV
# ============================================================

csv_path = os.path.join(
    RESULT_DIR,
    "dncnn_test_results.csv"
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
            "inference_time_sec"
        ]
    )

    writer.writeheader()

    writer.writerows(
        per_image_results
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULT_DIR,
    "dncnn_test_summary.txt"
)

with open(
    summary_path,
    "w"
) as f:

    f.write(
        "KLA DnCNN TEST RESULTS\n"
    )

    f.write(
        "=" * 60 + "\n"
    )

    f.write(
        f"Images evaluated: "
        f"{len(test_files)}\n"
    )

    f.write(
        f"Mean MSE: "
        f"{mean_mse:.8f}\n"
    )

    f.write(
        f"Mean PSNR: "
        f"{mean_psnr:.4f} dB\n"
    )

    f.write(
        f"Mean SSIM: "
        f"{mean_ssim:.6f}\n"
    )

    f.write(
        f"Mean inference time: "
        f"{mean_time * 1000:.2f} ms/image\n"
    )

    f.write(
        f"Median inference time: "
        f"{median_time * 1000:.2f} ms/image\n"
    )


# ============================================================
# VISUAL COMPARISONS
# ============================================================

for example in visual_examples:

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    # Noisy

    axes[0].imshow(
        example["noisy"],
        cmap="gray"
    )

    axes[0].set_title(
        "NoisyLR Input"
    )

    axes[0].axis("off")

    # Restored

    axes[1].imshow(
        example["prediction"],
        cmap="gray"
    )

    axes[1].set_title(
        f"DnCNN Restored\n"
        f"PSNR={example['psnr']:.2f} dB"
    )

    axes[1].axis("off")

    # GT

    axes[2].imshow(
        example["gt"],
        cmap="gray"
    )

    axes[2].set_title(
        f"Ground Truth\n"
        f"SSIM={example['ssim']:.4f}"
    )

    axes[2].axis("off")

    plt.tight_layout()

    output_name = (
        example["filename"]
        .replace(".npy", ".png")
    )

    plt.savefig(
        os.path.join(
            COMPARISON_DIR,
            output_name
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# PSNR DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    psnr_values,
    bins=30
)

plt.xlabel(
    "PSNR (dB)"
)

plt.ylabel(
    "Number of Images"
)

plt.title(
    "DnCNN Test PSNR Distribution"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "dncnn_psnr_distribution.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# SSIM DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    ssim_values,
    bins=30
)

plt.xlabel(
    "SSIM"
)

plt.ylabel(
    "Number of Images"
)

plt.title(
    "DnCNN Test SSIM Distribution"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "dncnn_ssim_distribution.png"
    ),
    dpi=150
)

plt.close()


print("\n")
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    f"CSV results : {csv_path}"
)

print(
    f"Summary     : {summary_path}"
)

print(
    f"Restored    : {RESTORED_DIR}"
)

print(
    f"Comparisons : {COMPARISON_DIR}"
)

print("=" * 70)