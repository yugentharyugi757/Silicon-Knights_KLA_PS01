import os
import numpy as np

GT_DIR = "train/GT"
NOISY_DIR = "train/NoisyLR"

files = sorted([
    f for f in os.listdir(NOISY_DIR)
    if f.endswith(".npy")
])

print("=" * 60)
print("DATASET SHAPE CHECK")
print("=" * 60)

for filename in files[:10]:

    noisy = np.load(
        os.path.join(NOISY_DIR, filename)
    )

    gt = np.load(
        os.path.join(GT_DIR, filename)
    )

    print(
        f"{filename:<30} "
        f"NoisyLR: {noisy.shape} "
        f"GT: {gt.shape}"
    )

print("=" * 60)