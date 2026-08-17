import torch
import numpy as np
import os

from evaluate_dncnn import (
    DnCNNRestorer,
    CHECKPOINT_PATH,
    NOISY_DIR,
    DEVICE
)

# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

config = checkpoint.get("config", {})

num_features = config.get("num_features", 64)
num_blocks = config.get("num_blocks", 10)

model = DnCNNRestorer(
    num_features=num_features,
    num_blocks=num_blocks
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

# ------------------------------------------------------------
# Load one noisy image
# ------------------------------------------------------------

filename = sorted([
    f for f in os.listdir(NOISY_DIR)
    if f.endswith(".npy")
])[0]

image = np.load(
    os.path.join(NOISY_DIR, filename)
).astype(np.float32)

noisy = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

noisy = noisy.to(DEVICE)

# ------------------------------------------------------------
# Model inference
# ------------------------------------------------------------

with torch.no_grad():

    output = model(noisy)

# ------------------------------------------------------------
# Print shapes
# ------------------------------------------------------------

print("=" * 60)
print("MODEL SHAPE VERIFICATION")
print("=" * 60)

print("Input NumPy shape :", image.shape)
print("Input tensor      :", noisy.shape)
print("Output tensor     :", output.shape)

print("=" * 60)