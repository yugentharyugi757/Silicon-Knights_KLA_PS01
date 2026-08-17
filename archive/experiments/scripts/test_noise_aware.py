import torch

from models.noise_aware_dncnn import NoiseAwareDnCNN


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = NoiseAwareDnCNN(
    num_features=64,
    num_blocks=10
).to(device)

x = torch.randn(
    2,
    2,
    128,
    128
).to(device)

with torch.no_grad():

    y = model(x)

print("=" * 60)
print("NOISE-AWARE MODEL SHAPE TEST")
print("=" * 60)

print("Input :", x.shape)
print("Output:", y.shape)

print("=" * 60)