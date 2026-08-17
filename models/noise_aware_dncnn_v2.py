import torch
import torch.nn as nn


# ============================================================
# RESIDUAL BLOCK V2
# ============================================================
# V2 removes BatchNorm to preserve image and noise statistics
# that are important for accurate image restoration and noise
# estimation. BatchNorm normalization can interfere with the
# model's ability to learn task-specific feature statistics.
# ============================================================

class ResidualBlockV2(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out = out + residual
        out = self.relu(out)

        return out


# ============================================================
# NOISE-AWARE DnCNN V2
# ============================================================
# V2 Architecture ablation: removes BatchNorm from residual blocks.
# All other aspects remain identical to V1:
# - 2-channel input (noisy image + sigma map)
# - 64 features
# - 10 residual blocks
# - Global residual connection
# - 2x PixelShuffle upsampling
# - Single-channel output
# ============================================================

class NoiseAwareDnCNNV2(nn.Module):

    def __init__(
        self,
        num_features=64,
        num_blocks=10
    ):

        super().__init__()

        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------
        # Channel 0 = noisy low-resolution grayscale image
        # Channel 1 = estimated noise sigma map
        # Input shape: [B, 2, 128, 128]
        # ----------------------------------------------------

        self.head = nn.Sequential(

            nn.Conv2d(
                2,
                num_features,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )

        # ----------------------------------------------------
        # RESIDUAL BODY (V2: BatchNorm removed)
        # ----------------------------------------------------

        blocks = []

        for _ in range(num_blocks):

            blocks.append(
                ResidualBlockV2(num_features)
            )

        self.body = nn.Sequential(*blocks)

        # ----------------------------------------------------
        # RECONSTRUCTION
        # ----------------------------------------------------

        self.reconstruction = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # 2X UPSAMPLING WITH PixelShuffle
        # ----------------------------------------------------

        self.upsample = nn.Sequential(

            nn.Conv2d(
                num_features,
                num_features * 4,
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(2),

            nn.ReLU(inplace=True)
        )

        # ----------------------------------------------------
        # OUTPUT
        # Output shape: [B, 1, 256, 256]
        # ----------------------------------------------------

        self.output = nn.Conv2d(
            num_features,
            1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        # Initial feature extraction
        features = self.head(x)

        # Deep residual feature extraction
        body = self.body(features)

        # Global residual connection
        body = body + features

        # Reconstruction layer
        body = self.reconstruction(body)

        # 2x upsampling
        out = self.upsample(body)

        # Final single-channel output
        out = self.output(out)

        return out
