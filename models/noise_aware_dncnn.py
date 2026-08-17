import torch
import torch.nn as nn


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

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

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
# NOISE-AWARE DnCNN
# ============================================================

class NoiseAwareDnCNN(nn.Module):

    def __init__(
        self,
        num_features=64,
        num_blocks=10
    ):

        super().__init__()

        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------
        # Channel 0 = noisy LR image
        # Channel 1 = estimated noise sigma map
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
        # RESIDUAL BODY
        # ----------------------------------------------------

        blocks = []

        for _ in range(num_blocks):

            blocks.append(
                ResidualBlock(num_features)
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
        # 2X UPSAMPLING
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
        # ----------------------------------------------------

        self.output = nn.Conv2d(
            num_features,
            1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        features = self.head(x)

        body = self.body(features)

        # Global residual connection
        body = body + features

        body = self.reconstruction(body)

        out = self.upsample(body)

        out = self.output(out)

        return out