import torch
import torch.nn as nn


class SRCNN(nn.Module):
    """Classical SRCNN for grayscale super-resolution."""

    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        first_features=64,
        first_kernel_size=9,
        second_features=32,
        second_kernel_size=1,
        third_kernel_size=5,
    ):
        super().__init__()

        self.feature_extraction = nn.Conv2d(
            input_channels,
            first_features,
            kernel_size=first_kernel_size,
            padding=first_kernel_size // 2,
        )
        self.activation_1 = nn.ReLU(inplace=True)

        self.non_linear_mapping = nn.Conv2d(
            first_features,
            second_features,
            kernel_size=second_kernel_size,
            padding=second_kernel_size // 2,
        )
        self.activation_2 = nn.ReLU(inplace=True)

        self.reconstruction = nn.Conv2d(
            second_features,
            output_channels,
            kernel_size=third_kernel_size,
            padding=third_kernel_size // 2,
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.activation_1(x)
        x = self.non_linear_mapping(x)
        x = self.activation_2(x)
        x = self.reconstruction(x)
        return x
