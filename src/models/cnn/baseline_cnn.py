"""
Simple 3D CNN baseline for lung nodule 3-class classification. No attention,
no multi-scale fusion, no quantum components -- this establishes the
performance floor that the eventual hybrid architecture needs to beat.
"""

import torch
import torch.nn as nn


class BaselineCNN3D(nn.Module):
    """
    A straightforward 3D CNN: three conv blocks (conv -> batchnorm -> relu ->
    maxpool), then a small classifier head. Input: (B, 1, 32, 64, 64).
    """

    def __init__(self, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 1 -> 16 channels
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),  # (32,64,64) -> (16,32,32)

            # Block 2: 16 -> 32 channels
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),  # (16,32,32) -> (8,16,16)

            # Block 3: 32 -> 64 channels
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),  # (8,16,16) -> (4,8,8)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # (B, 64, 4, 8, 8) -> (B, 64, 1, 1, 1)
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = BaselineCNN3D(num_classes=3)
    dummy_input = torch.randn(2, 1, 32, 64, 64)  # batch of 2
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")