import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSRModel(nn.Module):
    def __init__(self, scale: int = 4, num_features: int = 64, num_blocks: int = 8):
        super().__init__()
        self.scale = scale

        self.head = nn.Conv2d(3, num_features, kernel_size=3, padding=1)
        self.body = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                )
                for _ in range(num_blocks)
            ]
        )
        self.tail = nn.Conv2d(num_features, 3 * scale * scale, kernel_size=3, padding=1)
        self.upsample = nn.PixelShuffle(scale)

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        _, _, lr_h, lr_w = pixel_values.shape
        target_size = (lr_h * self.scale, lr_w * self.scale)

        features = torch.relu(self.head(pixel_values))
        for block in self.body:
            features = features + block(features)

        sr = self.tail(features)
        sr = self.upsample(sr)

        base = F.interpolate(
            pixel_values,
            size=target_size,
            mode="bicubic",
            align_corners=False,
        )

        return sr + base
