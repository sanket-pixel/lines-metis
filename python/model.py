"""Standard DeepLSD inference CNN, adapted from cvg/DeepLSD (MIT).

Input: grayscale float32 NCHW in [0, 1]. Outputs: normalized distance and
orientation fraction, each N1HW. Denormalization/extraction stay on the CPU.
Checkpoint layer names and Conv -> ReLU -> BatchNorm ordering are preserved.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

CHECKPOINT = Path(__file__).resolve().parents[1] / 'weights/torch/deeplsd.pt'


def _block(in_channels, hidden_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_channels),
        nn.Conv2d(hidden_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
    )


def _head(activation):
    return nn.Sequential(*_block(64, 64, 64), nn.Conv2d(64, 1, 1), activation)


class VGGUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(2, 2)
        self.block1 = _block(1, 64, 64)
        self.block2 = _block(64, 128, 128)
        self.block3 = _block(128, 256, 256)
        self.block4 = _block(256, 512, 512)
        self.deblock4 = _block(512, 256, 256)
        self.deblock3 = _block(512, 256, 128)
        self.deblock2 = _block(256, 128, 64)
        self.deblock1 = _block(128, 64, 64)

    def forward(self, image):
        features = [self.block1(image)]
        for block in (self.block2, self.block3, self.block4):
            features.append(block(self.pool(features[-1])))
        out = self.deblock4(features[-1])
        for block, skip in zip(
            (self.deblock3, self.deblock2, self.deblock1), reversed(features[:-1])
        ):
            out = block(torch.cat([
                F.interpolate(out, skip.shape[2:4], mode='bilinear', align_corners=False),
                skip,
            ], dim=1))
        return out


class DeepLSD(nn.Module):
    """Inference-only standard model; no training/configuration framework."""
    def __init__(self):
        super().__init__()
        self.backbone = VGGUNet()
        self.df_head = _head(nn.ReLU())
        self.angle_head = _head(nn.Sigmoid())

    def forward(self, image):
        features = self.backbone(image)
        return self.df_head(features), self.angle_head(features)


def load_model(checkpoint=CHECKPOINT):
    """Strictly load tensor-only MegaDepth weights on CPU (see setup.sh)."""
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = DeepLSD()
    model.load_state_dict(state, strict=True)
    return model.eval()
