import torch
import torch.nn as nn
from torch.nn import functional as F
from modules import AttentionBlock, ResidualBlock

from torchinfo import summary

class Discriminator(nn.Module):
    def __init__(self,):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),

            ResidualBlock(256, 256),

            nn.GroupNorm(32, 256),

            nn.SiLU(),

            nn.Conv2d(256, 8, kernel_size=3, padding=1),

            nn.Conv2d(8, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        # x: (Batch_Size, Channel, Height, Width)

        for layers in self.layers:

            if getattr(layers, 'stride', None) == (2, 2):  # Padding at downsampling should be asymmetric (see #8)
                # Pad: (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom).
                # Pad with zeros on the right and bottom.
                # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Channel, Height + Padding_Top + Padding_Bottom, Width + Padding_Left + Padding_Right) = (Batch_Size, Channel, Height + 1, Width + 1)
                x = F.pad(x, (0, 1, 0, 1))

            x = layers(x)
        return torch.sigmoid(x)