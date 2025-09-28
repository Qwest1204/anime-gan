import torch
import torch.nn as nn
from torch.nn import functional as F
from model.modules import AttentionBlock, ResidualBlock

from torchinfo import summary

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, padding=0),
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),

            ResidualBlock(128, 128),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(128, 64),
            ResidualBlock(64, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),

            ResidualBlock(64, 32),
            ResidualBlock(32, 32),
            nn.GroupNorm(16, 32),

            nn.SiLU(),

            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return torch.tanh(self.layers(x))