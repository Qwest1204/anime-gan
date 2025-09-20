import torch
import torch.nn as nn
from torch.nn import functional as F
from modules import AttentionBlock, ResidualBlock

from torchinfo import summary

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, padding=0),
            nn.Conv2d(3, 256, kernel_size=3, padding=1),

            ResidualBlock(256, 256),
            AttentionBlock(256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            ResidualBlock(256, 256),
            ResidualBlock(256, 256),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            ResidualBlock(256, 128),
            ResidualBlock(128, 128),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),

            ResidualBlock(128, 64),
            ResidualBlock(64, 64),

            nn.GroupNorm(32, 64),

            nn.SiLU(),

            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return torch.tanh(self.layers(x))

def test():
    x = torch.randn((1, 3, 128, 128))
    model = Generator()
    summary(model, (1, 3, 128, 128))
    out = model(x)
    print(out.shape)

if __name__ == "__main__":
    test()