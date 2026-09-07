# ml/model.py
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.15)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class MathSymbolCNN(nn.Module):
    def __init__(self, num_classes: int = 52):
        super().__init__()
        # Entrada: [B, 1, 32, 32]
        self.layer1 = ConvBlock(1, 32)    # -> [B, 32, 16, 16]
        self.layer2 = ConvBlock(32, 64)   # -> [B, 64, 8, 8]
        self.layer3 = ConvBlock(64, 128)  # -> [B, 128, 4, 4]
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.classifier(x)