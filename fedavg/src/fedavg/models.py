from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTCNN(nn.Module):
    """CNN ligera para MNIST (1x28x28)."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*7*7, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.conv(x))


class SmallVGG(nn.Module):
    """VGG-like pequeña para CIFAR (3x32x32)."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(dataset: str, num_classes: int, in_ch: int):
    if dataset.lower() == "mnist":
        return MNISTCNN(num_classes)
    else:
        return SmallVGG(num_classes)
