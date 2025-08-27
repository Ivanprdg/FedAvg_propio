from __future__ import annotations
from typing import Tuple
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _mnist_transforms():
    # 28x28 -> 28x28, normalización estándar MNIST
    return (
        transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))]),
        transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))]),
    )


def _cifar_transforms(augment: bool = True):
    # 32x32 CIFAR, augs ligeros para train
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return train_tf, test_tf


def get_datasets(name: str, root: str | Path = "./data"):
    name = name.lower()
    if name == "mnist":
        train_tf, test_tf = _mnist_transforms()
        train = datasets.MNIST(root, train=True, download=True, transform=train_tf)
        test = datasets.MNIST(root, train=False, download=True, transform=test_tf)
        num_classes = 10
        in_channels = 1
    elif name == "cifar10":
        train_tf, test_tf = _cifar_transforms(True)
        train = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
        test = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
        num_classes = 10
        in_channels = 3
    elif name == "cifar100":
        train_tf, test_tf = _cifar_transforms(True)
        train = datasets.CIFAR100(root, train=True, download=True, transform=train_tf)
        test = datasets.CIFAR100(root, train=False, download=True, transform=test_tf)
        num_classes = 100
        in_channels = 3
    else:
        raise ValueError("dataset debe ser [mnist|cifar10|cifar100]")
    return train, test, num_classes, in_channels


def make_loader(ds, batch_size: int, num_workers: int, pin: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
