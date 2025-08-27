from __future__ import annotations
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm


class Client:
    def __init__(self, cid: int, model: torch.nn.Module, loader: DataLoader, device: torch.device,
                 lr: float, weight_decay: float, amp: bool):
        self.id = cid
        self.device = device
        self.model = model
        self.loader = loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        self.amp = amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    def set_state(self, state: dict) -> None:
        self.model.load_state_dict(state)

    def get_state(self) -> dict:
        return self.model.state_dict()

    def num_samples(self) -> int:
        return len(self.loader.dataset)

    def local_train(self, epochs: int = 1) -> float:
        self.model.train()
        total_loss = 0.0
        batches = 0
        for _ in range(epochs):
            for x, y in self.loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                if self.amp and self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        logits = self.model(x)
                        loss = self.criterion(logits, y)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                    loss.backward()
                    self.optimizer.step()
                total_loss += float(loss.detach().cpu())
                batches += 1
        self.model.eval()
        return total_loss / max(1, batches)
