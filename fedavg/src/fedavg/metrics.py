from __future__ import annotations
from typing import Dict, List
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix


class MetricsWriter:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.rows: List[Dict] = []
        self.csv_path = self.out_dir / "metrics_per_round.csv"

    def add_round(self, row: Dict) -> None:
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.csv_path, index=False)

    @staticmethod
    def save_confusion_matrix(model: torch.nn.Module, loader, device, out_dir: Path) -> None:
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                logits = model(x)
                preds = logits.argmax(1).cpu().numpy().tolist()
                y_true.extend(y.numpy().tolist())
                y_pred.extend(preds)
        cm = confusion_matrix(y_true, y_pred)
        np.save(Path(out_dir) / "cm_final.npy", cm)
