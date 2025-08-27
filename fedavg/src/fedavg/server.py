from __future__ import annotations
from typing import List, Tuple
import math, time, logging, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from .aggregator import weighted_average
from .metrics import MetricsWriter
from .utils import exp_dir, save_json, device_info
from .energy import round_tracker, read_round_energy_wh, clean_emissions_csv


logger = logging.getLogger("fedavg")


def accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(1, total)


class Server:
    def __init__(self, model: torch.nn.Module, clients, test_loader: DataLoader,
                 device: torch.device, out_dir: Path):
        self.model = model
        self.clients = clients
        self.test_loader = test_loader
        self.device = device
        self.out_dir = Path(out_dir)
        self.metrics = MetricsWriter(self.out_dir)
        self.energy_dir = self.out_dir / "emissions"
        self.energy_cum_wh = 0.0

    def run(self, max_rounds: int, local_epochs: int, participation: float,
            target_test_acc: float | None, max_energy_wh: float | None) -> None:
        n = len(self.clients)
        m = max(1, int(math.ceil(participation * n)))
        best_acc = 0.0

        for rnd in range(1, max_rounds + 1):
            # seleccionar participantes
            selected = random.sample(self.clients, m)
            for c in selected:
                c.set_state(self.model.state_dict())

            # medición de energía por ronda
            pname = f"round_{rnd:04d}"
            tracker = round_tracker(self.energy_dir, pname)
            tracker.start()
            t0 = time.time()

            # entrenamiento local
            updates: List[Tuple[dict, int, float]] = []
            for c in selected:
                train_loss = c.local_train(epochs=local_epochs)
                updates.append((c.get_state(), c.num_samples(), train_loss))

            # stop energy tracker
            tracker.stop()

            d_wh = read_round_energy_wh(self.energy_dir, pname)
            clean_emissions_csv(self.energy_dir, pname)
            self.energy_cum_wh += d_wh

            # agregación FedAvg (ponderado por muestras)
            states_and_counts = [(sd, n) for (sd, n, _) in updates]
            new_state = weighted_average(states_and_counts, device=self.device)
            self.model.load_state_dict(new_state)

            # evaluación
            test_acc = accuracy(self.model, self.test_loader, self.device)
            best_acc = max(best_acc, test_acc)
            train_loss_avg = float(np.mean([tl for (_, _, tl) in updates])) if updates else 0.0

            row = {
                "round": rnd,
                "participating_clients": m,
                "train_loss": train_loss_avg,
                "test_acc": test_acc,
                "energy_round_Wh": d_wh,
                "energy_cum_Wh": self.energy_cum_wh,
                "duration_s": time.time() - t0,
            }
            self.metrics.add_round(row)
            logger.info(f"[r{rnd:03d}] acc={test_acc:.4f} ΔE={d_wh:.3f}Wh ΣE={self.energy_cum_wh:.3f}Wh")

            # criterios de parada
            if target_test_acc is not None and test_acc >= target_test_acc:
                logger.info(f"Parada por precisión objetivo: {test_acc:.4f} ≥ {target_test_acc:.4f}")
                break
            if max_energy_wh is not None and self.energy_cum_wh >= max_energy_wh:
                logger.info(f"Parada por presupuesto energético: {self.energy_cum_wh:.3f}Wh ≥ {max_energy_wh:.3f}Wh")
                break

        # resumen
        summary = {
            "best_acc": best_acc,
            "rounds_executed": rnd,
            "energy_total_wh": self.energy_cum_wh,
            "device": str(self.device),
            **device_info(),
        }
        save_json(self.out_dir / "summary.json", summary)
