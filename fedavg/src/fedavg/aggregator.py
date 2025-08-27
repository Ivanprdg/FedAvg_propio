from __future__ import annotations
from typing import Dict, List, Tuple
import torch


def weighted_average(states_and_counts: List[Tuple[dict, int]], device: torch.device | None = None) -> dict:
    total = sum(n for _, n in states_and_counts) or 1
    avg = {}
    for sd, n in states_and_counts:
        w = n / total
        for k, v in sd.items():
            t = v
            if device is not None:
                t = t.to(device)
            if k not in avg:
                avg[k] = t * w
            else:
                avg[k] += t * w
    # volver a CPU para estado portable
    return {k: v.detach().cpu() for k, v in avg.items()}
