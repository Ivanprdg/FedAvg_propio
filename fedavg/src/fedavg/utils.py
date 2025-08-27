from __future__ import annotations
import os, json, random, time, platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(pref: str = "auto") -> torch.device:
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if pref == "mps" or (pref == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def exp_dir(base: str | Path, **tags: Any) -> Path:
    name = "_".join([f"{k}{v}" for k, v in tags.items()])
    d = Path(base) / f"exp_{now_stamp()}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "emissions").mkdir(exist_ok=True, parents=True)
    return d


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def device_info() -> Dict[str, Any]:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mps_available": torch.backends.mps.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    return info
