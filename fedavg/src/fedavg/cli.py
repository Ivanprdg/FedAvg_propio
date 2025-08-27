from __future__ import annotations
import argparse, logging, yaml
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch.utils.data import Subset
from .datasets import get_datasets, make_loader
from .models import build_model
from .partition import (
    iid_partition, dirichlet_partition, quantity_skew_partition,
    class_less_partition, dirichlet_partition_min1
)
from .client import Client
from .server import Server
from .metrics import MetricsWriter
from .utils import set_seed, select_device, exp_dir


def build_partitions(train_ds, n_clients: int, ptype: str, alpha: float, skew: float,
                     class_private_pct: float | None = None):
    y = np.array(getattr(train_ds, "targets"))
    if ptype == "iid":
        parts = iid_partition(len(train_ds), n_clients)
    elif ptype == "non_iid_dirichlet":
        parts = dirichlet_partition(y, n_clients, alpha)
    elif ptype == "non_iid_dirichlet_min1":
        parts = dirichlet_partition_min1(y, n_clients, alpha)
    elif ptype == "non_iid_quantity":
        parts = quantity_skew_partition(len(train_ds), n_clients, skew)
    elif ptype == "class_less":
        if class_private_pct is None:
            raise ValueError("class_less requiere --class_private_pct")
        parts = class_less_partition(y, n_clients, alpha, class_private_pct)
    else:
        raise ValueError("partition debe ser [iid|non_iid_dirichlet|non_iid_dirichlet_min1|non_iid_quantity|class_less]")

    return [Subset(train_ds, idx.tolist()) for idx in parts]




def main(argv: List[str] | None = None):
    ap = argparse.ArgumentParser(prog="fedavg", description="FedAvg clásico standalone")
    ap.add_argument("--config", type=str, default=None, help="Ruta a YAML con configuración por defecto")
    ap.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10", "cifar100"])
    ap.add_argument("--n_clients", type=int, default=10)
    ap.add_argument("--participation", type=float, default=1.0, help="Fracción de clientes por ronda C")
    ap.add_argument("--partition", type=str, default="non_iid_dirichlet",
                choices=["iid", "non_iid_dirichlet", "non_iid_quantity", "class_less", "non_iid_dirichlet_min1"])
    ap.add_argument("--alpha", type=float, default=0.3, help="Concentración Dirichlet")
    ap.add_argument("--skew", type=float, default=0.7, help="Grado de quantity-skew")
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--max_rounds", type=int, default=50)
    ap.add_argument("--target_test_acc", type=float, default=None)
    ap.add_argument("--max_energy_wh", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    ap.add_argument("--save_dir", type=str, default="results")
    ap.add_argument("--class_private_pct", type=float, default=0.2,
                help="Fracción de clases privadas por cliente (class_less)")
    args = ap.parse_args(argv)

    # Cargar config YAML si procede
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if getattr(args, k, None) is not None:
                setattr(args, k, v)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(args.seed)
    device = select_device(args.device)

    # Datos y modelo
    train_ds, test_ds, num_classes, in_ch = get_datasets(args.dataset)
    pin = (device.type == "cuda")
    test_loader = make_loader(test_ds, batch_size=512, num_workers=args.num_workers, pin=pin)
    model = build_model(args.dataset, num_classes, in_ch).to(device)

    # Particiones y clientes
    subsets = build_partitions(
        train_ds,
        args.n_clients,
        args.partition,
        args.alpha,
        args.skew,
        args.class_private_pct
    )

    clients = []
    from torch.utils.data import DataLoader
    for cid, sub in enumerate(subsets):
        loader = DataLoader(sub, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=pin)
        cli = Client(cid, model.__class__(**{"num_classes": num_classes}).to(device),
                     loader, device, lr=args.lr, weight_decay=args.weight_decay, amp=args.amp)
        clients.append(cli)

    # Carpeta de experimento
    tags = dict(ds=args.dataset, n=args.n_clients, part=args.partition, a=args.alpha, skew=args.skew, seed=args.seed)
    if args.partition == "class_less":
        tags["cp"] = args.class_private_pct
    out = exp_dir(args.save_dir, **tags)


    # Servidor (orquestador)
    server = Server(model=model, clients=clients, test_loader=test_loader, device=device, out_dir=out)
    server.run(max_rounds=args.max_rounds,
               local_epochs=args.local_epochs,
               participation=args.participation,
               target_test_acc=args.target_test_acc,
               max_energy_wh=args.max_energy_wh)

    # Matriz de confusión final
    from .metrics import MetricsWriter
    MetricsWriter.save_confusion_matrix(model, test_loader, device, out)

    print(f"✔ Fin. Resultados en: {out}")


if __name__ == "__main__":
    main()
