from __future__ import annotations
from typing import List
import numpy as np
from collections import defaultdict


def iid_partition(n_samples: int, n_clients: int) -> List[np.ndarray]:
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    return np.array_split(idx, n_clients)


def dirichlet_partition(labels: np.ndarray, n_clients: int, alpha: float) -> List[np.ndarray]:
    # Label-skew por Dirichlet por clase
    n_classes = int(labels.max() + 1)
    idx_by_class = [np.where(labels == c)[0] for c in range(n_classes)]
    for c in range(n_classes):
        np.random.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        counts = np.random.dirichlet(alpha=[alpha]*n_clients)
        # repartir índices de esta clase según counts
        n_c = len(idx_by_class[c])
        splits = (np.cumsum(counts) * n_c).astype(int)
        splits = np.clip(splits, 0, n_c)
        prev = 0
        parts = []
        for s in splits:
            parts.append(idx_by_class[c][prev:s])
            prev = s
        # ajustar número de partes == n_clients
        while len(parts) < n_clients:
            parts.append(np.array([], dtype=int))
        for i in range(n_clients):
            client_indices[i].extend(parts[i].tolist())
    return [np.array(sorted(ci), dtype=int) for ci in client_indices]


def quantity_skew_partition(n_samples: int, n_clients: int, skew: float) -> List[np.ndarray]:
    # Quantity-skew: algunos clientes reciben más muestras (skew en [0,1))
    # construimos proporciones ~ Dirichlet sesgada
    base = np.ones(n_clients)
    # sesgo: aumentar peso del primer porcentaje de clientes
    heavy = max(1, int(n_clients * skew))
    base[:heavy] = 5.0
    props = np.random.dirichlet(base)
    counts = (props * n_samples).astype(int)
    # ajustar suma
    diff = n_samples - counts.sum()
    counts[0] += diff
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    splits = []
    start = 0
    for c in counts:
        splits.append(idx[start:start+c])
        start += c
    return splits


def dirichlet_partition_min1(labels: np.ndarray, n_clients: int, alpha: float) -> List[np.ndarray]:
    """Tu Dirichlet con garantía de ≥1 muestra por cliente y clase (calc. por clase)."""
    n_classes = int(labels.max() + 1)
    idx_by_class = [np.where(labels == c)[0].tolist() for c in range(n_classes)]
    client_indices = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        idxs = idx_by_class[c]
        np.random.shuffle(idxs)
        N = len(idxs)
        props = np.random.dirichlet(alpha * np.ones(n_clients))
        raw = props * N
        counts = np.floor(raw).astype(int)
        counts[counts == 0] = 1  # ≥1 por cliente
        residual = raw - np.floor(raw)
        total = counts.sum()
        while total < N:
            i = np.argmax(residual); counts[i] += 1; residual[i] = 0; total += 1
        while total > N:
            valid = counts > 1
            j = np.argmin(np.where(valid, residual, np.inf))
            counts[j] -= 1; residual[j] = 1; total -= 1
        start = 0
        for cid, cnt in enumerate(counts):
            if cnt > 0:
                split = idxs[start:start+cnt]
                client_indices[cid].extend(split)
                start += cnt
    return [np.array(sorted(ci), dtype=int) for ci in client_indices]

def class_less_partition(labels: np.ndarray, n_clients: int, alpha: float, private_pct: float) -> List[np.ndarray]:
    """
    Replica tu 'class_less': oculta un % de clases (privadas); reparte por Dirichlet
    solo entre clientes que "ven" la clase; arregla clientes vacíos moviendo
    muestras de clases compartidas desde donantes.
    """
    n_classes = int(labels.max() + 1)
    class_indices = [np.where(labels == c)[0] for c in range(n_classes)]
    all_classes = set(range(n_classes))

    # Elegir clases privadas globales
    num_private = min(int(n_classes * private_pct), n_classes)
    private_classes = list(np.random.choice(list(all_classes), size=num_private, replace=False))
    shared_classes = list(all_classes - set(private_classes))

    # Asignar clases visibles por cliente (round-robin para privadas + picks aleatorios de compartidas)
    client_classes = [set() for _ in range(n_clients)]
    for k, c in enumerate(private_classes):
        client_classes[k % n_clients].add(c)
    if shared_classes:
        for i in range(n_clients):
            k = np.random.randint(1, len(shared_classes) + 1)
            picks = set(np.random.choice(shared_classes, size=k, replace=False))
            client_classes[i].update(picks)

    # Asegurar cobertura de compartidas
    union = set().union(*client_classes)
    missing = set(shared_classes) - union
    for c in missing:
        candidates = [i for i, s in enumerate(client_classes) if len(s) < n_classes - 1]
        chosen = np.random.choice(candidates)
        client_classes[chosen].add(c)

    # Evitar que alguien tenga todas las clases
    for i, s in enumerate(client_classes):
        if len(s) == n_classes:
            compartidas = [c for c in s if sum(c in s2 for s2 in client_classes) > 1]
            if compartidas:
                drop = np.random.choice(compartidas)
                s.remove(drop)

    # Reparto por clase → Dirichlet entre clientes elegibles
    client_indices = [[] for _ in range(n_clients)]
    for c, idxs in enumerate(class_indices):
        idxs = idxs.copy()
        np.random.shuffle(idxs)
        elig = [i for i, s in enumerate(client_classes) if c in s] or list(range(n_clients))
        props = np.random.dirichlet(alpha * np.ones(len(elig)))
        cuts = (np.cumsum(props) * len(idxs)).astype(int)[:-1]
        splits = np.split(idxs, cuts)
        for cli, part in zip(elig, splits):
            client_indices[cli].extend(part.tolist())

    # Fix clientes vacíos moviendo SOLO compartidas desde donantes
    sizes = [len(li) for li in client_indices]
    empties = [i for i, L in enumerate(sizes) if L == 0]
    if empties:
        targets = labels  # para conocer la clase de cada índice
        shared_set = set(shared_classes)
        donors = [i for i, L in enumerate(sizes) if L > 1]
        for cli in empties:
            moved = False
            np.random.shuffle(donors)
            for d in donors:
                for k, src_idx in enumerate(client_indices[d]):
                    if int(targets[src_idx]) in shared_set:
                        client_indices[cli].append(src_idx)
                        del client_indices[d][k]
                        moved = True
                        if len(client_indices[d]) <= 1:
                            donors.remove(d)
                        break
                if moved: break
            if not moved:
                raise RuntimeError("No se encontró muestra compartida para cliente vacío (caso extremo)")

    return [np.array(sorted(ci), dtype=int) for ci in client_indices]
