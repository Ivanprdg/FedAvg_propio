import numpy as np
from src.fedavg.partition import iid_partition, dirichlet_partition, quantity_skew_partition


def test_iid_partition_covers_all():
    n = 100
    parts = iid_partition(n, 5)
    assert sum(len(p) for p in parts) == n
    concat = np.concatenate(parts)
    assert len(np.unique(concat)) == n


def test_dirichlet_partition_shapes():
    y = np.array([0, 1, 0, 1, 2, 2, 2, 1, 0, 2])
    parts = dirichlet_partition(y, 3, alpha=0.5)
    assert len(parts) == 3
    assert sum(len(p) for p in parts) == len(y)


def test_quantity_skew_partition_sum():
    n = 123
    parts = quantity_skew_partition(n, 7, skew=0.5)
    assert sum(len(p) for p in parts) == n
