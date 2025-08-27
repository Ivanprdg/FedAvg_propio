import torch
from src.fedavg.aggregator import weighted_average


def test_weighted_average_basic():
    s1 = {"w": torch.tensor([1.0, 1.0])}
    s2 = {"w": torch.tensor([3.0, 5.0])}
    out = weighted_average([(s1, 1), (s2, 3)])
    # (1*1 + 3*3)/4 = (1 + 9)/4 = 2.5 ; (1*1 + 5*3)/4 = (1 + 15)/4 = 4
    assert torch.allclose(out["w"], torch.tensor([2.5, 4.0]))
