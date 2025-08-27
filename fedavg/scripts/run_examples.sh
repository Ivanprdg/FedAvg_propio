#!/usr/bin/env bash
set -e

# CIFAR-10 parada por rondas
python -m fedavg --dataset cifar10 --n_clients 10 --partition non_iid_dirichlet --alpha 0.3 --local_epochs 1 --max_rounds 50

# MNIST parada por precisión
python -m fedavg --dataset mnist --n_clients 8 --partition iid --local_epochs 1 --target_test_acc 0.985

# CIFAR-100 parada por energía
python -m fedavg --dataset cifar100 --n_clients 20 --partition non_iid_quantity --skew 0.7 --local_epochs 2 --max_energy_wh 0.8
