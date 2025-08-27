PYTHON ?= python

install:
	$(PYTHON) -m pip install -e .[tests]

lint:
	$(PYTHON) -m pip show flake8 >/dev/null 2>&1 || $(PYTHON) -m pip install flake8
	flake8 src tests

test:
	$(PYTHON) -m pytest -q

run_mnist_example:
	$(PYTHON) -m fedavg --dataset mnist --n_clients 8 --partition iid --local_epochs 1 --max_rounds 5

run_cifar10_example:
	$(PYTHON) -m fedavg --dataset cifar10 --n_clients 10 --partition non_iid_dirichlet --alpha 0.3 --local_epochs 1 --max_rounds 5

run_cifar100_energy_example:
	$(PYTHON) -m fedavg --dataset cifar100 --n_clients 20 --partition non_iid_quantity --skew 0.7 --local_epochs 2 --max_energy_wh 0.8
