"""Checks of the circuit simulator and of the head sizes. Run with: python -m pytest tests -q (PennyLane required)."""
import math

import numpy as np
import pytest
import torch

from qpd.config import DEVICE
from qpd.models import make_head, n_params
from qpd.simulator import VQC

qml = pytest.importorskip("pennylane")


def _circuit(n, L, device, p=0.0):
    @qml.qnode(device)
    def circ(a, w):
        for q in range(n): qml.RY(a[q], wires=q)
        for layer in range(L):
            for q in range(n):
                qml.RY(w[layer, q, 0], wires=q); qml.RZ(w[layer, q, 1], wires=q)
            for i in range(n): qml.CNOT(wires=[i, (i + 1) % n])
            if p > 0:
                for q in range(n): qml.DepolarizingChannel(p, wires=q)
        return [qml.expval(qml.PauliZ(q)) for q in range(n)]
    return circ


def test_exact_expectation_values():
    n, L = 8, 3
    vqc = VQC(n, L).to(DEVICE)
    with torch.no_grad(): vqc.weights.normal_(0, 1)
    ang = torch.rand(4, n, device=DEVICE) * 2 * math.pi
    with torch.no_grad(): mine = vqc(ang).cpu().numpy()
    circ = _circuit(n, L, qml.device("default.qubit", wires=n)); w = vqc.weights.detach().cpu().numpy()
    ref = np.array([np.array(circ(ang[b].cpu().numpy(), w)) for b in range(4)])
    assert np.abs(mine - ref).max() < 1e-4


def test_depolarising_noise_matches_density_matrix():
    n, L, p = 6, 2, 0.05
    vqc = VQC(n, L).to(DEVICE)
    with torch.no_grad(): vqc.weights.normal_(0, 1)
    a = torch.rand(1, n, device=DEVICE) * 2 * math.pi
    with torch.no_grad(): noisy = vqc(a.repeat(20000, 1), p_dep=p).mean(0).cpu().numpy()
    ref = np.array(_circuit(n, L, qml.device("default.mixed", wires=n), p)(a[0].cpu().numpy(), vqc.weights.detach().cpu().numpy()))
    assert np.abs(noisy - ref).max() < 0.03


def test_head_parameter_counts():
    assert n_params(make_head("quantum_4x2")) == 313
    assert n_params(make_head("mlp_300")) == 297
    assert n_params(make_head("bottleneck_4")) == 297
