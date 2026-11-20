"""Batched exact state-vector simulator of the policy circuit, with shot and depolarising-noise models."""
import math

import numpy as np
import torch
import torch.nn as nn

from .config import DEVICE


class VQC(nn.Module):
    """Exact differentiable state-vector simulator (batched, GPU).
    Circuit: RY angle embedding -> L x [RY, RZ on every qubit, CNOT ring] -> <Z_i> on every qubit.
    Optional deployment noise: finite measurement shots, and per-layer single-qubit depolarising
    noise simulated with Monte-Carlo Pauli-error trajectories (one trajectory per sample)."""
    def __init__(self, n_qubits, n_layers, entangle=True, trainable=True):
        super().__init__()
        self.n, self.L = n_qubits, n_layers
        self.weights = nn.Parameter(0.3 * torch.randn(n_layers, n_qubits, 2), requires_grad=trainable)
        N = 2 ** n_qubits; idx = np.arange(N); f = idx.copy()
        for i in (range(n_qubits) if entangle else []):
            c, t = i, (i + 1) % n_qubits
            cbit = (f >> (n_qubits - 1 - c)) & 1
            f = f ^ (cbit << (n_qubits - 1 - t))
        perm = np.empty(N, dtype=np.int64); perm[f] = idx
        z = np.array([[1 - 2 * ((b >> (n_qubits - 1 - q)) & 1) for q in range(n_qubits)] for b in range(N)], dtype=np.float32)
        paulis = torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=torch.complex64)
        self.register_buffer('perm', torch.from_numpy(perm))
        self.register_buffer('zmat', torch.from_numpy(z))
        self.register_buffer('paulis', paulis)

    def _apply1(self, psi, U, q):
        B = psi.shape[0]
        x = psi.reshape(B, 2 ** q, 2, 2 ** (self.n - 1 - q))
        y = torch.einsum('ij,bajc->baic', U, x) if U.dim() == 2 else torch.einsum('bij,bajc->baic', U, x)
        return y.reshape(B, -1)

    def forward(self, angles, shots=None, p_dep=0.0):
        B, n = angles.shape[0], self.n
        c, s = torch.cos(angles / 2), torch.sin(angles / 2)
        psi = torch.ones(B, 1, device=angles.device)
        for q in range(n):
            v = torch.stack([c[:, q], s[:, q]], 1)
            psi = (psi.unsqueeze(2) * v.unsqueeze(1)).reshape(B, -1)
        psi = psi.to(torch.complex64)
        for l in range(self.L):
            th, ph = self.weights[l, :, 0], self.weights[l, :, 1]
            cc, ss = torch.cos(th / 2), torch.sin(th / 2)
            em = torch.complex(torch.cos(ph / 2), -torch.sin(ph / 2)); ep = em.conj()
            U = torch.stack([torch.stack([em * cc, -em * ss], -1), torch.stack([ep * ss, ep * cc], -1)], -2)  # (n,2,2) = RZ.RY
            for q in range(n):
                psi = self._apply1(psi, U[q], q)
            psi = psi[:, self.perm]
            if p_dep > 0:
                for q in range(n):
                    r = torch.rand(B, device=psi.device)
                    k = torch.where(r < p_dep, torch.randint(1, 4, (B,), device=psi.device), torch.zeros(B, dtype=torch.long, device=psi.device))
                    psi = self._apply1(psi, self.paulis[k], q)
        probs = psi.real ** 2 + psi.imag ** 2
        if shots is not None:
            probs = probs / probs.sum(1, keepdim=True)
            idx = torch.multinomial(probs, shots, replacement=True)
            counts = torch.zeros_like(probs).scatter_add_(1, idx, torch.ones_like(idx, dtype=probs.dtype))
            probs = counts / shots
        return probs @ self.zmat

def verify_simulator():
    """Cross-check the simulator against PennyLane (exact values, shots, depolarising noise)."""
    rep = {}
    try:
        import pennylane as qml
    except Exception as e:
        print('PennyLane not available, skipping verification:', e); return rep
    def build(n, L, dev, p=0.0):
        @qml.qnode(dev)
        def circ(a, w):
            for q in range(n): qml.RY(a[q], wires=q)
            for l in range(L):
                for q in range(n):
                    qml.RY(w[l, q, 0], wires=q); qml.RZ(w[l, q, 1], wires=q)
                for i in range(n): qml.CNOT(wires=[i, (i + 1) % n])
                if p > 0:
                    for q in range(n): qml.DepolarizingChannel(p, wires=q)
            return [qml.expval(qml.PauliZ(q)) for q in range(n)]
        return circ
    n, L = 8, 3
    vqc = VQC(n, L).to(DEVICE)
    with torch.no_grad(): vqc.weights.normal_(0, 1)
    ang = torch.rand(4, n, device=DEVICE) * 2 * math.pi
    with torch.no_grad(): mine = vqc(ang).cpu().numpy()
    circ = build(n, L, qml.device('default.qubit', wires=n)); w = vqc.weights.detach().cpu().numpy()
    ref = np.array([np.array(circ(ang[b].cpu().numpy(), w)) for b in range(4)])
    rep['max_abs_diff_exact'] = float(np.abs(mine - ref).max())
    with torch.no_grad(): sh = vqc(ang[:1].repeat(1, 1), shots=200000).cpu().numpy()
    rep['max_abs_diff_shots_200k'] = float(np.abs(sh - ref[:1]).max())
    n2, L2, p = 6, 2, 0.05
    v2 = VQC(n2, L2).to(DEVICE)
    with torch.no_grad(): v2.weights.normal_(0, 1)
    a2 = (torch.rand(1, n2, device=DEVICE) * 2 * math.pi)
    with torch.no_grad(): noisy = v2(a2.repeat(20000, 1), p_dep=p).mean(0).cpu().numpy()
    ref2 = np.array(build(n2, L2, qml.device('default.mixed', wires=n2), p)(a2[0].cpu().numpy(), v2.weights.detach().cpu().numpy()))
    rep['max_abs_diff_depolarising_20k_traj'] = float(np.abs(noisy - ref2).max())
    print('Simulator verification vs PennyLane:', rep)
    return rep
