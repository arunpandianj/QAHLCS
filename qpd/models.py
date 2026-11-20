"""Encoder, policy heads and critic."""
import math

import torch
import torch.nn as nn

from .config import CFG, IMG, P, SPECS
from .data import mask_to_pix
from .simulator import VQC

class Encoder(nn.Module):
    """Shared perception module: masked image (+ mask channel) -> feature h -> linear classifier."""
    def __init__(self, n_classes, feat=CFG['FEAT']):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        s = IMG // 4
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * s * s, 256), nn.ReLU(), nn.Linear(256, feat))
        self.cls = nn.Linear(feat, n_classes)
    def forward(self, img_u8, mask, aug=False):
        x = (img_u8.float() / 255 - 0.5) / 0.5
        if aug:                                                    # per-sample contrast / brightness jitter
            B = x.shape[0]
            x = x * (1 + 0.2 * (2 * torch.rand(B, 1, 1, 1, device=x.device) - 1)) + 0.1 * (2 * torch.rand(B, 1, 1, 1, device=x.device) - 1)
        m = mask_to_pix(mask)
        h = self.fc(self.conv(torch.cat([x * m, m], 1)))
        return h, self.cls(h)

class QuantumHead(nn.Module):
    def __init__(self, in_dim, out_dim, nq, nl, entangle=True, trainable=True):
        super().__init__()
        self.proj = nn.Linear(in_dim, nq); self.vqc = VQC(nq, nl, entangle, trainable); self.out = nn.Linear(nq, out_dim)
        self.shots, self.p_dep = None, 0.0
    def forward(self, x):
        return self.out(self.vqc(math.pi * torch.tanh(self.proj(x)), self.shots, self.p_dep))

class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__(); self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    def forward(self, x): return self.net(x)

class BottleneckHead(nn.Module):
    """Quantum head with the circuit replaced by tanh. The features lie in [-1, 1] like the circuit outputs, so the same
    perturbations can be applied: Gaussian noise with the finite-shot variance (1 - z^2) / K, and a contraction by
    (1 - 4p/3)^L that mimics the damping of expectation values under depolarising noise in a circuit of L layers."""
    def __init__(self, in_dim, out_dim, nq, nl_equiv=2):
        super().__init__()
        self.proj = nn.Linear(in_dim, nq); self.out = nn.Linear(nq, out_dim)
        self.shots, self.p_dep, self.nl_equiv = None, 0.0, nl_equiv
    def forward(self, x):
        z = torch.tanh(self.proj(x))
        if self.shots: z = z + torch.randn_like(z) * torch.sqrt((1 - z ** 2).clamp(min=0) / self.shots)
        if self.p_dep > 0: z = z * (1 - 4 * self.p_dep / 3) ** self.nl_equiv
        return self.out(z)

def n_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

IN_DIM, OUT_DIM = CFG['FEAT'] + 1 + P + 3, P + 1   # input: features, step fraction, revealed-mask, 3 confidence features; output: P patches + STOP

def make_head(name):
    s = SPECS[name]
    if s['kind'] == 'quantum': return QuantumHead(IN_DIM, OUT_DIM, s['nq'], s['nl'], s.get('entangle', True), s.get('trainable', True))
    if s['kind'] == 'bottleneck': return BottleneckHead(IN_DIM, OUT_DIM, s['nq'])
    if 'hidden' in s: return MLPHead(IN_DIM, OUT_DIM, s['hidden'])
    target = n_params(make_head(s['match']))
    return MLPHead(IN_DIM, OUT_DIM, max(1, round((target - OUT_DIM) / (IN_DIM + 1 + OUT_DIM))))

def make_critic(): return nn.Sequential(nn.Linear(IN_DIM, 64), nn.ReLU(), nn.Linear(64, 1))
