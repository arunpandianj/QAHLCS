"""MedMNIST loading and patch-grid utilities."""
from pathlib import Path

import medmnist
import numpy as np
import torch
from medmnist import INFO

from .config import DEVICE, G, IMG, P, PS


def find_medmnist_root(flag):
    """Use a local copy of the .npz file (e.g. a Kaggle input dataset) if one exists; otherwise download."""
    fn = f'{flag}.npz' if IMG == 28 else f'{flag}_{IMG}.npz'
    for base in ('/kaggle/input', str(Path.home() / '.medmnist')):
        if Path(base).exists():
            for p in Path(base).rglob(fn): return str(p.parent)
    return None

def load_medmnist(flag, splits=('train', 'val', 'test')):
    info = INFO[flag]; Cls = getattr(medmnist, info['python_class'])
    out = {}; root = find_medmnist_root(flag)
    for sp in splits:
        kw = dict(root=root, download=False) if root else dict(download=True)
        try:
            ds = Cls(split=sp, size=IMG, **kw)
        except TypeError:
            ds = Cls(split=sp, **kw)
        x = np.asarray(ds.imgs)
        if x.ndim == 3: x = np.repeat(x[..., None], 3, axis=3)
        x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()          # uint8, N x 3 x H x W
        y = torch.from_numpy(np.asarray(ds.labels).reshape(-1)).long()
        out[sp] = (x.to(DEVICE), y.to(DEVICE))
    return out, len(info['label']), info['label']

def mask_to_pix(mask):                                                     # (B,P) -> (B,1,H,W)
    return mask.view(-1, 1, G, G).repeat_interleave(PS, 2).repeat_interleave(PS, 3)

RASTER = list(range(P))
_c = (G - 1) / 2
CENTER = sorted(range(P), key=lambda i: ((i // G - _c) ** 2 + (i % G - _c) ** 2, i))
