"""Global configuration. Values can be overridden through environment variables set by run_experiments.py."""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

QUICK = os.environ.get("QPD_QUICK", "0") == "1"
_OV = json.loads(os.environ.get("QPD_OVERRIDES", "{}"))

CFG = dict(
    DATASETS=["pathmnist", "dermamnist", "bloodmnist"],
    SEEDS=[0, 1, 2, 3, 4],
    LAMBDAS=[0.005, 0.015, 0.04],                     # cost per glimpse
    POLICIES=["quantum_4x2", "mlp_300", "bottleneck_4"],
    IMG_SIZE=int(os.environ.get("QPD_IMG", "28")), GRID=4, T_MAX=8, FEAT=32,
    BALANCED=True,
    THRESHOLDS=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    PRE_STEPS=6000, PRE_BS=256, PRE_LR=1e-3,
    BC_UPDATES=300, BC_LR=3e-3, BC_SMOOTH=0.1,
    CRIT_WARM=60, RL_UPDATES=800, RL_BS=512, RL_LR=5e-4,
    VAL_EVERY=100, VAL_N=2000,
    ENT_START=0.01, ENT_END=0.002, GRAD_CLIP=1.0,
    EVAL_BS=1024,
    NOISE_SHOTS=[8, 32, 128, 512, 2048], NOISE_DEP=[0.001, 0.003, 0.01, 0.03],
    ABLATION=False, ABL_DATASET="pathmnist", ABL_LAMBDA=0.015, ABL_CIRCUITS=[],
)
EXT = dict(
    SEEDS=[0, 1, 2], LAM=0.015, STAGES=["main"],
    HEADS=["quantum_4x2", "mlp_300", "bottleneck_4"],
    CIRCUIT_HEADS=["quantum_2x2", "quantum_4x1", "quantum_4x3", "quantum_6x2", "quantum_8x2", "quantum_8x3",
                   "quantum_frozen_4x2", "quantum_noent_4x2", "mlp_620", "bottleneck_8", "mlp_h16", "mlp_h64"],
    TUNE_LR=[2e-4, 1.5e-3],
)
SPECS = {
    "quantum_4x2": dict(kind="quantum", nq=4, nl=2), "mlp_300": dict(kind="mlp", match="quantum_4x2"),
    "bottleneck_4": dict(kind="bottleneck", nq=4), "quantum_8x3": dict(kind="quantum", nq=8, nl=3),
    "mlp_620": dict(kind="mlp", match="quantum_8x3"), "bottleneck_8": dict(kind="bottleneck", nq=8),
    "mlp_h16": dict(kind="mlp", hidden=16), "mlp_h64": dict(kind="mlp", hidden=64),     # larger classical heads
    "quantum_frozen_4x2": dict(kind="quantum", nq=4, nl=2, trainable=False),   # random fixed circuit
    "quantum_noent_4x2": dict(kind="quantum", nq=4, nl=2, entangle=False),     # no CNOT gates
}
for _nq, _nl in [(2, 2), (4, 1), (4, 3), (6, 2), (8, 2), (8, 3)]:
    SPECS.setdefault(f"quantum_{_nq}x{_nl}", dict(kind="quantum", nq=_nq, nl=_nl))

if QUICK:   # tiny smoke-test configuration
    CFG.update(DATASETS=["pathmnist"], SEEDS=[0], LAMBDAS=[0.015], PRE_STEPS=300, RL_UPDATES=60, BC_UPDATES=30,
               CRIT_WARM=10, VAL_EVERY=30, VAL_N=300, NOISE_SHOTS=[32, 512], NOISE_DEP=[0.01])
    EXT.update(SEEDS=[0], CIRCUIT_HEADS=["quantum_frozen_4x2", "quantum_noent_4x2", "mlp_620"], TUNE_LR=[1.5e-3])
for _k, _v in _OV.get("CFG", {}).items(): CFG[_k] = _v
for _k, _v in _OV.get("EXT", {}).items(): EXT[_k] = _v

SIG = hashlib.md5(json.dumps({k: CFG[k] for k in ["IMG_SIZE", "GRID", "T_MAX", "FEAT", "BALANCED", "THRESHOLDS", "PRE_STEPS", "PRE_BS",
      "RL_UPDATES", "RL_BS", "RL_LR", "ENT_START", "ENT_END", "BC_UPDATES", "BC_LR", "BC_SMOOTH", "CRIT_WARM", "VAL_EVERY", "VAL_N"]},
      sort_keys=True).encode()).hexdigest()[:6]

ENC_SIG = hashlib.md5(json.dumps({k: CFG[k] for k in ["IMG_SIZE", "GRID", "T_MAX", "FEAT", "BALANCED", "THRESHOLDS", "PRE_STEPS", "PRE_BS"]},
      sort_keys=True).encode()).hexdigest()[:6]   # encoders and baselines depend on these settings only

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = Path(os.environ.get("QPD_RESULTS", "results_quick" if QUICK else "results"))
for _sub in ["runs", "enc", "baselines", "figures", "tables"]:
    (RESULTS / _sub).mkdir(parents=True, exist_ok=True)

G = CFG["GRID"]; P = G * G; IMG = CFG["IMG_SIZE"]; PS = IMG // G; T = CFG["T_MAX"]
LAMS = [round(x, 4) for x in CFG["LAMBDAS"]]
MID_LAM = LAMS[len(LAMS) // 2]
LAM_X = round(EXT["LAM"], 4)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
