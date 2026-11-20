"""Command-line entry point.

Examples
    python run_experiments.py --quick                         # smoke test, a few minutes on a GPU
    python run_experiments.py                                 # main experiment (Protocol B, 3 costs, 5 seeds)
    python run_experiments.py --stages main factorial circuit tuning
    python run_experiments.py --img-size 64 --stages main     # larger images
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="tiny smoke-test configuration")
ap.add_argument("--results", default=None, help="output folder (default: results)")
ap.add_argument("--datasets", nargs="+", default=None, help="subset of pathmnist dermamnist bloodmnist")
ap.add_argument("--seeds", nargs="+", type=int, default=None, help="seeds of the main stage")
ap.add_argument("--ext-seeds", nargs="+", type=int, default=None, help="seeds of the extension stages")
ap.add_argument("--stages", nargs="+", default=None, choices=["main", "factorial", "circuit", "tuning"])
ap.add_argument("--lambdas", nargs="+", type=float, default=None, help="patch costs of the main stage")
ap.add_argument("--rl-updates", type=int, default=None, help="policy updates of Protocol B (training-budget check)")
ap.add_argument("--heads", nargs="+", default=None, help="policy heads of the main stage")
ap.add_argument("--deadline", type=float, default=None, help="epoch seconds after which no new run is started")
ap.add_argument("--circuit-heads", nargs="+", default=None, help="heads of the circuit stage")
ap.add_argument("--img-size", type=int, default=28, choices=[28, 64])
a = ap.parse_args()

ov = {"CFG": {}, "EXT": {}}
if a.datasets: ov["CFG"]["DATASETS"] = a.datasets
if a.seeds: ov["CFG"]["SEEDS"] = a.seeds
if a.lambdas: ov["CFG"]["LAMBDAS"] = a.lambdas
if a.heads: ov["CFG"]["POLICIES"] = a.heads
if a.rl_updates: ov["CFG"]["RL_UPDATES"] = a.rl_updates
if a.ext_seeds: ov["EXT"]["SEEDS"] = a.ext_seeds
if a.circuit_heads: ov["EXT"]["CIRCUIT_HEADS"] = a.circuit_heads
if a.deadline: os.environ["QPD_DEADLINE"] = str(a.deadline)
if a.stages: ov["EXT"]["STAGES"] = a.stages
os.environ["QPD_OVERRIDES"] = json.dumps(ov)
os.environ["QPD_IMG"] = str(a.img_size)
if a.quick: os.environ["QPD_QUICK"] = "1"
if a.results: os.environ["QPD_RESULTS"] = a.results

from qpd.runner import run_all  # noqa: E402  (imported after the environment is set)

if __name__ == "__main__":
    run_all()
