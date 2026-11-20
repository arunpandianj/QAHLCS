"""Build tables, statistics and figures from the stored run files, then pack everything into a ZIP.

Example: python run_analysis.py --results results
"""
import argparse
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=None); ap.add_argument("--quick", action="store_true")
ap.add_argument("--datasets", nargs="+", default=None)
ap.add_argument("--seeds", nargs="+", type=int, default=None)
ap.add_argument("--lambdas", nargs="+", type=float, default=None)
ap.add_argument("--rl-updates", type=int, default=None)
ap.add_argument("--img-size", type=int, default=28, choices=[28, 64])
ap.add_argument("--extension", action="store_true", help="also analyse the factorial, circuit and tuning stages")
a = ap.parse_args()
import json
ov = {"CFG": {}, "EXT": {}}
if a.datasets: ov["CFG"]["DATASETS"] = a.datasets
if a.seeds: ov["CFG"]["SEEDS"] = a.seeds
if a.lambdas: ov["CFG"]["LAMBDAS"] = a.lambdas
if a.rl_updates: ov["CFG"]["RL_UPDATES"] = a.rl_updates
os.environ["QPD_OVERRIDES"] = json.dumps(ov)
os.environ["QPD_IMG"] = str(a.img_size)
if a.quick: os.environ["QPD_QUICK"] = "1"
if a.results: os.environ["QPD_RESULTS"] = a.results

from qpd.config import RESULTS  # noqa: E402
import qpd.analyze_main  # noqa: E402,F401  (the module builds all tables and figures when imported)
if a.extension:
    import qpd.analyze_extension  # noqa: E402,F401
print("ZIP written:", shutil.make_archive(str(RESULTS), "zip", RESULTS))
