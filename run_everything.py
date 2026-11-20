"""Run the experiments of the study in one call, in order of importance, and pack the results.

    python run_everything.py --hours 8.0
    python run_everything.py --hours 8.0 --only budget tuning factorial circuit --ext-seeds 0 1 2 3 4

The phases run one after another until the time budget is used. No new run is started after the deadline, every finished run is kept,
and the analysis and the ZIP are produced from whatever has finished. Running the command again continues where it stopped.
"""
import argparse
import glob
import subprocess
import sys
import time
import zipfile

ap = argparse.ArgumentParser()
ap.add_argument("--hours", type=float, default=8.0, help="time budget for starting new runs")
ap.add_argument("--quick", action="store_true", help="smoke test of the whole chain")
ap.add_argument("--only", nargs="+", default=None, choices=["samples", "main", "factorial", "circuit", "tuning", "budget", "more"], help="phases to run, in this order")
ap.add_argument("--ext-seeds", nargs="+", default=None, help="seeds of the factorial, circuit and tuning phases")
a = ap.parse_args()
t0 = time.time(); deadline = t0 + a.hours * 3600
Q = ["--quick"] if a.quick else []
ES = ["--ext-seeds"] + a.ext_seeds if a.ext_seeds else []
CH = ["quantum_4x1", "quantum_4x3", "quantum_6x2", "quantum_8x3", "quantum_frozen_4x2", "quantum_noent_4x2", "mlp_620", "bottleneck_8", "mlp_h64"]
if a.quick: CH = ["quantum_frozen_4x2", "quantum_noent_4x2", "mlp_620"]
LONG = ["--lambdas", "0.015", "--seeds", "0", "1", "2", "--rl-updates", "2400", "--results", "results_long"]
MORE = ["--datasets", "organamnist", "octmnist", "pneumoniamnist", "--seeds", "0", "1", "2", "--results", "results_more"]

# key: (title, arguments of run_experiments.py, arguments of run_analysis.py); the default order is the order of importance
PHASES = {
    "samples": ("sample-data figures", [], []),
    "main": ("main stage with matched noise, AUROC and per-class recall", [], []),
    "factorial": ("cloning x confidence-feature factorial", ["--stages", "factorial"] + ES, ["--extension"]),
    "circuit": ("circuit ablation and larger classical heads", ["--stages", "circuit", "--circuit-heads"] + CH + ES, ["--extension"]),
    "tuning": ("learning-rate tuning", ["--stages", "tuning"] + ES, ["--extension"]),
    "budget": ("training-budget check", LONG, LONG),
    "more": ("three additional datasets", MORE, MORE),
}
order = a.only or (["samples", "main", "factorial", "circuit", "tuning"] if a.quick else list(PHASES))


def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True); return subprocess.run(cmd).returncode


for key in order:
    name, ex_args, an_args = PHASES[key]
    if time.time() > deadline: print("Time budget used; skipped:", name); continue
    print(f"\n######## {name}  (elapsed {(time.time() - t0) / 3600:.2f} h)", flush=True)
    if key == "samples": run([sys.executable, "make_sample_figures.py"] + Q); continue
    run([sys.executable, "run_experiments.py", "--deadline", str(deadline)] + Q + ex_args)

print("\n######## analysis", flush=True)
seen = set()
for key in order:
    name, ex_args, an_args = PHASES[key]
    if key == "samples" or tuple(an_args) in seen: continue
    seen.add(tuple(an_args))
    if run([sys.executable, "run_analysis.py"] + Q + an_args) != 0:
        print(f"analysis of '{name}' could not be completed, probably because of unfinished runs; the run files are kept")

with zipfile.ZipFile("final_results_all.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for d in sorted(glob.glob("results*")):
        if d.endswith(".zip"): continue
        for f in glob.glob(d + "/**/*", recursive=True):
            if "/enc/" not in f: z.write(f)
print(f"\nDone in {(time.time() - t0) / 3600:.2f} h. Download final_results_all.zip")
