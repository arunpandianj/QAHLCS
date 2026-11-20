# Quantum policy heads for budget-aware medical image diagnosis

Python implementation of the study "Variational Quantum Policy Heads for Budget-Aware Medical Image Diagnosis: Controlled Comparison with Classical Heads and Confidence-Stopping Baselines".
An agent reveals patches of a medical image one at a time and stops when another patch is no longer worth its cost. The reward is balanced
accuracy minus a fixed cost per patch. A variational quantum circuit, simulated exactly on a GPU, is used as the policy head and is compared with
classical heads of equal size and with eight non-learned baselines. No quantum hardware is needed.

## Contents

| Path | Purpose |
|---|---|
| `qpd/config.py` | all settings, policy-head specifications, result folder |
| `qpd/data.py` | MedMNIST loading and the 4 x 4 patch grid |
| `qpd/simulator.py` | batched exact state-vector simulator, finite-shot and depolarising-noise models |
| `qpd/models.py` | frozen encoder, quantum head, MLP head, bottleneck head, critic |
| `qpd/agent.py` | environment rollout, encoder pre-training, behaviour cloning, advantage actor-critic, evaluation, non-learned baselines |
| `qpd/runner.py` | experiment stages and resumable job loop |
| `qpd/analyze_main.py` | tables, paired statistical tests and figures for the main experiment |
| `qpd/analyze_extension.py` | tables and figures for the extension stages |
| `run_experiments.py`, `run_analysis.py` | command-line entry points |
| `tests/test_simulator.py` | simulator check against PennyLane and head-size check |

## Installation

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
A CUDA GPU is strongly recommended. The MedMNIST files (PathMNIST, DermaMNIST, BloodMNIST) are downloaded automatically on first use
(https://medmnist.com/) and are found automatically if a copy of the `.npz` files already exists in `~/.medmnist`.

## Usage

Smoke test, a few minutes on a GPU:
```
python run_experiments.py --quick
python run_analysis.py --quick
```
Main experiment: three patch costs, three datasets, five seeds, three heads, Protocol B:
```
python run_experiments.py --results results
python run_analysis.py --results results
```
Extension stages at cost 0.015:
```
python run_experiments.py --results results --stages main factorial circuit tuning
python run_analysis.py --results results --extension
```
| Stage | Content |
|---|---|
| `main` | Protocol B, behaviour cloning of the greedy-order teacher followed by refinement; noise sweep of the quantum head |
| `factorial` | 2 x 2 design of cloning and confidence features |
| `circuit` | qubits, layers, random fixed circuit, circuit without CNOT gates, classical references for the 8-qubit circuit |
| `tuning` | policy learning rate per head, chosen on validation utility |

Useful options: `--datasets`, `--seeds`, `--ext-seeds`, `--img-size 64`. Every finished run is written to `results/runs/*.json`, so an interrupted
job continues where it stopped. The file names carry a configuration signature, so changing a training setting starts new runs and does not mix old ones.

One call for everything: `python run_everything.py --hours 8` runs the main stage, the factorial, circuit and tuning stages, a training-budget
check and three additional datasets in order of importance. No new run starts after the time budget, finished runs are kept, and the
analysis and `final_results_all.zip` are produced from what has finished. Running the same command again continues the unfinished work.

Sample-data figures: `python make_sample_figures.py` writes one random test image per class for each dataset and the output of a trained policy head on the first test images, using the stored run files of the main stage.

## Outputs

`results/tables` contains the CSV and LaTeX tables, `results/figures` the figures as PNG and PDF, `results/baselines` the non-learned baseline
metrics for each dataset and seed, and `results/enc` the frozen encoder weights. `run_analysis.py` also packs the folder into `results.zip`.

## Notes

* Protocol A (learning from scratch, no confidence features) corresponds to the cell `A00` of the `factorial` stage.
* Training times on one Tesla T4 GPU: main stage 202 minutes for 135 runs, factorial stage 117 minutes for 81 runs, circuit stage 141 minutes for 81 runs.
* The learning-rate tuning stage was only partly run in the reported experiments and its results are not used in the paper.
