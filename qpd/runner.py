"""Experiment runner. Every finished run is stored as one JSON file, so interrupted jobs can be resumed."""
import json
import os
import time
import zlib

import numpy as np
import torch

from . import agent
from .agent import evaluate, eval_baselines, get_encoder, train_policy
from .config import CFG, DEVICE, ENC_SIG, EXT, LAM_X, LAMS, RESULTS, SIG, SPECS, set_seed
from .data import load_medmnist
from .models import n_params

# cond: clone = behaviour-cloning start, conf = confidence features in the observation
DEADLINE = float(os.environ.get("QPD_DEADLINE", "inf"))   # epoch seconds; no new job is started after it

COND = {"B": dict(clone=True, conf=True), "B00": dict(clone=True, conf=False),
        "A01": dict(clone=False, conf=True), "A00": dict(clone=False, conf=False)}
SCRATCH = dict(lr=3e-3, updates=1000, ent=(0.02, 0.002), crit_warm=0)   # settings of the from-scratch protocol A


def jsonable(o):
    if isinstance(o, dict): return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o


def build_jobs(ds, seed):
    """Stages: main (Protocol B, three costs), factorial, circuit, tuning (extension stages at cost LAM_X)."""
    jobs, stages = [], EXT["STAGES"]
    def add(stage, head, lam, cond="B", lr=None):
        jobs.append(dict(stage=stage, ds=ds, seed=seed, head=head, lam=lam, cond=cond, lr=lr))
    if "main" in stages and seed in CFG["SEEDS"]:
        for lam in LAMS:
            for h in CFG["POLICIES"]: add("main", h, lam)
    if seed in EXT["SEEDS"]:
        if "factorial" in stages:
            for h in EXT["HEADS"]:
                for c in ("B00", "A01", "A00"): add("factorial", h, LAM_X, c)
        if "circuit" in stages:
            for h in EXT["CIRCUIT_HEADS"]:
                if h not in EXT["HEADS"]: add("circuit", h, LAM_X)
        if "tuning" in stages:
            for h in EXT["HEADS"]:
                for lr in EXT["TUNE_LR"]: add("tuning", h, LAM_X, "B", lr)
    return jobs


def run_job(j, enc, data, C, base):
    cd, lam, spec = COND[j["cond"]], j["lam"], SPECS[j["head"]]
    agent.CONF_ON = cd["conf"]
    set_seed(zlib.crc32((j["head"] + j["cond"] + str(j["lr"])).encode()) % 100_000 + 1000 * j["seed"] + int(lam * 10_000))
    t0 = time.time(); order = base["greedy_order"]
    tau = max(CFG["THRESHOLDS"], key=lambda t_: base["greedy"]["val"]["thresh"][str(t_)]["bacc"] - lam * base["greedy"]["val"]["thresh"][str(t_)]["glimpses"])
    kw = dict(clone=cd["clone"])
    if not cd["clone"]: kw.update(lr=SCRATCH["lr"], updates=SCRATCH["updates"], ent=SCRATCH["ent"], crit_warm=SCRATCH["crit_warm"])
    if j["lr"]: kw["lr"] = j["lr"]
    head, critic, curve, init_t, selected, val_u = train_policy(enc, data, j["head"], lam, C, order, tau, **kw)
    res = evaluate(enc, head, data["test"], lam, C, detail=True)
    init = {k: init_t[k] for k in ("acc", "bacc", "glimpses", "utility")}
    res.update(stage=j["stage"], main=j["stage"] == "main", dataset=j["ds"], seed=j["seed"], policy=j["head"], kind=spec["kind"],
               nq=spec.get("nq"), nl=spec.get("nl"), cond=j["cond"], lr=kw.get("lr", CFG["RL_LR"]), lam=lam, params=n_params(head),
               curve=curve, init=init, bc=init, selected=selected, val_utility=val_u, teacher_tau=tau)
    if spec["kind"] in ("quantum", "bottleneck") and j["stage"] == "main":   # deployment noise, evaluated without retraining
        set_seed(777); noise = {"shots": {}, "dep": {}}
        for s in CFG["NOISE_SHOTS"]:
            head.shots, head.p_dep = s, 0.0; noise["shots"][str(s)] = evaluate(enc, head, data["test"], lam, C)
        for p_ in CFG["NOISE_DEP"]:
            head.shots, head.p_dep = None, p_; noise["dep"][str(p_)] = evaluate(enc, head, data["test"], lam, C)
        head.shots, head.p_dep = None, 0.0; res["noise"] = noise
    res["train_time"] = time.time() - t0
    agent.CONF_ON = True
    return res


def run_all():
    seeds = sorted(set(CFG["SEEDS"]) | set(EXT["SEEDS"]))
    n_total = sum(len(build_jobs(d, s)) for d in CFG["DATASETS"] for s in seeds)
    print(f"Total runs: {n_total}; stages: {EXT['STAGES']}; device: {DEVICE}"); T0 = time.time(); done_t = done_n = 0
    for ds in CFG["DATASETS"]:
        print(f"\n=== {ds} ==="); data, C, _ = load_medmnist(ds); ytr = data["train"][1]
        data["sw"] = (1.0 / torch.bincount(ytr, minlength=C).float()[ytr]) if CFG["BALANCED"] else torch.ones(len(ytr), device=DEVICE)
        gv = torch.Generator().manual_seed(99); sel = torch.randperm(len(data["val"][0]), generator=gv)[:CFG["VAL_N"]].to(DEVICE)
        data["val_sub"] = (data["val"][0][sel], data["val"][1][sel])
        for seed in seeds:
            jobs = build_jobs(ds, seed)
            if not jobs: continue
            if time.time() > DEADLINE: print("Time limit reached; stopping. Finished runs are kept and a later call resumes."); return
            enc = get_encoder(ds, seed, data, C); fb = RESULTS / "baselines" / f"{ds}_s{seed}_{ENC_SIG}.json"
            if not fb.exists(): set_seed(5000 + seed); json.dump(jsonable(eval_baselines(enc, data, C)), open(fb, "w"))
            base = json.load(open(fb))
            for j in jobs:
                f = RESULTS / "runs" / f"{j['stage']}__{ds}__{j['head']}__{j['cond']}__lam{j['lam']}__lr{j['lr'] or 'def'}__s{seed}__{SIG}.json"
                if f.exists(): continue
                if time.time() > DEADLINE: print("Time limit reached; stopping. Finished runs are kept and a later call resumes."); return
                res = run_job(j, enc, data, C, base); json.dump(jsonable(res), open(f, "w"))
                done_t += res["train_time"]; done_n += 1
                rem = n_total - len(list((RESULTS / "runs").glob(f"*__{SIG}.json")))
                print(f"  {j['stage']:9s} {j['head']:19s} {j['cond']:4s} lam={j['lam']:<6} seed={seed} | util={res['utility']:.3f} "
                      f"(start {res['init']['utility']:.3f}, kept {res['selected']}) | {res['train_time']:.0f}s | est. remaining {done_t / done_n * rem / 60:.0f} min")
        del data
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
    print(f"\nFinished in {(time.time() - T0) / 60:.1f} min")
