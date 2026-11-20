"""Tables, statistical tests and figures for the extension stages (factorial, circuit, tuning)."""
import json

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import CFG, ENC_SIG, EXT, G, LAMS, LAM_X, MID_LAM, P, PS, RESULTS, SIG, SPECS, T  # noqa: E402
from .data import load_medmnist  # noqa: E402

display = print
from .runner import COND  # noqa: E402

plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
DSN = {'pathmnist': 'PathMNIST', 'dermamnist': 'DermaMNIST', 'bloodmnist': 'BloodMNIST', 'organamnist': 'OrganAMNIST', 'octmnist': 'OCTMNIST', 'pneumoniamnist': 'PneumoniaMNIST'}
ORDERS = ['random', 'raster', 'center', 'greedy']
def kcol(p): return {'quantum': '#1f77b4', 'mlp': '#d62728', 'bottleneck': '#2ca02c'}[SPECS[p]['kind']]
RAW = [json.load(open(f)) for f in sorted((RESULTS / 'runs').glob(f'*__{SIG}.json'))]
BASEL = {}
for f in (RESULTS / 'baselines').glob(f'*_{ENC_SIG}.json'):
    BASEL[(f.stem.split('_s')[0], int(f.stem.split('_s')[1].split('_')[0]))] = json.load(open(f))

def best_adapt(b, lam):
    bb = None
    for nm in ORDERS:
        for t_ in CFG['THRESHOLDS']:
            v, te = b[nm]['val']['thresh'][str(t_)], b[nm]['test']['thresh'][str(t_)]; u = v['bacc'] - lam * v['glimpses']
            if bb is None or u > bb[0]: bb = (u, te['bacc'] - lam * te['glimpses'])
    return bb[1]

rows = []
for d in RAW:
    rows.append(dict(stage=d['stage'], lam=d['lam'], dataset=d['dataset'], seed=d['seed'], policy=d['policy'], cond=d['cond'], lr=d['lr'], params=d['params'],
                     util=d['utility'], bacc=d['bacc'], f1=d['f1'], auroc=d['auroc'], glimpses=d['glimpses'], min_recall=min(r for r in d['recall'] if r is not None),
                     start_util=d['init']['utility'], val_util=d['val_utility'], selected=d['selected'],
                     gain_adapt=d['utility'] - best_adapt(BASEL[(d['dataset'], d['seed'])], d['lam'])))
DF = pd.DataFrame(rows); DF.to_csv(RESULTS / 'tables' / 'all_ext_runs.csv', index=False)
CORE = DF[(DF.stage == 'main') & (DF.lam == LAM_X)]; print(len(DF), 'runs loaded')

def boot_ci(d, rng=np.random.default_rng(0)):
    return np.percentile([rng.choice(d, len(d)).mean() for _ in range(5000)], [2.5, 97.5])
def paired_s(a, b):
    """paired test on the (dataset, seed) pairs that both series contain"""
    idx = a.index.intersection(b.index)
    return paired(a.loc[idx].values, b.loc[idx].values)


def paired(a, b):
    """a, b: aligned arrays over (dataset, seed) pairs"""
    d = np.asarray(a) - np.asarray(b); out = dict(n=len(d), mean_diff=d.mean())
    if len(d) > 1:
        lo, hi = boot_ci(d); out.update(ci_lo=lo, ci_hi=hi, dz=d.mean() / (d.std(ddof=1) + 1e-12))
        try: out['p_wilcoxon'] = stats.wilcoxon(d).pvalue
        except Exception: out['p_wilcoxon'] = np.nan
    return out
def aligned(df, col, **flt):
    g = df
    for k, v in flt.items(): g = g[g[k] == v]
    return g.sort_values(['dataset', 'seed']).set_index(['dataset', 'seed'])[col]
def save_table(df, name, **kw):
    df.to_csv(RESULTS / 'tables' / f'{name}.csv')
    try: (RESULTS / 'tables' / f'{name}.tex').write_text(df.to_latex(**kw))
    except Exception as e: print('LaTeX export skipped:', name, e)

# ---- E1: factorial study of cloning and confidence features
E1 = DF[DF.stage.isin(['main', 'factorial']) & (DF.lam == LAM_X)]
if 'factorial' in set(DF.stage):
    t = E1.groupby(['policy', 'cond']).agg(utility=('util', 'mean'), sd=('util', 'std'), gain_vs_adaptive=('gain_adapt', 'mean'), glimpses=('glimpses', 'mean'), auroc=('auroc', 'mean')).round(4)
    save_table(t, 'E1_factorial_cells'); display(t)
    r = []
    for h in EXT['HEADS']:
        u = lambda c: aligned(E1, 'util', policy=h, cond=c)
        for lab, a, b in (('cloning effect with features (B - A01)', 'B', 'A01'), ('cloning effect without features (B00 - A00)', 'B00', 'A00'),
                          ('feature effect with cloning (B - B00)', 'B', 'B00'), ('feature effect from scratch (A01 - A00)', 'A01', 'A00')):
            if len(u(a)) and len(u(b)): r.append(dict(head=h, contrast=lab, **paired_s(u(a), u(b))))
    for c in COND:
        for h in EXT['HEADS'][1:]:
            qa, ha = aligned(E1, 'util', policy=EXT['HEADS'][0], cond=c), aligned(E1, 'util', policy=h, cond=c)
            if len(qa) and len(ha): r.append(dict(head=f'{EXT["HEADS"][0]} vs {h}', contrast=f'head difference in cell {c}', **paired_s(qa, ha)))
    E1t = pd.DataFrame(r).round(4).set_index(['head', 'contrast']); save_table(E1t, 'E1_factorial_tests'); display(E1t)

# ---- E2: circuit ablation (core cell and circuit stage)
E2 = DF[DF.stage.isin(['main', 'circuit']) & (DF.cond == 'B') & (DF.lam == LAM_X)]
if 'circuit' in set(DF.stage):
    t = E2.groupby('policy').agg(trainable_params=('params', 'first'), utility=('util', 'mean'), sd=('util', 'std'), bacc=('bacc', 'mean'), glimpses=('glimpses', 'mean'),
                                 auroc=('auroc', 'mean'), gain_vs_adaptive=('gain_adapt', 'mean')).round(4).sort_values('trainable_params')
    save_table(t, 'E2_circuit_ablation'); display(t)
    r = []
    for a, b in (('quantum_4x2', 'quantum_frozen_4x2'), ('quantum_4x2', 'quantum_noent_4x2'), ('quantum_8x3', 'mlp_620'), ('quantum_8x3', 'bottleneck_8'), ('quantum_8x3', 'quantum_4x2'),
                 ('quantum_4x2', 'quantum_2x2'), ('quantum_4x2', 'quantum_4x1'), ('quantum_4x3', 'quantum_4x2')):
        x, y = aligned(E2, 'util', policy=a), aligned(E2, 'util', policy=b)
        if len(x) and len(y): r.append(dict(comparison=f'{a} - {b}', **paired_s(x, y)))
    E2t = pd.DataFrame(r).round(4).set_index('comparison'); save_table(E2t, 'E2_circuit_tests'); display(E2t)

# ---- E3: learning-rate sensitivity, selection by validation utility
E3 = DF[DF.stage.isin(['main', 'tuning']) & (DF.cond == 'B') & DF.policy.isin(EXT['HEADS']) & (DF.lam == LAM_X)]
if 'tuning' in set(DF.stage):
    t = E3.groupby(['policy', 'lr']).agg(val_utility=('val_util', 'mean'), test_utility=('util', 'mean'), test_sd=('util', 'std'), glimpses=('glimpses', 'mean')).round(4)
    save_table(t, 'E3_lr_sensitivity'); display(t)
    sel = {h: E3[E3.policy == h].groupby('lr').val_util.mean().idxmax() for h in EXT['HEADS']}; print('learning rate chosen on validation:', sel)
    pick = {h: aligned(E3, 'util', policy=h, lr=sel[h]) for h in EXT['HEADS']}
    r = [dict(comparison=f'{EXT["HEADS"][0]} - {h} (validation-selected learning rates)', **paired_s(pick[EXT['HEADS'][0]], pick[h])) for h in EXT['HEADS'][1:]]
    E3t = pd.DataFrame(r).round(4).set_index('comparison'); save_table(E3t, 'E3_tuned_head_tests'); display(E3t)

# ---- E4: extended metrics of the main cell (AUROC, worst-class recall)
t = CORE.groupby(['dataset', 'policy']).agg(bacc=('bacc', 'mean'), macro_f1=('f1', 'mean'), auroc=('auroc', 'mean'), min_class_recall=('min_recall', 'mean'), glimpses=('glimpses', 'mean')).round(4)
save_table(t, 'E4_extended_metrics'); display(t)
pr = []
for d in RAW:
    if d['stage'] == 'main' and d['lam'] == LAM_X: pr.append(dict(dataset=d['dataset'], policy=d['policy'], seed=d['seed'], **{f'class{c}': r_ for c, r_ in enumerate(d['recall'])}))
pd.DataFrame(pr).groupby(['dataset', 'policy']).mean(numeric_only=True).round(3).to_csv(RESULTS / 'tables' / 'E4_per_class_recall.csv')


def savefig(fig, name):
    fig.savefig(RESULTS / 'figures' / f'{name}.png'); fig.savefig(RESULTS / 'figures' / f'{name}.pdf'); plt.close(fig)
HM = {'B': 'clone + features', 'B00': 'clone, no features', 'A01': 'scratch + features', 'A00': 'scratch, no features'}
HATCH = {'B': '', 'B00': '//', 'A01': '..', 'A00': 'xx'}

if 'factorial' in set(DF.stage):
    fig, ax = plt.subplots(figsize=(6.4, 3.3)); w = 0.2
    for k, c in enumerate(['A00', 'A01', 'B00', 'B']):
        g = E1[E1.cond == c].groupby('policy').gain_adapt.agg(['mean', 'std']).reindex(EXT['HEADS'])
        ax.bar(np.arange(3) + (k - 1.5) * w, g['mean'], w, yerr=g['std'], color=[kcol(p) for p in EXT['HEADS']], hatch=HATCH[c], edgecolor='white', capsize=2, label=HM[c])
    ax.axhline(0, color='k', lw=0.7); ax.set_xticks(range(3)); ax.set_xticklabels(EXT['HEADS']); ax.set_ylabel('Utility minus best confidence-stop baseline'); ax.legend(fontsize=6.5)
    savefig(fig, 'E1_factorial')
if 'circuit' in set(DF.stage):
    fig, ax = plt.subplots(figsize=(6.4, 3.4)); g = E2.groupby('policy').agg(p=('params', 'first'), m=('util', 'mean'), s=('util', 'std'))
    for pol, r in g.iterrows():
        ax.errorbar(r.p, r.m, yerr=r.s, fmt='o', color=kcol(pol), capsize=2, ms=5); ax.annotate(pol.replace('quantum_', 'q'), (r.p, r.m), fontsize=6, xytext=(3, 3), textcoords='offset points')
    ax.set_xscale('log'); ax.set_xlabel('Trainable head parameters'); ax.set_ylabel('Utility (mean $\\pm$ s.d.)'); savefig(fig, 'E2_circuit_ablation')
if 'tuning' in set(DF.stage):
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for h in EXT['HEADS']:
        g = E3[E3.policy == h].groupby('lr')[['util', 'val_util']].mean().sort_index()
        ax.plot(g.index, g.util, 'o-', color=kcol(h), label=f'{h}, test'); ax.plot(g.index, g.val_util, 's--', color=kcol(h), alpha=0.5, label=f'{h}, validation')
    ax.set_xscale('log'); ax.set_xlabel('Policy learning rate'); ax.set_ylabel('Utility'); ax.legend(fontsize=6); savefig(fig, 'E3_lr_sensitivity')
fig, ax = plt.subplots(figsize=(5.4, 3.0))
for k, h in enumerate(EXT['HEADS']):
    g = CORE[CORE.policy == h].groupby('dataset').auroc.agg(['mean', 'std']).reindex(CFG['DATASETS'])
    ax.errorbar(np.arange(len(g)) + (k - 1) * 0.15, g['mean'], yerr=g['std'], fmt='o', color=kcol(h), capsize=2, label=h)
ax.set_xticks(range(len(CFG['DATASETS']))); ax.set_xticklabels([DSN[d] for d in CFG['DATASETS']]); ax.set_ylabel('Macro AUROC at stopping time'); ax.legend(fontsize=6.5)
savefig(fig, 'E4_auroc')
