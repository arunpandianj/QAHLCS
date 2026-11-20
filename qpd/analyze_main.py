"""Tables, statistical tests and figures for the main stage (Protocol B)."""
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

plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
COLK = {'quantum': '#1f77b4', 'mlp': '#d62728', 'bottleneck': '#2ca02c'}
def pcol(p): return COLK[SPECS[p]['kind']]
DSN = {'pathmnist': 'PathMNIST', 'dermamnist': 'DermaMNIST', 'bloodmnist': 'BloodMNIST', 'organamnist': 'OrganAMNIST', 'octmnist': 'OCTMNIST', 'pneumoniamnist': 'PneumoniaMNIST'}
KS = list(range(0, T + 1)); ORDERS = ['random', 'raster', 'center', 'greedy']

RAW = [json.load(open(f)) for f in sorted((RESULTS / 'runs').glob(f'*__{SIG}.json'))]
BASEL = {}
for f in (RESULTS / 'baselines').glob(f'*_{ENC_SIG}.json'):
    BASEL[(f.stem.split('_s')[0], int(f.stem.split('_s')[1].split('_')[0]))] = json.load(open(f))

def fixed_curve(b, name, split='test', key='bacc'): return np.array([b[name][split]['fixed'][str(k)][key] for k in KS])

def pick_baseline(b, lam):
    """Best fixed-budget and best adaptive-stopping baseline for this lambda, chosen on the VALIDATION split, scored on TEST."""
    res = {}
    for fam in ('fixed', 'adaptive'):
        best = None
        for nm in ORDERS:
            if fam == 'fixed': cands = [(str(k), 'fixed', str(k)) for k in range(1, T + 1)]
            else: cands = [(str(t), 'thresh', str(t)) for t in CFG['THRESHOLDS']]
            for key, kind, kk in cands:
                v, te = b[nm]['val'][kind][kk], b[nm]['test'][kind][kk]
                u = v['bacc'] - lam * v['glimpses']
                if best is None or u > best[0]: best = (u, nm, key, te['bacc'] - lam * te['glimpses'], te['glimpses'], te['bacc'])
        res[fam] = best
    return res

rows = []
for d in RAW:
    pb = pick_baseline(BASEL[(d['dataset'], d['seed'])], d['lam'])
    rows.append(dict(dataset=d['dataset'], policy=d['policy'], kind=d['kind'], lam=d['lam'], seed=d['seed'], nq=d['nq'], nl=d['nl'], main=d['main'],
                     params=d['params'], acc=d['acc'], bacc=d['bacc'], f1=d['f1'], ece=d['ece'], glimpses=d['glimpses'], utility=d['utility'],
                     base_fixed_util=pb['fixed'][3], base_adapt_util=pb['adaptive'][3],
                     base_fixed_cfg=f"{pb['fixed'][1]}-k{pb['fixed'][2]}", base_adapt_cfg=f"{pb['adaptive'][1]}-tau{pb['adaptive'][2]}",
                     gain_fixed=d['utility'] - pb['fixed'][3], gain_adapt=d['utility'] - pb['adaptive'][3], train_time=d['train_time'],
                     bc_utility=d['bc']['utility'], bc_glimpses=d['bc']['glimpses'], bc_gain_adapt=d['bc']['utility'] - pb['adaptive'][3],
                     rl_delta=d['utility'] - d['bc']['utility'], selected=d['selected']))
DF = pd.DataFrame(rows); DF.to_csv(RESULTS / 'tables' / 'all_runs.csv', index=False)
if not DF.main.any(): raise SystemExit('No main-stage runs found in this results folder.')
MAIN = DF[DF.main].copy(); POLS = [p for p in CFG['POLICIES'] if p in MAIN.policy.unique()]
print(len(DF), 'runs loaded;', len(MAIN), 'main runs')

def save_table(df, name, **kw):
    df.to_csv(RESULTS / 'tables' / f'{name}.csv')
    try: (RESULTS / 'tables' / f'{name}.tex').write_text(df.to_latex(**kw))
    except Exception as e: print('LaTeX export skipped for', name, e)

def pm(g, col, nd=3): return f'{g[col].mean():.{nd}f} ± {g[col].std(ddof=1) if len(g) > 1 else 0:.{nd}f}'

# ---- Table 1: main results (mean ± std over seeds)
t1 = []
for (ds, lam, pol), g in MAIN.groupby(['dataset', 'lam', 'policy']):
    t1.append(dict(Dataset=DSN[ds], Lambda=lam, Policy=pol, BalancedAcc=pm(g, 'bacc'), Accuracy=pm(g, 'acc'), MacroF1=pm(g, 'f1'),
                   Glimpses=pm(g, 'glimpses', 2), Utility=pm(g, 'utility'), CloneUtility=pm(g, 'bc_utility'), RLminusClone=pm(g, 'rl_delta'), GainVsFixed=pm(g, 'gain_fixed'), GainVsAdaptive=pm(g, 'gain_adapt'), ECE=pm(g, 'ece')))
T1 = pd.DataFrame(t1).set_index(['Dataset', 'Lambda', 'Policy']); save_table(T1, 'table1_main_results', escape=True); display(T1)

# ---- Table 2: parameter counts
T2 = DF.groupby('policy').params.first().to_frame('head_parameters'); save_table(T2, 'table2_parameters'); display(T2)

# ---- Table 3: non-RL baselines (test balanced accuracy, mean over seeds) and the validation-selected reference per lambda
t3 = []
for ds in CFG['DATASETS']:
    for nm in ORDERS:
        r = {'Dataset': DSN[ds], 'Order': nm}
        for k in (0, 1, 2, 4, 8, P): r[f'fixed k={k}'] = np.mean([BASEL[(ds, s)][nm]['test']['fixed'][str(k)]['bacc'] for s in CFG['SEEDS'] if (ds, s) in BASEL])
        for tau in (0.8, 0.95):
            if tau in CFG['THRESHOLDS']:
                r[f'stop tau={tau}'] = '%.3f @ %.2f gl.' % (np.mean([BASEL[(ds, s)][nm]['test']['thresh'][str(tau)]['bacc'] for s in CFG['SEEDS'] if (ds, s) in BASEL]),
                                                         np.mean([BASEL[(ds, s)][nm]['test']['thresh'][str(tau)]['glimpses'] for s in CFG['SEEDS'] if (ds, s) in BASEL]))
        t3.append(r)
T3 = pd.DataFrame(t3).set_index(['Dataset', 'Order']); save_table(T3, 'table3_baselines'); display(T3.round(3))
ref = MAIN.groupby(['dataset', 'lam']).agg(best_fixed_utility=('base_fixed_util', 'mean'), best_adaptive_utility=('base_adapt_util', 'mean'),
                                           fixed_cfg=('base_fixed_cfg', lambda s: s.mode().iloc[0]), adaptive_cfg=('base_adapt_cfg', lambda s: s.mode().iloc[0])).round(4)
save_table(ref, 'table3b_baseline_reference'); display(ref)

# ---- Table 4: paired tests between policies + one-sample tests of the gain over the best baselines
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for i, j in enumerate(o): run = max(run, (len(p) - i) * p[j]); adj[j] = min(1, run)
    return adj

def boot_ci(d, rng=np.random.default_rng(0)):
    bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]; return np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def paired(main, metric, a, b):
    pv = main.pivot_table(index=['dataset', 'lam'], columns='policy', values=metric, aggfunc='mean')
    d = (pv[a] - pv[b]).dropna().values
    ps = main.pivot_table(index=['dataset', 'lam', 'seed'], columns='policy', values=metric)
    ds_ = (ps[a] - ps[b]).dropna().values
    out = dict(comparison=f'{a} - {b}', metric=metric, n_units=len(d), mean_diff=d.mean(), seed_level_wins=f'{int((ds_ > 0).sum())}/{len(ds_)}')
    if len(d) > 1:
        out.update(ci95_lo=boot_ci(d)[0], ci95_hi=boot_ci(d)[1], cohen_dz=d.mean() / (d.std(ddof=1) + 1e-12), p_ttest=stats.ttest_1samp(d, 0).pvalue)
        try: out['p_wilcoxon'] = stats.wilcoxon(d).pvalue
        except Exception: out['p_wilcoxon'] = np.nan
    return out

def one_sample(main, pol, metric):
    d = main[main.policy == pol].groupby(['dataset', 'lam'])[metric].mean().values
    out = dict(comparison=f'{pol} gain', metric=metric, n_units=len(d), mean_diff=d.mean(), seed_level_wins=f'{int((main[main.policy == pol][metric] > 0).sum())}/{int((main.policy == pol).sum())}')
    if len(d) > 1:
        out.update(ci95_lo=boot_ci(d)[0], ci95_hi=boot_ci(d)[1], cohen_dz=d.mean() / (d.std(ddof=1) + 1e-12), p_ttest=stats.ttest_1samp(d, 0).pvalue)
        try: out['p_wilcoxon'] = stats.wilcoxon(d).pvalue
        except Exception: out['p_wilcoxon'] = np.nan
    return out

T4 = None
if MAIN.groupby(['dataset', 'lam']).ngroups > 1:
    qn = [p for p in POLS if SPECS[p]['kind'] == 'quantum'][0]
    t4 = [paired(MAIN, m, qn, b) for b in POLS if b != qn for m in ('utility', 'bacc', 'acc', 'f1', 'glimpses', 'ece')]
    t4 += [one_sample(MAIN, p, m) for p in POLS for m in ('gain_fixed', 'gain_adapt', 'bc_gain_adapt', 'rl_delta')]
    T4 = pd.DataFrame(t4)
    T4['p_wilcoxon_holm'] = holm(T4.p_wilcoxon.fillna(1).values)
    T4 = T4.round(4).set_index(['comparison', 'metric']); save_table(T4, 'table4_paired_tests'); display(T4)
else: print('Not enough conditions for paired tests (QUICK_TEST?).')

# ---- Table 4b: behaviour-cloned start versus final policy versus the teacher baseline (mean over all main runs)
T4b = MAIN.groupby('policy').agg(clone_utility=('bc_utility', 'mean'), final_utility=('utility', 'mean'), best_adaptive_baseline=('base_adapt_util', 'mean'),
                                 best_fixed_baseline=('base_fixed_util', 'mean'), clone_glimpses=('bc_glimpses', 'mean'), final_glimpses=('glimpses', 'mean')).round(4)
T4b['kept_clone_fraction'] = MAIN.groupby('policy').selected.apply(lambda s: s.isin(['bc', 'clone']).mean()).round(3)
save_table(T4b, 'table4b_clone_vs_final'); display(T4b)

# ---- Table 5: noise robustness of the quantum policies (deployment noise)
t5 = []
for d in RAW:
    if 'noise' not in d: continue
    for kind, lab in (('shots', 'shots'), ('dep', 'depolarising p')):
        for lv, m in d['noise'][kind].items():
            t5.append(dict(policy=d['policy'], dataset=d['dataset'], lam=d['lam'], seed=d['seed'], kind=lab, level=float(lv), utility=m['utility'], bacc=m['bacc'], glimpses=m['glimpses'], clean_utility=d['utility']))
T5raw = pd.DataFrame(t5)
if len(T5raw):
    T5raw['rel_utility'] = T5raw.utility / T5raw.clean_utility
    T5 = T5raw.groupby(['policy', 'kind', 'level']).agg(utility=('utility', 'mean'), bacc=('bacc', 'mean'), glimpses=('glimpses', 'mean'), rel_utility=('rel_utility', 'mean')).round(4)
    save_table(T5, 'table5_noise_robustness'); display(T5)

# ---- Table 6: circuit-size ablation (all heads at the ablation dataset and lambda)
ab = DF[(DF.dataset == CFG['ABL_DATASET']) & (DF.lam == round(CFG['ABL_LAMBDA'], 4))]
if CFG['ABLATION'] and len(ab):
    T6 = ab.groupby('policy').agg(params=('params', 'first'), bacc=('bacc', 'mean'), glimpses=('glimpses', 'mean'), utility=('utility', 'mean'), utility_sd=('utility', 'std')).round(4).sort_values('params')
    save_table(T6, 'table6_ablation'); display(T6)


def savefig(fig, name):
    fig.savefig(RESULTS / 'figures' / f'{name}.png'); fig.savefig(RESULTS / 'figures' / f'{name}.pdf'); plt.close(fig)

DSS = CFG['DATASETS']; RAWM = [d for d in RAW if d['main']]
QPOL = [p for p in POLS if SPECS[p]['kind'] == 'quantum'][0]

# Fig 1: balanced accuracy vs mean glimpses, with the strongest non-RL baselines
fig, axs = plt.subplots(1, len(DSS), figsize=(4.4 * len(DSS), 3.5), squeeze=False)
for ax, ds in zip(axs[0], DSS):
    bs = [BASEL[(ds, s)] for s in CFG['SEEDS'] if (ds, s) in BASEL]
    for nm, st in (('random', ('--', 'gray')), ('center', ('-.', 'gray')), ('greedy', ('-', 'k'))):
        ax.plot(KS, np.mean([fixed_curve(b, nm) for b in bs], 0), st[0], color=st[1], lw=1, label=f'fixed order: {nm}')
    for nm, col in (('greedy', 'darkorange'), ('center', 'goldenrod')):
        gx = [np.mean([b[nm]['test']['thresh'][str(t)]['glimpses'] for b in bs]) for t in CFG['THRESHOLDS']]
        gy = [np.mean([b[nm]['test']['thresh'][str(t)]['bacc'] for b in bs]) for t in CFG['THRESHOLDS']]
        ax.plot(gx, gy, ':', marker='s', ms=3, color=col, label=f'{nm} order + confidence stop')
    for pol in POLS:
        g = MAIN[(MAIN.dataset == ds) & (MAIN.policy == pol)].groupby('lam')[['glimpses', 'bacc']].agg(['mean', 'std']).sort_index()
        ax.errorbar(g[('glimpses', 'mean')], g[('bacc', 'mean')], xerr=g[('glimpses', 'std')].fillna(0), yerr=g[('bacc', 'std')].fillna(0),
                    marker='o', color=pcol(pol), capsize=2, label=pol)
    ax.set_title(DSN[ds]); ax.set_xlabel('Mean glimpses'); ax.set_ylabel('Test balanced accuracy')
axs[0][0].legend(fontsize=5.5); savefig(fig, 'fig1_balanced_accuracy_vs_glimpses')

# Fig 2: utility gain over the best validation-selected baselines
units = MAIN.groupby(['dataset', 'lam']).size().index.tolist()
fig, axs = plt.subplots(2, 1, figsize=(max(6, 0.9 * len(units) + 2), 5.4), sharex=True); w = 0.8 / len(POLS)
for ax, col, ttl in zip(axs, ('gain_fixed', 'gain_adapt'), ('best fixed-budget baseline', 'best adaptive-stopping baseline')):
    for j, pol in enumerate(POLS):
        m = [MAIN[(MAIN.dataset == u[0]) & (MAIN.lam == u[1]) & (MAIN.policy == pol)][col] for u in units]
        ax.bar(np.arange(len(units)) + j * w, [x.mean() for x in m], w, yerr=[x.std(ddof=1) if len(x) > 1 else 0 for x in m], color=pcol(pol), capsize=2, label=pol)
    ax.axhline(0, color='k', lw=0.6); ax.set_ylabel(f'Utility gain over\n{ttl}', fontsize=8)
axs[1].set_xticks(np.arange(len(units)) + w); axs[1].set_xticklabels([f'{DSN[u[0]][:5]}\nλ={u[1]}' for u in units], fontsize=7); axs[0].legend(fontsize=6.5)
savefig(fig, 'fig2_gain_over_baselines')

# Fig 3: learning curves
fig, axs = plt.subplots(1, len(DSS), figsize=(4.2 * len(DSS), 3.2), squeeze=False)
for ax, ds in zip(axs[0], DSS):
    for pol in POLS:
        cs = [pd.DataFrame(d['curve']).set_index('update').ret for d in RAWM if d['dataset'] == ds and d['policy'] == pol and d['lam'] == MID_LAM]
        if not cs: continue
        c = pd.concat(cs, axis=1); mu, sd = c.mean(1), c.std(1).fillna(0)
        ax.plot(mu.index, mu, color=pcol(pol), label=pol); ax.fill_between(mu.index, mu - sd, mu + sd, color=pcol(pol), alpha=0.2)
    ax.set_title(f'{DSN[ds]} (λ={MID_LAM})'); ax.set_xlabel('Policy update'); ax.set_ylabel('Training return')
axs[0][0].legend(fontsize=6.5); savefig(fig, 'fig3_learning_curves')

# Fig 4: noise robustness of the quantum policy
if len(T5raw):
    fig, axs = plt.subplots(2, len(DSS), figsize=(4.2 * len(DSS), 6), squeeze=False)
    for j, ds in enumerate(DSS):
        for i, (kind, lab, logx) in enumerate((('shots', 'Measurement shots per circuit', True), ('depolarising p', 'Depolarising probability per qubit per layer', False))):
            ax = axs[i][j]; g = T5raw[(T5raw.policy == QPOL) & (T5raw.dataset == ds) & (T5raw.lam == MID_LAM) & (T5raw.kind == kind)].groupby('level').utility.agg(['mean', 'std']).sort_index()
            ax.errorbar(g.index, g['mean'], yerr=g['std'].fillna(0), marker='o', color=pcol(QPOL), capsize=2, label=f'{QPOL}, noisy deployment')
            gb = T5raw[(T5raw.policy == 'bottleneck_4') & (T5raw.dataset == ds) & (T5raw.lam == MID_LAM) & (T5raw.kind == kind)].groupby('level').utility.agg(['mean', 'std']).sort_index()
            if len(gb): ax.errorbar(gb.index, gb['mean'], yerr=gb['std'].fillna(0), marker='s', color=pcol('bottleneck_4'), capsize=2, label='bottleneck_4, matched perturbation')
            for pol in POLS:
                u = MAIN[(MAIN.dataset == ds) & (MAIN.lam == MID_LAM) & (MAIN.policy == pol)].utility.mean()
                ax.axhline(u, color=pcol(pol), ls='-' if pol == QPOL else '--', lw=1, label=f'{pol}, noiseless')
            if logx: ax.set_xscale('log')
            ax.set_xlabel(lab); ax.set_ylabel('Utility (bal. acc − λ·glimpses)'); ax.set_title(f'{DSN[ds]} (λ={MID_LAM})')
    axs[0][0].legend(fontsize=6); savefig(fig, 'fig4_noise_robustness')

# Fig 5: where do the policies look?
fig, axs = plt.subplots(len(DSS), len(POLS), figsize=(2.6 * len(POLS), 2.4 * len(DSS)), squeeze=False)
for i, ds in enumerate(DSS):
    for j, pol in enumerate(POLS):
        ms = [np.array(d['patch_counts']).sum(0) for d in RAWM if d['dataset'] == ds and d['policy'] == pol and d['lam'] == MID_LAM]
        ax = axs[i][j]
        if ms:
            m = np.mean([x / x.sum() for x in ms], 0).reshape(G, G); im = ax.imshow(m, cmap='viridis'); plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if i == 0: ax.set_title(pol, fontsize=8)
        if j == 0: ax.set_ylabel(DSN[ds])
fig.suptitle(f'Patch selection frequency (λ={MID_LAM})', y=1.0); savefig(fig, 'fig5_patch_heatmaps')

# Fig 6: example glimpse trajectories
def show_traces(ds, lam, seed, n=6):
    d = next((d for d in RAWM if d['dataset'] == ds and d['policy'] == QPOL and d['lam'] == lam and d['seed'] == seed), None)
    if d is None: return None
    xs = load_medmnist(ds, splits=('test',))[0]['test'][0][:8].cpu().numpy().transpose(0, 2, 3, 1)
    fig, axs = plt.subplots(1, n, figsize=(1.9 * n, 2.3), squeeze=False)
    for i in range(n):
        ax = axs[0][i]; ax.imshow(xs[i]); tr = d['traces'][i]
        order = [a for a in tr['actions'] if 0 <= a < P]
        for p_ in range(P):
            r, c = divmod(p_, G)
            if p_ not in order: ax.add_patch(plt.Rectangle((c * PS - .5, r * PS - .5), PS, PS, color='k', alpha=0.7))
            else: ax.text(c * PS + PS / 2 - .5, r * PS + PS / 2 - .5, str(order.index(p_) + 1), color='w', ha='center', va='center', fontsize=8, weight='bold')
        ax.set_title(f"true {tr['label']} / pred {tr['pred']}", fontsize=7); ax.axis('off')
    fig.suptitle(f'{DSN[ds]}: glimpse order of {QPOL} (λ={lam}); dark = never inspected', fontsize=8); return fig
for ds in DSS:
    fg = show_traces(ds, MID_LAM, CFG['SEEDS'][0])
    if fg is not None: savefig(fg, f'fig6_trajectories_{ds}')

# Fig 7: adaptivity of the stopping decision
fig, axs = plt.subplots(len(LAMS), len(DSS), figsize=(3.6 * len(DSS), 2.4 * len(LAMS)), squeeze=False)
for i, lam in enumerate(LAMS):
    for j, ds in enumerate(DSS):
        ax = axs[i][j]; w = 0.8 / len(POLS)
        for k_, pol in enumerate(POLS):
            hs = [np.array(d['glimpse_hist']) / np.sum(d['glimpse_hist']) for d in RAWM if d['dataset'] == ds and d['policy'] == pol and d['lam'] == lam]
            if hs: ax.bar(np.arange(1, T + 1) + k_ * w - 0.3, np.mean(hs, 0)[1:], w, color=pcol(pol), label=pol)
        ax.set_title(f'{DSN[ds]}, λ={lam}', fontsize=8); ax.set_xlabel('Glimpses before stopping'); ax.set_ylabel('Fraction of test images')
axs[0][0].legend(fontsize=6); savefig(fig, 'fig7_glimpse_histograms')

# Fig 8: circuit-size ablation with classical references
if CFG['ABLATION'] and len(ab):
    g = ab.groupby('policy').agg(params=('params', 'first'), m=('utility', 'mean'), s=('utility', 'std')).fillna(0).sort_values('params')
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(range(len(g)), g['m'], yerr=g['s'], color=[pcol(p) for p in g.index], capsize=3); ax.set_xticks(range(len(g)))
    ax.set_xticklabels([f'{p}\n({int(n)} par.)' for p, n in zip(g.index, g.params)], fontsize=6.5, rotation=0)
    ax.set_ylim(max(0, (g['m'] - g['s']).min() - 0.05), None)
    ax.set_ylabel('Utility'); ax.set_title(f'{DSN[CFG["ABL_DATASET"]]}, λ={CFG["ABL_LAMBDA"]}'); savefig(fig, 'fig8_ablation')

# Fig 9: teacher baseline, behaviour-cloned policy, final (RL-refined) policy
fig, axs = plt.subplots(1, len(DSS), figsize=(4.2 * len(DSS), 3.3), squeeze=False)
for ax, ds in zip(axs[0], DSS):
    xs_ = np.arange(len(LAMS)); w = 0.8 / (len(POLS) * 2 + 1)
    ref_ = [MAIN[(MAIN.dataset == ds) & (MAIN.lam == l)].base_adapt_util.mean() for l in LAMS]
    ax.bar(xs_, ref_, w, color='gray', label='best adaptive baseline')
    for j, pol in enumerate(POLS):
        for k_, (col, hatch) in enumerate((('bc_utility', '//'), ('utility', ''))):
            v = [MAIN[(MAIN.dataset == ds) & (MAIN.lam == l) & (MAIN.policy == pol)][col].mean() for l in LAMS]
            e = [MAIN[(MAIN.dataset == ds) & (MAIN.lam == l) & (MAIN.policy == pol)][col].std() for l in LAMS]
            ax.bar(xs_ + (1 + 2 * j + k_) * w, v, w, yerr=np.nan_to_num(e), color=pcol(pol), hatch=hatch, alpha=0.6 if hatch else 1, capsize=1.5,
                   label=f'{pol} ' + ('clone only' if hatch else 'final'))
    allv = ref_ + [MAIN[(MAIN.dataset == ds) & (MAIN.lam == l)].utility.mean() for l in LAMS]
    ax.set_ylim(min(allv) - 0.06, max(allv) + 0.03)
    ax.set_xticks(xs_ + 3 * w); ax.set_xticklabels([f'λ={l}' for l in LAMS]); ax.set_title(DSN[ds]); ax.set_ylabel('Utility')
axs[0][0].legend(fontsize=5.5); savefig(fig, 'fig9_clone_vs_final')
