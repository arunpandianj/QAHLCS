"""Environment rollout, training protocols, evaluation and non-learned baselines."""
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.distributions import Categorical

from .config import CFG, DEVICE, ENC_SIG, G, P, RESULTS, T, set_seed
from .data import CENTER, RASTER
from .models import IN_DIM, Encoder, make_critic, make_head

def random_masks(B, k):
    return (torch.rand(B, P, device=DEVICE).argsort(1).argsort(1) < k.view(-1, 1)).float()

def sample_idx(data, B):                       # class-balanced (or uniform) sampling of training images
    return torch.multinomial(data['sw'], B, replacement=True)

def bacc_t(pred, y, C):
    r = [(pred[y == c] == c).float().mean() for c in range(C) if (y == c).any()]
    return torch.stack(r).mean().item()

def pretrain_encoder(data, C, seed):
    set_seed(10_000 + seed)
    enc = Encoder(C).to(DEVICE); opt = torch.optim.Adam(enc.parameters(), lr=CFG['PRE_LR'], weight_decay=1e-4)
    steps = CFG['PRE_STEPS']; sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    xs, ys = data['train']; xv, yv = data['val']
    g = torch.Generator(device=DEVICE).manual_seed(123)
    vmasks = {k: (torch.rand(len(xv), P, generator=g, device=DEVICE).argsort(1).argsort(1) < k).float() for k in (2, 4, 8)}
    def val_score():
        enc.eval(); sc = []
        with torch.no_grad():
            for k, m in vmasks.items():
                pr = torch.cat([enc(xv[i:i + 1024], m[i:i + 1024])[1].argmax(1) for i in range(0, len(xv), 1024)])
                sc.append(bacc_t(pr, yv, C))
        enc.train(); return float(np.mean(sc))
    best, best_state = -1, None
    for it in range(1, steps + 1):
        idx = sample_idx(data, CFG['PRE_BS'])
        x = xs[idx]; m = random_masks(len(idx), torch.randint(0, 11, (len(idx),), device=DEVICE))
        if random.random() < 0.5: x = x.flip(3); m = m.view(-1, G, G).flip(2).reshape(-1, P)
        if random.random() < 0.5: x = x.flip(2); m = m.view(-1, G, G).flip(1).reshape(-1, P)
        loss = F.cross_entropy(enc(x, m, aug=True)[1], ys[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % max(1, steps // 8) == 0 or it == steps:
            a = val_score()
            if a > best: best, best_state = a, {k_: v.clone() for k_, v in enc.state_dict().items()}
    enc.load_state_dict(best_state); enc.eval()
    for p_ in enc.parameters(): p_.requires_grad_(False)
    print(f'   encoder pre-trained, best val balanced acc (random 2/4/8 patches): {best:.3f}')
    return enc

def get_encoder(ds, seed, data, C):
    f = RESULTS / 'enc' / f'{ds}_s{seed}_{ENC_SIG}.pt'
    enc = Encoder(C).to(DEVICE)
    if f.exists():
        enc.load_state_dict(torch.load(f, map_location=DEVICE)); enc.eval()
        for p_ in enc.parameters(): p_.requires_grad_(False)
        return enc
    enc = pretrain_encoder(data, C, seed); torch.save(enc.state_dict(), f); return enc

CONF_ON = True      # False removes the three confidence features (they are set to zero; input size unchanged)

def policy_inputs(h, cl, mask, t):
    """Policy observation: encoder features, step fraction, revealed mask, and 3 classifier-confidence features."""
    B = h.shape[0]; pr = F.softmax(cl, 1); top2 = pr.topk(2, 1).values
    ent = -(pr * torch.log(pr.clamp_min(1e-8))).sum(1, keepdim=True) / math.log(pr.shape[1])
    feats = torch.cat([top2[:, :1], ent, top2[:, :1] - top2[:, 1:2]], 1)
    if not CONF_ON: feats = feats * 0
    return torch.cat([h, torch.full((B, 1), t / T, device=h.device), mask, feats], 1)

def rollout(enc, head, critic, x, y, lam, greedy=False, grad=False):
    B, dev = x.shape[0], x.device
    mask = torch.zeros(B, P, device=dev); active = torch.ones(B, dtype=torch.bool, device=dev)
    glimpses = torch.zeros(B, device=dev); final_cl = torch.zeros(B, enc.cls.out_features, device=dev)
    stop_t = torch.full((B,), T, dtype=torch.long, device=dev)
    LP, EN, VV, RR, AC, AA = [], [], [], [], [], []
    with (torch.enable_grad() if grad else torch.no_grad()):
        for t in range(T + 1):
            with torch.no_grad(): h, cl = enc(x, mask)
            inp = policy_inputs(h, cl, mask, t)
            logits = head(inp)
            v = critic(inp).squeeze(1) if critic is not None else torch.zeros(B, device=dev)
            invalid = torch.cat([mask > 0, torch.zeros(B, 1, dtype=torch.bool, device=dev)], 1)
            if t == 0: invalid[:, P] = True            # at least one glimpse
            if t == T: invalid[:, :P] = True           # budget exhausted: forced stop
            logits = logits.masked_fill(invalid, -1e9)
            dist = Categorical(logits=logits)
            a = logits.argmax(1) if greedy else dist.sample()
            lp, ent = dist.log_prob(a), dist.entropy()
            is_stop = a == P
            correct = (cl.argmax(1) == y).float()
            r = torch.where(active, torch.where(is_stop, correct, torch.full_like(correct, -lam)), torch.zeros_like(correct))
            move, stopping = active & ~is_stop, active & is_stop
            final_cl = torch.where(stopping[:, None], cl, final_cl)
            stop_t = torch.where(stopping, torch.full_like(stop_t, t), stop_t)
            glimpses = glimpses + move.float()
            mask = (mask + F.one_hot(a.clamp(max=P - 1), P).float() * move[:, None].float()).clamp(max=1)
            LP.append(lp); EN.append(ent); VV.append(v); RR.append(r); AC.append(active)
            AA.append(torch.where(active, a, torch.full_like(a, -1)))
            active = active & ~is_stop
    return dict(lp=torch.stack(LP), ent=torch.stack(EN), v=torch.stack(VV), r=torch.stack(RR), act=torch.stack(AC),
                actions=torch.stack(AA), final_cl=final_cl, glimpses=glimpses, stop_t=stop_t)

@torch.no_grad()
def teacher_batch(enc, x, order, tau):
    """Teacher = fixed (validation-optimised) patch order + stop when classifier confidence >= tau."""
    B = x.shape[0]; ord_t = torch.tensor(order, device=DEVICE)
    r = torch.empty(P, dtype=torch.long, device=DEVICE); r[ord_t] = torch.arange(P, device=DEVICE)
    rank = r.unsqueeze(0).expand(B, P); inps, masks, confs = [], [], []
    for t in range(T + 1):
        mask = (rank < t).float(); h, cl = enc(x, mask)
        inps.append(policy_inputs(h, cl, mask, t)); masks.append(mask); confs.append(F.softmax(cl, 1).max(1).values)
    conf = torch.stack(confs)                                   # (T+1, B)
    hit = conf[1:] >= tau; hit[-1] = True
    s_star = hit.float().argmax(0) + 1                          # first step t>=1 that reaches the threshold
    acts, am = [], []
    for t in range(T + 1):
        acts.append(torch.where(s_star == t, torch.full_like(s_star, P), ord_t[t].expand_as(s_star))); am.append(t <= s_star)
    return torch.stack(inps), torch.stack(masks), torch.stack(acts), torch.stack(am)

def bc_pretrain(head, enc, data, order, tau):
    opt = torch.optim.Adam(head.parameters(), lr=CFG['BC_LR']); U = CFG['BC_UPDATES']
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, U, eta_min=CFG['BC_LR'] * 0.1); xs, ys = data['train']
    for u in range(U):
        idx = sample_idx(data, CFG['RL_BS'])
        inps, masks, acts, am = teacher_batch(enc, xs[idx], order, tau); B = len(idx)
        tt = torch.arange(T + 1, device=DEVICE).repeat_interleave(B)
        logits = head(inps.reshape(-1, IN_DIM))
        invalid = torch.cat([masks.reshape(-1, P) > 0, torch.zeros(len(tt), 1, dtype=torch.bool, device=DEVICE)], 1)
        invalid[tt == 0, P] = True; invalid[tt == T, :P] = True
        logp = F.log_softmax(logits.masked_fill(invalid, -1e9), 1); valid = (~invalid).float()
        a = acts.reshape(-1); keep = am.reshape(-1)
        nll = -logp.gather(1, a.unsqueeze(1)).squeeze(1)
        smooth = -(logp * valid).sum(1) / valid.sum(1)
        eps = CFG['BC_SMOOTH']; loss = (((1 - eps) * nll + eps * smooth) * keep.float()).sum() / keep.float().sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), CFG['GRAD_CLIP']); opt.step(); sched.step()

def clone_state(m): return {k: v.clone() for k, v in m.state_dict().items()}

def train_policy(enc, data, name, lam, C, order, tau, clone=True, lr=None, updates=None, ent=None, crit_warm=None):
    lr = lr or CFG['RL_LR']; U = updates or CFG['RL_UPDATES']; ent = ent or (CFG['ENT_START'], CFG['ENT_END'])
    cw = CFG['CRIT_WARM'] if crit_warm is None else crit_warm
    head = make_head(name).to(DEVICE); critic = make_critic().to(DEVICE); xs, ys = data['train']
    if clone: bc_pretrain(head, enc, data, order, tau)
    head.eval()
    init_test = evaluate(enc, head, data['test'], lam, C)
    best_u = evaluate(enc, head, data['val_sub'], lam, C)['utility']; best = (clone_state(head), clone_state(critic), 'clone' if clone else 'init')
    params = [p for p in list(head.parameters()) + list(critic.parameters()) if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, U, eta_min=lr * 0.1); curve = []
    for u in range(U):
        ent_c = ent[0] + (ent[1] - ent[0]) * u / max(1, U - 1)
        idx = sample_idx(data, CFG['RL_BS'])
        o = rollout(enc, head, critic, xs[idx], ys[idx], lam, greedy=False, grad=True)
        r, act = o['r'], o['act'].float()
        Gt = torch.flip(torch.cumsum(torch.flip(r, [0]), 0), [0])
        adv = Gt - o['v'].detach(); m = o['act']
        advn = (adv - adv[m].mean()) / (adv[m].std() + 1e-6)
        n_act = act.sum(); vl = 0.5 * (((o['v'] - Gt) ** 2) * act).sum() / n_act
        if u < cw: loss = vl
        else: loss = -(o['lp'] * advn * act).sum() / n_act + vl - ent_c * (o['ent'] * act).sum() / n_act
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, CFG['GRAD_CLIP']); opt.step(); sched.step()
        if u % 20 == 0 or u == U - 1:
            curve.append(dict(update=u, ret=r.sum(0).mean().item(), glimpses=o['glimpses'].mean().item()))
        if (u + 1) % CFG['VAL_EVERY'] == 0 or u == U - 1:
            head.eval(); vu = evaluate(enc, head, data['val_sub'], lam, C)['utility']
            if vu > best_u: best_u, best = vu, (clone_state(head), clone_state(critic), f'update{u + 1}')
    head.load_state_dict(best[0]); head.eval()
    return head, critic, curve, init_test, best[2], best_u

def ece_score(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)

@torch.no_grad()
def evaluate(enc, head, split, lam, C, detail=False):
    xs, ys = split; preds, confs, gls, stops, acts, probs_l = [], [], [], [], [], []
    for i in range(0, len(xs), CFG['EVAL_BS']):
        o = rollout(enc, head, None, xs[i:i + CFG['EVAL_BS']], ys[i:i + CFG['EVAL_BS']], lam, greedy=True)
        pr = F.softmax(o['final_cl'], 1)
        probs_l.append(pr); preds.append(pr.argmax(1)); confs.append(pr.max(1).values); gls.append(o['glimpses']); stops.append(o['stop_t'])
        acts.append(o['actions'].T)
    pred = torch.cat(preds).cpu().numpy(); conf = torch.cat(confs).cpu().numpy(); gl = torch.cat(gls).cpu().numpy()
    yt = ys.cpu().numpy(); correct = (pred == yt).astype(float)
    bacc = float(balanced_accuracy_score(yt, pred))
    res = dict(acc=float(correct.mean()), bacc=bacc, f1=float(f1_score(yt, pred, average='macro')), ece=ece_score(conf, correct),
               glimpses=float(gl.mean()), utility=bacc - lam * float(gl.mean()), utility_acc=float(correct.mean() - lam * gl.mean()))
    PA = torch.cat(probs_l).cpu().numpy().astype('float64'); PA = PA / PA.sum(1, keepdims=True)
    try: res['auroc'] = float(roc_auc_score(yt, PA, multi_class='ovr', average='macro'))
    except Exception: res['auroc'] = float('nan')
    res['recall'] = [float((pred[yt == c] == c).mean()) if (yt == c).any() else None for c in range(PA.shape[1])]
    if detail:
        A = torch.cat(acts).cpu().numpy()                                    # N x (T+1), -1 inactive, P = stop
        pc = np.zeros((T + 1, P)); cc = np.zeros((C, P))
        for t in range(T + 1):
            for p_ in range(P): pc[t, p_] = (A[:, t] == p_).sum()
        for c in range(C): cc[c] = [(A[yt == c] == p_).sum() for p_ in range(P)]
        res.update(glimpse_hist=np.bincount(gl.astype(int), minlength=T + 1).tolist(), patch_counts=pc.tolist(),
                   class_patch_counts=cc.tolist(), class_glimpses=[float(gl[yt == c].mean()) if (yt == c).any() else None for c in range(C)],
                   traces=[dict(actions=A[i].tolist(), label=int(yt[i]), pred=int(pred[i])) for i in range(8)])
    return res

# ---------------- non-RL baselines: fixed orders, validation-optimised static order, confidence-threshold stopping
@torch.no_grad()
def order_probs(enc, split, order, C):
    xs, ys = split; N = len(xs); ts = list(range(T + 1)) + [P]
    probs = torch.zeros(len(ts), N, C, device=DEVICE)
    for i in range(0, N, CFG['EVAL_BS']):
        xb = xs[i:i + CFG['EVAL_BS']]; B = len(xb)
        if order is None: rank = torch.rand(B, P, device=DEVICE).argsort(1).argsort(1)
        else:
            r = torch.empty(P, dtype=torch.long, device=DEVICE); r[torch.tensor(order, device=DEVICE)] = torch.arange(P, device=DEVICE)
            rank = r.unsqueeze(0).expand(B, P)
        for j, t in enumerate(ts): probs[j, i:i + B] = F.softmax(enc(xb, (rank < t).float())[1], 1)
    return probs                                                   # (T+2, N, C)

def family_metrics(probs, ys):
    yt = ys.cpu().numpy(); out = {'fixed': {}, 'thresh': {}}
    def met(pred, g):
        return dict(acc=float((pred == yt).mean()), bacc=float(balanced_accuracy_score(yt, pred)),
                    f1=float(f1_score(yt, pred, average='macro')), glimpses=float(g))
    for j, k in enumerate(list(range(T + 1)) + [P]): out['fixed'][str(k)] = met(probs[j].argmax(1).cpu().numpy(), k)
    conf, pr = probs[1:T + 1].max(2)                               # (T, N)
    for tau in CFG['THRESHOLDS']:
        hit = conf >= tau; hit[-1] = True
        si = hit.float().argmax(0)                                 # first step that reaches the threshold
        pred = pr[si, torch.arange(conf.shape[1], device=DEVICE)].cpu().numpy()
        out['thresh'][str(tau)] = met(pred, (si + 1).float().mean().item())
    return out

@torch.no_grad()
def greedy_static_order(enc, val, C):
    xs, ys = val; sel = torch.randperm(len(xs))[:4000].to(DEVICE); xs, ys = xs[sel], ys[sel]; n = len(xs)
    cnt = torch.bincount(ys, minlength=C).float(); w = 1.0 / cnt[ys]; w = w / w.sum(); chosen = []
    for _ in range(T):
        best, bp = -1.0, None
        for p in range(P):
            if p in chosen: continue
            m = torch.zeros(n, P, device=DEVICE); m[:, chosen + [p]] = 1; score = 0.0
            for i in range(0, n, CFG['EVAL_BS']):
                pr = F.softmax(enc(xs[i:i + CFG['EVAL_BS']], m[i:i + CFG['EVAL_BS']])[1], 1)
                score += (w[i:i + CFG['EVAL_BS']] * pr.gather(1, ys[i:i + CFG['EVAL_BS']].unsqueeze(1)).squeeze(1)).sum().item()
            if score > best: best, bp = score, p
        chosen.append(bp)
    return chosen + [p for p in range(P) if p not in chosen]

def eval_baselines(enc, data, C):
    gorder = greedy_static_order(enc, data['val'], C)
    orders = {'random': None, 'raster': RASTER, 'center': CENTER, 'greedy': gorder}
    out = {'greedy_order': gorder}
    for name, order in orders.items():
        out[name] = {sp: family_metrics(order_probs(enc, data[sp], order, C), data[sp][1]) for sp in ('val', 'test')}
    return out
