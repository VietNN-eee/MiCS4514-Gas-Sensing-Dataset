# Cross-sensor classification (manuscript v8): Table II; Tables S6 (pairs), S8 (block bootstrap), S9 (alternative normalizations), S10 (SVM), S11 (pair matrix), S12 (robustness); confusion matrices for Fig. 5(a) and Fig. S6
import json, sys, numpy as np
from core import *
from sklearn.metrics import f1_score
SEEDS = [0, 1, 2]
R = {}

def eng_feats(s):
    dv = X[:, s] - airmean[s]; o = []
    for ch in (0, 1):
        v = dv[:, ch]; o += [v.mean(1), v.min(1), v.argmin(1), v.max(1), v.argmax(1), v.sum(1), np.diff(v, axis=1).mean(1), np.abs(np.diff(v, axis=1)).max(1)]
    o.append(X[:, s, 0].mean(1) / X[:, s, 1].mean(1)); return np.column_stack(o)
import core
_featmat = core.featmat
def featmat2(s, mode, pts=None):
    return eng_feats(s) if mode == 'eng' else _featmat(s, mode, pts)
core.featmat = featmat2

def mean_acc(mode, y, model, folds_fn=folds_block, seeds=SEEDS, **kw):
    return float(np.mean([offdiag(transfer(mode, y, model, folds_fn(sd), **kw)) for sd in seeds]))

part = sys.argv[1]
if part == 'table2':
    T = {}
    for mode in ['raw', 'dv', 'ratio', 'z', 'eng']:
        for m in ['SVM', 'LDA', 'kNN', 'RF']:
            T[f'{mode}|{m}|21'] = mean_acc(mode, cls_id, m)
        for m in ['SVM', 'LDA', 'kNN', 'RF', 'GBDT']:
            T[f'{mode}|{m}|3'] = mean_acc(mode, gas, m)
        print(mode, {k: round(v, 3) for k, v in T.items() if k.startswith(mode)}, flush=True)
    # transductive reference (old protocol) for comparison
    T['z|LDA|21|transductive_random'] = mean_acc('z', cls_id, 'LDA', folds_random, strict=False)
    T['z|LDA|21|strict_random'] = mean_acc('z', cls_id, 'LDA', folds_random)
    for k in ['z|LDA', 'raw|SVM', 'raw|LDA']:
        mode, m = k.split('|')
        v = [offdiag(transfer(mode, cls_id, m, folds_block(sd))) for sd in range(8)]
        T[k + '|21|sd8'] = [float(np.mean(v)), float(np.std(v))]
    R = T
elif part == 'pairs':
    acc21 = np.zeros((3, 3)); f121 = np.zeros((3, 3)); acc3 = np.zeros((3, 3)); f13 = np.zeros((3, 3))
    pairs_modes = {}
    for sd in SEEDS:
        a21, P21 = transfer('z', cls_id, 'LDA', folds_block(sd), return_pred=True)
        a3, P3 = transfer('z', gas, 'LDA', folds_block(sd), return_pred=True)
        acc21 += a21 / 3; acc3 += a3 / 3
        for a in range(3):
            for b in range(3):
                f121[a, b] += f1_score(cls_id, P21[(a, b)], average='macro') / 3
                f13[a, b] += f1_score(gas, P3[(a, b)], average='macro') / 3
    for mode, m in [('raw', 'SVM'), ('raw', 'LDA'), ('dv', 'LDA'), ('ratio', 'LDA'), ('z', 'SVM'), ('z', 'kNN'), ('z', 'RF'), ('z', 'LDA')]:
        A = transfer(mode, cls_id, m, folds_block(0))
        pairs_modes[f'{mode}|{m}'] = [float(A[a, b]) for a, b in PAIRS] + [offdiag(A)]
    # confusion S1->S2 (seed 0) for Fig. 5(a), and raw+SVM confusions S1->S2, S2->S1 for Fig. S6
    _, Pz = transfer('z', cls_id, 'LDA', folds_block(0), return_pred=True)
    _, Pr = transfer('raw', cls_id, 'SVM', folds_block(0), return_pred=True)
    np.savez('pred_conf.npz', z01=Pz[(0, 1)], r01=Pr[(0, 1)], r10=Pr[(1, 0)], y=cls_id)
    R = dict(acc21=acc21.tolist(), f121=f121.tolist(), acc3=acc3.tolist(), f13=f13.tolist(), pairs=pairs_modes)
elif part == 'boot':
    g = block_id(); ug = np.unique(g)
    CFG = {'z-LDA': ('z', 'LDA', {}), 'z-kNN': ('z', 'kNN', {}), 'z-SVM': ('z', 'SVM', {}), 'raw-SVM': ('raw', 'SVM', {}),
           'raw-LDA': ('raw', 'LDA', {}), 'dv-SVM': ('dv', 'SVM', {})}
    per = {}
    for k, (mode, m, kw) in CFG.items():
        _, P = transfer(mode, cls_id, m, folds_block(0), return_pred=True, **kw)
        per[k] = np.mean([(P[p] == cls_id) for p in PAIRS], 0)          # per-cycle fraction correct over 6 pairs
    rng = np.random.RandomState(0); B = 4000
    gi = {u: np.where(g == u)[0] for u in ug}
    samp = [np.concatenate([gi[u] for u in rng.choice(ug, len(ug))]) for _ in range(B)]
    def ci(v): bs = np.array([v[s].mean() for s in samp]); return [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    out = {k: ci(v) for k, v in per.items()}
    for a, b in [('z-LDA', 'raw-SVM'), ('z-LDA', 'raw-LDA'), ('z-LDA', 'z-kNN'), ('z-LDA', 'z-SVM')]:
        out[f'{a} - {b}'] = ci(per[a] - per[b])
    R = out
elif part == 'alt':
    T = {}
    for mode in ['mc', 'z', 'minmax', 'robust', 'l2', 'airz']:
        for m in ['LDA', 'SVM', 'kNN']:
            if m == 'kNN' and mode != 'airz': continue
            T[f'{mode}|{m}'] = mean_acc(mode, cls_id, m)
        print(mode, flush=True)
    R = T
elif part == 'svm':
    T = {}
    for name, kw in [('RBF, C=1', dict(C=1)), ('RBF, C=10', dict(C=10)), ('RBF, C=100', dict(C=100)), ('RBF, C=1000', dict(C=1000)),
                     ('Linear, C=1', dict(kernel='linear', C=1)), ('Linear, C=100', dict(kernel='linear', C=100))]:
        T[name] = [mean_acc('raw', cls_id, 'SVM', **kw), mean_acc('z', cls_id, 'SVM', **kw)]
        print(name, T[name], flush=True)
    R = T
elif part == 'robust':
    def temporal(frac, rev):
        tr = []; te = []
        for c in range(21):
            ix = np.where(cls_id == c)[0]; ix = ix[np.argsort(pos[ix])]; k = int(np.ceil(frac * len(ix)))
            a, b = (ix[:k], ix[k:]) if not rev else (ix[len(ix) - k:], ix[:len(ix) - k]); tr += list(a); te += list(b)
        return [(np.array(tr), np.array(te))]
    T = {}
    for mode, m in [('raw', 'SVM'), ('raw', 'LDA'), ('raw', 'kNN'), ('z', 'SVM'), ('z', 'LDA')]:
        r = [offdiag(transfer(mode, cls_id, m, temporal(0.6, rev))) for rev in (False, True)]
        # multi-unit: train on two sensors (each with own training statistics), test on third
        accs = []
        for sd in SEEDS:
            for t_ in range(3):
                src = [s for s in range(3) if s != t_]
                F = {s: featmat(s, mode) for s in range(3)}
                pred = np.full(N, -1)
                for tr, te in folds_block(sd):
                    if mode in PER_SENSOR:
                        Ts = {s: fit_norm(F[s][tr], mode) for s in range(3)}
                    else:
                        Ts = {s: (lambda A: A) for s in range(3)}
                    Xtr = np.vstack([Ts[s](F[s][tr]) for s in src]); ytr = np.concatenate([cls_id[tr]] * 2)
                    mm = make_model(m); Xte = Ts[t_](F[t_][te])
                    if m in ('SVM', 'kNN'):
                        sc = StandardScaler().fit(Xtr); mm.fit(sc.transform(Xtr), ytr); p = mm.predict(sc.transform(Xte))
                    else:
                        mm.fit(Xtr, ytr); p = mm.predict(Xte)
                    pred[te] = p
                accs.append((pred == cls_id).mean())
        T[f'{mode}|{m}'] = r + [float(np.mean(accs))]
        print(mode, m, T[f'{mode}|{m}'], flush=True)
    R = T
json.dump(R, open(f'res_{part}.json', 'w'), indent=1)
print('done', part)
