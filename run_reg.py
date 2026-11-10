# Cross-sensor quantification (manuscript v8): Table IV, Fig. 7, Table S16 (per-level error), Table S17 (PLS LVs), Table S18 (LOCO, fixed-LV columns)
import json, sys, numpy as np
from core import *
from sklearn.ensemble import RandomForestRegressor as RFR, GradientBoostingRegressor as GBR
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
SEEDS = [0, 1, 2]
y_all = np.log10(np.where(conc > 0, conc, 1))
bid = block_id()

def norm_pair(mode, a, b, ref_a, ref_b):
    Fa, Fb = featmat(a, mode), featmat(b, mode)
    if mode in PER_SENSOR:
        return fit_norm(Fa[ref_a], mode)(Fa), fit_norm(Fb[ref_b], mode)(Fb)
    return Fa, Fb

def pls_inner_k(Xs, ys, groups, kmax=6):
    """choose LV by grouped (block) CV inside the source training data only"""
    ug = np.unique(groups); rng = np.random.RandomState(0); ug = rng.permutation(ug)
    parts = np.array_split(ug, 5); best = None
    for k in range(1, kmax + 1):
        e = []
        for pg in parts:
            te = np.isin(groups, pg); tr = ~te
            if tr.sum() <= k or te.sum() == 0: continue
            e.append(((PLSRegression(k, scale=False).fit(Xs[tr], ys[tr]).predict(Xs[te]).ravel() - ys[te]) ** 2).mean())
        if best is None or np.mean(e) < best[0]: best = (np.mean(e), k)
    return best[1]

def model_fit_predict(name, Xtr, ytr, Xte, groups=None, k=None):
    if name == 'RF': return RFR(300, random_state=0, n_jobs=-1).fit(Xtr, ytr).predict(Xte), None
    if name == 'GBDT': return GBR(random_state=0).fit(Xtr, ytr).predict(Xte), None
    if name.startswith('PLS'):
        kk = k if k is not None else pls_inner_k(Xtr, ytr, groups)
        return PLSRegression(kk, scale=False).fit(Xtr, ytr).predict(Xte).ravel(), kk

part = sys.argv[1]
R = {}
if part == 'table5':
    # cross-sensor and same-sensor regression; z statistics from training cycles (all gases) of each fold
    store = {}
    for mode in ['raw', 'dv', 'ratio', 'z']:
        for mdl in ['RF', 'GBDT', 'PLS']:
            for gi, g in enumerate(GASES):
                yt, yp, ks = [], [], []
                same = []
                perpair = {p: [] for p in PAIRS}
                levels = {}
                for sd in SEEDS:
                    for a in range(3):
                        for b in range(3):
                            pred = np.zeros(N)
                            for tr, te in folds_block(sd):
                                Fa, Fb = norm_pair(mode, a, b, tr, tr)
                                trg, teg = tr[gas[tr] == gi], te[gas[te] == gi]
                                p, k = model_fit_predict(mdl, Fa[trg], y_all[trg], Fb[teg], groups=bid[trg])
                                pred[teg] = p
                                if k: ks.append(k)
                            mk = gas == gi
                            if a == b:
                                same.append(r2_score(y_all[mk], pred[mk]))
                            else:
                                yt.append(y_all[mk]); yp.append(pred[mk]); perpair[(a, b)].append(pred[mk])
                yt = np.concatenate(yt); yp = np.concatenate(yp)
                key = f'{mode}|{mdl}|{g}'
                R[key] = dict(r2=float(r2_score(yt, yp)), rmse=float(np.sqrt(np.mean((yt - yp) ** 2))),
                              same_r2=float(np.mean(same)), meanLV=float(np.mean(ks)) if ks else None)
                if mode == 'z' and mdl == 'RF':
                    mk = gas == gi; ytrue = y_all[mk]
                    R[key]['pair_rmse'] = {f'{a}{b}': float(np.sqrt(np.mean([(p - ytrue) ** 2 for p in perpair[(a, b)]]))) for a, b in PAIRS}
                    # per-level median relative error (cross-sensor)
                    cs = np.unique(conc[mk]); cc = conc[mk]
                    R[key]['mre'] = {f'{c:g}': float(np.median(np.concatenate([np.abs(10 ** p[cc == c] - c) / c for k_ in PAIRS for p in perpair[k_]])) * 100) for c in cs}
                    np.savez(f'reg_{g}.npz', yt=yt, yp=yp)
                print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R[key].items() if k in ('r2', 'rmse', 'same_r2', 'meanLV')}, flush=True)
    # same-sensor median relative error for z+RF (seed 0) for Fig. 7(d)-(f)
elif part == 'samemre':
    for gi, g in enumerate(GASES):
        mk = gas == gi; preds = []
        for sd in SEEDS:
            for a in range(3):
                pred = np.zeros(N)
                for tr, te in folds_block(sd):
                    Fa, _ = norm_pair('z', a, a, tr, tr)
                    trg, teg = tr[gas[tr] == gi], te[gas[te] == gi]
                    pred[teg] = RFR(300, random_state=0, n_jobs=-1).fit(Fa[trg], y_all[trg]).predict(Fa[teg])
                preds.append(pred[mk])
        cc = conc[mk]
        R[g] = {f'{c:g}': float(np.median(np.concatenate([np.abs(10 ** p[cc == c] - c) / c for p in preds])) * 100) for c in np.unique(cc)}
        print(g, R[g], flush=True)
elif part == 'plsk':
    # PLS sensitivity to fixed LV (seed 0) + nested selection on two source sensors
    for mode in ['raw', 'z']:
        for gi, g in enumerate(GASES):
            mk = gas == gi
            for k in range(1, 7):
                yt, yp = [], []
                for a, b in PAIRS:
                    pred = np.zeros(N)
                    for tr, te in folds_block(0):
                        Fa, Fb = norm_pair(mode, a, b, tr, tr)
                        trg, teg = tr[gas[tr] == gi], te[gas[te] == gi]
                        pred[teg] = PLSRegression(k, scale=False).fit(Fa[trg], y_all[trg]).predict(Fb[teg]).ravel()
                    yt.append(y_all[mk]); yp.append(pred[mk])
                yt = np.concatenate(yt); yp = np.concatenate(yp)
                R[f'{mode}|{g}|{k}'] = [float(r2_score(yt, yp)), float(np.sqrt(np.mean((yt - yp) ** 2)))]
            # nested: choose LV by cross-prediction between the two source sensors, test on third
            yt, yp, ks = [], [], []
            for sd in SEEDS:
                for c in range(3):
                    a, b = [s for s in range(3) if s != c]
                    pred = np.zeros(N)
                    for tr, te in folds_block(sd):
                        F = {s: (fit_norm(featmat(s, mode)[tr], mode)(featmat(s, mode)) if mode in PER_SENSOR else featmat(s, mode)) for s in range(3)}
                        trg, teg = tr[gas[tr] == gi], te[gas[te] == gi]
                        errs = [sum(((PLSRegression(k, scale=False).fit(F[u][trg], y_all[trg]).predict(F[v][trg]).ravel() - y_all[trg]) ** 2).mean()
                                    for u, v in ((a, b), (b, a))) for k in range(1, 7)]
                        k = int(np.argmin(errs)) + 1; ks.append(k)
                        pred[teg] = PLSRegression(k, scale=False).fit(np.vstack([F[a][trg], F[b][trg]]), np.concatenate([y_all[trg]] * 2)).predict(F[c][teg]).ravel()
                    yt.append(y_all[mk]); yp.append(pred[mk])
            yt = np.concatenate(yt); yp = np.concatenate(yp)
            R[f'{mode}|{g}|nested'] = [float(r2_score(yt, yp)), float(np.sqrt(np.mean((yt - yp) ** 2))), float(np.mean(ks))]
            print(mode, g, R[f'{mode}|{g}|nested'], flush=True)
elif part == 'loco':
    # Outer LOCO: withheld interior level excluded from ALL fitting (normalisation statistics, model, LV selection).
    # PLS LV chosen by inner leave-one-level-out on the SOURCE sensor over the remaining interior levels.
    for gi, g in enumerate(GASES):
        mk = gas == gi; levels = np.unique(conc[mk])
        rows = []
        for held in levels[1:-1]:
            te_all = np.where(mk & (conc == held))[0]
            ref = np.where(~(mk & (conc == held)))[0]          # all other cycles (any gas) for z statistics
            trg = np.where(mk & (conc != held))[0]
            e = {'RF': [], 'PLSnest': [], 'PLS2': [], 'PLS3': []}; kk = []
            for a, b in PAIRS:
                Fa, Fb = norm_pair('z', a, b, ref, ref)
                p = RFR(300, random_state=0, n_jobs=-1).fit(Fa[trg], y_all[trg]).predict(Fb[te_all]); e['RF'].append(((p - y_all[te_all]) ** 2).mean())
                # inner leave-one-level-out (interior remaining levels) on source sensor only
                inner_lv = [l for l in levels[1:-1] if l != held]; errs = []
                for k in range(1, 7):
                    ee = []
                    for l in inner_lv:
                        itr = trg[conc[trg] != l]; ite = trg[conc[trg] == l]
                        ee.append(((PLSRegression(k, scale=False).fit(Fa[itr], y_all[itr]).predict(Fa[ite]).ravel() - y_all[ite]) ** 2).mean())
                    errs.append(np.mean(ee))
                k = int(np.argmin(errs)) + 1; kk.append(k)
                p = PLSRegression(k, scale=False).fit(Fa[trg], y_all[trg]).predict(Fb[te_all]).ravel(); e['PLSnest'].append(((p - y_all[te_all]) ** 2).mean())
                for nm, kf in (('PLS2', 2), ('PLS3', 3)):
                    p = PLSRegression(kf, scale=False).fit(Fa[trg], y_all[trg]).predict(Fb[te_all]).ravel(); e[nm].append(((p - y_all[te_all]) ** 2).mean())
            rows.append(dict(held=float(held), **{k: float(np.sqrt(np.mean(v))) for k, v in e.items()}, meanLV=float(np.mean(kk))))
            print(g, rows[-1], flush=True)
        pooled = {k: float(np.sqrt(np.mean([r[k] ** 2 for r in rows]))) for k in ('RF', 'PLSnest', 'PLS2', 'PLS3')}
        R[g] = dict(rows=rows, pooled=pooled, meanLV=float(np.mean([r['meanLV'] for r in rows])))
json.dump(R, open(f'res_reg_{part}.json', 'w'), indent=1)
print('done', part)

if part == 'loco2':
    # LOCO with two labelled source units: LV chosen by cross-prediction between the two sources over the training levels;
    # model trained on both sources, tested on the third unit at the withheld level. RF trained on both sources for comparison.
    R = {}
    for gi, g in enumerate(GASES):
        mk = gas == gi; levels = np.unique(conc[mk]); rows = []
        for held in levels[1:-1]:
            te_all = np.where(mk & (conc == held))[0]
            ref = np.where(~(mk & (conc == held)))[0]
            trg = np.where(mk & (conc != held))[0]
            e = {'RF2': [], 'PLS2src': []}; kk = []
            for c in range(3):
                a, b = [s for s in range(3) if s != c]
                F = {s: fit_norm(featmat(s, 'z')[ref], 'z')(featmat(s, 'z')) for s in range(3)}
                errs = [sum(((PLSRegression(k, scale=False).fit(F[u][trg], y_all[trg]).predict(F[v][trg]).ravel() - y_all[trg]) ** 2).mean()
                            for u, v in ((a, b), (b, a))) for k in range(1, 7)]
                k = int(np.argmin(errs)) + 1; kk.append(k)
                Xtr = np.vstack([F[a][trg], F[b][trg]]); ytr = np.concatenate([y_all[trg]] * 2)
                p = PLSRegression(k, scale=False).fit(Xtr, ytr).predict(F[c][te_all]).ravel(); e['PLS2src'].append(((p - y_all[te_all]) ** 2).mean())
                p = RFR(300, random_state=0, n_jobs=-1).fit(Xtr, ytr).predict(F[c][te_all]); e['RF2'].append(((p - y_all[te_all]) ** 2).mean())
            rows.append(dict(held=float(held), **{k_: float(np.sqrt(np.mean(v))) for k_, v in e.items()}, meanLV=float(np.mean(kk))))
            print(g, rows[-1], flush=True)
        pooled = {k_: float(np.sqrt(np.mean([r[k_] ** 2 for r in rows]))) for k_ in ('RF2', 'PLS2src')}
        R[g] = dict(rows=rows, pooled=pooled, meanLV=float(np.mean([r['meanLV'] for r in rows])))
    json.dump(R, open('res_reg_loco2.json', 'w'), indent=1); print('done loco2')
