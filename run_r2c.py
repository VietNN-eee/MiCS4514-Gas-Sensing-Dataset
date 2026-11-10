# Manuscript v8: PLS with LVs selected in nested loops (statistics refitted in every inner fold): Table IV (PLS column), Table S17 (two-source), Table S18 (LOCO)
import json, sys, numpy as np
from core import *
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
SEEDS = [0, 1, 2]; KS = range(1, 7)
y_all = np.log10(conc); bid = block_id()
FZ = {s: featmat(s, 'z') for s in range(3)}
def Z(s, ref): return fit_norm(FZ[s][ref], 'z')(FZ[s])
def pls(k, Xtr, ytr, Xte): return PLSRegression(k, scale=False).fit(Xtr, ytr).predict(Xte).ravel()
def inner_parts(trg):
    ug = np.unique(bid[trg]); ug = np.random.RandomState(0).permutation(ug); return [trg[np.isin(bid[trg], pg)] for pg in np.array_split(ug, 5)]
R = {}
part = sys.argv[1]
if part == 'tab5':     # one source, LV by inner block CV on the source, scaler refit per inner fold
    for gi, g in enumerate(GASES):
        yt, yp, ks = [], [], []
        mk = gas == gi
        for sd in SEEDS:
            for a, b in PAIRS:
                pred = np.zeros(N)
                for tr, te in folds_block(sd):
                    trg, teg = tr[gas[tr] == gi], te[gas[te] == gi]
                    parts = inner_parts(trg); err = []
                    for k in KS:
                        e = []
                        for pt in parts:
                            ref = np.setdiff1d(tr, pt); itr = np.setdiff1d(trg, pt); Fa = Z(a, ref)
                            e.append(((pls(k, Fa[itr], y_all[itr], Fa[pt]) - y_all[pt]) ** 2).mean())
                        err.append(np.mean(e))
                    k = int(np.argmin(err)) + 1; ks.append(k)
                    pred[teg] = pls(k, Z(a, tr)[trg], y_all[trg], Z(b, tr)[teg])
                yt.append(y_all[mk]); yp.append(pred[mk])
        yt = np.concatenate(yt); yp = np.concatenate(yp)
        R[g] = dict(r2=float(r2_score(yt, yp)), rmse=float(np.sqrt(np.mean((yt - yp) ** 2))), meanLV=float(np.mean(ks)))
        print(g, R[g], flush=True)
if part == 'two':      # two sources, LV by inner block-grouped cross-prediction (train u, validate v on held-out blocks), scaler refit
    for gi, g in enumerate(GASES):
        yt, yp, ks = [], [], []
        mk = gas == gi
        for sd in SEEDS:
            for c in range(3):
                a, b = [s for s in range(3) if s != c]; pred = np.zeros(N)
                for tr, te in folds_block(sd):
                    trg, teg = tr[gas[tr] == gi], te[gas[te] == gi]
                    parts = inner_parts(trg); err = []
                    for k in KS:
                        e = []
                        for pt in parts:
                            ref = np.setdiff1d(tr, pt); itr = np.setdiff1d(trg, pt); F = {s: Z(s, ref) for s in (a, b)}
                            for u, v in ((a, b), (b, a)): e.append(((pls(k, F[u][itr], y_all[itr], F[v][pt]) - y_all[pt]) ** 2).mean())
                        err.append(np.mean(e))
                    k = int(np.argmin(err)) + 1; ks.append(k)
                    F = {s: Z(s, tr) for s in range(3)}
                    pred[teg] = pls(k, np.vstack([F[a][trg], F[b][trg]]), np.concatenate([y_all[trg]] * 2), F[c][teg])
                yt.append(y_all[mk]); yp.append(pred[mk])
        yt = np.concatenate(yt); yp = np.concatenate(yp)
        R[g] = dict(r2=float(r2_score(yt, yp)), rmse=float(np.sqrt(np.mean((yt - yp) ** 2))), meanLV=float(np.mean(ks)))
        print(g, R[g], flush=True)
if part == 'loco':     # outer LOCO; inner leave-one-level-out with scaler refit; one source (PLS1) and two sources (PLS2, RF2)
    for gi, g in enumerate(GASES):
        mk = gas == gi; levels = np.unique(conc[mk]); rows = []
        for held in levels[1:-1]:
            te_all = np.where(mk & (conc == held))[0]; ref = np.where(~(mk & (conc == held)))[0]; trg = np.where(mk & (conc != held))[0]
            inner = [l for l in levels[1:-1] if l != held]
            Fi = {l: {s: Z(s, np.setdiff1d(ref, np.where(mk & (conc == l))[0])) for s in range(3)} for l in inner}
            Fo = {s: Z(s, ref) for s in range(3)}
            e = {'RF1': [], 'PLS1': [], 'RF2': [], 'PLS2': []}; k1s, k2s = [], []
            for a, b in PAIRS:
                err = []
                for k in KS:
                    err.append(np.mean([((pls(k, Fi[l][a][trg[conc[trg] != l]], y_all[trg[conc[trg] != l]], Fi[l][a][trg[conc[trg] == l]]) - y_all[trg[conc[trg] == l]]) ** 2).mean() for l in inner]))
                k = int(np.argmin(err)) + 1; k1s.append(k)
                e['PLS1'].append(((pls(k, Fo[a][trg], y_all[trg], Fo[b][te_all]) - y_all[te_all]) ** 2).mean())
                e['RF1'].append(((RFR(300, random_state=0, n_jobs=-1).fit(Fo[a][trg], y_all[trg]).predict(Fo[b][te_all]) - y_all[te_all]) ** 2).mean())
            for c in range(3):
                a, b = [s for s in range(3) if s != c]; err = []
                for k in KS:
                    ee = []
                    for l in inner:
                        itr = trg[conc[trg] != l]; ite = trg[conc[trg] == l]
                        for u, v in ((a, b), (b, a)): ee.append(((pls(k, Fi[l][u][itr], y_all[itr], Fi[l][v][ite]) - y_all[ite]) ** 2).mean())
                    err.append(np.mean(ee))
                k = int(np.argmin(err)) + 1; k2s.append(k)
                Xtr = np.vstack([Fo[a][trg], Fo[b][trg]]); ytr = np.concatenate([y_all[trg]] * 2)
                e['PLS2'].append(((pls(k, Xtr, ytr, Fo[c][te_all]) - y_all[te_all]) ** 2).mean())
                e['RF2'].append(((RFR(300, random_state=0, n_jobs=-1).fit(Xtr, ytr).predict(Fo[c][te_all]) - y_all[te_all]) ** 2).mean())
            rows.append(dict(held=float(held), **{k_: float(np.sqrt(np.mean(v))) for k_, v in e.items()}, LV1=float(np.mean(k1s)), LV2=float(np.mean(k2s))))
            print(g, {k_: round(v, 3) for k_, v in rows[-1].items()}, flush=True)
        R[g] = dict(rows=rows, pooled={k_: float(np.sqrt(np.mean([r[k_] ** 2 for r in rows]))) for k_ in ('RF1', 'PLS1', 'RF2', 'PLS2')},
                    LV1=float(np.mean([r['LV1'] for r in rows])), LV2=float(np.mean([r['LV2'] for r in rows])))
        print(g, 'pooled', {k_: round(v, 3) for k_, v in R[g]['pooled'].items()}, flush=True)
json.dump(R, open(f'res_r2_{part}.json', 'w'), indent=1); print('done', part)
