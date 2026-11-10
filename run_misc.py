# Manuscript v8: Fig. 6 and Table S13 (ablation), Table III and Table S14 (reference design), Table S15 (DS/CORAL), Tables S4-S5 (sensitivity, resolution)
import json, sys, numpy as np
from core import *
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier as RFC
SEEDS = [0, 1, 2]
part = sys.argv[1]; R = {}
conc_rank = np.zeros(N, int)
for gi in range(3):
    mk = gas == gi; conc_rank[mk] = np.searchsorted(np.unique(conc[mk]), conc[mk])
low_mask = np.array([c <= np.median(conc[gas == g]) for c, g in zip(conc, gas)])

if part == 'ablation':
    cfg = {'z': ('z', 'LDA', {}), 'r': ('raw', 'SVM', {})}
    def acc_pts(pts, mode, m, seeds=SEEDS):
        return float(np.mean([offdiag(transfer(mode, cls_id, m, folds_block(sd), pts=pts)) for sd in seeds]))
    for k, (mode, m, _) in cfg.items():
        single = [acc_pts([p], mode, m) for p in range(45)]
        R[f'{k}|single'] = single
        R[f'{k}|n'] = {n: acc_pts(list(np.linspace(0, 44, n).round().astype(int)), mode, m) for n in (3, 5, 15, 45)}
        R[f'{k}|regions'] = {nm: acc_pts(list(r), mode, m) for nm, r in [('first', range(0, 15)), ('middle', range(15, 30)), ('last', range(30, 45))]}
        print(k, np.mean(single), np.max(single), int(np.argmax(single)) + 1, R[f'{k}|n'], R[f'{k}|regions'], flush=True)
    # single point selected INSIDE training: for each outer fold choose the point with best inner same-sensor CV on the source
    for k, (mode, m, _) in cfg.items():
        accs, chosen = [], []
        for sd in SEEDS:
            for a, b in PAIRS:
                pred = np.full(N, -1)
                Fa_all = {p: featmat(a, mode, [p]) for p in range(45)}; Fb_all = {p: featmat(b, mode, [p]) for p in range(45)}
                for tr, te in folds_block(sd):
                    best = None
                    inner = folds_block(sd + 100, idx=tr)
                    for p in range(45):
                        Fa = Fa_all[p]
                        Ta = fit_norm(Fa[tr], mode) if mode in PER_SENSOR else (lambda A: A)
                        ok = []
                        for itr, ite in inner:
                            Ti = fit_norm(Fa[itr], mode) if mode in PER_SENSOR else (lambda A: A)
                            mm = make_model(m)
                            if m == 'SVM':
                                sc = StandardScaler().fit(Ti(Fa[itr])); mm.fit(sc.transform(Ti(Fa[itr])), cls_id[itr]); pp = mm.predict(sc.transform(Ti(Fa[ite])))
                            else:
                                mm.fit(Ti(Fa[itr]), cls_id[itr]); pp = mm.predict(Ti(Fa[ite]))
                            ok.append((pp == cls_id[ite]).mean())
                        if best is None or np.mean(ok) > best[0]: best = (np.mean(ok), p)
                    p = best[1]; chosen.append(p + 1)
                    Fa, Fb = Fa_all[p], Fb_all[p]
                    Ta = fit_norm(Fa[tr], mode) if mode in PER_SENSOR else (lambda A: A)
                    Tb = fit_norm(Fb[tr], mode) if mode in PER_SENSOR else (lambda A: A)
                    mm = make_model(m)
                    if m == 'SVM':
                        sc = StandardScaler().fit(Ta(Fa[tr])); mm.fit(sc.transform(Ta(Fa[tr])), cls_id[tr]); pp = mm.predict(sc.transform(Tb(Fb[te])))
                    else:
                        mm.fit(Ta(Fa[tr]), cls_id[tr]); pp = mm.predict(Tb(Fb[te]))
                    pred[te] = pp
                accs.append((pred == cls_id).mean())
        R[f'{k}|selected_single'] = [float(np.mean(accs)), float(np.median(chosen)), [int(c) for c in np.bincount(chosen).argsort()[::-1][:3]]]
        print(k, 'selected single', R[f'{k}|selected_single'], flush=True)
    imp = np.zeros((2, 45))
    for s in range(3):
        F = featmat(s, 'z'); F = fit_norm(F, 'z')(F)
        imp += RFC(300, random_state=0, n_jobs=-1).fit(F, cls_id).feature_importances_.reshape(2, 45) / 3
    R['imp'] = imp.tolist()
elif part == 'refdesign':
    folds0 = folds_block(0)
    def zstats(s, idx):
        f = X[idx][:, s].reshape(len(idx), -1); return f.mean(0), f.std(0) + 1e-9
    def balanced(rng, tr, k, ranks=None):
        idx = []
        for c in range(21):
            if ranks is not None and conc_rank[cls_id == c][0] not in ranks: continue
            pool = tr[cls_id[tr] == c]; idx += list(rng.choice(pool, size=min(k, len(pool)), replace=False))
        return np.array(idx)
    def ref_design(src_fn, tgt_fn, reps=5):
        res = []
        for r in range(reps):
            rng = np.random.RandomState(r); accs = []
            for a, b in PAIRS:
                for tr, te in folds0:
                    ia, ib = src_fn(rng, tr), tgt_fn(rng, tr)
                    mu_a, sd_a = zstats(a, ia); mu_b, sd_b = zstats(b, ib)
                    Ftr = (X[tr, a].reshape(len(tr), -1) - mu_a) / sd_a; Fte = (X[te, b].reshape(len(te), -1) - mu_b) / sd_b
                    accs.append((make_model('LDA').fit(Ftr, cls_id[tr]).predict(Fte) == cls_id[te]).mean())
            res.append(np.mean(accs))
        return [float(np.mean(res)), float(np.std(res))]
    TR = lambda rng, tr: tr
    R['train_all'] = ref_design(TR, TR, 1)
    for k in (1, 2, 5): R[f'bal{k}'] = ref_design(lambda r, t, k=k: balanced(r, t, k), lambda r, t, k=k: balanced(r, t, k), 10)
    for nm, rk in [('lv4', [0, 2, 4, 6]), ('lv3', [0, 3, 6]), ('lv2', [1, 5]), ('lv1', [3])]:
        R[nm] = ref_design(lambda r, t, rk=rk: balanced(r, t, 1, rk), lambda r, t, rk=rk: balanced(r, t, 1, rk), 10)
    R['mis_bal5'] = ref_design(TR, lambda r, t: balanced(r, t, 5), 10)
    for m in (3, 10, 21, 42, 84):
        R[f'mis_rand{m}'] = ref_design(TR, lambda r, t, m=m: r.choice(t, size=min(m, len(t)), replace=False), 10)
    R['mis_low'] = ref_design(TR, lambda r, t: t[low_mask[t]], 1)
    R['airz'] = {m: float(np.mean([offdiag(transfer('airz', cls_id, m, folds_block(sd))) for sd in SEEDS])) for m in ('LDA', 'SVM', 'kNN')}
    print(R, flush=True)
elif part == 'dscoral':
    folds0 = folds_block(0)
    def _sym_pow(Cm, p):
        w, V = np.linalg.eigh(Cm); w = np.clip(w, 1e-9, None); return (V * (w ** p)) @ V.T
    def coral_acc(lam):
        accs = []
        for a, b in PAIRS:
            Fa = featmat(a, 'raw'); Fb = featmat(b, 'raw')
            for tr, te in folds0:
                mu_a, sd_a = Fa[tr].mean(0), Fa[tr].std(0) + 1e-9; mu_b, sd_b = Fb[tr].mean(0), Fb[tr].std(0) + 1e-9
                Za = (Fa[tr] - mu_a) / sd_a; Zb_tr = (Fb[tr] - mu_b) / sd_b; Zb_te = (Fb[te] - mu_b) / sd_b
                Zs = Za @ _sym_pow(np.cov(Za.T) + lam * np.eye(90), -0.5) @ _sym_pow(np.cov(Zb_tr.T) + lam * np.eye(90), 0.5)
                accs.append((make_model('LDA').fit(Zs, cls_id[tr]).predict(Zb_te) == cls_id[te]).mean())
        return float(np.mean(accs))
    R['coral'] = {str(l): coral_acc(l) for l in (0.1, 1, 10)}
    print(R, flush=True)
    def ds_acc(alpha, ranks=None, pairing='classmean', reps=3):
        out = []
        for r in range(reps):
            rng = np.random.RandomState(r); accs = []
            for a, b in PAIRS:
                Fa = featmat(a, 'raw'); Fb = featmat(b, 'raw')
                for tr, te in folds0:
                    Xt, Ys = [], []
                    for c in range(21):
                        if ranks is not None and conc_rank[cls_id == c][0] not in ranks: continue
                        pool = tr[cls_id[tr] == c]
                        i = rng.choice(pool)
                        Xt.append(Fb[i])
                        if pairing == 'classmean': Ys.append(Fa[pool].mean(0))
                        elif pairing == 'othercycle': Ys.append(Fa[rng.choice([j for j in pool if j != i])])
                        else: Ys.append(Fa[i])
                    mp = Ridge(alpha=alpha).fit(np.array(Xt), np.array(Ys))
                    accs.append((make_model('LDA').fit(Fa[tr], cls_id[tr]).predict(mp.predict(Fb[te])) == cls_id[te]).mean())
            out.append(np.mean(accs))
        return [float(np.mean(out)), float(np.std(out))]
    for pairing in ('classmean', 'othercycle', 'event'):
        for nm, rk in [('21', None), ('9', [0, 3, 6]), ('6', [1, 5]), ('3', [3])]:
            R[f'ds|{pairing}|{nm}'] = {str(al): ds_acc(al, rk, pairing) for al in (1e2, 1e4, 1e5, 1e6)}
            print(pairing, nm, R[f'ds|{pairing}|{nm}'], flush=True)
elif part == 'lod':
    cm = {(s, ch): AIR[:, s, ch].mean(1) for s in range(3) for ch in range(2)}
    def sig_diff(v, exclude_jumps=False):
        d = np.diff(v)
        if exclude_jumps:
            med = np.median(d); mad = 1.4826 * np.median(np.abs(d - med)); d = d[np.abs(d - med) <= 5 * mad]
        return d.std() / np.sqrt(2)
    out = {}
    for s in range(3):
        for ch in range(2):
            d = np.diff(cm[(s, ch)]); med = np.median(d); mad = 1.4826 * np.median(np.abs(d - med))
            out[f'jumps|{s}|{ch}'] = [int(i + 1) for i in np.where(np.abs(d - med) > 5 * mad)[0]]
            out[f'lag1|{s}|{ch}'] = float(np.corrcoef(cm[(s, ch)][:-1] - cm[(s, ch)][:-1].mean(), cm[(s, ch)][1:] - cm[(s, ch)][1:].mean())[0, 1])
    prim = [('CO', 1), ('NO2', 0), ('H2S', 0), ('H2S', 1)]
    for g, ch in prim:
        gi = GASES.index(g); mk = gas == gi; cs = np.unique(conc[mk])
        for s in range(3):
            dv = X[mk][:, s, ch].mean(1) - AIR[:, s, ch].mean()
            mu = np.array([dv[conc[mk] == c].mean() for c in cs]); sdv = np.array([dv[conc[mk] == c].std() for c in cs])
            k_log = np.polyfit(np.log10(cs), mu, 1)[0]
            k_lin = np.polyfit(cs[:3], mu[:3], 1)[0]
            s_all = cm[(s, ch)].std(); s_d = sig_diff(cm[(s, ch)]); s_dj = sig_diff(cm[(s, ch)], True)
            out[f'{g}|{ch}|{s}'] = dict(k_log=float(k_log), k_lin=float(k_lin), sig_all=float(s_all), sig_diff=float(s_d), sig_diff_nojump=float(s_dj),
                                        lod_diff=float(3 * s_d / abs(k_lin)), lod_diff_nojump=float(3 * s_dj / abs(k_lin)), lod_all=float(3 * s_all / abs(k_lin)),
                                        dv_lowest=float(mu[0]), sd_lowest=float(sdv[0]), mu3=[float(x) for x in mu[:3]])
            print(g, ch, s, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[f'{g}|{ch}|{s}'].items()}, flush=True)
    R = out
json.dump(R, open(f'res_misc_{part}.json', 'w'), indent=1)
print('done', part)
