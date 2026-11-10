# Core data loading and cross-sensor protocol for the MiCS-4514 calibration-transfer study
# (contiguous-block CV within each condition; per-sensor statistics fitted on training cycles only)
import numpy as np, pandas as pd, os, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC, GradientBoostingClassifier as GBC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import StandardScaler

# ---------------- paths ----------------
# Default layout:  <DATA_DIR>/*.xlsx,  <DATA_DIR>/code/*.py,  <DATA_DIR>/results/,  <DATA_DIR>/figures/
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get('ENOSE_DATA_DIR', os.path.dirname(CODE_DIR))
RES_DIR = os.environ.get('ENOSE_RES_DIR', os.path.join(DATA_DIR, 'results'))
FIG_DIR = os.environ.get('ENOSE_FIG_DIR', os.path.join(DATA_DIR, 'figures'))
os.makedirs(RES_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
os.chdir(RES_DIR)                      # all analysis scripts read/write their result files here

# ---------------- data ----------------
LEVELS = {'CO': [1, 5, 10, 25, 50, 75, 100], 'NO2': [0.1, 0.5, 1, 2.5, 5, 7.5, 10], 'H2S': [0.1, 0.5, 1, 2.5, 5, 7.5, 10]}
def _channels(df):
    """Sensor columns in the order S1, S2, S3 (OX, RED). Accepts S1..S3 or the original S2..S4 naming."""
    for ids in (('S1', 'S2', 'S3'), ('S2', 'S3', 'S4')):
        cols = [f'{s}_{c}_Vout' for s in ids for c in ('OX', 'RED')]
        if all(c in df.columns for c in cols): return cols
    raise KeyError('sensor columns not found')
def to_array(df):
    n = len(df) // 45
    return df[_channels(df)].values.astype(float).reshape(n, 45, 3, 2).transpose(0, 2, 3, 1)   # cycle, sensor, channel(OX,RED), point

def load():
    air = pd.read_excel(os.path.join(DATA_DIR, 'Base_Air_45points_Vo.xlsx'))
    D = {'Air': (to_array(air), np.zeros(len(air) // 45))}
    for g in ['CO', 'NO2', 'H2S']:
        df = pd.read_excel(os.path.join(DATA_DIR, f'{g}_45points_Vo.xlsx'))
        df = df[np.isin(np.round(df.Nong_do_khi.astype(float), 3), LEVELS[g])]        # the seven levels used in the study
        n = len(df) // 45
        D[g] = (to_array(df), df.Nong_do_khi.values.reshape(n, 45)[:, 0].astype(float))
    return D

_cache = os.path.join(RES_DIR, 'data_cache.npz')
if os.path.exists(_cache):
    z = np.load(_cache); AIR, X, conc, gas = z['AIR'], z['X'], z['conc'], z['gas']
else:
    D = load(); AIR = D['Air'][0]; gases = ['CO', 'NO2', 'H2S']
    X = np.concatenate([D[g][0] for g in gases]); conc = np.concatenate([D[g][1] for g in gases])
    gas = np.concatenate([[i] * len(D[g][1]) for i, g in enumerate(gases)])
    np.savez(_cache, AIR=AIR, X=X, conc=conc, gas=gas)
GASES = ['CO', 'NO2', 'H2S']
cls = np.array([f'{g}{c}' for g, c in zip(gas, conc)]); _, cls_id = np.unique(cls, return_inverse=True)
airmean = AIR.mean(0)
N = len(X)
# position of each cycle within its condition (chronological order in the files)
pos = np.zeros(N, int)
for c in np.unique(cls_id):
    idx = np.where(cls_id == c)[0]; pos[idx] = np.arange(len(idx))

# ---------------- features ----------------
def base(s, mode):
    x = X[:, s]
    if mode in ('raw', 'z', 'mc', 'minmax', 'robust'): return x
    if mode == 'dv': return x - airmean[s]
    if mode == 'ratio': return x / airmean[s]
    if mode == 'l2':
        f = x.reshape(N, -1); return (f / np.linalg.norm(f, axis=1, keepdims=True)).reshape(x.shape)
    if mode == 'airz':
        a = AIR[:, s]; return (x - a.mean(0)) / a.std(0)
    raise ValueError(mode)

def fit_norm(F, mode):
    """Return a transform fitted on F (rows = reference cycles) for per-sensor normalisation."""
    if mode == 'z':
        mu, sd = F.mean(0), F.std(0); sd[sd == 0] = 1; return lambda A: (A - mu) / sd
    if mode == 'mc':
        mu = F.mean(0); return lambda A: A - mu
    if mode == 'minmax':
        lo, hi = F.min(0), F.max(0); return lambda A: (A - lo) / (hi - lo + 1e-12)
    if mode == 'robust':
        med = np.median(F, 0); iqr = np.subtract(*np.percentile(F, [75, 25], 0)); iqr[iqr == 0] = 1
        return lambda A: (A - med) / iqr
    return lambda A: A

PER_SENSOR = ('z', 'mc', 'minmax', 'robust')

def featmat(s, mode, pts=None):
    f = base(s, mode)
    if pts is not None: f = f[:, :, pts]
    return f.reshape(N, -1)

# ---------------- splits ----------------
def folds_random(seed=0, y=None):
    y = cls_id if y is None else y
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))

def folds_block(seed=0, idx=None):
    """Contiguous-block 5-fold: within each condition, consecutive cycles are kept together.
    10 cycles -> 5 blocks of 2; 7 cycles -> blocks of 2,2,1,1,1. Block-to-fold mapping shuffled per condition by seed."""
    idx = np.arange(N) if idx is None else idx
    rng = np.random.RandomState(seed)
    fold_of = np.zeros(N, int)
    for c in np.unique(cls_id[idx]):
        ii = idx[cls_id[idx] == c]; ii = ii[np.argsort(pos[ii])]
        blocks = np.array_split(np.arange(len(ii)), 5)
        perm = rng.permutation(5)
        for b, k in zip(blocks, perm): fold_of[ii[b]] = k
    return [(idx[fold_of[idx] != k], idx[fold_of[idx] == k]) for k in range(5)]

def block_id():
    """Group id of the contiguous block (condition, block) used for block bootstrap."""
    g = np.zeros(N, int); k = 0
    for c in np.unique(cls_id):
        ii = np.where(cls_id == c)[0]; ii = ii[np.argsort(pos[ii])]
        for b in np.array_split(np.arange(len(ii)), 5):
            g[ii[b]] = k; k += 1
    return g

# ---------------- models ----------------
def make_model(name, C=10, kernel='rbf'):
    return {'LDA': lambda: LDA(solver='lsqr', shrinkage='auto'), 'SVM': lambda: SVC(C=C, kernel=kernel),
            'RF': lambda: RFC(200, random_state=0, n_jobs=-1), 'kNN': lambda: KNN(5),
            'GBDT': lambda: GBC(random_state=0)}[name]()

def transfer(mode, y, model='LDA', folds=None, pts=None, strict=True, return_pred=False, **mk):
    """acc[a,b]: train on sensor a (train cycles), test on sensor b (held-out cycles).
    strict=True: per-sensor statistics are fitted on the TRAINING cycles of each sensor only."""
    F = {s: featmat(s, mode, pts) for s in range(3)}
    acc = np.zeros((3, 3)); P = {}
    for a in range(3):
        for b in range(3):
            pred = np.full(len(y), -1)
            for tr, te in folds:
                if mode in PER_SENSOR:
                    ref_a = tr if strict else np.arange(N)
                    ref_b = tr if strict else np.arange(N)
                    Ta, Tb = fit_norm(F[a][ref_a], mode), fit_norm(F[b][ref_b], mode)
                    Xa, Xb = Ta(F[a][tr]), Tb(F[b][te])
                else:
                    Xa, Xb = F[a][tr], F[b][te]
                m = make_model(model, **mk)
                if model in ('SVM', 'kNN'):
                    sc = StandardScaler().fit(Xa); m.fit(sc.transform(Xa), y[tr]); p = m.predict(sc.transform(Xb))
                else:
                    m.fit(Xa, y[tr]); p = m.predict(Xb)
                pred[te] = p
            done = pred >= 0
            acc[a, b] = (pred[done] == y[done]).mean(); P[(a, b)] = pred
    return (acc, P) if return_pred else acc

def offdiag(acc): return float(np.mean([acc[a, b] for a in range(3) for b in range(3) if a != b]))
PAIRS = [(a, b) for a in range(3) for b in range(3) if a != b]
