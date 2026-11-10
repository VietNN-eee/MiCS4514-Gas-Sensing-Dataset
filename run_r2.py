# Table S7 (training-only vs transductive z-score statistics on identical folds)
import json, numpy as np
from core import *
out={}
for fn,name in [(folds_random,'random'),(folds_block,'block')]:
    for strict in (True,False):
        v=[offdiag(transfer('z',cls_id,'LDA',fn(seed),strict=strict)) for seed in range(8)]
        out[f'{name}|{"train" if strict else "trans"}']={'mean3':float(np.mean(v[:3])),'mean8':float(np.mean(v)),'sd8':float(np.std(v,ddof=1)),'seed0':v[0],'all':v}
        print(name,strict,np.round(v,4),np.mean(v[:3]))
json.dump(out,open('res_r2_matrix.json','w'),indent=1)
