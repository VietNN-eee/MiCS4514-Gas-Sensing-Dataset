# Same-sensor RMSE of RF + z-score (diagonal of Table S11)
import json, numpy as np
from core import *
from sklearn.ensemble import RandomForestRegressor as RFR
y_all=np.log10(conc); out={}
for gi,g in enumerate(GASES):
    mk=gas==gi
    for a in range(3):
        errs=[]
        for sd in [0,1,2]:
            pred=np.zeros(N)
            for tr,te in folds_block(sd):
                F=featmat(a,'z'); F=fit_norm(F[tr],'z')(F)
                trg,teg=tr[gas[tr]==gi],te[gas[te]==gi]
                pred[teg]=RFR(300,random_state=0,n_jobs=-1).fit(F[trg],y_all[trg]).predict(F[teg])
            errs.append(np.mean((pred[mk]-y_all[mk])**2))
        out[f'{g}|{a}']=float(np.sqrt(np.mean(errs))); print(g,a,out[f'{g}|{a}'],flush=True)
json.dump(out,open('res_samermse.json','w'))
