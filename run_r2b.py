# Within-segment clean-air noise: first-difference SD, detrended residual SD, lag-one autocorrelation (Table S5)
import json, numpy as np
from core import *
seg_b=[0,130,217,255,334]
out={}
for s in range(3):
    for ch in range(2):
        v=AIR[:,s,ch].mean(1); r1=[];dsd=[];res=[];lens=[]
        diffs=[]
        for a,b in zip(seg_b[:-1],seg_b[1:]):
            x=v[a:b]; diffs.append(np.diff(x))
            xc=x-x.mean(); r1.append(np.corrcoef(xc[:-1],xc[1:])[0,1])
            t=np.arange(len(x)); res.append(x-np.polyval(np.polyfit(t,x,1),t))
        d=np.concatenate(diffs); R=np.concatenate(res)
        rr=np.concatenate([np.corrcoef(r[:-1],r[1:])[0,1]*np.ones(1) for r in res])
        out[f'{s}|{ch}']=dict(sig_diff=float(d.std()/np.sqrt(2)),sig_detrend=float(R.std()),lag1_raw=[float(x) for x in r1],lag1_detr=[float(x) for x in rr])
        print(s,ch,{k:np.round(v_,3) for k,v_ in out[f'{s}|{ch}'].items()})
json.dump(out,open('res_r2_noise.json','w'),indent=1)
