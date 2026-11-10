# Fig. 5(b) and graphical abstract: cross-sensor accuracy by source sensor for four representations
import json, numpy as np
from core import *
out={}
for m in ['raw','dv','ratio','z']:
    A=np.mean([transfer(m,cls_id,'LDA',folds_block(sd)) for sd in [0,1,2]],0)
    out[m]=[float(np.mean([A[a,b] for b in range(3) if b!=a])) for a in range(3)]
json.dump(out,open('res_fig7b.json','w')); print(out)
