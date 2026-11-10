# -*- coding: utf-8 -*-
"""Generate all figures of the MiCS-4514 cross-sensor manuscript (main text Fig. 1-7, Supplementary Fig. S1-S6, graphical abstract),
numbered as in manuscript v8.

Figures 1-4 and S1-S5 are computed directly from the Excel data; Figures 5-7 and S6 use the result files written by
the analysis scripts (run_all.py runs everything). Output: <DATA_DIR>/figures/*.png
"""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from core import *

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300, 'font.family': 'DejaVu Sans'})
SC = ['#1f77b4', '#d62728', '#2ca02c']; CHN = ['OX', 'RED']; SN = ['S1', 'S2', 'S3']; t = np.arange(1, 46)
GL = {'CO': 'CO', 'NO2': r'NO$_2$', 'H2S': r'H$_2$S'}; gases = GASES
D = {g: (X[gas == i], conc[gas == i]) for i, g in enumerate(gases)}
def lab(ax, s): ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='right')
DPI = int(os.environ.get('ENOSE_DPI', '300'))   # >= 300 dpi for submission
def save(fig, name, tight=True):
    if tight: fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name + '.png'), dpi=DPI, facecolor='white'); plt.close(fig); print('saved', name)
def feats(s, mode):
    """Features for visualisation (PCA): raw V_out or per-sensor z-score over all gas cycles."""
    f = X[:, s].reshape(len(X), -1)
    return (f - f.mean(0)) / f.std(0) if mode == 'z' else f

# ---------------- Fig. 1: system, read-out/heater circuit, cross-sensor split
def box(ax,x,y,w,h,txt,fc='#ffffff',fs=8,lw=1.2,bold=False,ec='#333'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=0.8',fc=fc,ec=ec,lw=lw))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal',linespacing=1.15)
def arr(ax,x1,y1,x2,y2,txt=None,tx=0,ty=0,fs=7,ha='center',style='->'):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle=style,lw=1.2,color='#222'))
    if txt: ax.text((x1+x2)/2+tx,(y1+y2)/2+ty,txt,fontsize=fs,ha=ha,va='center')
fig=plt.figure(figsize=(8.8,7.0)); gs=fig.add_gridspec(2,2,height_ratios=[1,1.12],width_ratios=[1.05,1])
# ---------------- (a) system block diagram
ax=fig.add_subplot(gs[0,:]); ax.set_xlim(0,100); ax.set_ylim(0,44); ax.axis('off')
ax.text(0,43.5,'(a)',fontsize=11,fontweight='bold',va='top'); ax.text(4,43.5,'Experimental system',fontsize=9.5,va='top')
box(ax,1,26,17,10,'Gas cylinders\nCO, NO$_2$, H$_2$S\n(diluted in air)','#fff4e0')
box(ax,1,9,17,10,'Air compressor,\nfilter, dry-air\ngenerator','#fff4e0')
box(ax,24,12,13,24,'Mass flow\ncontrollers\n(MFC)','#eef3fb')
arr(ax,18,31,24,28); arr(ax,18,14,24,20)
box(ax,43,12,30,26,'','#f5f5f5')
ax.text(58,35.3,'Gas chamber',ha='center',fontsize=8.5,fontweight='bold')
for i,(c,nm) in enumerate(zip(['#1f77b4','#d62728','#2ca02c'],['S1','S2','S3'])):
    x0=45+i*9.3; ax.add_patch(FancyBboxPatch((x0,15),8,16,boxstyle='round,pad=0,rounding_size=0.5',fc='white',ec='#333',lw=1))
    ax.text(x0+4,25.5,f'MiCS-\n4514\n{nm}',ha='center',va='center',fontsize=7.5); ax.add_patch(Rectangle((x0+0.4,15.6),7.2,1.6,fc=c,ec='none'))
arr(ax,37,24,43,24); arr(ax,73,24,80,24); box(ax,80,19,12,10,'Exhaust\n(pump)','#fff4e0')
# electronics row
box(ax,43,0.5,30,7,'ESP32: PWM generation (5 kHz)\n+ 2 × ADS1115 (16-bit) read-out','#e8f5e9',fs=7.5)
arr(ax,53,7.5,53,12,'PWM heater drive\n(250→30, 45 s)',tx=-0.8,ha='right',fs=7)
arr(ax,63,12,63,7.5,r'$V_{out}$ (6 channels)',tx=0.8,ha='left',fs=7)
box(ax,18,0.5,19,7,'PC (LabVIEW)\nlogging, MFC control','#fde9e7',fs=7.5)
arr(ax,43,4,37,4,'serial',ty=2.0,fs=7,style='<->')
arr(ax,27.5,7.5,27.5,12,'flow\nset-points',tx=0.8,ha='left',fs=7)
# ---------------- (b) circuit
ax=fig.add_subplot(gs[1,0]); ax.set_xlim(0,17.2); ax.set_ylim(-1.4,12.2); ax.axis('off')
ax.text(0,12.1,'(b)',fontsize=11,fontweight='bold',va='top'); ax.text(1.4,12.1,'Read-out and heater circuit (per package)',fontsize=9.5,va='top')
def vz(x,y1,y2,n=6,a=.35):
    ys=np.linspace(y1,y2,2*n+1); xs=np.full_like(ys,x); xs[1:-1:2]+=a; xs[2:-1:2]-=a
    ax.plot(xs,ys,color='#222',lw=1.2)
def wire(xs,ys): ax.plot(xs,ys,color='#222',lw=1.2)
top,bot=10,0
wire([.6,14.4],[top,top]); wire([.6,14.4],[bot,bot]); ax.text(.5,top+.45,'$V_c$ = +5 V',fontsize=8); ax.text(.5,bot-.75,'GND',fontsize=8)
chan=[('OX',1.6,4.6,'#1f77b4'),('RED',9.2,12.2,'#d62728')]
for nm,xh,xs,col in chan:
    # heater branch: +5V -> R_H (inside package) -> R_P -> Q -> GND
    wire([xh,xh],[top,9.3]); vz(xh,9.3,7.0); wire([xh,xh],[7.0,5.8]); vz(xh,5.8,3.8); wire([xh,xh],[3.8,2.9])
    ax.add_patch(Rectangle((xh-.55,1.3),1.1,1.6,fc='white',ec='#222',lw=1.2)); ax.text(xh,2.1,'Q',ha='center',va='center',fontsize=8)
    wire([xh,xh],[1.3,bot]); ax.plot([xh],[bot],'o',ms=2.5,color='#222'); ax.plot([xh],[top],'o',ms=2.5,color='#222')
    ax.annotate('',xy=(xh-.55,2.1),xytext=(xh-1.2,2.1),arrowprops=dict(arrowstyle='->',lw=1.1))
    ax.text(xh-1.25,2.1,'PWM',ha='right',va='center',fontsize=7)
    ax.text(xh+.55,8.15,'$R_H$',fontsize=8,va='center'); ax.text(xh+.55,4.8,'$R_P$ = 100 Ω',fontsize=7.5,va='center')
    # sensing branch: +5V -> R_S (inside package) -> node V_out -> R_L -> GND  (V_out taken across R_L)
    wire([xs,xs],[top,9.3]); vz(xs,9.3,7.0); wire([xs,xs],[7.0,5.9]); ax.plot([xs],[5.9],'o',ms=3,color='#222')
    wire([xs,xs],[5.9,4.9]); vz(xs,4.9,2.7); wire([xs,xs],[2.7,bot]); ax.plot([xs],[bot],'o',ms=2.5,color='#222'); ax.plot([xs],[top],'o',ms=2.5,color='#222')
    ax.text(xs+.55,8.15,'$R_S$',fontsize=8,va='center'); ax.text(xs+.55,3.8,'$R_L$ = '+('10 kΩ' if nm=='OX' else '100 kΩ'),fontsize=7.5,va='center')
    wire([xs,xs+1.5],[5.9,5.9]); ax.annotate('',xy=(xs+2.1,5.9),xytext=(xs+1.5,5.9),arrowprops=dict(arrowstyle='->',lw=1.1))
    ax.text(xs+1.7,6.45,f'$V_{{{nm}}}$→ADS1115',fontsize=7,ha='left')
    ax.text((xh+xs)/2,11.0,f'{nm} element',ha='center',fontsize=8.5,fontweight='bold',color=col)
ax.add_patch(FancyBboxPatch((.6,6.5),12.9,3.4,boxstyle='round,pad=0,rounding_size=0.3',fc='none',ec='#777',lw=1,ls='--'))
ax.text(13.6,8.2,'MiCS-4514',fontsize=7.5,ha='left',va='center',color='#555')
# ---------------- (c) data split
ax=fig.add_subplot(gs[1,1]); ax.set_xlim(0,60); ax.set_ylim(0,44); ax.axis('off')
ax.text(0,43.5,'(c)',fontsize=11,fontweight='bold',va='top'); ax.text(4,43.5,'Cross-sensor data split (3 rotations)',fontsize=9.5,va='top')
ax.text(4,37.5,'Train (2 channels)',fontsize=8.5,fontweight='bold'); ax.text(30,37.5,'Test (other 2 sensors)',fontsize=8.5,fontweight='bold')
rows=[('S1',['S2','S3']),('S2',['S1','S3']),('S3',['S1','S2'])]
for i,(tr,te) in enumerate(rows):
    y=26-i*12.5
    box(ax,2,y,16,8.5,f'{tr}\n(OX+RED)','#cfe0f2',fs=8)
    arr(ax,19,y+4.25,29,y+4.25,f'Rotation {i+1}',ty=2.2,fs=7.5)
    for j,tn in enumerate(te): box(ax,30+j*15,y,13.5,8.5,f'{tn}\n(OX+RED)','#fbe7d8',fs=8)
fig.tight_layout(); save(fig,'Fig1')

# ---------------- Fig. S1: clean-air response along the PWM ramp
# ---------- Fig 2
fig,axs=plt.subplots(2,2,figsize=(7.4,5.6)); axs=axs.ravel()
ax=axs[0]; pwm=250-5*(t-1)
ax.plot(t,pwm,color='k',lw=1.5); ax.plot(t,pwm,'o',color='k',ms=3)
ax.set_xlabel('Time in cycle (s)'); ax.set_ylabel('Heater PWM (duty level)'); ax.set_title('Linearly decreasing PWM ramp\n(250 to 30, 45 points, 1 Hz)',fontsize=9)
ax.set_ylim(15,265); ax.set_xlim(0,46); ax.set_yticks([30,90,150,210,250])
ax.annotate('1 sample/s, step of 5',xy=(22,140),xytext=(15,215),arrowprops=dict(arrowstyle='->'),fontsize=8)
lab(axs[0],'(a)')
m=AIR.mean(0).mean(0)  # ch,pt (mean over sensors)
sd=AIR.reshape(len(AIR),-1,2,45).std(0) if False else AIR.transpose(0,1,2,3).reshape(-1,2,45).std(0)
ax=axs[1]
for c,col in zip(range(2),['#1f77b4','#d62728']):
    ax.plot(t,m[c],color=col,label=CHN[c]); ax.fill_between(t,m[c]-sd[c],m[c]+sd[c],color=col,alpha=.2)
ax.set_xlabel('Time in cycle (s)'); ax.set_ylabel(r'$V_{out}$ (mV)'); ax.set_title('Clean air: mean ± SD\n(334 cycles, 3 sensors)',fontsize=9); ax.legend(frameon=False); lab(ax,'(b)')
ax=axs[2]
for c,col in zip(range(2),['#1f77b4','#d62728']):
    y=(m[c]-m[c].min())/(m[c].max()-m[c].min()); ax.plot(t,y,color=col,label=CHN[c])
ax.set_xlabel('Time in cycle (s)'); ax.set_ylabel(r'Min–max normalised $V_{out}$'); ax.set_title('Trajectory shape',fontsize=9); ax.legend(frameon=False); lab(ax,'(c)')
ax=axs[3]
for c,col in zip(range(2),['#1f77b4','#d62728']):
    ax.plot(t[1:],np.diff(m[c]),color=col,label=CHN[c])
ax.axhline(0,color='k',lw=.5); ax.set_xlabel('Time in cycle (s)'); ax.set_ylabel(r'$\Delta V$ between consecutive points (mV)'); ax.set_title('Point-to-point change',fontsize=9); ax.legend(frameon=False); lab(ax,'(d)')
save(fig,'FigS1')

# ---------------- Fig. 2: response trajectories of S1
# ---------- Fig 3 (sensor S1 = first sensor)
s=0
fig,axs=plt.subplots(3,2,figsize=(6.6,7.2),sharex=True)
for i,g in enumerate(gases):
    Xg,cg=D[g]; cs=sorted(set(cg)); cmap=plt.cm.viridis(np.linspace(0,.95,len(cs)))
    for j in range(2):
        ax=axs[i,j]
        for k,col in zip(cs,cmap):
            v=Xg[cg==k][:,s,j]; mu=v.mean(0); sg=v.std(0)
            ax.plot(t,mu,color=col,lw=1.2,label=f'{k:g}'); ax.fill_between(t,mu-sg,mu+sg,color=col,alpha=.2,lw=0)
        ax.plot(t,airmean[s,j],'k--',lw=1,label='Air')
        if j==0: ax.set_ylabel(f'{GL[g]}\n'+r'$V_{out}$ (mV)')
        if i==0: ax.set_title(f'{CHN[j]} channel')
        if i==2: ax.set_xlabel('Time in cycle (s)')
        if j==1: ax.legend(title='ppm',frameon=False,fontsize=7,title_fontsize=7,loc='upper right',ncol=2)
for ax,l in zip(axs.ravel(),'abcdef'): lab(ax,f'({l})')
save(fig,'Fig2')

# ---------------- Fig. 3: calibration curves
# ---------- Fig 4 calibration
fig,axs=plt.subplots(3,2,figsize=(6.6,7.2))
for i,g in enumerate(gases):
    Xg,cg=D[g]; cs=np.array(sorted(set(cg)))
    for j in range(2):
        ax=axs[i,j]
        for sx in range(3):
            dv=Xg[:,sx,j].mean(1)-AIR[:,sx,j].mean()
            mu=np.array([dv[cg==k].mean() for k in cs]); sg=np.array([dv[cg==k].std() for k in cs])
            ax.errorbar(cs*(1+0.02*(sx-1)),mu,yerr=sg,fmt='o-',ms=3,lw=1,capsize=2,color=SC[sx],label=SN[sx])
        ax.set_xscale('log'); ax.axhline(0,color='k',lw=.4)
        if j==0: ax.set_ylabel(f'{GL[g]}\n'+r'$\Delta V_{out}$ (mV)')
        if i==0: ax.set_title(f'{CHN[j]} channel')
        if i==2: ax.set_xlabel('Concentration (ppm)')
        if i==0 and j==0: ax.legend(frameon=False,fontsize=7)
for ax,l in zip(axs.ravel(),'abcdef'): lab(ax,f'({l})')
save(fig,'Fig3')

# ---------------- Fig. 4: sensor-to-sensor variability
# ---------- Fig 5 inter-sensor variability
ref=AIR[:20].mean(0); norm=AIR/ref
fig=plt.figure(figsize=(7.8,4.4)); gs=fig.add_gridspec(2,3,width_ratios=[1,1,1])
def traj(ax,A,ch,ylabel,title):
    for sx in range(3):
        mu=A[:,sx,ch].mean(0); sg=A[:,sx,ch].std(0)
        ax.plot(t,mu,color=SC[sx],label=SN[sx],lw=1.3); ax.fill_between(t,mu-sg,mu+sg,color=SC[sx],alpha=.2,lw=0)
    ax.set_ylabel(ylabel); ax.set_title(title,fontsize=9)
a=fig.add_subplot(gs[0,0]); traj(a,AIR,0,r'$V_{out}$ (mV)','OX, raw'); a.legend(frameon=False,fontsize=7); lab(a,'(a)')
b=fig.add_subplot(gs[0,1]); traj(b,AIR,1,r'$V_{out}$ (mV)','RED, raw'); lab(b,'(b)')
c=fig.add_subplot(gs[1,0]); traj(c,norm,0,r'$V_{out}/V_{ref}$','OX, normalised'); c.set_xlabel('Time in cycle (s)'); lab(c,'(c)')
d=fig.add_subplot(gs[1,1]); traj(d,norm,1,r'$V_{out}/V_{ref}$','RED, normalised'); d.set_xlabel('Time in cycle (s)'); lab(d,'(d)')
e=fig.add_subplot(gs[:,2])
cvf=lambda A,ch:(A.mean(0)[:,ch].std(0)/A.mean(0)[:,ch].mean(0))*100
cvs={}
for ch,col in zip(range(2),['#1f77b4','#d62728']):
    r=cvf(AIR,ch); n=cvf(norm,ch); cvs[ch]=(r.mean(),n.mean())
    e.plot(t,r,color=col,ls='-',label=f'{CHN[ch]} raw'); e.plot(t,n,color=col,ls='--',label=f'{CHN[ch]} normalised')
e.set_xlabel('Time in cycle (s)'); e.set_ylabel('CV across 3 sensors (%)'); e.legend(frameon=False,fontsize=7); e.set_title('Inter-sensor CV per point',fontsize=9); lab(e,'(e)')
save(fig,'Fig4')

# ---------------- Fig. S5: PCA fitted on S1
mk_=['o','s','^']
# ---------- Fig 6 PCA
fig,axs=plt.subplots(1,2,figsize=(8,3.8))
for ax,mode,tt,l in zip(axs,['raw','z'],['Raw $V_{out}$','Per-sensor z-score'],'ab'):
    F={sx:feats(sx,mode) for sx in range(3)}
    p=PCA(2).fit(F[0])
    for sx in range(3):
        Z=p.transform(F[sx])
        for gi in range(3):
            mk=gas==gi
            if sx==0: ax.scatter(Z[mk,0],Z[mk,1],marker=mk_[gi],s=22,color=SC[sx],alpha=.8,lw=.5)
            else: ax.scatter(Z[mk,0],Z[mk,1],marker=mk_[gi],s=26,facecolors='none',edgecolors=SC[sx],lw=.9)
    ax.set_xlabel(f'PC1 ({p.explained_variance_ratio_[0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({p.explained_variance_ratio_[1]*100:.1f}%)'); ax.set_title(tt,fontsize=9); lab(ax,f'({l})')
h=[Line2D([],[],marker='o',ls='',color=SC[0],label='S1 (train, filled)'),Line2D([],[],marker='o',ls='',mfc='none',color=SC[1],label='S2 (test, hollow)'),Line2D([],[],marker='o',ls='',mfc='none',color=SC[2],label='S3 (test, hollow)')]
h+=[Line2D([],[],marker=mk_[i],ls='',color='gray',label=GL[g]) for i,g in enumerate(gases)]
fig.legend(handles=h,loc='lower center',ncol=6,frameon=False,fontsize=7.5)
fig.tight_layout(rect=(0,0.07,1,1)); save(fig,'FigS5',tight=False)

# ---------------- Fig. S2: clean-air baseline with segment boundaries
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.2))
for j in range(2):
    for s in range(3): axs[j].plot(np.arange(1, len(AIR) + 1), AIR[:, s, j].mean(1), color=SC[s], lw=1, label=SN[s])
    for b in (130.5, 217.5, 255.5): axs[j].axvline(b, color='k', ls='--', lw=.7)
    axs[j].set_xlabel('Clean-air cycle index'); axs[j].set_ylabel(r'Cycle-mean $V_{out}$ (mV)'); axs[j].set_title(f'{CHN[j]} channel', fontsize=9); lab(axs[j], f'({"ab"[j]})')
axs[0].legend(frameon=False, fontsize=7); save(fig, 'FigS2')

# ---------------- Fig. S3, S4: response trajectories of S2 and S3
for s, fn in [(1,'FigS3'),(2,'FigS4')]:
    fig,axs = plt.subplots(3,2,figsize=(6.6,7.2),sharex=True)
    for i,g in enumerate(gases):
        Xg,cg = D[g]; cs = sorted(set(cg)); cmap = plt.cm.viridis(np.linspace(0,.95,len(cs)))
        for j in range(2):
            ax = axs[i,j]
            for k,col in zip(cs,cmap):
                v = Xg[cg==k][:,s,j]; mu = v.mean(0); sg = v.std(0)
                ax.plot(t,mu,color=col,lw=1.2,label=f'{k:g}'); ax.fill_between(t,mu-sg,mu+sg,color=col,alpha=.2,lw=0)
            ax.plot(t,airmean[s,j],'k--',lw=1,label='Air')
            if j==0: ax.set_ylabel(f'{GL[g]}\n'+r'$V_{out}$ (mV)')
            if i==0: ax.set_title(f'{CHN[j]} channel')
            if i==2: ax.set_xlabel('Time in cycle (s)')
            if j==1: ax.legend(title='ppm',frameon=False,fontsize=7,title_fontsize=7,loc='upper right',ncol=2)
    for ax,l in zip(axs.ravel(),'abcdef'): lab(ax,f'({l})')
    save(fig,fn)

# ---------------- Fig. 5 and Fig. S6 (needs results of run_cls.py and run_fig7b.py)

labels = [f'{GL[g]} {c:g}' for gi, g in enumerate(GASES) for c in np.unique(conc[gas == gi])]
P = np.load('pred_conf.npz'); y = P['y']

def conf(ax, pred, title):
    cm = confusion_matrix(y, pred, labels=range(21))
    im = ax.imshow(cm / cm.sum(1, keepdims=True), cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(21)); ax.set_yticks(range(21)); ax.set_xticklabels(labels, rotation=90, fontsize=5.5); ax.set_yticklabels(labels, fontsize=5.5)
    for i in range(21):
        for j in range(21):
            if cm[i, j] > 0: ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=5, color='w' if cm[i, j] / cm[i].sum() > .5 else 'k')
    ax.set_xlabel('Predicted class (gas, ppm)'); ax.set_ylabel('True class (gas, ppm)'); ax.set_title(title, fontsize=9)
    return im

# ---------------- Fig 7
fig, axs = plt.subplots(1, 2, figsize=(8.6, 4.0), gridspec_kw={'width_ratios': [1.1, 1]})
acc01 = (P['z01'] == y).mean()
im = conf(axs[0], P['z01'], f'S1→S2, z-score + LDA, 21 classes\n(accuracy {acc01:.2f})'); lab(axs[0], '(a)')
cb = fig.colorbar(im, ax=axs[0], fraction=.04, pad=.02); cb.set_label('Row-normalised fraction', fontsize=8)
src = json.load(open('res_fig7b.json'))
ax = axs[1]; w = .2; cols = ['#9ecae1', '#6baed6', '#3182bd', '#08306b']
names = [('raw', 'Raw $V_{out}$'), ('dv', r'$\Delta V$'), ('ratio', r'$V/V_{air}$'), ('z', 'Per-sensor z-score')]
for k, (m, nm) in enumerate(names):
    v = src[m]
    ax.bar(np.arange(3) + (k - 1.5) * w, v, w, color=cols[k], label=nm, edgecolor='white', linewidth=.5)
    for x_, y_ in zip(np.arange(3) + (k - 1.5) * w, v): ax.text(x_, y_ + .01, f'{y_:.2f}', ha='center', fontsize=6, rotation=90)
ax.set_xticks(range(3)); ax.set_xticklabels(['S1→S2,S3', 'S2→S1,S3', 'S3→S1,S2']); ax.set_ylabel('Cross-sensor accuracy (21 classes, LDA)')
ax.set_ylim(0, 1.12); ax.set_yticks([0, .2, .4, .6, .8, 1.0])
ax.legend(frameon=False, fontsize=7, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0), columnspacing=.8, handlelength=1); ax.set_xlabel('Train sensor → test sensors'); lab(ax, '(b)')
save(fig, 'Fig5')

# ---------------- Fig S4
fig, axs = plt.subplots(1, 2, figsize=(8.6, 4.2))
conf(axs[0], P['r01'], f'S1→S2, raw + SVM (accuracy {(P["r01"] == y).mean():.2f})'); lab(axs[0], '(a)')
conf(axs[1], P['r10'], f'S2→S1, raw + SVM (accuracy {(P["r10"] == y).mean():.2f})'); lab(axs[1], '(b)')
save(fig, 'FigS6')

# ---------------- Fig. 6: ablation (needs run_misc.py ablation)
A = json.load(open('res_misc_ablation.json'))
cfg = {'z': ('Per-sensor z-score + LDA', '#08306b'), 'r': ('Raw $V_{out}$ + SVM', '#d95f02')}
fig = plt.figure(figsize=(8.8, 3.2)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15])
a = fig.add_subplot(gs[0]); ns = [3, 5, 15, 45]
for k, (nm, col) in cfg.items():
    single = A[f'{k}|single']
    a.plot([1] + ns, [np.mean(single)] + [A[f'{k}|n'][str(n)] for n in ns], 'o-', color=col, label=nm, ms=4)
    a.vlines(1, min(single), max(single), color=col, lw=3, alpha=.35)
    a.plot(1, A[f'{k}|selected_single'][0], marker='*', ms=9, color=col, ls='')
a.set_xscale('log'); a.set_xticks([1, 3, 5, 15, 45]); a.set_xticklabels([1, 3, 5, 15, 45]); a.set_xlabel('Points per channel')
a.set_ylabel('Cross-sensor accuracy (21 classes)'); a.set_ylim(.4, 1.02); a.legend(frameon=False, fontsize=7, loc='center right'); lab(a, '(a)')
a.text(1.08, .43, 'bar: min–max of 45 single points\n★: point selected in training', fontsize=6.3)
b = fig.add_subplot(gs[1]); w = .38
regs = ['first', 'middle', 'last']
for i, (k, (nm, col)) in enumerate(cfg.items()):
    vals = [A[f'{k}|regions'][r] for r in regs] + [A[f'{k}|n']['45']]
    b.bar(np.arange(4) + (i - .5) * w, vals, w, color=col, label=nm)
b.set_xticks(range(4)); b.set_xticklabels(['First\n(1–15)', 'Middle\n(16–30)', 'Last\n(31–45)', 'Full\n(1–45)'], fontsize=7)
b.set_ylabel('Cross-sensor accuracy (21 classes)'); b.set_ylim(0, 1.05); b.set_xlabel('Cycle region'); lab(b, '(b)')
c = fig.add_subplot(gs[2]); imp = np.array(A['imp'])
im = c.imshow(imp * 100, aspect='auto', cmap='magma', extent=[.5, 45.5, 1.5, -.5]); c.set_yticks([0, 1]); c.set_yticklabels(['OX', 'RED'])
c.set_xlabel('Point index in cycle (= time, s)'); c.set_title('RF feature importance', fontsize=9); lab(c, '(c)')
cb = fig.colorbar(im, ax=c, fraction=.05, pad=.02); cb.set_label('Importance (%)', fontsize=8)
save(fig, 'Fig6')

# ---------------- Fig. 7: cross-sensor quantification (needs run_reg.py)
T5 = json.load(open('res_reg_table5.json')); SM = json.load(open('res_reg_samemre.json'))
fig, axs = plt.subplots(2, 3, figsize=(8.4, 4.7))
for gi, g in enumerate(GASES):
    d = np.load(f'reg_{g}.npz'); yt, yp = d['yt'], d['yp']; r = T5[f'z|RF|{g}']
    ax = axs[0, gi]; ax.scatter(10 ** yt, 10 ** yp, s=8, alpha=.2, color='#1f77b4', lw=0)
    lim = [10 ** yt.min() * .6, 10 ** yt.max() * 1.6]; ax.plot(lim, lim, 'k--', lw=.8); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('True (ppm)'); ax.set_ylabel('Predicted (ppm)')
    ax.text(0.04, 0.96, f'{GL[g]}\n$R^2$ = {r["r2"]:.2f}\nRMSE = {r["rmse"]:.2f}', transform=ax.transAxes, va='top', fontsize=7.5); lab(ax, f'({"abc"[gi]})')
    cs = [float(k) for k in r['mre']]; ax = axs[1, gi]
    ax.plot(cs, list(r['mre'].values()), 'o-', color='#d62728', label='Cross-sensor')
    ax.plot(cs, [SM[g][k] for k in r['mre']], 's--', color='#2ca02c', label='Same sensor')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_: f'{v:g}')); ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1,2,5))); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xlabel('Concentration (ppm)'); ax.set_ylabel('Median rel. error (%)'); lab(ax, f'({"def"[gi]})')
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi * 3.5)
    if gi == 0: ax.legend(frameon=False, fontsize=7, ncol=2, loc='upper left', columnspacing=1)
save(fig, 'Fig7')
# ---------------- Graphical abstract
src = json.load(open('res_fig7b.json'))
fig=plt.figure(figsize=(9.6,4.4),dpi=300,facecolor='white')
# ---- left: rotations
ax=fig.add_axes([0.01,0.03,0.47,0.86]); ax.set_xlim(-0.4,10.2); ax.set_ylim(0,10); ax.axis('off')
fig.text(0.02,0.94,'Cross-sensor data split (3 rotations)',fontsize=12.5,weight='bold',color='#1f2933')
ax.text(1.25,9.35,'Train (1 sensor)',ha='center',fontsize=10.5,weight='bold',color='#1f2933')
ax.text(7.25,9.35,'Test (other 2 sensors)',ha='center',fontsize=10.5,weight='bold',color='#1f2933')
def box(x,y,t,fc,ec):
    ax.add_patch(FancyBboxPatch((x,y),2.3,1.9,boxstyle='round,pad=0.02,rounding_size=0.25',fc=fc,ec=ec,lw=1.6))
    ax.text(x+1.15,y+0.95,t,ha='center',va='center',fontsize=10.5,color='#1f2933')
rows=[('S1',('S2','S3')),('S2',('S1','S3')),('S3',('S1','S2'))]
for i,(tr,te) in enumerate(rows):
    y=6.7-i*3.0
    box(0.1,y,f'{tr}\n(OX + RED)','#d6e6f5','#2b6ca3')
    ax.add_patch(FancyArrowPatch((2.65,y+0.95),(4.6,y+0.95),arrowstyle='-|>',mutation_scale=14,lw=1.6,color='#333'))
    ax.text(3.62,y+1.25,f'Rotation {i+1}',ha='center',fontsize=9,color='#555')
    box(4.85,y,f'{te[0]}\n(OX + RED)','#fbe5d6','#c0703a'); box(7.45,y,f'{te[1]}\n(OX + RED)','#fbe5d6','#c0703a')
# ---- right: bars
bx=fig.add_axes([0.585,0.15,0.40,0.68])
names=[('raw','Raw $V_{out}$'),('dv',r'$\Delta V$'),('ratio',r'$V/V_{air}$'),('z','Per-sensor z-score')]
cols=['#c6dbef','#9ecae1','#4292c6','#08306b']; w=0.2; x=np.arange(3)
for k,(m,nm) in enumerate(names):
    v=src[m]; xs=x+(k-1.5)*w
    bx.bar(xs,v,w,color=cols[k],label=nm,edgecolor='white',lw=0.6)
    for xx,vv in zip(xs,v): bx.text(xx,vv+0.015,f'{vv:.2f}',ha='center',va='bottom',fontsize=7,rotation=90,color='#1f2933',weight='bold' if m=='z' else 'normal')
bx.set_xticks(x); bx.set_xticklabels(['S1→S2,S3','S2→S1,S3','S3→S1,S2'],fontsize=10)
bx.set_ylim(0,1.15); bx.set_yticks([0,0.2,0.4,0.6,0.8,1.0]); bx.set_ylabel('Cross-sensor accuracy\n(21 classes, LDA)',fontsize=10)
bx.set_xlabel('Train sensor → test sensors',fontsize=10)
for s in ('top','right'): bx.spines[s].set_visible(False)
bx.legend(frameon=False,fontsize=8.5,ncol=4,loc='lower center',bbox_to_anchor=(0.5,1.0),columnspacing=0.9,handlelength=1.1)
fig.text(0.585,0.94,'Per-sensor z-score restores transfer',fontsize=12.5,weight='bold',color='#1f2933')
fig.savefig(os.path.join(FIG_DIR, 'Graphical_Abstract.png'), dpi=DPI, facecolor='white'); plt.close(fig); print('saved Graphical_Abstract')
