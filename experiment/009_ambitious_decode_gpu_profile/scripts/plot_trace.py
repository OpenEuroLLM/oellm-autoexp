"""Render exported aggregates, preserving per-GPU and per-step units."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
a=p.parse_args()
data=[json.loads(f.read_text()) for f in a.directory.glob('*.json') if f.stem.startswith(('ep4_b','tp4_b'))]
data.sort(key=lambda r:(r['label'].split('_')[0],int(r['label'].split('_b')[1])))
colors={'compute':'#2979b5','collective':'#e78a25','copy':'#50a178','idle':'#dedede'}
fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
labels=[]
for i,r in enumerate(data):
    labels.append(r['label'].replace('_b',' / B='))
    left=0
    for kind,color in colors.items():
        ms=sum(d['exclusive_ms_per_step'][kind] for d in r['devices'])/len(r['devices'])
        ax.barh(i,ms,left=left,color=color,label=kind if i==0 else None)
        left+=ms
    ax.text(left+1,i,f"{r['step_ms']:.1f} ms",va='center',fontsize=9)
ax.set_yticks(range(len(labels)),labels)
ax.invert_yaxis()
ax.set_xlabel('ms per decode step (mean across four GPUs; exclusive interval unions)')
ax.set_title('Ambitious 2ph: GPU activity and gaps during decode')
ax.legend(loc='lower right')
ax.set_xlim(0,max(r['step_ms'] for r in data)*1.15)
fig.savefig(a.directory/'gpu_activity.png',dpi=160)
fig.savefig(a.directory/'gpu_activity.svg')
plt.close(fig)
fig,axes=plt.subplots(len(data),1,figsize=(13,2.1*len(data)),layout='constrained',squeeze=False)
for ax,r in zip(axes[:,0],data):
    for device,rows in r['timeline'].items():
        for kind,color in colors.items():
            if kind=='idle':
                continue
            spans=[(e['start'],e['end']-e['start']) for e in rows if e['kind']==kind]
            ax.broken_barh(spans,(int(device)-.3,.6),facecolors=color,linewidths=0)
    ax.set_yticks(range(4),[f'GPU {i}' for i in range(4)])
    ax.set_xlim(0,r['individual_step_ms'][0])
    ax.set_title(r['label'].replace('_b',' / batch '),loc='left',fontsize=11)
    ax.set_facecolor('#f1f1f1')
axes[-1,0].set_xlabel('ms from start of middle decode step; each panel has its own time scale')
fig.legend(handles=[Patch(color=v,label=k) for k,v in colors.items() if k!='idle'],loc='outside upper center',ncol=3)
fig.savefig(a.directory/'decode_timeline.png',dpi=160)
fig.savefig(a.directory/'decode_timeline.svg')
