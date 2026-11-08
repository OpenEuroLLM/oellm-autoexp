"""Plot measured node output throughput with a separate scale for each batch."""
import argparse
import json
import os
from pathlib import Path
import statistics
os.environ.setdefault('MPLCONFIGDIR','/tmp/fusion010-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
data=json.loads((a.directory/'comparison.json').read_text())['decode']
fig,axes=plt.subplots(1,3,figsize=(11,4),layout='constrained')
for ax,batch in zip(axes,[1,32,256]):
 rows=[r for r in data if r['batch']==batch]
 controls=[r['decode_output_tok_s'] for r in rows if r['variant'].startswith('reference')]
 values=[statistics.mean(controls)]
 labels=['Eager']
 colors=['#888888']
 for variant,label,color in [('fused','Fused','#3e8dbc'),('graphs','+ graphs','#43a780'),('graphs_repeat','Repeat','#276c54')]:
  match=next((r for r in rows if r['variant']==variant),None)
  if match:values.append(match['decode_output_tok_s']);labels.append(label);colors.append(color)
 ax.bar(range(len(values)),values,color=colors)
 ax.set_xticks(range(len(values)),labels,rotation=25,ha='right')
 ax.set_title(f'Batch {batch}')
 ax.set_ylim(0,max(values)*1.22)
 for i,v in enumerate(values):ax.text(i,v+max(values)*.025,f'{v:.1f}' if batch==1 else f'{v:.0f}',ha='center',fontsize=9)
 ax.spines[['right','top']].set_visible(False)
axes[0].set_ylabel('Output tok/s across four GPUs')
fig.suptitle('Ambitious 2ph: paired decode-window controls')
fig.savefig(a.directory/'decode_speed.png',dpi=160)
fig.savefig(a.directory/'decode_speed.svg')
