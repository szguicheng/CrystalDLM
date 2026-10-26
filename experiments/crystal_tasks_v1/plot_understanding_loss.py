"""Plot recorded AR answer-token CE; adapted from plot-from-data's loss inset style."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, ScalarFormatter
import numpy as np

RUN = Path(__file__).resolve().parent/'phase3_understanding_20261026'
SOURCE = RUN/'full/trainer_state.json'
OUT = RUN/'loss_curves'
OUT.mkdir(exist_ok=True)
state = json.loads(SOURCE.read_text())
train = [row for row in state['log_history'] if 'train_answer_ce' in row]
val = [row for row in state['log_history'] if 'eval_answer_ce' in row]
assert len(train)==909 and len(val)==18 and state['global_step']==9093
tx = np.array([r['step'] for r in train]); ty = np.array([r['train_answer_ce'] for r in train])
vx = np.array([r['step'] for r in val]); vy = np.array([r['eval_answer_ce'] for r in val])
assert np.all(np.isfinite(ty)) and np.all(ty>0) and np.all(vy>0)
best = int(np.argmin(vy)); assert vx[best]==7200

plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'], 'text.usetex':False,
    'font.size':10,'axes.labelsize':11,'axes.titlesize':12,'axes.unicode_minus':False,
    'pdf.fonttype':42,'ps.fonttype':42})
blue, orange = '#1F77B4', '#FF7F0E'
fig, ax = plt.subplots(figsize=(10.6,6.0))
fig.subplots_adjust(left=.105,right=.975,bottom=.21,top=.82)
fig.suptitle('AR-understanding: training and validation loss',x=.105,y=.975,ha='left',fontsize=16)
fig.text(.105,.925,'Qwen3-8B + LoRA  |  3 epochs  |  36,362 training / 4,501 validation examples',fontsize=10,color='#555555')
ax.plot(tx,ty,color=blue,lw=.9,alpha=.78,label='Training (10-update windows)')
ax.plot(vx,vy,color=orange,lw=1.8,marker='o',ms=4,label='Validation (full split)',zorder=4)
ax.set(yscale='log',xlim=(0,9300),ylim=(2e-7,6),xlabel='Optimizer update',ylabel='Answer-token cross-entropy (nats/token; log scale)')
ax.xaxis.set_major_locator(MultipleLocator(1500))
ax.grid(axis='y',which='major',color='#E0E0E0',linestyle=':',lw=.8)
ax.spines[['top','right']].set_visible(False)
ax.tick_params(direction='out')
for step in (3031,6062,9093):ax.axvline(step,color='#999999',ls='--',lw=.7,alpha=.45,zorder=0)
top=ax.secondary_xaxis('top',functions=(lambda x:x/3031,lambda x:x*3031))
top.set_xticks([0,1,2,3]);top.set_xlabel('Epoch',labelpad=6)
top.spines['top'].set_visible(False)
ax.legend(loc='upper left',bbox_to_anchor=(.15,1.0),frameon=False,fontsize=10)

inset=ax.inset_axes([.56,.55,.40,.32])
late=vx>=6000
inset.plot(vx[late],vy[late],color=orange,lw=1.5,marker='o',ms=3.8)
inset.scatter(vx[best],vy[best],s=50,facecolors='white',edgecolors=orange,linewidths=1.8,zorder=5)
inset.set(xlim=(5850,9250),ylim=(1.15e-4,1.59e-4),title='Late validation (linear scale)')
inset.set_xticks([6000,7200,9000]);inset.set_yticks([1.2e-4,1.4e-4,1.5e-4])
formatter=ScalarFormatter(useMathText=True);formatter.set_powerlimits((-4,-4));inset.yaxis.set_major_formatter(formatter)
inset.tick_params(labelsize=8)
for spine in inset.spines.values():spine.set_linewidth(1.0);spine.set_color('#555555')
inset.annotate('best: step 7,200',xy=(vx[best],vy[best]),xytext=(6260,1.535e-4),fontsize=8,
    arrowprops={'arrowstyle':'->','lw':.7,'color':'#555555'})
fig.text(.105,.075,'Raw recorded values; no smoothing. Prompt and padding tokens are excluded.',fontsize=9,color='#555555')
fig.text(.105,.035,f'Last training: {ty[-1]:.3e} (step {tx[-1]:,})     Best validation: {vy[best]:.3e}     Final validation: {vy[-1]:.3e}',fontsize=9)
fig.savefig(OUT/'understanding_training_validation_loss.png',dpi=300,facecolor='white')
fig.savefig(OUT/'understanding_training_validation_loss.pdf',facecolor='white')
plt.close(fig)

with (OUT/'loss_values.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=['split','step','epoch','answer_token_ce','trainer_logged_loss','answer_tokens'])
    writer.writeheader()
    for name,rows,prefix in [('training',train,'train'),('validation',val,'eval')]:
        for row in rows:writer.writerow({'split':name,'step':row['step'],'epoch':row['epoch'],
            'answer_token_ce':row[f'{prefix}_answer_ce'], 'trainer_logged_loss':row['loss' if name=='training' else 'eval_loss'],
            'answer_tokens':row[f'{prefix}_answer_tokens']})
meta={'source':'full/trainer_state.json','source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'training_points':len(train),'validation_points':len(val),'metric_fields':['train_answer_ce','eval_answer_ce'],
    'normalization':'sum CE / supervised answer-token count', 'training_scope':'logged 10-update windows',
    'validation_scope':'all 4501 validation examples', 'smoothing':False,'main_y_scale':'log',
    'best_validation_step':int(vx[best]), 'best_validation_ce':float(vy[best]), 'final_validation_ce':float(vy[-1]),
    'last_training_step':int(tx[-1]),'last_training_ce':float(ty[-1])}
(OUT/'plot_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
