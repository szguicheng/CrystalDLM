"""Plot full explicit-site-constraint program counts for the verified 512 CIFs."""
import csv
import json
from pathlib import Path
import shlex

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/'outputs/explicit_constraint_scatter_20260918'
OUT.mkdir(parents=True,exist_ok=True)
rows=json.loads((ROOT/'corpus_512/records.json').read_text())
assert len(rows)==512 and all(r['status']=='pass' for r in rows)
for r in rows:
    case=ROOT/'corpus_512'/r['id']
    assert len((case/'encoded.sym').read_text().split())==r['program_tokens']
    lexer=shlex.shlex((case/'input.cif').read_text(),posix=True)
    lexer.whitespace_split=True;lexer.commenters='#'
    assert len(list(lexer))==r['cif_tokens']
x=np.array([r['cif_tokens'] for r in rows]);y=np.array([r['program_tokens'] for r in rows])
short=y<=x
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral','DejaVu Serif'],
    'mathtext.fontset':'stix','font.size':12,'axes.labelsize':15,'axes.titlesize':17,
    'xtick.labelsize':12,'ytick.labelsize':12,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(8.4,8))
fig.subplots_adjust(left=.14,right=.97,top=.90,bottom=.17)
limit=int(np.ceil(max(x.max(),y.max())/20)*20)
ax.set_xlim(0,limit);ax.set_ylim(0,limit);ax.set_aspect('equal',adjustable='box')
ax.set_xticks(range(0,limit+1,20));ax.set_yticks(range(0,limit+1,20))
ax.plot([0,limit],[0,limit],color='#777F86',lw=1.3,ls='--',label='Equal count (y = x)',zorder=1)
ax.scatter(x[short],y[short],s=31,c='#287E6B',alpha=.40,linewidths=0,label=f'At or below CIF ({short.sum()})',zorder=2)
ax.scatter(x[~short],y[~short],s=48,c='#C56A3A',alpha=.95,edgecolors='white',linewidths=.6,label=f'Above CIF ({(~short).sum()})',zorder=3)
lookup={r['id']:r for r in rows}
for mid,label,offset in [('mp-30009',r'KrF$_2$  (59, 47)',(-50,-29)),('mp-1095422',r'YbBPt$_2$  (122, 49)',(30,-29))]:
    r=lookup[mid];px,py=r['cif_tokens'],r['program_tokens']
    ax.scatter([px],[py],s=75,facecolors='none',edgecolors='#214E75',linewidths=1.25,zorder=4)
    ax.annotate(label,(px,py),xytext=offset,textcoords='offset points',fontsize=11.5,color='#214E75',
                ha='center',va='top',arrowprops={'arrowstyle':'-','color':'#214E75','lw':.8},
                bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':1.5},zorder=5)
ax.set_xlabel('CIF tokens',fontweight='bold',labelpad=9)
ax.set_ylabel('Explicit site-constraint tokens',fontweight='bold',labelpad=9)
ax.set_title('CIF vs explicit site-constraint token counts',fontweight='bold',pad=15)
ax.tick_params(direction='in',length=4,width=.8)
for sp in ax.spines.values(): sp.set_color('#333333');sp.set_linewidth(.9)
ax.grid(True,color='#E0E0E0',linestyle=':',linewidth=.7);ax.set_axisbelow(True)
ax.legend(loc='upper left',frameon=True,facecolor='white',edgecolor='#CCCCCC',fontsize=11,markerscale=1.2)
fig.text(.14,.078,'512 materials; full programs include lattice, generators, precision and residuals.',fontsize=10.7,color='#535C63')
fig.text(.14,.048,'One arithmetic primitive, scalar or label = one unit (not BPE); no hidden dictionary.',fontsize=10.7,color='#535C63')
for ext in ['png','pdf','svg']: fig.savefig(OUT/f'cif_vs_explicit_constraints.{ext}',dpi=300,facecolor='white')
plt.close(fig)
with (OUT/'token_counts_512.csv').open('w',newline='') as stream:
    writer=csv.writer(stream);writer.writerow(['material_id','formula','cif_tokens','explicit_site_constraint_tokens'])
    for r in rows: writer.writerow([r['id'],r['formula'],r['cif_tokens'],r['program_tokens']])
stats={'materials':len(rows),'at_or_below_CIF':int(short.sum()),'above_CIF':int((~short).sum()),
       'total_cif_tokens':int(x.sum()),'total_program_tokens':int(y.sum()),'total_reduction':float(1-y.sum()/x.sum()),
       'median_ratio':float(np.median(y/x)),'max_ratio':float((y/x).max()),'all_counts_rechecked_against_actual_files':True}
(OUT/'statistics.json').write_text(json.dumps(stats,indent=2))
print(json.dumps({'output':str(OUT),**stats}))
