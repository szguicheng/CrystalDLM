"""Plot verified ten-CIF field-token counts; adapt plot-from-data scatter style."""
import csv
import json
from pathlib import Path
import re
import shlex

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/'outputs/token_count_scatter_20260917'
OUT.mkdir(parents=True,exist_ok=True)
rows=json.loads((ROOT/'fixed_dictionary_results/records.json').read_text())
assert len(rows)==10
for row in rows:
    tokens=(ROOT/'fixed_dictionary_results'/row['id']/'encoded.tokens').read_text().split()
    lexer=shlex.shlex((ROOT/'results'/row['id']/'input.cif').read_text(),posix=True)
    lexer.whitespace_split=True; lexer.commenters='#'
    assert len(tokens)==row['actual_stream_tokens']
    assert len(list(lexer))==row['input_cif_tokens']

plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral','DejaVu Serif'],
    'mathtext.fontset':'stix','font.size':12,'axes.labelsize':15,'axes.titlesize':18,
    'xtick.labelsize':12,'ytick.labelsize':12,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(9,7))
fig.subplots_adjust(left=.12,right=.97,top=.89,bottom=.19)
ax.set_xlim(40,180); ax.set_ylim(0,620)
ax.set_xticks(range(40,181,20)); ax.set_yticks(range(0,601,100))
ax.plot([40,180],[40,180],color='#7D848A',lw=1.3,ls='--',label='Equal count (y = x)',zorder=1)
offsets={'mp-4047':(10,-15),'mp-1205515':(10,9),'mp-30009':(-8,10),
    'mp-1188285':(10,9),'mp-1207068':(10,-9),'mp-1205902':(-10,10),
    'mp-1229077':(10,9),'mp-1019259':(-10,9),'mp-1079976':(-10,9),'mp-1095422':(10,8)}
for row in rows:
    x,y=row['input_cif_tokens'],row['actual_stream_tokens']
    color='#21866B' if y<x else '#315F89'
    ax.scatter(x,y,s=78,c=color,edgecolors='white',linewidths=.8,zorder=3)
    formula=re.sub(r'(\d+)',r'_{\1}',row['formula'])
    dx,dy=offsets[row['id']]
    ax.annotate(r'$\mathrm{'+formula+'}$',(x,y),xytext=(dx,dy),textcoords='offset points',
                ha='left' if dx>0 else 'right',va='center',fontsize=13,color='#233746')
ax.set_xlabel('CIF tokens',fontweight='bold',labelpad=9)
ax.set_ylabel('Edge-orbit tokens',fontweight='bold',labelpad=9)
ax.set_title('CIF vs edge-orbit token counts',fontweight='bold',pad=14)
ax.grid(True,color='#E0E0E0',lw=.7,linestyle=':',zorder=0)
ax.set_axisbelow(True)
for spine in ax.spines.values(): spine.set_color('#333333'); spine.set_linewidth(.9)
ax.tick_params(direction='in',length=4,width=.8)
ax.legend(loc='upper right',frameon=True,edgecolor='#CCCCCC',facecolor='white',fontsize=11)
fig.text(.12,.082,'10 materials; frozen dictionary; full edge-orbit sequence includes geometry and headers.',fontsize=11,color='#535C63')
fig.text(.12,.047,'One scalar or label = one unit (not BPE). The shared dictionary is counted separately.',fontsize=11,color='#535C63')
for suffix in ['png','pdf','svg']:
    fig.savefig(OUT/f'cif_vs_edge_orbit_tokens.{suffix}',dpi=300,facecolor='white')
plt.close(fig)
with (OUT/'token_counts.csv').open('w',newline='') as stream:
    writer=csv.writer(stream)
    writer.writerow(['material_id','formula','cif_tokens','edge_orbit_tokens','decoded_cif_tokens'])
    for r in rows: writer.writerow([r['id'],r['formula'],r['input_cif_tokens'],r['actual_stream_tokens'],r['decoded_cif_tokens']])
(OUT/'counting_notes.txt').write_text('X: original CIF grammar tokens, excluding comments; quoted values count once.\nY: fixed-dictionary full positional sequence, including lattice, independent free parameters and five framing fields.\nShared dictionary excluded per crystal. Counts are structural scalar/label units, not BPE or bytes.\nAll 10 pairs were checked against the actual input.cif and encoded.tokens files before plotting.\n')
print(OUT)
