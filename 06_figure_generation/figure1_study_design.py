# Figure 1: cohort assembly and directional state rule
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from _utils.plot_utils import setup_style, save_fig, PALETTE, COLORS, _lighten
setup_style()
OUT = Path('figures')
OUT.mkdir(exist_ok=True)

def draw_sankey(ax, source_nodes, target_nodes, flows, left_x=0.16, right_x=0.84, node_width=0.035, node_gap=0.05):
    total_src = sum(v for _, v in source_nodes); total_tgt = sum(v for _, v in target_nodes)
    if abs(total_src - total_tgt) > 1e-6: raise ValueError('Sankey totals must balance')
    margin = 0.12
    avs = 1 - 2*margin - node_gap*max(0, len(source_nodes)-1)
    avt = 1 - 2*margin - node_gap*max(0, len(target_nodes)-1)
    src_ranges=[]; y=1-margin
    for _,v in source_nodes:
        h=v/total_src*avs; src_ranges.append((y-h,y)); y -= h+node_gap
    tgt_ranges=[]; y=1-margin
    for _,v in target_nodes:
        h=v/total_tgt*avt; tgt_ranges.append((y-h,y)); y -= h+node_gap
    src_colors=[PALETTE[i % len(PALETTE)] for i in range(len(source_nodes))]
    for i,((name,v),(y0,y1)) in enumerate(zip(source_nodes,src_ranges)):
        ax.add_patch(Rectangle((left_x-node_width/2,y0),node_width,y1-y0,facecolor=src_colors[i],edgecolor='none',alpha=.92,zorder=3))
        ax.text(left_x-node_width/2-.015,(y0+y1)/2,f'{name}\n{v:,}',ha='right',va='center',fontsize=9,color=COLORS['text'],fontweight='bold')
    for i,((name,v),(y0,y1)) in enumerate(zip(target_nodes,tgt_ranges)):
        ax.add_patch(Rectangle((right_x-node_width/2,y0),node_width,y1-y0,facecolor=COLORS['text'],edgecolor='none',alpha=.92,zorder=3))
        ax.text(right_x+node_width/2+.015,(y0+y1)/2,f'{name}\n{v:,}',ha='left',va='center',fontsize=9,color=COLORS['text'],fontweight='bold')
    s_used=[r[1] for r in src_ranges]; t_used=[r[1] for r in tgt_ranges]
    for f in flows:
        si,ti,val=f['source_idx'],f['target_idx'],f['value']; sh=val/total_src*avs; th=val/total_tgt*avt
        stop=s_used[si]; sbot=stop-sh; s_used[si]=sbot; ttop=t_used[ti]; tbot=ttop-th; t_used[ti]=tbot
        mx=(left_x+right_x)/2
        pts=[(MplPath.MOVETO,(left_x+node_width/2,stop)),(MplPath.CURVE4,(mx,stop)),(MplPath.CURVE4,(mx,ttop)),(MplPath.CURVE4,(right_x-node_width/2,ttop)),(MplPath.LINETO,(right_x-node_width/2,tbot)),(MplPath.CURVE4,(mx,tbot)),(MplPath.CURVE4,(mx,sbot)),(MplPath.CURVE4,(left_x+node_width/2,sbot)),(MplPath.CLOSEPOLY,(left_x+node_width/2,stop))]
        codes,verts=zip(*pts); ax.add_patch(PathPatch(MplPath(verts,codes),facecolor=src_colors[si],edgecolor='none',alpha=.42,zorder=2))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

fig = plt.figure(figsize=(11, 7.0))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.35], hspace=.22)
ax = fig.add_subplot(gs[0]); draw_sankey(ax,[('ELSA',5621),('CHARLS',6283),('HRS',8978)],[('Trajectory gate',20882)],[{'source_idx':0,'target_idx':0,'value':5621},{'source_idx':1,'target_idx':0,'value':6283},{'source_idx':2,'target_idx':0,'value':8978}])
ax.text(.5,.98,'Development cohorts and trajectory gate',ha='center',va='top',fontsize=12,fontweight='bold',color=COLORS['text'])
ax.text(.5,.04,'Known outcome category: 18,888 of 20,882 trajectory participants',ha='center',va='bottom',fontsize=9,color=COLORS['text'])

ax2 = fig.add_subplot(gs[1])
# Directional rule as an empirical 2x2 map; percentages are pooled development-cohort shares.
ax2.axhline(0,color=COLORS['ref_line'],lw=1.2); ax2.axvline(0,color=COLORS['ref_line'],lw=1.2)
state_info=[(-.5,.5,'Coupled IC decline\n+ FI accumulation','30.6%',PALETTE[3]),(.5,.5,'FI accumulation only','20.6%',PALETTE[2]),(-.5,-.5,'IC decline only','27.0%',PALETTE[1]),(.5,-.5,'Preserved IC\n/ low accumulation','21.8%',PALETTE[0])]
for x,y,label,pct,col in state_info:
    ax2.add_patch(Rectangle((x-.46,y-.38),.92,.76,facecolor=_lighten(col,.78),edgecolor=col,lw=1.8,alpha=.9))
    ax2.text(x,y+.05,label,ha='center',va='center',fontsize=10,color=COLORS['text'],fontweight='bold')
    ax2.text(x,y-.22,pct,ha='center',va='center',fontsize=10,color=col,fontweight='bold')
ax2.set_xlim(-1.05,1.05); ax2.set_ylim(-1.0,1.0)
ax2.set_xlabel('IC change (negative = decline)',fontsize=10); ax2.set_ylabel('FI change (positive = accumulation)',fontsize=10)
ax2.set_title('Directional four-state construction in the development cohorts',fontsize=12,fontweight='bold',color=COLORS['text'],pad=10)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False); ax2.grid(False)
fig.text(.5,.005,'State labels are empirical directional patterns; development-cohort pooled shares are shown.',ha='center',va='bottom',fontsize=8.5,color=COLORS['text'])
fig.subplots_adjust(left=.10,right=.95,top=.96,bottom=.10)
for ext in ('png','tif'):
    fig.savefig(OUT/f'Figure1_study_design.{ext}',dpi=350,facecolor='white')
save_fig(fig, str(OUT/'Figure1_study_design.pdf'))
