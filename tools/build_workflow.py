#!/usr/bin/env python3
"""Rebuild the public workflow diagram. Matplotlib figure is editable source."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib as mpl
mpl.rcParams.update({
  "font.family":"DejaVu Sans",
  "font.size":10,
  "mathtext.fontset":"dejavusans",
  "svg.fonttype":"none",
})
W,H=1057,528
fig,ax=plt.subplots(figsize=(W/100,H/100),dpi=100)
fig.patch.set_facecolor("white")
ax.set_xlim(0,W);ax.set_ylim(H,0);ax.axis("off")
fig.subplots_adjust(0,0,1,1)
blue="#1962F4"; orange="#F56D19"; fillblue="#EAF3FF"; fillorange="#FFF4E7"; gray="#89939E"
def panel(x,y,w,h):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=15",linewidth=1.3,edgecolor="#BDCCDE",facecolor="#FCFDFE"))
def box(x,y,w,h,txt,kind="blue",fs=11,weight="normal",lw=1.3):
 color=blue if kind=="blue" else orange if kind=="orange" else gray
 fill=fillblue if kind=="blue" else fillorange if kind=="orange" else "#F8F8F8"
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=6",edgecolor=color,facecolor=fill,linewidth=lw))
 ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,weight=weight,color="black")
def arrow(x1,y1,x2,y2,color=blue,style="-",lw=1.45):
 ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=12,color=color,linewidth=lw,linestyle=style,shrinkA=0,shrinkB=0))
def wire(points,color=blue,style="-",lw=1.4):
 for i in range(len(points)-2):
  a,b=points[i],points[i+1];ax.plot([a[0],b[0]],[a[1],b[1]],color=color,linewidth=lw,linestyle=style)
 arrow(*points[-2],*points[-1],color=color,style=style,lw=lw)
panel(1,2,1055,351);panel(1,363,1055,163)
ax.text(18,31,"a",fontsize=22,weight="bold")
ax.text(57,31,"MagPES-MACE framework for magnetization-conditioned multi-surface PES learning",fontsize=17,weight="bold")
box(865,14,39,18,"",kind="orange")
ax.text(913,27,"this work",fontsize=10,va="center")
box(865,42,39,18,"",kind="blue")
ax.text(913,55,"original MACE module",fontsize=10,va="center")

box(38,73,88,39,r"$Z$",fs=17)
box(177,73,156,39,"node embedding",fs=11)
box(376,59,262,64,"",kind="orange")
ax.text(507,82,"magnetic FiLM modulation",ha="center",fontsize=12.5,weight="bold")
ax.text(507,109,r"$x_i^\prime=\gamma(m_i^{in})\odot x_i+\beta(m_i^{in})$",fontsize=14,ha="center")
arrow(126,93,177,93);arrow(333,93,376,93)
box(38,148,88,39,r"$R,\mathrm{CELL}$",fs=13)
box(177,136,156,39,"spherical harmonics",fs=11)
box(177,188,156,39,"radial embedding",fs=11)
wire([(126,168),(154,168),(154,155),(177,155)])
wire([(154,168),(154,207),(177,207)])
box(675,71,165,214,"",kind="blue")
ax.text(758,91,"MACE interactions",ha="center",weight="bold",fontsize=13)
box(692,111,132,33,"message passing",fs=10)
box(692,149,132,33,"tensor product",fs=10)
box(692,188,132,33,"readout features",fs=10)
ax.text(758,255,"E(3)-equivariant",color=blue,ha="center",fontsize=11)
ax.text(758,272,"representation",color=blue,ha="center",fontsize=11)
arrow(638,92,675,92)
arrow(333,155,675,155)
arrow(333,207,675,207)
box(882,73,139,39,"energy readout",fs=11)
arrow(840,92,882,92)
box(888,135,125,39,"scale shift",fs=11)
arrow(952,112,952,135)
box(877,190,58,36,r"$E$",fs=18)
box(966,190,59,36,r"$F$",fs=18)
arrow(935,174,906,190);arrow(964,174,996,190)

box(36,258,116,39,r"$m^{in}$",kind="orange",fs=19)
box(166,258,178,39,"magnetic conditioning",kind="orange",fs=11)
arrow(152,278,166,278,color=orange)
wire([(344,278),(356,278),(356,100),(376,100)],color=orange)
# The optional box is a separate selectable conditioning path, not default.
ax.add_patch(FancyBboxPatch((380,222),280,118,boxstyle="round,pad=0,rounding_size=6",edgecolor=orange,facecolor=fillorange,linewidth=1.1,linestyle=(0,(5,3))))
ax.text(520,240,"Optional global magnetic-reversal symmetry",ha="center",fontsize=11.2,weight="bold",color="black")
ax.text(520,265,r"$q_i=(s_i^2,\;s_i\bar{s}_i)$",ha="center",fontsize=14.5)
ax.text(520,285,"symmetry-preserving magnetic conditioning",ha="center",fontsize=9.9,color="black")
ax.text(520,307,r"$E(R,m)=E(R,-m)$",ha="center",fontsize=13)
ax.text(520,327,r"$\hat{m}(R,-m)=-\hat{m}(R,m)$",ha="center",fontsize=13)
wire([(344,278),(380,278)],color=orange,style="--")
wire([(504,222),(504,167),(504,124)],color=orange,style="--")
ax.text(540,190,"optional",fontsize=10,color=orange,ha="center")
ax.text(540,205,"conditioning",fontsize=10,color=orange,ha="center")
box(880,242,147,42,"magnetic-moment\nreadout",kind="orange",fs=10.5)
arrow(840,263,880,263,color=orange)
box(908,309,91,36,r"$m^{pred}$",kind="orange",fs=17)
arrow(954,284,954,309,color=orange)
# Bottom panel
ax.text(19,392,"b",fontsize=22,weight="bold")
ax.text(57,392,"Closed-loop magnetic-state propagation in MHC-MD",fontsize=17,weight="bold")
box(41,408,79,36,r"$R_t$",fs=17)
box(41,469,79,36,r"$m_t^{in}$",kind="orange",fs=17)
box(175,427,157,46,"MagPES-MACE",kind="orange",fs=12.3,weight="bold")
arrow(120,426,175,444);arrow(120,487,175,456,color=orange)
box(389,399,80,34,r"$E_t$",fs=16)
box(389,435,80,34,r"$F_t$",fs=16)
box(389,472,80,34,r"$m_t^{pred}$",kind="orange",fs=16)
arrow(332,440,389,416);arrow(332,454,389,451);arrow(332,463,389,489,color=orange)
box(540,410,134,39,"MD integrator",fs=11.5)
wire([(469,451),(502,451),(502,430),(540,430)])
wire([(469,416),(505,416),(505,430)],lw=1.1)
box(747,410,91,39,r"$R_{t+1}$",fs=16)
arrow(674,430,747,430)
box(539,472,159,40,r"$m_{t+1}^{in}=m_t^{pred}$",kind="orange",fs=14)
arrow(469,490,539,490,color=orange)
ax.add_patch(FancyBboxPatch((841,449),136,63,boxstyle="round,pad=0,rounding_size=8",edgecolor="#888",facecolor="#FBFBFB",linewidth=1.3,linestyle="--"))
ax.text(909,472,"next",ha="center",fontsize=11)
ax.text(909,493,"MagPES-MACE",ha="center",fontsize=11)
wire([(838,430),(906,430),(906,449)])
arrow(698,490,841,490,color=orange)
ax.text(997,480,r"$\cdots$",fontsize=23,ha="center")
root=Path(__file__).resolve().parents[1]
(root/"figures").mkdir(parents=True,exist_ok=True)
for ext in ("svg","png"):
 path=root/"figures"/("workflow."+ext)
 fig.savefig(path,dpi=200 if ext=="png" else 100,bbox_inches=None,pad_inches=0,facecolor="white")
 if ext == "svg":
  path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
 print("WROTE",path, path.stat().st_size)
plt.close(fig)
