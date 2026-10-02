#!/usr/bin/env python3
"""Render 5 SEPARATE small single-column figures (no large multi-panel image):
   sclacl_cycle_v2.png, sclacl_control_v2.png, sclacl_acc_v2.png, sclacl_tcldp_v2.png, sclacl_sparsity_v2.png
   (the file names used by the camera-ready paper)
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent   # repository root
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle
from matplotlib.lines import Line2D
INK="#10221a"; RED="#c0392b"; GREEN="#1f7a4d"; AMBER="#b4690e"; GREY="#6b7a72"
plt.rcParams.update({"font.family":"DejaVu Sans"})

def box(ax,x,y,w,h,t,fc,ec,tc="white",fs=6.6):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.05,rounding_size=0.1",fc=fc,ec=ec,lw=1.1,zorder=3))
    ax.text(x,y,t,ha="center",va="center",color=tc,fontsize=fs,fontweight="bold",zorder=4)
def arr(ax,p1,p2,c,lw=1.5,style="-|>"):
    ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle=style,mutation_scale=10,color=c,lw=lw,
                 shrinkA=0,shrinkB=0,zorder=2))

from PIL import Image
def crop(path, margin=8):
    """Trim the empty canvas matplotlib leaves around the drawing."""
    im=Image.open(path); a=np.asarray(im.convert("L")); r=np.where((a<245).any(1))[0]; c=np.where((a<245).any(0))[0]
    im.crop((max(c.min()-margin,0),max(r.min()-margin,0),min(c.max()+margin,a.shape[1]),min(r.max()+margin,a.shape[0]))).save(path,dpi=(170,170))

# ---------- 1. paradox cycle: arcs run behind the boxes and stop at their edges ----------
fig,ax=plt.subplots(figsize=(3.5,2.9)); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
cx,cy,R=5,4.8,3.25
nodes=["Concept\ndrift","Continual\nlearning","NVM P/E\nwrites","Memory\nwear-out","Premature\ndeath","Embodied\ncarbon ↑"]
angd=[90,30,-30,-90,-150,-210]
gap={90:24,-90:24,30:17,-30:17,-150:17,-210:17}          # degrees kept clear around each box
pt=lambda a:(cx+R*np.cos(np.radians(a)),cy+R*np.sin(np.radians(a)))
for i in range(6):
    a0=angd[i]-gap[angd[i]]; a1=angd[(i+1)%6]+gap[angd[(i+1)%6]] if i<5 else angd[0]-360+gap[90]
    th=np.linspace(a0,a1,60); xs=cx+R*np.cos(np.radians(th)); ys=cy+R*np.sin(np.radians(th))
    ax.plot(xs[:-2],ys[:-2],color=RED,lw=1.5,zorder=1,solid_capstyle="round")
    arr(ax,(xs[-4],ys[-4]),(xs[-1],ys[-1]),RED)
for a,t in zip(angd,nodes):
    fc=RED if("carbon" in t or "death" in t) else "#e8736a" if "wear" in t else "#f0a59e"
    box(ax,*pt(a),2.0,1.05,t,fc,RED,fs=6.4)
ax.text(cx,cy,"operational\nsavings\nNEGATED",ha="center",va="center",color=RED,fontsize=6.8,fontweight="bold",style="italic")
fig.tight_layout(); fig.savefig(str(ROOT/"figures"/"sclacl_cycle_v2.png"),dpi=170,bbox_inches="tight"); plt.close(fig)

# ---------- 2. SC-LACL control loop: wider boxes, arrows end at box edges ----------
fig,ax=plt.subplots(figsize=(3.5,2.9)); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
for t,yy in [("$Q_D$ drift",8.4),("$Q_E$ energy",6.9),("$Q_Z$ endurance",5.4)]:
    box(ax,1.9,yy,2.7,0.8,t,"#eaf3ee",GREEN,INK,6.6)
box(ax,5.3,6.9,2.6,1.5,"Lyapunov\ndrift-plus-\npenalty",GREEN,GREEN,fs=6.8)          # x 4.0-6.6, y 6.15-7.65
for y0,y1 in [(8.4,7.35),(6.9,6.9),(5.4,6.45)]: arr(ax,(3.35,y0),(3.93,y1),GREEN)
box(ax,5.3,4.3,4.0,0.8,"Carbon break-even gate","#f6efe2",AMBER,INK,6.4)          # y 3.9-4.7
box(ax,5.3,2.75,4.0,0.9,"Endurance-aware\nsparse update","#f6efe2",AMBER,INK,6.2)   # y 2.3-3.2
box(ax,5.3,1.15,4.0,0.7,"Model on NVM",INK,INK,fs=6.8)                            # y 0.8-1.5
arr(ax,(5.3,6.05),(5.3,4.82),GREEN); arr(ax,(5.3,3.82),(5.3,3.3),AMBER); arr(ax,(5.3,2.22),(5.3,1.6),AMBER)
arr(ax,(7.4,1.15),(9.1,1.15),GREY,lw=1.2,style="-")
arr(ax,(9.1,1.15),(9.1,6.9),GREY,lw=1.2,style="-")
arr(ax,(9.1,6.9),(6.72,6.9),GREY,lw=1.2)
ax.text(9.4,4.0,"measured P/E + energy",rotation=90,va="center",ha="center",color=GREY,fontsize=5.8)
fig.tight_layout(); fig.savefig(str(ROOT/"figures"/"sclacl_control_v2.png"),dpi=170,bbox_inches="tight"); plt.close(fig)

# ---------- data figures ----------
def mean(s): return float(str(s).split("±")[0])
m=pd.read_csv(ROOT/"results"/"exp_main.csv"); a=pd.read_csv(ROOT/"results"/"exp_ablation.csv")
for c in ["acc","tCLDP"]: m[c]=m[c].map(mean)
m["Mops"]=m["Mops"].astype(float)
col={"STATIC":"#8a8a8a","NAIVE":"#d1495b","PERIODIC":"#e8a13a","EWC":"#7b5ea7","ER":"#3a78b5","LWF":"#1f9e8f","SCLACL":"#1f7a4d"}
nice={"STATIC":"Static","NAIVE":"Naive","PERIODIC":"Periodic","EWC":"EWC","ER":"ER","LWF":"LwF","SCLACL":"SC-LACL"}
order=["STATIC","NAIVE","PERIODIC","EWC","ER","LWF","SCLACL"]
m=m[m.policy.isin(order)]   # Figs 3-4 show the seven policies of Table 1; PRICE ties SC-LACL and is reported separately

# 3. accuracy vs writes
fig,ax=plt.subplots(figsize=(3.5,2.7))
mk={"ELEC2":"o","COVERTYPE":"^"}
for ds in ["ELEC2","COVERTYPE"]:
    for _,r in m[m.dataset==ds].iterrows():
        ax.scatter(max(r.Mops,0.03),r.acc,s=55,marker=mk[ds],color=col[r.policy],edgecolor="k",linewidth=.4,zorder=3)
ax.set_xscale("log")
ax.annotate("SC-LACL",(0.8,0.80),(0.06,0.845),fontsize=7.5,color=GREEN,fontweight="bold",arrowprops=dict(arrowstyle="->",color=GREEN,lw=1))
ax.annotate("wear-agnostic CL",(18,0.81),(1.5,0.70),fontsize=7.5,color=RED,arrowprops=dict(arrowstyle="->",color=RED,lw=1))
ax.set_xlabel("Lifetime writes (M P/E, log)",fontsize=8); ax.set_ylabel("Prequential accuracy",fontsize=8)
ax.tick_params(labelsize=7); ax.grid(alpha=.25)
ax.legend(handles=[Line2D([],[],marker="o",ls="",color="k",label="ELEC2"),Line2D([],[],marker="^",ls="",color="k",label="Covertype")],fontsize=7,loc="lower right")
fig.tight_layout(); fig.savefig(str(ROOT/"figures"/"sclacl_acc_v2.png"),dpi=170,bbox_inches="tight"); plt.close(fig)

# 4. tCLDP grouped bars
fig,ax=plt.subplots(figsize=(3.5,2.7))
dsets=["ELEC2","COVERTYPE","SEA"]; x=np.arange(3); w=0.115
for i,p in enumerate(order):
    vals=[m[(m.dataset==d)&(m.policy==p)].tCLDP.values[0] for d in dsets]
    ax.bar(x+(i-3)*w,vals,w,color=col[p],label=nice[p],edgecolor="k" if p=="SCLACL" else "none",linewidth=1.1)
ax.set_xticks(x); ax.set_xticklabels(["ELEC2","Cover","SEA"],fontsize=8)
ax.set_ylabel("tCLDP  (lower better)",fontsize=8); ax.tick_params(labelsize=7)
ax.legend(fontsize=5.8,ncol=4,loc="upper center",columnspacing=0.8,handlelength=1.0); ax.grid(alpha=.25,axis="y"); ax.set_ylim(0,3.2)
fig.tight_layout(); fig.savefig(str(ROOT/"figures"/"sclacl_tcldp_v2.png"),dpi=170,bbox_inches="tight"); plt.close(fig)

# ---------- 5. sparsity frontier: labels placed clear of the line and the frame ----------
sp=a[a.ablation.str.startswith("sparsity")].copy()
sp["tk"]=sp.ablation.str.extract(r"topk=([0-9.]+)").astype(float); sp=sp.sort_values("tk")
sp["Mops"]=sp.Mops.astype(float); sp["acc"]=sp.acc.astype(float)
fig,ax=plt.subplots(figsize=(3.5,2.7))
ax.plot(sp.Mops,sp.acc,"o-",color=GREEN,lw=2)
place={0.05:(6,-3,"left","top"),0.1:(6,-3,"left","top"),0.18:(6,-3,"left","top"),0.35:(0,-7,"center","top"),1.0:(-3,-7,"right","top")}
for _,r in sp.iterrows():
    dx,dy,ha,va=place[round(r.tk,2)]
    ax.annotate(f"k={r.tk:g}",(r.Mops,r.acc),fontsize=6.5,xytext=(dx,dy),textcoords="offset points",ha=ha,va=va)
ax.set_xlabel("Lifetime writes (M P/E)",fontsize=8); ax.set_ylabel("accuracy",fontsize=8)
ax.set_ylim(0.785,0.805); ax.set_xlim(0,4.95); ax.tick_params(labelsize=7); ax.grid(alpha=.25)
ax.text(0.04,0.97,"19× writes,\n0.7-pt accuracy",transform=ax.transAxes,va="top",fontsize=7.5,color=GREEN,fontweight="bold")
fig.tight_layout(); fig.savefig(str(ROOT/"figures"/"sclacl_sparsity_v2.png"),dpi=170,bbox_inches="tight"); plt.close(fig)
for n in ["sclacl_cycle_v2.png","sclacl_control_v2.png","sclacl_sparsity_v2.png"]: crop(str(ROOT/"figures"/n))
print("saved 5 single figures: cycle_v2, control_v2, acc_v2, tcldp_v2, sparsity_v2")
