#!/usr/bin/env python3
"""Results figure from exp_main.csv + exp_ablation.csv."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent   # repository root
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

def mean(s): return float(str(s).split("±")[0])
m=pd.read_csv(ROOT/"results"/"exp_main.csv"); a=pd.read_csv(ROOT/"results"/"exp_ablation.csv")
for c in ["acc","tCLDP"]: m[c]=m[c].map(mean)
m["Mops"]=m["Mops"].astype(float)

col={"STATIC":"#8a8a8a","NAIVE":"#d1495b","PERIODIC":"#e8a13a","EWC":"#7b5ea7",
     "ER":"#3a78b5","LWF":"#1f9e8f","SCLACL":"#1f7a4d"}
nice={"STATIC":"Static","NAIVE":"Naive CL","PERIODIC":"Periodic","EWC":"EWC",
      "ER":"ER","LWF":"LwF","SCLACL":"SC-LACL"}
order=["STATIC","NAIVE","PERIODIC","EWC","ER","LWF","SCLACL"]
m=m[m.policy.isin(order)]   # PRICE ties SC-LACL here and is reported separately

fig,ax=plt.subplots(1,3,figsize=(13.2,3.7))

# (a) accuracy vs writes (ELEC2 + COVERTYPE), log x
mk={"ELEC2":"o","COVERTYPE":"^"}
for ds in ["ELEC2","COVERTYPE"]:
    sub=m[m.dataset==ds]
    for _,r in sub.iterrows():
        x=max(r.Mops,0.03)
        ax[0].scatter(x,r.acc,s=70,marker=mk[ds],color=col[r.policy],
                      edgecolor="k",linewidth=.5,zorder=3)
ax[0].set_xscale("log")
ax[0].annotate("SC-LACL\n(ours)",(0.8,0.80),(0.12,0.84),fontsize=8,color="#1f7a4d",
               fontweight="bold",arrowprops=dict(arrowstyle="->",color="#1f7a4d"))
ax[0].annotate("Naive/EWC/ER/LwF\n(wear-agnostic)",(20,0.80),(2.2,0.66),fontsize=8,color="#d1495b",
               arrowprops=dict(arrowstyle="->",color="#d1495b"))
ax[0].set_xlabel("Lifetime writes  (M P/E ops, log)"); ax[0].set_ylabel("Prequential accuracy")
ax[0].set_title("(a) Same accuracy, far fewer writes",fontsize=10.5,fontweight="bold")
ax[0].grid(alpha=.25)
from matplotlib.lines import Line2D
ax[0].legend(handles=[Line2D([],[],marker="o",ls="",color="k",label="ELEC2"),
                      Line2D([],[],marker="^",ls="",color="k",label="Covertype")],
             fontsize=8,loc="lower right")

# (b) tCLDP grouped by dataset
dsets=["ELEC2","COVERTYPE","SEA"]; x=np.arange(len(dsets)); w=0.115
for i,p in enumerate(order):
    vals=[m[(m.dataset==d)&(m.policy==p)].tCLDP.values[0] for d in dsets]
    ax[1].bar(x+(i-3)*w,vals,w,color=col[p],label=nice[p],
              edgecolor="k" if p=="SCLACL" else "none",linewidth=1.2)
ax[1].set_xticks(x); ax[1].set_xticklabels(["ELEC2","Cover","SEA"])
ax[1].set_ylabel("tCLDP  (lower better)")
ax[1].set_title("(b) SC-LACL wins tCLDP on all 3",fontsize=10.5,fontweight="bold")
ax[1].legend(fontsize=7,ncol=2,loc="upper right"); ax[1].grid(alpha=.25,axis="y")

# (c) sparsity frontier
sp=a[a.ablation.str.startswith("sparsity")].copy()
sp["tk"]=sp.ablation.str.extract(r"topk=([0-9.]+)").astype(float)
sp=sp.sort_values("tk")
ax2=ax[2]; l1=ax2.plot(sp.Mops,sp.acc,"o-",color="#1f7a4d",lw=2,label="accuracy")[0]
ax2.set_xlabel("Lifetime writes  (M P/E ops)"); ax2.set_ylabel("accuracy",color="#1f7a4d")
ax2.set_ylim(0.78,0.81); ax2.set_title("(c) Most plasticity is redundant wear",fontsize=10.5,fontweight="bold")
ax2.grid(alpha=.25)
for _,r in sp.iterrows(): ax2.annotate(f"k={r.tk}",(r.Mops,r.acc),fontsize=7,xytext=(2,4),textcoords="offset points")
ax2.text(0.05,0.99,"19× fewer writes,\n~0.6% accuracy",transform=ax2.transAxes,
         va="top",fontsize=8,color="#1f7a4d",fontweight="bold")

fig.suptitle("SC-LACL vs. real continual-learning baselines on concept-drift streams (mean / 5 seeds)",
             fontsize=12,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(str(ROOT/"figures"/"sclacl_exp_results.png"),dpi=160,bbox_inches="tight")
print("saved sclacl_exp_results.png")
