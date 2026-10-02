#!/usr/bin/env python3
"""
SC-LACL broad validation harness
=================================
Real continual-learning baselines vs. SC-LACL on multiple concept-drift streams,
with uniform, genuine per-cell NVM program/erase (P/E) wear accounting.

Datasets (all streamed in time order):
  ELEC2      - real electricity-market drift (binary, 13 feat)
  COVERTYPE  - real forest-cover stream (7-class, 54 feat; 40k contiguous block)
  SEA        - synthetic SEA generator with known abrupt drifts (binary, 3 feat)
               -> ground-truth drift points let us measure recovery cleanly.

Policies (identical MLP + identical wear accounting):
  STATIC   frozen after cloud pretraining
  NAIVE    dense full-intensity adaptation every slot
  PERIODIC dense adaptation every K slots
  EWC      Elastic Weight Consolidation (Fisher penalty), dense daily
  ER       Experience Replay (reservoir buffer), dense daily
  LwF      Learning without Forgetting (distillation), dense daily
  SCLACL   carbon break-even gate + endurance-aware sparse wear-leveled updates

Key point: EWC/ER/LwF improve accuracy/stability but are wear-agnostic (dense
writes) -> they wear like NAIVE. SC-LACL is the only wear-aware learner.

Metrics: prequential accuracy, worst rolling-window accuracy (stability),
SEA post-drift recovery, total cell P/E ops, MTTF, device replacements,
blackouts, embodied carbon, tCLDP. Multi-seed with std.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent   # repository root
import json, numpy as np, pandas as pd
from dataclasses import dataclass, asdict

# --------------------------------------------------------------------------- #
@dataclass
class Cfg:
    periods_per_day: int = 48
    pretrain_days: int = 20
    hidden: int = 64
    lr: float = 0.05
    # wear (datasheet-grounded; commodity Flash)
    endurance_cycles: float = 1e4
    target_life_frac: float = 0.85
    # energy (normalized/day)
    harvest_mean: float = 0.95; season_amp: float = 0.30; noise: float = 0.22
    deep_fade_prob: float = 0.04; buffer_cap: float = 2.5
    e_infer: float = 0.20; e_write_per_kop: float = 0.020
    embodied_kg: float = 2.59
    # policy knobs
    naive_epochs: int = 6
    periodic_K: int = 21
    ewc_lambda: float = 80.0; ewc_interval: int = 45
    er_buffer: int = 512
    lwf_alpha: float = 0.6; lwf_interval: int = 30
    # SC-LACL
    V: float = 12.0; w_wear: float = 1.0; w_energy: float = 0.4
    sclacl_topk: float = 0.18; sclacl_max_epochs: int = 4
    seeds: int = 5
    # PRICE baseline (static shadow-price gate, calibrated once, then frozen)
    price_w: float = 0.0; price_e: float = 0.0


# ---- K-class MLP (numpy) with explicit dlogits backprop ---- #
class MLP:
    def __init__(self, d, h, K, rng):
        self.K=K
        self.W1=rng.normal(0,np.sqrt(2/d),(d,h)); self.b1=np.zeros(h)
        self.W2=rng.normal(0,np.sqrt(2/h),(h,K)); self.b2=np.zeros(K)
    def params(self): return [self.W1,self.b1,self.W2,self.b2]
    def forward(self,X):
        self.X=X; self.z1=X@self.W1+self.b1; self.a1=np.maximum(0,self.z1)
        z2=self.a1@self.W2+self.b2; z2=z2-z2.max(1,keepdims=True); e=np.exp(z2)
        self.p=e/e.sum(1,keepdims=True); return self.p
    def predict(self,X): return self.forward(X).argmax(1)
    def backward_from_dlogits(self,dz2):
        gW2=self.a1.T@dz2; gb2=dz2.sum(0)
        da1=dz2@self.W2.T; dz1=da1*(self.z1>0)
        gW1=self.X.T@dz1; gb1=dz1.sum(0)
        return [gW1,gb1,gW2,gb2]
    def ce_grads(self,X,y,soft=None,alpha=0.0):
        n=len(y); p=self.forward(X); oh=np.zeros_like(p); oh[np.arange(n),y]=1
        dz2=(p-oh)/n
        if soft is not None: dz2=dz2+alpha*(p-soft)/n   # LwF distillation
        return self.backward_from_dlogits(dz2)
    def clone_params(self): return [p.copy() for p in self.params()]


def apply_grads(model,grads,cfg,writes,topk=1.0,level=False):
    """Write-counted parameter update. Dense or top-k sparse (wear-leveled)."""
    ops=0
    for k,(p,gr) in enumerate(zip(model.params(),grads)):
        if topk<1.0 and gr.size>8:
            score=np.abs(gr).ravel().astype(float)
            if level:
                wc=writes[k].ravel(); score=score/(1.0+wc/(wc.mean()+1e-9))
            ntop=max(1,int(topk*gr.size)); sel=np.argpartition(score,-ntop)[-ntop:]
            p.ravel()[sel]-=cfg.lr*gr.ravel()[sel]; writes[k].ravel()[sel]+=1; ops+=ntop
        else:
            p-=cfg.lr*gr; writes[k]+=1; ops+=gr.size
    return ops

# --------------------------------------------------------------------------- #
def _std(X): return (X-X.mean(0))/(X.std(0)+1e-9)

def load_elec2(path=str(ROOT/"data"/"elec.arff")):
    rows=[r.strip().split(",") for r in open(path) if r.strip() and not r.startswith(("@","%"))]
    df=pd.DataFrame(rows,columns=["date","day","period","nswprice","nswdemand","vicprice","vicdemand","transfer","cls"])
    num=["period","nswprice","nswdemand","vicprice","vicdemand","transfer"]
    X=df[num].astype(float).values; oh=np.eye(7)[df["day"].astype(int).values-1]
    X=np.hstack([_std(X),oh]); y=(df["cls"].values=="UP").astype(int)
    return X.astype(float),y,2

def load_covertype(path=str(ROOT/"data"/"covertype.arff"),n=40000,start=60000):
    rows=[]
    for r in open(path):
        if r.startswith(("@","%")) or not r.strip(): continue
        rows.append(r.strip().split(","))
        if len(rows)>=start+n: break
    rows=rows[start:start+n]
    arr=np.array(rows,dtype=float); X=arr[:,:-1]; y=(arr[:,-1].astype(int)-1)
    # standardize the 10 continuous cols; binary soil/wilderness left as-is
    X[:,:10]=_std(X[:,:10])
    return X,y,7

def gen_sea(n=40000,seed=0):
    rng=np.random.default_rng(seed); X=rng.uniform(0,10,(n,3))
    thetas=[8.0,9.0,7.0,9.5]; seg=n//len(thetas); y=np.zeros(n,int); drifts=[]
    for i,th in enumerate(thetas):
        s=i*seg; e=n if i==len(thetas)-1 else (i+1)*seg
        if i>0: drifts.append(s)
        y[s:e]=((X[s:e,0]+X[s:e,1])<=th).astype(int)
    flip=rng.random(n)<0.10; y[flip]=1-y[flip]                 # 10% label noise
    X[:,:2]=_std(X[:,:2]); X[:,2]=_std(X[:,2])
    return X,y,2,drifts

def solar(T,cfg,rng):
    d=np.arange(T); h=cfg.harvest_mean+cfg.season_amp*np.sin(2*np.pi*d/365)+rng.normal(0,cfg.noise,T)
    h=np.clip(h,0,None); h[rng.random(T)<cfg.deep_fade_prob]*=0.15; return h

# --------------------------------------------------------------------------- #
def run(dataset, policy, cfg, seed, topk=None, level=True, gate=True):
    rng=np.random.default_rng(seed)
    if dataset=="ELEC2": X,y,K=load_elec2(); drifts=[]
    elif dataset=="COVERTYPE": X,y,K=load_covertype(); drifts=[]
    else: X,y,K,drifts=gen_sea(seed=seed)
    P=cfg.periods_per_day; T=len(y)//P; d=X.shape[1]
    if topk is None: topk=cfg.sclacl_topk
    model=MLP(d,cfg.hidden,K,rng); writes=[np.zeros_like(p) for p in model.params()]
    pi=cfg.pretrain_days*P
    for _ in range(12): apply_grads(model,model.ce_grads(X[:pi],y[:pi]),cfg,writes)
    writes=[np.zeros_like(p) for p in model.params()]
    horizon_yr=T*P*0.5/24/365.0; h=solar(T,cfg,rng)
    target_w=cfg.endurance_cycles*cfg.target_life_frac/T
    soc=cfg.buffer_cap; Qw=Qe=0.0; devices=1; first_death=None
    total_ops=0; blackouts=0; acc=np.zeros(T); wear_tr=np.zeros(T); recent=[]
    Qw_hist=[]; Qe_hist=[]
    # baseline-specific state
    buf_X=np.zeros((cfg.er_buffer,d)); buf_y=np.zeros(cfg.er_buffer,int); buf_n=0; seen=0
    F=[np.zeros_like(p) for p in model.params()]; theta_star=model.clone_params()
    old_params=model.clone_params()
    def peak(): return max(float(w.max()) for w in writes)/cfg.endurance_cycles

    for t in range(cfg.pretrain_days,T):
        sl=slice(t*P,(t+1)*P); Xb,yb=X[sl],y[sl]
        acc[t]=float((model.predict(Xb)==yb).mean()); recent.append(acc[t]); recent=recent[-7:]
        epochs,tk,lv=0,1.0,False
        if policy=="NAIVE": epochs=cfg.naive_epochs
        elif policy=="PERIODIC": epochs=cfg.naive_epochs if t%cfg.periodic_K==0 else 0
        elif policy in("EWC","ER","LWF"): epochs=cfg.naive_epochs
        elif policy=="SCLACL":
            drift=1.0-np.mean(recent)
            est_ops=topk*sum(p.size for p in model.params())*2
            est_w=est_ops/sum(p.size for p in model.params())
            est_e=cfg.e_infer+cfg.e_write_per_kop*est_ops/1000
            oa=cfg.V*drift+cfg.w_wear*Qw*(est_w/target_w-1)+cfg.w_energy*Qe*(est_e-h[t])
            os_=cfg.V*drift*1.20
            if (oa<os_) or (not gate):
                epochs=int(np.clip(1+round(3*drift),1,cfg.sclacl_max_epochs)); tk=topk; lv=level
        elif policy=="PRICE":
            # Static shadow-price gate: identical write mechanism to SC-LACL (same
            # top-k wear-leveled sparse update), but Qw/Qe (adaptive Lyapunov virtual
            # queues) are replaced by cfg.price_w/cfg.price_e -- fixed constants,
            # calibrated once (see calibrate_price()) and never updated by feedback.
            drift=1.0-np.mean(recent)
            est_ops=topk*sum(p.size for p in model.params())*2
            est_w=est_ops/sum(p.size for p in model.params())
            est_e=cfg.e_infer+cfg.e_write_per_kop*est_ops/1000
            oa=cfg.V*drift+cfg.w_wear*cfg.price_w*(est_w/target_w-1)+cfg.w_energy*cfg.price_e*(est_e-h[t])
            os_=cfg.V*drift*1.20
            if (oa<os_) or (not gate):
                epochs=int(np.clip(1+round(3*drift),1,cfg.sclacl_max_epochs)); tk=topk; lv=level
        if soc+h[t]<cfg.e_infer:
            blackouts+=1; soc=max(0,soc+h[t]); acc[t]=0.0; wear_tr[t]=peak()
            Qw=max(0,Qw-1); Qe=max(0,Qe+cfg.e_infer-h[t]); continue
        ops=0
        if epochs>0:
            # Energy check BEFORE training: every policy runs only as many epochs as the
            # buffer + today's harvest can pay for after inference (no free writes).
            step=sum((max(1,int(tk*p.size)) if (tk<1.0 and p.size>8) else p.size) for p in model.params())
            e_epoch=cfg.e_write_per_kop*int(np.ceil(len(yb)/16))*step/1000
            epochs=min(epochs,max(0,int((soc+h[t]-cfg.e_infer)//e_epoch)))
        if epochs>0:
            for _ in range(epochs):
                idx=rng.permutation(len(yb))
                for s in range(0,len(yb),16):
                    j=idx[s:s+16]; xb,yy=Xb[j],yb[j]
                    if policy=="ER" and buf_n>0:
                        m=min(16,buf_n); bi=rng.integers(0,buf_n,m)
                        xb=np.vstack([xb,buf_X[bi]]); yy=np.concatenate([yy,buf_y[bi]])
                    soft=None
                    if policy=="LWF":
                        # distill toward old model on current inputs
                        om=MLP(d,cfg.hidden,K,rng);
                        for pp,op in zip(om.params(),old_params): pp[...]=op
                        soft=om.forward(xb); g=model.ce_grads(xb,yy,soft=soft,alpha=cfg.lwf_alpha)
                    else:
                        g=model.ce_grads(xb,yy)
                    if policy=="EWC":
                        for kk,(pp) in enumerate(model.params()):
                            g[kk]=g[kk]+cfg.ewc_lambda*F[kk]*(pp-theta_star[kk])
                    ops+=apply_grads(model,g,cfg,writes,topk=tk,level=lv)
            total_ops+=ops
            if peak()>=1.0:
                if first_death is None: first_death=t
                devices+=1
                for w in writes: w.fill(0)
        # reservoir buffer update (ER)
        for ii in range(len(yb)):
            seen+=1
            if buf_n<cfg.er_buffer: buf_X[buf_n]=Xb[ii]; buf_y[buf_n]=yb[ii]; buf_n+=1
            else:
                r=rng.integers(0,seen)
                if r<cfg.er_buffer: buf_X[r]=Xb[ii]; buf_y[r]=yb[ii]
        # EWC consolidation
        if policy=="EWC" and t%cfg.ewc_interval==0:
            p=model.forward(Xb); oh=np.zeros_like(p); oh[np.arange(len(yb)),yb]=1
            dz2=(p-oh)/len(yb); gg=model.backward_from_dlogits(dz2)
            F=[0.9*f+np.square(g) for f,g in zip(F,gg)]; theta_star=model.clone_params()
        if policy=="LWF" and t%cfg.lwf_interval==0: old_params=model.clone_params()
        e_need=cfg.e_infer+cfg.e_write_per_kop*ops/1000
        if soc+h[t]<e_need: e_need=cfg.e_infer
        soc=float(np.clip(soc+h[t]-e_need,0,cfg.buffer_cap)); wear_tr[t]=peak()
        day_pe=ops/max(sum(p.size for p in model.params()),1)
        Qw=max(0,Qw+day_pe/target_w-1); Qe=max(0,Qe+e_need-h[t])
        Qw_hist.append(Qw); Qe_hist.append(Qe)

    mttf=(first_death/T*horizon_yr) if first_death else horizon_yr
    a=acc[cfg.pretrain_days:]; mean_acc=float(a.mean())
    roll=np.convolve(a,np.ones(30)/30,mode="valid"); worst=float(roll.min()) if len(roll) else mean_acc
    recov=None
    if drifts:
        rs=[]
        for dpt in drifts:
            dd=dpt//P
            if dd+25<T: rs.append(float(acc[dd+5:dd+25].mean()))
        recov=float(np.mean(rs)) if rs else None
    embodied=devices*cfg.embodied_kg; annual=embodied/horizon_yr; tcldp=annual/max(mean_acc,1e-3)
    return dict(dataset=dataset,policy=policy,mttf_yr=round(mttf,2),devices=devices,
                mean_acc=round(mean_acc,4),worst_acc=round(worst,4),
                recovery=None if recov is None else round(recov,4),
                Mops=round(total_ops/1e6,2),blackouts=blackouts,
                embodied_kg=round(embodied,2),tCLDP=round(tcldp,3),
                Qw_mean=float(np.mean(Qw_hist)) if Qw_hist else 0.0,
                Qe_mean=float(np.mean(Qe_hist)) if Qe_hist else 0.0,
                _acc=acc,_wear=wear_tr,_death=first_death,_hz=horizon_yr)


def calibrate_price(cfg):
    """Freeze PRICE's static gate at SC-LACL's own long-run average queue
    state on ELEC2 (5 seeds) -- a fair, once-calibrated-then-frozen baseline,
    not a strawman: it starts exactly where the adaptive controller settles,
    it just never adapts again, including on the other two streams."""
    rs=[run("ELEC2","SCLACL",cfg,sd) for sd in range(cfg.seeds)]
    pw=float(np.mean([r['Qw_mean'] for r in rs])); pe=float(np.mean([r['Qe_mean'] for r in rs]))
    return pw,pe

def main():
    cfg=Cfg()
    cfg.price_w,cfg.price_e=calibrate_price(cfg)
    print(f"PRICE calibration (frozen, from SC-LACL/ELEC2 mean queues): price_w={cfg.price_w:.4f} price_e={cfg.price_e:.4f}")
    pols=["STATIC","NAIVE","PERIODIC","EWC","ER","LWF","PRICE","SCLACL"]
    dsets=["ELEC2","COVERTYPE","SEA"]
    rows=[]; agg={}; rep={}
    for ds in dsets:
        for p in pols:
            rs=[run(ds,p,cfg,sd) for sd in range(cfg.seeds)]
            agg[(ds,p)]=rs
            if (ds=="ELEC2"): rep[p]=rs[0]
            m=lambda k:np.mean([r[k] for r in rs]); s=lambda k:np.std([r[k] for r in rs])
            rec=[r['recovery'] for r in rs if r['recovery'] is not None]
            rows.append(dict(dataset=ds,policy=p,
                acc=f"{m('mean_acc'):.3f}±{s('mean_acc'):.3f}",
                worst=f"{m('worst_acc'):.3f}",
                recov=("-" if not rec else f"{np.mean(rec):.3f}"),
                MTTF=f"{m('mttf_yr'):.2f}",dev=f"{m('devices'):.1f}",
                Mops=f"{m('Mops'):.1f}",tCLDP=f"{m('tCLDP'):.3f}±{s('tCLDP'):.3f}"))
    df=pd.DataFrame(rows); pd.set_option("display.width",200)
    print(df.to_string(index=False)); df.to_csv(str(ROOT/"results"/"exp_main.csv"),index=False)

    # ablations on ELEC2
    print("\n--- ABLATIONS (ELEC2, mean over 5 seeds) ---")
    abl=[]
    for tk in [0.05,0.10,0.18,0.35,1.0]:
        rs=[run("ELEC2","SCLACL",cfg,sd,topk=tk) for sd in range(cfg.seeds)]
        abl.append(dict(ablation=f"sparsity topk={tk}",acc=f"{np.mean([r['mean_acc'] for r in rs]):.3f}",
                        Mops=f"{np.mean([r['Mops'] for r in rs]):.2f}",tCLDP=f"{np.mean([r['tCLDP'] for r in rs]):.3f}"))
    for lv in [True,False]:
        rs=[run("ELEC2","SCLACL",cfg,sd,level=lv) for sd in range(cfg.seeds)]
        abl.append(dict(ablation=f"wear-leveling={lv}",acc=f"{np.mean([r['mean_acc'] for r in rs]):.3f}",
                        Mops=f"{np.mean([r['Mops'] for r in rs]):.2f}",tCLDP=f"{np.mean([r['tCLDP'] for r in rs]):.3f}"))
    for gt in [True,False]:
        rs=[run("ELEC2","SCLACL",cfg,sd,gate=gt) for sd in range(cfg.seeds)]
        abl.append(dict(ablation=f"break-even gate={gt}",acc=f"{np.mean([r['mean_acc'] for r in rs]):.3f}",
                        Mops=f"{np.mean([r['Mops'] for r in rs]):.2f}",tCLDP=f"{np.mean([r['tCLDP'] for r in rs]):.3f}"))
    adf=pd.DataFrame(abl); print(adf.to_string(index=False)); adf.to_csv(str(ROOT/"results"/"exp_ablation.csv"),index=False)
    json.dump(asdict(cfg),open(str(ROOT/"results"/"exp_config.json"),"w"),indent=2,default=str)
    return cfg,agg,rep,df,adf

if __name__=="__main__":
    main()
