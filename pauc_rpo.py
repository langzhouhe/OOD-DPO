#!/usr/bin/env python3
"""Can RPO beat balanced BCE by doing something BCE structurally cannot:
optimize a RESTRICTED REGION of the ROC curve (partial AUC / the FPR95 operating point)?

FPR95 = fraction of ID ranked above the threshold that keeps 95% of OOD.
It is governed by (a) ID samples with the HIGHEST scores (the false positives) and
(b) OOD samples with the LOWEST scores (the ones missed). A pairwise objective can
restrict its pairs to exactly that region; a pointwise BCE cannot express this.

Objectives (identical head/features/protocol, val-only hp selection):
  bce        : balanced BCE
  rpo        : plain pairwise logistic over sampled pairs
  pauc(a)    : pairwise logistic restricted to the top-a fraction of ID by score
               x the bottom-a fraction of OOD by score  (the FPR-relevant corner)
Reports FPR95 (primary) and AUROC (secondary)."""
import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
import json, pickle, argparse, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

CACHE=Path("cache/ood_dpo_cache")
BASE={"ec50_assay":"lbap_general_ec50_assay","ic50_assay":"lbap_general_ic50_assay",
      "ec50_scaffold":"lbap_general_ec50_scaffold","ic50_scaffold":"lbap_general_ic50_scaffold"}
KEYS=["train_id","train_ood","val_id","val_ood","test_id","test_ood"]
DESCS=[Descriptors.MolWt,Descriptors.MolLogP,Descriptors.TPSA,Descriptors.NumHDonors,
       Descriptors.NumHAcceptors,Descriptors.NumRotatableBonds,rdMolDescriptors.CalcNumRings,
       Descriptors.HeavyAtomCount,Descriptors.FractionCSP3,rdMolDescriptors.CalcNumAromaticRings]
BLK=["fm","um","mo","de"]; PROJ=256
ALPHAS=[0.05,0.1,0.25,0.5]
SEL=[1,2,3]; FIN=[11,12,13,14,15]

class Head(nn.Module):
    def __init__(s,dims):
        super().__init__()
        s.proj=nn.ModuleList([nn.Linear(d,PROJ) for d in dims])
        s.net=nn.Sequential(nn.Linear(PROJ*len(dims),256),nn.ReLU(),nn.Dropout(0.1),
                            nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.1),nn.Linear(128,1))
    def forward(s,bl): return s.net(torch.cat([p(b) for p,b in zip(s.proj,bl)],1)).squeeze(-1)

def featurize(cell):
    b=BASE[cell]
    fm=pickle.load(open(CACHE/f"{b}_minimol_features.pkl","rb"))["features"]
    um=pickle.load(open(CACHE/f"{b}_unimol_features.pkl","rb"))["features"]
    sp=json.load(open(CACHE/f"{b}_seed42_splits.json","rb"))["splits"]
    out={}
    for k in KEYS:
        A,B,C,D=[],[],[],[]
        for s in sp[k]:
            if s not in fm or s not in um: continue
            m=Chem.MolFromSmiles(s)
            if m is None: continue
            arr=np.zeros(2048,dtype=np.float32)
            DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048),arr)
            A.append(fm[s]); B.append(um[s]); C.append(arr)
            D.append(np.array([f(m) for f in DESCS],dtype=np.float32))
        out[k]={"fm":np.stack(A).astype(np.float32),"um":np.stack(B).astype(np.float32),
                "mo":np.stack(C),"de":np.stack(D).astype(np.float32)}
    return out

def mets(head,A,B):
    with torch.no_grad(): s=np.r_[head(A).numpy(),head(B).numpy()]
    y=np.r_[np.zeros(len(A[0])),np.ones(len(B[0]))]
    p,r,_=precision_recall_curve(y,s); fpr,tpr,_=roc_curve(y,s); i=np.searchsorted(tpr,0.95)
    return roc_auc_score(y,s),auc(r,p),fpr[min(i,len(fpr)-1)]

def train(V,kind,hp,seed,epochs=120,n_id=1500):
    g=torch.Generator().manual_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    dims=[b.shape[1] for b in V["train_id"]]
    iid=torch.randperm(len(V["train_id"][0]),generator=g)[:n_id]
    iod=torch.randperm(len(V["train_ood"][0]),generator=g)[:2000]
    Xid=[b[iid] for b in V["train_id"]]; Xood=[b[iod] for b in V["train_ood"]]
    lr=hp[1] if kind=="bce" else 1e-4
    head=Head(dims); opt=torch.optim.AdamW(head.parameters(),lr,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.StepLR(opt,10,0.9); best,bs=-1,None
    for ep in range(epochs):
        head.train(); opt.zero_grad(); Eid,Eood=head(Xid),head(Xood)
        if kind=="bce":
            lam=hp[0]
            core=0.5*F.binary_cross_entropy_with_logits(Eid,torch.zeros(len(Eid)))+ \
                 0.5*F.binary_cross_entropy_with_logits(Eood,torch.ones(len(Eood)))
        elif kind=="rpo":
            beta,lam=hp
            idx=torch.randint(len(Eid),(len(Eood),),generator=g)
            core=F.softplus(-beta*(Eood-Eid[idx])).mean()
        else:  # pauc: restrict to the FPR-relevant corner
            beta,lam,alpha=hp
            k_id=max(8,int(alpha*len(Eid))); k_od=max(8,int(alpha*len(Eood)))
            with torch.no_grad():
                hi_id=torch.topk(Eid,k_id).indices          # ID scored highest -> false positives
                lo_od=torch.topk(-Eood,k_od).indices        # OOD scored lowest  -> missed
            diff=Eood[lo_od].unsqueeze(0)-Eid[hi_id].unsqueeze(1)
            core=F.softplus(-beta*diff).mean()
        (core+lam*(Eid.pow(2).mean()+Eood.pow(2).mean())).backward()
        nn.utils.clip_grad_norm_(head.parameters(),1.0); opt.step(); sch.step()
        if (ep+1)%15==0:
            head.eval()
            with torch.no_grad():
                sv=np.r_[head(V["val_id"]).numpy(),head(V["val_ood"]).numpy()]
            yv=np.r_[np.zeros(len(V["val_id"][0])),np.ones(len(V["val_ood"][0]))]
            fpr,tpr,_=roc_curve(yv,sv); i=np.searchsorted(tpr,0.95)
            v=-fpr[min(i,len(fpr)-1)]                        # select on val FPR95 (lower=better)
            if v>best: best,bs=v,{k:t.clone() for k,t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval()
    return best,mets(head,V["test_id"],V["test_ood"])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cell",required=True); a=ap.parse_args()
    data=featurize(a.cell)
    st={p:(data["train_id"][p].mean(0),data["train_id"][p].std(0)+1e-6) for p in BLK}
    V={k:[torch.tensor((data[k][p]-st[p][0])/st[p][1]) for p in BLK] for k in KEYS}
    cfgs=[("bce",[(l,lr) for l in [0.,0.01,0.1] for lr in [1e-4,3e-4]]),
          ("rpo",[(b,l) for b in [1.,5.,10.] for l in [0.,0.1,0.5]])]
    cfgs+=[(f"pauc{al}",[(b,l,al) for b in [1.,5.,10.] for l in [0.,0.1,0.5]]) for al in ALPHAS]
    out={}
    for name,grid in cfgs:
        kind="pauc" if name.startswith("pauc") else name
        best=None
        for hp in grid:
            v=float(np.mean([train(V,kind,hp,s)[0] for s in SEL]))
            if best is None or v>best[1]: best=(hp,v)
        R=[train(V,kind,best[0],s)[1] for s in FIN]
        au=np.array([x[0] for x in R]); f95=np.array([x[2] for x in R])
        out[name]={"hp":str(best[0]),"auroc":float(au.mean()),"auroc_std":float(au.std(ddof=1)),
                   "fpr95":float(f95.mean()),"fpr95_std":float(f95.std(ddof=1)),
                   "aupr":float(np.mean([x[1] for x in R]))}
        print(f"[{a.cell}] {name:9} AUROC={au.mean():.4f}  FPR95={f95.mean():.4f}±{f95.std(ddof=1):.4f}  hp={best[0]}",flush=True)
    json.dump(out,open(f"repro/pauc_{a.cell}.json","w"),indent=2)
    print(f"PAUC_DONE {a.cell}",flush=True)

if __name__=="__main__": main()
