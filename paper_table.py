#!/usr/bin/env python3
"""Produce the paper's main table: all methods x all cells x AUROC/AUPR/FPR95, 5 seeds.

Method groups
  Post-hoc (NO auxiliary OOD; scored from an ID-only property classifier or ID features):
      MSP, ODIN, Energy, Mahalanobis, KNN, LOF
  Auxiliary-OOD supervised (same ID/OOD data, same head, same tuning budget):
      Balanced OOD Head (balanced BCE),  RPO (ours)
Representations: minimol | unimol | multiview(MiniMol+UniMol+Morgan+descriptors)

Everything matched: same splits, same head family, val-only model/hyperparameter
selection, 5 seeds; per-seed values stored so paired CIs can be computed.
"""
import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
import json, pickle, argparse, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

CACHE=Path("cache/ood_dpo_cache")
BASE={"ec50_scaffold":"lbap_general_ec50_scaffold","ec50_size":"lbap_general_ec50_size",
      "ec50_assay":"lbap_general_ec50_assay","ic50_scaffold":"lbap_general_ic50_scaffold",
      "ic50_size":"lbap_general_ic50_size","ic50_assay":"lbap_general_ic50_assay",
      "hiv_scaffold":"good_hiv_scaffold_covariate","hiv_size":"good_hiv_size_covariate",
      "pcba_scaffold":"good_pcba_scaffold_covariate","pcba_size":"good_pcba_size_covariate",
      "zinc_scaffold":"good_zinc_scaffold_covariate","zinc_size":"good_zinc_size_covariate"}
DRUGOOD={"ec50_scaffold","ec50_size","ec50_assay","ic50_scaffold","ic50_size","ic50_assay"}
KEYS=["train_id","train_ood","val_id","val_ood","test_id","test_ood"]
DESCS=[Descriptors.MolWt,Descriptors.MolLogP,Descriptors.TPSA,Descriptors.NumHDonors,
       Descriptors.NumHAcceptors,Descriptors.NumRotatableBonds,rdMolDescriptors.CalcNumRings,
       Descriptors.HeavyAtomCount,Descriptors.FractionCSP3,rdMolDescriptors.CalcNumAromaticRings]
PROJ=256; SEL=[1,2,3]; FIN=[11,12,13,14,15]
VIEWDEF={"minimol":["fm"],"unimol":["um"],
         "minimol_mv":["fm","mo","de"],"unimol_mv":["um","mo","de"],
         "multiview":["fm","um","mo","de"]}
RPO_GRID=[(b,l) for b in [1.,5.,10.] for l in [0.,0.1,0.5]]
BCE_GRID=[(l,lr) for l in [0.,0.01,0.1] for lr in [1e-4,3e-4]]

def mets(s_id,s_ood):
    y=np.r_[np.zeros(len(s_id)),np.ones(len(s_ood))]; s=np.r_[s_id,s_ood]
    p,r,_=precision_recall_curve(y,s); fpr,tpr,_=roc_curve(y,s); i=np.searchsorted(tpr,0.95)
    return roc_auc_score(y,s),auc(r,p),float(fpr[min(i,len(fpr)-1)])

# ---------------- features ----------------
def load_blocks(cell,need_um):
    b=BASE[cell]
    fm=pickle.load(open(CACHE/f"{b}_minimol_features.pkl","rb"))["features"]
    um=pickle.load(open(CACHE/f"{b}_unimol_features.pkl","rb"))["features"] if need_um else None
    sp=json.load(open(CACHE/f"{b}_seed42_splits.json","rb"))["splits"]
    lab={}
    if cell in DRUGOOD:
        d=json.load(open(f"data/raw/{b}.json"))["split"]
        for k in ["train","iid_val","iid_test","ood_val","ood_test"]:
            for it in d.get(k,[]):
                if it.get("smiles") and it.get("cls_label") is not None: lab[it["smiles"]]=int(it["cls_label"])
    out={}
    for k in KEYS:
        A,B,C,D,Y=[],[],[],[],[]
        for s in sp[k]:
            if s not in fm or (need_um and s not in um): continue
            m=Chem.MolFromSmiles(s)
            if m is None: continue
            arr=np.zeros(2048,dtype=np.float32)
            DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048),arr)
            A.append(fm[s]); C.append(arr)
            D.append(np.array([f(m) for f in DESCS],dtype=np.float32))
            if need_um: B.append(um[s])
            Y.append(lab.get(s,-1))
        out[k]={"fm":np.stack(A).astype(np.float32),"mo":np.stack(C),
                "de":np.stack(D).astype(np.float32),"y":np.array(Y)}
        if need_um: out[k]["um"]=np.stack(B).astype(np.float32)
    return out

def view_blocks(data,view):
    parts=VIEWDEF[view]
    st={p:(data["train_id"][p].mean(0),data["train_id"][p].std(0)+1e-6) for p in parts}
    return {k:[torch.tensor((data[k][p]-st[p][0])/st[p][1]) for p in parts] for k in KEYS}

# ---------------- detector head ----------------
class Head(nn.Module):
    def __init__(s,dims):
        super().__init__()
        s.proj=nn.ModuleList([nn.Linear(d,PROJ) for d in dims])
        s.net=nn.Sequential(nn.Linear(PROJ*len(dims),256),nn.ReLU(),nn.Dropout(0.1),
                            nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.1),nn.Linear(128,1))
    def forward(s,bl): return s.net(torch.cat([p(b) for p,b in zip(s.proj,bl)],1)).squeeze(-1)

def train_head(V,kind,hp,seed,epochs=120,n_id=1500):
    g=torch.Generator().manual_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    dims=[b.shape[1] for b in V["train_id"]]
    iid=torch.randperm(len(V["train_id"][0]),generator=g)[:n_id]
    iod=torch.randperm(len(V["train_ood"][0]),generator=g)[:2000]
    Xid=[b[iid] for b in V["train_id"]]; Xood=[b[iod] for b in V["train_ood"]]
    lr=1e-4 if kind=="rpo" else hp[1]
    head=Head(dims); opt=torch.optim.AdamW(head.parameters(),lr,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.StepLR(opt,10,0.9); best,bs=-1,None
    for ep in range(epochs):
        head.train(); opt.zero_grad(); Eid,Eood=head(Xid),head(Xood)
        if kind=="rpo":
            beta,lam=hp
            idx=torch.randint(len(Eid),(len(Eood),),generator=g)
            core=F.softplus(-beta*(Eood-Eid[idx])).mean()
        else:
            lam=hp[0]
            core=0.5*F.binary_cross_entropy_with_logits(Eid,torch.zeros(len(Eid)))+ \
                 0.5*F.binary_cross_entropy_with_logits(Eood,torch.ones(len(Eood)))
        (core+lam*(Eid.pow(2).mean()+Eood.pow(2).mean())).backward()
        nn.utils.clip_grad_norm_(head.parameters(),1.0); opt.step(); sch.step()
        if (ep+1)%15==0:
            head.eval()
            with torch.no_grad():
                v=roc_auc_score(np.r_[np.zeros(len(V["val_id"][0])),np.ones(len(V["val_ood"][0]))],
                                np.r_[head(V["val_id"]).numpy(),head(V["val_ood"]).numpy()])
            if v>best: best,bs=v,{k:t.clone() for k,t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval()
    with torch.no_grad():
        return best,mets(head(V["test_id"]).numpy(),head(V["test_ood"]).numpy())

def oe_method(V,kind):
    grid=RPO_GRID if kind=="rpo" else BCE_GRID
    best=None
    for hp in grid:
        v=float(np.mean([train_head(V,kind,hp,s)[0] for s in SEL]))
        if best is None or v>best[1]: best=(hp,v)
    R=[train_head(V,kind,best[0],s)[1] for s in FIN]
    return R,str(best[0])

# ---------------- post-hoc baselines (NO auxiliary OOD) ----------------
class Clf(nn.Module):
    def __init__(s,d,c=2):
        super().__init__(); s.f=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Dropout(0.5),
                                              nn.Linear(64,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(0.5))
        s.o=nn.Linear(64,c)
    def feat(s,x): return s.f(x)
    def forward(s,x): return s.o(s.f(x))

def posthoc(data,view,seed):
    parts=VIEWDEF[view]
    st={p:(data["train_id"][p].mean(0),data["train_id"][p].std(0)+1e-6) for p in parts}
    X={k:np.concatenate([(data[k][p]-st[p][0])/st[p][1] for p in parts],1).astype(np.float32) for k in KEYS}
    y=data["train_id"]["y"]
    torch.manual_seed(seed); np.random.seed(seed)
    res={}
    Xtr=torch.tensor(X["train_id"]); ytr=torch.tensor(np.where(y>=0,y,0))
    if (ytr.numpy()>=0).all() and len(set(ytr.numpy().tolist()))>1:
        clf=Clf(Xtr.shape[1]); opt=torch.optim.Adam(clf.parameters(),0.01,weight_decay=5e-4)
        for ep in range(200):
            clf.train(); opt.zero_grad(); F.cross_entropy(clf(Xtr),ytr).backward(); opt.step()
        clf.eval()
        def logits(a):
            with torch.no_grad(): return clf(torch.tensor(a))
        def feats(a):
            with torch.no_grad(): return clf.feat(torch.tensor(a)).numpy()
        li,lo=logits(X["test_id"]),logits(X["test_ood"])
        res["MSP"]=mets(1-F.softmax(li,1).max(1).values.numpy(),1-F.softmax(lo,1).max(1).values.numpy())
        res["Energy"]=mets(-torch.logsumexp(li,1).numpy(),-torch.logsumexp(lo,1).numpy())
        T=1000.
        res["ODIN"]=mets(1-F.softmax(li/T,1).max(1).values.numpy(),1-F.softmax(lo/T,1).max(1).values.numpy())
    # distance/density baselines operate on the FROZEN representation itself
    # (as in the paper), not on classifier penultimate activations
    ftr=X["train_id"]; fi,fo=X["test_id"],X["test_ood"]
    mu=ftr.mean(0); cov=np.cov(ftr,rowvar=False)+1e-3*np.eye(ftr.shape[1]); P=np.linalg.pinv(cov)
    md=lambda A: np.einsum("ij,jk,ik->i",A-mu,P,A-mu)
    res["Mahalanobis"]=mets(md(fi),md(fo))
    nn_=NearestNeighbors(n_neighbors=50).fit(ftr)
    res["KNN"]=mets(nn_.kneighbors(fi)[0].mean(1),nn_.kneighbors(fo)[0].mean(1))
    lof=LocalOutlierFactor(n_neighbors=20,novelty=True).fit(ftr)
    res["LOF"]=mets(-lof.score_samples(fi),-lof.score_samples(fo))
    return res

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cell",required=True)
    ap.add_argument("--views",nargs="+",default=["minimol","unimol","minimol_mv","unimol_mv"]); a=ap.parse_args()
    need_um=any(v in ("unimol","multiview") for v in a.views)
    data=load_blocks(a.cell,need_um)
    out={}
    for view in a.views:
        ph={m:[] for m in ["MSP","ODIN","Energy","Mahalanobis","KNN","LOF"]}
        for s in FIN:
            r=posthoc(data,view,s)
            for m,v in r.items(): ph[m].append(v)
        for m,vals in ph.items():
            if vals: out[f"{view}|{m}"]={"auroc":[v[0] for v in vals],"aupr":[v[1] for v in vals],"fpr95":[v[2] for v in vals]}
        V=view_blocks(data,view)
        for kind,name in [("bce","BalancedOODHead"),("rpo","RPO")]:
            R,hp=oe_method(V,kind)
            out[f"{view}|{name}"]={"auroc":[x[0] for x in R],"aupr":[x[1] for x in R],
                                   "fpr95":[x[2] for x in R],"hp":hp}
        print(f"[{a.cell}] {view:10} "+"  ".join(
            f"{n}={np.mean(out[f'{view}|{n}']['auroc']):.3f}" for n in
            ["MSP","Energy","KNN","BalancedOODHead","RPO"] if f"{view}|{n}" in out),flush=True)
    json.dump(out,open(f"repro/table_{a.cell}.json","w"),indent=2)
    print(f"TABLE_DONE {a.cell}",flush=True)

if __name__=="__main__": main()
