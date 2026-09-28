#!/usr/bin/env python3
"""Revision run implementing Phase 0 of REVISION_EXPERIMENT_ORGANIZATION_ZH.md.

Protocol fixes vs the earlier tables:
 1. EQUAL TUNING BUDGET - beta fixed to 1 (removes the beta/lambda confound, doc 4.0);
    both objectives search the SAME 3x3 grid over (score regulariser, lr) = 9 trials.
 2. EQUAL SAMPLE EXPOSURE - both objectives get the identical ID/OOD molecule subset
    and identical batch indices at every seed.
 3. CONVENTIONAL FPR95 - ID is the positive class: threshold = 95th percentile of ID
    scores, FPR95 = fraction of OOD falling below it (matches the paper's wording).
 4. VALIDATION-ONLY selection everywhere (no test-based model choice).
 5. PARAMETER-MATCHED head - every view config projects to the same total width, so
    single-view and multi-view share the identical scoring head.
 6. DOMAIN-DISJOINT train-OOD / validation-OOD for DrugOOD (by domain_id), because the
    molecule-random split leaves up to 100% domain overlap on the assay tasks.
 7. Primary endpoint = macro AUROC over the four DrugOOD non-size cells.
"""
import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
import json, pickle, argparse, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
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
PRIMARY=["ec50_scaffold","ec50_assay","ic50_scaffold","ic50_assay"]
KEYS=["train_id","train_ood","val_id","val_ood","test_id","test_ood"]
DESCS=[Descriptors.MolWt,Descriptors.MolLogP,Descriptors.TPSA,Descriptors.NumHDonors,
       Descriptors.NumHAcceptors,Descriptors.NumRotatableBonds,rdMolDescriptors.CalcNumRings,
       Descriptors.HeavyAtomCount,Descriptors.FractionCSP3,rdMolDescriptors.CalcNumAromaticRings]
VIEWDEF={"minimol":["fm"],"unimol":["um"],
         "minimol_mv":["fm","mo","de"],"unimol_mv":["um","mo","de"]}
TOTAL=768                      # fixed total projected width -> identical scoring head
BETA=1.0                       # fixed; only gamma (=lambda) and lr are tuned
GAMMAS=[0.001,0.01,0.1]; LRS=[1e-4,3e-4,1e-3]    # 3x3 = 9 trials, SAME grid for BOTH
# gamma spans the effective regulariser lambda/beta^2 of the earlier best RPO configs (~0.001-0.005)
SEL=[1,2,3]; FIN=[11,12,13,14,15]

# ---------- metrics (conventional FPR95: ID positive) ----------
def mets(s_id,s_ood):
    y=np.r_[np.zeros(len(s_id)),np.ones(len(s_ood))]; s=np.r_[s_id,s_ood]
    au=roc_auc_score(y,s)
    p,r,_=precision_recall_curve(y,s); ap=auc(r,p)
    tau=np.quantile(s_id,0.95)                 # keep 95% of ID below tau
    fpr95=float((np.asarray(s_ood)<=tau).mean())   # OOD wrongly accepted as ID
    return au,ap,fpr95

# ---------- data ----------
def domain_disjoint_split(cell,sp):
    """Re-partition the cached train_ood + val_ood molecules so that the two sets use
    DISJOINT domain_ids (molecule-random splitting leaves up to 100% overlap)."""
    if cell not in DRUGOOD: return sp,None
    d=json.load(open(f"data/raw/{BASE[cell]}.json"))["split"]
    dom={it["smiles"]:it.get("domain_id") for it in d.get("ood_val",[]) if it.get("smiles")}
    pool=[s for s in (list(sp["train_ood"])+list(sp["val_ood"])) if s in dom]
    if not pool: return sp,None
    by={}
    for s in pool: by.setdefault(dom[s],[]).append(s)
    ds=sorted(by,key=lambda k:(len(by[k]),str(k)))
    val,tr,n=[],[],0
    target=len(sp["val_ood"])
    for k in ds:
        if n<target: val+=by[k]; n+=len(by[k])
        else: tr+=by[k]
    sp=dict(sp); sp["train_ood"]=tr; sp["val_ood"]=val
    ov=len({dom[s] for s in val}&{dom[s] for s in tr})
    return sp,{"val_domains":len({dom[s] for s in val}),"overlap":ov,
               "n_train_ood":len(tr),"n_val_ood":len(val)}

def load(cell,need_um,domain_split=True):
    b=BASE[cell]
    fm=pickle.load(open(CACHE/f"{b}_minimol_features.pkl","rb"))["features"]
    um=pickle.load(open(CACHE/f"{b}_unimol_features.pkl","rb"))["features"] if need_um else None
    sp=json.load(open(CACHE/f"{b}_seed42_splits.json","rb"))["splits"]
    info=None
    if domain_split: sp,info=domain_disjoint_split(cell,sp)
    out={}
    for k in KEYS:
        A,B,C,D=[],[],[],[]
        for s in sp[k]:
            if s not in fm or (need_um and s not in um): continue
            m=Chem.MolFromSmiles(s)
            if m is None: continue
            arr=np.zeros(2048,dtype=np.float32)
            DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048),arr)
            A.append(fm[s]); C.append(arr)
            D.append(np.array([f(m) for f in DESCS],dtype=np.float32))
            if need_um: B.append(um[s])
        out[k]={"fm":np.stack(A).astype(np.float32),"mo":np.stack(C),"de":np.stack(D).astype(np.float32)}
        if need_um: out[k]["um"]=np.stack(B).astype(np.float32)
    return out,info

def blocks(data,view):
    parts=VIEWDEF[view]
    st={p:(data["train_id"][p].mean(0),data["train_id"][p].std(0)+1e-6) for p in parts}
    return {k:[torch.tensor((data[k][p]-st[p][0])/st[p][1]) for p in parts] for k in KEYS}

# ---------- head (parameter-matched: same total width & same scoring MLP) ----------
class Head(nn.Module):
    def __init__(s,dims):
        super().__init__()
        w=TOTAL//len(dims)
        s.proj=nn.ModuleList([nn.Linear(d,w) for d in dims])
        s.net=nn.Sequential(nn.Linear(w*len(dims),256),nn.ReLU(),nn.Dropout(0.1),
                            nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.1),nn.Linear(128,1))
    def forward(s,bl): return s.net(torch.cat([p(b) for p,b in zip(s.proj,bl)],1)).squeeze(-1)

def train(V,kind,gamma,lr,seed,epochs=120,n_id=1500,n_ood=2000):
    # identical subset + identical pairing indices for both objectives
    g=torch.Generator().manual_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    iid=torch.randperm(len(V["train_id"][0]),generator=g)[:n_id]
    iod=torch.randperm(len(V["train_ood"][0]),generator=g)[:n_ood]
    Xid=[b[iid] for b in V["train_id"]]; Xood=[b[iod] for b in V["train_ood"]]
    head=Head([b.shape[1] for b in V["train_id"]])
    opt=torch.optim.AdamW(head.parameters(),lr,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.StepLR(opt,10,0.9); best,bs=-1,None
    for ep in range(epochs):
        head.train(); opt.zero_grad(); Eid,Eood=head(Xid),head(Xood)
        if kind=="rpo":
            # identical molecules for both objectives; pairing is resampled each
            # step because that is intrinsic to a pairwise loss (BCE has no analogue)
            pair=torch.randint(len(Eid),(len(Eood),),generator=g)
            core=F.softplus(-BETA*(Eood-Eid[pair])).mean()
        else: core=0.5*F.binary_cross_entropy_with_logits(Eid,torch.zeros(len(Eid)))+ \
                   0.5*F.binary_cross_entropy_with_logits(Eood,torch.ones(len(Eood)))
        (core+gamma*(Eid.pow(2).mean()+Eood.pow(2).mean())).backward()
        nn.utils.clip_grad_norm_(head.parameters(),1.0); opt.step(); sch.step()
        if (ep+1)%15==0:
            head.eval()
            with torch.no_grad():
                v=roc_auc_score(np.r_[np.zeros(len(V["val_id"][0])),np.ones(len(V["val_ood"][0]))],
                                np.r_[head(V["val_id"]).numpy(),head(V["val_ood"]).numpy()])
            if v>best: best,bs=v,{k:t.clone() for k,t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval()
    with torch.no_grad():
        return best,mets(head(V["test_id"]).numpy(),head(V["test_ood"]).numpy()),head

def run_objective(V,kind):
    best=None
    for gm in GAMMAS:
        for lr in LRS:                                    # 9 trials, identical for both
            v=float(np.mean([train(V,kind,gm,lr,s)[0] for s in SEL]))
            if best is None or v>best[2]: best=(gm,lr,v)
    gm,lr,_=best
    R=[train(V,kind,gm,lr,s)[1] for s in FIN]
    return R,{"gamma":gm,"lr":lr}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cell",required=True)
    ap.add_argument("--views",nargs="+",default=list(VIEWDEF)); a=ap.parse_args()
    data,info=load(a.cell,True)
    out={"domain_split":info}
    for view in a.views:
        V=blocks(data,view)
        for kind,name in [("bce","BalancedOODHead"),("rpo","RPO")]:
            R,hp=run_objective(V,kind)
            out[f"{view}|{name}"]={"auroc":[x[0] for x in R],"aupr":[x[1] for x in R],
                                   "fpr95":[x[2] for x in R],"hp":hp}
        r=out[f"{view}|RPO"]; b=out[f"{view}|BalancedOODHead"]
        print(f"[{a.cell}] {view:11} RPO={np.mean(r['auroc']):.4f} BOH={np.mean(b['auroc']):.4f} "
              f"Δ={np.mean(r['auroc'])-np.mean(b['auroc']):+.4f} | FPR95 {np.mean(r['fpr95']):.3f}/{np.mean(b['fpr95']):.3f}",flush=True)
    json.dump(out,open(f"repro/rev_{a.cell}.json","w"),indent=2)
    print(f"REV_DONE {a.cell}",flush=True)

if __name__=="__main__": main()
