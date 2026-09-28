#!/usr/bin/env python3
"""Phase 1 (revised question): is the MULTI-VIEW gain concentrated on ID-OOD pairs where
the views DISAGREE?

The original question ("why does RPO beat BCE under multi-view?") is moot: under the
corrected Phase-0 protocol RPO and the Balanced OOD Head are statistically tied. What
still needs a mechanism is the multi-view gain itself, which is real and holds for both
backbones. So we compare SINGLE-VIEW vs MULTI-VIEW pairwise accuracy as a function of
cross-view disagreement, for both objectives.

Hypothesis (doc 3.4/4.1): FM, Morgan and descriptors give conflicting ranking evidence
for some pairs; a pointwise loss must fix absolute logits while a pairwise loss can
directly repair the relative order -> gain should grow with cross-view disagreement.

Procedure
  1. train a view-specific scorer for each block (fm|um, mo, de) with the SAME protocol;
  2. for every test ID-OOD pair compute per-view margins  D_v = s_v(ood) - s_v(id);
  3. disagreement d = 1 - |mean_v sign(D_v)|   (0 = views agree, 1 = maximal conflict);
  4. bucket pairs by d (low/mid/high, cut-points chosen on VALIDATION pairs);
  5. report pairwise accuracy of the final multi-view RPO and Balanced OOD Head per bucket.
Falsification: if the RPO-BCE gap does not grow with d, the multi-view mechanism story
is not supported and should not be claimed.
"""
import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
import json, argparse, warnings, logging
import numpy as np, torch
torch.set_num_threads(1)
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from revision_run import (load, blocks, train, run_objective, VIEWDEF, FIN, KEYS)

def scores(head,V,split):
    with torch.no_grad(): return head(V[split]).numpy()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cell",required=True)
    ap.add_argument("--backbone",default="minimol",choices=["minimol","unimol"])
    ap.add_argument("--max_pairs",type=int,default=400000); a=ap.parse_args()
    fmkey="fm" if a.backbone=="minimol" else "um"
    mv=f"{a.backbone}_mv"
    data,info=load(a.cell,True)

    # --- view-specific scorers (same protocol, validation-selected) ---
    vs={}
    for blk in [fmkey,"mo","de"]:
        VIEWDEF[f"_solo_{blk}"]=[blk]
        V=blocks(data,f"_solo_{blk}")
        R,hp=run_objective(V,"rpo")
        _,_,head=train(V,"rpo",hp["gamma"],hp["lr"],FIN[0])
        vs[blk]={"id":scores(head,V,"test_id"),"ood":scores(head,V,"test_ood"),
                 "vid":scores(head,V,"val_id"),"vood":scores(head,V,"val_ood")}

    # --- final models: single-view vs multi-view, for both objectives ---
    fin={}
    for viewname,tag in [(a.backbone,"SV"),(mv,"MV")]:
        Vx=blocks(data,viewname)
        for kind,kn in [("bce","BOH"),("rpo","RPO")]:
            R,hp=run_objective(Vx,kind)
            _,_,head=train(Vx,kind,hp["gamma"],hp["lr"],FIN[0])
            fin[f"{tag}-{kn}"]={"id":scores(head,Vx,"test_id"),"ood":scores(head,Vx,"test_ood"),
                                "vid":scores(head,Vx,"val_id"),"vood":scores(head,Vx,"val_ood")}

    def disagreement(pref):
        n_id=len(vs[fmkey]["id" if pref=="t" else "vid"]); n_od=len(vs[fmkey]["ood" if pref=="t" else "vood"])
        rng=np.random.default_rng(0)
        m=min(a.max_pairs,n_id*n_od)
        ii=rng.integers(0,n_id,m); oo=rng.integers(0,n_od,m)
        signs=[]
        for blk in [fmkey,"mo","de"]:
            si=vs[blk]["id" if pref=="t" else "vid"]; so=vs[blk]["ood" if pref=="t" else "vood"]
            signs.append(np.sign(so[oo]-si[ii]))
        d=1-np.abs(np.mean(signs,axis=0))
        return ii,oo,d

    vi,vo,vd=disagreement("v")
    q1,q2=np.quantile(vd,[1/3,2/3])          # cut-points from VALIDATION pairs
    ti,to,td=disagreement("t")
    buckets={"low":td<=q1,"mid":(td>q1)&(td<=q2),"high":td>q2}
    print(f"[{a.cell}|{a.backbone}] disagreement cutpoints (val): {q1:.3f}/{q2:.3f}")
    print(f"{'bucket':8}{'n_pairs':>10}{'SV acc':>10}{'MV acc':>10}{'MV-SV':>10}{'RPO-BCE(MV)':>11}")
    res={}
    for bn,mask in buckets.items():
        if mask.sum()==0: continue
        accs={}
        for name in ["MV-RPO","MV-BOH","SV-RPO","SV-BOH"]:
            si=fin[name]["id"][ti[mask]]; so=fin[name]["ood"][to[mask]]
            accs[name]=float((so>si).mean())          # correct order: OOD scored higher
        res[bn]={"n":int(mask.sum()),**accs,
                 "mv_minus_sv_bce":accs["MV-BOH"]-accs["SV-BOH"],
                 "mv_minus_sv_rpo":accs["MV-RPO"]-accs["SV-RPO"],
                 "rpo_minus_bce_mv":accs["MV-RPO"]-accs["MV-BOH"]}
        print(f"{bn:8}{mask.sum():>10}{accs['SV-BOH']:>10.4f}{accs['MV-BOH']:>10.4f}"
              f"{accs['MV-BOH']-accs['SV-BOH']:>+10.4f}{accs['MV-RPO']-accs['MV-BOH']:>+11.4f}")
    json.dump(res,open(f"repro/phase1_{a.cell}_{a.backbone}.json","w"),indent=2)
    print(f"PHASE1_DONE {a.cell} {a.backbone}",flush=True)

if __name__=="__main__": main()
