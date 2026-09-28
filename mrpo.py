#!/usr/bin/env python3
"""MRPO kill test: a 2x2 factorial over {pointwise, pairwise} x {pooled, mechanism-robust}.

                     pooled aggregation      mechanism-robust aggregation
    pointwise        bce                     cvar_bce
    pairwise         rpo                     mrpo

Within a row the ONLY difference is how per-mechanism risk is aggregated; within a column
the ONLY difference is pointwise vs pairwise risk. So the 2x2 identifies which factor (if
either) is responsible. This is the design that makes the test falsifiable: if mrpo beats
bce but not cvar_bce, the contribution is group robustness and has nothing to do with
preference/pairwise ranking.

Per-mechanism risk on group g (mechanisms = DrugOOD assay domain_id):
  pairwise   R_g(s) = mean_{i in ID, o in g} softplus(-beta (s(o) - s(i)))
  pointwise  R_g(s) = 0.5 mean_ID BCE(s,0) + 0.5 mean_{o in g} BCE(s,1)
Robust arms optimise CVaR_alpha over the REFERENCE-NORMALISED excess risk
  Delta_g(s) = R_g(s) - R_g(s_ref),   s_ref = the matched Balanced BCE detector,
plus a KL trust region tau * E_x KL(Bern(sigmoid s(x)) || Bern(sigmoid s_ref(x))).

Both pooled arms use ALL PAIRS / all molecules, so pooled-vs-robust differs only in
aggregation (E19 established all-pairs RPO and sampled RPO agree to within +-0.002).

Equal tuning budget: 9 trials each. Pooled arms search the full 3x3 (gamma, lr) grid.
Robust arms search 3 alpha x 3 pre-registered (gamma, lr) diagonal points, with tau fixed
at 0.1 a priori -- an extra knob costs coverage of the shared knobs.

Usage: python mrpo.py --cell ec50_assay --split_seed 7
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, pickle, argparse, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import revision_run as RR

CACHE = Path("cache/ood_dpo_cache"); MR = Path("cache/mrpo")
BASE = {"ec50_assay": "lbap_general_ec50_assay", "ic50_assay": "lbap_general_ic50_assay"}
KEYS = RR.KEYS
BETA = RR.BETA
ALPHAS = [0.1, 0.25, 0.5]
TAU = 0.1
DIAG = [(0.001, 1e-3), (0.01, 3e-4), (0.1, 1e-4)]
FULLGRID = [(g, l) for g in RR.GAMMAS for l in RR.LRS]
SEL = [1, 2, 3]; FIN = [11, 12, 13]
N_ID = 1500
ARMS = ["bce", "rpo", "cvar_bce", "mrpo"]


def load(cell, split_seed):
    b = BASE[cell]
    feats = pickle.load(open(CACHE / f"{b}_minimol_features.pkl", "rb"))["features"]
    rec = json.load(open(MR / f"{b}_mrpo_seed{split_seed}_splits.json"))
    sp, dom = rec["splits"], rec["domains"]
    X, G = {}, {}
    for k in KEYS:
        keep = [i for i, s in enumerate(sp[k]) if s in feats]
        X[k] = torch.tensor(np.stack([feats[sp[k][i]] for i in keep]), dtype=torch.float32)
        if k in dom:
            d = [dom[k][i] for i in keep]
            uniq = {v: j for j, v in enumerate(sorted(set(d)))}
            G[k] = torch.tensor([uniq[v] for v in d], dtype=torch.long)
    mu, sd = X["train_id"].mean(0), X["train_id"].std(0) + 1e-6
    X = {k: (v - mu) / sd for k, v in X.items()}
    return X, G, rec


def group_mean(v, gidx, n_g):
    s = torch.zeros(n_g, dtype=v.dtype).index_add_(0, gidx, v)
    c = torch.zeros(n_g, dtype=v.dtype).index_add_(0, gidx, torch.ones_like(v))
    return s / c.clamp(min=1)


def per_group_risk(kind, Eid, Eood, gidx, n_g):
    """Vector of per-mechanism risks; its plain mean is exactly the pooled objective."""
    if kind == "pairwise":
        # mean over ID for each OOD molecule, then mean within mechanism
        col = F.softplus(-BETA * (Eood.unsqueeze(0) - Eid.unsqueeze(1))).mean(0)
        return group_mean(col, gidx, n_g)
    idl = F.binary_cross_entropy_with_logits(Eid, torch.zeros(len(Eid)), reduction="none").mean()
    ool = F.binary_cross_entropy_with_logits(Eood, torch.ones(len(Eood)), reduction="none")
    return 0.5 * idl + 0.5 * group_mean(ool, gidx, n_g)


def kl_bern(E, Eref):
    p = torch.sigmoid(E).clamp(1e-6, 1 - 1e-6); q = torch.sigmoid(Eref).clamp(1e-6, 1 - 1e-6)
    return (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()).mean()


def train(X, G, arm, gamma, lr, alpha, seed, ref=None, epochs=120):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    iid = torch.randperm(len(X["train_id"]), generator=g)[:N_ID]
    Xid, Xood = X["train_id"][iid], X["train_ood"]
    gidx = G["train_ood"]; n_g = int(gidx.max()) + 1
    kind = "pairwise" if arm in ("rpo", "mrpo") else "pointwise"
    robust = arm in ("cvar_bce", "mrpo")
    head = RR.Head([Xid.shape[1]])
    opt = torch.optim.AdamW(head.parameters(), lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9)
    best, bs = -1, None
    for ep in range(epochs):
        head.train(); opt.zero_grad()
        Eid, Eood = head([Xid]), head([Xood])
        R = per_group_risk(kind, Eid, Eood, gidx, n_g)
        if robust:
            d = R - ref["R"]
            k = max(1, int(np.ceil(alpha * n_g)))
            core = torch.topk(d, k).values.mean() + TAU * (
                kl_bern(Eid, ref["id"][iid]) + kl_bern(Eood, ref["ood"])) * 0.5
        else:
            core = R.mean()
        (core + gamma * (Eid.pow(2).mean() + Eood.pow(2).mean())).backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
        if (ep + 1) % 15 == 0:
            head.eval()
            with torch.no_grad():
                v = RR.mets(head([X["val_id"]]).numpy(), head([X["val_ood"]]).numpy())[0]
            if v > best: best, bs = v, {k2: t.clone() for k2, t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval()
    with torch.no_grad():
        te = RR.mets(head([X["test_id"]]).numpy(), head([X["test_ood"]]).numpy())
    return best, te, head


def make_ref(X, G, hp, seed):
    """Reference risks must be evaluated on exactly the ID subsample that `train` will use
    at this seed, otherwise Delta_g mixes a real difference with a resampling difference."""
    _, _, h = train(X, G, "bce", hp[0], hp[1], None, seed)
    h.eval()
    g = torch.Generator().manual_seed(seed)
    iid = torch.randperm(len(X["train_id"]), generator=g)[:N_ID]
    with torch.no_grad():
        Eid, Eood = h([X["train_id"]]), h([X["train_ood"]])
        gidx = G["train_ood"]; n_g = int(gidx.max()) + 1
        return {"id": Eid, "ood": Eood,
                "R_pairwise": per_group_risk("pairwise", Eid[iid], Eood, gidx, n_g),
                "R_pointwise": per_group_risk("pointwise", Eid[iid], Eood, gidx, n_g)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--split_seed", type=int, default=7)
    a = ap.parse_args()
    X, G, rec = load(a.cell, a.split_seed)
    print(f"[{a.cell}] mechanisms train/val/test = "
          f"{int(G['train_ood'].max())+1}/{int(G['val_ood'].max())+1}/{int(G['test_ood'].max())+1} "
          f"| overlaps {rec['domain_overlap']}", flush=True)
    out = {"cell": a.cell, "split_seed": a.split_seed, "n_domains": rec["n_domains"], "arms": {}}

    # 1. pooled BCE first -- it is both an arm and the reference policy
    sc = [((gm, lr), float(np.mean([train(X, G, "bce", gm, lr, None, s)[0] for s in SEL])))
          for gm, lr in FULLGRID]
    hp_bce = max(sc, key=lambda t: t[1])[0]
    refs = {s: make_ref(X, G, hp_bce, s) for s in SEL + FIN}
    R_bce = [train(X, G, "bce", hp_bce[0], hp_bce[1], None, s)[1] for s in FIN]
    out["arms"]["bce"] = {"hp": {"gamma": hp_bce[0], "lr": hp_bce[1]},
                          "auroc": [x[0] for x in R_bce], "aupr": [x[1] for x in R_bce],
                          "fpr95": [x[2] for x in R_bce]}
    print(f"[{a.cell}] bce      AUROC={np.mean([x[0] for x in R_bce]):.4f} "
          f"FPR95={np.mean([x[2] for x in R_bce]):.4f} hp={hp_bce}", flush=True)

    for arm in ["rpo", "cvar_bce", "mrpo"]:
        robust = arm in ("cvar_bce", "mrpo")
        kind = "R_pairwise" if arm in ("rpo", "mrpo") else "R_pointwise"
        trials = ([(gm, lr, al) for al in ALPHAS for gm, lr in DIAG] if robust
                  else [(gm, lr, None) for gm, lr in FULLGRID])
        scored = []
        for gm, lr, al in trials:
            vs = [train(X, G, arm, gm, lr, al, s,
                        ref={"R": refs[s][kind], "id": refs[s]["id"], "ood": refs[s]["ood"]}
                        if robust else None)[0] for s in SEL]
            scored.append(((gm, lr, al), float(np.mean(vs))))
        hp = max(scored, key=lambda t: t[1])[0]
        R = [train(X, G, arm, hp[0], hp[1], hp[2], s,
                   ref={"R": refs[s][kind], "id": refs[s]["id"], "ood": refs[s]["ood"]}
                   if robust else None)[1] for s in FIN]
        out["arms"][arm] = {"hp": {"gamma": hp[0], "lr": hp[1], "alpha": hp[2], "tau": TAU if robust else None},
                            "auroc": [x[0] for x in R], "aupr": [x[1] for x in R],
                            "fpr95": [x[2] for x in R]}
        print(f"[{a.cell}] {arm:9} AUROC={np.mean([x[0] for x in R]):.4f} "
              f"FPR95={np.mean([x[2] for x in R]):.4f} hp={hp}", flush=True)

    json.dump(out, open(f"repro/mrpo_{a.cell}_s{a.split_seed}.json", "w"), indent=2)
    print(f"MRPO_DONE {a.cell}", flush=True)


if __name__ == "__main__":
    main()
