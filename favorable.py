#!/usr/bin/env python3
"""Three pre-registered axes on which a pairwise objective could still beat a pointwise one.

Every matched comparison so far used one head size, one auxiliary-OOD pool, and AUROC-based
model selection, and all of them tied.  This script varies the three things that a
mechanism argument says should matter, with the hypotheses fixed BEFORE the run:

  A. HEAD CAPACITY.  RPO's optimum is (1/beta) x BCE's logit plus a constant, so the two
     share a Bayes ranking only when the head can represent it.  Prediction: Delta =
     RPO - BCE is positive at small width and decays to 0 as width grows.  The claim is
     the SHAPE (monotone decay), not any single positive cell.

  B. OUTLIER-EXPOSURE DIFFICULTY.  A pairwise gradient concentrates on pairs that are
     close or mis-ordered; far outliers are easy for both losses.  Auxiliary OOD is split
     into near / medium / far tiers by an ID-ONLY statistic (mean distance to the 20
     nearest ID training molecules in the frozen feature space), plus a random tier of the
     same size as a budget-matched control and the full pool as a reference.  Prediction:
     Delta grows monotonically from far to near.  All tiers are reported, always.

  C. TAIL-ORIENTED SELECTION.  Screening cares about the tail, so hyper-parameters and
     checkpoints are also selected by a validation tail metric instead of validation
     AUROC.  Both selection rules are reported for every objective.

The controls are not optional.  E23 already showed that the FPR95 gain from restricting a
pairwise loss to the tail is reproduced by hard-example-weighted POINTWISE losses, which
means "concentrating on hard examples" is not evidence for pairwise structure.  So every
configuration also runs hard_bce and focal_bce; if they track RPO, the pairwise claim is
dead regardless of how large Delta is.

Equal budget: all six objectives search the SAME 3x3 (gamma x lr) grid = 9 trials.  The
extra knob of hard_bce (q=0.5), focal_bce (gamma_f=2), pauc (alpha=0.25) and
energy_margin (m=-3/+3) is FIXED at its a-priori value and never tuned, so no objective
gets a larger search than another.  Selection uses 5 seeds (not 3) because the arena runs
showed the tail criterion is itself a noisy selector, and evaluation uses 5 disjoint
paired seeds.

Usage: python favorable.py --cell ec50_assay --backbone minimol --width 8 --tier near
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, argparse, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import revision_run as RR
import matched_oe as MO

OBJS = ["bce", "rpo", "hard_bce", "focal_bce", "pauc", "energy_margin"]
TIERS = ["near", "medium", "far", "random", "all"]
SEL = [1, 2, 3, 4, 5]
FIN = [11, 12, 13, 14, 15]
N_ID = 1500
EPOCHS = 120
BETA = 1.0
Q_HARD, GAMMA_FOCAL, ALPHA_PAUC, M_IN, M_OUT = 0.5, 2.0, 0.25, -3.0, 3.0
TIER_FRAC = 0.30
KNN_K = 20


# ------------------------------- metrics -------------------------------
def metrics(s_id, s_ood):
    y = np.r_[np.zeros(len(s_id)), np.ones(len(s_ood))]; s = np.r_[s_id, s_ood]
    au = roc_auc_score(y, s)
    p, r, _ = precision_recall_curve(y, s); ap = auc(r, p)
    # accept 95% of ID -> how much OOD is wrongly accepted (the paper's FPR95)
    fpr_ood = float((s_ood <= np.quantile(s_id, 0.95)).mean())
    # detect 95% of OOD -> how much ID is wrongly rejected
    fpr_id = float((s_id >= np.quantile(s_ood, 0.05)).mean())
    return {"auroc": float(au), "aupr": float(ap),
            "fpr_ood_at95tpr_id": fpr_ood, "fpr_id_at95tpr_ood": fpr_id}


# ------------------------------- head -------------------------------
def make_head(dim, width):
    if width == "full":
        return RR.Head([dim])
    h = int(width)
    return nn.Sequential(nn.Linear(dim, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1))


class Wrap(nn.Module):
    """Uniform interface: RR.Head takes a list of blocks, the lite head takes a tensor."""
    def __init__(s, dim, width):
        super().__init__()
        s.width = width; s.m = make_head(dim, width)
    def forward(s, x):
        return s.m([x]) if s.width == "full" else s.m(x).squeeze(-1)


# ------------------------------- objectives -------------------------------
def loss_of(kind, Eid, Eood, g):
    if kind == "bce":
        return (0.5 * F.binary_cross_entropy_with_logits(Eid, torch.zeros(len(Eid)))
                + 0.5 * F.binary_cross_entropy_with_logits(Eood, torch.ones(len(Eood))))
    if kind == "rpo":
        pair = torch.randint(len(Eid), (len(Eood),), generator=g)
        return F.softplus(-BETA * (Eood - Eid[pair])).mean()
    if kind == "hard_bce":
        li = F.binary_cross_entropy_with_logits(Eid, torch.zeros(len(Eid)), reduction="none")
        lo = F.binary_cross_entropy_with_logits(Eood, torch.ones(len(Eood)), reduction="none")
        ki = max(1, int(Q_HARD * len(li))); ko = max(1, int(Q_HARD * len(lo)))
        return 0.5 * torch.topk(li, ki).values.mean() + 0.5 * torch.topk(lo, ko).values.mean()
    if kind == "focal_bce":
        pi = torch.sigmoid(Eid); po = torch.sigmoid(Eood)
        li = F.binary_cross_entropy_with_logits(Eid, torch.zeros(len(Eid)), reduction="none")
        lo = F.binary_cross_entropy_with_logits(Eood, torch.ones(len(Eood)), reduction="none")
        return (0.5 * (pi.pow(GAMMA_FOCAL) * li).mean()
                + 0.5 * ((1 - po).pow(GAMMA_FOCAL) * lo).mean())
    if kind == "pauc":
        ki = max(1, int(ALPHA_PAUC * len(Eid))); ko = max(1, int(ALPHA_PAUC * len(Eood)))
        hi = torch.topk(Eid, ki).values                 # the ID molecules that score worst
        lo = torch.topk(-Eood, ko).values.neg()         # the OOD molecules that score best
        return F.softplus(-(lo.unsqueeze(0) - hi.unsqueeze(1))).mean()
    if kind == "energy_margin":
        return (F.relu(Eid - M_IN).pow(2).mean() + F.relu(M_OUT - Eood).pow(2).mean())
    raise ValueError(kind)


def train(X, kind, gamma, lr, seed, sel_key):
    """One training run.  Checkpoint is chosen on the validation criterion `sel_key`, so
    tail-oriented selection changes BOTH the hyper-parameter and the checkpoint."""
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    iid = torch.randperm(len(X["train_id"]), generator=g)[:N_ID]
    Xid = X["train_id"][iid]; Xood = X["train_ood"]
    head = Wrap(Xid.shape[1], X["width"])
    opt = torch.optim.AdamW(head.parameters(), lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9)
    lower_better = sel_key != "auroc"
    best, bs = None, None
    for ep in range(EPOCHS):
        head.train(); opt.zero_grad()
        Eid, Eood = head(Xid), head(Xood)
        (loss_of(kind, Eid, Eood, g)
         + gamma * (Eid.pow(2).mean() + Eood.pow(2).mean())).backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
        if (ep + 1) % 15 == 0:
            head.eval()
            with torch.no_grad():
                m = metrics(head(X["val_id"]).numpy(), head(X["val_ood"]).numpy())
            v = -m[sel_key] if lower_better else m[sel_key]
            if best is None or v > best:
                best, bs = v, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval()
    with torch.no_grad():
        val = metrics(head(X["val_id"]).numpy(), head(X["val_ood"]).numpy())
        test = metrics(head(X["test_id"]).numpy(), head(X["test_ood"]).numpy())
    return val, test


def run_objective(X, kind, sel_key):
    lower_better = sel_key != "auroc"
    best = None
    for gm in RR.GAMMAS:
        for lr in RR.LRS:                                  # 9 trials, same grid for all
            v = float(np.mean([train(X, kind, gm, lr, s, sel_key)[0][sel_key] for s in SEL]))
            score = -v if lower_better else v
            if best is None or score > best[2]: best = (gm, lr, score)
    gm, lr, _ = best
    R = [train(X, kind, gm, lr, s, sel_key)[1] for s in FIN]
    return R, {"gamma": gm, "lr": lr}


# ------------------------------- exposure tiers -------------------------------
def tier_index(Xid, Xood, tier, n):
    """Rank auxiliary OOD by mean distance to the KNN_K nearest ID training molecules.
    The statistic uses ID molecules only -- no OOD labels, no test data."""
    d = NearestNeighbors(n_neighbors=min(KNN_K, len(Xid))).fit(Xid).kneighbors(Xood)[0].mean(1)
    o = np.argsort(d)
    if tier == "near":   idx = o[:n]
    elif tier == "far":  idx = o[-n:]
    elif tier == "medium":
        c = len(o) // 2; idx = o[max(0, c - n // 2):max(0, c - n // 2) + n]
    elif tier == "random":
        idx = np.random.RandomState(42).permutation(len(o))[:n]
    else:                idx = np.arange(len(o))
    return idx, float(d[idx].mean()), float(d.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--backbone", default="minimol")
    ap.add_argument("--width", required=True)
    ap.add_argument("--tier", required=True, choices=TIERS)
    a = ap.parse_args()
    D, info = MO.load_cell(a.cell, a.backbone)
    tr_id = D["train_id"][0]; tr_ood = D["train_ood"][0]
    n = int(TIER_FRAC * len(tr_ood))
    idx, d_tier, d_all = tier_index(tr_id, tr_ood, a.tier, n)
    width = a.width if a.width == "full" else int(a.width)

    X = {"train_id": torch.tensor(tr_id), "train_ood": torch.tensor(tr_ood[idx]),
         "val_id": torch.tensor(D["val_id"][0]), "val_ood": torch.tensor(D["val_ood"][0]),
         "test_id": torch.tensor(D["test_id"][0]), "test_ood": torch.tensor(D["test_ood"][0]),
         "width": width}
    npar = sum(p.numel() for p in Wrap(tr_id.shape[1], width).parameters())
    print(f"[{a.cell}/{a.backbone}/h={a.width}/{a.tier}] id={len(tr_id)} "
          f"ood_exposed={len(idx)}/{len(tr_ood)} params={npar:,} "
          f"tier_knn_dist={d_tier:.3f} (pool mean {d_all:.3f}) split={info}", flush=True)

    out = {"cell": a.cell, "backbone": a.backbone, "width": a.width, "tier": a.tier,
           "params": int(npar), "n_exposed": int(len(idx)), "tier_dist": d_tier,
           "pool_dist": d_all, "domain_split": info, "arms": {}}
    for sel_key in ("auroc", "fpr_ood_at95tpr_id"):
        for kind in OBJS:
            R, hp = run_objective(X, kind, sel_key)
            out["arms"][f"{kind}|sel_{sel_key}"] = {
                "hp": hp, **{k: [r[k] for r in R] for k in R[0]}}
            print(f"  sel={sel_key:18} {kind:14} AUROC {np.mean([r['auroc'] for r in R]):.4f} "
                  f"FPRood {np.mean([r['fpr_ood_at95tpr_id'] for r in R]):.3f} "
                  f"FPRid {np.mean([r['fpr_id_at95tpr_ood'] for r in R]):.3f} hp={hp}",
                  flush=True)
        b = np.array(out["arms"][f"bce|sel_{sel_key}"]["auroc"])
        for kind in OBJS:
            if kind == "bce": continue
            d = np.array(out["arms"][f"{kind}|sel_{sel_key}"]["auroc"]) - b
            out["arms"][f"{kind}|sel_{sel_key}"]["delta_vs_bce"] = float(d.mean())
            out["arms"][f"{kind}|sel_{sel_key}"]["pos_seeds"] = int((d > 0).sum())

    Path("repro").mkdir(exist_ok=True)
    json.dump(out, open(f"repro/fav_{a.cell}_{a.backbone}_h{a.width}_{a.tier}.json", "w"),
              indent=2)
    print(f"FAV_DONE {a.cell} {a.backbone} h{a.width} {a.tier}", flush=True)


if __name__ == "__main__":
    main()
