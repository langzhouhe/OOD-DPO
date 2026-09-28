#!/usr/bin/env python3
"""Matched-OE arena with FPR95 as the primary endpoint.

Answers the AC's Q1/Q3 head-on: every arm gets the SAME frozen features, the SAME
head, the SAME ID/OOD molecules, the SAME validation protocol, and -- critically --
the SAME NUMBER OF TUNING TRIALS (9). Protocol elements are imported from
revision_run so they cannot drift from E17.

Why FPR95 and not AUROC: for full AUROC, pairwise logistic and balanced BCE share the
Bayes-optimal ranking (revision doc §3), so no separation is possible and E2/E6/E17/E19
duly found none. A REGION-RESTRICTED objective is different: under a restricted
hypothesis class the minimiser of the pairwise loss restricted to the FPR-relevant
corner is NOT the minimiser of pointwise log-loss. That is the only place a pairwise
formulation can win, so that is what this measures.

Arms (all outputs: higher score = more OOD):
  bce        balanced BCE                              <- the AC's requested control
  rpo        pairwise logistic, resampled pairs         <- the paper's objective
  pauc       pairwise logistic restricted to the top-a fraction of ID by score x the
             bottom-a fraction of OOD by score, i.e. exactly the pairs that set FPR95
  focal_bce  balanced focal BCE                        <- KILLER CONTROL: pointwise
             reweighting toward hard examples. If this closes the gap, pauc adds nothing.
  hard_bce   balanced BCE on the top-q fraction by per-sample loss within each class
                                                       <- second killer control
  energy_oe  energy-bounded OE (Liu 2020) = the OE version of the paper's own Energy
             baseline, which beat RPO in E3 at shared default hyperparameters

Equal tuning budget: 2-parameter arms (bce, rpo) search the full 3x3 (gamma, lr) grid.
3-parameter arms search a PRE-REGISTERED 9-point subset: 3 values of their extra knob x
3 a-priori (gamma, lr) points on the grid diagonal. Having an extra knob therefore costs
coverage of the shared knobs, which is what "equal budget" has to mean. A secondary
FULL-grid (27-trial) result is also recorded, clearly labelled, so the effect of the
budget constraint itself is visible rather than hidden.

Both selection criteria are recorded: hyperparameters chosen by validation AUROC and by
validation FPR95. Checkpoint selection stays on validation AUROC for every arm (an E17
protocol element, applied identically), so only the hp criterion varies.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, argparse, warnings, logging, itertools
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import revision_run as RR

ARMS = ["bce", "rpo", "pauc", "focal_bce", "hard_bce", "energy_oe"]
EXTRA = {"pauc": [0.05, 0.1, 0.25], "focal_bce": [1.0, 2.0, 5.0],
         "hard_bce": [0.25, 0.5, 0.75], "energy_oe": [0.5, 1.0, 2.0], "bce": [None], "rpo": [None]}
# a-priori diagonal of the shared (gamma, lr) grid, fixed before seeing any result
DIAG = [(0.001, 1e-3), (0.01, 3e-4), (0.1, 1e-4)]
FULLGRID = [(g, l) for g in RR.GAMMAS for l in RR.LRS]


def loss_of(arm, Eid, Eood, extra, g):
    """Every arm is a function of the same two score vectors; nothing else differs."""
    if arm == "bce":
        return 0.5 * F.binary_cross_entropy_with_logits(Eid, torch.zeros(len(Eid))) + \
               0.5 * F.binary_cross_entropy_with_logits(Eood, torch.ones(len(Eood)))
    if arm == "rpo":
        pair = torch.randint(len(Eid), (len(Eood),), generator=g)
        return F.softplus(-RR.BETA * (Eood - Eid[pair])).mean()
    if arm == "pauc":
        # the FPR95 corner: ID scored too HIGH, OOD scored too LOW
        ki = max(1, int(round(extra * len(Eid)))); ko = max(1, int(round(extra * len(Eood))))
        hi = torch.topk(Eid, ki).values                     # worst ID (highest scores)
        lo = torch.topk(-Eood, ko).values.neg()             # worst OOD (lowest scores)
        return F.softplus(-RR.BETA * (lo.unsqueeze(0) - hi.unsqueeze(1))).mean()
    if arm == "focal_bce":
        def foc(E, y):
            p = torch.sigmoid(E); pt = p * y + (1 - p) * (1 - y)
            return ((1 - pt).clamp(min=1e-6) ** extra *
                    F.binary_cross_entropy_with_logits(E, y, reduction="none")).mean()
        return 0.5 * foc(Eid, torch.zeros(len(Eid))) + 0.5 * foc(Eood, torch.ones(len(Eood)))
    if arm == "hard_bce":
        def hard(E, y):
            l = F.binary_cross_entropy_with_logits(E, y, reduction="none")
            return torch.topk(l, max(1, int(round(extra * len(l))))).values.mean()
        return 0.5 * hard(Eid, torch.zeros(len(Eid))) + 0.5 * hard(Eood, torch.ones(len(Eood)))
    if arm == "energy_oe":
        return F.relu(Eid - (-extra)).pow(2).mean() + F.relu(extra - Eood).pow(2).mean()
    raise ValueError(arm)


def train(V, arm, gamma, lr, extra, seed, epochs=120, n_id=1500, n_ood=2000):
    """Byte-for-byte the E17 training loop except for the loss term."""
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    iid = torch.randperm(len(V["train_id"][0]), generator=g)[:n_id]
    iod = torch.randperm(len(V["train_ood"][0]), generator=g)[:n_ood]
    Xid = [b[iid] for b in V["train_id"]]; Xood = [b[iod] for b in V["train_ood"]]
    head = RR.Head([b.shape[1] for b in V["train_id"]])
    opt = torch.optim.AdamW(head.parameters(), lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9)
    best, bs = -1, None
    for ep in range(epochs):
        head.train(); opt.zero_grad()
        Eid, Eood = head(Xid), head(Xood)
        core = loss_of(arm, Eid, Eood, extra, g)
        (core + gamma * (Eid.pow(2).mean() + Eood.pow(2).mean())).backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
        if (ep + 1) % 15 == 0:
            head.eval()
            with torch.no_grad():
                si, so = head(V["val_id"]).numpy(), head(V["val_ood"]).numpy()
            v = roc_auc_score(np.r_[np.zeros(len(si)), np.ones(len(so))], np.r_[si, so])
            if v > best: best, bs = v, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval()
    with torch.no_grad():
        vau, _, vfp = RR.mets(head(V["val_id"]).numpy(), head(V["val_ood"]).numpy())
        te = RR.mets(head(V["test_id"]).numpy(), head(V["test_ood"]).numpy())
    return (vau, vfp), te


def run_arm(V, arm, grid_mode):
    extras = EXTRA[arm]
    if arm in ("bce", "rpo"):
        trials = [(g, l, None) for g, l in FULLGRID]
    elif grid_mode == "equal":
        trials = [(g, l, e) for e in extras for g, l in DIAG]      # 9, pre-registered
    else:
        trials = [(g, l, e) for e in extras for g, l in FULLGRID]  # 27, secondary
    scored = []
    for gm, lr, ex in trials:
        vs = [train(V, arm, gm, lr, ex, s)[0] for s in RR.SEL]
        scored.append(((gm, lr, ex), float(np.mean([v[0] for v in vs])),
                                     float(np.mean([v[1] for v in vs]))))
    out, cache = {}, {}
    for crit, key, better in (("auroc", 1, max), ("fpr95", 2, min)):
        hp = better(scored, key=lambda t: t[key])[0]
        if hp not in cache:
            cache[hp] = [train(V, arm, hp[0], hp[1], hp[2], s)[1] for s in RR.FIN]
        R = cache[hp]
        out[f"sel_{crit}"] = {"hp": {"gamma": hp[0], "lr": hp[1], "extra": hp[2]},
                              "auroc": [x[0] for x in R], "aupr": [x[1] for x in R],
                              "fpr95": [x[2] for x in R]}
    out["n_trials"] = len(trials)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--view", default="minimol", choices=list(RR.VIEWDEF))
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--grid", default="equal", choices=["equal", "full"])
    a = ap.parse_args()

    data, info = RR.load(a.cell, a.view.startswith("unimol"))
    V = RR.blocks(data, a.view)
    out = {"cell": a.cell, "view": a.view, "grid": a.grid, "domain_split": info, "arms": {}}
    for arm in a.arms:
        r = run_arm(V, arm, a.grid)
        out["arms"][arm] = r
        for crit in ("auroc", "fpr95"):
            d = r[f"sel_{crit}"]
            print(f"[{a.cell}/{a.view}/{a.grid}] {arm:10} sel-by-{crit:5} "
                  f"AUROC={np.mean(d['auroc']):.4f} FPR95={np.mean(d['fpr95']):.4f} "
                  f"hp={d['hp']}", flush=True)
    path = f"repro/arena_{a.cell}_{a.view}_{a.grid}.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"ARENA_DONE {path}", flush=True)


if __name__ == "__main__":
    main()
