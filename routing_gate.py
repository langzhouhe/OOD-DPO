#!/usr/bin/env python3
"""Gate 0': can the OE/distance complementarity be realised TARGET-FREE?

zeroood_diag showed Mahalanobis and an OE detector are nearly orthogonal per mechanism
(Spearman ~0) with a per-mechanism oracle ceiling ~+0.11 over the best single method. That
ceiling needs labels. This asks whether a combiner learned only from data available at
deployment recovers a useful share of it.

Strict separation, so nothing the combiner sees overlaps what it is tested on:
  id_fit     1500 ID  -> fits Mahalanobis/KNN/LOF and trains the OE detector
  S_oe       100 mech -> the OE detector's auxiliary outliers
  id_router   500 ID  -> combiner training, ID side          (disjoint from id_fit)
  S_router   100 mech -> combiner training, OOD side         (disjoint from S_oe)
  id_test + H1|H2     -> evaluation only, 200 unseen mechanisms

Arms:
  maha / knn / lof   zero-OOD, no auxiliary data at all
  oe                 exposure-based detector alone
  zsum               z-sum of maha and oe, standardised on ID only (no fitting)
  stack              logistic combiner on [z_maha, z_oe]      -- global weighting
  route              MLP on [z_maha,z_knn,z_lof,z_oe] + PCA16(x) -- x-dependent weighting
  ORACLE             per-mechanism max(maha, oe); upper bound, needs labels

Gate: `route` (or `stack`) beats the best single method by >= +0.030 on BOTH endpoints.

Usage: python routing_gate.py --cell ec50_assay
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, argparse, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4)
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import revision_run as RR
import mepoe_gate0 as MG

N_FIT, N_ROUTER = 1500, 500
SEEDS = (1, 2, 3)


def train_oe(Xid, Xood, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    head = RR.Head([Xid.shape[1]])
    opt = torch.optim.AdamW(head.parameters(), MG.LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9)
    for _ in range(MG.EPOCHS):
        head.train(); opt.zero_grad()
        Eid, Eood = head([Xid]), head([Xood])
        (0.5 * F.binary_cross_entropy_with_logits(Eid, torch.zeros(len(Eid)))
         + 0.5 * F.binary_cross_entropy_with_logits(Eood, torch.ones(len(Eood)))
         + MG.GAMMA * (Eid.pow(2).mean() + Eood.pow(2).mean())).backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
    head.eval()
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args()
    d, _ = MG.setup(a.cell, a.seed); MG.G.clear(); MG.G.update(d)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(d["id_train"]))
    id_fit = d["id_train"][idx[:N_FIT]]
    id_rt = d["id_train"][idx[N_FIT:N_FIT + N_ROUTER]]
    id_te = d["id_test"]

    Smech = d["S_mech"]; uS = np.unique(Smech)
    half = rng.permutation(len(uS))
    m_oe, m_rt = set(uS[half[:len(uS) // 2]]), set(uS[half[len(uS) // 2:]])
    S_oe = d["S"][torch.as_tensor(np.where(np.isin(Smech, list(m_oe)))[0])]
    S_rt = d["S"][torch.as_tensor(np.where(np.isin(Smech, list(m_rt)))[0])]

    # unseen evaluation universe
    Xun, mun, off = [], [], 0
    for n in ("H1", "H2"):
        Xun.append(d[n].numpy()); mun.append(d[n + "_mech"] + off)
        off += int(d[n + "_mech"].max()) + 1
    Xun = np.concatenate(Xun); mun = np.concatenate(mun); uids = np.unique(mun)

    # ---- zero-OOD scorers, fit on id_fit only ----
    tr = id_fit.numpy()
    mu = tr.mean(0); P = np.linalg.pinv(np.cov(tr, rowvar=False) + 1e-3 * np.eye(tr.shape[1]))
    nn_ = NearestNeighbors(n_neighbors=50).fit(tr)
    lof_ = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(tr)
    Z = {"maha": lambda A: np.einsum("ij,jk,ik->i", A - mu, P, A - mu),
         "knn": lambda A: nn_.kneighbors(A)[0].mean(1),
         "lof": lambda A: -lof_.score_samples(A)}

    # ---- OE detector, trained on id_fit + S_oe ----
    heads = [train_oe(id_fit, S_oe, s) for s in SEEDS]
    def oe(A):
        t = torch.as_tensor(A) if not torch.is_tensor(A) else A
        with torch.no_grad():
            return np.mean([h([t]).numpy() for h in heads], 0)

    pca = PCA(n_components=16, random_state=0).fit(tr)

    def feats(A):
        An = A.numpy() if torch.is_tensor(A) else A
        cols = [Z[k](An) for k in ("maha", "knn", "lof")] + [oe(An)]
        return np.column_stack(cols), pca.transform(An)

    # standardise every score using the ID ROUTER split only (never OOD, never test)
    f_rt, p_rt = feats(id_rt)
    m_, s_ = f_rt.mean(0), f_rt.std(0) + 1e-9
    norm = lambda F_: (F_ - m_) / s_

    Fr_id, Pr_id = norm(f_rt), p_rt
    f_so, p_so = feats(S_rt); Fr_od, Pr_od = norm(f_so), p_so
    Xtr = np.vstack([Fr_id, Fr_od]); Ptr = np.vstack([Pr_id, Pr_od])
    ytr = np.r_[np.zeros(len(Fr_id)), np.ones(len(Fr_od))]

    stack = LogisticRegression(max_iter=5000, class_weight="balanced").fit(Xtr[:, [0, 3]], ytr)
    route = MLPClassifier((64, 32), max_iter=3000, random_state=0,
                          early_stopping=True).fit(np.hstack([Xtr, Ptr]), ytr)

    f_te, p_te = feats(id_te); Fte, Pte = norm(f_te), p_te
    f_un, p_un = feats(Xun);   Fun, Pun = norm(f_un), p_un

    S_id = {"maha": Fte[:, 0], "knn": Fte[:, 1], "lof": Fte[:, 2], "oe": Fte[:, 3],
            "zsum": Fte[:, 0] + Fte[:, 3],
            "stack": stack.decision_function(Fte[:, [0, 3]]),
            "route": route.predict_proba(np.hstack([Fte, Pte]))[:, 1]}
    S_od = {"maha": Fun[:, 0], "knn": Fun[:, 1], "lof": Fun[:, 2], "oe": Fun[:, 3],
            "zsum": Fun[:, 0] + Fun[:, 3],
            "stack": stack.decision_function(Fun[:, [0, 3]]),
            "route": route.predict_proba(np.hstack([Fun, Pun]))[:, 1]}

    per = {k: np.array([RR.mets(S_id[k], S_od[k][mun == m])[0] for m in uids]) for k in S_id}
    per["ORACLE"] = np.maximum(per["maha"], per["oe"])
    best_single = max(per["maha"].mean(), per["oe"].mean())

    out = {"cell": a.cell, "n_unseen_mech": int(len(uids)),
           "per_mech_mean": {k: float(v.mean()) for k, v in per.items()},
           "best_single": float(best_single),
           "delta_vs_best_single": {k: float(per[k].mean() - best_single) for k in per}}
    print(f"[{a.cell}] {len(uids)} unseen mechanisms | id_fit {N_FIT} id_router {N_ROUTER} "
          f"| S_oe {len(m_oe)} mech, S_router {len(m_rt)} mech", flush=True)
    for k in ("maha", "knn", "lof", "oe", "zsum", "stack", "route", "ORACLE"):
        mark = ""
        if k in ("zsum", "stack", "route"):
            dd = per[k].mean() - best_single
            mark = f"   delta vs best single {dd:+.4f} {'PASS' if dd >= 0.030 else 'fail'}"
        print(f"    {k:8s} {per[k].mean():.4f}{mark}", flush=True)
    json.dump(out, open(f"repro/routing_{a.cell}.json", "w"), indent=2)
    print(f"ROUTING_DONE {a.cell}", flush=True)


if __name__ == "__main__":
    main()
