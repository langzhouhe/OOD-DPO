#!/usr/bin/env python3
"""When do zero-OOD (distance/density) detectors fail, and can you tell in advance?

Gate 0 showed mechanism selection carries real signal, but the reference check showed the
whole OE enterprise is dominated by free Mahalanobis on ec50_assay while Mahalanobis
collapses on ic50_assay. Before building an acquisition policy we need to know whether
that split is (a) an endpoint-level property, (b) a per-mechanism property, and (c)
predictable WITHOUT seeing the target OOD -- because if it is not predictable, an
acquisition policy has no way to know when it is even the right tool.

  D-A  per-mechanism distribution of zero-OOD AUROC (test_id vs one mechanism's molecules)
  D-B  target-free predictability: does zero-OOD AUROC measured on the VISIBLE catalog
       predict zero-OOD AUROC on UNSEEN mechanisms? Tested under random splits and under
       embedding-clustered splits, which make catalog and unseen structurally different
       and are therefore the honest stress test.
  D-C  does OE rescue exactly where zero-OOD fails? Per-mechanism correlation between
       Mahalanobis AUROC and a full-catalog OE detector's AUROC. A negative correlation
       is what would define the regime where acquisition is worth doing.

Zero-OOD scorers are fit on id_train only and never see any OOD.

Usage: python zeroood_diag.py --cell ec50_assay
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, argparse, warnings, logging
import numpy as np, torch
torch.set_num_threads(4)
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.cluster import KMeans
from scipy.stats import spearmanr, pearsonr
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import revision_run as RR
import mepoe_gate0 as MG

N_CLUST = 10
N_REP = 200


def auroc(s_id, s_ood):
    return RR.mets(s_id, s_ood)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args()
    d, rec = MG.setup(a.cell, a.seed)
    MG.G.clear(); MG.G.update(d)

    tr, te = d["id_train"].numpy(), d["id_test"].numpy()
    # one pooled mechanism universe with globally unique ids
    X, mech, off = [], [], 0
    for name in ("S", "H1", "H2"):
        X.append(d[name].numpy()); mech.append(d[name + "_mech"] + off)
        off += int(d[name + "_mech"].max()) + 1
    X = np.concatenate(X); mech = np.concatenate(mech)
    src = np.concatenate([np.full(len(d[n]), n) for n in ("S", "H1", "H2")])
    ids = np.unique(mech)
    print(f"[{a.cell}] {len(ids)} mechanisms, {len(X)} molecules, ID train {len(tr)} test {len(te)}",
          flush=True)

    mu = tr.mean(0); P = np.linalg.pinv(np.cov(tr, rowvar=False) + 1e-3 * np.eye(tr.shape[1]))
    md = lambda A: np.einsum("ij,jk,ik->i", A - mu, P, A - mu)
    nn_ = NearestNeighbors(n_neighbors=50).fit(tr)
    kn = lambda A: nn_.kneighbors(A)[0].mean(1)
    lof_ = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(tr)
    lf = lambda A: -lof_.score_samples(A)
    SCORERS = {"mahalanobis": md, "knn50": kn, "lof": lf}
    s_te = {k: f(te) for k, f in SCORERS.items()}
    s_X = {k: f(X) for k, f in SCORERS.items()}

    out = {"cell": a.cell, "n_mech": int(len(ids))}

    # ---- D-A: per-mechanism zero-OOD AUROC ----
    per = {k: np.array([auroc(s_te[k], s_X[k][mech == m]) for m in ids]) for k in SCORERS}
    out["D_A"] = {k: {"mean": float(v.mean()), "sd": float(v.std()),
                      "q10": float(np.quantile(v, .1)), "q50": float(np.median(v)),
                      "q90": float(np.quantile(v, .9)),
                      "frac_below_0.5": float((v < .5).mean())} for k, v in per.items()}
    print(f"[{a.cell}] D-A per-mechanism AUROC", flush=True)
    for k, v in per.items():
        print(f"    {k:12s} mean={v.mean():.4f} sd={v.std():.4f} "
              f"q10/50/90={np.quantile(v,.1):.3f}/{np.median(v):.3f}/{np.quantile(v,.9):.3f} "
              f"frac<0.5={(v<.5).mean():.2f}", flush=True)

    # ---- D-B: is the catalog estimate a valid stand-in for unseen mechanisms? ----
    cent = np.stack([X[mech == m].mean(0) for m in ids])
    lab = KMeans(N_CLUST, n_init=10, random_state=0).fit_predict(cent)
    rng = np.random.default_rng(5)
    res = {}
    for mode in ("random", "clustered"):
        cc, uu = [], []
        for _ in range(N_REP):
            if mode == "random":
                p = rng.permutation(len(ids)); cm = set(ids[p[:len(ids) // 2]])
            else:
                cl = rng.permutation(N_CLUST)[:N_CLUST // 2]
                cm = set(ids[np.isin(lab, cl)])
                if not (0 < len(cm) < len(ids)): continue
            inC = np.isin(mech, list(cm))
            cc.append(auroc(s_te["mahalanobis"], s_X["mahalanobis"][inC]))
            uu.append(auroc(s_te["mahalanobis"], s_X["mahalanobis"][~inC]))
        cc, uu = np.array(cc), np.array(uu)
        res[mode] = {"r_pearson": float(pearsonr(cc, uu).statistic) if cc.std() > 1e-9 else None,
                     "mean_abs_err": float(np.abs(cc - uu).mean()),
                     "q90_abs_err": float(np.quantile(np.abs(cc - uu), .9)),
                     "catalog_mean": float(cc.mean()), "unseen_mean": float(uu.mean()),
                     "n": int(len(cc))}
        print(f"[{a.cell}] D-B {mode:9s} catalog={cc.mean():.4f} unseen={uu.mean():.4f} "
              f"| mean|err|={np.abs(cc-uu).mean():.4f} q90|err|={np.quantile(np.abs(cc-uu),.9):.4f} "
              f"| r={res[mode]['r_pearson']}", flush=True)
    out["D_B"] = res

    # ---- D-C: does OE rescue where Mahalanobis fails? ----
    # full-catalog OE detector, same protocol as Gate 0, trained on S only
    Srows = np.where(src == "S")[0]
    import torch.nn as nn, torch.nn.functional as F
    unseen = np.isin(src, ["H1", "H2"])
    sc_acc = np.zeros(int(unseen.sum())); id_acc = np.zeros(len(te))
    for sd in (1, 2, 3):
        Xid = d["id_train"]; Xood = torch.tensor(X[Srows])
        torch.manual_seed(sd); np.random.seed(sd)
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
        with torch.no_grad():
            id_acc += head([d["id_test"]]).numpy() / 3
            sc_acc += head([torch.tensor(X[unseen])]).numpy() / 3
    um = mech[unseen]; uids = np.unique(um)
    oe_per = np.array([auroc(id_acc, sc_acc[um == m]) for m in uids])
    mh_per = np.array([auroc(s_te["mahalanobis"], s_X["mahalanobis"][mech == m]) for m in uids])
    r_s = float(spearmanr(mh_per, oe_per).statistic)
    lo = mh_per < np.median(mh_per)
    out["D_C"] = {"spearman_maha_vs_oe": r_s, "n_unseen_mech": int(len(uids)),
                  "oe_mean": float(oe_per.mean()), "maha_mean": float(mh_per.mean()),
                  "oe_minus_maha_where_maha_low": float((oe_per - mh_per)[lo].mean()),
                  "oe_minus_maha_where_maha_high": float((oe_per - mh_per)[~lo].mean())}
    print(f"[{a.cell}] D-C over {len(uids)} unseen mechanisms: "
          f"Maha={mh_per.mean():.4f} OE={oe_per.mean():.4f} Spearman(Maha,OE)={r_s:+.3f}", flush=True)
    print(f"    OE-Maha on mechanisms where Maha is LOW  : {(oe_per-mh_per)[lo].mean():+.4f}", flush=True)
    print(f"    OE-Maha on mechanisms where Maha is HIGH : {(oe_per-mh_per)[~lo].mean():+.4f}", flush=True)

    json.dump(out, open(f"repro/zeroood_{a.cell}.json", "w"), indent=2)
    print(f"ZEROOOD_DONE {a.cell}", flush=True)


if __name__ == "__main__":
    main()
