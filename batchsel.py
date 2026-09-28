#!/usr/bin/env python3
"""Gate 0'': label-free batch-level detector selection, on MIXED batches.

Motivation (zeroood_diag): distance-based and exposure-based detectors are nearly
orthogonal per mechanism (Spearman ~0), conditional gaps +-0.2..0.3, per-mechanism oracle
ceiling ~+0.11 over the best single detector. Gate 0' failed to realise it with a
per-molecule stacking router, because the oracle gain is a per-GROUP switch.

!! DEGENERATE SETTING TO AVOID !!  A first version of this script scored batches that were
each an ENTIRE mechanism, i.e. 100% OOD, against an ID reference. Under "this batch is
all OOD and here is my ID data", each detector's AUROC is itself a two-sample statistic
and is therefore directly computable with no labels -- the oracle is available at test
time and selection is trivial. It produced +0.087/+0.125 with 0.89 selector accuracy,
which was an artifact of that circularity, not a method.

This version uses MIXED batches of unknown composition: each batch is m ID molecules
(held out) plus m OOD molecules from one mechanism, and the target is the AUROC WITHIN
the batch. Now the label-free statistics cannot encode the answer: a detector's batch
score distribution being wide/bimodal is genuine evidence that it is separating something,
but it is not a monotone function of the within-batch AUROC.

Arms: maha, oe, best_fixed, heuristic (argmax batch score dispersion), learned
meta-selector trained on historical mechanisms, ORACLE (needs labels).
Gate: learned beats best_fixed by >= +0.030 on BOTH endpoints.

Usage: python batchsel.py --cell ec50_assay
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, argparse, warnings, logging
import numpy as np, torch
torch.set_num_threads(4)
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import ks_2samp, wasserstein_distance, skew, kurtosis
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import revision_run as RR
import mepoe_gate0 as MG
from routing_gate import train_oe, N_FIT, SEEDS

PREV = [0.5, 0.25, 0.10, 0.05]   # OOD prevalence in the batch; 0.5 is the easy case
REPEATS = 5


def batch_stats(zB, zREF):
    """Label-free description of one MIXED batch under one detector. Nothing here uses
    the batch's labels; dispersion/bimodality is the signal that the detector separates."""
    q95 = np.quantile(zREF, 0.95)
    s = np.sort(zB); h = len(s) // 2
    return [zB.mean(), zB.std(), float(skew(zB)) if len(zB) > 2 else 0.0,
            float(kurtosis(zB)) if len(zB) > 3 else 0.0,
            ks_2samp(zB, zREF).statistic, wasserstein_distance(zB, zREF),
            float((zB > q95).mean()),
            float(s[h:].mean() - s[:h].mean()),          # top-half minus bottom-half gap
            float(zB.std() / (zREF.std() + 1e-9))]


def build(cell, seed=21):
    d, _ = MG.setup(cell, seed); MG.G.clear(); MG.G.update(d)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(d["id_train"]))
    id_fit = d["id_train"][idx[:N_FIT]]
    Smech = d["S_mech"]; uS = np.unique(Smech); h = rng.permutation(len(uS))
    m_oe, m_rt = set(uS[h[:len(uS) // 2]]), set(uS[h[len(uS) // 2:]])
    S_oe = d["S"][torch.as_tensor(np.where(np.isin(Smech, list(m_oe)))[0])]

    tr = id_fit.numpy()
    mu = tr.mean(0); P = np.linalg.pinv(np.cov(tr, rowvar=False) + 1e-3 * np.eye(tr.shape[1]))
    md = lambda A: np.einsum("ij,jk,ik->i", A - mu, P, A - mu)
    heads = [train_oe(id_fit, S_oe, s) for s in SEEDS]
    def oe(A):
        t = torch.as_tensor(A) if not torch.is_tensor(A) else A
        with torch.no_grad(): return np.mean([hh([t]).numpy() for hh in heads], 0)
    ref = {"maha": md(tr), "oe": oe(tr)}
    z = {k: (lambda A, k=k: ((md(A) if k == "maha" else oe(A)) - ref[k].mean())
             / (ref[k].std() + 1e-9)) for k in ref}
    zREF = {k: (ref[k] - ref[k].mean()) / (ref[k].std() + 1e-9) for k in ref}
    return d, z, zREF, m_rt


def collect(z, zREF, X, mech, ids, id_pool, rng, prev):
    """One MIXED batch per mechanism: the mechanism's OOD molecules plus enough held-out ID
    molecules to hit the target OOD prevalence. Target = AUROC within the batch.
    Features never see which member is which."""
    F_, Y_, A_ = [], [], []
    for m in np.asarray(ids):
        rows = np.where(mech == m)[0]
        n_ood = len(rows)
        n_id = max(1, int(round(n_ood * (1 - prev) / prev)))
        if n_id > len(id_pool): continue
        idm = id_pool[rng.choice(len(id_pool), n_id, replace=False)]
        feat, au = [], {}
        for k in z:
            zo, zi = z[k](X[rows]), z[k](idm)
            zb = np.r_[zi, zo]
            feat += batch_stats(zb, zREF[k])
            au[k] = RR.mets(zi, zo)[0]
        F_.append(feat); Y_.append(1 if au["oe"] > au["maha"] else 0)
        A_.append([au["maha"], au["oe"]])
    return np.asarray(F_), np.asarray(Y_), np.asarray(A_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args()
    d, z, zREF, m_rt = build(a.cell, a.seed)
    id_pool = d["id_test"].numpy()
    rng = np.random.default_rng(3)

    Xs, ms = d["S"].numpy(), d["S_mech"]
    Xu, mu_, off = [], [], 0
    for n in ("H1", "H2"):
        Xu.append(d[n].numpy()); mu_.append(d[n + "_mech"] + off)
        off += int(d[n + "_mech"].max()) + 1
    Xu = np.concatenate(Xu); mu_ = np.concatenate(mu_)

    out = {"cell": a.cell, "prevalence": {}}
    for prev in PREV:
        R = {k: [] for k in ("maha", "oe", "best_fixed", "heuristic", "learned", "oracle",
                             "acc", "base")}
        for rep in range(REPEATS):
            rg = np.random.default_rng(100 + rep)
            Ftr, Ytr, _ = collect(z, zREF, Xs, ms, sorted(m_rt), id_pool, rg, prev)
            Fte, Yte, Ate = collect(z, zREF, Xu, mu_, np.unique(mu_), id_pool, rg, prev)
            if not len(Ate): continue
            n = len(Ate)
            maha, oe = Ate[:, 0].mean(), Ate[:, 1].mean()
            nf = len(Fte[0]) // 2
            pick_h = (Fte[:, nf + 8] > Fte[:, 8]).astype(int)
            if len(np.unique(Ytr)) > 1:
                clf = GradientBoostingClassifier(random_state=rep, n_estimators=200,
                                                 max_depth=2).fit(Ftr, Ytr)
                pick_l = clf.predict(Fte).astype(int)
            else:
                pick_l = np.full(n, int(Ytr[0]))
            R["maha"].append(maha); R["oe"].append(oe)
            R["best_fixed"].append(max(maha, oe)); R["oracle"].append(Ate.max(1).mean())
            R["heuristic"].append(Ate[np.arange(n), pick_h].mean())
            R["learned"].append(Ate[np.arange(n), pick_l].mean())
            R["acc"].append((pick_l == Yte).mean())
            R["base"].append(max(Yte.mean(), 1 - Yte.mean()))
        M = {k: float(np.mean(v)) for k, v in R.items()}
        S = {k: float(np.std(v)) for k, v in R.items()}
        dl = M["learned"] - M["best_fixed"]
        out["prevalence"][str(prev)] = {**M, "sd_learned": S["learned"],
                                        "delta_learned": dl,
                                        "delta_heuristic": M["heuristic"] - M["best_fixed"],
                                        "oracle_headroom": M["oracle"] - M["best_fixed"],
                                        "pass": bool(dl >= 0.030), "repeats": REPEATS}
        print(f"[{a.cell}] OOD prevalence {prev:.0%}  (batch = 8 OOD + "
              f"{max(1,int(round(8*(1-prev)/prev)))} ID)", flush=True)
        print(f"    maha {M['maha']:.4f} | oe {M['oe']:.4f} | best_fixed {M['best_fixed']:.4f} "
              f"| ORACLE {M['oracle']:.4f} (headroom {M['oracle']-M['best_fixed']:+.4f})", flush=True)
        print(f"    heuristic {M['heuristic']:+.4f}d   learned {M['learned']:.4f} "
              f"({dl:+.4f} +-{S['learned']:.4f}) {'PASS' if dl>=0.030 else 'FAIL'}"
              f"  | acc {M['acc']:.3f} vs base {M['base']:.3f}", flush=True)
    json.dump(out, open(f"repro/batchsel_{a.cell}.json", "w"), indent=2)
    print(f"BATCHSEL_DONE {a.cell}", flush=True)


if __name__ == "__main__":
    main()
