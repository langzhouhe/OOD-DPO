#!/usr/bin/env python3
"""Canonical export for the MolRoute paper numbers. LOG-ONLY: no algorithm or parameter
is changed, and every seed matches molroute_cf.py, so the summary values reproduce the
already-committed ones exactly. What this adds is the record-keeping that was missing.

Fixes carried out here:
  * per-mechanism / per-batch records were never written (the dict was initialised and
    left empty), so selection-frequency, mechanism figures and mechanism bootstrap were
    impossible for the Uni-Mol and CF runs
  * ONE canonical MolRoute-T. The first molroute.py run and molroute_cf.py differ by up to
    ~0.008 purely through ID resampling; the paper uses the CF-matched version throughout,
    so T and CF are computed on identical batches
  * the oracle was over the FULL 10-detector bank while MolRoute may only pick among the
    four routing members. Both are now reported: `oracle_routing` is the attainable
    ceiling for this method, `oracle_full` is the looser bank-wide one
  * additional fixed baselines, all read off the same calibrated scores at no extra cost:
    a mean-ensemble over the routing bank and over the full bank

Also reported: the GBDR learned router, trained on the same ood_val historical batches
with the recipe already frozen in decisive.py (no tuning), so Table 1 can compare against
a learned router on the official test set rather than only against HistBest.

Usage: python molroute_export.py --cell ec50_assay --backbone minimol
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, argparse, warnings, logging
import numpy as np, torch
torch.set_num_threads(4)
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import revision_run as RR
import batchsel_multi as BM
import decisive as DC
import sanity as SN

PREV = [0.10, 0.25, 0.50]
REPEATS = 5


def batch_pack(B, names, Do, Di):
    """Calibrated score matrix (K x n), label-free q, true per-detector AUROC, features."""
    C = np.stack([DC.mid_cdf(B["ecdf_ref"][n], np.r_[B["z"][n](Di), B["z"][n](Do)])
                  for n in names])
    y = np.r_[np.zeros(len(next(iter(Di.values())))), np.ones(len(next(iter(Do.values()))))]
    au = np.array([roc_auc_score(y, C[j]) for j in range(len(names))])
    feat = np.concatenate([BM.batch_stats(np.r_[B["z"][n](Di), B["z"][n](Do)], B["zCAL"][n])
                           for n in names])
    return C, y, au, feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--backbone", default="minimol", choices=["minimol", "unimol"])
    a = ap.parse_args()
    B, names, bank, bidx, hb, guard_on = SN.frozen_setup(a.cell, a.backbone)
    K = len(names); bidx = np.array(bidx)
    keep, cov = SN.common_test_smiles(a.cell)
    Xt, mech_t = SN.test_mols(a.cell, B, keep)
    IDf = B["IDf"]; npool = len(next(iter(IDf.values())))
    uniq = np.unique(mech_t)
    print(f"[{a.cell}/{a.backbone}] bank {bank} | hist_best {names[hb]} | "
          f"guard {'ROUTE' if guard_on else 'FALLBACK'} | {len(uniq)} mechanisms | "
          f"coverage {cov}", flush=True)

    out = {"cell": a.cell, "backbone": a.backbone, "detectors": names, "routing_bank": bank,
           "historical_best": names[hb], "guard_route": guard_on, "coverage": cov,
           "n_test_mech": int(len(uniq)), "summary": {}, "per_mech": {}, "select_freq": {}}

    for prev in PREV:
        # ---- historical batches (ood_val) for the GBDR router, frozen recipe ----
        Fh, Ah = [], []
        for rep in range(REPEATS):
            rg = np.random.default_rng(100 + rep)
            f, _, A = DC.collect(B, B["Xv"], B["Sm"], B["m_rt"], B["IDr"], rg, prev)
            if len(A): Fh.append(np.nan_to_num(f)); Ah.append(np.nan_to_num(A, nan=.5))
        Fh, Ah = np.vstack(Fh), np.vstack(Ah)
        nf = Fh.shape[1] // K
        Rtr = np.vstack([Fh[:, i * nf:(i + 1) * nf] for i in range(K)])
        ytr = np.concatenate([Ah[:, i] for i in range(K)])
        reg = GradientBoostingRegressor(random_state=0, n_estimators=300,
                                        max_depth=3).fit(Rtr, ytr)

        ARMS = ["hist_best", "molroute_T", "molroute_CF", "ens_bank", "ens_full",
                "gbdr", "oracle_routing", "oracle_full"]
        acc = {k: [] for k in ARMS}
        per = {k: np.zeros(len(uniq)) for k in ARMS}
        freq_T = np.zeros(K); freq_CF = np.zeros(K); nrec = 0
        q0, pick0 = None, None
        for rep in range(REPEATS):
            rg = np.random.default_rng(500 + rep)     # identical to molroute_cf.py
            vals = {k: [] for k in ARMS}
            qm, pk = [], []
            for mi, m in enumerate(uniq):
                rows = np.where(mech_t == m)[0]
                n_id = max(1, int(round(len(rows) * (1 - prev) / prev)))
                if n_id > npool: continue
                isel = rg.choice(npool, n_id, replace=False)
                Do = {v: Xt[v][rows] for v in Xt}; Di = {v: IDf[v][isel] for v in IDf}
                C, y, au, feat = batch_pack(B, names, Do, Di)
                q = C.mean(1)

                v = {}
                v["hist_best"] = au[hb]
                v["oracle_routing"] = au[bidx].max()
                v["oracle_full"] = au.max()
                v["ens_bank"] = roc_auc_score(y, C[bidx].mean(0))
                v["ens_full"] = roc_auc_score(y, C.mean(0))
                dT = bidx[int(np.argmax(q[bidx]))] if guard_on else hb
                v["molroute_T"] = au[dT]
                pg = np.array([reg.predict(feat[i * nf:(i + 1) * nf].reshape(1, -1))[0]
                               for i in range(K)])
                dG = bidx[int(np.argmax(pg[bidx]))] if guard_on else hb
                v["gbdr"] = au[dG]
                # cross-fitting
                idx = rg.permutation(len(y)); h = len(idx) // 2
                merged = np.empty(len(y)); cfd = []
                for src, dst in ((idx[:h], idx[h:]), (idx[h:], idx[:h])):
                    d = bidx[int(np.argmax(C[:, src].mean(1)[bidx]))] if guard_on else hb
                    merged[dst] = C[d, dst]; cfd.append(d)
                v["molroute_CF"] = roc_auc_score(y, merged)

                for k in ARMS: vals[k].append(v[k]); per[k][mi] += v[k] / REPEATS
                freq_T[dT] += 1
                for d in cfd: freq_CF[d] += 1
                nrec += 1
                if rep == 0:
                    qm.append([round(float(x), 5) for x in q])
                    pk.append({"mech": int(m), "T": names[dT],
                               "CF": [names[c] for c in cfd], "gbdr": names[dG],
                               "auroc": [round(float(x), 5) for x in au]})
            for k in ARMS: acc[k].append(float(np.mean(vals[k])))
            if rep == 0: q0, pick0 = qm, pk

        S = {k: float(np.mean(acc[k])) for k in ARMS}
        S.update({f"sd_{k}": float(np.std(acc[k])) for k in ARMS})
        for k in ARMS:
            if k != "hist_best": S[f"delta_{k}"] = S[k] - S["hist_best"]
        out["summary"][str(prev)] = S
        out["per_mech"][str(prev)] = {k: [round(float(x), 5) for x in per[k]] for k in ARMS}
        out["select_freq"][str(prev)] = {
            "T": {names[j]: round(float(freq_T[j] / max(nrec, 1)), 4) for j in range(K) if freq_T[j]},
            "CF": {names[j]: round(float(freq_CF[j] / max(2 * nrec, 1)), 4) for j in range(K) if freq_CF[j]}}
        out.setdefault("rep0", {})[str(prev)] = {"q": q0, "picks": pick0}
        print(f"[{a.cell}/{a.backbone}] prev {prev:.0%}  hist {S['hist_best']:.4f} | "
              f"T {S['molroute_T']:.4f} ({S['delta_molroute_T']:+.4f}) | "
              f"CF {S['molroute_CF']:.4f} ({S['delta_molroute_CF']:+.4f}) | "
              f"ens {S['ens_bank']:.4f} | gbdr {S['gbdr']:.4f} | "
              f"orc-route {S['oracle_routing']:.4f} orc-full {S['oracle_full']:.4f}", flush=True)

    sfx = "" if a.backbone == "minimol" else f"_{a.backbone}"
    json.dump(out, open(f"repro/export_{a.cell}{sfx}.json", "w"))
    print(f"EXPORT_DONE {a.cell} {a.backbone}", flush=True)


if __name__ == "__main__":
    main()
