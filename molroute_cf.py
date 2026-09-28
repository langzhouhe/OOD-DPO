#!/usr/bin/env python3
"""MolRoute-CF (2-fold cross-fitting) plus the ID_cal held-out FPR diagnostic.

MolRoute (transductive)
    select one detector using the WHOLE unlabelled batch, score the whole batch with it.
    Maximises batch ranking performance; the threshold calibration inherits a selection
    bias because the same molecules chose the detector and are scored by it.

MolRoute-CF (cross-fitted)
    split the batch in two: choose a detector on half A and score half B with it, choose
    on half B and score half A. Every molecule still gets a score -- unlike simply
    discarding half -- but no molecule is scored by a detector its own value helped pick.
    The two halves may end up on different detectors, so scores are merged on the common
    ID-ECDF scale F_0d(s_d(x)), which is uniform on ID for every d.

ID_cal held-out FPR
    ID_cal is split in two: thresholds from one half, FPR measured on the other. That is
    a same-pool reference, so
        calibration drift  ~  FPR_hist(ID_test) - FPR_hist(ID_cal holdout)
        selection inflation ~ FPR_same-batch    - FPR_crossfit
    which separates Ki's 0.0358 (below the nominal 5%) from anything routing did.

Bank, historical_best and guard are the frozen ones. This adds a reported variant; it
does not change the main method.

Usage: python molroute_cf.py --cell ec50_assay
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, argparse, warnings, logging
import numpy as np, torch
torch.set_num_threads(4)
from pathlib import Path
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import revision_run as RR
import batchsel_multi as BM
import decisive as DC
import sanity as SN

MR = Path("cache/molroute")
PREV = [0.10, 0.25, 0.50]
REPEATS = 5
N_PURE = 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--backbone", default="minimol", choices=["minimol", "unimol"])
    a = ap.parse_args()
    B, names, bank, bidx, hb, guard_on = SN.frozen_setup(a.cell, a.backbone)
    K = len(names)
    print(f"[{a.cell}] frozen: bank {bank} | historical_best {names[hb]} | "
          f"guard {'ROUTE' if guard_on else 'FALLBACK'}", flush=True)
    keep, cov = SN.common_test_smiles(a.cell)
    Xt, mech_t = SN.test_mols(a.cell, B, keep)
    print(f"[{a.cell}/{a.backbone}] ood_test coverage {cov} | scoring "
          f"{len(keep)} common molecules over {mech_t.max()+1} mechanisms", flush=True)
    IDf = B["IDf"]; npool = len(next(iter(IDf.values())))
    out = {"cell": a.cell, "backbone": a.backbone, "historical_best": names[hb],
           "guard_route": guard_on, "routing_bank": bank, "coverage": cov,
           "n_common_molecules": len(keep), "auroc": {}, "calibration": {}, "per_mech": {}}

    # ---------------- MolRoute vs MolRoute-CF on official ood_test ----------------
    for prev in PREV:
        hbv, tr_, cf_, orc = [], [], [], []
        for rep in range(REPEATS):
            rg = np.random.default_rng(500 + rep)
            hb_b, tr_b, cf_b, or_b = [], [], [], []
            for m in np.unique(mech_t):
                rows = np.where(mech_t == m)[0]
                n_id = max(1, int(round(len(rows) * (1 - prev) / prev)))
                if n_id > npool: continue
                isel = rg.choice(npool, n_id, replace=False)
                Do = {v: Xt[v][rows] for v in Xt}; Di = {v: IDf[v][isel] for v in IDf}
                y = np.r_[np.zeros(n_id), np.ones(len(rows))]
                # calibrated score of every detector on the whole batch
                C = np.stack([DC.mid_cdf(B["ecdf_ref"][n],
                                         np.r_[B["z"][n](Di), B["z"][n](Do)]) for n in names])
                q = C.mean(1)
                hb_b.append(roc_auc_score(y, C[hb]))
                or_b.append(max(roc_auc_score(y, C[j]) for j in range(K)))
                d_t = bidx[int(np.argmax(q[bidx]))] if guard_on else hb
                tr_b.append(roc_auc_score(y, C[d_t]))
                # cross-fitting over the batch
                idx = rg.permutation(len(y)); h = len(idx) // 2
                fa, fb = idx[:h], idx[h:]
                merged = np.empty(len(y))
                for src, dst in ((fa, fb), (fb, fa)):
                    qa = C[:, src].mean(1)
                    d = bidx[int(np.argmax(qa[bidx]))] if guard_on else hb
                    merged[dst] = C[d, dst]
                cf_b.append(roc_auc_score(y, merged))
            if hb_b:
                hbv.append(np.mean(hb_b)); tr_.append(np.mean(tr_b))
                cf_.append(np.mean(cf_b)); orc.append(np.mean(or_b))
        r = {"hist_best": float(np.mean(hbv)), "molroute": float(np.mean(tr_)),
             "molroute_cf": float(np.mean(cf_)), "oracle": float(np.mean(orc))}
        r["delta"] = r["molroute"] - r["hist_best"]
        r["delta_cf"] = r["molroute_cf"] - r["hist_best"]
        r["cf_retained"] = r["delta_cf"] / r["delta"] if abs(r["delta"]) > 1e-9 else float("nan")
        out["auroc"][str(prev)] = r
        print(f"[{a.cell}] prev {prev:.0%}  hist_best {r['hist_best']:.4f} | "
              f"MolRoute {r['molroute']:.4f} ({r['delta']:+.4f}) | "
              f"MolRoute-CF {r['molroute_cf']:.4f} ({r['delta_cf']:+.4f}, "
              f"retains {r['cf_retained']*100:.0f}%) | oracle {r['oracle']:.4f}", flush=True)

    # ---------------- calibration diagnostic ----------------
    cal = {n: B["zCAL"][n] for n in names}
    rs = np.random.default_rng(11)
    perm = rs.permutation(len(cal[names[0]]))
    ha, hb_ = perm[:len(perm) // 2], perm[len(perm) // 2:]
    zf = {n: B["z"][n](IDf) for n in names}
    for al in (0.05, 0.01):
        tA = np.quantile(cal[names[hb]][ha], 1 - al)
        fpr_hold = float((cal[names[hb]][hb_] > tA).mean())      # same pool, held out
        fpr_test = float((zf[names[hb]] > tA).mean())            # different ID pool
        # cross-fitted selection on pure-ID batches, for the inflation term
        tau = np.array([np.quantile(cal[n], 1 - al) for n in names])
        same, cfv = [], []
        for rep in range(REPEATS):
            rg = np.random.default_rng(700 + rep)
            for _ in range(N_PURE // REPEATS):
                sel = rg.choice(npool, 16, replace=False)
                Z = np.stack([zf[n][sel] for n in names])
                q = np.array([DC.mid_cdf(B["ecdf_ref"][names[j]], Z[j]).mean() for j in range(K)])
                d = bidx[int(np.argmax(q[bidx]))] if guard_on else hb
                same.append(float((Z[d] > tau[d]).mean()))
                idx = rg.permutation(16); h = 8
                flag = np.empty(16, dtype=bool)
                for src, dst in ((idx[:h], idx[h:]), (idx[h:], idx[:h])):
                    qa = np.array([DC.mid_cdf(B["ecdf_ref"][names[j]], Z[j, src]).mean()
                                   for j in range(K)])
                    dd = bidx[int(np.argmax(qa[bidx]))] if guard_on else hb
                    flag[dst] = Z[dd, dst] > tau[dd]
                cfv.append(float(flag.mean()))
        rec = {"nominal": al, "fpr_cal_holdout": fpr_hold, "fpr_id_test": fpr_test,
               "calibration_drift": fpr_test - fpr_hold,
               "fpr_same_batch": float(np.mean(same)), "fpr_crossfit": float(np.mean(cfv)),
               "selection_inflation": float(np.mean(same) - np.mean(cfv))}
        out["calibration"][str(al)] = rec
        print(f"[{a.cell}] FPR@{al:.0%}: cal-holdout {fpr_hold:.4f} -> ID_test {fpr_test:.4f} "
              f"(drift {rec['calibration_drift']:+.4f}) | same-batch {rec['fpr_same_batch']:.4f} "
              f"vs cross-fit {rec['fpr_crossfit']:.4f} (selection inflation "
              f"{rec['selection_inflation']:+.4f})", flush=True)

    sfx = "" if a.backbone == "minimol" else f"_{a.backbone}"
    json.dump(out, open(f"repro/molroutecf_{a.cell}{sfx}.json", "w"), indent=2)
    print(f"CF_DONE {a.cell}", flush=True)


if __name__ == "__main__":
    main()
