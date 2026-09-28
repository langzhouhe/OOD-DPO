#!/usr/bin/env python3
"""Two sanity checks for MolRoute. Bank, selector and guard are the frozen ones -- nothing
here may change them, and no threshold may be tuned on these results.

1) PURE-ID BATCHES (pi = 0). AUROC is undefined, so report false-positive rates at
   thresholds calibrated ONLY on ID_cal:
     - FPR@5% / FPR@1% per detector
     - MolRoute's FPR increment over historical_best
     - same-batch: select and score on the same molecules (the deployed behaviour)
     - split-half: select on one half, score the other (breaks the selection/scoring
       coupling, isolating selection-induced false-positive inflation)
   The question is how much inflation selection causes, NOT whether the guard fires.

2) MULTI-MECHANISM BATCHES. Total OOD held at 8 so batch size is not a confound:
   1 mechanism (8), 2 mechanisms (4+4), 4 mechanisms (2+2+2+2), grouped with a fixed seed
   from the same official ood_test mechanisms. Requirement is only that the gain over
   historical_best stays clearly positive.

Usage: python sanity.py --cell ec50_assay
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, argparse, warnings, logging
import numpy as np, torch
torch.set_num_threads(4)
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import revision_run as RR
import batchsel_multi as BM
import decisive as DC
import molroute as MRT

MR = Path("cache/molroute")
PREV = [0.10, 0.25, 0.50]
N_PURE = 400
GROUP_SEED = 4242
REPEATS = 5


def common_test_smiles(cell, backbones=("minimol", "unimol")):
    """ood_test molecules present in EVERY backbone cache. Both backbones are then scored
    on exactly the same molecules, so a coverage gap in one cannot shift the comparison."""
    b = BM.BASE[cell]
    rec = json.load(open(MR / f"{b}_molroute_test.json"))
    want = {s for ss in rec["catalog"].values() for s in ss}
    for bb in backbones:
        f = BM.CACHE / f"{b}_{bb}_features.pkl"
        if not f.exists(): return want, {bb: 0.0}
        want = want & set(BM.pickle.load(open(f, "rb"))["features"])
    cov = {}
    full = {s for ss in rec["catalog"].values() for s in ss}
    for bb in backbones:
        f = BM.CACHE / f"{b}_{bb}_features.pkl"
        cov[bb] = len(full & set(BM.pickle.load(open(f, "rb"))["features"])) / len(full)
    cov["common"] = len(want) / len(full)
    return want, cov


def frozen_setup(cell, backbone="minimol"):
    """Rebuild exactly what molroute.py used: bank, historical_best, guard.
    historical_best and the guard are re-derived from THIS backbone's ood_val -- detector
    identity is never inherited across backbones."""
    B = DC.build(cell, 21, backbone)
    names = B["names"]
    bank = [n for n in MRT.ROUTING_BANK if n in names]
    bidx = [names.index(n) for n in bank]
    Qh, Ah = [], []
    for prev in PREV:
        for rep in range(REPEATS):
            rg = np.random.default_rng(100 + rep)
            _, q, A = DC.collect(B, B["Xv"], B["Sm"], B["m_rt"], B["IDr"], rg, prev)
            if len(A): Qh.append(q); Ah.append(np.nan_to_num(A, nan=.5))
    Qh, Ah = np.vstack(Qh), np.vstack(Ah)
    hb = int(np.argmax(Ah.mean(0)))
    routed = Ah[np.arange(len(Ah)), np.array(bidx)[Qh[:, bidx].argmax(1)]].mean()
    guard_on = bool(routed > Ah[:, hb].mean())
    return B, names, bank, bidx, hb, guard_on


def test_mols(cell, B, keep=None):
    """keep: restrict to a molecule whitelist (the cross-backbone intersection)."""
    b = BM.BASE[cell]
    rec = json.load(open(MR / f"{b}_molroute_test.json"))
    if keep is None:
        keep = set(BM.pickle.load(
            open(BM.CACHE / f"{b}_minimol_features.pkl", "rb"))["features"])
    doms = sorted(rec["catalog"])
    smis, mech = [], []
    for j, dm in enumerate(doms):
        v = [s for s in rec["catalog"][dm] if s in keep]
        smis += v; mech += [j] * len(v)
    return B["project"](smis), np.asarray(mech)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    a = ap.parse_args()
    B, names, bank, bidx, hb, guard_on = frozen_setup(a.cell)
    K = len(names)
    print(f"[{a.cell}] frozen: bank {bank} | historical_best {names[hb]} | "
          f"guard {'ROUTE' if guard_on else 'FALLBACK'}", flush=True)

    # ---- thresholds from ID_cal ONLY ----
    tau = {}
    for al in (0.05, 0.01):
        tau[al] = np.array([np.quantile(B["zCAL"][n], 1 - al) for n in names])

    out = {"cell": a.cell, "detectors": names, "routing_bank": bank,
           "historical_best": names[hb], "guard_route": guard_on, "pure_id": {}, "mixture": {}}

    # ================= 1. pure-ID batches =================
    IDf = B["IDf"]; npool = len(next(iter(IDf.values())))
    zf = {n: B["z"][n](IDf) for n in names}          # score ID_test once per detector
    for prev in PREV:
        size = int(round(8 / prev))                  # same batch size as the pi>0 runs
        rows = {"per_det_fpr5": np.zeros(K), "per_det_fpr1": np.zeros(K)}
        same5, same1, half5, half1, hbf5, hbf1 = [], [], [], [], [], []
        for rep in range(REPEATS):
            rg = np.random.default_rng(700 + rep)
            for _ in range(N_PURE // REPEATS):
                sel = rg.choice(npool, size, replace=False)
                Z = np.stack([zf[n][sel] for n in names])            # K x size
                q = np.array([DC.mid_cdf(B["ecdf_ref"][names[j]], Z[j]).mean()
                              for j in range(K)])
                rows["per_det_fpr5"] += (Z > tau[0.05][:, None]).mean(1)
                rows["per_det_fpr1"] += (Z > tau[0.01][:, None]).mean(1)
                d_same = bidx[int(np.argmax(q[bidx]))] if guard_on else hb
                same5.append(float((Z[d_same] > tau[0.05][d_same]).mean()))
                same1.append(float((Z[d_same] > tau[0.01][d_same]).mean()))
                hbf5.append(float((Z[hb] > tau[0.05][hb]).mean()))
                hbf1.append(float((Z[hb] > tau[0.01][hb]).mean()))
                h = size // 2
                qa = np.array([DC.mid_cdf(B["ecdf_ref"][names[j]], Z[j, :h]).mean()
                               for j in range(K)])
                d_half = bidx[int(np.argmax(qa[bidx]))] if guard_on else hb
                half5.append(float((Z[d_half, h:] > tau[0.05][d_half]).mean()))
                half1.append(float((Z[d_half, h:] > tau[0.01][d_half]).mean()))
        nb = len(same5)
        rec = {"batch_size": size, "n_batches": nb,
               "hist_best_fpr5": float(np.mean(hbf5)), "hist_best_fpr1": float(np.mean(hbf1)),
               "molroute_same_fpr5": float(np.mean(same5)), "molroute_same_fpr1": float(np.mean(same1)),
               "molroute_half_fpr5": float(np.mean(half5)), "molroute_half_fpr1": float(np.mean(half1)),
               "per_detector_fpr5": {names[j]: round(float(rows["per_det_fpr5"][j] / nb), 4)
                                     for j in range(K)}}
        rec["inflation_same_5"] = rec["molroute_same_fpr5"] - rec["hist_best_fpr5"]
        rec["inflation_half_5"] = rec["molroute_half_fpr5"] - rec["hist_best_fpr5"]
        rec["inflation_same_1"] = rec["molroute_same_fpr1"] - rec["hist_best_fpr1"]
        out["pure_id"][str(prev)] = rec
        print(f"[{a.cell}] pure-ID batch of {size}: FPR@5% hist_best {rec['hist_best_fpr5']:.4f} | "
              f"MolRoute same-batch {rec['molroute_same_fpr5']:.4f} ({rec['inflation_same_5']:+.4f}) | "
              f"split-half {rec['molroute_half_fpr5']:.4f} ({rec['inflation_half_5']:+.4f})", flush=True)
        print(f"[{a.cell}]                    FPR@1% hist_best {rec['hist_best_fpr1']:.4f} | "
              f"same-batch {rec['molroute_same_fpr1']:.4f} ({rec['inflation_same_1']:+.4f}) | "
              f"split-half {rec['molroute_half_fpr1']:.4f}", flush=True)

    # ================= 2. multi-mechanism batches =================
    Xt, mech_t = test_mols(a.cell, B)
    uniq = np.unique(mech_t)
    for n_mech in (1, 2, 4):
        per = 8 // n_mech
        grp_rng = np.random.default_rng(GROUP_SEED)
        order = grp_rng.permutation(uniq)
        groups = [order[i:i + n_mech] for i in range(0, len(order) - n_mech + 1, n_mech)]
        for prev in PREV:
            hbv, sel, orc = [], [], []
            for rep in range(REPEATS):
                rg = np.random.default_rng(800 + rep)
                A_, Q_ = [], []
                for g in groups:
                    rows = np.concatenate([rg.choice(np.where(mech_t == m)[0], per, replace=False)
                                           for m in g])
                    n_id = max(1, int(round(len(rows) * (1 - prev) / prev)))
                    if n_id > len(next(iter(IDf.values()))): continue
                    isel = rg.choice(len(next(iter(IDf.values()))), n_id, replace=False)
                    Do = {v: Xt[v][rows] for v in Xt}
                    Di = {v: IDf[v][isel] for v in IDf}
                    q, au = [], []
                    for n in names:
                        zo, zi = B["z"][n](Do), B["z"][n](Di)
                        q.append(float(DC.mid_cdf(B["ecdf_ref"][n], np.r_[zi, zo]).mean()))
                        au.append(RR.mets(zi, zo)[0])
                    Q_.append(q); A_.append(au)
                if not A_: continue
                Q_, A_ = np.asarray(Q_), np.nan_to_num(np.asarray(A_), nan=.5)
                nb = len(A_)
                used = (np.array(bidx)[Q_[:, bidx].argmax(1)] if guard_on
                        else np.full(nb, hb))
                hbv.append(A_[:, hb].mean()); sel.append(A_[np.arange(nb), used].mean())
                orc.append(A_.max(1).mean())
            if not hbv: continue
            k = f"{n_mech}mech_prev{prev}"
            out["mixture"][k] = {"n_mech": n_mech, "per_mech": per, "prev": prev,
                                 "hist_best": float(np.mean(hbv)),
                                 "molroute": float(np.mean(sel)),
                                 "delta": float(np.mean(sel) - np.mean(hbv)),
                                 "oracle": float(np.mean(orc)), "n_batches": nb}
            r = out["mixture"][k]
            print(f"[{a.cell}] {n_mech} mech x {per} mol, prev {prev:.0%}: "
                  f"hist_best {r['hist_best']:.4f} -> MolRoute {r['molroute']:.4f} "
                  f"({r['delta']:+.4f})  oracle {r['oracle']:.4f}", flush=True)

    json.dump(out, open(f"repro/sanity_{a.cell}.json", "w"), indent=2)
    print(f"SANITY_DONE {a.cell}", flush=True)


if __name__ == "__main__":
    main()
