#!/usr/bin/env python3
"""How much does each detector actually contribute to batch-AUC routing?

Two cheap questions that decide how much of the original RPO paper survives:
  1. how often is each detector the one the router picks?
  2. what happens to the routed result if that detector is deleted from the bank?

A detector that is rarely picked AND whose removal costs nothing is a baseline, not a
contribution. A detector that is picked on a distinct subset of mechanisms and whose
removal hurts is a complementary ranking view -- which is a defensible role for RPO even
though it does not beat Balanced BCE on average.

Selection rule is the ANALYTIC one (argmax_d q_d, q_d = mean mid-CDF of the batch under
detector d against an independent ID calibration set). No learning, no target labels, no
knowledge of the batch OOD fraction. decisive.py established this matches the learned
router, so leave-one-out is measured on the rule we would actually ship.

Usage: python bankablate.py --cell ec50_assay --seed 21
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, argparse, warnings, logging
import numpy as np
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import decisive as DC

PREV = [0.5, 0.25, 0.10]
REPEATS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--backbone", default="minimol")
    a = ap.parse_args()
    B = DC.build(a.cell, a.seed, a.backbone)
    names = B["names"]; K = len(names)
    print(f"[{a.cell}/s{a.seed}] bank ({K}): {names}", flush=True)

    out = {"cell": a.cell, "seed": a.seed, "detectors": names, "prevalence": {}}
    for prev in PREV:
        freq = np.zeros(K); full = []; loo = {n: [] for n in names}
        hb_all, orc = [], []
        for rep in range(REPEATS):
            rg = np.random.default_rng(100 + rep)
            _, Qtr, Atr = DC.collect(B, B["Xv"], B["Sm"], B["m_rt"], B["IDr"], rg, prev)
            _, Qte, Ate = DC.collect(B, B["Xu"], B["mu"], np.unique(B["mu"]), B["IDf"], rg, prev)
            if not len(Ate): continue
            Atr = np.nan_to_num(Atr, nan=.5); Ate = np.nan_to_num(Ate, nan=.5)
            n = len(Ate)
            pick = Qte.argmax(1)
            freq += np.bincount(pick, minlength=K)
            full.append(Ate[np.arange(n), pick].mean())
            hb_all.append(Ate[:, int(np.argmax(Atr.mean(0)))].mean())
            orc.append(Ate.max(1).mean())
            for j, nm in enumerate(names):                 # delete detector j, re-route
                keep = [i for i in range(K) if i != j]
                p2 = np.array(keep)[Qte[:, keep].argmax(1)]
                loo[nm].append(Ate[np.arange(n), p2].mean())
        tot = freq.sum()
        rec = {"hist_best": float(np.mean(hb_all)), "routed_full": float(np.mean(full)),
               "oracle_batch": float(np.mean(orc)),
               "select_freq": {nm: round(float(freq[j] / tot), 4) for j, nm in enumerate(names)},
               "loo_drop": {nm: round(float(np.mean(full) - np.mean(loo[nm])), 4) for nm in names}}
        out["prevalence"][str(prev)] = rec
        print(f"[{a.cell}/s{a.seed}] prev {prev:.0%}  hist_best {rec['hist_best']:.4f}  "
              f"routed {rec['routed_full']:.4f}  oracle {rec['oracle_batch']:.4f}", flush=True)
        order = sorted(names, key=lambda n: -rec["loo_drop"][n])
        for nm in order:
            print(f"      {nm:14s} picked {rec['select_freq'][nm]*100:5.1f}%   "
                  f"removing it costs {rec['loo_drop'][nm]:+.4f}", flush=True)
    json.dump(out, open(f"repro/bankablate_{a.cell}_s{a.seed}_{a.backbone}.json", "w"), indent=2)
    print(f"BANKABLATE_DONE {a.cell} s{a.seed}", flush=True)


if __name__ == "__main__":
    main()
