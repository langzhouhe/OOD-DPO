#!/usr/bin/env python3
"""MolRoute on official ood_test. Protocol frozen in PROTOCOL_MOLROUTE.md.

Development (historical batches, historical_best, guard) uses ood_val mechanisms ONLY.
Evaluation uses official ood_test mechanisms, which share no domain_id with ood_val.

historical_best is one detector per cell, chosen on ood_val batches POOLED over
10/25/50% prevalence. The guard is one mixed-prevalence boolean per cell. Neither is
re-derived per prevalence and neither sees ood_test.

Usage: python molroute.py --cell ec50_assay [--full-bank]
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, argparse, warnings, logging
import numpy as np, torch
torch.set_num_threads(4)
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import revision_run as RR
import mepoe_gate0 as MG
import batchsel_multi as BM
import decisive as DC

MR = Path("cache/molroute")
ROUTING_BANK = ["OE-MV", "Mahalanobis", "LOF", "OE-pAUC"]      # frozen
PREV = [0.10, 0.25, 0.50]
REPEATS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--full-bank", action="store_true")
    a = ap.parse_args()

    B = DC.build(a.cell, 21, "minimol")
    names = B["names"]
    bank = names if a.full_bank else [n for n in ROUTING_BANK if n in names]
    bidx = [names.index(n) for n in bank]
    print(f"[{a.cell}] full bank {names}", flush=True)
    print(f"[{a.cell}] ROUTING bank {bank}", flush=True)

    # ---------- development on ood_val: historical_best + guard, pooled over prevalence ----
    Qh, Ah = [], []
    for prev in PREV:
        for rep in range(REPEATS):
            rg = np.random.default_rng(100 + rep)
            _, q, A = DC.collect(B, B["Xv"], B["Sm"], B["m_rt"], B["IDr"], rg, prev)
            if len(A): Qh.append(q); Ah.append(np.nan_to_num(A, nan=.5))
    Qh, Ah = np.vstack(Qh), np.vstack(Ah)
    hb = int(np.argmax(Ah.mean(0)))                      # one detector, pooled
    routed_dev = Ah[np.arange(len(Ah)), np.array(bidx)[Qh[:, bidx].argmax(1)]].mean()
    guard_on = bool(routed_dev > Ah[:, hb].mean())       # one boolean, pooled
    print(f"[{a.cell}] DEV: historical_best = {names[hb]} ({Ah[:,hb].mean():.4f}) | "
          f"routed_dev {routed_dev:.4f} -> guard {'ROUTE' if guard_on else 'FALLBACK'}",
          flush=True)

    # ---------- official ood_test ----------
    b = BM.BASE[a.cell]
    rec = json.load(open(MR / f"{b}_molroute_test.json"))
    feats = BM.pickle.load(open(BM.CACHE / f"{b}_minimol_features.pkl", "rb"))["features"]
    doms = sorted(rec["catalog"])
    smis, mech_t = [], []
    for j, dm in enumerate(doms):
        v = [s for s in rec["catalog"][dm] if s in feats]
        smis += v; mech_t += [j] * len(v)
    mech_t = np.asarray(mech_t)
    Xt = B["project"](smis)          # identical standardisation to the development views
    print(f"[{a.cell}] ood_test: {len(smis)} molecules over {mech_t.max()+1} mechanisms "
          f"| audit {rec['audit']}", flush=True)

    out = {"cell": a.cell, "detectors": names, "routing_bank": bank,
           "historical_best": names[hb], "guard_route": guard_on,
           "audit": rec["audit"], "n_test_mech": int(mech_t.max()) + 1, "prevalence": {}}
    per_mech = {}
    for prev in PREV:
        hbv, sel, orc, fb = [], [], [], []
        freq = np.zeros(len(names)); rec_rows = []
        for rep in range(REPEATS):
            rg = np.random.default_rng(500 + rep)
            _, Qt, At = DC.collect(B, Xt, mech_t, np.unique(mech_t), B["IDf"], rg, prev)
            if not len(At): continue
            At = np.nan_to_num(At, nan=.5); n = len(At)
            pick = np.array(bidx)[Qt[:, bidx].argmax(1)]
            used = pick if guard_on else np.full(n, hb)
            freq += np.bincount(used, minlength=len(names))
            hbv.append(At[:, hb].mean()); orc.append(At.max(1).mean())
            sel.append(At[np.arange(n), used].mean()); fb.append(0.0 if guard_on else 1.0)
            if rep == 0:
                for i in range(n):
                    rec_rows.append({"mech": int(i), "q": [round(float(x), 5) for x in Qt[i]],
                                     "auroc": [round(float(x), 5) for x in At[i]],
                                     "pick": names[int(used[i])]})
        M_ = lambda v: float(np.mean(v))
        out["prevalence"][str(prev)] = {
            "historical_best": M_(hbv), "molroute": M_(sel), "oracle_batch": M_(orc),
            "delta": M_(sel) - M_(hbv), "sd_delta": float(np.std(np.array(sel) - np.array(hbv))),
            "fallback": M_(fb),
            "select_freq": {names[j]: round(float(freq[j] / freq.sum()), 4)
                            for j in range(len(names)) if freq[j] > 0}}
        per_mech[str(prev)] = rec_rows
        r = out["prevalence"][str(prev)]
        print(f"[{a.cell}] prev {prev:.0%}  hist_best {r['historical_best']:.4f} -> "
              f"MolRoute {r['molroute']:.4f}  ({r['delta']:+.4f} +-{r['sd_delta']:.4f})  "
              f"oracle {r['oracle_batch']:.4f}", flush=True)
    tag = "_fullbank" if a.full_bank else ""
    json.dump(out, open(f"repro/molroute_{a.cell}{tag}.json", "w"), indent=2)
    json.dump(per_mech, open(f"repro/molroute_permech_{a.cell}{tag}.json", "w"))
    print(f"MOLROUTE_DONE {a.cell}", flush=True)


if __name__ == "__main__":
    main()
