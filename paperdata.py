#!/usr/bin/env python3
"""Assemble every number the MolRoute paper figures and tables need, from the canonical
export only. No computation beyond aggregation and a mechanism bootstrap.

Usage: python paperdata.py
"""
import json, glob, os
import numpy as np

ASSAY = [("ec50_assay", "EC50-Assay"), ("ic50_assay", "IC50-Assay"), ("ki_assay", "Ki-Assay")]
SCAF = [("ec50_scaffold", "EC50-Scaffold"), ("ic50_scaffold", "IC50-Scaffold")]
BB = ["minimol", "unimol"]
PREV = ["0.1", "0.25", "0.5"]
ARMS = ["hist_best", "ens_bank", "gbdr", "molroute_T", "molroute_CF",
        "oracle_routing", "oracle_full"]
OUT = {}


def E(c, bb):
    f = f"repro/export_{c}.json" if bb == "minimol" else f"repro/export_{c}_unimol.json"
    return json.load(open(f))


def boot(per_a, per_b, n=2000, seed=0):
    """Bootstrap the paired per-mechanism difference, resampling MECHANISMS."""
    a, b = np.asarray(per_a), np.asarray(per_b)
    rng = np.random.default_rng(seed); m = len(a)
    d = np.array([(a[i] - b[i]).mean() for i in (rng.integers(0, m, (n, m)))])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---------- Table 1 + Fig 3 ----------
OUT["table1"] = {}
for c, lab in ASSAY + SCAF:
    OUT["table1"][c] = {"label": lab}
    for bb in BB:
        d = E(c, bb); rec = {"hist_best_name": d["historical_best"],
                             "guard": "ROUTE" if d["guard_route"] else "FALLBACK"}
        for p in PREV:
            s = d["summary"][p]
            row = {k: round(s[k], 4) for k in ARMS}
            row["delta_T"] = round(s["delta_molroute_T"], 4)
            row["delta_CF"] = round(s["delta_molroute_CF"], 4)
            pm = d["per_mech"][p]
            lo, hi = boot(pm["molroute_T"], pm["hist_best"])
            row["ci_T"] = [round(lo, 4), round(hi, 4)]
            lo, hi = boot(pm["molroute_CF"], pm["hist_best"])
            row["ci_CF"] = [round(lo, 4), round(hi, 4)]
            hr = s["oracle_routing"] - s["hist_best"]
            row["cap_T"] = round((s["molroute_T"] - s["hist_best"]) / hr, 4) if hr > 1e-9 else None
            row["cap_CF"] = round((s["molroute_CF"] - s["hist_best"]) / hr, 4) if hr > 1e-9 else None
            rec[p] = row
        OUT["table1"][c][bb] = rec

# ---------- Fig 1: per-mechanism detector reversal ----------
OUT["fig1"] = {}
for c, lab in ASSAY:
    d = E(c, "minimol")
    dets = d["detectors"]; bank = d["routing_bank"]
    picks = d["rep0"]["0.5"]["picks"]
    A = np.array([p["auroc"] for p in picks])
    i, j = dets.index(bank[1]), dets.index(bank[0])      # Mahalanobis vs OE-MV
    OUT["fig1"][c] = {"label": lab, "x_name": dets[i], "y_name": dets[j],
                      "pts": [[round(float(A[k, i]), 4), round(float(A[k, j]), 4)]
                              for k in range(len(A))],
                      "spearman": round(float(np.corrcoef(
                          A[:, i].argsort().argsort(), A[:, j].argsort().argsort())[0, 1]), 3),
                      "frac_x_wins": round(float((A[:, i] > A[:, j]).mean()), 3)}
OUT["fig1_hb"] = {c: {bb: E(c, bb)["historical_best"] for bb in BB} for c, _ in ASSAY + SCAF}

# ---------- Fig 4a: pure-ID FPR ; Fig 4b: mechanism mixture ----------
OUT["fig4"] = {"pure_id": {}, "mixture": {}}
for c, lab in ASSAY:
    s = json.load(open(f"repro/sanity_{c}.json"))
    OUT["fig4"]["pure_id"][c] = {"label": lab, **{p: {
        "batch": s["pure_id"][p]["batch_size"],
        "hist": round(s["pure_id"][p]["hist_best_fpr5"], 4),
        "same": round(s["pure_id"][p]["molroute_same_fpr5"], 4),
        "half": round(s["pure_id"][p]["molroute_half_fpr5"], 4)} for p in PREV}}
    mx = {}
    for nm in (1, 2, 4):
        v = [s["mixture"][f"{nm}mech_prev{p}"] for p in (0.5, 0.25)]
        hr = np.mean([x["oracle"] - x["hist_best"] for x in v])
        got = np.mean([x["molroute"] - x["hist_best"] for x in v])
        mx[nm] = {"delta": round(float(got), 4), "headroom": round(float(hr), 4),
                  "captured": round(float(got / hr), 4)}
    OUT["fig4"]["mixture"][c] = {"label": lab, **{str(k): v for k, v in mx.items()}}

# ---------- appendix ----------
OUT["select_freq"] = {c: {bb: E(c, bb)["select_freq"]["0.5"] for bb in BB} for c, _ in ASSAY}
OUT["calibration"] = {}
for c, lab in ASSAY + SCAF:
    f = f"repro/molroutecf_{c}.json"
    if os.path.exists(f):
        OUT["calibration"][c] = json.load(open(f))["calibration"]
OUT["per_detector"] = {}
for c, lab in ASSAY:
    d = E(c, "minimol"); dets = d["detectors"]
    A = np.array([p["auroc"] for p in d["rep0"]["0.5"]["picks"]])
    OUT["per_detector"][c] = {dets[k]: round(float(A[:, k].mean()), 4) for k in range(len(dets))}

OUT["meta"] = {"n_mech_per_cell": 400, "molecules_per_mech": 8,
               "test_split": "official ood_test", "test_seed": 777,
               "routing_bank": E("ki_assay", "minimol")["routing_bank"],
               "full_bank": E("ki_assay", "minimol")["detectors"],
               "coverage": {c: E(c, "minimol")["coverage"] for c, _ in ASSAY + SCAF}}

json.dump(OUT, open("repro/paperdata.json", "w"), indent=1)
print("wrote repro/paperdata.json")
for k in OUT: print(f"  {k}")
print()
print("abstract ranges:")
for arm in ("delta_T", "delta_CF"):
    v = [OUT["table1"][c][bb][p][arm] for c, _ in ASSAY for bb in BB for p in PREV]
    v25 = [OUT["table1"][c][bb][p][arm] for c, _ in ASSAY for bb in BB for p in ("0.25", "0.5")]
    print(f"  {arm}: {min(v):+.4f}..{max(v):+.4f}   >=25%: {min(v25):+.4f}..{max(v25):+.4f}")
