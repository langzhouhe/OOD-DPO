#!/usr/bin/env python3
"""Shortcut audit: how much of each benchmark cell is solvable by a single trivial
molecular descriptor, with NO training at all?

For every cell we take the EXACT molecule subsets the experiments use
(cache/ood_dpo_cache/*_seed42_splits.json) and score OOD-ness by one raw scalar
feature.  Three things are reported per (cell, feature):

  sep_test   orientation-free AUROC on (test_id vs test_ood) = max(a, 1-a).
             How much of the cell one number already explains.
  gap        True if NO test-ID molecule and NO test-OOD molecule share a value,
             i.e. a single threshold separates them with an empty margin.
  dir_flip   True if the feature's ID/OOD orientation on (train_id, train_ood)
             is OPPOSITE to its orientation on the test split.  A flip means
             outlier exposure teaches the detector the WRONG direction, which is
             a mechanism for negative transfer.

Why this matters: a cell where sep_test ~= 1.0 with a gap is not measuring OOD
detection.  Any method given auxiliary OOD is handed the decision rule, so it
scores ~1.000 for free, while ID-only methods cannot recover it.  That makes the
published leaderboard on those cells a ranking of OOD-access, not of detection.

Writes repro/shortcut_audit.json.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, glob, warnings, logging
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from sklearn.metrics import roc_auc_score

CACHE = "cache/ood_dpo_cache"

FEATURES = {
    "heavy_atoms": lambda m: m.GetNumHeavyAtoms(),
    "molwt":       Descriptors.MolWt,
    "n_rings":     rdMolDescriptors.CalcNumRings,
    "logp":        Descriptors.MolLogP,
    "tpsa":        Descriptors.TPSA,
    "n_rotbonds":  Descriptors.NumRotatableBonds,
}


def featurize(smiles):
    out = {k: [] for k in FEATURES}
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        for k, f in FEATURES.items():
            out[k].append(float(f(m)))
    return {k: np.asarray(v) for k, v in out.items()}


def audit_pair(a, b):
    """a = ID values, b = OOD values. Returns (signed_auroc, orientation_free, gap)."""
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    s = np.r_[a, b]
    au = roc_auc_score(y, s)
    gap = bool(a.min() > b.max() or b.min() > a.max())
    return au, max(au, 1.0 - au), gap


def main():
    rows = []
    for f in sorted(glob.glob(f"{CACHE}/*_seed42_splits.json")):
        cell = os.path.basename(f).replace("_seed42_splits.json", "")
        sp = json.load(open(f))["splits"]
        F = {k: featurize(sp[k]) for k in
             ("train_id", "train_ood", "test_id", "test_ood")}
        for feat in FEATURES:
            au_tr, sep_tr, _ = audit_pair(F["train_id"][feat], F["train_ood"][feat])
            au_te, sep_te, gap = audit_pair(F["test_id"][feat], F["test_ood"][feat])
            rows.append(dict(cell=cell, feature=feat,
                             sep_train=round(sep_tr, 4), sep_test=round(sep_te, 4),
                             gap=gap,
                             dir_flip=bool((au_tr - 0.5) * (au_te - 0.5) < 0),
                             med_id=round(float(np.median(F["test_id"][feat])), 2),
                             med_ood=round(float(np.median(F["test_ood"][feat])), 2)))

    json.dump(rows, open("repro/shortcut_audit.json", "w"), indent=1)

    # ---- headline table: the single most predictive trivial feature per cell ----
    best = {}
    for r in rows:
        if r["cell"] not in best or r["sep_test"] > best[r["cell"]]["sep_test"]:
            best[r["cell"]] = r
    print("Most predictive TRIVIAL feature per cell (no training, raw scalar score)\n")
    print(f"{'cell':34s} {'feature':12s} {'sep_tr':>7s} {'sep_te':>7s} {'gap':>4s} "
          f"{'flip':>5s}  med ID/OOD")
    print("-" * 92)
    for cell, r in sorted(best.items(), key=lambda kv: -kv[1]["sep_test"]):
        print(f"{cell:34s} {r['feature']:12s} {r['sep_train']:7.4f} {r['sep_test']:7.4f} "
              f"{'YES' if r['gap'] else '-':>4s} {'YES' if r['dir_flip'] else '-':>5s}  "
              f"{r['med_id']:g} / {r['med_ood']:g}")

    print("\nheavy_atoms alone, every cell (the feature the size splits are built on)\n")
    print(f"{'cell':34s} {'sep_tr':>7s} {'sep_te':>7s} {'gap':>4s} {'flip':>5s}")
    print("-" * 64)
    for r in sorted([r for r in rows if r["feature"] == "heavy_atoms"],
                    key=lambda r: -r["sep_test"]):
        print(f"{r['cell']:34s} {r['sep_train']:7.4f} {r['sep_test']:7.4f} "
              f"{'YES' if r['gap'] else '-':>4s} {'YES' if r['dir_flip'] else '-':>5s}")

    flips = [r for r in rows if r["dir_flip"] and r["sep_test"] > 0.55]
    if flips:
        print("\nDIRECTION FLIPS (train-OOD orientation opposite to test-OOD) -- "
              "outlier exposure learns the wrong sign here:")
        for r in flips:
            print(f"  {r['cell']:32s} {r['feature']:12s} sep_tr={r['sep_train']:.4f} "
                  f"sep_te={r['sep_test']:.4f}")
    print("\nwrote repro/shortcut_audit.json")


if __name__ == "__main__":
    main()
