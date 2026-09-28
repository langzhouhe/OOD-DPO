#!/usr/bin/env python3
"""
Zero-OOD reference: how much do the FEW auxiliary OOD samples actually buy?

Compares detectors that see NO OOD at all (KNN / Mahalanobis on ID features
only) against the few-sample OE heads (RPO/BCE @ K). Same frozen features,
same fixed test set. Directly quantifies whether "few-sample RPO is good"
mainly reflects access to OOD or genuine value beyond zero-OOD methods.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, pickle, warnings, logging
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

CACHE = Path("cache/ood_dpo_cache")
CELLS = {
    "ec50_assay":    ("lbap_general_ec50_assay_minimol_features.pkl",     "lbap_general_ec50_assay_seed42_splits.json"),
    "ic50_assay":    ("lbap_general_ic50_assay_minimol_features.pkl",     "lbap_general_ic50_assay_seed42_splits.json"),
    "ec50_scaffold": ("lbap_general_ec50_scaffold_minimol_features.pkl",  "lbap_general_ec50_scaffold_seed42_splits.json"),
    "zinc_scaffold": ("good_zinc_scaffold_covariate_minimol_features.pkl","good_zinc_scaffold_covariate_seed42_splits.json"),
}


def load(cell):
    fc, sc = CELLS[cell]
    feats = pickle.load(open(CACHE / fc, "rb"))["features"]
    sp = json.load(open(CACHE / sc, "rb"))["splits"]
    def mat(smis): return np.stack([feats[s] for s in smis if s in feats])
    return {k: mat(sp[k]) for k in ["train_id", "test_id", "test_ood"]}


def auroc(s_id, s_ood):
    return roc_auc_score(np.r_[np.zeros(len(s_id)), np.ones(len(s_ood))], np.r_[s_id, s_ood])


def knn_score(train_id, X, k=50):
    nn = NearestNeighbors(n_neighbors=k).fit(train_id)
    return nn.kneighbors(X)[0].mean(1)      # mean distance to k nearest ID; higher = more OOD


def maha_score(train_id, X):
    mu = train_id.mean(0)
    cov = np.cov(train_id, rowvar=False) + 1e-3 * np.eye(train_id.shape[1])
    P = np.linalg.pinv(cov)
    d = X - mu
    return np.einsum("ij,jk,ik->i", d, P, d)  # Mahalanobis^2 to ID; higher = more OOD


print(f"{'cell':14}{'KNN(K=0)':>10}{'Maha(K=0)':>11}")
print("-" * 36)
res = {}
for cell in CELLS:
    d = load(cell)
    tid = d["train_id"][:2000]
    a_knn = auroc(knn_score(tid, d["test_id"]), knn_score(tid, d["test_ood"]))
    a_mah = auroc(maha_score(tid, d["test_id"]), maha_score(tid, d["test_ood"]))
    res[cell] = {"knn": a_knn, "maha": a_mah}
    print(f"{cell:14}{a_knn:>10.3f}{a_mah:>11.3f}")
json.dump(res, open("repro/zero_ood_reference.json", "w"), indent=2)
print("ZERO_OOD_DONE")
