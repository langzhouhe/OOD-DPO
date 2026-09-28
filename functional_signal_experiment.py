#!/usr/bin/env python3
"""
Is there a FUNCTIONAL signal (FM downstream prediction behavior) that detects
the assay (functional) OOD axis where STRUCTURAL embedding detectors fail
(cross-shift assay ~0.44)?

Train an ID-only property classifier (MLP on frozen features, cls_label) and use
its post-hoc uncertainty (MSP / energy / entropy) as an OOD score -- this uses
NO OOD labels at all. Compare on each shift's test set against the structural
in-criterion numbers. If functional uncertainty detects assay >> structural
cross-shift, a multi-signal method has real headroom.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, pickle, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from pathlib import Path
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

CACHE = Path("cache/ood_dpo_cache")
SHIFTS = ["size", "scaffold", "assay"]

# SMILES -> cls_label from raw DrugOOD json; SMILES -> FM vec from cache
LAB = {}
for s in SHIFTS:
    d = json.load(open(f"data/raw/lbap_general_ec50_{s}.json"))["split"]
    for k in ["train", "iid_val", "iid_test", "ood_val", "ood_test"]:
        for it in d.get(k, []):
            if it.get("smiles") is not None and it.get("cls_label") is not None:
                LAB[it["smiles"]] = int(it["cls_label"])
FM = {}
SPL = {}
for s in SHIFTS:
    FM.update(pickle.load(open(CACHE / f"lbap_general_ec50_{s}_minimol_features.pkl", "rb"))["features"])
    SPL[s] = json.load(open(CACHE / f"lbap_general_ec50_{s}_seed42_splits.json", "rb"))["splits"]


def xy(smis):
    X, y = [], []
    for s in smis:
        if s in FM and s in LAB:
            X.append(FM[s]); y.append(LAB[s])
    return np.stack(X).astype(np.float32), np.array(y)


class Clf(nn.Module):
    def __init__(s, d=512, c=2):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, c))
    def forward(s, x): return s.net(x)


def auroc(a, b): return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])


def train_clf(shift, seed=1, epochs=200):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = xy(SPL[shift]["train_id"]); mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xt = torch.tensor((Xtr - mu) / sd); yt = torch.tensor(ytr)
    clf = Clf(); opt = torch.optim.AdamW(clf.parameters(), 1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        clf.train(); opt.zero_grad(); loss = F.cross_entropy(clf(Xt), yt); loss.backward(); opt.step()
    clf.eval()
    return clf, mu, sd


def scores(clf, mu, sd, smis):
    X, _ = xy(smis) if any(s in LAB for s in smis) else (np.stack([FM[s] for s in smis if s in FM]).astype(np.float32), None)
    with torch.no_grad():
        logit = clf(torch.tensor((X - mu) / sd))
        p = F.softmax(logit, 1).numpy()
        msp = 1 - p.max(1)                          # higher = more OOD
        energy = -torch.logsumexp(logit, 1).numpy() # higher(=less negative) ~ more OOD
        ent = -(p * np.log(p + 1e-9)).sum(1)
    return {"MSP": msp, "energy": energy, "entropy": ent}


print(f"{'shift':10}{'MSP':>8}{'energy':>8}{'entropy':>9}   (functional OOD detection, ID-only classifier)")
print("-" * 52)
for shift in SHIFTS:
    clf, mu, sd = train_clf(shift)
    sid = scores(clf, mu, sd, SPL[shift]["test_id"]); sood = scores(clf, mu, sd, SPL[shift]["test_ood"])
    row = {m: auroc(sid[m], sood[m]) for m in ["MSP", "energy", "entropy"]}
    print(f"{shift:10}{row['MSP']:>8.3f}{row['energy']:>8.3f}{row['entropy']:>9.3f}", flush=True)
print("FUNC_DONE")
