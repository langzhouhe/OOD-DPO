#!/usr/bin/env python3
"""
Cross-shift (unknown-shift) OOD detection: train the detector on ONE shift's
auxiliary OOD, test on ANOTHER shift's OOD. Diagonal = in-criterion (the easy
task the benchmark currently uses); off-diagonal = the realistic "unknown shift"
task. Question: does the FM representation transfer across shifts better than a
trivial molecular-size feature? If yes -> real headroom for a method that
detects OOD robustly across unseen shift types.

3x3 transfer matrix (train-shift x test-shift) for two feature types:
  FM   : frozen MiniMol 512-d + RPO head
  size : [heavy atoms, MolWt] + same head
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, pickle, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from pathlib import Path
from sklearn.metrics import roc_auc_score
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

CACHE = Path("cache/ood_dpo_cache")
SHIFTS = ["size", "scaffold", "assay"]
FEATPKL = {s: f"lbap_general_ec50_{s}_minimol_features.pkl" for s in SHIFTS}
SPLIT = {s: f"lbap_general_ec50_{s}_seed42_splits.json" for s in SHIFTS}


def load(shift):
    fm = pickle.load(open(CACHE / FEATPKL[shift], "rb"))["features"]
    sp = json.load(open(CACHE / SPLIT[shift], "rb"))["splits"]
    return fm, sp


def size_feat(smis):
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        out.append([m.GetNumHeavyAtoms(), Descriptors.MolWt(m)] if m else None)
    return out


def build(shift, kind):
    fm, sp = load(shift)
    keys = ["train_id", "ood_val", "val_id", "val_ood", "test_id", "test_ood"]
    # splits use train_ood/val_ood/test_ood naming; map
    kmap = {"ood_val": "train_ood"}
    F = {}
    for k in keys:
        sk = kmap.get(k, k)
        smis = sp[sk]
        if kind == "fm":
            v = [fm.get(s) for s in smis]
        else:
            v = size_feat(smis)
        F[k] = np.stack([x for x in v if x is not None]).astype(np.float32)
    return F


class Head(nn.Module):
    def __init__(s, d):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(s, x): return s.net(x).squeeze(-1)


def auroc(a, b): return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])


def train_head(Ftr, mu, sd, seed=1, epochs=300):
    torch.manual_seed(seed); np.random.seed(seed)
    Xid = torch.tensor((Ftr["train_id"][:2000] - mu) / sd); Xood = torch.tensor((Ftr["ood_val"][:2000] - mu) / sd)
    vid = torch.tensor((Ftr["val_id"] - mu) / sd); vood = torch.tensor((Ftr["val_ood"] - mu) / sd)
    head = Head(Xid.shape[1]); opt = torch.optim.AdamW(head.parameters(), 1e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9); best, bs = -1, None
    for ep in range(epochs):
        head.train(); opt.zero_grad(); Eid, Eood = head(Xid), head(Xood)
        loss = F.softplus(-0.1 * (Eood.unsqueeze(0) - Eid.unsqueeze(1))).mean() + 0.01 * (Eid.pow(2).mean() + Eood.pow(2).mean())
        loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
        if (ep + 1) % 15 == 0:
            head.eval()
            with torch.no_grad(): v = auroc(head(vid).numpy(), head(vood).numpy())
            if v > best: best, bs = v, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval()
    return head


for kind in ["fm", "size"]:
    feats = {s: build(s, kind) for s in SHIFTS}
    print(f"\n===== {kind.upper()} : train-shift (rows) x test-shift (cols), test AUROC =====")
    hdr = "train|test"
    print(f"{hdr:12}" + "".join(f"{s:>10}" for s in SHIFTS))
    for tr in SHIFTS:
        mu = feats[tr]["train_id"].mean(0); sd = feats[tr]["train_id"].std(0) + 1e-6
        head = train_head(feats[tr], mu, sd)
        row = []
        for te in SHIFTS:
            F2 = feats[te]
            with torch.no_grad():
                sid = head(torch.tensor((F2["test_id"] - mu) / sd)).numpy()
                sood = head(torch.tensor((F2["test_ood"] - mu) / sd)).numpy()
            row.append(auroc(sid, sood))
        print(f"{tr:12}" + "".join(f"{x:>10.3f}" for x in row))
print("\nCROSS_SHIFT_DONE")
