#!/usr/bin/env python3
"""
Leave-one-shift-out (LOSO) unknown-shift OOD detection.
For held-out shift H: train the detector on the OTHER two shifts' auxiliary OOD
(diverse, multi-shift training), test on H's unseen OOD. Compares:
  - train on 1 other shift  (single-source, from the 3x3 matrix)
  - train on 2 other shifts  (LOSO / diverse)
  - train on H itself        (in-criterion upper bound)
and RPO(dpo) vs BCE, to see whether (a) diverse training generalizes to unseen
shifts and (b) the pairwise objective generalizes better than pointwise.
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

# merge FM features across the 3 subsets into one global SMILES->vec dict
FM = {}
SPL = {}
for s in SHIFTS:
    FM.update(pickle.load(open(CACHE / f"lbap_general_ec50_{s}_minimol_features.pkl", "rb"))["features"])
    SPL[s] = json.load(open(CACHE / f"lbap_general_ec50_{s}_seed42_splits.json", "rb"))["splits"]


def vecs(smis):
    return np.stack([FM[s] for s in smis if s in FM]).astype(np.float32)


def pool(shifts, key):
    smis = []
    for s in shifts:
        smis += SPL[s][key]
    return list(dict.fromkeys(smis))   # dedup, keep order


class Head(nn.Module):
    def __init__(s, d=512):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(s, x): return s.net(x).squeeze(-1)


def auroc(a, b): return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])


def train_head(Xid, Xood, vid, vood, mu, sd, loss_type, seed=1, epochs=300):
    torch.manual_seed(seed); np.random.seed(seed)
    Xid = torch.tensor((Xid - mu) / sd); Xood = torch.tensor((Xood - mu) / sd)
    vid = torch.tensor((vid - mu) / sd); vood = torch.tensor((vood - mu) / sd)
    head = Head(); opt = torch.optim.AdamW(head.parameters(), 1e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9); best, bs = -1, None
    for ep in range(epochs):
        head.train(); opt.zero_grad(); Eid, Eood = head(Xid), head(Xood)
        if loss_type == "dpo":
            core = F.softplus(-0.1 * (Eood.unsqueeze(0) - Eid.unsqueeze(1))).mean()
        else:
            lg = torch.cat([Eid, Eood]); lb = torch.cat([torch.zeros(len(Eid)), torch.ones(len(Eood))])
            core = F.binary_cross_entropy_with_logits(lg, lb)
        loss = core + 0.01 * (Eid.pow(2).mean() + Eood.pow(2).mean())
        loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
        if (ep + 1) % 15 == 0:
            head.eval()
            with torch.no_grad(): v = auroc(head(vid).numpy(), head(vood).numpy())
            if v > best: best, bs = v, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval(); return head


def run(train_shifts, test_shift, loss_type):
    Xid = vecs(pool(train_shifts, "train_id"))[:3000]
    Xood = vecs(pool(train_shifts, "train_ood"))[:3000]
    vid = vecs(pool(train_shifts, "val_id"))[:1500]; vood = vecs(pool(train_shifts, "val_ood"))[:1500]
    mu = Xid.mean(0); sd = Xid.std(0) + 1e-6
    head = train_head(Xid, Xood, vid, vood, mu, sd, loss_type)
    tid = vecs(SPL[test_shift]["test_id"]); tood = vecs(SPL[test_shift]["test_ood"])
    with torch.no_grad():
        return auroc(head(torch.tensor((tid - mu) / sd)).numpy(), head(torch.tensor((tood - mu) / sd)).numpy())


print(f"{'held-out':10}{'loss':6}{'train=other2':>14}{'train=self':>12}{'best-single-other':>18}")
print("-" * 62)
for H in SHIFTS:
    others = [s for s in SHIFTS if s != H]
    for lt in ["dpo", "bce"]:
        a_two = run(others, H, lt)          # LOSO: diverse multi-shift training -> unseen shift
        a_self = run([H], H, lt)            # in-criterion upper bound
        a_single = max(run([o], H, lt) for o in others)  # best single other shift
        print(f"{H:10}{lt:6}{a_two:>14.3f}{a_self:>12.3f}{a_single:>18.3f}", flush=True)
print("CROSS_METHOD_DONE")
