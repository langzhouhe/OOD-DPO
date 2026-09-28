#!/usr/bin/env python3
"""
Direction (1): hybrid ViM-style score = learned OE head (RPO) + feature-space
residual (Mahalanobis). The two are complementary (RPO wins on scaffold, Maha
on assay). Combination weight alpha is tuned on VALIDATION only (no test
leakage); report TEST AUROC of each component and the hybrid.

Honest eval: matched features, best-val head, alpha selected on val.
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
CELLS = {
    "ec50_assay":    ("lbap_general_ec50_assay_minimol_features.pkl",     "lbap_general_ec50_assay_seed42_splits.json"),
    "ic50_assay":    ("lbap_general_ic50_assay_minimol_features.pkl",     "lbap_general_ic50_assay_seed42_splits.json"),
    "ec50_scaffold": ("lbap_general_ec50_scaffold_minimol_features.pkl",  "lbap_general_ec50_scaffold_seed42_splits.json"),
    "zinc_scaffold": ("good_zinc_scaffold_covariate_minimol_features.pkl","good_zinc_scaffold_covariate_seed42_splits.json"),
}


class Head(nn.Module):
    def __init__(s, d=512, h=256):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1),
                              nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h // 2, 1))
    def forward(s, x): return s.net(x).squeeze(-1)


def load(cell):
    fc, sc = CELLS[cell]
    feats = pickle.load(open(CACHE / fc, "rb"))["features"]
    sp = json.load(open(CACHE / sc, "rb"))["splits"]
    def mat(smis): return np.stack([feats[s] for s in smis if s in feats]).astype(np.float32)
    return {k: mat(sp[k]) for k in ["train_id", "train_ood", "val_id", "val_ood", "test_id", "test_ood"]}


def auroc(s_id, s_ood): return roc_auc_score(np.r_[np.zeros(len(s_id)), np.ones(len(s_ood))], np.r_[s_id, s_ood])


def train_rpo(d, seed=1, epochs=300, beta=0.1, lam=0.01):
    torch.manual_seed(seed); np.random.seed(seed)
    Xid = torch.tensor(d["train_id"][:2000]); Xood = torch.tensor(d["train_ood"][:2000])
    vid = torch.tensor(d["val_id"]); vood = torch.tensor(d["val_ood"])
    head = Head(); opt = torch.optim.AdamW(head.parameters(), 1e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9); best, bs = -1, None
    for ep in range(epochs):
        head.train(); opt.zero_grad()
        Eid, Eood = head(Xid), head(Xood)
        loss = F.softplus(-beta * (Eood.unsqueeze(0) - Eid.unsqueeze(1))).mean() + lam * (Eid.pow(2).mean() + Eood.pow(2).mean())
        loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
        if (ep + 1) % 15 == 0:
            head.eval()
            with torch.no_grad():
                v = auroc(head(vid).numpy(), head(vood).numpy())
            if v > best: best, bs = v, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval()
    def E(X):
        with torch.no_grad(): return head(torch.tensor(X)).numpy()
    return E


def maha_scorer(train_id):
    mu = train_id.mean(0); cov = np.cov(train_id, rowvar=False) + 1e-3 * np.eye(train_id.shape[1]); P = np.linalg.pinv(cov)
    return lambda X: np.einsum("ij,jk,ik->i", X - mu, P, X - mu)


def z(v, ref): return (v - ref.mean()) / (ref.std() + 1e-8)


print(f"{'cell':14}{'RPO':>8}{'Maha':>8}{'Hybrid':>8}{'best_a':>8}")
print("-" * 46)
res = {}
for cell in CELLS:
    d = load(cell)
    E = train_rpo(d); M = maha_scorer(d["train_id"][:2000])
    # component scores
    parts = {}
    for split in ["val_id", "val_ood", "test_id", "test_ood"]:
        parts[("e", split)] = E(d[split]); parts[("m", split)] = M(d[split])
    # z-normalize using validation ID as reference
    eref = parts[("e", "val_id")]; mref = parts[("m", "val_id")]
    def hyb(a, split): return a * z(parts[("e", split)], eref) + (1 - a) * z(parts[("m", split)], mref)
    # tune alpha on validation
    best_a, best_v = 1.0, -1
    for a in np.linspace(0, 1, 21):
        v = auroc(hyb(a, "val_id"), hyb(a, "val_ood"))
        if v > best_v: best_v, best_a = v, a
    a_rpo = auroc(parts[("e", "test_id")], parts[("e", "test_ood")])
    a_mah = auroc(parts[("m", "test_id")], parts[("m", "test_ood")])
    a_hyb = auroc(hyb(best_a, "test_id"), hyb(best_a, "test_ood"))
    res[cell] = {"rpo": a_rpo, "maha": a_mah, "hybrid": a_hyb, "alpha": best_a}
    print(f"{cell:14}{a_rpo:>8.3f}{a_mah:>8.3f}{a_hyb:>8.3f}{best_a:>8.2f}", flush=True)
json.dump(res, open("repro/hybrid_results.json", "w"), indent=2)
print("HYBRID_DONE")
