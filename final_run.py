#!/usr/bin/env python3
"""5-seed final: locked (val-chosen) configs, full settings (epochs 150, n_id 2000).
Tuned RPO (one-to-one, val-selected beta,lambda) vs tuned balanced-BCE (val-selected
lambda,lr), matched everything. Reports AUROC/AUPR/FPR95 mean+/-std and paired
Delta(RPO-BCE) with 95% CI over the 5 matched seeds."""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, pickle, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

CACHE = Path("cache/ood_dpo_cache")
BASE = {"ec50_assay": "lbap_general_ec50_assay", "ec50_scaffold": "lbap_general_ec50_scaffold", "ec50_size": "lbap_general_ec50_size",
        "ic50_assay": "lbap_general_ic50_assay", "ic50_scaffold": "lbap_general_ic50_scaffold", "ic50_size": "lbap_general_ic50_size",
        "hiv_scaffold": "good_hiv_scaffold_covariate", "hiv_size": "good_hiv_size_covariate",
        "pcba_scaffold": "good_pcba_scaffold_covariate", "pcba_size": "good_pcba_size_covariate",
        "zinc_scaffold": "good_zinc_scaffold_covariate", "zinc_size": "good_zinc_size_covariate"}
# (rpo_beta, rpo_lambda, bce_lambda, bce_lr) from validation search; size cells = default (saturated)
CFG = {"ec50_assay": (10., .5, .1, 3e-4), "ic50_assay": (5., 0., .1, 3e-4),
       "ec50_scaffold": (10., .5, .1, 3e-4), "ic50_scaffold": (10., .5, .1, 3e-4),
       "hiv_scaffold": (10., .5, .1, 3e-4), "pcba_scaffold": (10., .5, .1, 3e-4), "zinc_scaffold": (10., .5, .1, 3e-4)}
DEFAULT = (0.1, 0.01, 0.01, 1e-4)


class Head(nn.Module):
    def __init__(s, d=512):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(s, x): return s.net(x).squeeze(-1)


def load(cell):
    b = BASE[cell]; fm = pickle.load(open(CACHE / f"{b}_minimol_features.pkl", "rb"))["features"]
    sp = json.load(open(CACHE / f"{b}_seed42_splits.json", "rb"))["splits"]
    def mat(s): return torch.tensor(np.stack([fm[x] for x in s if x in fm]).astype(np.float32))
    return {k: mat(sp[k]) for k in ["train_id", "train_ood", "val_id", "val_ood", "test_id", "test_ood"]}


def mets(head, a, b):
    with torch.no_grad(): s = np.r_[head(a).numpy(), head(b).numpy()]
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    p, r, _ = precision_recall_curve(y, s); fpr, tpr, _ = roc_curve(y, s); i = np.searchsorted(tpr, 0.95)
    return roc_auc_score(y, s), auc(r, p), fpr[min(i, len(fpr) - 1)]


def train(d, kind, hp, seed, epochs=150, n_id=2000):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    Xid = d["train_id"][torch.randperm(len(d["train_id"]), generator=g)[:n_id]]
    Xood = d["train_ood"][torch.randperm(len(d["train_ood"]), generator=g)[:2000]]
    lr = 1e-4 if kind == "rpo" else hp[1]
    head = Head(); opt = torch.optim.AdamW(head.parameters(), lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9); best, bs = -1, None
    for ep in range(epochs):
        head.train(); opt.zero_grad(); Eid, Eood = head(Xid), head(Xood)
        if kind == "rpo":
            beta, lam = hp
            idx = torch.randint(len(Eid), (len(Eood),), generator=g)
            core = F.softplus(-beta * (Eood - Eid[idx])).mean()
        else:
            lam = hp[0]
            core = 0.5 * F.binary_cross_entropy_with_logits(Eid, torch.zeros(len(Eid))) + \
                   0.5 * F.binary_cross_entropy_with_logits(Eood, torch.ones(len(Eood)))
        (core + lam * (Eid.pow(2).mean() + Eood.pow(2).mean())).backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
        if (ep + 1) % 15 == 0:
            head.eval()
            with torch.no_grad(): v = roc_auc_score(np.r_[np.zeros(len(d["val_id"])), np.ones(len(d["val_ood"]))],
                                                    np.r_[head(d["val_id"]).numpy(), head(d["val_ood"]).numpy()])
            if v > best: best, bs = v, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval()
    return mets(head, d["test_id"], d["test_ood"])


res = {}
print(f"{'cell':15}| {'RPO AUROC':>16} {'AUPR':>6} {'FPR95':>6} | {'balBCE AUROC':>16} {'AUPR':>6} {'FPR95':>6} | {'dAUROC[95%CI]':>18}")
for cell in BASE:
    d = load(cell); rb, rl, bl, blr = CFG.get(cell, DEFAULT)
    rpo = [train(d, "rpo", (rb, rl), s) for s in range(1, 6)]
    bce = [train(d, "bce", (bl, blr), s) for s in range(1, 6)]
    ra = np.array([x[0] for x in rpo]); ba = np.array([x[0] for x in bce])
    dd = ra - ba; ci = 2.776 * dd.std(ddof=1) / np.sqrt(5)
    res[cell] = {"rpo": {"auroc": ra.tolist(), "aupr": [x[1] for x in rpo], "fpr95": [x[2] for x in rpo]},
                 "bce": {"auroc": ba.tolist(), "aupr": [x[1] for x in bce], "fpr95": [x[2] for x in bce]},
                 "delta_auroc": float(dd.mean()), "ci95": float(ci)}
    print(f"{cell:15}| {ra.mean():.3f}±{ra.std(ddof=1):.3f}    {np.mean([x[1] for x in rpo]):.3f}  {np.mean([x[2] for x in rpo]):.3f} | "
          f"{ba.mean():.3f}±{ba.std(ddof=1):.3f}    {np.mean([x[1] for x in bce]):.3f}  {np.mean([x[2] for x in bce]):.3f} | "
          f"{dd.mean():+.3f} [±{ci:.3f}]", flush=True)
json.dump(res, open("repro/final_5seed.json", "w"), indent=2)
print("FINAL_DONE", flush=True)
