#!/usr/bin/env python3
"""
P1 mechanism-identification (CONTROLLED protocol): same head, optimizer, reg,
epochs, best-val selection for ALL losses. Only the objective differs. Goal is
NOT to pick a winner but to attribute the low-budget gap to (a) class balancing,
(b) averaging each scarce OOD against many ID regions, or (c) loss shape.

Losses (round 1 discriminating subset):
  bce_natural   : pooled BCE (OOD underweighted at low K)  -- the fragile ref
  bce_balanced  : 0.5*mean BCE(ID,0) + 0.5*mean BCE(OOD,1) -- explicit class balance
  bce_wnorm     : weighted BCE, weights normalized to mean 1 (no grad-scale change)
  mse_balanced  : 0.5*mean(E_id^2) + 0.5*mean((E_ood-1)^2)
  pair_log_m1   : pairwise logistic, each OOD vs 1 sampled ID  (sampled-m=1)
  pair_log_all  : pairwise logistic, all ID x OOD pairs        (sampled-m=all)
  hybrid        : 0.5*bce_balanced + 0.5*pair_log_all
All losses share the same lambda*E^2 gauge reg and beta.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, pickle, sys, argparse, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

CACHE = Path("cache/ood_dpo_cache")
CELLS = {
    "ec50_assay":    ("lbap_general_ec50_assay_minimol_features.pkl",     "lbap_general_ec50_assay_seed42_splits.json"),
    "ic50_assay":    ("lbap_general_ic50_assay_minimol_features.pkl",     "lbap_general_ic50_assay_seed42_splits.json"),
    "ec50_scaffold": ("lbap_general_ec50_scaffold_minimol_features.pkl",  "lbap_general_ec50_scaffold_seed42_splits.json"),
    "zinc_scaffold": ("good_zinc_scaffold_covariate_minimol_features.pkl","good_zinc_scaffold_covariate_seed42_splits.json"),
}
CONFIGS = ["bce_natural", "bce_balanced", "bce_wnorm", "mse_balanced", "pair_log_m1", "pair_log_all", "hybrid"]
BETA, LAM = 0.1, 0.01


class Head(nn.Module):
    def __init__(s, d=512):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(s, x): return s.net(x).squeeze(-1)


def load(cell):
    fc, sc = CELLS[cell]
    fm = pickle.load(open(CACHE / fc, "rb"))["features"]
    sp = json.load(open(CACHE / sc, "rb"))["splits"]
    def mat(smis): return torch.tensor(np.stack([fm[s] for s in smis if s in fm]).astype(np.float32))
    return {k: mat(sp[k]) for k in ["train_id", "train_ood", "val_id", "val_ood", "test_id", "test_ood"]}


def metrics(s_id, s_ood):
    y = np.r_[np.zeros(len(s_id)), np.ones(len(s_ood))]; s = np.r_[s_id, s_ood]
    au = roc_auc_score(y, s)
    p, r, _ = precision_recall_curve(y, s); ap = auc(r, p)
    fpr, tpr, _ = roc_curve(y, s); idx = np.searchsorted(tpr, 0.95); fpr95 = fpr[min(idx, len(fpr) - 1)]
    return au, ap, fpr95


def core_loss(cfg, Eid, Eood, g):
    if cfg == "bce_natural":
        lg = torch.cat([Eid, Eood]); tg = torch.cat([torch.zeros(len(Eid)), torch.ones(len(Eood))])
        return F.binary_cross_entropy_with_logits(lg, tg)
    if cfg == "bce_balanced":
        return 0.5 * F.binary_cross_entropy_with_logits(Eid, torch.zeros(len(Eid))) + \
               0.5 * F.binary_cross_entropy_with_logits(Eood, torch.ones(len(Eood)))
    if cfg == "bce_wnorm":
        lg = torch.cat([Eid, Eood]); tg = torch.cat([torch.zeros(len(Eid)), torch.ones(len(Eood))])
        per = F.binary_cross_entropy_with_logits(lg, tg, reduction="none")
        w = torch.cat([torch.ones(len(Eid)), (len(Eid) / len(Eood)) * torch.ones(len(Eood))]); w = w / w.mean()
        return (w * per).mean()
    if cfg == "mse_balanced":
        return 0.5 * (Eid - 0.0).pow(2).mean() + 0.5 * (Eood - 1.0).pow(2).mean()
    if cfg == "pair_log_m1":
        idx = torch.randint(len(Eid), (len(Eood),), generator=g)
        return F.softplus(-BETA * (Eood - Eid[idx])).mean()
    if cfg == "pair_log_all":
        return F.softplus(-BETA * (Eood.unsqueeze(0) - Eid.unsqueeze(1))).mean()
    if cfg == "hybrid":
        bal = 0.5 * F.binary_cross_entropy_with_logits(Eid, torch.zeros(len(Eid))) + \
              0.5 * F.binary_cross_entropy_with_logits(Eood, torch.ones(len(Eood)))
        pair = F.softplus(-BETA * (Eood.unsqueeze(0) - Eid.unsqueeze(1))).mean()
        return 0.5 * bal + 0.5 * pair
    raise ValueError(cfg)


def run(d, cfg, k, seed, epochs=150, n_id=2000):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    Xid = d["train_id"][torch.randperm(len(d["train_id"]), generator=g)[:n_id]]
    Xood = d["train_ood"][torch.randperm(len(d["train_ood"]), generator=g)[:k]]
    vid, vood = d["val_id"], d["val_ood"]
    head = Head(); opt = torch.optim.AdamW(head.parameters(), 1e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9); best, bs = -1, None
    for ep in range(epochs):
        head.train(); opt.zero_grad(); Eid, Eood = head(Xid), head(Xood)
        loss = core_loss(cfg, Eid, Eood, g) + LAM * (Eid.pow(2).mean() + Eood.pow(2).mean())
        loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
        if (ep + 1) % 15 == 0:
            head.eval()
            with torch.no_grad(): v = roc_auc_score(np.r_[np.zeros(len(vid)), np.ones(len(vood))],
                                                    np.r_[head(vid).numpy(), head(vood).numpy()])
            if v > best: best, bs = v, {kk: t.clone() for kk, t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval()
    with torch.no_grad():
        return metrics(head(d["test_id"]).numpy(), head(d["test_ood"]).numpy())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budgets", nargs="+", type=int, default=[20, 50, 200, 2000]); a = ap.parse_args()
    d = load(a.task); out = {}
    for k in a.budgets:
        for cfg in CONFIGS:
            res = [run(d, cfg, k, s) for s in range(1, a.seeds + 1)]
            au = [r[0] for r in res]
            out[f"{k}|{cfg}"] = {"auroc": au, "aupr": [r[1] for r in res], "fpr95": [r[2] for r in res]}
        line = f"[{a.task}] K={k:<4} " + " ".join(f"{c.split('_')[0][:4]}{'.'+c.split('_')[1][:2] if '_' in c else ''}={np.mean(out[f'{k}|{c}']['auroc']):.3f}" for c in CONFIGS)
        print(line, flush=True)
    json.dump(out, open(f"repro/lossid_{a.task}.json", "w"), indent=2)
    print(f"LOSSID_DONE {a.task}", flush=True)


if __name__ == "__main__":
    main()
