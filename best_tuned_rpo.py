#!/usr/bin/env python3
"""
Best-tuned RPO: per (dataset,shift) pick RPO hyperparameters by VALIDATION AUROC,
then report default-RPO vs best-tuned-RPO on test. Fixed MLP head.
Stage A searches beta x lambda (the params the paper's own sensitivity study flags);
lr fixed 1e-4, pairing = random one-to-one (shown ~= all-pairs, much cheaper),
hard-pair fraction = 1.0. Refinement of lr/pair/hardfrac/K is Stage B.
Selection uses ONLY validation; test seen once at the val-chosen config.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, pickle, argparse, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

CACHE = Path("cache/ood_dpo_cache")
CELLS = {
    "ec50_assay":    "lbap_general_ec50_assay",
    "ic50_assay":    "lbap_general_ic50_assay",
    "ec50_scaffold": "lbap_general_ec50_scaffold",
    "ic50_scaffold": "lbap_general_ic50_scaffold",
    "hiv_scaffold":  "good_hiv_scaffold_covariate",
    "pcba_scaffold": "good_pcba_scaffold_covariate",
    "zinc_scaffold": "good_zinc_scaffold_covariate",
}
BETAS = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.5]


class Head(nn.Module):
    def __init__(s, d=512):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(s, x): return s.net(x).squeeze(-1)


def load(cell):
    base = CELLS[cell]
    fm = pickle.load(open(CACHE / f"{base}_minimol_features.pkl", "rb"))["features"]
    sp = json.load(open(CACHE / f"{base}_seed42_splits.json", "rb"))["splits"]
    def mat(smis): return torch.tensor(np.stack([fm[s] for s in smis if s in fm]).astype(np.float32))
    return {k: mat(sp[k]) for k in ["train_id", "train_ood", "val_id", "val_ood", "test_id", "test_ood"]}


def mets(s_id, s_ood):
    y = np.r_[np.zeros(len(s_id)), np.ones(len(s_ood))]; s = np.r_[s_id, s_ood]
    au = roc_auc_score(y, s); p, r, _ = precision_recall_curve(y, s); ap = auc(r, p)
    fpr, tpr, _ = roc_curve(y, s); i = np.searchsorted(tpr, 0.95); f95 = fpr[min(i, len(fpr) - 1)]
    return au, ap, f95


def rpo_loss(Eid, Eood, beta, lam, pair, hardfrac, g):
    if pair == "all":
        diff = (Eood.unsqueeze(0) - Eid.unsqueeze(1)).reshape(-1)
    else:  # one-to-one
        idx = torch.randint(len(Eid), (len(Eood),), generator=g); diff = Eood - Eid[idx]
    per = F.softplus(-beta * diff)
    if hardfrac < 1.0:
        kk = max(1, int(hardfrac * per.numel())); per = torch.topk(per, kk).values
    return per.mean() + lam * (Eid.pow(2).mean() + Eood.pow(2).mean())


def run(d, beta, lam, seed, lr=1e-4, pair="1to1", hardfrac=1.0, k=None, epochs=150, n_id=2000):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    ko = len(d["train_ood"]) if k is None else min(k, len(d["train_ood"]))
    Xid = d["train_id"][torch.randperm(len(d["train_id"]), generator=g)[:n_id]]
    Xood = d["train_ood"][torch.randperm(len(d["train_ood"]), generator=g)[:ko]]
    head = Head(); opt = torch.optim.AdamW(head.parameters(), lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9); best, bs, bestv = -1, None, -1
    for ep in range(epochs):
        head.train(); opt.zero_grad()
        loss = rpo_loss(head(Xid), head(Xood), beta, lam, pair, hardfrac, g)
        loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
        if (ep + 1) % 15 == 0:
            head.eval()
            with torch.no_grad(): v = roc_auc_score(np.r_[np.zeros(len(d["val_id"])), np.ones(len(d["val_ood"]))],
                                                    np.r_[head(d["val_id"]).numpy(), head(d["val_ood"]).numpy()])
            if v > best: best, bs, bestv = v, {kk: t.clone() for kk, t in head.state_dict().items()}, v
    head.load_state_dict(bs); head.eval()
    with torch.no_grad(): te = mets(head(d["test_id"]).numpy(), head(d["test_ood"]).numpy())
    return bestv, te


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cell", required=True); ap.add_argument("--seeds", type=int, default=3); a = ap.parse_args()
    d = load(a.cell); grid = {}
    for b in BETAS:
        for l in LAMBDAS:
            V, T = [], []
            for s in range(1, a.seeds + 1):
                v, te = run(d, b, l, s); V.append(v); T.append(te[0])
            grid[f"{b}|{l}"] = {"val": float(np.mean(V)), "test": float(np.mean(T))}
    # default RPO
    dv = grid["0.1|0.01"]["test"]
    # val-selected best config
    bestk = max(grid, key=lambda kk: grid[kk]["val"])
    bt = grid[bestk]["test"]; bb, bl = bestk.split("|")
    print(f"[{a.cell}] default(b0.1,l0.01) test={dv:.3f} | best-tuned test={bt:.3f} (val-chosen b={bb},l={bl}) | gain={bt-dv:+.3f}", flush=True)
    json.dump(grid, open(f"repro/tuned_{a.cell}.json", "w"), indent=2)
    print(f"TUNED_DONE {a.cell}", flush=True)


if __name__ == "__main__":
    main()
