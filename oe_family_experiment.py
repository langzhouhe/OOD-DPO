#!/usr/bin/env python3
"""
Matched Outlier-Exposure (OE) family comparison.

Same frozen MiniMol features, same auxiliary ID/OOD data, same head
(512->256->128->1, dropout 0.1), same AdamW/StepLR/grad-clip, same seeds,
best-val-AUROC checkpoint selection. Only the training objective differs:

  dpo    : pairwise logistic  softplus(-beta*(E_ood - E_id))          [+ lambda*E^2, as in paper]
  hinge  : pairwise margin     relu(m - (E_ood - E_id))               [+ lambda*E^2, as in paper]
  bce    : pointwise binary    BCEWithLogits(E, y)  (OOD=1)           [+ lambda*E^2, as in paper]
  mse    : pointwise regression (E_id-0)^2 + (E_ood-1)^2              [+ lambda*E^2, as in paper]
  energy : energy-bounded OE (Liu 2020) relu(E_id-m_in)^2 + relu(m_out-E_ood)^2   [own regularization]

Score convention: higher E(x) = more OOD; AUROC with OOD as positive.
train_ood is subsampled to K to probe the scarce-auxiliary-OOD regime.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, pickle, sys, warnings, logging, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)  # tiny MLP; avoid spawning a large intraop threadpool (box near thread cap)
from pathlib import Path
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

CACHE = Path("cache/ood_dpo_cache")
DEV = torch.device("cpu")
CELLS = {
    "ec50_assay":    ("lbap_general_ec50_assay_minimol_features.pkl",     "lbap_general_ec50_assay_seed42_splits.json"),
    "ic50_assay":    ("lbap_general_ic50_assay_minimol_features.pkl",     "lbap_general_ic50_assay_seed42_splits.json"),
    "ec50_scaffold": ("lbap_general_ec50_scaffold_minimol_features.pkl",  "lbap_general_ec50_scaffold_seed42_splits.json"),
    "zinc_scaffold": ("good_zinc_scaffold_covariate_minimol_features.pkl","good_zinc_scaffold_covariate_seed42_splits.json"),
}
OBJS = ["dpo", "hinge", "bce", "mse", "energy"]


class Head(nn.Module):
    def __init__(self, d=512, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(h // 2, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def load_cell(cell):
    fc, sc = CELLS[cell]
    feats = pickle.load(open(CACHE / fc, "rb"))["features"]
    sp = json.load(open(CACHE / sc, "rb"))["splits"]
    def mat(smis):
        v = [feats[s] for s in smis if s in feats]
        return torch.tensor(np.stack(v), dtype=torch.float32)
    return {k: mat(sp[k]) for k in ["train_id", "train_ood", "val_id", "val_ood", "test_id", "test_ood"]}


def auroc(head, Xid, Xood):
    head.eval()
    with torch.no_grad():
        s = np.r_[head(Xid).cpu().numpy(), head(Xood).cpu().numpy()]
    y = np.r_[np.zeros(len(Xid)), np.ones(len(Xood))]
    return roc_auc_score(y, s)


def loss_fn(obj, Eid, Eood, beta=0.1, lam=0.01, margin=1.0, m_in=-1.0, m_out=1.0):
    if obj == "dpo":
        core = F.softplus(-beta * (Eood.unsqueeze(0) - Eid.unsqueeze(1))).mean()
        return core + lam * (Eid.pow(2).mean() + Eood.pow(2).mean())
    if obj == "hinge":
        core = F.relu(margin - (Eood.unsqueeze(0) - Eid.unsqueeze(1))).mean()
        return core + lam * (Eid.pow(2).mean() + Eood.pow(2).mean())
    if obj == "bce":
        logits = torch.cat([Eid, Eood]); labels = torch.cat([torch.zeros(len(Eid)), torch.ones(len(Eood))])
        return F.binary_cross_entropy_with_logits(logits, labels) + lam * (Eid.pow(2).mean() + Eood.pow(2).mean())
    if obj == "mse":
        return (Eid - 0.0).pow(2).mean() + (Eood - 1.0).pow(2).mean() + lam * (Eid.pow(2).mean() + Eood.pow(2).mean())
    if obj == "energy":  # energy-bounded OE, self-regularized by margins
        return F.relu(Eid - m_in).pow(2).mean() + F.relu(m_out - Eood).pow(2).mean()
    raise ValueError(obj)


def train_eval(data, obj, k_ood, seed, epochs=300, n_id=2000):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    tid, tood = data["train_id"], data["train_ood"]
    Xid = tid[torch.randperm(len(tid), generator=g)[:min(n_id, len(tid))]].to(DEV)
    Xood = tood[torch.randperm(len(tood), generator=g)[:min(k_ood, len(tood))]].to(DEV)
    head = Head().to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)
    best_val, best_state = -1, None
    for ep in range(epochs):
        head.train(); opt.zero_grad()
        loss = loss_fn(obj, head(Xid), head(Xood))
        loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sched.step()
        if (ep + 1) % 15 == 0 or ep == epochs - 1:      # best-val-checkpoint selection
            v = auroc(head, data["val_id"].to(DEV), data["val_ood"].to(DEV))
            if v > best_val:
                best_val = v; best_state = {k: t.clone() for k, t in head.state_dict().items()}
    if best_state: head.load_state_dict(best_state)
    return auroc(head, data["test_id"].to(DEV), data["test_ood"].to(DEV))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=list(CELLS))
    ap.add_argument("--budgets", nargs="+", type=int, default=[50, 100, 500, 2000])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    data = load_cell(args.cell)
    out = {}
    for k in args.budgets:
        line = [f"[{args.cell}] K={k:<4}"]
        for obj in OBJS:
            vals = [train_eval(data, obj, k, s) for s in args.seeds]
            m, sd = float(np.mean(vals)), float(np.std(vals))
            out[f"{k}|{obj}"] = {"mean": m, "std": sd, "vals": vals}
            line.append(f"{obj}={m:.3f}±{sd:.3f}")
        print("  ".join(line), flush=True)
    Path("repro").mkdir(exist_ok=True)
    json.dump(out, open(f"repro/oe_family_{args.cell}.json", "w"), indent=2)
    print(f"OE_CELL_DONE {args.cell}", flush=True)


if __name__ == "__main__":
    main()
