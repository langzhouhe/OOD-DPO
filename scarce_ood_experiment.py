#!/usr/bin/env python3
"""
Scarce-auxiliary-OOD experiment: does the pairwise (RPO/DPO) objective beat a
matched pointwise BCE head when auxiliary OOD samples are FEW?

Faithful to the paper: same frozen features (cached), same head
(512->256->128->1, dropout 0.1), same optimizer (AdamW), StepLR(10,0.9),
grad-clip 1.0, beta=0.1, lambda=0.01. Only the loss differs. Test set is held
FIXED at full size; only the number of training OOD samples varies. Multiple
seeds; report mean +/- std test AUROC.
"""
import json, pickle, sys, warnings, logging, argparse
import numpy as np, torch, torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

CACHE = Path("cache/ood_dpo_cache")
DEV = torch.device("cpu")

CELLS = {  # cell -> (feature_cache, splits_cache)
    "ec50_assay":    ("lbap_general_ec50_assay_minimol_features.pkl",    "lbap_general_ec50_assay_seed42_splits.json"),
    "ic50_assay":    ("lbap_general_ic50_assay_minimol_features.pkl",    "lbap_general_ic50_assay_seed42_splits.json"),
    "ec50_scaffold": ("lbap_general_ec50_scaffold_minimol_features.pkl", "lbap_general_ec50_scaffold_seed42_splits.json"),
    "zinc_scaffold": ("good_zinc_scaffold_covariate_minimol_features.pkl","good_zinc_scaffold_covariate_seed42_splits.json"),
}


class Head(nn.Module):
    def __init__(self, d=512, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(h // 2, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_cell(cell):
    fc, sc = CELLS[cell]
    feats = pickle.load(open(CACHE / fc, "rb"))["features"]
    sp = json.load(open(CACHE / sc, "rb"))["splits"]
    def mat(smis):
        v = [feats[s] for s in smis if s in feats]
        return torch.tensor(np.stack(v), dtype=torch.float32)
    return {k: mat(sp[k]) for k in ["train_id", "train_ood", "test_id", "test_ood"]}


def train_eval(data, loss_type, k_ood, seed, epochs=300, beta=0.1, lam=0.01, n_id=2000):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed); np.random.seed(seed)
    tid, tood = data["train_id"], data["train_ood"]
    id_idx = torch.randperm(len(tid), generator=g)[:min(n_id, len(tid))]
    ood_idx = torch.randperm(len(tood), generator=g)[:min(k_ood, len(tood))]
    Xid, Xood = tid[id_idx].to(DEV), tood[ood_idx].to(DEV)
    head = Head().to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)
    bce = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        head.train(); opt.zero_grad()
        Eid, Eood = head(Xid), head(Xood)
        if loss_type == "dpo":
            diff = Eood.unsqueeze(0) - Eid.unsqueeze(1)          # [n_id, k_ood]
            loss = torch.nn.functional.softplus(-beta * diff).mean()
        else:  # bce: OOD=1 (high energy), ID=0
            logits = torch.cat([Eid, Eood]); labels = torch.cat([torch.zeros(len(Eid)), torch.ones(len(Eood))])
            loss = bce(logits, labels)
        loss = loss + lam * (Eid.pow(2).mean() + Eood.pow(2).mean())
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); sched.step()
    head.eval()
    with torch.no_grad():
        s_id = head(data["test_id"].to(DEV)).cpu().numpy()
        s_ood = head(data["test_ood"].to(DEV)).cpu().numpy()
    y = np.r_[np.zeros(len(s_id)), np.ones(len(s_ood))]
    return roc_auc_score(y, np.r_[s_id, s_ood])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--budgets", nargs="+", type=int, default=[10, 25, 50, 100, 250, 500, 2000])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    print(f"{'cell':14}{'K_ood':>6}  {'DPO(mean±std)':>18}  {'BCE(mean±std)':>18}  {'Δ(DPO-BCE)':>11}")
    print("-" * 74)
    summary = {}
    for cell in args.cells:
        data = load_cell(cell)
        for k in args.budgets:
            dpo = [train_eval(data, "dpo", k, s) for s in args.seeds]
            bce = [train_eval(data, "bce", k, s) for s in args.seeds]
            dm, ds = np.mean(dpo), np.std(dpo); bm, bs = np.mean(bce), np.std(bce)
            summary[f"{cell}|{k}"] = {"dpo": dpo, "bce": bce}
            print(f"{cell:14}{k:>6}  {dm:>7.3f} ± {ds:<7.3f}  {bm:>7.3f} ± {bs:<7.3f}  {dm-bm:>+11.3f}", flush=True)
        print("-" * 74)
    json.dump(summary, open("repro/scarce_ood_summary.json", "w"), indent=2)
    print("SCARCE_DONE")


if __name__ == "__main__":
    main()
