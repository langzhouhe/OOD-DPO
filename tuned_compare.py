#!/usr/bin/env python3
"""
The #1 reviewer/AC ask: matched, equally-tuned RPO vs balanced-BCE.
Same encoder/head/ID-OOD data/validation/seeds. Both get a validation-AUROC
hyperparameter search of comparable size:
  RPO           : beta x lambda  (7x7=49), lr=1e-4, one-to-one pairing
  balanced-BCE  : lambda x lr    (7x3=21)   [0.5*BCE(id,0)+0.5*BCE(ood,1)+lam*E^2]
Search with 3 seeds (epochs 100); report default-RPO, tuned-RPO, tuned-bal-BCE
test AUROC at the validation-chosen config, per cell.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, pickle, argparse, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from pathlib import Path
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

CACHE = Path("cache/ood_dpo_cache")
CELLS = {"ec50_assay": "lbap_general_ec50_assay", "ic50_assay": "lbap_general_ic50_assay",
         "ec50_scaffold": "lbap_general_ec50_scaffold", "ic50_scaffold": "lbap_general_ic50_scaffold",
         "hiv_scaffold": "good_hiv_scaffold_covariate", "pcba_scaffold": "good_pcba_scaffold_covariate",
         "zinc_scaffold": "good_zinc_scaffold_covariate"}
BETAS = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.5]
LRS = [3e-5, 1e-4, 3e-4]


class Head(nn.Module):
    def __init__(s, d=512):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(s, x): return s.net(x).squeeze(-1)


def load(cell):
    b = CELLS[cell]; fm = pickle.load(open(CACHE / f"{b}_minimol_features.pkl", "rb"))["features"]
    sp = json.load(open(CACHE / f"{b}_seed42_splits.json", "rb"))["splits"]
    def mat(s): return torch.tensor(np.stack([fm[x] for x in s if x in fm]).astype(np.float32))
    return {k: mat(sp[k]) for k in ["train_id", "train_ood", "val_id", "val_ood", "test_id", "test_ood"]}


def auc(head, a, b):
    with torch.no_grad(): return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[head(a).numpy(), head(b).numpy()])


def train(d, kind, hp, seed, lr, epochs=100, n_id=1500):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    Xid = d["train_id"][torch.randperm(len(d["train_id"]), generator=g)[:n_id]]
    Xood = d["train_ood"][torch.randperm(len(d["train_ood"]), generator=g)[:2000]]
    head = Head(); opt = torch.optim.AdamW(head.parameters(), lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9); best, bs = -1, None
    for ep in range(epochs):
        head.train(); opt.zero_grad(); Eid, Eood = head(Xid), head(Xood)
        if kind == "rpo":
            beta, lam = hp
            idx = torch.randint(len(Eid), (len(Eood),), generator=g)
            core = F.softplus(-beta * (Eood - Eid[idx])).mean()
        else:  # balanced BCE
            lam = hp
            core = 0.5 * F.binary_cross_entropy_with_logits(Eid, torch.zeros(len(Eid))) + \
                   0.5 * F.binary_cross_entropy_with_logits(Eood, torch.ones(len(Eood)))
        loss = core + lam * (Eid.pow(2).mean() + Eood.pow(2).mean())
        loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
        if (ep + 1) % 15 == 0:
            head.eval(); v = auc(head, d["val_id"], d["val_ood"])
            if v > best: best, bs = v, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval()
    return best, auc(head, d["test_id"], d["test_ood"])


def search(d, kind, seeds=3):
    grid = {}
    if kind == "rpo":
        for b in BETAS:
            for l in LAMBDAS:
                r = [train(d, "rpo", (b, l), s, 1e-4) for s in range(1, seeds + 1)]
                grid[f"{b}|{l}"] = (float(np.mean([x[0] for x in r])), float(np.mean([x[1] for x in r])))
    else:
        for l in LAMBDAS:
            for lr in LRS:
                r = [train(d, "bce", l, s, lr) for s in range(1, seeds + 1)]
                grid[f"{l}|{lr}"] = (float(np.mean([x[0] for x in r])), float(np.mean([x[1] for x in r])))
    bk = max(grid, key=lambda k: grid[k][0])
    return grid, bk, grid[bk][1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cell", required=True); a = ap.parse_args()
    d = load(a.cell)
    rgrid, rbk, rtest = search(d, "rpo"); bgrid, bbk, btest = search(d, "bce")
    dflt = rgrid["0.1|0.01"][1]
    print(f"[{a.cell}] default_RPO={dflt:.3f} | tuned_RPO={rtest:.3f}(β,λ={rbk}) | tuned_balBCE={btest:.3f}(λ,lr={bbk}) | RPO-BCE={rtest-btest:+.3f}", flush=True)
    json.dump({"rpo": rgrid, "bce": bgrid, "default_rpo": dflt, "tuned_rpo": rtest, "tuned_bce": btest,
               "rpo_cfg": rbk, "bce_cfg": bbk}, open(f"repro/tunedcmp_{a.cell}.json", "w"), indent=2)
    print(f"CMP_DONE {a.cell}", flush=True)


if __name__ == "__main__":
    main()
