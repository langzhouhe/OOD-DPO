#!/usr/bin/env python3
"""
Phase 1: exposure-strength sweep + ZINC diagnosis.
Loss  L_rho = (1-rho)*BCE_natural + rho*BCE_balanced  (rho in [0,1]).
  rho=0  -> natural (weak OE, OOD under-weighted)
  rho=1  -> balanced (strong OE)
Records VAL and TEST AUROC/AUPR/FPR95, so we can test the key question:
  does argmax_rho AUROC_val pick a rho that AVOIDS negative transfer (esp. ZINC)?
Also records, at K=50, the heavy-atom-count range of the sampled train_ood vs
test_ood (size-shortcut check) and per-seed test AUROC (consistency).
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, pickle, argparse, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from rdkit import Chem, RDLogger; RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

CACHE = Path("cache/ood_dpo_cache")
CELLS = {
    "ec50_assay":    ("lbap_general_ec50_assay_minimol_features.pkl",     "lbap_general_ec50_assay_seed42_splits.json"),
    "ec50_scaffold": ("lbap_general_ec50_scaffold_minimol_features.pkl",  "lbap_general_ec50_scaffold_seed42_splits.json"),
    "ic50_assay":    ("lbap_general_ic50_assay_minimol_features.pkl",     "lbap_general_ic50_assay_seed42_splits.json"),
    "zinc_scaffold": ("good_zinc_scaffold_covariate_minimol_features.pkl","good_zinc_scaffold_covariate_seed42_splits.json"),
}
RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]; LAM = 0.01


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
    d = {k: mat(sp[k]) for k in ["train_id", "train_ood", "val_id", "val_ood", "test_id", "test_ood"]}
    d["_ood_smiles"] = [s for s in sp["train_ood"] if s in fm]
    d["_test_ood_smiles"] = [s for s in sp["test_ood"] if s in fm]
    return d


def mets(s_id, s_ood):
    y = np.r_[np.zeros(len(s_id)), np.ones(len(s_ood))]; s = np.r_[s_id, s_ood]
    au = roc_auc_score(y, s); p, r, _ = precision_recall_curve(y, s); ap = auc(r, p)
    fpr, tpr, _ = roc_curve(y, s); i = np.searchsorted(tpr, 0.95); f95 = fpr[min(i, len(fpr) - 1)]
    return au, ap, f95


def rho_loss(rho, Eid, Eood):
    lg = torch.cat([Eid, Eood]); tg = torch.cat([torch.zeros(len(Eid)), torch.ones(len(Eood))])
    nat = F.binary_cross_entropy_with_logits(lg, tg)
    bal = 0.5 * F.binary_cross_entropy_with_logits(Eid, torch.zeros(len(Eid))) + \
          0.5 * F.binary_cross_entropy_with_logits(Eood, torch.ones(len(Eood)))
    return (1 - rho) * nat + rho * bal


def run(d, rho, k, seed, epochs=150, n_id=2000):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    perm = torch.randperm(len(d["train_ood"]), generator=g)[:k]
    Xid = d["train_id"][torch.randperm(len(d["train_id"]), generator=g)[:n_id]]
    Xood = d["train_ood"][perm]
    head = Head(); opt = torch.optim.AdamW(head.parameters(), 1e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9); best, bs = -1, None
    for ep in range(epochs):
        head.train(); opt.zero_grad(); Eid, Eood = head(Xid), head(Xood)
        loss = rho_loss(rho, Eid, Eood) + LAM * (Eid.pow(2).mean() + Eood.pow(2).mean())
        loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
        if (ep + 1) % 15 == 0:
            head.eval()
            with torch.no_grad(): v = roc_auc_score(np.r_[np.zeros(len(d["val_id"])), np.ones(len(d["val_ood"]))],
                                                    np.r_[head(d["val_id"]).numpy(), head(d["val_ood"]).numpy()])
            if v > best: best, bs = v, {kk: t.clone() for kk, t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval()
    with torch.no_grad():
        te = mets(head(d["test_id"]).numpy(), head(d["test_ood"]).numpy())
    return best, te, perm  # val-auroc, (test au/ap/f95), sampled ood indices


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budgets", nargs="+", type=int, default=[20, 50, 100, 200, 2000]); a = ap.parse_args()
    d = load(a.task); out = {}
    for k in a.budgets:
        for rho in RHOS:
            V, TE = [], []
            for s in range(1, a.seeds + 1):
                v, te, _ = run(d, rho, k, s); V.append(v); TE.append(te)
            out[f"{k}|{rho}"] = {"val_auroc": V, "test_auroc": [t[0] for t in TE],
                                 "test_aupr": [t[1] for t in TE], "test_fpr95": [t[2] for t in TE]}
        # val-selected rho vs fixed rho=0 / rho=1 (per seed, then mean)
        rows = []
        for s_i in range(a.seeds):
            vals = {rho: out[f"{k}|{rho}"]["val_auroc"][s_i] for rho in RHOS}
            rstar = max(vals, key=vals.get)
            rows.append((out[f"{k}|{rstar}"]["test_auroc"][s_i], out[f"{k}|0.0"]["test_auroc"][s_i], out[f"{k}|1.0"]["test_auroc"][s_i], rstar))
        tstar = np.mean([r[0] for r in rows]); t0 = np.mean([r[1] for r in rows]); t1 = np.mean([r[2] for r in rows])
        rstars = [r[3] for r in rows]
        print(f"[{a.task}] K={k:<4} test@rho*(val)={tstar:.3f}  test@rho=0={t0:.3f}  test@rho=1={t1:.3f}  chosen_rho={rstars}", flush=True)
    json.dump(out, open(f"repro/exposure_{a.task}.json", "w"), indent=2)
    # ZINC/size diagnostic at K=50
    if True:
        from rdkit.Chem import Descriptors
        def hac(smis): return [Chem.MolFromSmiles(x).GetNumHeavyAtoms() for x in smis if Chem.MolFromSmiles(x)]
        g = torch.Generator().manual_seed(1); perm = torch.randperm(len(d["train_ood"]), generator=g)[:50].tolist()
        sel = [d["_ood_smiles"][i] for i in perm]
        sh = hac(sel); th = hac(d["_test_ood_smiles"][:1000])
        print(f"[{a.task}] size-check: sampled train_ood heavy-atoms med={np.median(sh):.0f} [{np.percentile(sh,10):.0f}-{np.percentile(sh,90):.0f}] | test_ood med={np.median(th):.0f} [{np.percentile(th,10):.0f}-{np.percentile(th,90):.0f}]", flush=True)
    print(f"EXP_DONE {a.task}", flush=True)


if __name__ == "__main__":
    main()
