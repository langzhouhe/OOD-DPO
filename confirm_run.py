#!/usr/bin/env python3
"""
Independent confirmation of the 4/4 signal, on 5 FRESH seeds (11-15) that were
NOT used during method development.

Fixed input : MiniMol + Morgan + descriptors (single candidate, no feature search)
Fixed hp    : per-cell, validation-selected during development (not re-tuned here)
Compared    : RPO (pairwise logistic) vs balanced BCE, fully matched --
              same data subsets per seed (paired), same fusion head/params,
              same epochs/optimizer/early-stopping.
Reports per-seed paired Delta AUROC, mean+/-std, 95% CI, plus AUPR and FPR95.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, pickle, argparse, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

CACHE = Path("cache/ood_dpo_cache")
BASE = {"ec50_assay": "lbap_general_ec50_assay", "ic50_assay": "lbap_general_ic50_assay",
        "ec50_scaffold": "lbap_general_ec50_scaffold", "ic50_scaffold": "lbap_general_ic50_scaffold"}
HP = {"ec50_assay": ((10.0, 0.5), (0.1, 3e-4)), "ic50_assay": ((1.0, 0.1), (0.1, 3e-4)),
      "ec50_scaffold": ((10.0, 0.1), (0.1, 3e-4)), "ic50_scaffold": ((10.0, 0.0), (0.1, 3e-4))}
SPLITS = ["train_id", "train_ood", "val_id", "val_ood", "test_id", "test_ood"]
DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, Descriptors.NumHDonors,
         Descriptors.NumHAcceptors, Descriptors.NumRotatableBonds, rdMolDescriptors.CalcNumRings,
         Descriptors.HeavyAtomCount, Descriptors.FractionCSP3, rdMolDescriptors.CalcNumAromaticRings]
BLOCKS = ["fm", "morgan", "desc"]; PROJ = 256
SEEDS = [11, 12, 13, 14, 15]          # fresh, unused in development


class FusionHead(nn.Module):
    def __init__(s, dims):
        super().__init__()
        s.proj = nn.ModuleList([nn.Linear(d, PROJ) for d in dims])
        h = PROJ * len(dims)
        s.net = nn.Sequential(nn.Linear(h, 256), nn.ReLU(), nn.Dropout(0.1),
                              nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(s, bl): return s.net(torch.cat([p(b) for p, b in zip(s.proj, bl)], 1)).squeeze(-1)


def featurize(cell):
    b = BASE[cell]
    fm = pickle.load(open(CACHE / f"{b}_minimol_features.pkl", "rb"))["features"]
    sp = json.load(open(CACHE / f"{b}_seed42_splits.json", "rb"))["splits"]
    out = {}
    for k in SPLITS:
        A, B, C = [], [], []
        for s in sp[k]:
            if s not in fm: continue
            m = Chem.MolFromSmiles(s)
            if m is None: continue
            arr = np.zeros(2048, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), arr)
            A.append(fm[s]); B.append(arr); C.append(np.array([f(m) for f in DESCS], dtype=np.float32))
        out[k] = {"fm": np.stack(A).astype(np.float32), "morgan": np.stack(B), "desc": np.stack(C).astype(np.float32)}
    return out


def mets(head, A, B):
    with torch.no_grad(): s = np.r_[head(A).numpy(), head(B).numpy()]
    y = np.r_[np.zeros(len(A[0])), np.ones(len(B[0]))]
    p, r, _ = precision_recall_curve(y, s); fpr, tpr, _ = roc_curve(y, s); i = np.searchsorted(tpr, 0.95)
    return roc_auc_score(y, s), auc(r, p), fpr[min(i, len(fpr) - 1)]


def train(V, kind, hp, seed, epochs=120, n_id=1500):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    dims = [V["train_id"][i].shape[1] for i in range(3)]
    iid = torch.randperm(len(V["train_id"][0]), generator=g)[:n_id]
    iod = torch.randperm(len(V["train_ood"][0]), generator=g)[:2000]
    Xid = [b[iid] for b in V["train_id"]]; Xood = [b[iod] for b in V["train_ood"]]
    lr = 1e-4 if kind == "rpo" else hp[1]
    head = FusionHead(dims); opt = torch.optim.AdamW(head.parameters(), lr, weight_decay=1e-5)
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
            with torch.no_grad():
                v = roc_auc_score(np.r_[np.zeros(len(V["val_id"][0])), np.ones(len(V["val_ood"][0]))],
                                  np.r_[head(V["val_id"]).numpy(), head(V["val_ood"]).numpy()])
            if v > best: best, bs = v, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval()
    return mets(head, V["test_id"], V["test_ood"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cell", required=True); a = ap.parse_args()
    data = featurize(a.cell)
    st = {p: (data["train_id"][p].mean(0), data["train_id"][p].std(0) + 1e-6) for p in BLOCKS}
    V = {k: [torch.tensor((data[k][p] - st[p][0]) / st[p][1]) for p in BLOCKS] for k in SPLITS}
    rhp, bhp = HP[a.cell]
    R = [train(V, "rpo", rhp, s) for s in SEEDS]
    B = [train(V, "bce", bhp, s) for s in SEEDS]
    ra = np.array([x[0] for x in R]); ba = np.array([x[0] for x in B]); dd = ra - ba
    ci = 2.776 * dd.std(ddof=1) / np.sqrt(len(dd))
    res = {"seeds": SEEDS, "rpo_auroc": ra.tolist(), "bce_auroc": ba.tolist(),
           "delta": dd.tolist(), "mean": float(dd.mean()), "ci95": float(ci),
           "rpo_aupr": float(np.mean([x[1] for x in R])), "bce_aupr": float(np.mean([x[1] for x in B])),
           "rpo_fpr95": float(np.mean([x[2] for x in R])), "bce_fpr95": float(np.mean([x[2] for x in B]))}
    json.dump(res, open(f"repro/confirm_{a.cell}.json", "w"), indent=2)
    print(f"[{a.cell}] RPO={ra.mean():.4f}±{ra.std(ddof=1):.4f} BCE={ba.mean():.4f}±{ba.std(ddof=1):.4f} "
          f"Δ={dd.mean():+.4f} [95%CI ±{ci:.4f}] per-seed Δ={[round(x,4) for x in dd]} "
          f"| AUPR {res['rpo_aupr']:.3f}/{res['bce_aupr']:.3f} | FPR95 {res['rpo_fpr95']:.3f}/{res['bce_fpr95']:.3f}", flush=True)
    print(f"CONFIRM_DONE {a.cell}", flush=True)


if __name__ == "__main__":
    main()
