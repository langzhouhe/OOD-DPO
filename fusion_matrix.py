#!/usr/bin/env python3
"""
The decisive matrix: does fusion give RPO an advantage OVER balanced BCE, or
does it merely raise a ceiling both share?

4 feature inputs x 2 objectives, everything else matched:
  inputs : fm | morgan | fm+morgan | fm+morgan+desc
  losses : balanced BCE | RPO (pairwise logistic)
Dimension confound removed: EVERY view is per-block standardized and linearly
projected to 256-d, then concatenated -> identical scoring head input size and
parameter count for both objectives and across views.
3 seeds; validation-only selection of loss hyperparameters; reports AUROC/AUPR/FPR95.
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
        "ec50_scaffold": "lbap_general_ec50_scaffold", "ic50_scaffold": "lbap_general_ic50_scaffold",
        "hiv_scaffold": "good_hiv_scaffold_covariate", "pcba_scaffold": "good_pcba_scaffold_covariate"}
SPLITS = ["train_id", "train_ood", "val_id", "val_ood", "test_id", "test_ood"]
DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, Descriptors.NumHDonors,
         Descriptors.NumHAcceptors, Descriptors.NumRotatableBonds, rdMolDescriptors.CalcNumRings,
         Descriptors.HeavyAtomCount, Descriptors.FractionCSP3, rdMolDescriptors.CalcNumAromaticRings]
VIEWS = ["fm", "morgan", "fm+morgan", "fm+morgan+desc"]
PROJ = 256


class FusionHead(nn.Module):
    """Per-block linear projection to PROJ dims, concat, then the standard MLP head.
    Identical architecture for both objectives => no dimension/parameter confound."""
    def __init__(s, dims):
        super().__init__()
        s.proj = nn.ModuleList([nn.Linear(d, PROJ) for d in dims])
        h = PROJ * len(dims)
        s.net = nn.Sequential(nn.Linear(h, 256), nn.ReLU(), nn.Dropout(0.1),
                              nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(s, blocks):
        z = torch.cat([p(b) for p, b in zip(s.proj, blocks)], 1)
        return s.net(z).squeeze(-1)


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
        out[k] = {"fm": np.stack(A).astype(np.float32), "morgan": np.stack(B),
                  "desc": np.stack(C).astype(np.float32)}
    return out


def blocks_for(data, view, stats):
    return [torch.tensor((data[p] - stats[p][0]) / stats[p][1]) for p in view.split("+")]


def mets(head, A, B):
    with torch.no_grad(): s = np.r_[head(A).numpy(), head(B).numpy()]
    y = np.r_[np.zeros(len(A[0])), np.ones(len(B[0]))]
    p, r, _ = precision_recall_curve(y, s); fpr, tpr, _ = roc_curve(y, s); i = np.searchsorted(tpr, 0.95)
    return roc_auc_score(y, s), auc(r, p), fpr[min(i, len(fpr) - 1)]


def sub(blocks, idx): return [b[idx] for b in blocks]


def train(V, view, kind, hp, seed, epochs=120, n_id=1500):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    dims = [V["train_id"][i].shape[1] for i in range(len(view.split("+")))]
    tid, tood = V["train_id"], V["train_ood"]
    iid = torch.randperm(len(tid[0]), generator=g)[:n_id]; iod = torch.randperm(len(tood[0]), generator=g)[:2000]
    Xid, Xood = sub(tid, iid), sub(tood, iod)
    lr = hp[1] if kind == "bce" else 1e-4
    head = FusionHead(dims); opt = torch.optim.AdamW(head.parameters(), lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9); best, bs = -1, None
    for ep in range(epochs):
        head.train(); opt.zero_grad(); Eid, Eood = head(Xid), head(Xood)
        if kind == "bce":
            lam = hp[0]
            core = 0.5 * F.binary_cross_entropy_with_logits(Eid, torch.zeros(len(Eid))) + \
                   0.5 * F.binary_cross_entropy_with_logits(Eood, torch.ones(len(Eood)))
        else:
            beta, lam = hp
            idx = torch.randint(len(Eid), (len(Eood),), generator=g)
            core = F.softplus(-beta * (Eood - Eid[idx])).mean()
        (core + lam * (Eid.pow(2).mean() + Eood.pow(2).mean())).backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
        if (ep + 1) % 15 == 0:
            head.eval()
            with torch.no_grad():
                v = roc_auc_score(np.r_[np.zeros(len(V["val_id"][0])), np.ones(len(V["val_ood"][0]))],
                                  np.r_[head(V["val_id"]).numpy(), head(V["val_ood"]).numpy()])
            if v > best: best, bs = v, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval()
    return best, mets(head, V["test_id"], V["test_ood"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cell", required=True); a = ap.parse_args()
    data = featurize(a.cell)
    stats = {p: (data["train_id"][p].mean(0), data["train_id"][p].std(0) + 1e-6) for p in ["fm", "morgan", "desc"]}
    out = {}
    print(f"{'view':17}{'loss':5}{'AUROC':>8}{'AUPR':>7}{'FPR95':>7}", flush=True)
    for view in VIEWS:
        V = {k: blocks_for(data[k], view, stats) for k in SPLITS}
        for kind, grid in [("bce", [(l, lr) for l in [0.0, 0.01, 0.1] for lr in [1e-4, 3e-4]]),
                           ("rpo", [(b, l) for b in [1., 5., 10.] for l in [0.0, 0.1, 0.5]])]:
            best = None
            for hp in grid:
                r = [train(V, view, kind, hp, s) for s in (1, 2, 3)]
                val = float(np.mean([x[0] for x in r]))
                rec = {"hp": str(hp), "val": val, "auroc": float(np.mean([x[1][0] for x in r])),
                       "aupr": float(np.mean([x[1][1] for x in r])), "fpr95": float(np.mean([x[1][2] for x in r]))}
                if best is None or val > best["val"]: best = rec
            out[f"{view}|{kind}"] = best
            print(f"{view:17}{kind:5}{best['auroc']:>8.3f}{best['aupr']:>7.3f}{best['fpr95']:>7.3f}   hp={best['hp']}", flush=True)
    for view in VIEWS:
        d = out[f"{view}|rpo"]["auroc"] - out[f"{view}|bce"]["auroc"]
        print(f">>> [{a.cell}] {view:17} RPO-BCE = {d:+.3f}", flush=True)
    json.dump(out, open(f"repro/fmatrix_{a.cell}.json", "w"), indent=2)
    print(f"FMATRIX_DONE {a.cell}", flush=True)


if __name__ == "__main__":
    main()
