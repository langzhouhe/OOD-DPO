#!/usr/bin/env python3
"""
Raise the CEILING, not the loss: multi-view feature fusion.
Theory says RPO and balanced-BCE share the same Bayes-optimal ranking, so the
lever for absolute performance is the INFORMATION fed to the detector, not the
surrogate loss. Morgan fingerprints and MiniMol score near-identically yet are
different views (substructure bits vs pretrained continuous embedding), so they
may be complementary.

Views: fm (MiniMol 512) | morgan (ECFP4 2048) | desc (10 physchem)
       + concatenations fm+morgan, fm+desc, fm+morgan+desc
Detector: same MLP head; loss chosen per-view on VALIDATION (balanced-BCE or RPO).
Reports test AUROC/AUPR/FPR95; validation-only selection; 3 seeds.
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
VIEWS = ["fm", "morgan", "desc", "fm+morgan", "fm+desc", "fm+morgan+desc"]


class Head(nn.Module):
    def __init__(s, d):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(s, x): return s.net(x).squeeze(-1)


def featurize(cell):
    b = BASE[cell]
    fm = pickle.load(open(CACHE / f"{b}_minimol_features.pkl", "rb"))["features"]
    sp = json.load(open(CACHE / f"{b}_seed42_splits.json", "rb"))["splits"]
    out = {}
    for k in SPLITS:
        smis = [s for s in sp[k] if s in fm]
        F_fm, F_mo, F_de = [], [], []
        for s in smis:
            m = Chem.MolFromSmiles(s)
            if m is None: continue
            arr = np.zeros(2048, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), arr)
            F_fm.append(fm[s]); F_mo.append(arr)
            F_de.append(np.array([f(m) for f in DESCS], dtype=np.float32))
        out[k] = {"fm": np.stack(F_fm).astype(np.float32), "morgan": np.stack(F_mo),
                  "desc": np.stack(F_de).astype(np.float32)}
    return out


def make_view(data, view, stats=None):
    parts = view.split("+")
    blocks, st = [], {}
    for p in parts:
        X = data[p]
        if p in ("fm", "desc"):
            mu, sd = (stats[p] if stats else (X.mean(0), X.std(0) + 1e-6))
            X = (X - mu) / sd; st[p] = (mu, sd)
        blocks.append(X)
    return np.concatenate(blocks, 1).astype(np.float32), st


def mets(head, a, b):
    with torch.no_grad(): s = np.r_[head(a).numpy(), head(b).numpy()]
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    p, r, _ = precision_recall_curve(y, s); fpr, tpr, _ = roc_curve(y, s); i = np.searchsorted(tpr, 0.95)
    return roc_auc_score(y, s), auc(r, p), fpr[min(i, len(fpr) - 1)]


def train(V, loss_kind, seed, epochs=120, n_id=1500):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    Xid = torch.tensor(V["train_id"])[torch.randperm(len(V["train_id"]), generator=g)[:n_id]]
    Xood = torch.tensor(V["train_ood"])[torch.randperm(len(V["train_ood"]), generator=g)[:2000]]
    vid, vood = torch.tensor(V["val_id"]), torch.tensor(V["val_ood"])
    lr = 3e-4 if loss_kind == "bce" else 1e-4
    head = Head(Xid.shape[1]); opt = torch.optim.AdamW(head.parameters(), lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9); best, bs = -1, None
    for ep in range(epochs):
        head.train(); opt.zero_grad(); Eid, Eood = head(Xid), head(Xood)
        if loss_kind == "bce":
            core = 0.5 * F.binary_cross_entropy_with_logits(Eid, torch.zeros(len(Eid))) + \
                   0.5 * F.binary_cross_entropy_with_logits(Eood, torch.ones(len(Eood))); lam = 0.1
        else:
            idx = torch.randint(len(Eid), (len(Eood),), generator=g)
            core = F.softplus(-10.0 * (Eood - Eid[idx])).mean(); lam = 0.5
        (core + lam * (Eid.pow(2).mean() + Eood.pow(2).mean())).backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
        if (ep + 1) % 15 == 0:
            head.eval()
            with torch.no_grad(): v = roc_auc_score(np.r_[np.zeros(len(vid)), np.ones(len(vood))],
                                                    np.r_[head(vid).numpy(), head(vood).numpy()])
            if v > best: best, bs = v, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval()
    return best, mets(head, torch.tensor(V["test_id"]), torch.tensor(V["test_ood"]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cell", required=True); a = ap.parse_args()
    data = featurize(a.cell); out = {}
    for view in VIEWS:
        _, st = make_view(data["train_id"], view)           # stats from train_id only
        V = {k: make_view(data[k], view, st)[0] for k in SPLITS}
        best = None
        for lk in ["bce", "rpo"]:
            r = [train(V, lk, s) for s in (1, 2, 3)]
            val = float(np.mean([x[0] for x in r]))
            rec = {"loss": lk, "val": val, "auroc": float(np.mean([x[1][0] for x in r])),
                   "aupr": float(np.mean([x[1][1] for x in r])), "fpr95": float(np.mean([x[1][2] for x in r]))}
            if best is None or val > best["val"]: best = rec
        out[view] = best
        print(f"[{a.cell}] {view:16} loss={best['loss']:4} AUROC={best['auroc']:.3f} AUPR={best['aupr']:.3f} FPR95={best['fpr95']:.3f}", flush=True)
    bv = max(out, key=lambda k: out[k]["val"])
    print(f"[{a.cell}] >>> val-selected view = {bv} : AUROC={out[bv]['auroc']:.3f} (fm-only {out['fm']['auroc']:.3f}, Δ={out[bv]['auroc']-out['fm']['auroc']:+.3f})", flush=True)
    json.dump(out, open(f"repro/fusion_{a.cell}.json", "w"), indent=2)
    print(f"FUSION_DONE {a.cell}", flush=True)


if __name__ == "__main__":
    main()
