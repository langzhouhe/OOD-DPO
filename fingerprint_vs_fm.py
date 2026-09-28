#!/usr/bin/env python3
"""
Is the "frozen FM + OE head" gain from the FM's pretraining, or would trivial
features do it? (The exact diagnostic reviewers ynTJ/daed and Meta-Q5 asked for.)

Same OE head (dpo, best-val, matched), same splits, only the INPUT features
differ:
  size   : [num heavy atoms, MolWt]              (the "is it just molecular size?" test)
  morgan : ECFP4 2048-bit fingerprint
  desc   : ~10 RDKit physicochemical descriptors
  fm     : frozen MiniMol 512-d embedding        (cached)
Report TEST AUROC per feature type per cell.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, pickle, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from pathlib import Path
from sklearn.metrics import roc_auc_score
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

CACHE = Path("cache/ood_dpo_cache")
CELLS = {
    "ec50_size":     ("lbap_general_ec50_size_minimol_features.pkl",      "lbap_general_ec50_size_seed42_splits.json"),
    "ec50_scaffold": ("lbap_general_ec50_scaffold_minimol_features.pkl",  "lbap_general_ec50_scaffold_seed42_splits.json"),
    "ec50_assay":    ("lbap_general_ec50_assay_minimol_features.pkl",     "lbap_general_ec50_assay_seed42_splits.json"),
    "hiv_scaffold":  ("good_hiv_scaffold_covariate_minimol_features.pkl", "good_hiv_scaffold_covariate_seed42_splits.json"),
    "zinc_scaffold": ("good_zinc_scaffold_covariate_minimol_features.pkl","good_zinc_scaffold_covariate_seed42_splits.json"),
}
DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, Descriptors.NumHDonors,
         Descriptors.NumHAcceptors, Descriptors.NumRotatableBonds, rdMolDescriptors.CalcNumRings,
         Descriptors.HeavyAtomCount, Descriptors.FractionCSP3, rdMolDescriptors.CalcNumAromaticRings]


def featurize(smis, kind, fmfeat=None):
    out = []
    for s in smis:
        if kind == "fm":
            out.append(fmfeat.get(s)); continue   # None if FM couldn't encode it (dropped downstream)
        m = Chem.MolFromSmiles(s)
        if m is None:
            out.append(None); continue
        if kind == "size":
            out.append(np.array([m.GetNumHeavyAtoms(), Descriptors.MolWt(m)], dtype=np.float32))
        elif kind == "morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
            arr = np.zeros(2048, dtype=np.float32); from rdkit import DataStructs; DataStructs.ConvertToNumpyArray(fp, arr); out.append(arr)
        elif kind == "desc":
            out.append(np.array([f(m) for f in DESCS], dtype=np.float32))
    return out


class Head(nn.Module):
    def __init__(s, d):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(s, x): return s.net(x).squeeze(-1)


def auroc(a, b): return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])


def standardize(splits, mu, sd):
    return {k: (v - mu) / sd for k, v in splits.items()}


def train_eval(F, kind, seed=1, epochs=300):
    torch.manual_seed(seed); np.random.seed(seed)
    Xid = torch.tensor(F["train_id"][:2000]); Xood = torch.tensor(F["train_ood"][:2000])
    vid, vood = torch.tensor(F["val_id"]), torch.tensor(F["val_ood"])
    head = Head(Xid.shape[1]); opt = torch.optim.AdamW(head.parameters(), 1e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9); best, bs = -1, None
    for ep in range(epochs):
        head.train(); opt.zero_grad(); Eid, Eood = head(Xid), head(Xood)
        loss = F_.softplus(-0.1 * (Eood.unsqueeze(0) - Eid.unsqueeze(1))).mean() + 0.01 * (Eid.pow(2).mean() + Eood.pow(2).mean())
        loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
        if (ep + 1) % 15 == 0:
            head.eval()
            with torch.no_grad(): v = auroc(head(vid).numpy(), head(vood).numpy())
            if v > best: best, bs = v, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval()
    with torch.no_grad(): return auroc(head(torch.tensor(F["test_id"])).numpy(), head(torch.tensor(F["test_ood"])).numpy())


F_ = F
print(f"{'cell':14}{'size':>8}{'morgan':>8}{'desc':>8}{'FM':>8}")
print("-" * 46)
res = {}
for cell, (fc, sc) in CELLS.items():
    fmfeat = pickle.load(open(CACHE / fc, "rb"))["features"]
    sp = json.load(open(CACHE / sc, "rb"))["splits"]
    keys = ["train_id", "train_ood", "val_id", "val_ood", "test_id", "test_ood"]
    row = {}
    for kind in ["size", "morgan", "desc", "fm"]:
        feats = {}
        ok = True
        for k in keys:
            fv = featurize(sp[k], kind, fmfeat)
            pairs = [(s, x) for s, x in zip(sp[k], fv) if x is not None]
            feats[k] = np.stack([x for _, x in pairs]).astype(np.float32)
        # standardize (size/desc/fm) using train_id stats; morgan left as bits
        if kind in ("size", "desc", "fm"):
            allid = feats["train_id"]; mu = allid.mean(0); sd = allid.std(0) + 1e-6
            feats = {k: (v - mu) / sd for k, v in feats.items()}
        row[kind] = train_eval(feats, kind)
    res[cell] = row
    print(f"{cell:14}{row['size']:>8.3f}{row['morgan']:>8.3f}{row['desc']:>8.3f}{row['fm']:>8.3f}", flush=True)
json.dump(res, open("repro/fingerprint_vs_fm.json", "w"), indent=2)
print("FP_VS_FM_DONE")
