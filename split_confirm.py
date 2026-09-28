#!/usr/bin/env python3
"""
Split-level confirmation: does the multi-view RPO > balanced-BCE advantage
survive BENCHMARK RESAMPLING (new data-split seeds), not just new init seeds?

EVERYTHING IS FROZEN (no selection here):
  view      : MiniMol + Morgan + descriptors
  head      : per-block Linear->256, concat, MLP 256->128->1 (identical both losses)
  hp        : per-cell RPO (beta,lambda) and BCE (lambda,lr) fixed from earlier
              validation selection; NOT re-tuned
  protocol  : epochs 120, AdamW, StepLR(10,0.9), clip 1.0, best-val checkpoint
  init seed : single fixed value (1) per split -- training randomness already
              checked with seeds 11-15
Varies ONLY data_seed in {43,44,45,46,47}, replicating data_loader.py's
deterministic subsampling (train/val/test subsets ALL change).
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, pickle, random, argparse, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

CACHE = Path("cache/ood_dpo_cache")
RAW = {"ec50_assay": "lbap_general_ec50_assay", "ic50_assay": "lbap_general_ic50_assay",
       "ec50_scaffold": "lbap_general_ec50_scaffold", "ic50_scaffold": "lbap_general_ic50_scaffold"}
HP = {"ec50_assay": ((10.0, 0.5), (0.1, 3e-4)), "ic50_assay": ((1.0, 0.1), (0.1, 3e-4)),
      "ec50_scaffold": ((10.0, 0.1), (0.1, 3e-4)), "ic50_scaffold": ((10.0, 0.0), (0.1, 3e-4))}
KEYS = ["train_id", "train_ood", "val_id", "val_ood", "test_id", "test_ood"]
TARGETS = {"train_id": 2000, "train_ood": 2000, "val_id": 600, "val_ood": 600, "test_id": 1000, "test_ood": 1000}
OFFSETS = {"train_id": 0, "train_ood": 1, "val_id": 2, "val_ood": 3, "test_id": 4, "test_ood": 5}
DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, Descriptors.NumHDonors,
         Descriptors.NumHAcceptors, Descriptors.NumRotatableBonds, rdMolDescriptors.CalcNumRings,
         Descriptors.HeavyAtomCount, Descriptors.FractionCSP3, rdMolDescriptors.CalcNumAromaticRings]
DATA_SEEDS = [43, 44, 45, 46, 47]; INIT_SEED = 1; PROJ = 256


def build_split(cell, data_seed):
    """Replicates utils.process_drugood_data + data_loader._select_final_smiles."""
    d = json.load(open(f"data/raw/{RAW[cell]}.json"))["split"]
    g = lambda k: [it["smiles"] for it in d.get(k, []) if it.get("smiles")]
    train_id, val_id, test_id = g("train"), g("iid_val"), g("iid_test")
    train_ood_all, test_ood = g("ood_val"), g("ood_test")
    tro = list(train_ood_all); random.Random(data_seed).shuffle(tro)     # val_ood split off train_ood
    vs = min(3000, len(tro) // 5); val_ood, train_ood = tro[:vs], tro[vs:]
    raw = {"train_id": train_id, "train_ood": train_ood, "val_id": val_id,
           "val_ood": val_ood, "test_id": test_id, "test_ood": test_ood}
    out = {}
    for k in KEYS:
        n = min(TARGETS[k], len(raw[k]))
        rng = np.random.default_rng(abs(data_seed + OFFSETS[k]) % (2 ** 31))
        out[k] = [str(x) for x in rng.choice(raw[k], size=n, replace=False)]
    return out


def ensure_features(cell, needed):
    """Load cached MiniMol features; encode any missing molecules and persist."""
    path = CACHE / f"{RAW[cell]}_minimol_features.pkl"
    obj = pickle.load(open(path, "rb")); feats = obj["features"]
    missing = sorted({s for s in needed if s not in feats})
    if missing:
        print(f"[{cell}] encoding {len(missing)} new molecules with MiniMol...", flush=True)
        from model import MinimolEncoder
        enc = MinimolEncoder()
        B = 500
        for i in range(0, len(missing), B):
            chunk = missing[i:i + B]
            try:
                out = enc.encode_smiles(chunk)
                for s, f in zip(chunk, out): feats[s] = f.detach().cpu().numpy().astype(np.float32)
            except Exception:
                for s in chunk:
                    try: feats[s] = enc.encode_smiles([s])[0].detach().cpu().numpy().astype(np.float32)
                    except Exception: pass
            if (i // B) % 4 == 0: print(f"[{cell}]   {min(i+B,len(missing))}/{len(missing)}", flush=True)
        obj["features"] = feats; pickle.dump(obj, open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    return feats


def blocks(smis, feats):
    A, B, C = [], [], []
    for s in smis:
        if s not in feats: continue
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        arr = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), arr)
        A.append(feats[s]); B.append(arr); C.append(np.array([f(m) for f in DESCS], dtype=np.float32))
    return {"fm": np.stack(A).astype(np.float32), "morgan": np.stack(B), "desc": np.stack(C).astype(np.float32)}


class FusionHead(nn.Module):
    def __init__(s, dims):
        super().__init__()
        s.proj = nn.ModuleList([nn.Linear(d, PROJ) for d in dims])
        s.net = nn.Sequential(nn.Linear(PROJ * len(dims), 256), nn.ReLU(), nn.Dropout(0.1),
                              nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(s, bl): return s.net(torch.cat([p(b) for p, b in zip(s.proj, bl)], 1)).squeeze(-1)


def mets(head, A, B):
    with torch.no_grad(): s = np.r_[head(A).numpy(), head(B).numpy()]
    y = np.r_[np.zeros(len(A[0])), np.ones(len(B[0]))]
    p, r, _ = precision_recall_curve(y, s); fpr, tpr, _ = roc_curve(y, s); i = np.searchsorted(tpr, 0.95)
    return roc_auc_score(y, s), auc(r, p), fpr[min(i, len(fpr) - 1)]


def train(V, kind, hp, seed=INIT_SEED, epochs=120, n_id=1500):
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
    splits = {ds: build_split(a.cell, ds) for ds in DATA_SEEDS}
    need = {s for sp in splits.values() for k in KEYS for s in sp[k]}
    feats = ensure_features(a.cell, need)
    rhp, bhp = HP[a.cell]; out = {}
    for ds in DATA_SEEDS:
        data = {k: blocks(splits[ds][k], feats) for k in KEYS}
        st = {p: (data["train_id"][p].mean(0), data["train_id"][p].std(0) + 1e-6) for p in ["fm", "morgan", "desc"]}
        V = {k: [torch.tensor((data[k][p] - st[p][0]) / st[p][1]) for p in ["fm", "morgan", "desc"]] for k in KEYS}
        r = train(V, "rpo", rhp); b = train(V, "bce", bhp)
        out[ds] = {"rpo": r, "bce": b, "delta": r[0] - b[0]}
        print(f"[{a.cell}] data_seed={ds} RPO={r[0]:.4f} BCE={b[0]:.4f} Δ={r[0]-b[0]:+.4f} "
              f"| AUPR {r[1]:.3f}/{b[1]:.3f} | FPR95 {r[2]:.3f}/{b[2]:.3f}", flush=True)
    dd = np.array([out[ds]["delta"] for ds in DATA_SEEDS])
    print(f"[{a.cell}] SPLIT-LEVEL: mean Δ={dd.mean():+.4f} ±{2.776*dd.std(ddof=1)/np.sqrt(5):.4f} "
          f"| positive on {int((dd>0).sum())}/5 splits", flush=True)
    json.dump({str(k): {"rpo": list(v["rpo"]), "bce": list(v["bce"]), "delta": v["delta"]} for k, v in out.items()},
              open(f"repro/splitconf_{a.cell}.json", "w"), indent=2)
    print(f"SPLITCONF_DONE {a.cell}", flush=True)


if __name__ == "__main__":
    main()
