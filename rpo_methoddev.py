#!/usr/bin/env python3
"""RPO method development: can RPO-specific structure beat matched balanced-BCE?
Two mechanisms BCE cannot express:
  (1) hard-pair mining: optimize only the top-q% hardest pairwise losses;
  (2) local+random pairing: pair each OOD with its k NEAREST ID (hard, boundary)
      plus k RANDOM ID (global ordering) -- 'this OOD must rank behind these
      SIMILAR ID molecules'.
Grid over pair/q/beta on 4 main DrugOOD cells; validation-only selection;
report AUROC/AUPR/FPR95 vs balanced-BCE. FPR95 is where hard mining likely helps.
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
BASE = {"ec50_assay": "lbap_general_ec50_assay", "ic50_assay": "lbap_general_ic50_assay",
        "ec50_scaffold": "lbap_general_ec50_scaffold", "ic50_scaffold": "lbap_general_ic50_scaffold"}

# config list: balanced-BCE reference + RPO variants
CONFIGS = [("bce", None, None, None)]
for pair in ["all"]:
    for beta in [5., 10.]:
        CONFIGS.append(("rpo", pair, 100, beta))              # original all-pairs RPO
for pair in ["local32", "local128"]:
    for q in [25, 50, 100]:
        for beta in [5., 10.]:
            CONFIGS.append(("rpo", pair, q, beta))            # hard-mined local+random RPO
LAM_RPO, LR_RPO = 0.5, 1e-4
LAM_BCE, LR_BCE = 0.1, 3e-4


class Head(nn.Module):
    def __init__(s, d=512):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(s, x): return s.net(x).squeeze(-1)


def load(cell):
    b = BASE[cell]; fm = pickle.load(open(CACHE / f"{b}_minimol_features.pkl", "rb"))["features"]
    sp = json.load(open(CACHE / f"{b}_seed42_splits.json", "rb"))["splits"]
    def mat(s): return torch.tensor(np.stack([fm[x] for x in s if x in fm]).astype(np.float32))
    return {k: mat(sp[k]) for k in ["train_id", "train_ood", "val_id", "val_ood", "test_id", "test_ood"]}


def mets(head, a, b):
    with torch.no_grad(): s = np.r_[head(a).numpy(), head(b).numpy()]
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    p, r, _ = precision_recall_curve(y, s); fpr, tpr, _ = roc_curve(y, s); i = np.searchsorted(tpr, 0.95)
    return roc_auc_score(y, s), auc(r, p), fpr[min(i, len(fpr) - 1)]


def build_pairs(Xid, Xood, pair, g):
    n_id, n_ood = len(Xid), len(Xood)
    if pair == "all":
        oo = torch.arange(n_ood).repeat_interleave(n_id); ii = torch.arange(n_id).repeat(n_ood)
    else:
        k = int(pair.replace("local", ""))
        with torch.no_grad(): D = torch.cdist(Xood, Xid)          # frozen features -> fixed
        nn_idx = D.topk(k, largest=False).indices                 # k nearest ID per OOD (hard)
        rd_idx = torch.randint(n_id, (n_ood, k), generator=g)     # k random ID (global)
        cat = torch.cat([nn_idx, rd_idx], 1)
        oo = torch.arange(n_ood).repeat_interleave(2 * k); ii = cat.reshape(-1)
    return oo, ii


def train(d, cfg, seed, epochs=120, n_id=1500):
    kind, pair, q, beta = cfg
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    Xid = d["train_id"][torch.randperm(len(d["train_id"]), generator=g)[:n_id]]
    Xood = d["train_ood"][torch.randperm(len(d["train_ood"]), generator=g)[:2000]]
    if kind == "rpo": oo, ii = build_pairs(Xid, Xood, pair, g)
    lr = LR_RPO if kind == "rpo" else LR_BCE
    head = Head(); opt = torch.optim.AdamW(head.parameters(), lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9); best, bs = -1, None
    for ep in range(epochs):
        head.train(); opt.zero_grad(); Eid, Eood = head(Xid), head(Xood)
        if kind == "rpo":
            per = F.softplus(-beta * (Eood[oo] - Eid[ii]))
            if q < 100:
                kk = max(1, int(q / 100 * per.numel())); per = torch.topk(per, kk).values
            core = per.mean(); lam = LAM_RPO
        else:
            core = 0.5 * F.binary_cross_entropy_with_logits(Eid, torch.zeros(len(Eid))) + \
                   0.5 * F.binary_cross_entropy_with_logits(Eood, torch.ones(len(Eood))); lam = LAM_BCE
        (core + lam * (Eid.pow(2).mean() + Eood.pow(2).mean())).backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
        if (ep + 1) % 15 == 0:
            head.eval()
            with torch.no_grad(): v = roc_auc_score(np.r_[np.zeros(len(d["val_id"])), np.ones(len(d["val_ood"]))],
                                                    np.r_[head(d["val_id"]).numpy(), head(d["val_ood"]).numpy()])
            if v > best: best, bs = v, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval()
    return best, mets(head, d["test_id"], d["test_ood"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cell", required=True); a = ap.parse_args()
    d = load(a.cell); out = {}
    for cfg in CONFIGS:
        r = [train(d, cfg, s) for s in (1, 2, 3)]
        name = "bce" if cfg[0] == "bce" else f"rpo_{cfg[1]}_q{cfg[2]}_b{int(cfg[3])}"
        out[name] = {"val": float(np.mean([x[0] for x in r])),
                     "auroc": float(np.mean([x[1][0] for x in r])), "aupr": float(np.mean([x[1][1] for x in r])),
                     "fpr95": float(np.mean([x[1][2] for x in r]))}
    bce = out["bce"]
    rpos = {k: v for k, v in out.items() if k != "bce"}
    best_au = max(rpos, key=lambda k: rpos[k]["val"])              # val-selected best RPO
    best_fpr = min(rpos, key=lambda k: rpos[k]["fpr95"])
    print(f"[{a.cell}] BCE au={bce['auroc']:.3f} fpr95={bce['fpr95']:.3f} | "
          f"bestRPO(val)={best_au} au={rpos[best_au]['auroc']:.3f}(Δ{rpos[best_au]['auroc']-bce['auroc']:+.3f}) fpr95={rpos[best_au]['fpr95']:.3f}(Δ{rpos[best_au]['fpr95']-bce['fpr95']:+.3f}) | "
          f"minFPR-RPO={best_fpr} fpr95={rpos[best_fpr]['fpr95']:.3f}(Δ{rpos[best_fpr]['fpr95']-bce['fpr95']:+.3f})", flush=True)
    json.dump(out, open(f"repro/methoddev_{a.cell}.json", "w"), indent=2)
    print(f"MDEV_DONE {a.cell}", flush=True)


if __name__ == "__main__":
    main()
