#!/usr/bin/env python3
"""Matched outlier-exposure comparison: every baseline gets the same auxiliary OOD.

The reviewers' objection to Table 1 is an information asymmetry, not a loss-function
question: RPO trains on auxiliary OOD molecules and the six post-hoc baselines do not, so
the reported margin is mostly "OE vs no OE".  This script removes the asymmetry by giving
every baseline an outlier-exposed counterpart trained on the SAME molecules, then reports
both halves in one table.

Fourteen methods, all on the identical frozen features, the identical ID/OOD subsets, the
identical domain-disjoint validation split, and exactly NINE validation trials each:

  no OE (as in the original Table 1)      matched OE
  --------------------------------       ----------------------------------------------
  MSP                                     MSP-OE      uniform-output loss on aux OOD
  ODIN        temperature only            ODIN-OE     same temperature on the OE classifier
  Energy                                  Energy-OE   Liu et al. energy margin on aux OOD
  Mahalanobis ID Gaussian                 OE-Mahalanobis  two-sample Gaussian log-LR
  KNN         mean kNN distance to ID     OE-KNN      distance to ID minus distance to OOD
  LOF         ID local density            OE-LOF      ID minus OOD local-density outlierness
                                          Balanced OOD Head   class-balanced BCE
                                          RPO         pairwise logistic (the paper's loss)

The three "OE-" distance methods are NOT standard named methods in the literature; they are
two-sample adaptations built here so the distance baselines get the same information.  They
must be described that way in the paper.

Protocol is inherited from revision_run.py (Phase 0) by import, not by copy: the same
domain-disjoint split, the same parameter-matched head, the same conventional FPR95, the
same validation-only selection, the same 3 selection seeds and 5 final seeds.

Usage: python matched_oe.py --cell ec50_assay --backbone minimol
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, pickle, argparse, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from pathlib import Path
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import revision_run as RR

# Ki-Assay is a DrugOOD cell that RR.BASE never listed because it had no split cache.
# build_ki_splits.py now produces one whose construction was verified element-for-element
# against an existing cell, so register it here for every downstream script.
RR.BASE.setdefault("ki_assay", "lbap_general_ki_assay")
RR.DRUGOOD.add("ki_assay")

N_ID, N_OOD = 1500, 2000            # identical to RR.train, so every method sees the
SEL, FIN = RR.SEL, RR.FIN           # same molecules at the same seed
CLF_EPOCHS = 300
GOOD_LABELS = {"hiv": "hiv", "pcba": "pcba", "zinc": "zinc"}

# ---------------- nine-trial grids (one per method, no method gets more) ----------------
G_CLF    = [(lr, wd) for lr in (3e-3, 1e-2, 3e-2) for wd in (5e-4, 5e-3, 5e-2)]
G_T      = [1., 2., 5., 10., 20., 50., 100., 200., 1000.]
G_OECLF  = [(lam, lr) for lam in (0.1, 0.5, 1.0) for lr in (3e-3, 1e-2, 3e-2)]
G_ENOE   = [(lam, m) for lam in (0.03, 0.1, 0.3)
            for m in ((-7., -5.), (-5., -3.), (-3., -1.))]
G_MAHA   = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
G_KNN    = [1, 2, 5, 10, 20, 30, 50, 75, 100]
G_LOF    = [5, 10, 20, 30, 50, 75, 100, 150, 200]
G_MAHAOE = [(e, m) for e in (1e-3, 1e-2, 1e-1) for m in ("shared", "sep", "diag")]
G_KNNOE  = [(a, b) for a in (5, 20, 50) for b in (5, 20, 50)]
G_LOFOE  = [(a, b) for a in (10, 20, 50) for b in (10, 20, 50)]
CLF_DEF, OECLF_DEF = (1e-2, 5e-4), (0.5, 1e-2)     # fixed classifier for the ODIN arms


# ---------------------------------- data ----------------------------------
def load_cell(cell, backbone):
    """Frozen features + row-aligned SMILES, on RR's cached splits with RR's
    domain-disjoint train-OOD / val-OOD re-partition."""
    b = RR.BASE[cell]
    feat = pickle.load(open(RR.CACHE / f"{b}_{backbone}_features.pkl", "rb"))["features"]
    sp = json.load(open(RR.CACHE / f"{b}_seed42_splits.json"))["splits"]
    sp, info = RR.domain_disjoint_split(cell, sp)
    out = {}
    for k in RR.KEYS:
        smis = [s for s in sp[k] if s in feat]
        out[k] = (np.stack([feat[s] for s in smis]).astype(np.float32), smis)
    mu = out["train_id"][0].mean(0); sd = out["train_id"][0].std(0) + 1e-6
    return {k: ((v[0] - mu) / sd, v[1]) for k, v in out.items()}, info


def label_map(cell):
    if cell in RR.DRUGOOD:
        raw = json.load(open(f"data/raw/{RR.BASE[cell]}.json"))["split"]
        lab = {}
        for part in ("train", "iid_val", "iid_test"):
            for it in raw.get(part, []):
                if it.get("smiles") is not None and it.get("cls_label") is not None:
                    lab.setdefault(it["smiles"], int(it["cls_label"]))
        return lab
    tag = cell.split("_")[0]
    f = Path(f"cache/good_labels_{GOOD_LABELS.get(tag, '')}.json")
    return json.load(open(f))["labels"] if f.exists() else {}


def subset(seed, n_id_pool, n_ood_pool):
    """Byte-for-byte the same draw RR.train makes, so the trained heads and the post-hoc
    detectors are fitted on identical molecules at every seed."""
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    iid = torch.randperm(n_id_pool, generator=g)[:N_ID].numpy()
    iod = torch.randperm(n_ood_pool, generator=g)[:N_OOD].numpy()
    return iid, iod


# ------------------------------- classifiers -------------------------------
class Clf(nn.Module):
    def __init__(s, d, c):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                              nn.Linear(64, c))
    def forward(s, x): return s.net(x)


def fit_clf(Xid, y, Xood, mode, hp, seed):
    """mode: 'plain' = CE on ID only; 'oe' = + uniform-output loss on aux OOD;
    'energy' = + Liu et al. squared energy margins on ID and aux OOD."""
    torch.manual_seed(seed); np.random.seed(seed)
    C = int(y.max()) + 1
    Xi = torch.as_tensor(Xid); yi = torch.as_tensor(y, dtype=torch.long)
    Xo = torch.as_tensor(Xood) if Xood is not None else None
    lr, wd = (hp if mode == "plain" else
              (hp[1], 5e-4) if mode == "oe" else (CLF_DEF[0], CLF_DEF[1]))
    clf = Clf(Xid.shape[1], C)
    opt = torch.optim.Adam(clf.parameters(), lr, weight_decay=wd)
    for _ in range(CLF_EPOCHS):
        clf.train(); opt.zero_grad()
        loss = F.cross_entropy(clf(Xi), yi)
        if mode == "oe":
            lo = F.log_softmax(clf(Xo), 1)
            loss = loss + hp[0] * (-lo.mean(1)).mean()          # push towards uniform
        elif mode == "energy":
            lam, (m_in, m_out) = hp
            Ei = -torch.logsumexp(clf(Xi), 1); Eo = -torch.logsumexp(clf(Xo), 1)
            loss = loss + lam * (F.relu(Ei - m_in).pow(2).mean() +
                                 F.relu(m_out - Eo).pow(2).mean())
        loss.backward(); opt.step()
    clf.eval()
    return clf


def _logits(clf, A):
    with torch.no_grad(): return clf(torch.as_tensor(A, dtype=torch.float32))


# ------------------------------ post-hoc detectors ------------------------------
def _gauss(tr, eps):
    mu = tr.mean(0)
    S = np.cov(tr, rowvar=False) + eps * np.eye(tr.shape[1])
    P = np.linalg.pinv(S)
    sign, logdet = np.linalg.slogdet(S)
    return mu, P, float(logdet)


def maha_id(tr, eps):
    mu, P, _ = _gauss(tr, eps)
    return lambda A: np.einsum("ij,jk,ik->i", A - mu, P, A - mu)


def maha_oe(tr, tro, eps, mode):
    """Two-sample Gaussian log-likelihood ratio log p_ood(x) - log p_id(x), i.e. the same
    density ratio the trained OE head estimates, but in closed form.  Constant terms
    (log-determinants) are dropped: they shift every score equally and cannot change a
    ranking metric.  The three modes are three covariance estimators, all needed because
    512 dimensions with 1500 ID molecules is a rank-deficient regime.
      shared -- pooled within-class covariance (an LDA-style direction)
      sep    -- one full covariance per class (QDA)
      diag   -- one diagonal covariance per class (the strongest regulariser)"""
    mi, mo = tr.mean(0), tro.mean(0)
    if mode == "diag":
        vi = tr.var(0) + eps; vo = tro.var(0) + eps
        return lambda A: (((A - mi) ** 2 / vi).sum(1) - ((A - mo) ** 2 / vo).sum(1))
    if mode == "shared":
        pooled = np.vstack([tr - mi, tro - mo])
        _, P, _ = _gauss(pooled, eps)
        Pi = Po = P
    else:
        _, Pi, _ = _gauss(tr, eps)
        _, Po, _ = _gauss(tro, eps)
    return lambda A: (np.einsum("ij,jk,ik->i", A - mi, Pi, A - mi)
                      - np.einsum("ij,jk,ik->i", A - mo, Po, A - mo))


def knn_id(tr, k):
    nn_ = NearestNeighbors(n_neighbors=min(k, len(tr))).fit(tr)
    return lambda A: nn_.kneighbors(A)[0].mean(1)


def knn_oe(tr, tro, k1, k2):
    a = NearestNeighbors(n_neighbors=min(k1, len(tr))).fit(tr)
    b = NearestNeighbors(n_neighbors=min(k2, len(tro))).fit(tro)
    return lambda A: a.kneighbors(A)[0].mean(1) - b.kneighbors(A)[0].mean(1)


def lof_id(tr, n):
    m = LocalOutlierFactor(n_neighbors=min(n, len(tr) - 1), novelty=True).fit(tr)
    return lambda A: -m.score_samples(A)


def lof_oe(tr, tro, n1, n2):
    a = LocalOutlierFactor(n_neighbors=min(n1, len(tr) - 1), novelty=True).fit(tr)
    b = LocalOutlierFactor(n_neighbors=min(n2, len(tro) - 1), novelty=True).fit(tro)
    return lambda A: (-a.score_samples(A)) - (-b.score_samples(A))


# ------------------------------ method registry ------------------------------
# build(ctx, hp, seed) -> scorer(features) -> higher means more OOD
def _b_msp(c, hp, s):
    clf = fit_clf(c["Xid_lab"], c["y"], None, "plain", hp, s)
    return lambda A: (1 - F.softmax(_logits(clf, A), 1).max(1).values).numpy()

def _b_energy(c, hp, s):
    clf = fit_clf(c["Xid_lab"], c["y"], None, "plain", hp, s)
    return lambda A: (-torch.logsumexp(_logits(clf, A), 1)).numpy()

def _b_odin(c, hp, s):
    clf = fit_clf(c["Xid_lab"], c["y"], None, "plain", CLF_DEF, s)
    return lambda A: (1 - F.softmax(_logits(clf, A) / hp, 1).max(1).values).numpy()

def _b_msp_oe(c, hp, s):
    clf = fit_clf(c["Xid_lab"], c["y"], c["Xood"], "oe", hp, s)
    return lambda A: (1 - F.softmax(_logits(clf, A), 1).max(1).values).numpy()

def _b_odin_oe(c, hp, s):
    clf = fit_clf(c["Xid_lab"], c["y"], c["Xood"], "oe", OECLF_DEF, s)
    return lambda A: (1 - F.softmax(_logits(clf, A) / hp, 1).max(1).values).numpy()

def _b_energy_oe(c, hp, s):
    clf = fit_clf(c["Xid_lab"], c["y"], c["Xood"], "energy", hp, s)
    return lambda A: (-torch.logsumexp(_logits(clf, A), 1)).numpy()

METHODS = {
    "MSP":            ("clf",  G_CLF,    _b_msp),
    "ODIN":           ("clf",  G_T,      _b_odin),
    "Energy":         ("clf",  G_CLF,    _b_energy),
    "Mahalanobis":    ("post", G_MAHA,   lambda c, h, s: maha_id(c["Xid"], h)),
    "KNN":            ("post", G_KNN,    lambda c, h, s: knn_id(c["Xid"], h)),
    "LOF":            ("post", G_LOF,    lambda c, h, s: lof_id(c["Xid"], h)),
    "MSP-OE":         ("clf",  G_OECLF,  _b_msp_oe),
    "ODIN-OE":        ("clf",  G_T,      _b_odin_oe),
    "Energy-OE":      ("clf",  G_ENOE,   _b_energy_oe),
    "OE-Mahalanobis": ("post", G_MAHAOE, lambda c, h, s: maha_oe(c["Xid"], c["Xood"], *h)),
    "OE-KNN":         ("post", G_KNNOE,  lambda c, h, s: knn_oe(c["Xid"], c["Xood"], *h)),
    "OE-LOF":         ("post", G_LOFOE,  lambda c, h, s: lof_oe(c["Xid"], c["Xood"], *h)),
}
NEEDS_LABEL = {"MSP", "ODIN", "Energy", "MSP-OE", "ODIN-OE", "Energy-OE"}
USES_OE = {"MSP-OE", "ODIN-OE", "Energy-OE", "OE-Mahalanobis", "OE-KNN", "OE-LOF",
           "BalancedOODHead", "RPO"}


def context(D, y_all, seed):
    iid, iod = subset(seed, len(D["train_id"][0]), len(D["train_ood"][0]))
    Xid = D["train_id"][0][iid]; Xood = D["train_ood"][0][iod]
    y = y_all[iid]
    ok = y >= 0
    return {"Xid": Xid, "Xood": Xood, "y": y[ok], "Xid_lab": Xid[ok], "n_lab": int(ok.sum())}


def evaluate(fn, D, split_id, split_ood):
    s_id = np.nan_to_num(np.asarray(fn(D[split_id][0]), dtype=np.float64))
    s_ood = np.nan_to_num(np.asarray(fn(D[split_ood][0]), dtype=np.float64))
    return s_id, s_ood


def run_method(name, D, y_all, ctxs):
    kind, grid, build = METHODS[name]
    best = None
    for hp in grid:                                  # exactly 9 validation trials
        vals = []
        for s in SEL:
            fn = build(ctxs[s], hp, s)
            si, so = evaluate(fn, D, "val_id", "val_ood")
            vals.append(RR.mets(si, so)[0])
        v = float(np.mean(vals))
        if best is None or v > best[1]: best = (hp, v)
    hp = best[0]
    R = []
    for s in FIN:
        c = ctxs.get(s) or context(D, y_all, s)
        fn = build(c, hp, s)
        si, so = evaluate(fn, D, "test_id", "test_ood")
        R.append(RR.mets(si, so))
    return R, hp, best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--backbone", default="minimol")
    a = ap.parse_args()
    D, info = load_cell(a.cell, a.backbone)
    lab = label_map(a.cell)
    y_all = np.array([lab.get(s, -1) for s in D["train_id"][1]])
    ok = y_all >= 0
    logit_ok = ok.sum() > 50 and len(np.unique(y_all[ok])) > 1
    print(f"[{a.cell}/{a.backbone}] id={len(D['train_id'][0])} ood={len(D['train_ood'][0])} "
          f"labelled={int(ok.sum())} classes={len(np.unique(y_all[ok])) if ok.any() else 0} "
          f"logit_trio={'yes' if logit_ok else 'NO'} split={info}", flush=True)

    ctxs = {s: context(D, y_all, s) for s in set(SEL) | set(FIN)}
    out = {"cell": a.cell, "backbone": a.backbone, "domain_split": info,
           "n_labelled": int(ok.sum()), "logit_trio": bool(logit_ok), "methods": {}}

    for name in METHODS:
        if name in NEEDS_LABEL and not logit_ok:
            print(f"  {name:16} SKIPPED (no usable class label)", flush=True); continue
        R, hp, v = run_method(name, D, y_all, ctxs)
        out["methods"][name] = {"auroc": [r[0] for r in R], "aupr": [r[1] for r in R],
                                "fpr95": [r[2] for r in R], "hp": str(hp), "val": v,
                                "oe": name in USES_OE}
        print(f"  {name:16} AUROC {np.mean([r[0] for r in R]):.4f} "
              f"±{np.std([r[0] for r in R]):.4f}  FPR95 {np.mean([r[2] for r in R]):.3f}  hp={hp}",
              flush=True)

    # the two trained exposure heads go through RR's own code path, unmodified
    V = {k: [torch.tensor(D[k][0])] for k in RR.KEYS}
    for kind, name in [("bce", "BalancedOODHead"), ("rpo", "RPO")]:
        R, hp = RR.run_objective(V, kind)
        out["methods"][name] = {"auroc": [r[0] for r in R], "aupr": [r[1] for r in R],
                                "fpr95": [r[2] for r in R], "hp": str(hp), "oe": True}
        print(f"  {name:16} AUROC {np.mean([r[0] for r in R]):.4f} "
              f"±{np.std([r[0] for r in R]):.4f}  FPR95 {np.mean([r[2] for r in R]):.3f}  hp={hp}",
              flush=True)

    Path("repro").mkdir(exist_ok=True)
    json.dump(out, open(f"repro/moe_{a.cell}_{a.backbone}.json", "w"), indent=2)
    print(f"MOE_DONE {a.cell} {a.backbone}", flush=True)


if __name__ == "__main__":
    main()
