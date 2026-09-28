#!/usr/bin/env python3
"""Batch-level label-free detector selection over the FULL detector bank.

Scales batchsel.py from {Mahalanobis, OE} to the paper's whole baseline set plus the
exposure detector:

  MSP, ODIN, Energy          logit-based, from a property classifier trained on ID
                             cls_label over the same frozen features
  Mahalanobis, KNN, LOF      distance/density, fit on ID only
  OE                         balanced BCE head trained on ID + the S_oe mechanisms

Setting is unchanged and non-degenerate: queries arrive as one MIXED batch of unknown
composition (a mechanism's molecules plus held-out ID at a given OOD prevalence), the
target is the AUROC WITHIN that batch, and the selector sees only label-free statistics of
each detector's score distribution on the batch. A meta-selector is trained on HISTORICAL
mechanisms (S_router) and evaluated on 200 disjoint unseen mechanisms (H1+H2).

Arms:
  each detector alone
  best_fixed   the best single detector, picked on the evaluation set (conservative)
  learned      shared regressor: (one detector's batch statistics) -> (that detector's
               within-batch AUROC), trained across all detectors and batches, then
               argmax at test time. This shares strength across detectors, gives
               N_batches x K training rows instead of N_batches, and scales to any K --
               a K-way classifier has too few samples and is corrupted by the argmax of
               K noisy AUROC estimates.
  learned_pruned  same, after dropping detectors whose mean AUROC on the HISTORICAL
               mechanisms is below chance (a training-side decision, no test information)
  learned_guarded selection is only worth doing where it actually helps. The guard
               CROSS-VALIDATES the selector on the HISTORICAL mechanisms (2 folds: fit on
               one, score the other) and falls back to the best historical fixed detector
               unless the held-out gain is positive. Purely training-side, free at
               deployment. An earlier version instead compared the top-2 historical
               detectors by a margin; that breaks with two backbones in the bank, because
               mi:OE and un:OE are both strong, the top-2 margin collapses, and the guard
               stops firing exactly where it is needed (scaffold splits went to -0.07).
               Cross-validating the decision itself has no such failure mode.
  ORACLE       per-batch max over the bank; needs labels, and note it is biased upward by
               the max of K noisy per-batch estimates

Usage: python batchsel_multi.py --cell ec50_assay --seed 21
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, pickle, argparse, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4)
from pathlib import Path
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import ks_2samp, wasserstein_distance, skew, kurtosis
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import revision_run as RR
import mepoe_gate0 as MG
from routing_gate import train_oe, N_FIT, SEEDS

CACHE = Path("cache/ood_dpo_cache")
BASE = {"ec50_assay": "lbap_general_ec50_assay", "ic50_assay": "lbap_general_ic50_assay",
        "ki_assay": "lbap_general_ki_assay",
        "ec50_scaffold": "lbap_general_ec50_scaffold",
        "ic50_scaffold": "lbap_general_ic50_scaffold"}
DETS = ["MSP", "ODIN", "Energy", "Mahalanobis", "KNN", "LOF", "OE",
        "RPO", "OE-pAUC", "OE-MV"]
MV = os.environ.get("MV", "1") == "1"
PREV = [float(x) for x in os.environ.get("PREV_LIST", "0.5,0.25,0.10").split(",")]
REPEATS = 5
T_ODIN = 1000.0
GUARD_MARGIN = 0.0     # required held-out gain on historical mechanisms


class Clf(nn.Module):
    def __init__(s, d, c):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        s.o = nn.Linear(64, c)
    def forward(s, x): return s.o(s.f(x))


def _oe(heads, A):
    t = torch.as_tensor(A, dtype=torch.float32)
    with torch.no_grad():
        return np.mean([hh([t]).numpy() for hh in heads], 0)


def id_labels(cell, smis):
    """cls_label for the ID molecules, from the official DrugOOD json."""
    raw = json.load(open(f"data/raw/{BASE[cell]}.json"))["split"]
    lab = {}
    for part in ("train", "iid_val", "iid_test"):
        for it in raw.get(part, []):
            if it.get("smiles") is not None and it.get("cls_label") is not None:
                lab.setdefault(it["smiles"], int(it["cls_label"]))
    return np.array([lab.get(s, -1) for s in smis])


def common_smiles(cell, backbones):
    """SMILES present in every requested backbone cache, so the two feature views index
    the same molecules in the same order."""
    b = BASE[cell]; keep = None
    for bb in backbones:
        f = CACHE / f"{b}_{bb}_features.pkl"
        s = set(pickle.load(open(f, "rb"))["features"])
        keep = s if keep is None else (keep & s)
    return keep


def train_oe_rpo(Xid, Xood, seed, beta=1.0):
    """The paper's own objective: pairwise logistic (Bradley-Terry) over ID/OOD pairs,
    pairs resampled each step. Identical head/optimiser/budget to the BCE-trained OE head,
    so any difference between them is the objective and nothing else."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed); np.random.seed(seed)
    head = RR.Head([Xid.shape[1]])
    opt = torch.optim.AdamW(head.parameters(), MG.LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9)
    for _ in range(MG.EPOCHS):
        head.train(); opt.zero_grad()
        Eid, Eood = head([Xid]), head([Xood])
        pair = torch.randint(len(Eid), (len(Eood),), generator=g)
        core = F.softplus(-beta * (Eood - Eid[pair])).mean()
        (core + MG.GAMMA * (Eid.pow(2).mean() + Eood.pow(2).mean())).backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
    head.eval()
    return head


def train_oe_pauc(Xid, Xood, seed, alpha=0.25):
    """Exposure head trained on the FPR-relevant corner only: the top-alpha fraction of ID
    by score crossed with the bottom-alpha fraction of OOD. Same head/optimiser as the
    plain OE detector, so it enters the bank as a genuinely different scorer rather than a
    reparameterisation."""
    torch.manual_seed(seed); np.random.seed(seed)
    head = RR.Head([Xid.shape[1]])
    opt = torch.optim.AdamW(head.parameters(), MG.LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9)
    for _ in range(MG.EPOCHS):
        head.train(); opt.zero_grad()
        Eid, Eood = head([Xid]), head([Xood])
        ki = max(1, int(round(alpha * len(Eid)))); ko = max(1, int(round(alpha * len(Eood))))
        hi = torch.topk(Eid, ki).values
        lo = torch.topk(-Eood, ko).values.neg()
        core = F.softplus(-(lo.unsqueeze(0) - hi.unsqueeze(1))).mean()
        (core + MG.GAMMA * (Eid.pow(2).mean() + Eood.pow(2).mean())).backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
    head.eval()
    return head


def extra_views(smi_lists):
    """Morgan ECFP4 + the 10 RDKit descriptors used throughout this project, computed for
    row-aligned SMILES so a multi-view detector can be added to the bank."""
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    out = {}
    for key, smis in smi_lists.items():
        MO, DE = [], []
        for s in smis:
            m = Chem.MolFromSmiles(s)
            arr = np.zeros(2048, dtype=np.float32)
            if m is not None:
                DataStructs.ConvertToNumpyArray(
                    AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), arr)
                DE.append([f(m) for f in RR.DESCS])
            else:
                DE.append([0.0] * len(RR.DESCS))
            MO.append(arr)
        out[key] = (np.stack(MO), np.asarray(DE, dtype=np.float32))
    return out


def view_detectors(cell, v, rec, idx, oe_rows, tag, v_mv=None):
    """Build the whole detector bank on ONE feature view. Returns raw score functions
    (they take a plain feature matrix from that view) plus the ID reference scores.
    Entries are (fn, view_suffix); "_mv" entries are scored on the multi-view matrix."""
    id_fit = v["id_train"][idx[:N_FIT]]
    S_oe = v["S"][torch.as_tensor(oe_rows)]
    tr = id_fit.numpy()

    # ---- logit-based trio: property classifier on ID cls_label ----
    y = id_labels(cell, rec["id_train"])[idx[:N_FIT]]
    ok = y >= 0
    logit_ok = ok.sum() > 50 and len(np.unique(y[ok])) > 1
    if logit_ok:
        Xc = torch.tensor(tr[ok]); yc = torch.tensor(y[ok], dtype=torch.long)
        torch.manual_seed(0)
        clf = Clf(tr.shape[1], int(yc.max()) + 1)
        opt = torch.optim.Adam(clf.parameters(), 0.01, weight_decay=5e-4)
        for _ in range(300):
            clf.train(); opt.zero_grad(); F.cross_entropy(clf(Xc), yc).backward(); opt.step()
        clf.eval()
        def logits(A):
            with torch.no_grad(): return clf(torch.as_tensor(A, dtype=torch.float32))
    else:
        print(f"[{cell}/{tag}] WARNING: no usable cls_label -> MSP/ODIN/Energy dropped",
              flush=True)

    mu = tr.mean(0); P = np.linalg.pinv(np.cov(tr, rowvar=False) + 1e-3 * np.eye(tr.shape[1]))
    nn_ = NearestNeighbors(n_neighbors=50).fit(tr)
    lof_ = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(tr)
    heads = [train_oe(id_fit, S_oe, s) for s in SEEDS]

    heads_p = [train_oe_pauc(id_fit, S_oe, s) for s in SEEDS]
    heads_r = [train_oe_rpo(id_fit, S_oe, s) for s in SEEDS]

    raw_fn = {
        "Mahalanobis": (lambda A: np.einsum("ij,jk,ik->i", A - mu, P, A - mu), ""),
        "KNN": (lambda A: nn_.kneighbors(A)[0].mean(1), ""),
        "LOF": (lambda A: -lof_.score_samples(A), ""),
        "OE": (lambda A: _oe(heads, A), ""),
        "OE-pAUC": (lambda A: _oe(heads_p, A), ""),
        "RPO": (lambda A: _oe(heads_r, A), ""),
    }
    if logit_ok:
        raw_fn["MSP"] = (lambda A: 1 - F.softmax(logits(A), 1).max(1).values.numpy(), "")
        raw_fn["ODIN"] = (lambda A: 1 - F.softmax(logits(A) / T_ODIN, 1).max(1).values.numpy(), "")
        raw_fn["Energy"] = (lambda A: -torch.logsumexp(logits(A), 1).numpy(), "")
    if v_mv is not None:
        mv_fit = v_mv["id_train"][idx[:N_FIT]]
        mv_oe = v_mv["S"][torch.as_tensor(oe_rows)]
        heads_mv = [train_oe(mv_fit, mv_oe, s) for s in SEEDS]
        raw_fn["OE-MV"] = (lambda A: _oe(heads_mv, A), "_mv")

    local = [n for n in DETS if n in raw_fn]
    trmv = v_mv["id_train"][idx[:N_FIT]].numpy() if v_mv is not None else None
    ref = {n: np.nan_to_num(np.asarray(
              raw_fn[n][0](trmv if raw_fn[n][1] == "_mv" else tr), dtype=np.float64))
           for n in local}
    dead = [n for n in local if ref[n].std() < 1e-9]
    if dead:
        print(f"[{cell}/{tag}] dropping degenerate detectors "
              f"(zero score variance on ID): {dead}", flush=True)
        local = [n for n in local if n not in dead]
    return raw_fn, ref, local


def build_bank(cell, seed, backbone="minimol"):
    """With backbone='both', the two views are restricted to molecules present in BOTH
    caches so their rows index the same molecules, and detector names are prefixed."""
    bbs = ["minimol", "unimol"] if backbone == "both" else [backbone]
    keep = common_smiles(cell, bbs) if len(bbs) > 1 else None
    views, rec = {}, None
    for bb in bbs:
        views[bb], rec = MG.setup(cell, seed, backbone=bb, keep=keep)
    d = views[bbs[0]]
    MG.G.clear(); MG.G.update(d)
    for bb in bbs[1:]:
        assert len(views[bb]["S"]) == len(d["S"]) and \
               len(views[bb]["id_test"]) == len(d["id_test"]), "view rows misaligned"

    # one ID subsample and one mechanism split, shared by every view
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(d["id_train"]))
    Smech = d["S_mech"]; uS = np.unique(Smech); h = rng.permutation(len(uS))
    m_oe, m_rt = set(uS[h[:len(uS) // 2]]), set(uS[h[len(uS) // 2:]])
    oe_rows = np.where(np.isin(Smech, list(m_oe)))[0]

    # multi-view feature blocks (backbone | Morgan | descriptors), row-aligned
    if MV:
        smi = {k: d[k + "_smi"] for k in ("id_train", "id_test", "S", "H1", "H2")}
        ex = extra_views(smi)
        for bb in list(bbs):
            base = views[bb]
            st = None
            mvv = {}
            for k in ("id_train", "id_test", "S", "H1", "H2"):
                blk = np.concatenate([base[k].numpy(), ex[k][0], ex[k][1]], 1)
                if st is None:
                    st = (blk.mean(0), blk.std(0) + 1e-6)
                mvv[k] = torch.tensor((blk - st[0]) / st[1], dtype=torch.float32)
                if k in ("S", "H1", "H2"): mvv[k + "_mech"] = base[k + "_mech"]
            views[bb + "_mv"] = mvv

    z, zREF, names, view_of = {}, {}, [], {}
    for bb in bbs:
        raw_fn, ref, local = view_detectors(cell, views[bb], rec, idx, oe_rows, bb,
                                            views.get(bb + "_mv"))
        pref = "" if len(bbs) == 1 else f"{bb[:2]}:"
        for n in local:
            key = pref + n
            vk = bb + raw_fn[n][1]
            names.append(key); view_of[key] = vk
            z[key] = (lambda D, fn=raw_fn[n][0], r=ref[n], v=vk: np.nan_to_num(
                (np.asarray(fn(D[v]), dtype=np.float64) - r.mean()) / (r.std() + 1e-9)))
            zREF[key] = (ref[n] - ref[n].mean()) / (ref[n].std() + 1e-9)
    return d, z, zREF, m_rt, names, view_of, views


def batch_stats(zB, zREF):
    """A near-degenerate detector (e.g. ODIN at T=1000 saturates to an almost constant
    score) gives a zero-variance batch, for which skew/kurtosis are undefined; those
    become 0 rather than NaN so the detector simply looks uninformative to the selector."""
    q95 = np.quantile(zREF, 0.95); s = np.sort(zB); h = max(1, len(s) // 2)
    v = [zB.mean(), zB.std(),
         float(skew(zB)) if len(zB) > 2 and zB.std() > 1e-12 else 0.0,
         float(kurtosis(zB)) if len(zB) > 3 and zB.std() > 1e-12 else 0.0,
         ks_2samp(zB, zREF).statistic, wasserstein_distance(zB, zREF),
         float((zB > q95).mean()), float(s[h:].mean() - s[:h].mean()),
         float(zB.std() / (zREF.std() + 1e-9))]
    return [float(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)) for x in v]


def collect(z, zREF, names, Xv, mech, ids, IDv, n_idpool, rng, prev, n_ood=None):
    """Xv / IDv are {backbone: matrix}; every view is indexed with the SAME rows, so a
    detector built on one backbone still scores exactly the same molecules.
    n_ood subsamples each mechanism, varying batch SIZE independently of prevalence."""
    F_, A_ = [], []
    for m in np.asarray(ids):
        rows = np.where(mech == m)[0]
        if n_ood is not None and len(rows) > n_ood:
            rows = rows[rng.choice(len(rows), n_ood, replace=False)]
        n_id = max(1, int(round(len(rows) * (1 - prev) / prev)))
        if n_id > n_idpool: continue
        sel = rng.choice(n_idpool, n_id, replace=False)
        Do = {v: Xv[v][rows] for v in Xv}
        Di = {v: IDv[v][sel] for v in IDv}
        feat, au = [], []
        for n in names:
            zo, zi = z[n](Do), z[n](Di)
            feat += batch_stats(np.r_[zi, zo], zREF[n])
            au.append(RR.mets(zi, zo)[0])
        F_.append(feat); A_.append(au)
    return np.asarray(F_), np.asarray(A_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--n_ood", type=int, default=None,
                    help="subsample this many molecules per mechanism (batch-size axis)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--backbone", default="minimol", choices=["minimol", "unimol", "both"])
    a = ap.parse_args()
    d, z, zREF, m_rt, names, view_of, views = build_bank(a.cell, a.seed, a.backbone)
    bbs = list(views)          # includes the "<backbone>_mv" pseudo-views when MV is on
    IDv = {v: views[v]["id_test"].numpy() for v in bbs}
    n_idpool = len(d["id_test"])
    Xsv = {v: views[v]["S"].numpy() for v in bbs}
    ms = d["S_mech"]
    Xuv, mu_, off = {v: [] for v in bbs}, [], 0
    for n in ("H1", "H2"):
        for v in bbs: Xuv[v].append(views[v][n].numpy())
        mu_.append(d[n + "_mech"] + off)
        off += int(d[n + "_mech"].max()) + 1
    Xuv = {v: np.concatenate(Xuv[v]) for v in bbs}
    mu_ = np.concatenate(mu_)
    print(f"[{a.cell}/s{a.seed}/{a.backbone}] detector bank ({len(names)}): {names}", flush=True)

    out = {"cell": a.cell, "seed": a.seed, "backbone": a.backbone,
           "detectors": names, "prevalence": {}}
    for prev in PREV:
        acc = {k: [] for k in ["best_fixed", "learned", "learned_pruned", "learned_guarded",
                               "oracle", "oracle_pruned", "sel_acc", "n_kept", "dominated",
                               "cv_gain"]}
        per_det = {n: [] for n in names}
        for rep in range(REPEATS):
            rg = np.random.default_rng(100 + rep)
            Ftr, Atr = collect(z, zREF, names, Xsv, ms, sorted(m_rt),
                               IDv, n_idpool, rg, prev, a.n_ood)
            Fte, Ate = collect(z, zREF, names, Xuv, mu_, np.unique(mu_),
                               IDv, n_idpool, rg, prev, a.n_ood)
            if not len(Ate): continue
            Ftr = np.nan_to_num(Ftr); Fte = np.nan_to_num(Fte)
            Atr = np.nan_to_num(Atr, nan=0.5); Ate = np.nan_to_num(Ate, nan=0.5)
            n = len(Ate)
            for i, nm in enumerate(names): per_det[nm].append(Ate[:, i].mean())
            K, nf = len(names), Ftr.shape[1] // len(names)
            # stack (batch, detector) rows: features of that detector -> its own AUROC
            Rtr = np.vstack([Ftr[:, i * nf:(i + 1) * nf] for i in range(K)])
            ytr = np.concatenate([Atr[:, i] for i in range(K)])
            reg = GradientBoostingRegressor(random_state=rep, n_estimators=300,
                                            max_depth=3).fit(Rtr, ytr)
            pred = np.column_stack([reg.predict(Fte[:, i * nf:(i + 1) * nf]) for i in range(K)])
            pick = pred.argmax(1)
            keep = [i for i in range(K) if Atr[:, i].mean() >= 0.5]      # training-side prune
            if not keep: keep = list(range(K))
            pick_p = np.array(keep)[pred[:, keep].argmax(1)]
            acc["best_fixed"].append(Ate.mean(0).max())
            acc["learned"].append(Ate[np.arange(n), pick].mean())
            acc["learned_pruned"].append(Ate[np.arange(n), pick_p].mean())
            acc["oracle"].append(Ate.max(1).mean())
            acc["oracle_pruned"].append(Ate[:, keep].max(1).mean())
            acc["sel_acc"].append((pick_p == Ate.argmax(1)).mean())
            acc["n_kept"].append(len(keep))
            # --- guard: does selection beat best-fixed on HELD-OUT historical mechanisms? ---
            hist = Atr.mean(0); best_hist = int(np.argmax(hist))
            nb = len(Atr); half = nb // 2
            perm = np.random.default_rng(rep).permutation(nb)
            cv_gain = []
            for fa, fb in ((perm[:half], perm[half:]), (perm[half:], perm[:half])):
                Ra = np.vstack([Ftr[fa][:, i * nf:(i + 1) * nf] for i in range(K)])
                ya = np.concatenate([Atr[fa][:, i] for i in range(K)])
                rg2 = GradientBoostingRegressor(random_state=rep, n_estimators=300,
                                                max_depth=3).fit(Ra, ya)
                pb = np.column_stack([rg2.predict(Ftr[fb][:, i * nf:(i + 1) * nf])
                                      for i in range(K)]).argmax(1)
                sel_b = Atr[fb][np.arange(len(fb)), pb].mean()
                fix_b = Atr[fb][:, int(np.argmax(Atr[fa].mean(0)))].mean()
                cv_gain.append(sel_b - fix_b)
            use_sel = bool(np.mean(cv_gain) > GUARD_MARGIN)
            acc["dominated"].append(float(not use_sel))
            acc["cv_gain"].append(float(np.mean(cv_gain)))
            acc["learned_guarded"].append(
                Ate[np.arange(n), pick].mean() if use_sel else Ate[:, best_hist].mean())
        M = {k: float(np.mean(v)) for k, v in acc.items()}
        S = {k: float(np.std(v)) for k, v in acc.items()}
        D = {nm: float(np.mean(v)) for nm, v in per_det.items()}
        dl = M["learned"] - M["best_fixed"]; dp = M["learned_pruned"] - M["best_fixed"]
        dg = M["learned_guarded"] - M["best_fixed"]
        out["prevalence"][str(prev)] = {"per_detector": D, **M,
                                        "sd_learned": S["learned"],
                                        "sd_learned_pruned": S["learned_pruned"],
                                        "delta_learned": dl, "delta_learned_pruned": dp,
                                        "delta_learned_guarded": dg,
                                        "oracle_headroom": M["oracle"] - M["best_fixed"],
                                        "pass": bool(max(dl, dp) >= 0.030)}
        print(f"[{a.cell}/s{a.seed}] prevalence {prev:.0%}", flush=True)
        print("     " + "  ".join(f"{nm}={D[nm]:.4f}" for nm in names), flush=True)
        print(f"     best_fixed {M['best_fixed']:.4f} | learned {M['learned']:.4f} ({dl:+.4f}) "
              f"| learned_pruned {M['learned_pruned']:.4f} ({dp:+.4f} +-{S['learned_pruned']:.4f}) "
              f"{'PASS' if max(dl,dp)>=0.030 else 'FAIL'}", flush=True)
        print(f"     guarded {M['learned_guarded']:.4f} ({dg:+.4f}) "
              f"| guard fell back on {M['dominated']:.0%} of repeats "
              f"(held-out CV gain {M['cv_gain']:+.4f})", flush=True)
        print(f"     ORACLE {M['oracle']:.4f} / pruned {M['oracle_pruned']:.4f} "
              f"| kept {M['n_kept']:.1f}/{len(names)} detectors | sel acc {M['sel_acc']:.3f}",
              flush=True)
    out["n_ood"] = a.n_ood
    suff = f"_{a.tag}" if a.tag else ""
    json.dump(out, open(f"repro/batchmulti_{a.cell}_s{a.seed}{suff}.json", "w"), indent=2)
    print(f"BATCHMULTI_DONE {a.cell} s{a.seed}", flush=True)


if __name__ == "__main__":
    main()
