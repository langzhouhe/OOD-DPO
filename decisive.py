#!/usr/bin/env python3
"""Decisive control: is the learned router doing anything an analytic estimator cannot?

For detector d, calibrate the empirical mid-CDF F_0d on an INDEPENDENT ID set and, on an
unlabelled batch B, compute

    q_d = mean_{x in B} F_0d(s_d(x)).

For an ID molecule the probability-integral transform gives E = 1/2; for an OOD molecule it
gives exactly the Mann-Whitney statistic. So with batch OOD fraction pi,

    E[q_d] = (1 - pi)/2 + pi * AUROC_d,

and because pi is shared by every detector within a batch, argmax_d q_d ranks detectors by
their true AUROC -- with no learning, no historical OOD, and no knowledge of pi. The GBDR
router's features (batch mean, KS and Wasserstein against the ID reference, tail fraction)
are all monotone-ish proxies for this same quantity, so the learned router may simply be a
noisy estimator of q_d. If so the learned-router framing has to go.

Protocol defects in batchsel_multi.py that this script fixes:
  * historical and evaluation batches drew their ID members from the SAME id_test pool
  * `best_fixed` was max over the TEST matrix, i.e. an oracle, not a deployable baseline
  * the ECDF reference and the detector-fitting ID set were the same molecules

Clean, mutually disjoint ID splits:
  ID_fit    1200 from id_train  fits Maha/KNN/LOF, trains OE heads + property classifier
  ID_cal     800 from id_train  ECDF calibration and the batch-statistic reference
  ID_router  500 from id_test   ID members of the HISTORICAL batches
  ID_final   500 from id_test   ID members of the EVALUATION batches

Arms (all scored on identical batches):
  hist_best     best detector by mean AUROC over historical batches -- DEPLOYABLE baseline
  ecdf          argmax_d q_d -- analytic, no training, no historical OOD
  ecdf_guard    ecdf, falling back to hist_best when cross-validation says selection loses
  gbdr_guard    the current learned router with the same guard
  oracle_fixed  best single detector chosen on the evaluation set -- UPPER BOUND
  oracle_batch  per-batch max -- UPPER BOUND on any selection rule

Also reports an unknown-prevalence condition: the learned router is trained on a MIXTURE of
prevalences and tested at each one, since a deployed router does not know pi. The analytic
selector needs no such training.

Usage: python decisive.py --cell ec50_assay --seed 21
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, argparse, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4)
from sklearn.ensemble import GradientBoostingRegressor
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import revision_run as RR
import mepoe_gate0 as MG
import batchsel_multi as BM

N_FIT, N_CAL, N_ROUTER, N_FINAL = 1200, 800, 500, 500
PREV = [0.5, 0.25, 0.10]
REPEATS = 5
SEEDS = (1, 2, 3)


def mid_cdf(ref_sorted, x):
    lo = np.searchsorted(ref_sorted, x, "left")
    hi = np.searchsorted(ref_sorted, x, "right")
    return (lo + hi) / (2.0 * len(ref_sorted))


def build(cell, seed, backbone="minimol"):
    bbs = ["minimol", "unimol"] if backbone == "both" else [backbone]
    keep = BM.common_smiles(cell, bbs) if len(bbs) > 1 else None
    views, rec = {}, None
    for bb in bbs:
        views[bb], rec = MG.setup(cell, seed, backbone=bb, keep=keep)
    d = views[bbs[0]]; MG.G.clear(); MG.G.update(d)

    rng = np.random.default_rng(0)
    tr_perm = rng.permutation(len(d["id_train"]))
    te_perm = rng.permutation(len(d["id_test"]))
    fit_i, cal_i = tr_perm[:N_FIT], tr_perm[N_FIT:N_FIT + N_CAL]
    rou_i, fin_i = te_perm[:N_ROUTER], te_perm[N_ROUTER:N_ROUTER + N_FINAL]
    assert not (set(fit_i) & set(cal_i)) and not (set(rou_i) & set(fin_i))

    Sm = d["S_mech"]; uS = np.unique(Sm); h = rng.permutation(len(uS))
    m_oe, m_rt = set(uS[h[:len(uS) // 2]]), set(uS[h[len(uS) // 2:]])
    oe_rows = np.where(np.isin(Sm, list(m_oe)))[0]

    if BM.MV:
        smi = {k: d[k + "_smi"] for k in ("id_train", "id_test", "S", "H1", "H2")}
        ex = BM.extra_views(smi)
        for bb in list(bbs):
            base, st, mvv = views[bb], None, {}
            for k in ("id_train", "id_test", "S", "H1", "H2"):
                blk = np.concatenate([base[k].numpy(), ex[k][0], ex[k][1]], 1)
                if st is None: st = (blk.mean(0), blk.std(0) + 1e-6)
                mvv[k] = torch.tensor((blk - st[0]) / st[1], dtype=torch.float32)
                if k in ("S", "H1", "H2"): mvv[k + "_mech"] = base[k + "_mech"]
            views[bb + "_mv"] = mvv

    # view_detectors slices idx[:N_FIT] internally, so hand it our fit indices first
    idx = np.concatenate([fit_i, cal_i, tr_perm[N_FIT + N_CAL:]])
    old = BM.N_FIT; BM.N_FIT = N_FIT
    z, zREF, names, view_of = {}, {}, [], {}
    try:
        for bb in bbs:
            raw_fn, ref_fit, local = BM.view_detectors(cell, views[bb], rec, idx, oe_rows, bb,
                                                       views.get(bb + "_mv"))
            pref = "" if len(bbs) == 1 else f"{bb[:2]}:"
            for n in local:
                key, vk = pref + n, bb + raw_fn[n][1]
                names.append(key); view_of[key] = vk
                r = ref_fit[n]
                z[key] = (lambda D, fn=raw_fn[n][0], rr=r, v=vk: np.nan_to_num(
                    (np.asarray(fn(D[v]), dtype=np.float64) - rr.mean()) / (rr.std() + 1e-9)))
    finally:
        BM.N_FIT = old

    # ID_cal is the reference for BOTH the ECDF and the batch statistics
    calD = {v: views[v]["id_train"].numpy()[cal_i] for v in views}
    zCAL = {n: z[n](calD) for n in names}
    ecdf_ref = {n: np.sort(zCAL[n]) for n in names}
    Xv = {v: views[v]["S"].numpy() for v in views}
    Xu, mu_, off = {v: [] for v in views}, [], 0
    for k in ("H1", "H2"):
        for v in views: Xu[v].append(views[v][k].numpy())
        mu_.append(d[k + "_mech"] + off); off += int(d[k + "_mech"].max()) + 1
    Xu = {v: np.concatenate(Xu[v]) for v in views}; mu_ = np.concatenate(mu_)
    IDr = {v: views[v]["id_test"].numpy()[rou_i] for v in views}
    IDf = {v: views[v]["id_test"].numpy()[fin_i] for v in views}

    # --- exact transform for NEW molecules (e.g. official ood_test), so held-out data is
    # standardised with the identical statistics the development views used ---
    import pickle as _pk
    raw_feats = {bb: _pk.load(open(BM.CACHE / f"{BM.BASE[cell]}_{bb}_features.pkl", "rb"))["features"]
                 for bb in bbs}
    tr_smi = d["id_train_smi"]
    raw_stats = {}
    for bb in bbs:
        A = np.stack([raw_feats[bb][s] for s in tr_smi]).astype(np.float32)
        raw_stats[bb] = (A.mean(0), A.std(0) + 1e-6)
    mv_stats = {}
    if BM.MV:
        exx = BM.extra_views({"tr": tr_smi})
        for bb in bbs:
            base_std = (np.stack([raw_feats[bb][s] for s in tr_smi]).astype(np.float32)
                        - raw_stats[bb][0]) / raw_stats[bb][1]
            blk = np.concatenate([base_std, exx["tr"][0], exx["tr"][1]], 1)
            mv_stats[bb] = (blk.mean(0), blk.std(0) + 1e-6)

    def project(smis):
        ex = BM.extra_views({"q": smis}) if BM.MV else None
        out = {}
        for bb in bbs:
            A = np.stack([raw_feats[bb][s] for s in smis]).astype(np.float32)
            base_std = (A - raw_stats[bb][0]) / raw_stats[bb][1]
            out[bb] = base_std.astype(np.float32)
            if BM.MV:
                blk = np.concatenate([base_std, ex["q"][0], ex["q"][1]], 1)
                out[bb + "_mv"] = ((blk - mv_stats[bb][0]) / mv_stats[bb][1]).astype(np.float32)
        return out

    return dict(z=z, zCAL=zCAL, ecdf_ref=ecdf_ref, names=names, m_rt=sorted(m_rt),
                Sm=Sm, Xv=Xv, Xu=Xu, mu=mu_, IDr=IDr, IDf=IDf, project=project)


def collect(B, X, mech, ids, IDpool, rng, prev, n_ood=None):
    """Per batch: label-free stats, the analytic q_d, and each detector's true AUROC."""
    z, names, zCAL = B["z"], B["names"], B["zCAL"]
    npool = len(next(iter(IDpool.values())))
    Fs, Qs, As = [], [], []
    for m in np.asarray(ids):
        rows = np.where(mech == m)[0]
        if n_ood is not None and len(rows) > n_ood:
            rows = rows[rng.choice(len(rows), n_ood, replace=False)]
        n_id = max(1, int(round(len(rows) * (1 - prev) / prev)))
        if n_id > npool: continue
        sel = rng.choice(npool, n_id, replace=False)
        Do = {v: X[v][rows] for v in X}; Di = {v: IDpool[v][sel] for v in IDpool}
        f, q, a = [], [], []
        for n in names:
            zo, zi = z[n](Do), z[n](Di)
            f += BM.batch_stats(np.r_[zi, zo], zCAL[n])
            q.append(float(mid_cdf(B["ecdf_ref"][n], np.r_[zi, zo]).mean()))
            a.append(RR.mets(zi, zo)[0])
        Fs.append(f); Qs.append(q); As.append(a)
    return np.asarray(Fs), np.asarray(Qs), np.asarray(As)


def gbdr_pick(Ftr, Atr, Fte, K, rep):
    nf = Ftr.shape[1] // K
    R = np.vstack([Ftr[:, i * nf:(i + 1) * nf] for i in range(K)])
    y = np.concatenate([Atr[:, i] for i in range(K)])
    reg = GradientBoostingRegressor(random_state=rep, n_estimators=300, max_depth=3).fit(R, y)
    P = np.column_stack([reg.predict(Fte[:, i * nf:(i + 1) * nf]) for i in range(K)])
    return P.argmax(1)


def cv_gain(Ftr, Atr, K, rep):
    """Held-out gain of the FITTED router over the historical-best fixed detector.
    Only the GBDR router needs this; the analytic selector fits nothing, so its guard is
    evaluated directly on the historical batches."""
    nb = len(Atr); half = nb // 2
    perm = np.random.default_rng(rep).permutation(nb); g = []
    for fa, fb in ((perm[:half], perm[half:]), (perm[half:], perm[:half])):
        pb = gbdr_pick(Ftr[fa], Atr[fa], Ftr[fb], K, rep)
        sel = Atr[fb][np.arange(len(fb)), pb].mean()
        fix = Atr[fb][:, int(np.argmax(Atr[fa].mean(0)))].mean()
        g.append(sel - fix)
    return float(np.mean(g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--backbone", default="minimol")
    a = ap.parse_args()
    B = build(a.cell, a.seed, a.backbone)
    K = len(B["names"])
    print(f"[{a.cell}/s{a.seed}] {K} detectors | ID fit/cal/router/final "
          f"{N_FIT}/{N_CAL}/{N_ROUTER}/{N_FINAL} (disjoint)", flush=True)

    out = {"cell": a.cell, "seed": a.seed, "detectors": B["names"], "prevalence": {}}
    for prev in PREV:
        acc = {k: [] for k in ["hist_best", "ecdf", "ecdf_guard", "gbdr_guard",
                               "gbdr_mixprev", "oracle_fixed", "oracle_batch",
                               "ecdf_acc", "gbdr_acc"]}
        for rep in range(REPEATS):
            rg = np.random.default_rng(100 + rep)
            Ftr, Qtr, Atr = collect(B, B["Xv"], B["Sm"], B["m_rt"], B["IDr"], rg, prev)
            Fte, Qte, Ate = collect(B, B["Xu"], B["mu"], np.unique(B["mu"]), B["IDf"], rg, prev)
            if not len(Ate): continue
            Ftr, Fte = np.nan_to_num(Ftr), np.nan_to_num(Fte)
            Atr, Ate = np.nan_to_num(Atr, nan=.5), np.nan_to_num(Ate, nan=.5)
            n = len(Ate); truth = Ate.argmax(1)

            hb = int(np.argmax(Atr.mean(0)))
            acc["hist_best"].append(Ate[:, hb].mean())
            acc["oracle_fixed"].append(Ate.mean(0).max())
            acc["oracle_batch"].append(Ate.max(1).mean())

            p_ecdf = Qte.argmax(1)
            acc["ecdf"].append(Ate[np.arange(n), p_ecdf].mean())
            acc["ecdf_acc"].append((p_ecdf == truth).mean())
            # guard for ecdf: it needs no fitting, so evaluate it directly on history
            g_e = Atr[np.arange(len(Atr)), Qtr.argmax(1)].mean() - Atr[:, hb].mean()
            acc["ecdf_guard"].append(acc["ecdf"][-1] if g_e > 0 else acc["hist_best"][-1])

            p_g = gbdr_pick(Ftr, Atr, Fte, K, rep)
            acc["gbdr_acc"].append((p_g == truth).mean())
            g_g = cv_gain(Ftr, Atr, K, rep)
            acc["gbdr_guard"].append(Ate[np.arange(n), p_g].mean() if g_g > 0
                                     else Ate[:, hb].mean())

            # unknown-prevalence: train the router on a MIX of prevalences
            Fm, Am = [Ftr], [Atr]
            for q in PREV:
                if q == prev: continue
                f2, _, a2 = collect(B, B["Xv"], B["Sm"], B["m_rt"], B["IDr"],
                                    np.random.default_rng(900 + rep), q)
                if len(a2): Fm.append(np.nan_to_num(f2)); Am.append(np.nan_to_num(a2, nan=.5))
            Fm, Am = np.vstack(Fm), np.vstack(Am)
            p_mix = gbdr_pick(Fm, Am, Fte, K, rep)
            acc["gbdr_mixprev"].append(Ate[np.arange(n), p_mix].mean())

        M = {k: float(np.mean(v)) for k, v in acc.items() if v}
        S = {k: float(np.std(v)) for k, v in acc.items() if v}
        base = M["hist_best"]
        out["prevalence"][str(prev)] = {**M, "sd": S,
                                        "d_ecdf": M["ecdf"] - base,
                                        "d_ecdf_guard": M["ecdf_guard"] - base,
                                        "d_gbdr_guard": M["gbdr_guard"] - base,
                                        "d_gbdr_mixprev": M["gbdr_mixprev"] - base,
                                        "gbdr_minus_ecdf": M["gbdr_guard"] - M["ecdf_guard"]}
        print(f"[{a.cell}/s{a.seed}] prev {prev:.0%}  hist_best {base:.4f}", flush=True)
        print(f"    ecdf {M['ecdf']:.4f} ({M['ecdf']-base:+.4f})  "
              f"ecdf+guard {M['ecdf_guard']:.4f} ({M['ecdf_guard']-base:+.4f})  "
              f"[sel acc {M['ecdf_acc']:.3f}]", flush=True)
        print(f"    gbdr+guard {M['gbdr_guard']:.4f} ({M['gbdr_guard']-base:+.4f})  "
              f"mix-prev {M['gbdr_mixprev']:.4f} ({M['gbdr_mixprev']-base:+.4f})  "
              f"[sel acc {M['gbdr_acc']:.3f}]", flush=True)
        print(f"    GBDR - ECDF = {M['gbdr_guard']-M['ecdf_guard']:+.4f}   "
              f"| oracle fixed {M['oracle_fixed']:.4f}  per-batch {M['oracle_batch']:.4f}",
              flush=True)
    json.dump(out, open(f"repro/decisive_{a.cell}_s{a.seed}_{a.backbone}.json", "w"), indent=2)
    print(f"DECISIVE_DONE {a.cell} s{a.seed}", flush=True)


if __name__ == "__main__":
    main()
