#!/usr/bin/env python3
"""Reference-Anchored RPO: use the strongest matched-OE detector as a DPO reference model.

Motivation.  The matched-OE table (E34) put two zero-training two-sample detectors ahead of
RPO on the clean assay cells, with OE-Mahalanobis first.  A pairwise objective that treats
that detector as a COMPETITOR loses; a pairwise objective that treats it as the reference
policy -- which is what DPO prescribes and what the original RPO omitted -- only has to
learn the ranking residual the reference gets wrong.

  s_theta(x) = m(x) + alpha * g_theta(h(x))            m = frozen OE-Mahalanobis, z-scored
  w_io       = sigmoid(-tau * [m(x_o) - m(x_i)]),  normalised to mean 1
  L          = mean_io w_io * softplus(-beta[s(x_o) - s(x_i)]) + lambda * E[g_theta(x)^2]

TEMPERATURE CALIBRATION (protocol note, fixed before formal evaluation).
Fixed temperatures were replaced before formal evaluation because their effective
concentration depends strongly on the reference-margin geometry.  On z-scored reference
scores a typical pair has m_o - m_i of order 1-3, so tau=5 gives sigmoid(-10) ~ 5e-5 and
the gradient collapses onto a handful of pairs.  Temperature is therefore calibrated using
the training-pair effective sample size, with no validation or test outcome involved:

  w_k(tau) = sigmoid(-tau * dm_k),   r(tau) = (sum_k w_k)^2 / (N * sum_k w_k^2)

and tau is the smallest value with r(tau) = 0.5.  The target is 0.5, NOT 0.25: as
tau -> infinity, w -> 1[m_o < m_i], so r -> 1 - AUROC_ref, which is 0.24-0.26 for this
reference.  A 0.25 target therefore sits on the asymptote, pushes tau into saturation and
may have no solution.  r is monotone decreasing in tau, so bisection is exact.
Targets 0.25 / 0.75 and fixed tau in {1, 5, 10} are run as sensitivity arms only and never
participate in selection.

Ablation rungs -- a win must survive all four, not just beat the reference:

  1. ref_only          g = 0                                (= OE-Mahalanobis)
  2. ref_bce           pointwise BCE residual               (is a residual head enough?)
  3. ref_weighted_bce  pointwise BCE with the pair weights  (is it just hard-example
                       marginalised to per-sample weights    weighting?)
  4. ref_rpo           unweighted pairwise residual         (is pairwise enough?)
  5. ref_rpo_ess       reference-weighted pairwise          (the full method)

The decisive comparison is ref_rpo_ess > ref_bce (and > ref_weighted_bce), not merely
> ref_only.  Pre-registered bar: same direction in at least 3 of the 4 backbone x assay
configurations and a macro gain of about +0.003 to +0.005.  Size cells are excluded by
construction.

EQUAL BUDGET.  Every residual arm searches the same 9 trials over alpha x lr; tau is
calibrated, not tuned.  The reference itself (eps, covariance mode) is picked by the same
9-trial validation search matched_oe.py uses, then frozen before any residual is trained.

Usage: python reference_rpo.py --cell ec50_assay --backbone minimol
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, argparse, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import revision_run as RR
import matched_oe as MO

ALPHAS = [0.25, 0.5, 1.0]
LRS = [1e-4, 3e-4, 1e-3]          # 3 x 3 = 9 trials for every residual arm
ESS_PRIMARY = 0.5
ESS_SENS = [0.25, 0.75]
TAUS_SENS = [1.0, 5.0, 10.0]
GAMMA = 0.01                      # residual regulariser, fixed (middle of RR.GAMMAS)
BETA = 1.0
EPOCHS = 120
SEL, FIN = RR.SEL, RR.FIN


# ------------------------------- reference -------------------------------
def fit_reference(ctx, hp):
    """OE-Mahalanobis on this seed's ID/OOD subsets, z-scored on the ID training scores so
    alpha is on a meaningful scale.  z-scoring is affine and cannot change AUROC."""
    fn = MO.maha_oe(ctx["Xid"], ctx["Xood"], *hp)
    mu = float(np.mean(fn(ctx["Xid"]))); sd = float(np.std(fn(ctx["Xid"]))) + 1e-8
    return lambda A: (np.nan_to_num(np.asarray(fn(A), dtype=np.float64)) - mu) / sd


def select_reference(D, ctxs):
    """The same 9-trial VALIDATION search matched_oe.py runs, then frozen."""
    best = None
    for hp in MO.G_MAHAOE:
        v = float(np.mean([RR.mets(fit_reference(ctxs[s], hp)(D["val_id"][0]),
                                   fit_reference(ctxs[s], hp)(D["val_ood"][0]))[0]
                           for s in SEL]))
        if best is None or v > best[1]: best = (hp, v)
    return best


# ------------------------------- weighting -------------------------------
def _r_of_tau(dm, tau):
    w = torch.sigmoid(-tau * dm)
    return float(w.sum() ** 2 / (len(dm) * (w.pow(2).sum() + 1e-12)))


def calibrate_tau(mid, mood, seed, target, n=20000):
    """Smallest tau with training-pair ESS ratio = target.  Uses training-side reference
    margins only -- no validation or test information."""
    g = torch.Generator().manual_seed(seed + 9999)
    i = torch.randint(len(mid), (n,), generator=g)
    o = torch.randint(len(mood), (n,), generator=g)
    dm = mood[o] - mid[i]
    lo, hi = 1e-4, 200.0
    if _r_of_tau(dm, hi) > target:        # target unreachable: saturated regime
        return hi, False
    for _ in range(80):
        t = 0.5 * (lo + hi)
        if _r_of_tau(dm, t) > target: lo = t
        else: hi = t
    return 0.5 * (lo + hi), True


def pair_weights(mood, mid_paired, tau):
    """Normalised to mean 1, so the weighting cannot change the effective learning rate."""
    w = torch.sigmoid(-tau * (mood - mid_paired))
    return w / (w.mean() + 1e-12)


def marginal_weights(mid, mood, tau):
    """Marginalise the pair weights onto single molecules: u_o = E_i[w_io], u_i = E_o[w_io].
    A pointwise loss with these weights is 'hard-example weighting derived from the same
    reference', which is the control that decides whether the pairwise structure matters."""
    W = torch.sigmoid(-tau * (mood.unsqueeze(1) - mid.unsqueeze(0)))     # [n_ood, n_id]
    u_ood = W.mean(1); u_id = W.mean(0)
    ess_global = float(W.sum() ** 2 / (W.numel() * (W.pow(2).sum() + 1e-12)))
    return u_id / (u_id.mean() + 1e-12), u_ood / (u_ood.mean() + 1e-12), ess_global


# ------------------------------- training -------------------------------
def train_residual(D, ctx, M, kind, alpha, lr, seed, tau_spec):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    Xid = torch.tensor(ctx["Xid"]); Xood = torch.tensor(ctx["Xood"])
    mid = torch.tensor(M["fit_id"], dtype=torch.float32)
    mood = torch.tensor(M["fit_ood"], dtype=torch.float32)

    tau, solved, ess_g, u_id, u_ood = None, True, None, None, None
    if kind in ("ref_rpo_w", "ref_weighted_bce"):
        mode, val = tau_spec
        tau, solved = calibrate_tau(mid, mood, seed, val) if mode == "ess" else (val, True)
        if kind == "ref_weighted_bce":
            u_id, u_ood, ess_g = marginal_weights(mid, mood, tau)

    head = RR.Head([Xid.shape[1]])
    opt = torch.optim.AdamW(head.parameters(), lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9)
    best, bs, mb_ess = -1, None, []

    def score(X, m):
        g = head([torch.tensor(X)])
        if kind == "fusion":                      # independently trained head -> z-score so
            with torch.no_grad(): gt = head([Xid])   # the fusion weight is on m's scale
            g = (g - gt.mean()) / (gt.std() + 1e-8)
        return torch.tensor(m, dtype=torch.float32) + alpha * g

    for ep in range(EPOCHS):
        head.train(); opt.zero_grad()
        gid, good = head([Xid]), head([Xood])
        sid, sood = mid + alpha * gid, mood + alpha * good
        if kind == "fusion":
            # the control the reference arms must beat: an ORIGINAL RPO head that never
            # sees the reference during training, combined with it afterwards by a
            # validation-selected linear weight.  Same head, same budget, same grid --
            # the only difference from ref_rpo is whether the pairwise loss is applied to
            # the TOTAL score or to the residual alone.  If fusion matches ref_rpo, the
            # contribution is plain score addition, not joint residual ranking.
            pair = torch.randint(len(gid), (len(good),), generator=g)
            core = F.softplus(-BETA * (good - gid[pair])).mean()
        elif kind == "ref_bce":
            core = (0.5 * F.binary_cross_entropy_with_logits(sid, torch.zeros(len(sid)))
                    + 0.5 * F.binary_cross_entropy_with_logits(sood, torch.ones(len(sood))))
        elif kind == "ref_weighted_bce":
            li = F.binary_cross_entropy_with_logits(sid, torch.zeros(len(sid)), reduction="none")
            lo = F.binary_cross_entropy_with_logits(sood, torch.ones(len(sood)), reduction="none")
            core = 0.5 * (u_id * li).mean() + 0.5 * (u_ood * lo).mean()
        else:
            pair = torch.randint(len(sid), (len(sood),), generator=g)
            per = F.softplus(-BETA * (sood - sid[pair]))
            if kind == "ref_rpo":
                core = per.mean()
            else:
                w = pair_weights(mood, mid[pair], tau).detach()
                mb_ess.append(float(w.sum() ** 2 / (len(w) * w.pow(2).sum())))
                core = (w * per).mean()
        (core + GAMMA * (gid.pow(2).mean() + good.pow(2).mean())).backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
        if (ep + 1) % 15 == 0:
            head.eval()
            with torch.no_grad():
                v = RR.mets(score(D["val_id"][0], M["val_id"]).numpy(),
                            score(D["val_ood"][0], M["val_ood"]).numpy())[0]
            if v > best: best, bs = v, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval()
    diag = {"tau": tau, "tau_solved": solved, "ess_global": ess_g,
            "mb_ess_median": float(np.median(mb_ess)) if mb_ess else None,
            "mb_ess_p5": float(np.quantile(mb_ess, 0.05)) if mb_ess else None}
    with torch.no_grad():
        return best, RR.mets(score(D["test_id"][0], M["test_id"]).numpy(),
                             score(D["test_ood"][0], M["test_ood"]).numpy()), diag


def run_arm(D, ctxs, Ms, kind, tau_spec):
    best = None
    for a in ALPHAS:
        for lr in LRS:                                   # 9 trials, same for every arm
            v = float(np.mean([train_residual(D, ctxs[s], Ms[s], kind, a, lr, s, tau_spec)[0]
                               for s in SEL]))
            if best is None or v > best[2]: best = (a, lr, v)
    a, lr, _ = best
    R, Dg = [], []
    for s in FIN:
        _, m, d = train_residual(D, ctxs[s], Ms[s], kind, a, lr, s, tau_spec)
        R.append(m); Dg.append(d)
    return R, {"alpha": a, "lr": lr, "tau_spec": list(tau_spec)}, Dg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--backbone", default="minimol")
    a = ap.parse_args()
    D, info = MO.load_cell(a.cell, a.backbone)
    lab = MO.label_map(a.cell)
    y_all = np.array([lab.get(s, -1) for s in D["train_id"][1]])
    ctxs = {s: MO.context(D, y_all, s) for s in set(SEL) | set(FIN)}

    hp_ref, val_ref = select_reference(D, ctxs)
    Ms = {}
    for s in ctxs:
        m = fit_reference(ctxs[s], hp_ref)
        Ms[s] = {k: m(D[k][0]) for k in RR.KEYS}
        Ms[s]["fit_id"] = m(ctxs[s]["Xid"]); Ms[s]["fit_ood"] = m(ctxs[s]["Xood"])
    ref_test = [RR.mets(Ms[s]["test_id"], Ms[s]["test_ood"]) for s in FIN]
    base = np.array([r[0] for r in ref_test])
    print(f"[{a.cell}/{a.backbone}] reference = OE-Mahalanobis {hp_ref}; "
          f"VALIDATION AUROC {val_ref:.4f} (selection only), TEST AUROC {base.mean():.4f} "
          f"(= ref_only). split={info}", flush=True)

    out = {"cell": a.cell, "backbone": a.backbone, "domain_split": info,
           "reference_hp": str(hp_ref), "reference_val_auroc": val_ref, "arms": {}}
    out["arms"]["ref_only"] = {"auroc": base.tolist(), "aupr": [r[1] for r in ref_test],
                               "fpr95": [r[2] for r in ref_test], "hp": "frozen reference",
                               "primary": True}
    print(f"  {'ref_only':24} AUROC {base.mean():.4f}", flush=True)

    P = ("ess", ESS_PRIMARY)
    jobs = [("ref_bce", "ref_bce", P), ("ref_weighted_bce", "ref_weighted_bce", P),
            ("fusion", "fusion", P),
            ("ref_rpo", "ref_rpo", P), ("ref_rpo_ess", "ref_rpo_w", P)]
    jobs += [(f"ref_rpo_ess{t:g}", "ref_rpo_w", ("ess", t)) for t in ESS_SENS]
    jobs += [(f"ref_rpo_tau{t:g}", "ref_rpo_w", ("fixed", t)) for t in TAUS_SENS]

    for name, kind, spec in jobs:
        R, hp, Dg = run_arm(D, ctxs, Ms, kind, spec)
        au = np.array([r[0] for r in R]); d = au - base
        out["arms"][name] = {"auroc": au.tolist(), "aupr": [r[1] for r in R],
                             "fpr95": [r[2] for r in R], "hp": hp, "diag": Dg,
                             "delta_vs_ref": float(d.mean()),
                             "pos_seeds": int((d > 0).sum()),
                             "primary": name in ("ref_bce", "ref_weighted_bce", "fusion",
                                                 "ref_rpo", "ref_rpo_ess")}
        t = [x["tau"] for x in Dg if x["tau"] is not None]
        me = [x["mb_ess_median"] for x in Dg if x["mb_ess_median"] is not None]
        p5 = [x["mb_ess_p5"] for x in Dg if x["mb_ess_p5"] is not None]
        eg = [x["ess_global"] for x in Dg if x["ess_global"] is not None]
        extra = ""
        if t: extra += f"  tau={np.mean(t):.3f}"
        if eg: extra += f" ESSglob={np.mean(eg):.3f}"
        if me: extra += f" mbESS med={np.mean(me):.3f} p5={np.mean(p5):.3f}"
        if t and not all(x["tau_solved"] for x in Dg): extra += "  [TAU UNSOLVED]"
        print(f"  {name:24} AUROC {au.mean():.4f}  Δvs-ref {d.mean():+.4f} "
              f"({int((d > 0).sum())}/5)  a={hp['alpha']} lr={hp['lr']}{extra}", flush=True)

    Path("repro").mkdir(exist_ok=True)
    json.dump(out, open(f"repro/refrpo_{a.cell}_{a.backbone}.json", "w"), indent=2)
    print(f"REFRPO_DONE {a.cell} {a.backbone}", flush=True)


if __name__ == "__main__":
    main()
