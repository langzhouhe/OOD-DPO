#!/usr/bin/env python3
"""MePOE Gate 1: target-free, CROSS-ENDPOINT preference-optimised OE acquisition.

Train a mechanism-utility policy from subset preferences measured on a SOURCE endpoint,
then apply it zero-shot to a TARGET endpoint -- never touching the target's validation or
test OOD. Detector is always a fixed Balanced BCE, so any difference comes from WHICH
mechanisms were acquired.

Honesty note on the objective: for equal-budget subsets under a UNIFORM reference policy,
log pi_ref(A) = log pi_ref(B), so the DPO objective collapses exactly to a Bradley-Terry
loss on the set utility U(S) = mean_{m in S} u(m). The reference/KL term only becomes
non-trivial for unequal budgets or a non-uniform prior. We therefore keep a variance
trust-region tau*Var(u) as the reference analogue and do NOT claim the DPO form itself is
the contribution -- the content is the additive set-utility parameterisation over
endpoint-invariant mechanism features plus cross-endpoint transfer.

Mechanism features are all z-scored against the endpoint's OWN ID distribution, so they
carry no endpoint-specific scale and can transfer.

Selectors compared at equal budget:
  random / closest / farthest / diverse(k-means) / reward-regression / preference(BT)
  plus the zero-OOD Mahalanobis reference, because a selector that wins the OE
  sub-competition while losing to a method that uses no OOD at all has won nothing.

Usage: python mepoe_gate1.py --source ec50_assay --target ic50_assay
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, argparse, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from multiprocessing import Pool
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import revision_run as RR
import mepoe_gate0 as MG

N_SUBSET, SEEDS, BUDGET = 128, (1, 2, 3), 25
TAU = 0.01
_C = {}


def mech_features(d):
    """Endpoint-invariant descriptors of each source mechanism (ID-relative, z-scored)."""
    tr = d["id_train"].numpy()
    mu = tr.mean(0); P = np.linalg.pinv(np.cov(tr, rowvar=False) + 1e-3 * np.eye(tr.shape[1]))
    md = lambda A: np.einsum("ij,jk,ik->i", A - mu, P, A - mu)
    nn_ = NearestNeighbors(n_neighbors=50).fit(tr)
    lof_ = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(tr)
    ref = {"md": md(tr), "kn": nn_.kneighbors(tr)[0].mean(1), "lf": -lof_.score_samples(tr)}
    z = lambda v, k: (v - ref[k].mean()) / (ref[k].std() + 1e-9)
    X, mech = d["S"].numpy(), d["S_mech"]
    cen_id = tr.mean(0)
    rows = []
    for m in np.unique(mech):
        A = X[mech == m]
        c = A.mean(0)
        dc = A - c
        rows.append([z(md(A), "md").mean(), z(md(A), "md").std(),
                     z(nn_.kneighbors(A)[0].mean(1), "kn").mean(),
                     z(-lof_.score_samples(A), "lf").mean(),
                     np.linalg.norm(c - cen_id) / (np.linalg.norm(cen_id) + 1e-9),
                     np.sqrt((dc ** 2).sum(1)).mean(),
                     float(np.median(md(A)))])
    Fm = np.asarray(rows, dtype=np.float64)
    Fm[:, -1] = (Fm[:, -1] - ref["md"].mean()) / (ref["md"].std() + 1e-9)
    return (Fm - Fm.mean(0)) / (Fm.std(0) + 1e-9), md, tr


def _eval(job):
    rows, seed = job
    return MG.run_one((rows, seed))


def subset_utilities(cell, seed=21, workers=22):
    """Measure each random subset's transfer utility on the source's own H1."""
    d, _ = MG.setup(cell, seed); MG.G.clear(); MG.G.update(d)
    mech = d["S_mech"]; n = int(mech.max()) + 1
    rng = np.random.default_rng(1000 + BUDGET)
    subs = [np.sort(rng.choice(n, size=BUDGET, replace=False)) for _ in range(N_SUBSET)]
    jobs = [(np.where(np.isin(mech, s))[0], sd) for s in subs for sd in SEEDS]
    with Pool(workers) as p:
        r = p.map(_eval, jobs, chunksize=1)
    A = np.array(r).reshape(N_SUBSET, len(SEEDS), 2)
    return subs, A[:, :, 0].mean(1), d


def fit_preference(Fm, subs, util, epochs=800, beta=4.0):
    """Bradley-Terry on set utility (== DPO with a uniform reference at equal budget)."""
    Fm_t = torch.tensor(Fm, dtype=torch.float32)
    M = torch.zeros(len(subs), len(Fm))
    for i, s in enumerate(subs): M[i, s] = 1.0 / len(s)
    u_t = torch.tensor(util, dtype=torch.float32)
    ii, jj = np.triu_indices(len(subs), 1)
    keep = np.abs(util[ii] - util[jj]) >= np.quantile(np.abs(util[ii] - util[jj]), 0.5)
    ii, jj = ii[keep], jj[keep]
    win = torch.tensor(np.where(u_t[ii] > u_t[jj], 1.0, -1.0), dtype=torch.float32)
    net = nn.Sequential(nn.Linear(Fm.shape[1], 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-4)
    torch.manual_seed(0)
    for _ in range(epochs):
        opt.zero_grad()
        u = net(Fm_t).squeeze(-1)
        U = M @ u
        loss = F.softplus(-beta * win * (U[ii] - U[jj])).mean() + TAU * u.var()
        loss.backward(); opt.step()
    with torch.no_grad():
        return net(Fm_t).squeeze(-1).numpy(), net


def selectors(Fm, md_fn, d, k, u_pref, u_reward):
    mech = d["S_mech"]; X = d["S"].numpy()
    mdm = np.array([md_fn(X[mech == m]).mean() for m in np.unique(mech)])
    cen = np.stack([X[mech == m].mean(0) for m in np.unique(mech)])
    km = KMeans(k, n_init=10, random_state=0).fit(cen)
    div = [int(np.argmin(((cen - c) ** 2).sum(1))) for c in km.cluster_centers_]
    return {"closest": np.argsort(mdm)[:k], "farthest": np.argsort(-mdm)[:k],
            "diverse": np.array(sorted(set(div))), "reward_reg": np.argsort(-u_reward)[:k],
            "preference": np.argsort(-u_pref)[:k]}


def evaluate(d, mech_ids, seeds=SEEDS):
    rows = np.where(np.isin(d["S_mech"], list(mech_ids)))[0]
    r = np.array([MG.run_one((rows, s)) for s in seeds])
    return r[:, 0], r[:, 1]        # per-seed AUROC on H1, H2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--workers", type=int, default=22)
    a = ap.parse_args()

    subs, util, d_src = subset_utilities(a.source, workers=a.workers)
    Fs, _, _ = mech_features(d_src)
    u_pref, net = fit_preference(Fs, subs, util)
    M = np.zeros((len(subs), len(Fs)))
    for i, s in enumerate(subs): M[i, s] = 1.0 / len(s)
    u_rew_net = Ridge(alpha=1.0).fit(M @ Fs, util)
    u_reward_src = Fs @ u_rew_net.coef_

    d_tgt, _ = MG.setup(a.target, 21); MG.G.clear(); MG.G.update(d_tgt)
    Ft, md_t, tr_t = mech_features(d_tgt)
    with torch.no_grad():
        u_pref_t = net(torch.tensor(Ft, dtype=torch.float32)).squeeze(-1).numpy()
    u_rew_t = Ft @ u_rew_net.coef_

    sel = selectors(Ft, md_t, d_tgt, BUDGET, u_pref_t, u_rew_t)
    res = {}
    for name, ids in sel.items():
        h1, h2 = evaluate(d_tgt, ids)
        res[name] = {"H1": h1.tolist(), "H2": h2.tolist(),
                     "mean_H2": float(h2.mean()), "mean_H1": float(h1.mean())}
    rng = np.random.default_rng(7)
    n_mech = int(d_tgt["S_mech"].max()) + 1
    rnd = [evaluate(d_tgt, rng.choice(n_mech, BUDGET, replace=False))[1].mean() for _ in range(12)]
    res["random"] = {"mean_H2": float(np.mean(rnd)), "sd": float(np.std(rnd)), "n_draws": 12}

    # zero-OOD reference on the target
    mu = tr_t.mean(0); P = np.linalg.pinv(np.cov(tr_t, rowvar=False) + 1e-3 * np.eye(tr_t.shape[1]))
    mdf = lambda A: np.einsum("ij,jk,ik->i", A - mu, P, A - mu)
    Xun, mun, off = [], [], 0
    for n_ in ("H1", "H2"):
        Xun.append(d_tgt[n_].numpy()); mun.append(d_tgt[n_ + "_mech"] + off)
        off += int(d_tgt[n_ + "_mech"].max()) + 1
    Xun = np.concatenate(Xun); mun = np.concatenate(mun)
    h2rows = np.isin(mun, np.unique(d_tgt["H2_mech"] + (int(d_tgt["H1_mech"].max()) + 1)))
    res["ZERO_OOD_maha"] = {"mean_H2": float(RR.mets(mdf(d_tgt["id_test"].numpy()),
                                                     mdf(Xun[h2rows]))[0])}

    best_oe = max(v["mean_H2"] for k, v in res.items() if k not in ("preference", "ZERO_OOD_maha"))
    pref = res["preference"]["mean_H2"]
    gate = {"vs_best_selector": pref - best_oe,
            "vs_reward_reg": pref - res["reward_reg"]["mean_H2"],
            "vs_zero_ood": pref - res["ZERO_OOD_maha"]["mean_H2"],
            "pass_selector": pref - best_oe >= 0.010,
            "pass_reward": pref - res["reward_reg"]["mean_H2"] >= 0.005,
            "pass_zero_ood": pref - res["ZERO_OOD_maha"]["mean_H2"] > 0}
    out = {"source": a.source, "target": a.target, "budget": BUDGET, "res": res, "gate": gate}
    print(f"\n[{a.source} -> {a.target}] budget {BUDGET} mechanisms, AUROC on target H2", flush=True)
    for k in ("random", "closest", "farthest", "diverse", "reward_reg", "preference", "ZERO_OOD_maha"):
        print(f"    {k:14s} {res[k]['mean_H2']:.4f}", flush=True)
    print(f"    gate: vs best selector {gate['vs_best_selector']:+.4f} (need >=+0.010) "
          f"{'PASS' if gate['pass_selector'] else 'FAIL'}", flush=True)
    print(f"          vs reward-reg    {gate['vs_reward_reg']:+.4f} (need >=+0.005) "
          f"{'PASS' if gate['pass_reward'] else 'FAIL'}", flush=True)
    print(f"          vs ZERO-OOD      {gate['vs_zero_ood']:+.4f} (need >0)       "
          f"{'PASS' if gate['pass_zero_ood'] else 'FAIL'}", flush=True)
    json.dump(out, open(f"repro/gate1_{a.source}_to_{a.target}.json", "w"), indent=2)
    print(f"GATE1_DONE {a.source}->{a.target}", flush=True)


if __name__ == "__main__":
    main()
