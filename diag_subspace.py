#!/usr/bin/env python3
"""D2 (shortcut-subspace ablation) + D3 (effective dimension of the score).

Reuses the E17 Phase-0 protocol by importing revision_run (equal 3x3 tuning grid,
domain-disjoint validation OOD, parameter-matched head, validation-only selection,
conventional FPR95) so the protocol cannot drift.

D2 arms -- all use the BalancedOODHead (BCE) objective, because E17 established the
two objectives are statistically tied and this diagnostic is about the REPRESENTATION:

  full         frozen MiniMol 512-d, unmodified                    (reproduces E17 cell)
  resid_lin    linear span of the 10 trivial descriptors projected out
  resid_poly2  degree-2 descriptor span (66 terms) projected out
  desc_only    the 10 descriptors alone, same trained head          (trivial upper bound)
  resid_rand   a RANDOM subspace of the same dim as resid_lin       (falsification control)
  resid_pca    the top-k principal components removed, k = dim(resid_lin) (control)

The two controls are what make the ablation interpretable: if removing an arbitrary
subspace of equal size costs as much AUROC as removing the descriptor subspace, then
the descriptor result says nothing about shortcuts. Variance removed is reported for
every arm so the comparison is auditable.

Residualisation is LABEL-FREE: fitted on train_id + train_ood pooled without using the
ID/OOD label, then applied unchanged to val and test.

D3, computed on the `full` arm's final-seed heads:
  pr            participation ratio of the input-gradient spectrum (effective #directions)
  auroc_top1/3  AUROC recoverable after projecting features onto the top 1/3 gradient PCs
  auroc_linear  AUROC of a plain logistic detector on the same features
  r2_desc/poly2 fraction of the MLP score explained by the trivial descriptors
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, argparse, warnings, logging
import numpy as np, torch
torch.set_num_threads(1)
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import revision_run as RR

KEYS = RR.KEYS
ARMS = ["full", "resid_lin", "resid_poly2", "desc_only", "resid_rand", "resid_pca"]


# ---------------- descriptor design matrices ----------------
def poly2(D):
    """[linear | squares | pairwise cross terms] of standardised descriptors."""
    n, d = D.shape
    cols = [D, D ** 2]
    cols += [(D[:, i] * D[:, j])[:, None] for i in range(d) for j in range(i + 1, d)]
    return np.concatenate(cols, 1)


def design(data, expand):
    """Standardise descriptors on the pooled TRAIN molecules (label-free), then expand."""
    pool = np.concatenate([data["train_id"]["de"], data["train_ood"]["de"]], 0)
    mu, sd = pool.mean(0), pool.std(0) + 1e-6
    out = {}
    for k in KEYS:
        Z = (data[k]["de"] - mu) / sd
        Z = poly2(Z) if expand else Z
        out[k] = np.concatenate([Z, np.ones((len(Z), 1), dtype=np.float32)], 1).astype(np.float32)
    return out


def residualise(data, block, X):
    """Project the descriptor-predictable component out of `block`. Ridge-regularised
    least squares fitted on pooled train (no labels), applied to every split."""
    A = np.concatenate([X["train_id"], X["train_ood"]], 0)
    Y = np.concatenate([data["train_id"][block], data["train_ood"][block]], 0)
    G = A.T @ A + 1e-3 * np.eye(A.shape[1], dtype=np.float64)
    W = np.linalg.solve(G, A.T @ Y.astype(np.float64))
    out = {k: (data[k][block] - X[k] @ W).astype(np.float32) for k in KEYS}
    return out, A.shape[1] - 1          # -1: intercept removes no direction


def remove_subspace(data, block, B):
    """Project out the column space of B (orthonormalised)."""
    Q, _ = np.linalg.qr(B)
    return {k: (data[k][block] - (data[k][block] @ Q) @ Q.T).astype(np.float32) for k in KEYS}


def var_kept(data, new, block):
    """Fraction of test-set feature variance surviving the projection."""
    a = data["test_id"][block].var(0).sum() + data["test_ood"][block].var(0).sum()
    b = new["test_id"].var(0).sum() + new["test_ood"].var(0).sum()
    return float(b / max(a, 1e-12))


# ---------------- Phase-0 training on an arbitrary block list ----------------
def make_V(data, parts):
    st = {p: (data["train_id"][p].mean(0), data["train_id"][p].std(0) + 1e-6) for p in parts}
    return {k: [torch.tensor((data[k][p] - st[p][0]) / st[p][1]) for p in parts] for k in KEYS}


def run_arm(V, kind="bce"):
    """Identical 9-trial validation-only selection as revision_run.run_objective,
    but also returns the final-seed heads so D3 can reuse them."""
    best = None
    for gm in RR.GAMMAS:
        for lr in RR.LRS:
            v = float(np.mean([RR.train(V, kind, gm, lr, s)[0] for s in RR.SEL]))
            if best is None or v > best[2]: best = (gm, lr, v)
    gm, lr, _ = best
    R, heads = [], []
    for s in RR.FIN:
        _, m, h = RR.train(V, kind, gm, lr, s)
        R.append(m); heads.append(h)
    return R, {"gamma": gm, "lr": lr}, heads


# ---------------- D3 ----------------
def d3(V, heads, data, seeds, n_id=1500, n_ood=2000):
    """Every probe is FIT ON TRAIN and SCORED ON TEST, using the same ID/OOD subsample
    the corresponding head was trained on -- an in-sample probe on 512 dims and ~2k test
    points overfits badly and would report a fictitious linear baseline.
    The participation ratio involves no fitting and is a property of the score itself."""
    Xi, Xo = V["test_id"][0], V["test_ood"][0]
    Xte = torch.cat([Xi, Xo], 0)
    yte = np.r_[np.zeros(len(Xi)), np.ones(len(Xo))]
    Xd = {False: design(data, False), True: design(data, True)}
    r = {k: [] for k in ("pr", "auroc_top1", "auroc_top3", "auroc_linear", "r2_desc", "r2_poly2")}
    r["dim"] = int(Xte.shape[1])
    for h, seed in zip(heads, seeds):
        h.eval()
        g = torch.Generator().manual_seed(seed)           # replicates revision_run.train
        iid = torch.randperm(len(V["train_id"][0]), generator=g)[:n_id]
        iod = torch.randperm(len(V["train_ood"][0]), generator=g)[:n_ood]
        Xtr = torch.cat([V["train_id"][0][iid], V["train_ood"][0][iod]], 0)
        ytr = np.r_[np.zeros(len(iid)), np.ones(len(iod))]

        Xg = Xte.clone().requires_grad_(True)
        h([Xg]).sum().backward()
        G = Xg.grad.detach().numpy()                       # n_test x d input gradients
        e = np.linalg.svd(G, compute_uv=False) ** 2
        r["pr"].append(float(e.sum() ** 2 / (e ** 2).sum()))
        _, _, Vt = np.linalg.svd(G, full_matrices=False)

        for k, key in ((1, "auroc_top1"), (3, "auroc_top3")):
            P = Vt[:k]
            m = LogisticRegression(max_iter=5000).fit(Xtr.numpy() @ P.T, ytr)
            r[key].append(roc_auc_score(yte, m.decision_function(Xte.numpy() @ P.T)))
        m = LogisticRegression(max_iter=5000).fit(Xtr.numpy(), ytr)
        r["auroc_linear"].append(roc_auc_score(yte, m.decision_function(Xte.numpy())))

        with torch.no_grad():
            sc_tr, sc_te = h([Xtr]).numpy(), h([Xte]).numpy()
        for expand, key in ((False, "r2_desc"), (True, "r2_poly2")):
            Atr = np.concatenate([Xd[expand]["train_id"][iid.numpy()],
                                  Xd[expand]["train_ood"][iod.numpy()]], 0).astype(np.float64)
            Ate = np.concatenate([Xd[expand]["test_id"], Xd[expand]["test_ood"]], 0).astype(np.float64)
            w = np.linalg.solve(Atr.T @ Atr + 1e-3 * np.eye(Atr.shape[1]), Atr.T @ sc_tr)
            resid = sc_te - Ate @ w
            r[key].append(float(1 - resid.var() / max(sc_te.var(), 1e-12)))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--block", default="fm", choices=["fm", "um"])
    ap.add_argument("--arms", nargs="+", default=ARMS)
    a = ap.parse_args()

    data, info = RR.load(a.cell, a.block == "um")
    out = {"domain_split": info, "block": a.block, "arms": {}}
    rng = np.random.default_rng(12345)

    # descriptor-residual blocks (also fixes k for the two controls)
    r_lin, k_lin = residualise(data, a.block, design(data, False))
    r_p2, k_p2 = residualise(data, a.block, design(data, True))
    d = data["train_id"][a.block].shape[1]
    r_rand = remove_subspace(data, a.block, rng.standard_normal((d, min(k_lin, d - 1))))
    pool = np.concatenate([data["train_id"][a.block], data["train_ood"][a.block]], 0)
    pool = pool - pool.mean(0)
    _, _, Vt = np.linalg.svd(pool, full_matrices=False)
    r_pca = remove_subspace(data, a.block, Vt[:min(k_lin, d - 1)].T)

    variants = {"resid_lin": (r_lin, k_lin), "resid_poly2": (r_p2, k_p2),
                "resid_rand": (r_rand, min(k_lin, d - 1)), "resid_pca": (r_pca, min(k_lin, d - 1))}

    for arm in a.arms:
        if arm == "full":
            dd, parts, removed, vk = data, [a.block], 0, 1.0
        elif arm == "desc_only":
            dd, parts, removed, vk = data, ["de"], 0, 1.0
        else:
            new, removed = variants[arm]
            vk = var_kept(data, new, a.block)
            dd = {k: dict(data[k]) for k in KEYS}
            for k in KEYS: dd[k][a.block] = new[k]
            parts = [a.block]
        V = make_V(dd, parts)
        R, hp, heads = run_arm(V)
        rec = {"auroc": [x[0] for x in R], "aupr": [x[1] for x in R], "fpr95": [x[2] for x in R],
               "hp": hp, "dims_removed": int(removed), "var_kept": vk}
        if arm == "full":
            rec["d3"] = d3(V, heads, data, RR.FIN)
        out["arms"][arm] = rec
        print(f"[{a.cell}/{a.block}] {arm:12} AUROC={np.mean(rec['auroc']):.4f} "
              f"FPR95={np.mean(rec['fpr95']):.3f} removed={removed} var_kept={vk:.3f}", flush=True)

    # merge into any existing file so re-running a subset of arms never clobbers the rest
    path = f"repro/diag_{a.cell}_{a.block}.json"
    if os.path.exists(path):
        prev = json.load(open(path))
        prev.get("arms", {}).update(out["arms"]); prev.update({k: v for k, v in out.items() if k != "arms"})
        out = prev
    json.dump(out, open(path, "w"), indent=2)
    print(f"DIAG_DONE {a.cell} {a.block} arms={sorted(out['arms'])}", flush=True)


if __name__ == "__main__":
    main()
