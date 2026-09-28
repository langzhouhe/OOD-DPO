#!/usr/bin/env python3
"""Context-Matched RPO: same architecture, same loss, only the pairing distribution changes.

Every arm sees the IDENTICAL molecules (1500 ID anchors, 2000 auxiliary anchors), the
identical head, the identical optimiser, and the identical nine-trial (gamma x lr) budget.
They differ only in what they are allowed to know about how those molecules are related.

Frozen training distribution
    P_CM(i,j) = 0.75 * 1[(i,j) in E]/|E|  +  0.25 * 1/(N_I*N_O)
with the ANALYTIC marginals
    mu_i = 0.75*d_i/|E| + 0.25/N_I ,   nu_j = 0.75*d_j/|E| + 0.25/N_O
so a pointwise control can be handed exactly the same molecule emphasis instead of an
emphasis estimated by counting simulated epochs.

  bce           balanced BCE                       knows nothing
  mp_bce        marginal-weighted BCE              knows mu, nu exactly
  mp_ctx_bce    marginal-weighted BCE + context    knows mu, nu AND gets free additive main
                main effects, dropped at test      effects for protein / activity class /
                                                   pChEMBL bin
  mp_rpo        pairwise, i ~ mu and j ~ nu drawn  same marginals, NO coupling
                independently
  broken_rpo    pairwise on the rewired edge set   same degrees, same source mix, same
                                                   chemical-distance distribution, protein
                                                   context destroyed
  cm_rpo        pairwise on the real edge set      the full method
  random_rpo    pairwise, uniform product coupling the original paper's pairing

`mp_ctx_bce` uses SHARED main effects, not one intercept per sparse interaction block: with
a free intercept per block, a singleton block reduces exactly to a pairwise logistic in
(r_o - r_i) at a different temperature, so it would not be an independent pointwise
control at all.
`broken_rpo`, not the within-block shuffle, is the decisive negative control -- a within-block
shuffle keeps the same protein and bin, so it would still cancel the context offset, and most
blocks are singletons and cannot be shuffled at all.  The within-block shuffle is reported in
the appendix only.

Model selection: target iid_val (ID) vs the assay-disjoint auxiliary validation pool (OOD).
Final evaluation: target iid_test (ID) vs the target's official ood_test (OOD), once.

Usage: python cmrpo.py --target ec50 --backbone minimol
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, pickle, argparse, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import revision_run as RR
import matched_oe as MO

DATA = Path("cache/cmrpo")
ARMS = ["bce", "mp_bce", "mp_ctx_bce", "mp_rpo", "broken_rpo", "cm_rpo", "random_rpo"]
OE_BASELINES = ["MSP-OE", "ODIN-OE", "Energy-OE", "OE-Mahalanobis", "OE-KNN", "OE-LOF"]
MATCHED_FRAC = 0.75          # frozen in advance, never tuned
PAIRS_PER_STEP = 2000
BETA = 1.0
EPOCHS = 120
SEL, FIN = RR.SEL, RR.FIN
GAMMAS, LRS = RR.GAMMAS, RR.LRS


def load(target, backbone):
    rec = json.load(open(DATA / f"{target}_cmrpo.json"))
    fp = {}
    f2 = DATA / f"{target}_{backbone}_features.pkl"
    if f2.exists(): fp.update(pickle.load(open(f2, "rb"))["features"])
    for t in ("ec50", "ic50", "ki"):
        g = RR.CACHE / f"lbap_general_{t}_assay_{backbone}_features.pkl"
        if g.exists():
            for k, v in pickle.load(open(g, "rb"))["features"].items(): fp.setdefault(k, v)

    keys = ["id_train", "ood_train", "val_id", "val_ood", "test_id", "test_ood"]
    idx = {k: [n for n, s in enumerate(rec[k]) if s in fp] for k in keys}
    X = {k: np.stack([fp[rec[k][n]] for n in idx[k]]).astype(np.float32) for k in keys}
    mu = X["id_train"].mean(0); sd = X["id_train"].std(0) + 1e-6
    X = {k: (v - mu) / sd for k, v in X.items()}

    remap_i = {o: n for n, o in enumerate(idx["id_train"])}
    remap_o = {o: n for n, o in enumerate(idx["ood_train"])}
    def keep(E):
        return np.array([[remap_i[i], remap_o[j]] for i, j, *_ in E
                         if i in remap_i and j in remap_o], dtype=np.int64)
    E = keep(rec["edges"]); EB = keep(rec["edges_broken"])
    ctx_i = np.array([rec["id_ctx"][n] for n in idx["id_train"]], dtype=np.int64)
    ctx_o = np.array([rec["ood_ctx"][n] for n in idx["ood_train"]], dtype=np.int64)
    lab = np.array([rec["id_labels"][n] for n in idx["id_train"]], dtype=np.int64)
    return X, E, EB, ctx_i, ctx_o, lab, rec


def marginals(E, n_i, n_o):
    """The exact marginals of the frozen mixture -- not counted from simulated epochs."""
    d_i = np.bincount(E[:, 0], minlength=n_i).astype(np.float64)
    d_o = np.bincount(E[:, 1], minlength=n_o).astype(np.float64)
    m = MATCHED_FRAC * d_i / max(len(E), 1) + (1 - MATCHED_FRAC) / n_i
    n = MATCHED_FRAC * d_o / max(len(E), 1) + (1 - MATCHED_FRAC) / n_o
    return m / m.sum(), n / n.sum()


class CtxHead(nn.Module):
    """RR.Head plus additive, SHARED main effects for protein / activity class / bin.
    The main effects are training-time nuisance absorbers and are dropped at test time."""
    def __init__(s, dim, sizes):
        super().__init__()
        s.h = RR.Head([dim])
        s.fx = nn.ModuleList([nn.Embedding(n + 1, 1) for n in sizes])
        for e in s.fx: nn.init.zeros_(e.weight)
    def forward(s, x, ctx=None):
        r = s.h([x])
        if ctx is None: return r
        return r + sum(e(ctx[:, k] + 1).squeeze(-1) for k, e in enumerate(s.fx))


def train(X, E, EB, ctx_i, ctx_o, arm, gamma, lr, seed, sizes, mu, nu):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    Xid = torch.tensor(X["id_train"]); Xood = torch.tensor(X["ood_train"])
    n_i, n_o = len(Xid), len(Xood)
    Ci = torch.tensor(ctx_i); Co = torch.tensor(ctx_o)
    use_ctx = arm == "mp_ctx_bce"
    head = CtxHead(Xid.shape[1], sizes) if use_ctx else RR.Head([Xid.shape[1]])
    opt = torch.optim.AdamW(head.parameters(), lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9)
    w_i = torch.tensor(mu * n_i, dtype=torch.float32)      # mean-1 normalised
    w_o = torch.tensor(nu * n_o, dtype=torch.float32)
    edge = {"cm_rpo": E, "broken_rpo": EB}.get(arm)
    n_m = int(round(MATCHED_FRAC * PAIRS_PER_STEP))
    best, bs = -1, None

    def raw(A):
        A = torch.as_tensor(A)
        return head(A, None) if use_ctx else head([A])

    for ep in range(EPOCHS):
        head.train(); opt.zero_grad()
        Eid = head(Xid, Ci) if use_ctx else head([Xid])
        Eood = head(Xood, Co) if use_ctx else head([Xood])
        if arm in ("bce",):
            core = (0.5 * F.binary_cross_entropy_with_logits(Eid, torch.zeros(n_i))
                    + 0.5 * F.binary_cross_entropy_with_logits(Eood, torch.ones(n_o)))
        elif arm in ("mp_bce", "mp_ctx_bce"):
            li = F.binary_cross_entropy_with_logits(Eid, torch.zeros(n_i), reduction="none")
            lo = F.binary_cross_entropy_with_logits(Eood, torch.ones(n_o), reduction="none")
            core = 0.5 * (w_i * li).mean() + 0.5 * (w_o * lo).mean()
        else:
            if arm == "random_rpo":
                I = torch.randint(n_i, (PAIRS_PER_STEP,), generator=g)
                J = torch.randint(n_o, (PAIRS_PER_STEP,), generator=g)
            elif arm == "mp_rpo":                       # same marginals, no coupling
                I = torch.multinomial(torch.tensor(mu, dtype=torch.float32),
                                      PAIRS_PER_STEP, replacement=True, generator=g)
                J = torch.multinomial(torch.tensor(nu, dtype=torch.float32),
                                      PAIRS_PER_STEP, replacement=True, generator=g)
            else:
                k = torch.randint(len(edge), (n_m,), generator=g)
                ek = torch.tensor(edge, dtype=torch.long)[k]
                I = torch.cat([ek[:, 0], torch.randint(n_i, (PAIRS_PER_STEP - n_m,), generator=g)])
                J = torch.cat([ek[:, 1], torch.randint(n_o, (PAIRS_PER_STEP - n_m,), generator=g)])
            core = F.softplus(-BETA * (Eood[J] - Eid[I])).mean()
        (core + gamma * (Eid.pow(2).mean() + Eood.pow(2).mean())).backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
        if (ep + 1) % 15 == 0:
            head.eval()
            with torch.no_grad():
                v = RR.mets(raw(X["val_id"]).numpy(), raw(X["val_ood"]).numpy())[0]
            if v > best: best, bs = v, {k2: t.clone() for k2, t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval()
    with torch.no_grad():                    # main effects are dropped at test time
        return best, RR.mets(raw(X["test_id"]).numpy(), raw(X["test_ood"]).numpy())


def run_arm(X, E, EB, ctx_i, ctx_o, arm, sizes, mu, nu):
    best = None
    for gm in GAMMAS:
        for lr in LRS:
            v = float(np.mean([train(X, E, EB, ctx_i, ctx_o, arm, gm, lr, s, sizes, mu, nu)[0]
                               for s in SEL]))
            if best is None or v > best[2]: best = (gm, lr, v)
    gm, lr, _ = best
    return [train(X, E, EB, ctx_i, ctx_o, arm, gm, lr, s, sizes, mu, nu)[1] for s in FIN], \
           {"gamma": gm, "lr": lr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--backbone", default="minimol")
    a = ap.parse_args()
    X, E, EB, ctx_i, ctx_o, lab, rec = load(a.target, a.backbone)
    n_i, n_o = len(X["id_train"]), len(X["ood_train"])
    mu, nu = marginals(E, n_i, n_o)
    sizes = [int(max(ctx_i[:, k].max(), ctx_o[:, k].max())) + 1 for k in range(3)]
    print(f"[{a.target}/{a.backbone}] ID {n_i} OOD {n_o} edges {len(E)} broken {len(EB)} "
          f"ctx sizes {sizes} labelled {int((lab >= 0).sum())}", flush=True)

    out = {"target": a.target, "backbone": a.backbone, "n_id": n_i, "n_ood": n_o,
           "n_edges": int(len(E)), "audit": rec["audit"], "arms": {}}
    for arm in ARMS:
        R, hp = run_arm(X, E, EB, ctx_i, ctx_o, arm, sizes, mu, nu)
        au = np.array([r[0] for r in R])
        out["arms"][arm] = {"auroc": au.tolist(), "aupr": [r[1] for r in R],
                            "fpr95": [r[2] for r in R], "hp": hp}
        print(f"  {arm:14} AUROC {au.mean():.4f} ±{au.std():.4f}  "
              f"FPR95 {np.mean([r[2] for r in R]):.3f}  hp={hp}", flush=True)

    D = {k: (X[k], None) for k in ("train_id", "train_ood", "val_id", "val_ood",
                                   "test_id", "test_ood")} if False else {
        "train_id": (X["id_train"], None), "train_ood": (X["ood_train"], None),
        "val_id": (X["val_id"], None), "val_ood": (X["val_ood"], None),
        "test_id": (X["test_id"], None), "test_ood": (X["test_ood"], None)}
    ctxs = {s: MO.context(D, lab, s) for s in set(SEL) | set(FIN)}
    ok = (lab >= 0).sum() > 50 and len(np.unique(lab[lab >= 0])) > 1
    for name in OE_BASELINES:
        if name in MO.NEEDS_LABEL and not ok:
            print(f"  {name:14} SKIPPED (no usable cls_label)", flush=True); continue
        R, hp, _ = MO.run_method(name, D, lab, ctxs)
        au = np.array([r[0] for r in R])
        out["arms"][name] = {"auroc": au.tolist(), "aupr": [r[1] for r in R],
                             "fpr95": [r[2] for r in R], "hp": str(hp)}
        print(f"  {name:14} AUROC {au.mean():.4f} ±{au.std():.4f}  "
              f"FPR95 {np.mean([r[2] for r in R]):.3f}  hp={hp}", flush=True)

    Path("repro").mkdir(exist_ok=True)
    json.dump(out, open(f"repro/cmrpo_{a.target}_{a.backbone}.json", "w"), indent=2)
    print(f"CMRPO_DONE {a.target} {a.backbone}", flush=True)


if __name__ == "__main__":
    main()
