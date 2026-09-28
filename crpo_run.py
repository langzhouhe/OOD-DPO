#!/usr/bin/env python3
"""C-RPO: does replacing RPO's PRODUCT coupling with a chemical OT coupling help?

RPO's loss is E_{i~p, o~q}[softplus(-beta*(s_o - s_i))] -- the ID and OOD draws are
INDEPENDENT, i.e. the coupling is the product measure p (x) q.  Under a product
coupling the pairwise risk decomposes and shares balanced BCE's Bayes-optimal
ranking (revision doc S3), which is why E17 found the two statistically tied.
The only way a pairwise loss can express something pointwise BCE cannot is a
NON-PRODUCT coupling pi(x_i, x_o) that does not factor into two sample weights.

Arms (identical head, identical molecules, identical 9-trial grid, val-only selection):
  bce       balanced BCE (Balanced OOD Head)                       -- the incumbent
  rpo       E17's sampled pairwise RPO (1 random ID per OOD/step)  -- continuity check
  rpo_prod  all-pairs with UNIFORM product coupling                -- isolates all-pairs
  ot_fixed  all-pairs with entropic-OT coupling on Tanimoto cost   -- chemical C-RPO
  ot_adv    all-pairs, coupling re-solved adversarially vs current
            loss, anchored to the chemical cost                    -- adversarial C-RPO
  ot_shuf   all-pairs with OT coupling on a ROW/COL-PERMUTED cost  -- FALSIFICATION

ot_shuf is the load-bearing control: it has the identical coupling structure
(uniform marginals, identical entropy/ESS, identical value spectrum) but the pair
assignments carry no chemical meaning.  If ot_shuf ~= ot_fixed then any gain comes
from merely having a non-product coupling, not from chemistry, and the coupling
hypothesis is dead.

Everything else is frozen to revision_run.py (E17 Phase 0): beta=1, gamma x lr on
the SAME 3x3 grid for every arm, domain-disjoint train/val OOD, parameter-matched
head, conventional FPR95, 1500 ID / 2000 OOD / 120 steps, val-only selection.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, argparse, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

# reuse E17's data/head/metric code paths verbatim so the protocol cannot drift
from revision_run import (load, blocks, Head, mets, VIEWDEF, GAMMAS, LRS, BETA,
                          SEL, FIN)

EPS_OT   = 0.02      # entropic reg; calibrated from cost geometry alone (knee of the
                     # ESS-vs-chemical-pull curve: ~65% of max pull, ~12k effective pairs)
LAM_CHEM = 1.0       # adversary chemical anchoring (scale-checked below)
SINK_IT  = 50        # Sinkhorn iterations
ADV_EVERY = 5        # re-solve the adversarial coupling every k steps
N_ID, N_OOD, EPOCHS = 1500, 2000, 120     # revision_run.train() budget (line 120)


# ---------- chemical cost + couplings ----------
def tanimoto_cost(mo_id, mo_ood):
    """1 - Tanimoto over Morgan ECFP4 bit vectors -> [0,1] cost matrix (n_id x n_ood)."""
    A = torch.as_tensor(mo_id, dtype=torch.float32)
    B = torch.as_tensor(mo_ood, dtype=torch.float32)
    inter = A @ B.T
    a = A.sum(1, keepdim=True); b = B.sum(1, keepdim=True).T
    return (1.0 - inter / (a + b - inter + 1e-8)).clamp_(0.0, 1.0)


def sinkhorn_log(logK, n_it=SINK_IT):
    """Log-domain Sinkhorn to uniform marginals. Returns pi summing to 1."""
    n, m = logK.shape
    f = torch.zeros(n); g = torch.zeros(m)
    log_mu = -np.log(n); log_nu = -np.log(m)
    for _ in range(n_it):
        f = log_mu - torch.logsumexp(logK + g[None, :], dim=1)
        g = log_nu - torch.logsumexp(logK + f[:, None], dim=0)
    return torch.exp(logK + f[:, None] + g[None, :])


def ess(pi):
    """Effective support size of the coupling (1 = degenerate, n*m = uniform)."""
    p = pi.flatten()
    return float(1.0 / (p.pow(2).sum() + 1e-30))


def build_coupling(cost, kind, seed):
    """Fixed couplings. Returns (pi, diagnostics) with pi summing to 1."""
    n, m = cost.shape
    if kind == "rpo_prod":
        return torch.full((n, m), 1.0 / (n * m)), {"ess_frac": 1.0}
    c = cost
    if kind == "ot_shuf":
        # independent row/col permutation: identical coupling STRUCTURE
        # (same marginals, same entropy, same value spectrum), no chemical meaning
        g = torch.Generator().manual_seed(10_000 + seed)
        c = cost[torch.randperm(n, generator=g)][:, torch.randperm(m, generator=g)]
    pi = sinkhorn_log(-c / EPS_OT)
    return pi, {"ess_frac": ess(pi) / (n * m),
                "mean_cost": float((pi * cost).sum())}


def adversarial_coupling(loss_mat, cost):
    """Inner max over couplings: mass on HARD pairs, anchored to chemically
    plausible ones, subject to uniform marginals (which is exactly what naive
    top-q% hard mining lacked -- E7 collapsed onto a handful of samples)."""
    return sinkhorn_log((loss_mat.detach() - LAM_CHEM * cost) / EPS_OT)


# ---------- training (mirrors revision_run.train, differing only in the coupling) ----------
def train(V, kind, gamma, lr, seed, cost=None, epochs=EPOCHS, n_id=N_ID, n_ood=N_OOD):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    iid = torch.randperm(len(V["train_id"][0]), generator=g)[:n_id]
    iod = torch.randperm(len(V["train_ood"][0]), generator=g)[:n_ood]
    Xid = [b[iid] for b in V["train_id"]]; Xood = [b[iod] for b in V["train_ood"]]

    C = cost[iid][:, iod] if cost is not None else None
    pi, diag = (build_coupling(C, kind, seed) if kind in
                ("rpo_prod", "ot_fixed", "ot_shuf") else (None, {}))

    head = Head([b.shape[1] for b in V["train_id"]])
    opt = torch.optim.AdamW(head.parameters(), lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9); best, bs = -1, None
    adv_pi = None
    for ep in range(epochs):
        head.train(); opt.zero_grad()
        Eid, Eood = head(Xid), head(Xood)
        if kind == "bce":
            core = 0.5 * F.binary_cross_entropy_with_logits(Eid, torch.zeros(len(Eid))) + \
                   0.5 * F.binary_cross_entropy_with_logits(Eood, torch.ones(len(Eood)))
        elif kind == "rpo":
            pair = torch.randint(len(Eid), (len(Eood),), generator=g)
            core = F.softplus(-BETA * (Eood - Eid[pair])).mean()
        else:
            L = F.softplus(-BETA * (Eood[None, :] - Eid[:, None]))   # n_id x n_ood
            if kind == "ot_adv":
                if ep % ADV_EVERY == 0:
                    adv_pi = adversarial_coupling(L, C)
                core = (adv_pi * L).sum()
            else:
                core = (pi * L).sum()
        (core + gamma * (Eid.pow(2).mean() + Eood.pow(2).mean())).backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
        if (ep + 1) % 15 == 0:
            head.eval()
            with torch.no_grad():
                v = roc(V, head)
            if v > best: best, bs = v, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval()
    if kind == "ot_adv" and adv_pi is not None:
        diag = {"ess_frac": ess(adv_pi) / adv_pi.numel(),
                "mean_cost": float((adv_pi * C).sum())}
    with torch.no_grad():
        return best, mets(head(V["test_id"]).numpy(), head(V["test_ood"]).numpy()), diag


def roc(V, head):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(
        np.r_[np.zeros(len(V["val_id"][0])), np.ones(len(V["val_ood"][0]))],
        np.r_[head(V["val_id"]).numpy(), head(V["val_ood"]).numpy()])


def run_arm(V, kind, cost):
    """Identical 9-trial validation-only grid for every arm."""
    best = None
    for gm in GAMMAS:
        for lr in LRS:
            v = float(np.mean([train(V, kind, gm, lr, s, cost)[0] for s in SEL]))
            if best is None or v > best[2]: best = (gm, lr, v)
    gm, lr, _ = best
    R = [train(V, kind, gm, lr, s, cost) for s in FIN]
    dg = [r[2].get("ess_frac") for r in R if r[2].get("ess_frac") is not None]
    mc = [r[2].get("mean_cost") for r in R if r[2].get("mean_cost") is not None]
    return ([r[1] for r in R],
            {"gamma": gm, "lr": lr,
             "ess_frac": float(np.mean(dg)) if dg else None,
             "mean_pair_cost": float(np.mean(mc)) if mc else None})


ARMS = ["bce", "rpo", "rpo_prod", "ot_fixed", "ot_adv", "ot_shuf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--views", nargs="+", default=["minimol", "minimol_mv"])
    ap.add_argument("--arms", nargs="+", default=ARMS)
    a = ap.parse_args()

    data, info = load(a.cell, need_um=False)
    cost = tanimoto_cost(data["train_id"]["mo"], data["train_ood"]["mo"])
    out = {"domain_split": info, "eps_ot": EPS_OT, "lam_chem": LAM_CHEM,
           "cost": "1 - Tanimoto(MorganECFP4,2048b)",
           "cost_stats": {"mean": float(cost.mean()), "min": float(cost.min()),
                          "p5": float(cost.flatten().kthvalue(int(0.05 * cost.numel()))[0])}}
    print(f"[{a.cell}] chemical cost: mean={cost.mean():.4f} min={cost.min():.4f} "
          f"| domain_split={info}", flush=True)

    for view in a.views:
        V = blocks(data, view)
        for kind in a.arms:
            R, hp = run_arm(V, kind, cost)
            au = [x[0] for x in R]
            out[f"{view}|{kind}"] = {"auroc": au, "aupr": [x[1] for x in R],
                                     "fpr95": [x[2] for x in R], "hp": hp}
            base = out.get(f"{view}|bce", {}).get("auroc")
            d = f" Δvs BOH={np.mean(au)-np.mean(base):+.4f}" if base and kind != "bce" else ""
            e = f" ess={hp['ess_frac']:.3f}" if hp.get("ess_frac") else ""
            print(f"  {view:11} {kind:9} AUROC={np.mean(au):.4f}±{np.std(au):.4f}"
                  f"{d}{e} hp=({hp['gamma']},{hp['lr']})", flush=True)
        # per-view filename when a single view is run, so parallel single-view
        # processes for the same cell cannot clobber each other's results
        tag = a.cell if len(a.views) > 1 else f"{a.cell}_{a.views[0]}"
        json.dump(out, open(f"repro/crpo_{tag}.json", "w"), indent=2)
    print(f"CRPO_DONE {a.cell}", flush=True)


if __name__ == "__main__":
    main()
