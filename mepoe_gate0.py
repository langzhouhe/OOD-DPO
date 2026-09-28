#!/usr/bin/env python3
"""MePOE Gate 0: is there a stable, transferable signal in WHICH mechanisms you expose?

For a fixed OE budget of k mechanisms, sample many random subsets of the source catalog
S, train the SAME Balanced BCE detector on each, and score every subset against two
disjoint pseudo-target mechanism populations H1 and H2.

  crit 1  headroom          top-decile subset beats the random mean by >= 0.020 on H1
  crit 2  preference xfer   sign(H1 preference) agrees with H2 on >= 65% of subset pairs
  crit 3  seed stability    preferences agree across detector seeds on >= 80% of pairs
  crit 4  selection value   subsets chosen on H1 beat the random mean on H2 by >= 0.010

Criterion 4 is the honest one: 1 is measured on the same population used to rank, so it
only bounds available headroom. A variance decomposition is also reported -- if
between-seed variance swamps between-subset variance there is nothing to learn no matter
how good the policy, and the gate fails for a diagnosable reason.

The detector is deliberately fixed (no per-subset tuning, no checkpoint selection, fixed
epoch count) so that differences between subsets come from the EXPOSURE, not from a
subset-dependent training protocol.

Usage: python mepoe_gate0.py --cell ec50_assay --seed 21
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, pickle, argparse, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from pathlib import Path
from multiprocessing import Pool
from scipy.stats import spearmanr
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import revision_run as RR

CACHE = Path("cache/ood_dpo_cache"); MP = Path("cache/mepoe")
BASE = {"ec50_assay": "lbap_general_ec50_assay", "ic50_assay": "lbap_general_ic50_assay",
        "ki_assay": "lbap_general_ki_assay",
        "ec50_scaffold": "lbap_general_ec50_scaffold",
        "ic50_scaffold": "lbap_general_ic50_scaffold"}
BUDGETS = [12, 25]
N_SUBSET = 128
SEEDS = [1, 2, 3]
GAMMA, LR, EPOCHS = 0.01, 3e-4, 120
G = {}


def setup(cell, seed, backbone="minimol", keep=None):
    """keep: optional SMILES whitelist, used when two backbones must be aligned row-for-row
    (a molecule missing from one cache would otherwise shift the mechanism indices)."""
    b = BASE[cell]
    feats = pickle.load(open(CACHE / f"{b}_{backbone}_features.pkl", "rb"))["features"]
    if keep is not None:
        feats = {k: v for k, v in feats.items() if k in keep}
    rec = json.load(open(MP / f"{b}_mepoe_seed{seed}.json"))
    def mat(smis):
        return np.stack([feats[s] for s in smis if s in feats]).astype(np.float32)
    tr_smi = [s for s in rec["id_train"] if s in feats]
    te_smi = [s for s in rec["id_test"] if s in feats]
    idtr, idte = mat(rec["id_train"]), mat(rec["id_test"])
    mu, sd = idtr.mean(0), idtr.std(0) + 1e-6
    out = {"id_train": torch.tensor((idtr - mu) / sd), "id_test": torch.tensor((idte - mu) / sd),
           "id_train_smi": tr_smi, "id_test_smi": te_smi}
    for name in ("S", "H1", "H2"):
        doms = sorted(rec["catalog"][name])
        rows, midx = [], []
        for j, d in enumerate(doms):
            v = [s for s in rec["catalog"][name][d] if s in feats]
            rows += v; midx += [j] * len(v)
        out[name] = torch.tensor((mat(rows) - mu) / sd)
        out[name + "_mech"] = np.asarray(midx)
        out[name + "_smi"] = rows          # row-aligned SMILES, for extra feature views
    return out, rec


def run_one(job):
    k_rows, seed = job
    Xid, Xood = G["id_train"], G["S"][torch.as_tensor(k_rows)]
    torch.manual_seed(seed); np.random.seed(seed)
    head = RR.Head([Xid.shape[1]])
    opt = torch.optim.AdamW(head.parameters(), LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9)
    for _ in range(EPOCHS):
        head.train(); opt.zero_grad()
        Eid, Eood = head([Xid]), head([Xood])
        loss = 0.5 * F.binary_cross_entropy_with_logits(Eid, torch.zeros(len(Eid))) + \
               0.5 * F.binary_cross_entropy_with_logits(Eood, torch.ones(len(Eood))) + \
               GAMMA * (Eid.pow(2).mean() + Eood.pow(2).mean())
        loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); sch.step()
    head.eval()
    with torch.no_grad():
        s_id = head([G["id_test"]]).numpy()
        return tuple(RR.mets(s_id, head([G[h]]).numpy())[0] for h in ("H1", "H2"))


def pref_acc(x, y, mask_frac=0.0):
    """Fraction of subset pairs on which the sign of the x-difference matches y."""
    n = len(x); dx = x[:, None] - x[None, :]; dy = y[:, None] - y[None, :]
    iu = np.triu_indices(n, 1); a, b = dx[iu], dy[iu]
    if mask_frac > 0:
        keep = np.abs(a) >= np.quantile(np.abs(a), mask_frac)
        a, b = a[keep], b[keep]
    nz = a != 0
    return float((np.sign(a[nz]) == np.sign(b[nz])).mean()), int(nz.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    data, rec = setup(a.cell, a.seed)
    G.update(data)
    n_mech = int(G["S_mech"].max()) + 1
    print(f"[{a.cell}] catalog {n_mech} mechanisms | ID train {len(G['id_train'])} "
          f"test {len(G['id_test'])} | H1 {len(G['H1'])} H2 {len(G['H2'])} mols", flush=True)
    print(f"[{a.cell}] canonical-SMILES overlaps: {rec['canon_overlap']}", flush=True)

    out = {"cell": a.cell, "seed": a.seed, "budgets": {}}
    for k in BUDGETS:
        rng = np.random.default_rng(1000 + k)
        subs = [np.sort(rng.choice(n_mech, size=k, replace=False)) for _ in range(N_SUBSET)]
        jobs = [(np.where(np.isin(G["S_mech"], s))[0], sd) for s in subs for sd in SEEDS]
        with Pool(a.workers) as p:
            res = p.map(run_one, jobs, chunksize=1)
        A = np.array(res).reshape(N_SUBSET, len(SEEDS), 2)
        a1, a2 = A[:, :, 0].mean(1), A[:, :, 1].mean(1)
        top = np.argsort(-a1)[:max(1, N_SUBSET // 10)]

        c1 = float(a1[top].mean() - a1.mean())
        c2, npair = pref_acc(a1, a2)
        c2_big, _ = pref_acc(a1, a2, mask_frac=0.5)
        ss = [pref_acc(A[:, i, 0], A[:, j, 0])[0]
              for i in range(len(SEEDS)) for j in range(i + 1, len(SEEDS))]
        c3 = float(np.mean(ss))
        c4 = float(a2[top].mean() - a2.mean())
        v_sub = float(A[:, :, 0].mean(1).var())
        v_seed = float(A[:, :, 0].var(1).mean())
        rho = float(spearmanr(a1, a2).statistic)

        g = {"crit1_headroom_H1": c1, "crit2_pref_transfer": c2, "crit2_top50pct_margin": c2_big,
             "crit3_seed_stability": c3, "crit4_selection_value_H2": c4,
             "spearman_H1_H2": rho, "var_between_subset": v_sub, "var_between_seed": v_seed,
             "mean_auroc_H1": float(a1.mean()), "mean_auroc_H2": float(a2.mean()),
             "std_auroc_H1": float(a1.std()), "n_pairs": npair,
             "pass": {"c1": c1 >= 0.020, "c2": c2 >= 0.65, "c3": c3 >= 0.80, "c4": c4 >= 0.010}}
        out["budgets"][str(k)] = g
        print(f"\n[{a.cell}] budget k={k} mechanisms ({k*8} OE molecules)", flush=True)
        print(f"   mean AUROC H1={a1.mean():.4f} H2={a2.mean():.4f} | across-subset sd={a1.std():.4f}", flush=True)
        print(f"   var between-subset={v_sub:.2e}  between-seed={v_seed:.2e}  "
              f"ratio={v_sub/max(v_seed,1e-12):.2f}", flush=True)
        print(f"   c1 headroom(H1)      {c1:+.4f}  need >=+0.0200  {'PASS' if c1>=0.020 else 'FAIL'}")
        print(f"   c2 pref transfer     {c2:.3f}   need >=0.650    {'PASS' if c2>=0.65 else 'FAIL'}"
              f"   (top-50% margin pairs: {c2_big:.3f}, Spearman {rho:+.3f})")
        print(f"   c3 seed stability    {c3:.3f}   need >=0.800    {'PASS' if c3>=0.80 else 'FAIL'}")
        print(f"   c4 selection value   {c4:+.4f}  need >=+0.0100  {'PASS' if c4>=0.010 else 'FAIL'}",
              flush=True)
    json.dump(out, open(f"repro/mepoe_gate0_{a.cell}_s{a.seed}.json", "w"), indent=2)
    print(f"GATE0_DONE {a.cell}", flush=True)


if __name__ == "__main__":
    main()
