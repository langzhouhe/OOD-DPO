#!/usr/bin/env python3
"""RPO-Lite: does the pairwise objective help once the reliability head is small?

Every matched comparison in this project so far (E2/E5/E6/E7/E17/E19/E23/E24) used the
same 768-256-128-1 scoring head, and all of them ended in a tie.  The theoretical reason
for the tie -- RPO's optimum is (1/beta) x BCE's logit plus a constant, so the two share a
Bayes-optimal ranking -- only bites when the head has enough capacity to represent that
optimum.  Head capacity is the one axis that was never swept, so it is the only place a
finite-capacity difference between a ranking loss and a calibration loss could still live.

Sweep: hidden width h in {4, 8, 16, 32, 64, 128, 256} plus the original E17 head as the
reference point, RPO vs Balanced OOD Head at every width, same 9-trial (gamma x lr) budget,
same 3 selection seeds and 5 final seeds, same domain-disjoint validation split.

RR.Head is monkey-patched rather than reimplemented, so RR.train / RR.run_objective run
byte-for-byte the Phase-0 protocol at every width and no protocol drift is possible.

Usage: python capacity_sweep.py --cell ec50_assay --backbone minimol
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, argparse, warnings, logging
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import revision_run as RR
import matched_oe as MO

WIDTHS = [4, 8, 16, 32, 64, 128, 256, "full"]
FULL_HEAD = RR.Head


def lite(h):
    """Total projected width AND scoring width are both h, so h is the only capacity knob."""
    class LiteHead(nn.Module):
        def __init__(s, dims):
            super().__init__()
            w = max(1, h // len(dims))
            s.proj = nn.ModuleList([nn.Linear(d, w) for d in dims])
            s.net = nn.Sequential(nn.Linear(w * len(dims), h), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(h, 1))
        def forward(s, bl):
            return s.net(torch.cat([p(b) for p, b in zip(s.proj, bl)], 1)).squeeze(-1)
    return LiteHead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--backbone", default="minimol")
    a = ap.parse_args()
    D, info = MO.load_cell(a.cell, a.backbone)
    V = {k: [torch.tensor(D[k][0])] for k in RR.KEYS}
    dims = [V["train_id"][0].shape[1]]
    out = {"cell": a.cell, "backbone": a.backbone, "domain_split": info, "widths": {}}
    print(f"[{a.cell}/{a.backbone}] id={len(D['train_id'][0])} ood={len(D['train_ood'][0])}",
          flush=True)

    for h in WIDTHS:
        RR.Head = FULL_HEAD if h == "full" else lite(h)
        npar = sum(p.numel() for p in RR.Head(dims).parameters())
        rec = {"params": int(npar)}
        for kind, name in [("bce", "BalancedOODHead"), ("rpo", "RPO")]:
            R, hp = RR.run_objective(V, kind)
            rec[name] = {"auroc": [r[0] for r in R], "aupr": [r[1] for r in R],
                         "fpr95": [r[2] for r in R], "hp": hp}
        d = np.array(rec["RPO"]["auroc"]) - np.array(rec["BalancedOODHead"]["auroc"])
        rec["delta_mean"] = float(d.mean()); rec["delta_pos_seeds"] = int((d > 0).sum())
        out["widths"][str(h)] = rec
        print(f"  h={str(h):5} params={npar:>9,}  RPO={np.mean(rec['RPO']['auroc']):.4f} "
              f"BOH={np.mean(rec['BalancedOODHead']['auroc']):.4f} "
              f"Δ={d.mean():+.4f} ({int((d > 0).sum())}/5 seeds +)", flush=True)

    RR.Head = FULL_HEAD
    Path("repro").mkdir(exist_ok=True)
    json.dump(out, open(f"repro/cap_{a.cell}_{a.backbone}.json", "w"), indent=2)
    print(f"CAP_DONE {a.cell} {a.backbone}", flush=True)


if __name__ == "__main__":
    main()
