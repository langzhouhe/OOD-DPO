#!/usr/bin/env python3
"""Train a Phase-0 detector on each Uni-Mol encoder-ablation arm.

Consumes the caches written by ablate_unimol.py by pointing revision_run.CACHE at the
arm's directory; every other element of the protocol (equal 3x3 tuning grid,
domain-disjoint validation OOD, parameter-matched head, validation-only selection,
conventional FPR95, 5 final seeds) is inherited unchanged by import.

The `pretrained` reference is read from repro/rev_<cell>.json (E17's
unimol|BalancedOODHead entry) rather than recomputed: it is the identical protocol on
the identical features, so recomputing it would only add noise.

Objective is BalancedOODHead (BCE) throughout -- E17 established the two objectives are
statistically tied, and this asks a question about the ENCODER, not the loss.

Usage: python diag_encoder.py --cell ec50_assay
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, argparse, warnings, logging
import numpy as np, torch
torch.set_num_threads(1)
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

import revision_run as RR

ARMS = ["pre", "gbf_id", "emb_rand", "rand"]
ORIG = RR.CACHE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    a = ap.parse_args()

    out = {"cell": a.cell, "arms": {}}
    ref = Path(f"repro/rev_{a.cell}.json")
    if ref.exists():
        r = json.load(open(ref)).get("unimol|BalancedOODHead")
        if r:
            out["arms"]["pretrained"] = {k: r[k] for k in ("auroc", "aupr", "fpr95", "hp")}
            print(f"[{a.cell}] pretrained  AUROC={np.mean(r['auroc']):.4f} "
                  f"FPR95={np.mean(r['fpr95']):.3f}  (from E17)", flush=True)

    for arm in a.arms:
        RR.CACHE = Path(f"cache/abl_{arm}")
        if not (RR.CACHE / f"{RR.BASE[a.cell]}_unimol_features.pkl").exists():
            print(f"[{a.cell}] {arm}: cache missing -> skip", flush=True); continue
        data, info = RR.load(a.cell, True)
        V = RR.blocks(data, "unimol")
        R, hp = RR.run_objective(V, "bce")
        out["arms"][arm] = {"auroc": [x[0] for x in R], "aupr": [x[1] for x in R],
                            "fpr95": [x[2] for x in R], "hp": hp}
        print(f"[{a.cell}] {arm:11} AUROC={np.mean([x[0] for x in R]):.4f} "
              f"FPR95={np.mean([x[2] for x in R]):.3f}", flush=True)
    RR.CACHE = ORIG
    json.dump(out, open(f"repro/encabl_{a.cell}.json", "w"), indent=2)
    print(f"ENCABL_DONE {a.cell}", flush=True)


if __name__ == "__main__":
    main()
