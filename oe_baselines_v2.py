#!/usr/bin/env python3
"""Matched-OE baselines on the v2 selection/final seeds.

Run only after all four `rpo_opt_v2_selection_*.json` files exist.  This reuses the
established baseline implementations and grids from `matched_oe.py`, but evaluates them
on the exact v2 selection and final seed sets so the resulting compact table is not a
cross-protocol comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matched_oe as M
from rpo_opt_v2_final_one import FINAL_SEEDS
from rpo_opt_v2_screen import SELECTION_SEEDS


METHODS = (
    "MSP-OE",
    "ODIN-OE",
    "Energy-OE",
    "OE-Mahalanobis",
    "OE-KNN",
    "OE-LOF",
)


def matched_context(data, y_all: np.ndarray, seed: int) -> dict:
    """Use the exact NumPy permutation rule used by RPO-OE v2 final training."""
    rng = np.random.default_rng(seed)
    iid = rng.permutation(len(data["train_id"][0]))[: min(1500, len(data["train_id"][0]))]
    iod = rng.permutation(len(data["train_ood"][0]))[: min(2000, len(data["train_ood"][0]))]
    xid = data["train_id"][0][iid]
    xood = data["train_ood"][0][iod]
    labels = y_all[iid]
    keep = labels >= 0
    return {
        "Xid": xid,
        "Xood": xood,
        "y": labels[keep],
        "Xid_lab": xid[keep],
        "n_lab": int(keep.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=("ec50_assay", "ic50_assay"), required=True)
    ap.add_argument("--backbone", choices=("minimol", "unimol"), required=True)
    args = ap.parse_args()

    frozen = Path("repro") / f"rpo_opt_v2_selection_{args.cell}_{args.backbone}.json"
    if not frozen.exists():
        raise RuntimeError(f"RPO/BCE selection is not frozen: {frozen}")

    M.SEL = tuple(SELECTION_SEEDS)
    M.FIN = tuple(FINAL_SEEDS)
    data, audit = M.load_cell(args.cell, args.backbone)
    labels = M.label_map(args.cell)
    y_all = np.asarray([labels.get(s, -1) for s in data["train_id"][1]])
    ok = y_all >= 0
    if ok.sum() <= 50 or len(np.unique(y_all[ok])) <= 1:
        raise RuntimeError("OE logit baselines require usable downstream ID labels")
    contexts = {
        seed: matched_context(data, y_all, seed)
        for seed in set(SELECTION_SEEDS) | set(FINAL_SEEDS)
    }
    out = {
        "protocol": "matched-oe-v2-baselines-same-selection-and-final-seeds",
        "cell": args.cell,
        "backbone": args.backbone,
        "selection_seeds": list(SELECTION_SEEDS),
        "final_seeds": list(FINAL_SEEDS),
        "domain_split": audit,
        "methods": {},
    }
    for name in METHODS:
        runs, hp, mean_val = M.run_method(name, data, y_all, contexts)
        out["methods"][name] = {
            "auroc": [x[0] for x in runs],
            "aupr": [x[1] for x in runs],
            "fpr95": [x[2] for x in runs],
            "selected_hp": repr(hp),
            "mean_selection_val_auroc": mean_val,
            "n_validation_trials": len(M.METHODS[name][1]),
        }
        print(
            f"[{args.cell}/{args.backbone}] {name:16s} "
            f"val={mean_val:.5f} test={np.mean(out['methods'][name]['auroc']):.5f}",
            flush=True,
        )
    target = Path("repro") / f"oe_baselines_v2_{args.cell}_{args.backbone}.json"
    target.write_text(json.dumps(out, indent=2))
    print(f"OE_BASELINES_V2_DONE {target}")


if __name__ == "__main__":
    main()
