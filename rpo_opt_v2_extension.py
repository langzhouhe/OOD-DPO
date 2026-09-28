#!/usr/bin/env python3
"""Validation-only boundary extension fixed before v2 final evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import rpo_opt_screen as S
from rpo_opt_v2_screen import FULL_BUDGET, N_CHECKPOINTS, SELECTION_SEEDS


def extension_recipes() -> list[dict]:
    # Three validation-geometry neighbourhoods, fixed before any v2 final test:
    # H surrounds EC50/MiniMol recipe 13 at the high-lr/high-gamma boundary;
    # L surrounds IC50/MiniMol recipe 24 near the low-lr boundary; and
    # U locally refines the Ledoit-Wolf-whitened recipe 3 shared by both Uni-Mol cells.
    high = [
        (0.004, 1.0),
        (0.006, 1.0),
        (0.010, 1.0),
        (0.004, 3.0),
        (0.006, 3.0),
        (0.004, 10.0),
    ]
    low = [
        (1e-5, 0.001),
        (1e-5, 0.006),
        (3e-5, 0.001),
        (3e-5, 0.006),
        (8e-5, 0.001),
        (8e-5, 0.006),
    ]
    whiten = [
        (6e-4, 0.1),
        (6e-4, 0.6),
        (1.5e-3, 0.1),
        (1.5e-3, 0.6),
    ]
    out = []
    for offset, (lr, gamma) in enumerate(high):
        out.append(
            {
                "id": 32 + offset,
                "batch": 512,
                "lr": lr,
                "gamma": gamma,
                "weight_decay": 0.0,
                "dropout": 0.1,
                "normalization": "zscore",
            }
        )
    for offset, (lr, gamma) in enumerate(low):
        out.append(
            {
                "id": 38 + offset,
                "batch": 64,
                "lr": lr,
                "gamma": gamma,
                "weight_decay": 1e-5,
                "dropout": 0.2,
                "normalization": "zscore",
            }
        )
    for offset, (lr, gamma) in enumerate(whiten):
        out.append(
            {
                "id": 44 + offset,
                "batch": 128,
                "lr": lr,
                "gamma": gamma,
                "weight_decay": 1e-5,
                "dropout": 0.1,
                "normalization": "lw_whiten",
            }
        )
    assert [r["id"] for r in out] == list(range(32, 48))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=("ec50_assay", "ic50_assay"), required=True)
    ap.add_argument("--backbone", choices=("minimol", "unimol"), required=True)
    ap.add_argument("--seed", type=int, choices=SELECTION_SEEDS, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    raw, audit = S.load_validation(args.cell, args.backbone)
    by_norm = {kind: S.transformed(raw, kind) for kind in ("zscore", "lw_whiten")}
    grid = extension_recipes()
    old_budget, old_checkpoints = S.SCREEN_BUDGET, S.N_CHECKPOINTS
    S.SCREEN_BUDGET, S.N_CHECKPOINTS = FULL_BUDGET, N_CHECKPOINTS
    out = {
        "protocol": "rpo-opt-v2-boundary-extension-validation-only",
        "cell": args.cell,
        "backbone": args.backbone,
        "seed": args.seed,
        "domain_split": audit,
        "budget": FULL_BUDGET,
        "n_checkpoints": N_CHECKPOINTS,
        "recipe_hash": S.stable_hash(grid),
        "recipes": grid,
        "results": {"rpo": {}, "bce": {}},
    }
    try:
        for objective in ("rpo", "bce"):
            for rec in grid:
                result = S.train_one(
                    by_norm[rec["normalization"]],
                    objective,
                    rec,
                    args.seed,
                    torch.device(args.device),
                )
                out["results"][objective][str(rec["id"])] = result
                print(
                    f"[{args.cell}/{args.backbone}/s{args.seed}] {objective} "
                    f"ext={rec['id']} val={result['val_auroc']:.5f}",
                    flush=True,
                )
    finally:
        S.SCREEN_BUDGET, S.N_CHECKPOINTS = old_budget, old_checkpoints
    target = Path("repro") / (
        f"rpo_opt_v2_extension_{args.cell}_{args.backbone}_s{args.seed}.json"
    )
    target.write_text(json.dumps(out, indent=2))
    print(f"V2_EXTENSION_DONE {target}")


if __name__ == "__main__":
    main()
