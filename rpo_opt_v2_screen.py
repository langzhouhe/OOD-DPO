#!/usr/bin/env python3
"""Full per-cell validation-only recipe screen for RPO-OE tuning v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import rpo_opt_screen as S


FULL_BUDGET = 1_024_000
N_CHECKPOINTS = 100
SELECTION_SEEDS = (21, 22, 23, 24, 25)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=("ec50_assay", "ic50_assay"), required=True)
    ap.add_argument("--backbone", choices=("minimol", "unimol"), required=True)
    ap.add_argument("--seed", type=int, choices=SELECTION_SEEDS, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    raw, audit = S.load_validation(args.cell, args.backbone)
    by_norm = {kind: S.transformed(raw, kind) for kind in ("zscore", "lw_whiten")}
    grid = S.recipes()

    old_budget, old_checkpoints = S.SCREEN_BUDGET, S.N_CHECKPOINTS
    S.SCREEN_BUDGET, S.N_CHECKPOINTS = FULL_BUDGET, N_CHECKPOINTS
    out = {
        "protocol": "rpo-opt-v2-full-per-cell-validation-only",
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
                    by_norm[rec["normalization"]], objective, rec, args.seed, device
                )
                out["results"][objective][str(rec["id"])] = result
                print(
                    f"[{args.cell}/{args.backbone}/s{args.seed}] {objective} "
                    f"r={rec['id']:02d} val={result['val_auroc']:.5f} "
                    f"step={result['best_step']}/{result['steps']}",
                    flush=True,
                )
    finally:
        S.SCREEN_BUDGET, S.N_CHECKPOINTS = old_budget, old_checkpoints

    Path("repro").mkdir(exist_ok=True)
    target = Path("repro") / (
        f"rpo_opt_v2_screen_{args.cell}_{args.backbone}_s{args.seed}.json"
    )
    target.write_text(json.dumps(out, indent=2))
    print(f"V2_SCREEN_DONE {target} hash={S.stable_hash(out)}", flush=True)


if __name__ == "__main__":
    main()

