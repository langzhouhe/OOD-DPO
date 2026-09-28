#!/usr/bin/env python3
"""Validation-only 48-recipe screen for matched Pairwise-Hinge v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import pairwise_hinge_v2_common as H
import rpo_opt_screen as S
from rpo_opt_v2_screen import FULL_BUDGET, N_CHECKPOINTS, SELECTION_SEEDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=("ec50_assay", "ic50_assay"), required=True)
    parser.add_argument("--backbone", choices=("minimol", "unimol"), required=True)
    parser.add_argument("--seed", type=int, choices=SELECTION_SEEDS, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    raw, audit = S.load_validation(args.cell, args.backbone)
    arrays_by_norm = {
        kind: S.transformed(raw, kind) for kind in ("zscore", "lw_whiten")
    }
    recipes = H.frozen_recipes()
    out = {
        "protocol": "pairwise-hinge-v2-validation-only-48-recipes",
        "cell": args.cell,
        "backbone": args.backbone,
        "seed": args.seed,
        "domain_split": audit,
        "budget": FULL_BUDGET,
        "n_checkpoints": N_CHECKPOINTS,
        "ema_decay": S.EMA_DECAY,
        "margin": H.MARGIN,
        "recipe_hash": H.EXPECTED_UNION_RECIPE_HASH,
        "recipes": recipes,
        "results": {},
    }
    device = torch.device(args.device)
    for recipe in recipes:
        result = H.train_validation(
            arrays_by_norm[recipe["normalization"]], recipe, args.seed, device
        )
        out["results"][str(recipe["id"])] = result
        print(
            f"[{args.cell}/{args.backbone}/s{args.seed}] hinge "
            f"r={recipe['id']:02d} val={result['val_auroc']:.5f} "
            f"step={result['best_step']}/{result['steps']}",
            flush=True,
        )

    Path("repro").mkdir(exist_ok=True)
    target = Path("repro") / (
        f"pairwise_hinge_v2_screen_{args.cell}_{args.backbone}_s{args.seed}.json"
    )
    target.write_text(json.dumps(out, indent=2))
    print(f"HINGE_V2_SCREEN_DONE {target} hash={S.stable_hash(out)}", flush=True)


if __name__ == "__main__":
    main()

