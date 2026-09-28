#!/usr/bin/env python3
"""Multi-seed validation stage for recipes frozen by rpo_opt_screen.py.

No test metrics are computed here.  The recipe IDs below are the top-eight global recipes
for each objective from the seed-1, four-cell validation screen with recipe hash
b20fbbc9883ae9021659d8c6e4919446e8ee80d6e305d67dab8b6f6cc1632feb.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import rpo_opt_screen as S


FULL_BUDGET = 1_024_000
SEEDS = (1, 2, 3, 4, 5)
TOP = {
    "rpo": (3, 28, 16, 27, 11, 31, 1, 24),
    "bce": (3, 16, 28, 27, 24, 14, 1, 19),
}
EXPECTED_RECIPE_HASH = "b20fbbc9883ae9021659d8c6e4919446e8ee80d6e305d67dab8b6f6cc1632feb"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=("ec50_assay", "ic50_assay"), required=True)
    ap.add_argument("--backbone", choices=("minimol", "unimol"), required=True)
    ap.add_argument("--objective", choices=("rpo", "bce", "both"), default="both")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    grid = S.recipes()
    got = S.stable_hash(grid)
    if got != EXPECTED_RECIPE_HASH:
        raise RuntimeError(f"recipe drift: {got} != {EXPECTED_RECIPE_HASH}")
    raw, audit = S.load_validation(args.cell, args.backbone)
    by_norm = {k: S.transformed(raw, k) for k in ("zscore", "lw_whiten")}
    device = torch.device(args.device)
    objectives = ("rpo", "bce") if args.objective == "both" else (args.objective,)
    old_budget = S.SCREEN_BUDGET
    S.SCREEN_BUDGET = FULL_BUDGET
    out = {
        "protocol": "rpo-opt-stage2-v1-validation-only",
        "cell": args.cell,
        "backbone": args.backbone,
        "recipe_hash": got,
        "full_budget": FULL_BUDGET,
        "selection_seeds": list(SEEDS),
        "domain_split": audit,
        "top_recipe_ids": {k: list(v) for k, v in TOP.items()},
        "results": {},
    }
    try:
        for objective in objectives:
            out["results"][objective] = {}
            for rid in TOP[objective]:
                rec = grid[rid]
                runs = []
                for seed in SEEDS:
                    result = S.train_one(by_norm[rec["normalization"]], objective, rec, seed, device)
                    runs.append(result)
                    print(
                        f"[{args.cell}/{args.backbone}] {objective} r={rid:02d} s={seed} "
                        f"val={result['val_auroc']:.4f} step={result['best_step']}/"
                        f"{result['steps']} clip={result['clip_rate']:.3f}",
                        flush=True,
                    )
                out["results"][objective][str(rid)] = runs
    finally:
        S.SCREEN_BUDGET = old_budget

    Path("repro").mkdir(exist_ok=True)
    target = Path("repro") / f"rpo_opt_stage2_{args.cell}_{args.backbone}.json"
    target.write_text(json.dumps(out, indent=2))
    print(f"STAGE2_DONE {target} hash={S.stable_hash(out)}", flush=True)


if __name__ == "__main__":
    main()
