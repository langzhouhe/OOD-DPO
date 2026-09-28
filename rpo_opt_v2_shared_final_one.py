#!/usr/bin/env python3
"""One final paired seed under the frozen objective-blind shared recipe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import rpo_opt_final as F
import rpo_opt_screen as S
import rpo_opt_stage2 as P
from rpo_opt_v2_extension import extension_recipes
from rpo_opt_v2_final_one import FINAL_SEEDS
from rpo_opt_v2_screen import FULL_BUDGET, N_CHECKPOINTS


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=("ec50_assay", "ic50_assay"), required=True)
    parser.add_argument("--backbone", choices=("minimol", "unimol"), required=True)
    parser.add_argument("--seed", type=int, choices=FINAL_SEEDS, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    path = Path("repro") / f"rpo_opt_v2_shared_selection_{args.cell}_{args.backbone}.json"
    selection = json.loads(path.read_text())
    assert selection["protocol"] == "rpo-opt-v2-objective-blind-shared-recipe-selection"
    assert selection["recipe_hash"] == S.stable_hash(S.recipes() + extension_recipes())
    recipe = selection["selection"]["recipe"]
    raw, audit = F.load_all(args.cell, args.backbone)
    arrays = S.transformed(raw, recipe["normalization"])
    old_checkpoints, old_budget = S.N_CHECKPOINTS, P.FULL_BUDGET
    S.N_CHECKPOINTS, P.FULL_BUDGET = N_CHECKPOINTS, FULL_BUDGET
    out = {
        "protocol": "rpo-opt-v2-objective-blind-shared-recipe-final",
        "cell": args.cell,
        "backbone": args.backbone,
        "seed": args.seed,
        "selection_sha256": sha256(path),
        "recipe_id": selection["selection"]["recipe_id"],
        "recipe": recipe,
        "domain_split": audit,
        "results": {},
    }
    try:
        for objective in ("rpo", "bce"):
            out["results"][objective] = F.train_final(
                arrays, objective, recipe, args.seed, torch.device(args.device)
            )
            print(f"[{args.cell}/{args.backbone}/s{args.seed}] shared {objective} "
                  f"test={out['results'][objective]['test_auroc']:.5f}", flush=True)
    finally:
        S.N_CHECKPOINTS, P.FULL_BUDGET = old_checkpoints, old_budget
    target = Path("repro") / (
        f"rpo_opt_v2_shared_final_{args.cell}_{args.backbone}_s{args.seed}.json"
    )
    target.write_text(json.dumps(out, indent=2))
    print(f"V2_SHARED_FINAL_DONE {target}", flush=True)


if __name__ == "__main__":
    main()
