#!/usr/bin/env python3
"""One paired final seed for a frozen RPO-OE v2 selection."""
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
from rpo_opt_v2_screen import FULL_BUDGET, N_CHECKPOINTS


FINAL_SEEDS = tuple(range(301, 321))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=("ec50_assay", "ic50_assay"), required=True)
    ap.add_argument("--backbone", choices=("minimol", "unimol"), required=True)
    ap.add_argument("--seed", type=int, choices=FINAL_SEEDS, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    selection_path = Path("repro") / (
        f"rpo_opt_v2_selection_{args.cell}_{args.backbone}.json"
    )
    selection = json.loads(selection_path.read_text())
    assert selection["protocol"] == "rpo-opt-v2-selection-frozen-before-final"
    assert selection["recipe_hash"] == S.stable_hash(S.recipes() + extension_recipes())

    raw, audit = F.load_all(args.cell, args.backbone)
    device = torch.device(args.device)
    old_checkpoints, old_budget = S.N_CHECKPOINTS, P.FULL_BUDGET
    S.N_CHECKPOINTS, P.FULL_BUDGET = N_CHECKPOINTS, FULL_BUDGET
    out = {
        "protocol": "rpo-opt-v2-frozen-final-one-seed",
        "cell": args.cell,
        "backbone": args.backbone,
        "seed": args.seed,
        "selection_sha256": sha256(selection_path),
        "domain_split": audit,
        "results": {},
    }
    try:
        for objective in ("rpo", "bce"):
            chosen = selection["selection"][objective]
            recipe = chosen["recipe"]
            arrays = S.transformed(raw, recipe["normalization"])
            result = F.train_final(arrays, objective, recipe, args.seed, device)
            out["results"][objective] = {
                "recipe_id": chosen["recipe_id"],
                "recipe": recipe,
                **result,
            }
            print(
                f"[{args.cell}/{args.backbone}/s{args.seed}] {objective} "
                f"val={result['val_auroc']:.5f} test={result['test_auroc']:.5f}",
                flush=True,
            )
    finally:
        S.N_CHECKPOINTS, P.FULL_BUDGET = old_checkpoints, old_budget

    target = Path("repro") / (
        f"rpo_opt_v2_final_{args.cell}_{args.backbone}_s{args.seed}.json"
    )
    target.write_text(json.dumps(out, indent=2))
    print(f"V2_FINAL_ONE_DONE {target}", flush=True)


if __name__ == "__main__":
    main()
