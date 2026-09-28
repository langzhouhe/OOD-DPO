#!/usr/bin/env python3
"""Select one recipe per cell/objective from the complete v2 validation screen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import rpo_opt_screen as S
from rpo_opt_v2_extension import extension_recipes
from rpo_opt_v2_screen import FULL_BUDGET, N_CHECKPOINTS, SELECTION_SEEDS


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
    args = ap.parse_args()

    base_paths = [
        Path("repro") / f"rpo_opt_v2_screen_{args.cell}_{args.backbone}_s{s}.json"
        for s in SELECTION_SEEDS
    ]
    extension_paths = [
        Path("repro") / f"rpo_opt_v2_extension_{args.cell}_{args.backbone}_s{s}.json"
        for s in SELECTION_SEEDS
    ]
    base_docs = [json.loads(path.read_text()) for path in base_paths]
    extension_docs = [json.loads(path.read_text()) for path in extension_paths]
    base_hash = S.stable_hash(S.recipes())
    extension_hash = S.stable_hash(extension_recipes())
    all_recipes = S.recipes() + extension_recipes()
    recipe_by_id = {int(r["id"]): r for r in all_recipes}
    if len(recipe_by_id) != len(all_recipes):
        raise RuntimeError("duplicate recipe IDs in the frozen union")
    union_hash = S.stable_hash(all_recipes)
    for path, doc, seed in zip(base_paths, base_docs, SELECTION_SEEDS):
        assert doc["protocol"] == "rpo-opt-v2-full-per-cell-validation-only"
        assert doc["seed"] == seed
        assert doc["budget"] == FULL_BUDGET
        assert doc["n_checkpoints"] == N_CHECKPOINTS
        assert doc["recipe_hash"] == base_hash
    for path, doc, seed in zip(extension_paths, extension_docs, SELECTION_SEEDS):
        assert doc["protocol"] == "rpo-opt-v2-boundary-extension-validation-only"
        assert doc["seed"] == seed
        assert doc["budget"] == FULL_BUDGET
        assert doc["n_checkpoints"] == N_CHECKPOINTS
        assert doc["recipe_hash"] == extension_hash

    out = {
        "protocol": "rpo-opt-v2-selection-frozen-before-final",
        "cell": args.cell,
        "backbone": args.backbone,
        "recipe_hash": union_hash,
        "base_recipe_hash": base_hash,
        "extension_recipe_hash": extension_hash,
        "selection_seeds": list(SELECTION_SEEDS),
        "screen_sha256": {
            path.name: sha256(path) for path in base_paths + extension_paths
        },
        "selection": {},
        "ranking": {},
    }
    for objective in ("rpo", "bce"):
        rows = []
        for rid in sorted(recipe_by_id):
            docs = base_docs if rid < 32 else extension_docs
            vals = np.asarray([
                doc["results"][objective][str(rid)]["val_auroc"] for doc in docs
            ], dtype=np.float64)
            rows.append(
                {
                    "recipe_id": rid,
                    "mean_val": float(vals.mean()),
                    "worst_seed_val": float(vals.min()),
                    "seed_values": vals.tolist(),
                }
            )
        rows.sort(
            key=lambda x: (x["mean_val"], x["worst_seed_val"], -x["recipe_id"]),
            reverse=True,
        )
        out["ranking"][objective] = rows
        best = rows[0]
        out["selection"][objective] = {
            **best,
            "recipe": recipe_by_id[best["recipe_id"]],
        }

    target = Path("repro") / f"rpo_opt_v2_selection_{args.cell}_{args.backbone}.json"
    target.write_text(json.dumps(out, indent=2))
    print(
        f"V2_SELECTION_FROZEN {target} "
        f"RPO=r{out['selection']['rpo']['recipe_id']} "
        f"BCE=r{out['selection']['bce']['recipe_id']} hash={S.stable_hash(out)}"
    )


if __name__ == "__main__":
    main()
