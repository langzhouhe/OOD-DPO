#!/usr/bin/env python3
"""Freeze one Hinge recipe per cell/backbone from validation artifacts only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import pairwise_hinge_v2_common as H
import rpo_opt_screen as S
from rpo_opt_v2_screen import FULL_BUDGET, N_CHECKPOINTS, SELECTION_SEEDS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=("ec50_assay", "ic50_assay"), required=True)
    parser.add_argument("--backbone", choices=("minimol", "unimol"), required=True)
    args = parser.parse_args()

    paths = [
        Path("repro")
        / f"pairwise_hinge_v2_screen_{args.cell}_{args.backbone}_s{seed}.json"
        for seed in SELECTION_SEEDS
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"validation screen is incomplete: {missing}")
    docs = [json.loads(path.read_text()) for path in paths]
    recipes = H.frozen_recipes()
    recipe_by_id = {int(recipe["id"]): recipe for recipe in recipes}
    for path, doc, seed in zip(paths, docs, SELECTION_SEEDS):
        assert doc["protocol"] == "pairwise-hinge-v2-validation-only-48-recipes"
        assert doc["cell"] == args.cell and doc["backbone"] == args.backbone
        assert doc["seed"] == seed
        assert doc["budget"] == FULL_BUDGET
        assert doc["n_checkpoints"] == N_CHECKPOINTS
        assert doc["margin"] == H.MARGIN
        assert doc["recipe_hash"] == H.EXPECTED_UNION_RECIPE_HASH
        if sorted(map(int, doc["results"].keys())) != list(range(48)):
            raise RuntimeError(f"incomplete recipe results: {path}")

    ranking = []
    for recipe_id in sorted(recipe_by_id):
        values = np.asarray(
            [doc["results"][str(recipe_id)]["val_auroc"] for doc in docs],
            dtype=np.float64,
        )
        ranking.append(
            {
                "recipe_id": recipe_id,
                "mean_val": float(values.mean()),
                "worst_seed_val": float(values.min()),
                "seed_values": values.tolist(),
            }
        )
    ranking.sort(
        key=lambda row: (row["mean_val"], row["worst_seed_val"], -row["recipe_id"]),
        reverse=True,
    )
    best = ranking[0]
    selection = {**best, "recipe": recipe_by_id[best["recipe_id"]]}
    out = {
        "protocol": "pairwise-hinge-v2-selection-frozen-before-final",
        "cell": args.cell,
        "backbone": args.backbone,
        "margin": H.MARGIN,
        "recipe_hash": H.EXPECTED_UNION_RECIPE_HASH,
        "selection_seeds": list(SELECTION_SEEDS),
        "screen_sha256": {path.name: sha256(path) for path in paths},
        "selection": selection,
        "ranking": ranking,
    }
    Path("repro").mkdir(exist_ok=True)
    target = Path("repro") / (
        f"pairwise_hinge_v2_selection_{args.cell}_{args.backbone}.json"
    )
    target.write_text(json.dumps(out, indent=2))
    print(
        f"HINGE_V2_SELECTION_FROZEN {target} r={selection['recipe_id']} "
        f"hash={S.stable_hash(out)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

