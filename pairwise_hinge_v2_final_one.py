#!/usr/bin/env python3
"""Evaluate one final Hinge seed, gated on a frozen validation selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import pairwise_hinge_v2_common as H
import rpo_opt_final as RPO_FINAL
import rpo_opt_screen as S


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
    parser.add_argument("--seed", type=int, choices=H.FINAL_SEEDS, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Deliberately fail before loading any target-test array unless selection is frozen.
    selection_path = Path("repro") / (
        f"pairwise_hinge_v2_selection_{args.cell}_{args.backbone}.json"
    )
    if not selection_path.exists():
        raise RuntimeError(f"selection is not frozen: {selection_path}")
    selection_sha256 = sha256(selection_path)
    selection_doc = json.loads(selection_path.read_text())
    assert selection_doc["protocol"] == "pairwise-hinge-v2-selection-frozen-before-final"
    assert selection_doc["cell"] == args.cell
    assert selection_doc["backbone"] == args.backbone
    assert selection_doc["margin"] == H.MARGIN
    assert selection_doc["recipe_hash"] == H.EXPECTED_UNION_RECIPE_HASH
    recipes = {int(recipe["id"]): recipe for recipe in H.frozen_recipes()}
    chosen = selection_doc["selection"]
    recipe_id = int(chosen["recipe_id"])
    if chosen["recipe"] != recipes[recipe_id]:
        raise RuntimeError("selected recipe content does not match the frozen union")

    raw, audit = RPO_FINAL.load_all(args.cell, args.backbone)
    arrays = S.transformed(raw, chosen["recipe"]["normalization"])
    result = H.train_final(arrays, chosen["recipe"], args.seed, torch.device(args.device))
    out = {
        "protocol": "pairwise-hinge-v2-frozen-final-one-seed",
        "cell": args.cell,
        "backbone": args.backbone,
        "seed": args.seed,
        "margin": H.MARGIN,
        "recipe_hash": H.EXPECTED_UNION_RECIPE_HASH,
        "selection_sha256": selection_sha256,
        "domain_split": audit,
        "result": {"recipe_id": recipe_id, "recipe": chosen["recipe"], **result},
    }
    Path("repro").mkdir(exist_ok=True)
    target = Path("repro") / (
        f"pairwise_hinge_v2_final_{args.cell}_{args.backbone}_s{args.seed}.json"
    )
    target.write_text(json.dumps(out, indent=2))
    print(
        f"[{args.cell}/{args.backbone}/s{args.seed}] hinge "
        f"val={result['val_auroc']:.5f} test={result['test_auroc']:.5f}",
        flush=True,
    )
    print(f"HINGE_V2_FINAL_ONE_DONE {target}", flush=True)


if __name__ == "__main__":
    main()

