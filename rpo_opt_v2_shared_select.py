#!/usr/bin/env python3
"""Freeze an objective-blind shared recipe from v2 validation artifacts.

The selected recipe maximizes the average of the five-seed RPO and BCE validation
AUROCs.  It provides a loss-only comparison: head, recipe, batches, optimizer and
checkpoint budget are identical, with only the core objective changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import rpo_opt_screen as S
from rpo_opt_v2_extension import extension_recipes
from rpo_opt_v2_screen import SELECTION_SEEDS


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
    args = parser.parse_args()

    base_paths = [Path("repro") / f"rpo_opt_v2_screen_{args.cell}_{args.backbone}_s{s}.json"
                  for s in SELECTION_SEEDS]
    ext_paths = [Path("repro") / f"rpo_opt_v2_extension_{args.cell}_{args.backbone}_s{s}.json"
                 for s in SELECTION_SEEDS]
    base_docs = [json.loads(path.read_text()) for path in base_paths]
    ext_docs = [json.loads(path.read_text()) for path in ext_paths]
    recipes = S.recipes() + extension_recipes()
    by_id = {int(r["id"]): r for r in recipes}
    rows = []
    for rid in sorted(by_id):
        docs = base_docs if rid < 32 else ext_docs
        rpo = np.asarray([d["results"]["rpo"][str(rid)]["val_auroc"] for d in docs])
        bce = np.asarray([d["results"]["bce"][str(rid)]["val_auroc"] for d in docs])
        joint = 0.5 * (rpo + bce)
        rows.append({
            "recipe_id": rid,
            "mean_joint_val": float(joint.mean()),
            "worst_seed_joint_val": float(joint.min()),
            "mean_rpo_val": float(rpo.mean()),
            "mean_bce_val": float(bce.mean()),
            "seed_joint_values": joint.tolist(),
        })
    rows.sort(key=lambda x: (x["mean_joint_val"], x["worst_seed_joint_val"],
                             -x["recipe_id"]), reverse=True)
    chosen = rows[0]
    out = {
        "protocol": "rpo-opt-v2-objective-blind-shared-recipe-selection",
        "cell": args.cell,
        "backbone": args.backbone,
        "selection_seeds": list(SELECTION_SEEDS),
        "recipe_hash": S.stable_hash(recipes),
        "screen_sha256": {p.name: sha256(p) for p in base_paths + ext_paths},
        "selection": {**chosen, "recipe": by_id[chosen["recipe_id"]]},
        "ranking": rows,
    }
    target = Path("repro") / f"rpo_opt_v2_shared_selection_{args.cell}_{args.backbone}.json"
    target.write_text(json.dumps(out, indent=2))
    print(f"V2_SHARED_SELECTION_FROZEN {target} r={chosen['recipe_id']} "
          f"val={chosen['mean_joint_val']:.6f}")


if __name__ == "__main__":
    main()
