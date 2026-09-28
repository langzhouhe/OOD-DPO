#!/usr/bin/env python3
"""Aggregate complete equal-48 final artifacts without performing model selection.

The resulting summary is also the provenance boundary consumed by the paper-table
pipeline.  Consequently this script deliberately re-validates the frozen grid,
selection file, all 20 final seeds, and every selection/final file hash instead of
merely averaging whichever files happen to exist.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import oe_equal48 as E


def summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if len(array) != len(E.FINAL_SEEDS) or not np.isfinite(array).all():
        raise RuntimeError(
            f"expected {len(E.FINAL_SEEDS)} finite final values, got {len(array)}"
        )
    return {
        "values": values,
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "n": int(len(array)),
    }


def _assert_exact(actual, expected, description: str) -> None:
    if tuple(actual) != tuple(expected):
        raise RuntimeError(f"{description}: expected {tuple(expected)}, got {tuple(actual)}")


def _validate_selection(
    selection: dict, selection_path: Path, cell: str, backbone: str, method: str
) -> None:
    expected = {
        "protocol": E.PROTOCOL,
        "phase": "selection-frozen-before-final",
        "grid_protocol": E.GRID_PROTOCOL,
        "cell": cell,
        "backbone": backbone,
        "method": method,
        "n_recipe_trials": 48,
        "n_checkpoint_queries_per_neural_recipe": E.N_CHECKPOINTS,
        "recipe_hash": E.grid_hash(method),
        "all_grid_hash": E.all_grid_hash(),
    }
    for key, value in expected.items():
        if selection.get(key) != value:
            raise RuntimeError(
                f"selection provenance mismatch in {selection_path}: "
                f"{key}={selection.get(key)!r}, expected {value!r}"
            )
    _assert_exact(
        selection.get("selection_seeds", ()),
        E.SELECTION_SEEDS,
        f"selection seeds in {selection_path}",
    )
    selected = selection.get("selected", {})
    recipe_id = selected.get("recipe_id")
    if not isinstance(recipe_id, int) or not 0 <= recipe_id < 48:
        raise RuntimeError(f"invalid selected recipe id in {selection_path}: {recipe_id!r}")
    if selected.get("recipe") != E.recipes(method)[recipe_id]:
        raise RuntimeError(f"selected recipe drift in {selection_path}")
    per_seed = selected.get("per_seed_val_auroc", ())
    if len(per_seed) != len(E.SELECTION_SEEDS) or not np.isfinite(per_seed).all():
        raise RuntimeError(f"invalid validation values in {selection_path}")


def _validate_final(
    doc: dict,
    path: Path,
    selection: dict,
    selection_sha256: str,
    cell: str,
    backbone: str,
    method: str,
    seed: int,
) -> None:
    expected = {
        "protocol": E.PROTOCOL,
        "phase": "frozen-final-one-seed",
        "grid_protocol": E.GRID_PROTOCOL,
        "cell": cell,
        "backbone": backbone,
        "method": method,
        "seed": seed,
        "selection_sha256": selection_sha256,
        "recipe_hash": E.grid_hash(method),
        "selected_recipe_id": selection["selected"]["recipe_id"],
        "selected_recipe": selection["selected"]["recipe"],
    }
    for key, value in expected.items():
        if doc.get(key) != value:
            raise RuntimeError(
                f"final provenance mismatch in {path}: "
                f"{key}={doc.get(key)!r}, expected {value!r}"
            )
    _assert_exact(
        doc.get("final_seeds", ()), E.FINAL_SEEDS, f"final seeds declared in {path}"
    )
    test = doc.get("test", {})
    if set(test) != {"auroc", "aupr", "fpr95"}:
        raise RuntimeError(f"unexpected test metrics in {path}: {tuple(test)}")
    if not all(math.isfinite(float(test[metric])) for metric in test):
        raise RuntimeError(f"non-finite test metric in {path}")


def main() -> None:
    output = {
        "protocol": E.PROTOCOL,
        "phase": "complete-final-aggregate",
        "grid_protocol": E.GRID_PROTOCOL,
        "all_grid_hash": E.all_grid_hash(),
        "method_grid_hashes": {method: E.grid_hash(method) for method in E.METHODS},
        "selection_seeds": list(E.SELECTION_SEEDS),
        "final_seeds": list(E.FINAL_SEEDS),
        "n_recipe_trials_per_method": 48,
        "n_final_seeds": len(E.FINAL_SEEDS),
        "settings": {},
    }
    for cell in E.CELLS:
        for backbone in E.BACKBONES:
            setting = f"{cell}/{backbone}"
            output["settings"][setting] = {}
            for method in E.METHODS:
                selection_path = E._selection_path(cell, backbone, method)
                if not selection_path.is_file():
                    raise FileNotFoundError(selection_path)
                selection = json.loads(selection_path.read_text())
                _validate_selection(selection, selection_path, cell, backbone, method)
                selection_sha256 = E.file_sha256(selection_path)
                docs = []
                final_provenance = []
                for seed in E.FINAL_SEEDS:
                    path = E._final_path(cell, backbone, method, seed)
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    doc = json.loads(path.read_text())
                    _validate_final(
                        doc,
                        path,
                        selection,
                        selection_sha256,
                        cell,
                        backbone,
                        method,
                        seed,
                    )
                    docs.append(doc)
                    final_provenance.append(
                        {"path": str(path), "sha256": E.file_sha256(path), "seed": seed}
                    )
                output["settings"][setting][method] = {
                    "selected_recipe_id": selection["selected"]["recipe_id"],
                    "selected_recipe": selection["selected"]["recipe"],
                    "mean_selection_val_auroc": selection["selected"]["mean_val_auroc"],
                    "n_validation_trials": selection["n_recipe_trials"],
                    "recipe_hash": selection["recipe_hash"],
                    "all_grid_hash": selection["all_grid_hash"],
                    "selection_file": str(selection_path),
                    "selection_sha256": selection_sha256,
                    "final_files": final_provenance,
                    **{
                        metric: summary([float(doc["test"][metric]) for doc in docs])
                        for metric in ("auroc", "aupr", "fpr95")
                    },
                }
    target = Path("repro/oe_equal48_summary.json")
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(f"OE_EQUAL48_AGGREGATE_DONE {target}")


if __name__ == "__main__":
    main()
