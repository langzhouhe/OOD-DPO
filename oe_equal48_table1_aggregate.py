#!/usr/bin/env python3
"""Validate and aggregate the equal-48 classifier-score OE rows for Table 1."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import oe_equal48 as E
from oe_equal48_aggregate import (
    _validate_final,
    _validate_selection,
    summary,
)


METHODS = ("MSP-OE", "ODIN-OE", "Energy-OE")
DISPLAY = {
    "MSP-OE": "MSP-OE",
    "ODIN-OE": "ODIN-OE (temperature-only)",
    "Energy-OE": "Energy-OE",
}
TABLE_ORDER = (
    "ec50_scaffold",
    "ec50_size",
    "ec50_assay",
    "ic50_scaffold",
    "ic50_size",
    "ic50_assay",
    "hiv_scaffold",
    "hiv_size",
    "pcba_scaffold",
    "pcba_size",
    "zinc_scaffold",
    "zinc_size",
)
if E.TABLE1_CELLS != TABLE_ORDER:
    raise RuntimeError(
        "oe_equal48.TABLE1_CELLS drifted from the manuscript's fixed Table 1 order"
    )
REPRO = Path("repro")


def _assert_exact(actual, expected, description: str) -> None:
    if tuple(actual) != tuple(expected):
        raise RuntimeError(
            f"{description}: expected {tuple(expected)}, got {tuple(actual)}"
        )


def _assert_close(actual: float, expected: float, description: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"{description}: expected {expected!r}, got {actual!r}")


def _validate_screen(
    doc: dict, path: Path, cell: str, backbone: str, method: str, seed: int
) -> None:
    grid = E.recipes(method)
    expected = {
        "protocol": E.PROTOCOL,
        "phase": "validation-only-screen",
        "grid_protocol": E.GRID_PROTOCOL,
        "cell": cell,
        "backbone": backbone,
        "method": method,
        "seed": seed,
        "n_recipe_trials": 48,
        "n_checkpoint_queries_per_neural_recipe": E.N_CHECKPOINTS,
        "recipe_hash": E.grid_hash(method),
        "all_grid_hash": E.all_grid_hash(),
    }
    for key, value in expected.items():
        if doc.get(key) != value:
            raise RuntimeError(
                f"screen provenance mismatch in {path}: "
                f"{key}={doc.get(key)!r}, expected {value!r}"
            )
    _assert_exact(
        doc.get("selection_seeds", ()),
        E.SELECTION_SEEDS,
        f"selection seeds declared in {path}",
    )
    if doc.get("recipes") != grid:
        raise RuntimeError(f"frozen recipe grid drift in {path}")
    results = doc.get("results", {})
    expected_ids = {str(recipe["id"]) for recipe in grid}
    if set(results) != expected_ids:
        raise RuntimeError(f"screen recipe IDs are incomplete or unexpected in {path}")
    for recipe in grid:
        record = results[str(recipe["id"])]
        if record.get("recipe") != recipe:
            raise RuntimeError(f"screen recipe payload drift in {path}, r={recipe['id']}")
        metrics = record.get("val", {})
        if set(metrics) != {"auroc", "aupr", "fpr95"} or not all(
            math.isfinite(float(metrics[name])) for name in metrics
        ):
            raise RuntimeError(f"invalid validation metrics in {path}, r={recipe['id']}")
    _assert_exact(
        doc.get("data_audit", {}).get("loaded_splits", ()),
        E.SPLITS_BY_PHASE["screen"],
        f"validation-only loaded splits in {path}",
    )
    draw = doc.get("draw_audit", {})
    if not (0 < int(draw.get("n_id", 0)) <= E.N_ID):
        raise RuntimeError(f"invalid ID draw size in {path}")
    if not (0 < int(draw.get("n_ood", 0)) <= E.N_OOD):
        raise RuntimeError(f"invalid auxiliary-OOD draw size in {path}")
    if not (0 < int(draw.get("n_labelled_id", 0)) <= int(draw["n_id"])):
        raise RuntimeError(f"invalid labelled-ID draw size in {path}")
    for name in ("id_index_sha256", "ood_index_sha256"):
        value = draw.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"invalid {name} in {path}")


def _validated_selection_and_screens(
    cell: str, backbone: str, method: str
) -> tuple[dict, Path, list[dict]]:
    selection_path = E._selection_path(cell, backbone, method)
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selection = json.loads(selection_path.read_text())
    _validate_selection(selection, selection_path, cell, backbone, method)

    provenance = selection.get("screen_provenance", ())
    if len(provenance) != len(E.SELECTION_SEEDS):
        raise RuntimeError(f"invalid screen provenance count in {selection_path}")
    screens = []
    screen_files = []
    for seed, item in zip(E.SELECTION_SEEDS, provenance):
        expected_path = E._screen_path(cell, backbone, method, seed)
        if item.get("path") != str(expected_path):
            raise RuntimeError(
                f"screen provenance path mismatch in {selection_path}, seed {seed}"
            )
        if not expected_path.is_file():
            raise FileNotFoundError(expected_path)
        sha256 = E.file_sha256(expected_path)
        if item.get("sha256") != sha256:
            raise RuntimeError(
                f"screen SHA256 mismatch in {selection_path}, seed {seed}"
            )
        screen = json.loads(expected_path.read_text())
        _validate_screen(screen, expected_path, cell, backbone, method, seed)
        screens.append(screen)
        screen_files.append(
            {
                "path": str(expected_path),
                "sha256": sha256,
                "seed": seed,
                "draw_audit": screen["draw_audit"],
            }
        )

    # Recompute the complete validation ranking.  This proves that the frozen
    # selection is the winner of the declared 48-recipe rule, rather than merely
    # checking that its recipe ID happens to be in range.
    ranking = []
    for recipe in E.recipes(method):
        values = [
            float(screen["results"][str(recipe["id"])]["val"]["auroc"])
            for screen in screens
        ]
        ranking.append(
            {
                "recipe_id": recipe["id"],
                "mean_val_auroc": float(np.mean(values)),
                "worst_seed_val_auroc": float(np.min(values)),
                "per_seed_val_auroc": values,
            }
        )
    ranking.sort(
        key=lambda row: (
            row["mean_val_auroc"],
            row["worst_seed_val_auroc"],
            -row["recipe_id"],
        ),
        reverse=True,
    )
    expected_selected = {**ranking[0], "recipe": E.recipes(method)[ranking[0]["recipe_id"]]}
    actual_selected = selection.get("selected", {})
    if (
        actual_selected.get("recipe_id") != expected_selected["recipe_id"]
        or actual_selected.get("recipe") != expected_selected["recipe"]
        or actual_selected.get("per_seed_val_auroc")
        != expected_selected["per_seed_val_auroc"]
    ):
        raise RuntimeError(f"selection is not the recomputed grid winner in {selection_path}")
    _assert_close(
        actual_selected.get("mean_val_auroc"),
        expected_selected["mean_val_auroc"],
        f"selected mean validation AUROC in {selection_path}",
    )
    _assert_close(
        actual_selected.get("worst_seed_val_auroc"),
        expected_selected["worst_seed_val_auroc"],
        f"selected worst-seed validation AUROC in {selection_path}",
    )
    if selection.get("top5") != ranking[:5]:
        raise RuntimeError(f"stored top-5 validation ranking drift in {selection_path}")
    return selection, selection_path, screen_files


def aggregate() -> dict:
    output = {
        "protocol": E.PROTOCOL,
        "phase": "complete-table1-classifier-oe-aggregate",
        "scope_protocol": "equal48-table1-extension-frozen-before-additional-runs",
        "grid_protocol": E.GRID_PROTOCOL,
        "all_grid_hash": E.all_grid_hash(),
        "method_grid_hashes": {method: E.grid_hash(method) for method in METHODS},
        "cells": list(TABLE_ORDER),
        "backbones": list(E.BACKBONES),
        "methods": list(METHODS),
        "selection_seeds": list(E.SELECTION_SEEDS),
        "final_seeds": list(E.FINAL_SEEDS),
        "n_recipe_trials_per_method": 48,
        "n_final_seeds": len(E.FINAL_SEEDS),
        "settings": {},
    }
    paired_screen_draws: dict[tuple[str, str, int], dict] = {}
    paired_final_draws: dict[tuple[str, str, int], dict] = {}
    for backbone in E.BACKBONES:
        for cell in TABLE_ORDER:
            key = f"{cell}/{backbone}"
            output["settings"][key] = {}
            for method in METHODS:
                selection, selection_path, screen_files = _validated_selection_and_screens(
                    cell, backbone, method
                )
                for item in screen_files:
                    paired_key = (cell, backbone, item["seed"])
                    draw = item["draw_audit"]
                    if (
                        paired_key in paired_screen_draws
                        and paired_screen_draws[paired_key] != draw
                    ):
                        raise RuntimeError(
                            f"unpaired validation draw for "
                            f"{cell}/{backbone}/s{item['seed']}: {method}"
                        )
                    paired_screen_draws.setdefault(paired_key, draw)
                selection_sha = E.file_sha256(selection_path)
                docs = []
                final_files = []
                for seed in E.FINAL_SEEDS:
                    path = E._final_path(cell, backbone, method, seed)
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    doc = json.loads(path.read_text())
                    _validate_final(
                        doc,
                        path,
                        selection,
                        selection_sha,
                        cell,
                        backbone,
                        method,
                        seed,
                    )
                    paired_key = (cell, backbone, seed)
                    draw = doc.get("draw_audit", {})
                    if paired_key in paired_final_draws and paired_final_draws[paired_key] != draw:
                        raise RuntimeError(
                            f"unpaired molecule draw for {cell}/{backbone}/s{seed}: {method}"
                        )
                    paired_final_draws.setdefault(paired_key, draw)
                    docs.append(doc)
                    final_files.append(
                        {"path": str(path), "sha256": E.file_sha256(path), "seed": seed}
                    )
                output["settings"][key][method] = {
                    "selected_recipe_id": selection["selected"]["recipe_id"],
                    "selected_recipe": selection["selected"]["recipe"],
                    "mean_selection_val_auroc": selection["selected"]["mean_val_auroc"],
                    "n_validation_trials": selection["n_recipe_trials"],
                    "selection_file": str(selection_path),
                    "selection_sha256": selection_sha,
                    "recipe_hash": selection["recipe_hash"],
                    "all_grid_hash": selection["all_grid_hash"],
                    "screen_files": screen_files,
                    "final_files": final_files,
                    **{
                        metric: summary([float(doc["test"][metric]) for doc in docs])
                        for metric in ("auroc", "aupr", "fpr95")
                    },
                }
    return output


def latex_rows(doc: dict) -> str:
    lines = [
        "% Auto-generated by oe_equal48_table1_aggregate.py; do not edit by hand.",
        "% Column order is exactly the existing WSDM Table 1 header.",
    ]
    for backbone in E.BACKBONES:
        lines.append(f"% {backbone}")
        for method in METHODS:
            cells = []
            for cell in TABLE_ORDER:
                rec = doc["settings"][f"{cell}/{backbone}"][method]["auroc"]
                # The manuscript's existing Table 1 reports three decimals and no
                # uncertainty in-cell; keep that format unchanged.  The JSON and
                # Markdown artifacts retain full precision and sample SD.
                cells.append(f"{rec['mean']:.3f}")
            lines.append(DISPLAY[method] + " & " + " & ".join(cells) + " \\\\")
        lines.append("")
    lines.extend(
        [
            "% Caption note:",
            "% All OE rows use the same auxiliary pools, 48 method-specific validation",
            "% recipes, five selection seeds, and twenty paired final seeds. ODIN-OE",
            "% is the temperature-only frozen-feature adaptation used in this study.",
            "",
        ]
    )
    return "\n".join(lines)


def markdown(doc: dict) -> str:
    labels = [
        "EC50-Scaf", "EC50-Size", "EC50-Assay", "IC50-Scaf", "IC50-Size",
        "IC50-Assay", "HIV-Scaf", "HIV-Size", "PCBA-Scaf", "PCBA-Size",
        "ZINC-Scaf", "ZINC-Size",
    ]
    lines = [
        "# Equal-48 classifier-score OE rows for the existing Table 1",
        "",
        "Every entry is ordinary AUROC, mean +/- sample SD over final seeds 301--320.",
        "",
    ]
    for backbone in E.BACKBONES:
        lines += [f"## {backbone}", "", "| Method | " + " | ".join(labels) + " |",
                  "|---|" + "---:|" * len(labels)]
        for method in METHODS:
            vals = []
            for cell in TABLE_ORDER:
                rec = doc["settings"][f"{cell}/{backbone}"][method]["auroc"]
                vals.append(f"{rec['mean']:.4f} +/- {rec['sample_sd']:.4f}")
            lines.append("| " + DISPLAY[method] + " | " + " | ".join(vals) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    doc = aggregate()
    (REPRO / "oe_equal48_table1_summary.json").write_text(json.dumps(doc, indent=2) + "\n")
    (REPRO / "oe_equal48_table1_rows.tex").write_text(latex_rows(doc))
    (REPRO / "oe_equal48_table1_summary.md").write_text(markdown(doc))
    print("OE_EQUAL48_TABLE1_AGGREGATE_DONE")


if __name__ == "__main__":
    main()
