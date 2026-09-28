#!/usr/bin/env python3
"""Aggregate the 20 frozen final Hinge seeds into JSON and Markdown."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import pairwise_hinge_v2_common as H


SETTINGS = tuple(
    (cell, backbone)
    for cell in ("ec50_assay", "ic50_assay")
    for backbone in ("minimol", "unimol")
)
PRETTY = {
    ("ec50_assay", "minimol"): "EC50-Assay / MiniMol",
    ("ec50_assay", "unimol"): "EC50-Assay / Uni-Mol",
    ("ic50_assay", "minimol"): "IC50-Assay / MiniMol",
    ("ic50_assay", "unimol"): "IC50-Assay / Uni-Mol",
}


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def main() -> None:
    out = {
        "protocol": "pairwise-hinge-v2-frozen-final-aggregate",
        "margin": H.MARGIN,
        "recipe_hash": H.EXPECTED_UNION_RECIPE_HASH,
        "final_seeds": list(H.FINAL_SEEDS),
        "settings": {},
    }
    lines = [
        "# Pairwise-Hinge v2: frozen final results\n",
        "| Setting | AUROC | AUPR | FPR95 | Recipe |",
        "|---|---:|---:|---:|---:|",
    ]
    macro_auroc = []
    for cell, backbone in SETTINGS:
        paths = [
            Path("repro")
            / f"pairwise_hinge_v2_final_{cell}_{backbone}_s{seed}.json"
            for seed in H.FINAL_SEEDS
        ]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise RuntimeError(f"final artifacts are incomplete: {missing}")
        docs = [json.loads(path.read_text()) for path in paths]
        selection_hashes = {doc["selection_sha256"] for doc in docs}
        recipe_ids = {int(doc["result"]["recipe_id"]) for doc in docs}
        if len(selection_hashes) != 1 or len(recipe_ids) != 1:
            raise RuntimeError(f"selection drift for {cell}/{backbone}")
        record = {
            "selection_sha256": next(iter(selection_hashes)),
            "recipe_id": next(iter(recipe_ids)),
            "recipe": docs[0]["result"]["recipe"],
        }
        for metric in ("val_auroc", "test_auroc", "test_aupr", "test_fpr95"):
            values = [float(doc["result"][metric]) for doc in docs]
            mean, sd = mean_sd(values)
            record[metric] = {"values": values, "mean": mean, "sd": sd}
        out["settings"][f"{cell}|{backbone}"] = record
        macro_auroc.append(np.asarray(record["test_auroc"]["values"]))
        lines.append(
            f"| {PRETTY[(cell, backbone)]} | "
            f"{record['test_auroc']['mean']:.4f} +/- {record['test_auroc']['sd']:.4f} | "
            f"{record['test_aupr']['mean']:.4f} +/- {record['test_aupr']['sd']:.4f} | "
            f"{record['test_fpr95']['mean']:.4f} +/- {record['test_fpr95']['sd']:.4f} | "
            f"r{record['recipe_id']} |"
        )
    macro = np.stack(macro_auroc).mean(axis=0)
    out["macro_test_auroc"] = {
        "values": macro.tolist(),
        "mean": float(macro.mean()),
        "sd": float(macro.std(ddof=1)),
    }
    lines.append(
        f"| **Macro** | **{macro.mean():.4f} +/- {macro.std(ddof=1):.4f}** | - | - | - |"
    )
    Path("repro/pairwise_hinge_v2_final_summary.json").write_text(
        json.dumps(out, indent=2)
    )
    Path("repro/pairwise_hinge_v2_final_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

