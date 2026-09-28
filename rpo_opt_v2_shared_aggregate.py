#!/usr/bin/env python3
"""Aggregate the objective-blind shared-recipe loss-only comparison."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rpo_opt_v2_final_one import FINAL_SEEDS


SETTINGS = tuple((c, b) for c in ("ec50_assay", "ic50_assay")
                 for b in ("minimol", "unimol"))
LABEL = {
    ("ec50_assay", "minimol"): "EC50-Assay / MiniMol",
    ("ec50_assay", "unimol"): "EC50-Assay / Uni-Mol",
    ("ic50_assay", "minimol"): "IC50-Assay / MiniMol",
    ("ic50_assay", "unimol"): "IC50-Assay / Uni-Mol",
}


def ci(delta: np.ndarray) -> list[float]:
    half = 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))
    return [float(delta.mean() - half), float(delta.mean() + half)]


def main() -> None:
    out = {"protocol": "rpo-opt-v2-shared-recipe-final-aggregate", "settings": {}}
    lines = [
        "# RPO-OE v2 objective-blind shared-recipe comparison\n",
        "| Setting | Recipe | RPO AUROC | BCE AUROC | Delta | Paired 95% CI | Wins |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    r_all, b_all = [], []
    for cell, backbone in SETTINGS:
        docs = [json.loads((Path("repro") /
                f"rpo_opt_v2_shared_final_{cell}_{backbone}_s{s}.json").read_text())
                for s in FINAL_SEEDS]
        assert all(d["protocol"] == "rpo-opt-v2-objective-blind-shared-recipe-final"
                   for d in docs)
        assert len({d["recipe_id"] for d in docs}) == 1
        r = np.asarray([d["results"]["rpo"]["test_auroc"] for d in docs])
        b = np.asarray([d["results"]["bce"]["test_auroc"] for d in docs])
        delta = r - b
        interval = ci(delta)
        key = f"{cell}|{backbone}"
        out["settings"][key] = {
            "recipe_id": docs[0]["recipe_id"],
            "recipe": docs[0]["recipe"],
            "rpo": r.tolist(), "bce": b.tolist(),
            "rpo_mean": float(r.mean()), "bce_mean": float(b.mean()),
            "delta": float(delta.mean()), "ci95": interval,
            "rpo_wins": int((delta > 0).sum()),
        }
        lines.append(f"| {LABEL[(cell, backbone)]} | {docs[0]['recipe_id']} | "
                     f"{r.mean():.4f} | {b.mean():.4f} | {delta.mean():+.4f} | "
                     f"[{interval[0]:+.4f}, {interval[1]:+.4f}] | "
                     f"{int((delta > 0).sum())}/{len(delta)} |")
        r_all.append(r); b_all.append(b)
    r_macro = np.stack(r_all).mean(0)
    b_macro = np.stack(b_all).mean(0)
    delta = r_macro - b_macro
    interval = ci(delta)
    out["macro"] = {"rpo_mean": float(r_macro.mean()), "bce_mean": float(b_macro.mean()),
                    "delta": float(delta.mean()), "ci95": interval,
                    "rpo_wins": int((delta > 0).sum())}
    lines.append(f"| **Macro** | -- | **{r_macro.mean():.4f}** | "
                 f"**{b_macro.mean():.4f}** | **{delta.mean():+.4f}** | "
                 f"**[{interval[0]:+.4f}, {interval[1]:+.4f}]** | "
                 f"**{int((delta > 0).sum())}/{len(delta)}** |")
    Path("repro/rpo_opt_v2_shared_summary.json").write_text(json.dumps(out, indent=2))
    Path("repro/rpo_opt_v2_shared_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
