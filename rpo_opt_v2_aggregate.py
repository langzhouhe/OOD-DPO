#!/usr/bin/env python3
"""Aggregate v2 final seeds into canonical JSON and a paper-ready table."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rpo_opt_v2_final_one import FINAL_SEEDS


CELLS = tuple(
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


def ci95(x: np.ndarray) -> tuple[float, float]:
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() - 1.96 * se), float(x.mean() + 1.96 * se)


def main() -> None:
    out = {"protocol": "rpo-opt-v2-frozen-final-aggregate", "settings": {}}
    lines = [
        "# RPO-OE tuning v2: paper-ready frozen re-evaluation\n",
        "| Setting | RPO AUROC | Balanced BCE AUROC | Delta | Paired 95% CI | RPO wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    macro_delta = []
    rpo_macro, bce_macro = [], []
    for cell, backbone in CELLS:
        docs = [
            json.loads(
                (Path("repro") / f"rpo_opt_v2_final_{cell}_{backbone}_s{s}.json").read_text()
            )
            for s in FINAL_SEEDS
        ]
        rec = {"seeds": list(FINAL_SEEDS), "rpo": {}, "bce": {}}
        for objective in ("rpo", "bce"):
            for metric in ("test_auroc", "test_aupr", "test_fpr95", "val_auroc"):
                rec[objective][metric] = [d["results"][objective][metric] for d in docs]
            rec[objective]["recipe_id"] = docs[0]["results"][objective]["recipe_id"]
            rec[objective]["recipe"] = docs[0]["results"][objective]["recipe"]
        r = np.asarray(rec["rpo"]["test_auroc"])
        b = np.asarray(rec["bce"]["test_auroc"])
        delta = r - b
        lo, hi = ci95(delta)
        rec["paired"] = {
            "delta_auroc_mean": float(delta.mean()),
            "ci95": [lo, hi],
            "rpo_wins": int((delta > 0).sum()),
        }
        key = f"{cell}|{backbone}"
        out["settings"][key] = rec
        lines.append(
            f"| {PRETTY[(cell, backbone)]} | {r.mean():.4f} | {b.mean():.4f} | "
            f"{delta.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
            f"{int((delta > 0).sum())}/{len(delta)} |"
        )
        rpo_macro.append(r)
        bce_macro.append(b)
        macro_delta.append(delta)

    rpo_macro = np.stack(rpo_macro).mean(0)
    bce_macro = np.stack(bce_macro).mean(0)
    macro_delta = rpo_macro - bce_macro
    lo, hi = ci95(macro_delta)
    out["macro"] = {
        "rpo_auroc": float(rpo_macro.mean()),
        "bce_auroc": float(bce_macro.mean()),
        "delta": float(macro_delta.mean()),
        "ci95": [lo, hi],
        "rpo_wins": int((macro_delta > 0).sum()),
        "n": len(macro_delta),
    }
    lines.append(
        f"| **Macro** | **{rpo_macro.mean():.4f}** | **{bce_macro.mean():.4f}** | "
        f"**{macro_delta.mean():+.4f}** | **[{lo:+.4f}, {hi:+.4f}]** | "
        f"**{int((macro_delta > 0).sum())}/{len(macro_delta)}** |"
    )
    Path("repro/rpo_opt_v2_final_summary.json").write_text(json.dumps(out, indent=2))
    Path("repro/rpo_opt_v2_final_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

