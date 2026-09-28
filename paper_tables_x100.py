#!/usr/bin/env python3
"""Emit every paper table on the AUROC x 100 scale, with the columns a reviewer will ask for.

Scaling by 100 is the field's normal convention and changes nothing about effect size: a
+0.0012 AUROC difference is +0.12 AUROC points.  It must never be written as "+0.12%",
which reads as a 0.12 percent relative gain.

Each comparison table therefore carries the paired 95% CI and the per-seed win count
alongside the mean, because a mean difference with a CI spanning zero and a 5/10 win rate
cannot support the word "consistently".  Omitting those two columns is what makes a table
falsifiable by a single rerun.

Usage: python paper_tables_x100.py > repro/paper_tables_x100.md
"""
import json, glob
from pathlib import Path
import numpy as np

ORDER = ["MSP", "ODIN", "Energy", "Mahalanobis", "KNN", "LOF",
         "MSP-OE", "ODIN-OE", "Energy-OE", "OE-Mahalanobis", "OE-KNN", "OE-LOF",
         "BalancedOODHead", "RPO"]
NAME = {"BalancedOODHead": "Balanced OOD Head"}
NOOE = {"MSP", "ODIN", "Energy", "Mahalanobis", "KNN", "LOF"}
CELLS = [("ec50_assay", "EC50-Assay"), ("ic50_assay", "IC50-Assay"),
         ("ki_assay", "Ki-Assay"), ("ec50_scaffold", "EC50-Scaffold"),
         ("ic50_scaffold", "IC50-Scaffold"), ("hiv_scaffold", "HIV-Scaffold"),
         ("pcba_scaffold", "PCBA-Scaffold"), ("zinc_scaffold", "ZINC-Scaffold")]
SIZE = [("ec50_size", "EC50-Size"), ("ic50_size", "IC50-Size"), ("hiv_size", "HIV-Size"),
        ("pcba_size", "PCBA-Size"), ("zinc_size", "ZINC-Size")]
FINAL = [("ec50_assay", "minimol", "EC50-Assay / MiniMol"),
         ("ec50_assay", "unimol", "EC50-Assay / Uni-Mol"),
         ("ic50_assay", "minimol", "IC50-Assay / MiniMol"),
         ("ic50_assay", "unimol", "IC50-Assay / Uni-Mol")]


def ci(d):
    d = np.asarray(d, dtype=float)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    return 100 * d.mean(), 100 * (d.mean() - 1.96 * se), 100 * (d.mean() + 1.96 * se)


def moe():
    return {(d["cell"], d["backbone"]): d
            for d in (json.load(open(f)) for f in glob.glob("repro/moe_*.json"))}


def table_matched_oe(M, bb, cells, metric="auroc"):
    L = ["| Method | OE | " + " | ".join(n for _, n in cells) + " | Macro |",
         "|---|:--:|" + "---:|" * (len(cells) + 1)]
    for m in ORDER:
        row, vals = [], []
        for c, _ in cells:
            r = M.get((c, bb), {}).get("methods", {}).get(m)
            if r is None: row.append("--")
            else:
                x = 100 * float(np.mean(r[metric])); vals.append(x); row.append(f"{x:.2f}")
        if not vals: continue
        L.append(f"| {NAME.get(m, m)} | {'x' if m in NOOE else 'v'} | "
                 + " | ".join(row) + f" | {np.mean(vals):.2f} |")
    return "\n".join(L)


def main():
    M = moe()
    out = ["# Paper tables, AUROC x 100",
           "",
           "All numbers are AUROC x 100.  A difference of `+0.20` is **0.20 AUROC points**,",
           "i.e. 0.0020 AUROC.  Never write it as `+0.20%`.",
           ""]

    out += ["## Table 1 -- Matched outlier exposure (MiniMol)", "",
            "Every original baseline is paired with an outlier-exposed counterpart trained on",
            "the identical auxiliary molecules, with exactly nine validation trials each.", ""]
    out += [table_matched_oe(M, "minimol", CELLS), ""]
    out += ["## Table 1b -- Matched outlier exposure (Uni-Mol)", ""]
    out += [table_matched_oe(M, "unimol", CELLS), ""]
    out += ["## Table 1c -- Size cells (shortcut diagnostic)", "",
            "A single zero-training descriptor separates these cells perfectly, so every",
            "OE-capable method scores ~100 for free.  Reported as a diagnostic, not a ranking.",
            ""]
    out += [table_matched_oe(M, "minimol", SIZE), ""]

    out += ["## Table 2 -- Optimised RPO vs capacity- and data-matched balanced BCE", "",
            "Ten paired final seeds.  The CI and win-count columns are mandatory: without",
            "them the table cannot be distinguished from a null result.", "",
            "| Setting | RPO | Balanced BCE | Delta (points) | Paired 95% CI | Seeds RPO > BCE |",
            "|---|---:|---:|---:|---:|---:|"]
    rows, macro_r, macro_b = [], [], []
    for cell, bb, label in FINAL:
        f = Path(f"repro/rpo_opt_final_{cell}_{bb}.json")
        if not f.exists(): continue
        d = json.load(open(f))["results"]
        r = np.array([x["test_auroc"] for x in d["rpo"]])
        b = np.array([x["test_auroc"] for x in d["bce"]])
        m, lo, hi = ci(r - b)
        macro_r.append(r); macro_b.append(b)
        rows.append(f"| {label} | **{100*r.mean():.2f}** | {100*b.mean():.2f} | "
                    f"{m:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {int((r > b).sum())}/{len(r)} |")
    out += rows
    if macro_r:
        R = np.mean(macro_r, 0); B = np.mean(macro_b, 0)
        m, lo, hi = ci(R - B)
        out += [f"| **Macro** | **{100*R.mean():.2f}** | {100*B.mean():.2f} | {m:+.2f} | "
                f"[{lo:+.2f}, {hi:+.2f}] | {int((R > B).sum())}/{len(R)} |"]
    out += ["",
            "**What this table supports.** Under matched auxiliary data, matched head capacity",
            "and a matched 32-recipe tuning budget, RPO's mean AUROC is nominally above",
            "balanced BCE in all four primary settings.",
            "",
            "**What it does not support.** Every paired 95% CI includes zero and the per-seed",
            "win rate is 5/10 to 7/10, so the two objectives are statistically indistinguishable.",
            "The word *consistently* is not defensible at a 5/10 win rate; *nominally higher in",
            "all four settings* is.",
            ""]

    out += ["## Table 3 -- Where RPO sits among matched-OE detectors", "",
            "| Setting | RPO | Strongest matched-OE baseline | Gap (points) |",
            "|---|---:|---|---:|"]
    for cell, bb, label in FINAL:
        f = Path(f"repro/rpo_opt_final_{cell}_{bb}.json")
        mm = M.get((cell, bb), {}).get("methods", {})
        if not f.exists() or not mm: continue
        r = 100 * np.mean([x["test_auroc"]
                           for x in json.load(open(f))["results"]["rpo"]])
        best = max(((100 * np.mean(v["auroc"]), k) for k, v in mm.items()
                    if k not in NOOE and k not in ("RPO",)), default=(None, None))
        if best[0] is None: continue
        out.append(f"| {label} | {r:.2f} | {best[1]} {best[0]:.2f} | {r-best[0]:+.2f} |")
    out += ["",
            "RPO is above every logit-based and score-based OE method but below the",
            "two-sample density/distance detectors in all four settings.",
            ""]

    out += ["## Wording that survives review", "",
            "Supportable:",
            "",
            "> Under matched auxiliary data, capacity and tuning budget, RPO attains nominally",
            "> higher mean AUROC than balanced BCE in all four primary assay/backbone settings",
            "> (macro +0.12 AUROC points), although the paired confidence intervals include",
            "> zero; the two objectives should be read as statistically indistinguishable.",
            "",
            "Not supportable, and each is refutable from the artifacts in `repro/`:",
            "",
            "- \"RPO consistently outperforms BCE\"  (5/10 seeds on IC50-Assay/MiniMol)",
            "- \"the gain comes from the preference objective\"  (shared-recipe crossover",
            "  macro is -0.0001; the gain follows the training recipe)",
            "- \"+0.20%\"  (it is +0.20 points, i.e. +0.0020 AUROC)",
            "- \"RPO is the best OE detector\"  (below OE-Mahalanobis/OE-KNN in 4/4 settings)",
            ""]
    print("\n".join(out))


if __name__ == "__main__":
    main()
