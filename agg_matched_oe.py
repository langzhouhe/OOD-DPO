#!/usr/bin/env python3
"""Aggregate the matched-OE table and the RPO-Lite capacity sweep into paper-ready tables.

Outputs markdown to repro/matched_oe_tables.md and a machine-readable roll-up to
repro/matched_oe_summary.json.

Pre-registered gate for RPO-Lite (fixed before the runs, see the session record):
at some head width, RPO - Balanced OOD Head must be >= +0.010 AUROC on BOTH clean assay
cells with 5/5 paired seeds positive on each.
"""
import json, glob
from pathlib import Path
import numpy as np

CELLS = ["ec50_assay", "ic50_assay", "ec50_scaffold", "ic50_scaffold",
         "hiv_scaffold", "pcba_scaffold", "zinc_scaffold",
         "ec50_size", "ic50_size", "hiv_size", "pcba_size", "zinc_size"]
SHORT = {"ec50_assay": "EC50-Assay", "ic50_assay": "IC50-Assay",
         "ec50_scaffold": "EC50-Scaf", "ic50_scaffold": "IC50-Scaf",
         "hiv_scaffold": "HIV-Scaf", "pcba_scaffold": "PCBA-Scaf",
         "zinc_scaffold": "ZINC-Scaf", "ec50_size": "EC50-Size",
         "ic50_size": "IC50-Size", "hiv_size": "HIV-Size",
         "pcba_size": "PCBA-Size", "zinc_size": "ZINC-Size"}
ORDER = ["MSP", "ODIN", "Energy", "Mahalanobis", "KNN", "LOF",
         "MSP-OE", "ODIN-OE", "Energy-OE", "OE-Mahalanobis", "OE-KNN", "OE-LOF",
         "BalancedOODHead", "RPO"]
NAME = {"BalancedOODHead": "Balanced OOD Head"}
ASSAY = ["ec50_assay", "ic50_assay"]
PRIMARY = ["ec50_assay", "ic50_assay", "ec50_scaffold", "ic50_scaffold"]
GOODSCAF = ["hiv_scaffold", "pcba_scaffold", "zinc_scaffold"]
SIZE = ["ec50_size", "ic50_size", "hiv_size", "pcba_size", "zinc_size"]


def load(pat):
    out = {}
    for f in glob.glob(pat):
        d = json.load(open(f))
        out[(d["cell"], d["backbone"])] = d
    return out


def macro(res, bb, method, cells, metric="auroc"):
    v = [np.mean(res[(c, bb)]["methods"][method][metric])
         for c in cells if (c, bb) in res and method in res[(c, bb)]["methods"]]
    return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)


def table(res, bb, cells, metric="auroc"):
    L = [f"| Method | OE | " + " | ".join(SHORT[c] for c in cells) + " | macro |",
         "|---|:--:|" + "---:|" * (len(cells) + 1)]
    for m in ORDER:
        row, vals = [], []
        for c in cells:
            r = res.get((c, bb), {}).get("methods", {}).get(m)
            if r is None:
                row.append("—")
            else:
                x = float(np.mean(r[metric])); vals.append(x); row.append(f"{x:.4f}")
        if not vals:
            continue
        oe = "✓" if (m in ("BalancedOODHead", "RPO") or m.endswith("-OE")
                     or m.startswith("OE-")) else "×"
        L.append(f"| {NAME.get(m, m)} | {oe} | " + " | ".join(row) +
                 f" | {np.mean(vals):.4f} |")
    return "\n".join(L)


def paired(res, bb, cell, a="RPO", b="BalancedOODHead"):
    r = res.get((cell, bb), {}).get("methods", {})
    if a not in r or b not in r:
        return None
    d = np.array(r[a]["auroc"]) - np.array(r[b]["auroc"])
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    return dict(mean=float(d.mean()), pos=int((d > 0).sum()), n=len(d),
                lo=float(d.mean() - 1.96 * se), hi=float(d.mean() + 1.96 * se))


def main():
    res = load("repro/moe_*.json")
    cap = load("repro/cap_*.json")
    S, out = {}, []
    out.append("# Matched outlier-exposure comparison\n")
    out.append(f"Loaded {len(res)} matched-OE configs and {len(cap)} capacity configs.\n")

    for bb in ("minimol", "unimol"):
        if not any(k[1] == bb for k in res):
            continue
        out.append(f"\n## Backbone: {bb}\n")
        out.append("### Clean cells (assay) + scaffold — AUROC\n")
        out.append(table(res, bb, PRIMARY + GOODSCAF))
        out.append("\n### Size cells (shortcut diagnostic — single descriptor gives 1.000) — AUROC\n")
        out.append(table(res, bb, SIZE))
        out.append("\n### Clean cells — FPR95 (lower is better)\n")
        out.append(table(res, bb, ASSAY, "fpr95"))

        rank = []
        for m in ORDER:
            v, n = macro(res, bb, m, ASSAY)
            if n == len(ASSAY):
                rank.append((v, m))
        rank.sort(reverse=True)
        out.append("\n### Ranking on the two clean assay cells (macro AUROC)\n")
        out.append("| # | Method | OE | macro AUROC |")
        out.append("|---:|---|:--:|---:|")
        for i, (v, m) in enumerate(rank, 1):
            oe = "✓" if (m in ("BalancedOODHead", "RPO") or m.endswith("-OE")
                         or m.startswith("OE-")) else "×"
            out.append(f"| {i} | {NAME.get(m, m)} | {oe} | {v:.4f} |")
        S[f"assay_rank_{bb}"] = [(m, round(v, 4)) for v, m in rank]

        out.append("\n### RPO vs Balanced OOD Head, paired over the 5 final seeds\n")
        out.append("| cell | Δ AUROC | 95% CI | seeds RPO>BOH |")
        out.append("|---|---:|---|---:|")
        for c in CELLS:
            p = paired(res, bb, c)
            if p:
                out.append(f"| {SHORT[c]} | {p['mean']:+.4f} | "
                           f"[{p['lo']:+.4f}, {p['hi']:+.4f}] | {p['pos']}/{p['n']} |")
        S[f"rpo_vs_boh_{bb}"] = {c: paired(res, bb, c) for c in CELLS
                                 if paired(res, bb, c)}

    # ---------------- capacity sweep ----------------
    if cap:
        out.append("\n\n# RPO-Lite: head-capacity sweep (Δ = RPO − Balanced OOD Head)\n")
        widths = ["4", "8", "16", "32", "64", "128", "256", "full"]
        for bb in ("minimol", "unimol"):
            keys = [k for k in cap if k[1] == bb]
            if not keys:
                continue
            out.append(f"\n## Backbone: {bb}\n")
            out.append("| cell | " + " | ".join(f"h={w}" for w in widths) + " |")
            out.append("|---|" + "---:|" * len(widths))
            for c in CELLS:
                if (c, bb) not in cap:
                    continue
                w = cap[(c, bb)]["widths"]
                cells_ = []
                for x in widths:
                    if x in w:
                        cells_.append(f"{w[x]['delta_mean']:+.4f}"
                                      f"<br><sub>{w[x]['delta_pos_seeds']}/5</sub>")
                    else:
                        cells_.append("—")
                out.append(f"| {SHORT[c]} | " + " | ".join(cells_) + " |")
            out.append("\nAbsolute AUROC at each width:\n")
            out.append("| cell | obj | " + " | ".join(f"h={w}" for w in widths) + " |")
            out.append("|---|---|" + "---:|" * len(widths))
            for c in CELLS:
                if (c, bb) not in cap:
                    continue
                w = cap[(c, bb)]["widths"]
                for obj in ("BalancedOODHead", "RPO"):
                    vals = [f"{np.mean(w[x][obj]['auroc']):.4f}" if x in w else "—"
                            for x in widths]
                    out.append(f"| {SHORT[c]} | {NAME.get(obj, obj)} | " +
                               " | ".join(vals) + " |")

        # gate
        out.append("\n## Pre-registered gate\n")
        out.append("Requirement: at some width, Δ ≥ +0.010 on BOTH assay cells "
                   "with 5/5 paired seeds positive on each.\n")
        gate = {}
        for bb in ("minimol", "unimol"):
            for x in widths:
                ok, detail = True, []
                for c in ASSAY:
                    w = cap.get((c, bb), {}).get("widths", {}).get(x)
                    if w is None:
                        ok = False; detail.append(f"{SHORT[c]}: missing"); continue
                    d, p = w["delta_mean"], w["delta_pos_seeds"]
                    detail.append(f"{SHORT[c]}: {d:+.4f} ({p}/5)")
                    if not (d >= 0.010 and p == 5):
                        ok = False
                gate[f"{bb}|h={x}"] = {"pass": ok, "detail": detail}
                out.append(f"- **{bb}, h={x}** — {'PASS' if ok else 'FAIL'}: "
                           + "; ".join(detail))
        S["gate"] = gate
        S["gate_any_pass"] = any(v["pass"] for v in gate.values())
        out.append(f"\n**Overall gate: "
                   f"{'PASS' if S['gate_any_pass'] else 'FAIL'}**")

    Path("repro").mkdir(exist_ok=True)
    open("repro/matched_oe_tables.md", "w").write("\n".join(out) + "\n")
    json.dump(S, open("repro/matched_oe_summary.json", "w"), indent=2, default=float)
    print("\n".join(out))


if __name__ == "__main__":
    main()
