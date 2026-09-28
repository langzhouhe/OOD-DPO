#!/usr/bin/env python3
"""Aggregate the three favorable-setting axes, and apply the pre-registered tests.

The pre-registered claims are about SHAPE, not about any single winning configuration:

  A. Delta(RPO - BCE) decays as head width grows            -> Spearman(width, Delta) < 0,
                                                               same sign on both cells.
  B. Delta(RPO - BCE) grows from far to near exposure       -> Spearman(difficulty, Delta) > 0,
                                                               same sign on both cells.
  C. Any configuration where Delta >= +0.010 with 5/5 paired seeds only counts as evidence
     for pairwise structure if BOTH pointwise controls (hard_bce, focal_bce) fail to reach
     the same Delta.  E23 showed hard-example concentration reproduces pairwise gains, so a
     configuration where hard_bce >= RPO is evidence AGAINST the pairwise claim.

The multiplicity of the grid is reported explicitly: with 120 comparisons, a handful of
+0.010 cells is what noise alone produces.

Outputs repro/favorable_tables.md and repro/favorable_summary.json.
"""
import json, glob
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

WIDTHS = ["4", "8", "16", "32", "128", "full"]
TIERS = ["far", "random", "medium", "near"]          # far -> near = easy -> hard
ALLT = TIERS + ["all"]
OBJS = ["rpo", "hard_bce", "focal_bce", "pauc", "energy_margin"]
SELS = ["auroc", "fpr_ood_at95tpr_id"]
CELLS = ["ec50_assay", "ic50_assay"]
THRESH, NSEED = 0.010, 5


def load():
    R = {}
    for f in glob.glob("repro/fav_*.json"):
        d = json.load(open(f))
        R[(d["cell"], d["width"], d["tier"])] = d
    return R


def delta(d, obj, sel):
    a = d["arms"].get(f"{obj}|sel_{sel}"); b = d["arms"].get(f"bce|sel_{sel}")
    if a is None or b is None:
        return None
    x = np.array(a["auroc"]) - np.array(b["auroc"])
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return {"mean": float(x.mean()), "pos": int((x > 0).sum()), "n": len(x),
            "lo": float(x.mean() - 1.96 * se), "hi": float(x.mean() + 1.96 * se)}


def main():
    R = load()
    S, out = {}, ["# Favorable-setting matrix (2 clean assay cells, MiniMol)\n",
                  f"Loaded {len(R)} of {len(CELLS)*len(WIDTHS)*len(ALLT)} configurations.\n"]

    # ---------- per-cell grids ----------
    for sel in SELS:
        out.append(f"\n## Selection rule: validation {sel}\n")
        for cell in CELLS:
            out.append(f"\n### {cell} — Δ AUROC vs Balanced BCE (5 paired seeds)\n")
            out.append("| width | " + " | ".join(
                f"{t} / {o}" for t in ALLT for o in ("RPO", "hard", "focal")) + " |")
            out.append("|---|" + "---:|" * (len(ALLT) * 3))
            for w in WIDTHS:
                row = []
                for t in ALLT:
                    d = R.get((cell, w, t))
                    for o in ("rpo", "hard_bce", "focal_bce"):
                        x = delta(d, o, sel) if d else None
                        row.append("—" if x is None else
                                   f"{x['mean']:+.4f}<sub>{x['pos']}/{x['n']}</sub>")
                out.append(f"| h={w} | " + " | ".join(row) + " |")
            out.append("\nAbsolute AUROC (BCE / RPO):\n")
            out.append("| width | " + " | ".join(ALLT) + " |")
            out.append("|---|" + "---:|" * len(ALLT))
            for w in WIDTHS:
                row = []
                for t in ALLT:
                    d = R.get((cell, w, t))
                    if not d: row.append("—"); continue
                    b = np.mean(d["arms"][f"bce|sel_{sel}"]["auroc"])
                    r = np.mean(d["arms"][f"rpo|sel_{sel}"]["auroc"])
                    row.append(f"{b:.4f} / {r:.4f}")
                out.append(f"| h={w} | " + " | ".join(row) + " |")

    # ---------- axis A: capacity ----------
    out.append("\n\n## Axis A — pre-registered: Δ decays as width grows\n")
    out.append("| cell | tier | sel | Spearman(width, Δ) | p | Δ at h=4 | Δ at full |")
    out.append("|---|---|---|---:|---:|---:|---:|")
    A = {}
    for cell in CELLS:
        for t in ALLT:
            for sel in SELS:
                ds = [delta(R.get((cell, w, t)), "rpo", sel) for w in WIDTHS]
                xs = [(i, d["mean"]) for i, d in enumerate(ds) if d]
                if len(xs) < 4: continue
                rho, p = spearmanr([a for a, _ in xs], [b for _, b in xs])
                A[f"{cell}|{t}|{sel}"] = {"rho": float(rho), "p": float(p)}
                out.append(f"| {cell} | {t} | {sel} | {rho:+.3f} | {p:.3f} | "
                           f"{ds[0]['mean']:+.4f} |" if ds[0] else
                           f"| {cell} | {t} | {sel} | {rho:+.3f} | {p:.3f} | — |")
                out[-1] += f" {ds[-1]['mean']:+.4f} |" if ds[-1] else " — |"
    S["axisA"] = A

    # ---------- axis B: exposure difficulty ----------
    out.append("\n\n## Axis B — pre-registered: Δ grows from far to near exposure\n")
    out.append("| cell | width | sel | Spearman(difficulty, Δ) | p | far | medium | near |")
    out.append("|---|---|---|---:|---:|---:|---:|---:|")
    B = {}
    for cell in CELLS:
        for w in WIDTHS:
            for sel in SELS:
                ds = [delta(R.get((cell, w, t)), "rpo", sel) for t in TIERS]
                xs = [(i, d["mean"]) for i, d in enumerate(ds) if d]
                if len(xs) < 4: continue
                rho, p = spearmanr([a for a, _ in xs], [b for _, b in xs])
                B[f"{cell}|h{w}|{sel}"] = {"rho": float(rho), "p": float(p)}
                g = lambda i: f"{ds[i]['mean']:+.4f}" if ds[i] else "—"
                out.append(f"| {cell} | h={w} | {sel} | {rho:+.3f} | {p:.3f} | "
                           f"{g(0)} | {g(2)} | {g(3)} |")
    S["axisB"] = B

    # ---------- axis C: pointwise controls ----------
    out.append("\n\n## Axis C — every configuration clearing +0.010 with 5/5 seeds, "
               "against the pointwise controls\n")
    out.append("| cell | width | tier | sel | Δ RPO | Δ hard_bce | Δ focal_bce | "
               "Δ pAUC | pairwise-specific? |")
    out.append("|---|---|---|---|---:|---:|---:|---:|:--:|")
    hits, surviving, total = 0, 0, 0
    for cell in CELLS:
        for w in WIDTHS:
            for t in ALLT:
                d = R.get((cell, w, t))
                if not d: continue
                for sel in SELS:
                    total += 1
                    x = delta(d, "rpo", sel)
                    if not x or not (x["mean"] >= THRESH and x["pos"] == NSEED):
                        continue
                    hits += 1
                    ctrl = {o: delta(d, o, sel) for o in
                            ("hard_bce", "focal_bce", "pauc")}
                    beat = all(c is None or c["mean"] < x["mean"] for c in ctrl.values())
                    surviving += bool(beat)
                    g = lambda o: f"{ctrl[o]['mean']:+.4f}" if ctrl[o] else "—"
                    out.append(f"| {cell} | h={w} | {t} | {sel} | **{x['mean']:+.4f}** | "
                               f"{g('hard_bce')} | {g('focal_bce')} | {g('pauc')} | "
                               f"{'YES' if beat else 'no'} |")
    if hits == 0:
        out.append("| — | — | — | — | — | — | — | — | *no configuration cleared the bar* |")
    out.append(f"\n**{hits} of {total} comparisons cleared Δ ≥ +{THRESH:.3f} with "
               f"{NSEED}/{NSEED} paired seeds; {surviving} of those also beat every "
               f"pointwise control.**  With {total} comparisons, a handful of hits is "
               f"what noise alone produces — only the Axis-A/Axis-B shape tests and the "
               f"control column carry evidence.")
    S["axisC"] = {"total": total, "hits": hits, "surviving": surviving}

    # ---------- parameter efficiency ----------
    out.append("\n\n## Parameter efficiency — smallest head reaching the full head's BCE AUROC\n")
    out.append("| cell | tier | sel | full-head BCE | smallest RPO head ≥ that | params |")
    out.append("|---|---|---|---:|---|---:|")
    P = {}
    for cell in CELLS:
        for t in ALLT:
            for sel in SELS:
                f = R.get((cell, "full", t))
                if not f: continue
                target = np.mean(f["arms"][f"bce|sel_{sel}"]["auroc"])
                win = None
                for w in WIDTHS:
                    d = R.get((cell, w, t))
                    if not d: continue
                    if np.mean(d["arms"][f"rpo|sel_{sel}"]["auroc"]) >= target:
                        win = (w, d["params"]); break
                P[f"{cell}|{t}|{sel}"] = {"target": float(target),
                                          "width": win[0] if win else None,
                                          "params": win[1] if win else None}
                out.append(f"| {cell} | {t} | {sel} | {target:.4f} | "
                           f"{'h='+str(win[0]) if win else 'none'} | "
                           f"{win[1]:,} |" if win else
                           f"| {cell} | {t} | {sel} | {target:.4f} | none | — |")
    S["param_efficiency"] = P

    Path("repro").mkdir(exist_ok=True)
    open("repro/favorable_tables.md", "w").write("\n".join(out) + "\n")
    json.dump(S, open("repro/favorable_summary.json", "w"), indent=2, default=float)
    print("\n".join(out))


if __name__ == "__main__":
    main()
