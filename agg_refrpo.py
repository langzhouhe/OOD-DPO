#!/usr/bin/env python3
"""Judge the frozen Reference-Anchored RPO.

Primary metric, frozen: MACRO AUROC over all endpoint x backbone configurations, ranked
across SINGLE FIXED METHODS.  A per-cell "best external baseline" column is NOT a
deployable method -- it is a test oracle -- and is never used as the comparison target.

Decision rules, fixed before the confirmation runs:
  * Reference-Anchored RPO (= unweighted ref_rpo) must hold the best single-method macro.
  * It must stay above Reference-BCE (same reference, same head, pointwise residual).
  * It must beat `fusion` -- an independently trained original RPO head combined with the
    reference by a validation-selected linear weight.  If fusion matches it, the gain is
    plain score addition, not joint residual ranking.
  * Direction must hold in at least 3 of the 4 development configurations; the Ki-Assay
    configurations are post-freeze confirmation and are reported separately so the
    development macro can be read without them.

Outputs repro/refrpo_tables.md and repro/refrpo_summary.json.
"""
import json, glob
from pathlib import Path
import numpy as np

DEV = [("ec50_assay", "minimol"), ("ec50_assay", "unimol"),
       ("ic50_assay", "minimol"), ("ic50_assay", "unimol")]
CONF = [("ki_assay", "minimol"), ("ki_assay", "unimol")]
LADDER = ["ref_only", "ref_bce", "ref_weighted_bce", "fusion", "ref_rpo", "ref_rpo_ess"]
PRETTY = {"ref_only": "OE-Mahalanobis (reference)", "ref_bce": "Reference-BCE",
          "ref_weighted_bce": "Reference-weighted BCE", "fusion": "Reference + independent RPO (fusion)",
          "ref_rpo": "Reference-Anchored RPO", "ref_rpo_ess": "Reference-weighted RPO (ESS)"}
EXT = ["OE-Mahalanobis", "OE-KNN", "BalancedOODHead", "RPO", "Energy-OE", "OE-LOF"]
SENS = ["ref_rpo_ess0.25", "ref_rpo_ess0.75", "ref_rpo_tau1", "ref_rpo_tau5", "ref_rpo_tau10"]


def load():
    R = {(d["cell"], d["backbone"]): d
         for d in (json.load(open(f)) for f in glob.glob("repro/refrpo_*.json"))}
    M = {(d["cell"], d["backbone"]): d
         for d in (json.load(open(f)) for f in glob.glob("repro/moe_*.json"))}
    return R, M


def au(R, k, arm):
    a = R.get(k, {}).get("arms", {}).get(arm)
    return np.array(a["auroc"]) if a else None


def block(R, M, cfgs, title, out):
    have = [k for k in cfgs if k in R]
    if not have:
        return {}
    out.append(f"\n### {title}\n")
    out.append("| method | " + " | ".join(f"{c}/{b}" for c, b in have) + " | macro |")
    out.append("|---|" + "---:|" * (len(have) + 1))
    macro = {}
    for arm in LADDER:
        v = [au(R, k, arm) for k in have]
        if any(x is None for x in v): continue
        m = [float(x.mean()) for x in v]
        macro[PRETTY[arm]] = float(np.mean(m))
        out.append(f"| {PRETTY[arm]} | " + " | ".join(f"{x:.4f}" for x in m) +
                   f" | {np.mean(m):.4f} |")
    for e in EXT:
        v = [M.get(k, {}).get("methods", {}).get(e) for k in have]
        if any(x is None for x in v): continue
        m = [float(np.mean(x["auroc"])) for x in v]
        macro[e] = float(np.mean(m))
        out.append(f"| {e} | " + " | ".join(f"{x:.4f}" for x in m) +
                   f" | {np.mean(m):.4f} |")
    out.append(f"\n**Single-method ranking by macro AUROC ({title}):**\n")
    out.append("| # | method | macro |")
    out.append("|---:|---|---:|")
    for i, (n, x) in enumerate(sorted(macro.items(), key=lambda t: -t[1]), 1):
        out.append(f"| {i} | {n} | {x:.4f} |")
    return macro


def pairwise(R, cfgs, a, b, out, label):
    out.append(f"\n**{label}** — paired over 5 seeds\n")
    out.append("| config | Δ | seeds |")
    out.append("|---|---:|---:|")
    ds = []
    for k in cfgs:
        x, y = au(R, k, a), au(R, k, b)
        if x is None or y is None: continue
        d = x - y; ds.append(float(d.mean()))
        out.append(f"| {k[0]}/{k[1]} | {d.mean():+.4f} | {int((d > 0).sum())}/5 |")
    if ds:
        out.append(f"\nmacro Δ = **{np.mean(ds):+.4f}**, positive in "
                   f"**{sum(1 for x in ds if x > 0)}/{len(ds)}** configurations")
    return ds


def main():
    R, M = load()
    out = ["# Reference-Anchored RPO — frozen-method judgement\n",
           "Primary metric: macro AUROC across single fixed methods. "
           "A per-cell best-baseline column is a test oracle and is not used.\n"]
    S = {}
    S["dev_macro"] = block(R, M, DEV, "Development configurations (EC50/IC50 x 2 backbones)", out)
    S["conf_macro"] = block(R, M, CONF, "Post-freeze confirmation (Ki-Assay x 2 backbones)", out)
    S["all_macro"] = block(R, M, DEV + CONF, "All configurations", out)

    out.append("\n\n## Decision rules\n")
    for a, b, lab in [("ref_rpo", "ref_bce", "Reference-Anchored RPO vs Reference-BCE"),
                      ("ref_rpo", "fusion", "Reference-Anchored RPO vs fusion control"),
                      ("ref_rpo", "ref_only", "Reference-Anchored RPO vs the reference"),
                      ("ref_rpo_ess", "ref_rpo", "reference weighting vs no weighting")]:
        S[f"{a}_vs_{b}_dev"] = pairwise(R, DEV, a, b, out, lab + " — development")
        S[f"{a}_vs_{b}_conf"] = pairwise(R, CONF, a, b, out, lab + " — Ki confirmation")

    if S["all_macro"]:
        top = max(S["all_macro"].items(), key=lambda t: t[1])
        S["best_single_method"] = top
        out.append(f"\n\n## Verdict\n\nBest single fixed method over all configurations: "
                   f"**{top[0]}** at macro {top[1]:.4f}.")
        rr = S["all_macro"].get("Reference-Anchored RPO")
        if rr is not None:
            gap = {n: rr - x for n, x in S["all_macro"].items() if n != "Reference-Anchored RPO"}
            out.append("\nReference-Anchored RPO minus each competitor (macro):\n")
            out.append("| competitor | Δ |")
            out.append("|---|---:|")
            for n, g in sorted(gap.items(), key=lambda t: -t[1]):
                out.append(f"| {n} | {g:+.4f} |")

    out.append("\n\n## Temperature sensitivity (never used for selection)\n")
    out.append("| arm | macro Δ vs reference | mean tau | mb-ESS median |")
    out.append("|---|---:|---:|---:|")
    for s in SENS:
        v, t, e = [], [], []
        for k in DEV + CONF:
            a = R.get(k, {}).get("arms", {}).get(s)
            if not a: continue
            v.append(a["delta_vs_ref"])
            t += [x["tau"] for x in a["diag"] if x["tau"] is not None]
            e += [x["mb_ess_median"] for x in a["diag"] if x["mb_ess_median"] is not None]
        if v:
            out.append(f"| {s} | {np.mean(v):+.4f} | {np.mean(t):.3f} | "
                       f"{np.mean(e) if e else float('nan'):.3f} |")
    uns = [f"{k[0]}/{k[1]}:{s}" for k in R for s in SENS
           if s in R[k]["arms"] and not all(x["tau_solved"] for x in R[k]["arms"][s]["diag"])]
    if uns:
        out.append(f"\n⚠️ tau had NO solution (pinned at the bisection ceiling) for: "
                   f"{', '.join(uns)}.  As tau grows, w -> 1[m_o < m_i], so the attainable "
                   f"ESS ratio floors at 1 - AUROC_ref; a target below that is unreachable.")

    Path("repro").mkdir(exist_ok=True)
    open("repro/refrpo_tables.md", "w").write("\n".join(out) + "\n")
    json.dump(S, open("repro/refrpo_summary.json", "w"), indent=2, default=float)
    print("\n".join(out))


if __name__ == "__main__":
    main()
