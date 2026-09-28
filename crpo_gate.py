#!/usr/bin/env python3
"""Gate decision for the C-RPO coupling experiment.

Pre-registered criterion (set before the run):
  C-RPO must beat the Balanced OOD Head by >= 0.010 macro AUROC on BOTH assay
  cells, with the paired per-seed difference the same sign on all 5 seeds.
  Pass -> extend to Uni-Mol and the scaffold cells.
  Fail -> stop; do NOT try to rescue it with LoRA / fusion / more auxiliary OOD.

Arms share seeds FIN=[11..15]; a given seed fixes the molecule subset AND the head
init, so per-seed differences are properly paired.

The two falsification contrasts matter as much as the headline:
  ot_fixed - ot_shuf   chemistry vs an identically-concentrated meaningless coupling
  ot_fixed - rpo_prod  any non-product coupling vs the uniform product coupling
If ot_fixed ~= ot_shuf the gain is not chemical and the coupling hypothesis is dead
regardless of what the headline delta says.
"""
import json, glob, os
import numpy as np

GATE_DELTA = 0.010
ARMS = ["bce", "rpo", "rpo_prod", "ot_fixed", "ot_adv", "ot_shuf"]
CELLS = ["ec50_assay", "ic50_assay"]
VIEWS = ["minimol", "minimol_mv"]


def ci95(d):
    d = np.asarray(d, float)
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))


def load_cell_view(cell, view):
    for p in (f"repro/crpo_{cell}_{view}.json", f"repro/crpo_{cell}.json"):
        if os.path.exists(p):
            d = json.load(open(p))
            if any(k.startswith(f"{view}|") for k in d):
                return d
    return None


def main():
    summary = {}
    for view in VIEWS:
        print("=" * 100)
        print(f"VIEW: {view}")
        print("=" * 100)
        for cell in CELLS:
            d = load_cell_view(cell, view)
            if d is None:
                print(f"  {cell}: NO RESULTS"); continue
            base = d.get(f"{view}|bce")
            if base is None:
                print(f"  {cell}: no BOH baseline"); continue
            b_au = np.array(base["auroc"]); b_fp = np.array(base["fpr95"])
            print(f"\n  [{cell}]  BOH AUROC={b_au.mean():.4f}±{b_au.std():.4f}  "
                  f"FPR95={b_fp.mean():.4f}")
            print(f"  {'arm':10} {'AUROC':>15} {'Δ vs BOH':>10} {'paired 95%CI':>18} "
                  f"{'signs':>7} {'FPR95':>8} {'ESS':>7} {'cost':>7}")
            for arm in ARMS:
                r = d.get(f"{view}|{arm}")
                if r is None: continue
                au = np.array(r["auroc"]); fp = np.array(r["fpr95"])
                diff = au - b_au
                m, h = ci95(diff) if arm != "bce" else (0.0, 0.0)
                npos = int((diff > 0).sum())
                hp = r["hp"]
                e = f"{hp['ess_frac']:.4f}" if hp.get("ess_frac") is not None else "-"
                c = f"{hp['mean_pair_cost']:.3f}" if hp.get("mean_pair_cost") is not None else "-"
                flag = ""
                if arm != "bce":
                    flag = "  <<<" if (m >= GATE_DELTA and npos == len(diff)) else ""
                print(f"  {arm:10} {au.mean():.4f}±{au.std():.4f} {m:+10.4f} "
                      f"  [{m-h:+.4f},{m+h:+.4f}] {npos}/{len(diff):>3} "
                      f"{fp.mean():8.4f} {e:>7} {c:>7}{flag}")
                summary[(view, cell, arm)] = dict(mean=au.mean(), diff=m, ci=h,
                                                  npos=npos, n=len(diff), fpr95=fp.mean())
            # falsification contrasts
            for a, b in [("ot_fixed", "ot_shuf"), ("ot_fixed", "rpo_prod"),
                         ("ot_adv", "ot_shuf")]:
                ra, rb = d.get(f"{view}|{a}"), d.get(f"{view}|{b}")
                if ra and rb:
                    df = np.array(ra["auroc"]) - np.array(rb["auroc"])
                    m, h = ci95(df)
                    print(f"    contrast {a:9} - {b:9} = {m:+.4f} [{m-h:+.4f},{m+h:+.4f}]"
                          f"  {int((df>0).sum())}/{len(df)} positive")

    # ---- gate ----
    print("\n" + "=" * 100)
    print(f"GATE: C-RPO >= +{GATE_DELTA:.3f} AUROC vs BOH on BOTH assay cells, 5/5 seeds same sign")
    print("=" * 100)
    for view in VIEWS:
        for arm in ("ot_fixed", "ot_adv"):
            rows = [summary.get((view, c, arm)) for c in CELLS]
            if any(r is None for r in rows):
                print(f"  {view:11} {arm:9} INCOMPLETE"); continue
            ok = all(r["diff"] >= GATE_DELTA and r["npos"] == r["n"] for r in rows)
            detail = "  ".join(f"{c}={r['diff']:+.4f}({r['npos']}/{r['n']})"
                               for c, r in zip(CELLS, rows))
            print(f"  {view:11} {arm:9} {'PASS' if ok else 'FAIL'}   {detail}")
    print("\nReminder: a PASS still requires ot_fixed - ot_shuf to be clearly positive,")
    print("otherwise the gain is not chemical and the coupling hypothesis is not supported.")


if __name__ == "__main__":
    main()
