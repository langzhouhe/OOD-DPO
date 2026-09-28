#!/usr/bin/env python3
"""
Table 1 reproduction driver for RPO (Energy-DPO) + baselines, both backbones.

Runs main.py (RPO) and/or baselines.py across DrugOOD (EC50/IC50 x
assay/scaffold/size) and GOOD (HIV/PCBA/ZINC x scaffold/size), aggregates
AUROC/AUPR/FPR95 over seeds, and prints a table alongside the paper's
Table 1 reference values (table1_reference.json).

Examples:
  # validate two MiniMol cells vs Table 1
  python repro_table1.py --backbones minimol --cells ec50_scaffold ec50_size --methods rpo --seeds 1 --epochs 500
  # full RPO sweep, both backbones, 3 seeds
  python repro_table1.py --backbones minimol unimol --cells all --methods rpo --seeds 1 2 3 --epochs 500
"""
import argparse, json, os, subprocess, sys, statistics
from pathlib import Path

PY = "/root/miniconda3/envs/ood/bin/python"
ROOT = Path("/root/autodl-tmp/OOD-DPO")
REF = json.load(open(ROOT / "table1_reference.json"))

# Every Table 1 column. kind drugood -> subset lbap_general_<subset>; kind good -> good_<ds> + domain.
CELLS = {
    "ec50_scaffold": {"kind": "drugood", "ds": "ec50", "shift": "scaffold"},
    "ec50_size":     {"kind": "drugood", "ds": "ec50", "shift": "size"},
    "ec50_assay":    {"kind": "drugood", "ds": "ec50", "shift": "assay"},
    "ic50_scaffold": {"kind": "drugood", "ds": "ic50", "shift": "scaffold"},
    "ic50_size":     {"kind": "drugood", "ds": "ic50", "shift": "size"},
    "ic50_assay":    {"kind": "drugood", "ds": "ic50", "shift": "assay"},
    "hiv_scaffold":  {"kind": "good", "ds": "hiv", "shift": "scaffold"},
    "hiv_size":      {"kind": "good", "ds": "hiv", "shift": "size"},
    "pcba_scaffold": {"kind": "good", "ds": "pcba", "shift": "scaffold"},
    "pcba_size":     {"kind": "good", "ds": "pcba", "shift": "size"},
    "zinc_scaffold": {"kind": "good", "ds": "zinc", "shift": "scaffold"},
    "zinc_size":     {"kind": "good", "ds": "zinc", "shift": "size"},
}
BATCH = {"minimol": "512", "unimol": "256"}
ENV = {**os.environ, "HF_ENDPOINT": "https://hf-mirror.com", "TQDM_DISABLE": "1"}


def run(cmd, log):
    with open(log, "w") as f:
        p = subprocess.run(cmd, env=ENV, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def rpo_cell(backbone, cell, seed, epochs):
    c = CELLS[cell]
    outdir = ROOT / f"repro/{backbone}/{cell}/seed{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    common = ["--foundation_model", backbone, "--cache_root", str(ROOT / "cache"),
              "--seed", str(seed), "--data_seed", "42", "--output_dir", str(outdir),
              "--dpo_beta", "0.1", "--lambda_reg", "0.01", "--lr", "1e-4",
              "--batch_size", BATCH[backbone], "--eval_batch_size", "256", "--num_workers", "4"]
    if c["kind"] == "drugood":
        subset = f"lbap_general_{c['ds']}_{c['shift']}"
        dsargs = ["--dataset", subset, "--drugood_subset", subset,
                  "--data_file", str(ROOT / f"data/raw/{subset}.json")]
    else:
        dsargs = ["--dataset", f"good_{c['ds']}", "--good_domain", c["shift"],
                  "--good_shift", "covariate", "--data_path", str(ROOT / "data")]
    # train
    rc = run([PY, "main.py", "--mode", "train", "--epochs", str(epochs)] + dsargs + common,
             outdir / "train.log")
    if rc != 0:
        return None
    # eval
    run([PY, "main.py", "--mode", "eval", "--model_path", str(outdir / "best_model.pth")]
        + dsargs + common, outdir / "eval.log")
    res = outdir / "ood_evaluation_results.json"
    if res.exists():
        d = json.load(open(res))
        return {"auroc": d.get("auroc"), "aupr": d.get("aupr"), "fpr95": d.get("fpr95")}
    return None


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["minimol"], choices=["minimol", "unimol"])
    ap.add_argument("--cells", nargs="+", default=["all"])
    ap.add_argument("--methods", nargs="+", default=["rpo"])  # baselines added later
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--epochs", type=int, default=500)
    args = ap.parse_args()

    cells = list(CELLS) if args.cells == ["all"] else args.cells
    rows = []
    for backbone in args.backbones:
        for cell in cells:
            c = CELLS[cell]
            ref = REF[backbone].get(f"{c['ds']}|{c['shift']}")
            aurocs, auprs, fprs = [], [], []
            for seed in args.seeds:
                r = rpo_cell(backbone, cell, seed, args.epochs)
                print(f"[{backbone} {cell} seed{seed}] -> {r}", flush=True)
                if r:
                    aurocs.append(r["auroc"]); auprs.append(r["aupr"]); fprs.append(r["fpr95"])
            a = agg(aurocs)
            if a:
                diff = a[0] - ref if ref is not None else float("nan")
                rows.append((backbone, cell, a[0], a[1], agg(auprs)[0], agg(fprs)[0], ref, diff))

    print("\n" + "=" * 92)
    print(f"{'backbone':8} {'cell':15} {'AUROC':>7} {'±std':>6} {'AUPR':>7} {'FPR95':>7} {'paper':>7} {'Δ':>7}")
    print("-" * 92)
    for bb, cell, au, sd, ap_, fp, ref, diff in rows:
        rf = f"{ref:.3f}" if ref is not None else "  -  "
        df = f"{diff:+.3f}" if ref is not None else "  -  "
        print(f"{bb:8} {cell:15} {au:7.3f} {sd:6.3f} {ap_:7.3f} {fp:7.3f} {rf:>7} {df:>7}")
    print("=" * 92)
    json.dump([{"backbone": r[0], "cell": r[1], "auroc": r[2], "auroc_std": r[3],
                "aupr": r[4], "fpr95": r[5], "paper_auroc": r[6], "delta": r[7]} for r in rows],
              open(ROOT / "repro/summary.json", "w"), indent=2)
    print("Summary -> repro/summary.json")


if __name__ == "__main__":
    main()
