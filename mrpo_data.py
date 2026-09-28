#!/usr/bin/env python3
"""Rebuild mechanism-stratified auxiliary OOD for the MRPO kill test.

The existing cache subsamples 2000 OOD molecules at random across ~1200 assay domains,
leaving ~1.7 molecules per assay. A per-mechanism risk is meaningless at that granularity
-- CVaR over singleton groups degenerates into hard-example mining, which arena.py
already showed is matched by pointwise `hard_bce`. So the groups have to be rebuilt.

Design (per cell):
  train_ood  N_TR domains x M molecules   drawn from official ood_val
  val_ood    N_VA domains x M molecules   drawn from official ood_val, DOMAIN-DISJOINT
  test_ood   N_TE domains x M molecules   drawn from official ood_test
The benchmark already guarantees ood_val and ood_test share NO domain_id (verified: 0
overlap on both assay cells), so the test mechanisms are unseen by construction.

ID molecules are taken unchanged from the existing cache, so the only thing that differs
from the Phase-0 setup is the mechanism structure of the auxiliary OOD.

Usage: python mrpo_data.py --cell ec50_assay --split_seed 7
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json, pickle, argparse, warnings, logging, collections
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

CACHE = Path("cache/ood_dpo_cache")
OUT = Path("cache/mrpo")
BASE = {"ec50_assay": "lbap_general_ec50_assay", "ic50_assay": "lbap_general_ic50_assay"}
M = 8            # molecules kept per mechanism
N_TR, N_VA, N_TE = 250, 75, 125     # domains -> 2000 / 600 / 1000 OOD molecules


def stratify(items, rng, n_dom, exclude=frozenset()):
    by = collections.defaultdict(list)
    for it in items:
        s, d = it.get("smiles"), it.get("domain_id")
        if s and d is not None and d not in exclude:
            by[d].append(s)
    ok = sorted([d for d, v in by.items() if len(set(v)) >= M])
    rng.shuffle(ok)
    take = ok[:n_dom]
    out = {}
    for d in take:
        u = sorted(set(by[d]))
        idx = rng.choice(len(u), size=M, replace=False)
        out[d] = [u[i] for i in idx]
    return out, set(take)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--split_seed", type=int, default=7)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    b = BASE[a.cell]
    raw = json.load(open(f"data/raw/{b}.json"))["split"]
    rng = np.random.default_rng(a.split_seed)

    tr, dtr = stratify(raw["ood_val"], rng, N_TR)
    va, dva = stratify(raw["ood_val"], rng, N_VA, exclude=dtr)
    te, dte = stratify(raw["ood_test"], rng, N_TE)
    assert not (dtr & dva) and not (dtr & dte) and not (dva & dte), "domain leakage"

    old = json.load(open(CACHE / f"{b}_seed42_splits.json"))["splits"]
    groups = {"train_ood": tr, "val_ood": va, "test_ood": te}
    splits = {k: old[k] for k in ("train_id", "val_id", "test_id")}
    for k, g in groups.items():
        splits[k] = [s for d in sorted(g) for s in g[d]]
    dom = {k: [d for d in sorted(g) for _ in g[d]] for k, g in groups.items()}

    rec = {"splits": splits, "domains": dom, "split_seed": a.split_seed, "M": M,
           "n_domains": {k: len(g) for k, g in groups.items()},
           "domain_overlap": {"train_val": len(dtr & dva), "train_test": len(dtr & dte),
                              "val_test": len(dva & dte)}}
    json.dump(rec, open(OUT / f"{b}_mrpo_seed{a.split_seed}_splits.json", "w"))
    print(f"[{a.cell}] domains tr/va/te = {len(dtr)}/{len(dva)}/{len(dte)} | "
          f"mols {len(splits['train_ood'])}/{len(splits['val_ood'])}/{len(splits['test_ood'])} "
          f"| overlaps {rec['domain_overlap']}", flush=True)

    # encode whatever MiniMol features are missing
    fpath = CACHE / f"{b}_minimol_features.pkl"
    blob = pickle.load(open(fpath, "rb")); feats = blob["features"]
    need = sorted({s for k in groups for s in splits[k]} - set(feats))
    print(f"[{a.cell}] {len(need)} molecules need MiniMol encoding", flush=True)
    if need:
        from model import MinimolEncoder, MinimolEncodingError
        enc = MinimolEncoder()
        B, done = 256, 0
        for i in range(0, len(need), B):
            chunk = need[i:i + B]
            try:
                out = enc.encode_smiles(chunk)
                for s, f in zip(chunk, out):
                    feats[s] = np.asarray(f, dtype=np.float32)
                done += len(chunk)
            except Exception:
                for s in chunk:                      # fall back one at a time
                    try:
                        feats[s] = np.asarray(enc.encode_smiles([s])[0], dtype=np.float32); done += 1
                    except Exception:
                        pass
            print(f"  [{a.cell}] encoded {done}/{len(need)}", flush=True)
        blob["features"] = feats
        pickle.dump(blob, open(fpath, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print(f"MRPO_DATA_DONE {a.cell}", flush=True)


if __name__ == "__main__":
    main()
