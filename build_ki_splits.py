#!/usr/bin/env python3
"""Build the seed-42 split cache for lbap_general_ki_assay, byte-identical in construction
to the twelve cells that already have one.

Ki-Assay has cached MiniMol and Uni-Mol features but no `*_seed42_splits.json`, so
matched_oe / reference_rpo cannot load it.  Rather than invent a split, this reproduces the
exact pipeline in `utils.process_drugood_data` + `data_loader._prepare_splits`:

  train_id  <- split['train'],    val_id <- split['iid_val'],  test_id  <- split['iid_test']
  train_ood <- split['ood_val'],  test_ood <- split['ood_test']
  val_ood is absent for DrugOOD, so it is carved out of train_ood with
      random.Random(42).shuffle(train_ood); val_size = min(3000, len//5)
  then each split is subsampled to its target size with
      np.random.default_rng(abs(42 + offset) % 2**31).choice(raw, size, replace=False)
  offsets: train_id 0, train_ood 1, val_id 2, val_ood 3, test_id 4, test_ood 5

`--verify <cell>` regenerates a cell that already has a cache and diffs it, which is the
only acceptable evidence that the reconstruction matches.

Usage: python build_ki_splits.py --verify ec50_assay && python build_ki_splits.py --build
"""
import json, random, argparse, warnings, time
from pathlib import Path
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

from utils import process_drugood_data

CACHE = Path("cache/ood_dpo_cache")
SIZES = {"train_id": 2000, "train_ood": 2000, "val_id": 600, "val_ood": 600,
         "test_id": 1000, "test_ood": 1000}
OFFSETS = {"train_id": 0, "train_ood": 1, "val_id": 2, "val_ood": 3,
           "test_id": 4, "test_ood": 5}
SEED = 42


def build(subset):
    f = f"data/raw/{subset}.json"
    d = process_drugood_data(f, None)
    raw = {"train_id": d["train_id_smiles"], "train_ood": d["train_ood_smiles"],
           "val_id": d["val_id_smiles"], "val_ood": [],
           "test_id": d["test_id_smiles"], "test_ood": d["test_ood_smiles"]}
    if not raw["val_ood"]:                       # DrugOOD never ships one
        tr = list(raw["train_ood"])
        random.Random(SEED).shuffle(tr)
        n = min(3000, len(tr) // 5)
        raw["val_ood"], raw["train_ood"] = tr[:n], tr[n:]
    out = {}
    for k in SIZES:
        n = min(SIZES[k], len(raw[k]))
        rng = np.random.default_rng(abs(SEED + OFFSETS[k]) % (2 ** 31))
        out[k] = list(rng.choice(raw[k], size=n, replace=False))
    return out, {"dataset_name": subset, "data_seed": SEED, "max_samples": None,
                 "drugood_subset": subset, "data_file_mtime": Path(f).stat().st_mtime}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", default=None)
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()

    if a.verify:
        sub = f"lbap_general_{a.verify}"
        got, _ = build(sub)
        ref = json.load(open(CACHE / f"{sub}_seed42_splits.json"))["splits"]
        ok = True
        for k in SIZES:
            same = list(got[k]) == list(ref[k])
            ok &= same
            print(f"  {k:10} regenerated {len(got[k]):>5} vs cached {len(ref[k]):>5}  "
                  f"identical={'YES' if same else 'NO'}"
                  + ("" if same else f"  overlap={len(set(got[k]) & set(ref[k]))}"))
        print(f"VERIFY {'PASS' if ok else 'FAIL'} for {a.verify}")
        return

    if a.build:
        sub = "lbap_general_ki_assay"
        sp, meta = build(sub)
        p = CACHE / f"{sub}_seed42_splits.json"
        json.dump({"metadata": meta, "target_sizes": SIZES, "splits": sp,
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(p, "w"))
        print(f"wrote {p}")
        for k in SIZES: print(f"  {k:10} {len(sp[k])}")
        dom = {it["smiles"]: it.get("domain_id")
               for it in json.load(open(f"data/raw/{sub}.json"))["split"]["ood_val"]}
        print(f"  train_ood/val_ood domains available for the disjoint re-split: "
              f"{len({dom[s] for s in sp['train_ood'] if s in dom})} / "
              f"{len({dom[s] for s in sp['val_ood'] if s in dom})}")


if __name__ == "__main__":
    main()
