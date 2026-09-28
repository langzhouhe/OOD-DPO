#!/usr/bin/env python3
"""Extract a SMILES -> class-label map for the three GOOD datasets.

The logit-based trio (MSP / ODIN / Energy) and their outlier-exposure counterparts all
need a property classifier trained on ID molecules.  DrugOOD ships `cls_label` in its
official json; GOOD does not expose one through this repo's loader, which is why the six
GOOD cells had no MSP/ODIN/Energy column in every previous table.  The labels do exist
inside the processed PyG files, so we lift them out once and cache them.

Per dataset:
  HIV   y is a single binary activity column -> used as-is.
  PCBA  y is 128 assay columns that are mostly NaN -> we keep the single column with the
        most labelled molecules, so the classifier is trained on a real assay rather than
        an imputed one.  The chosen task index is recorded in the cache.
  ZINC  y is continuous (penalised logP) -> binarised at the median over the molecules
        that actually carry a label.  This is an adaptation, not a GOOD-defined task, and
        must be described as such in the paper.

Usage: python good_labels.py
"""
import json, warnings
from pathlib import Path
import numpy as np
import torch

warnings.filterwarnings("ignore")

CACHE = Path("cache")
SPLITS = ["covariate_train", "covariate_id_val", "covariate_id_test",
          "covariate_val", "covariate_test", "no_shift_train"]
DSETS = {"GOODHIV": "hiv", "GOODPCBA": "pcba", "GOODZINC": "zinc"}


def collect(ds):
    """SMILES -> raw target, pooled over every split file we can read."""
    smi, ys = [], []
    for domain in ("scaffold", "size"):
        for sp in SPLITS:
            f = Path(f"data/{ds}/{domain}/processed/{sp}.pt")
            if not f.exists():
                continue
            try:
                obj = torch.load(f, weights_only=False)
            except Exception as e:
                print(f"  skip {f}: {type(e).__name__}", flush=True)
                continue
            data = obj[0] if isinstance(obj, (list, tuple)) else obj
            s = getattr(data, "smiles", None)
            y = getattr(data, "y", None)
            if s is None or y is None:
                continue
            y = np.asarray(y.numpy(), dtype=np.float64)
            if y.ndim == 1:
                y = y[:, None]
            n = min(len(s), len(y))
            smi += list(s[:n]); ys.append(y[:n])
            print(f"  {domain}/{sp}: {n} molecules, y{y.shape[1:]}", flush=True)
    if not ys:
        return None, None
    return smi, np.concatenate(ys, 0)


def main():
    CACHE.mkdir(exist_ok=True)
    for ds, tag in DSETS.items():
        print(f"[{ds}]", flush=True)
        smi, y = collect(ds)
        if smi is None:
            print(f"  no labels found -> {tag} trio stays unavailable", flush=True)
            continue
        meta = {"dataset": ds}
        if y.shape[1] > 1:
            # PCBA: pick the column with the largest MINORITY class, not the one with the
            # most labels -- the densest column is 411796/71, which trains a degenerate
            # classifier and would make MSP/Energy constant.
            pos = np.nansum(y == 1, 0); neg = np.nansum(y == 0, 0)
            t = int(np.argmax(np.minimum(pos, neg)))
            col = y[:, t]
            meta.update(task=t, n_labelled=int(pos[t] + neg[t]),
                        rule="PCBA assay column with the largest minority class")
        else:
            col = y[:, 0]
            meta.update(task=0, n_labelled=int((~np.isnan(col)).sum()))
        ok = ~np.isnan(col)
        vals = np.unique(col[ok])
        if len(vals) > 10:                       # ZINC: continuous target
            thr = float(np.median(col[ok]))
            lab = np.where(col > thr, 1, 0)
            meta.update(rule=f"binarised at median {thr:.4f} (continuous target)",
                        threshold=thr)
        else:
            lab = col.astype(int, copy=False)
        m = {}
        for s, l, o in zip(smi, lab, ok):
            if o:
                m.setdefault(s, int(l))
        meta.update(n_unique_smiles=len(m),
                    class_counts={str(k): int(v) for k, v in
                                  zip(*np.unique(list(m.values()), return_counts=True))})
        json.dump({"meta": meta, "labels": m},
                  open(CACHE / f"good_labels_{tag}.json", "w"))
        print(f"  -> cache/good_labels_{tag}.json  {meta}", flush=True)


if __name__ == "__main__":
    main()
