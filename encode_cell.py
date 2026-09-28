#!/usr/bin/env python3
"""Encode a cell's seed-42 split molecules with MiniMol or Uni-Mol.

Needed for Ki-Assay: its feature caches were built for the MolRoute mechanism substrate,
so they cover only ~27% of the molecules in the standard seed-42 splits (train_id 73/2000).
Everything here reuses the existing encoders and, for Uni-Mol, molroute_unimol_enc.py's
size-sorted dynamic batching and OOM-halving -- the Gaussian basis materialises a
(B, N, N, 128) tensor, so a fixed batch of 256 can need tens of GB on one large molecule.

  MiniMol : conda env `ood`   -> python encode_cell.py --cell ki_assay --backbone minimol
  Uni-Mol : conda env `umgpu` -> CUDA_VISIBLE_DEVICES=0 python encode_cell.py \
                                     --cell ki_assay --backbone unimol
"""
import os, json, pickle, argparse, time, warnings, logging
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

CACHE = Path("cache/ood_dpo_cache")
CMRPO = Path("cache/cmrpo")
BASE = {"ki_assay": "lbap_general_ki_assay", "ec50_assay": "lbap_general_ec50_assay",
        "ic50_assay": "lbap_general_ic50_assay"}
KEYS = ["id_train", "ood_train", "val_id", "val_ood", "test_id", "test_ood"]


def wanted(cell):
    """Molecules to encode.  `--cell cmrpo:<target>` reads the Context-Matched substrate,
    whose auxiliary molecules come from OTHER endpoints' train pools and are therefore
    absent from every per-cell feature cache."""
    if cell.startswith("cmrpo:"):
        rec = json.load(open(CMRPO / f"{cell.split(':', 1)[1]}_cmrpo.json"))
        sp = {k: rec[k] for k in KEYS}
        return sorted({s for v in sp.values() for s in v}), sp
    sp = json.load(open(CACHE / f"{BASE[cell]}_seed42_splits.json"))["splits"]
    return sorted({s for v in sp.values() for s in v}), sp


def report(cell, sp, feats, tag):
    for k, v in sp.items():
        print(f"  {k:10} {sum(1 for s in v if s in feats):>5}/{len(v):<5} covered {tag}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True,
                    help="a cell name, or cmrpo:<target> for the Context-Matched substrate")
    ap.add_argument("--backbone", required=True, choices=["minimol", "unimol"])
    a = ap.parse_args()
    cm = a.cell.startswith("cmrpo:")
    b = a.cell.split(":", 1)[1] if cm else BASE[a.cell]
    want, sp = wanted(a.cell)
    fp = (CMRPO / f"{b}_{a.backbone}_features.pkl") if cm else \
         (CACHE / f"{b}_{a.backbone}_features.pkl")
    if cm:   # seed the cache with anything already encoded for the three endpoints
        pre = {}
        for t in ("ec50", "ic50", "ki"):
            g = CACHE / f"lbap_general_{t}_assay_{a.backbone}_features.pkl"
            if g.exists(): pre.update(pickle.load(open(g, "rb"))["features"])
        if not fp.exists():
            fp.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump({"features": {s: pre[s] for s in want if s in pre},
                         "foundation_model": a.backbone, "dataset_name": b},
                        open(fp, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    blob = pickle.load(open(fp, "rb")) if fp.exists() else {
        "features": {}, "foundation_model": a.backbone, "dataset_name": b,
        "drugood_subset": b}
    feats = blob["features"]
    need = [s for s in want if s not in feats]
    print(f"[{a.cell}/{a.backbone}] {len(want)} split molecules, {len(need)} to encode",
          flush=True)
    report(a.cell, sp, feats, "before")
    if not need:
        print(f"ENC_DONE {a.cell} {a.backbone}", flush=True); return

    t0 = time.time()
    if a.backbone == "minimol":
        from model import MinimolEncoder
        enc = MinimolEncoder(); done = 0
        for i in range(0, len(need), 256):
            ch = need[i:i + 256]
            try:
                for s, f in zip(ch, enc.encode_smiles(ch)):
                    feats[s] = np.asarray(f, dtype=np.float32); done += 1
            except Exception:
                for s in ch:                       # fall back to one-at-a-time
                    try:
                        feats[s] = np.asarray(enc.encode_smiles([s])[0], dtype=np.float32)
                        done += 1
                    except Exception:
                        pass
            print(f"  encoded {done}/{len(need)} ({time.time()-t0:.0f}s)", flush=True)
    else:
        import torch, concurrent.futures as cf
        import molroute_unimol_enc as UM
        from model import UniMolEncoder
        enc = UniMolEncoder()
        print(f"  UniMol on {next(enc.model.parameters()).device}", flush=True)
        graphs = {}
        with cf.ProcessPoolExecutor(max_workers=UM.WORKERS) as ex:
            for s, g in ex.map(UM.build_graph, need, chunksize=16):
                if g is not None: graphs[s] = g
        inputs = [graphs.get(s, s) for s in need]
        batches = UM.make_batches(need, graphs, UM.BUDGET, UM.BATCH)
        print(f"  3D built in {time.time()-t0:.0f}s, {len(batches)} batches", flush=True)
        done = 0; torch.cuda.empty_cache()
        for bi, bidx in enumerate(batches):
            try:
                UM.encode_batch(enc, [inputs[i] for i in bidx], [need[i] for i in bidx], feats)
                done += len(bidx)
            except Exception as e:
                print(f"  batch {bi} failed: {type(e).__name__}", flush=True)
            if bi % 10 == 0:
                print(f"  encoded {done}/{len(need)} ({time.time()-t0:.0f}s)", flush=True)

    blob["features"] = feats
    pickle.dump(blob, open(fp, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    report(a.cell, sp, feats, "after")
    print(f"ENC_DONE {a.cell} {a.backbone} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
