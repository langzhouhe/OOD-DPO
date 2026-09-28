#!/usr/bin/env python3
"""
Precompute Uni-Mol 512-dim features on GPU for all Table-1 cells.

3D-conformer building (RDKit, CPU) is parallelized across many workers; the
Uni-Mol transformer forward runs on GPU. Writes feature caches in the exact
format data_loader expects, so subsequent `main.py --foundation_model unimol`
runs load them and skip encoding entirely.

Usage:
  CUDA_VISIBLE_DEVICES=0 python precompute_unimol.py <cell> [<cell> ...]
  cells: ec50_scaffold ec50_size ec50_assay ic50_scaffold ic50_size ic50_assay
         hiv_scaffold hiv_size pcba_scaffold pcba_size zinc_scaffold zinc_size
"""
import json, pickle, sys, time, warnings, logging, os
import numpy as np
import concurrent.futures as cf
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from utils import smiles2graph

CACHE = Path("cache/ood_dpo_cache")
WORKERS = int(os.environ.get("GRAPH_WORKERS", "48"))
BATCH = int(os.environ.get("UNIMOL_BATCH", "256"))

# cell -> (splits_cache, feature_cache, dataset_name, drugood_subset)
CELLS = {}
for d in ["ec50_scaffold", "ec50_size", "ec50_assay", "ic50_scaffold", "ic50_size", "ic50_assay"]:
    sub = f"lbap_general_{d}"
    CELLS[d] = (f"{sub}_seed42_splits.json", f"{sub}_unimol_features.pkl", sub, sub)
for ds in ["hiv", "pcba", "zinc"]:
    for dom in ["scaffold", "size"]:
        name = f"good_{ds}"
        CELLS[f"{ds}_{dom}"] = (f"{name}_{dom}_covariate_seed42_splits.json",
                                f"{name}_{dom}_covariate_unimol_features.pkl", name, "")


def build_graph(s):
    try:
        g = smiles2graph(s)
        if isinstance(g, dict) and "mol" in g:
            g = {k: v for k, v in g.items() if k != "mol"}
        return s, g
    except Exception:
        return s, None


def main():
    cells = sys.argv[1:] or list(CELLS)
    from model import UniMolEncoder
    enc = UniMolEncoder()
    import torch
    dev = next(enc.model.parameters()).device
    print(f"UniMol encoder on device: {dev}", flush=True)

    for key in cells:
        scache, fcache, dsname, sub = CELLS[key]
        fpath = CACHE / fcache
        if fpath.exists():
            print(f"[{key}] feature cache exists -> skip", flush=True); continue
        sp = json.load(open(CACHE / scache))["splits"]
        smis = sorted(set(s for v in sp.values() for s in v))
        t0 = time.time()
        # parallel 3D graph build
        graphs = {}
        with cf.ProcessPoolExecutor(max_workers=WORKERS) as ex:
            for s, g in ex.map(build_graph, smis, chunksize=16):
                if g is not None:
                    graphs[s] = g
        t_g = time.time() - t0
        # GPU forward; fall back to raw SMILES (inline build) for any failed graph
        inputs = [graphs.get(s, s) for s in smis]
        feats = {}
        t1 = time.time()
        for i in range(0, len(inputs), BATCH):
            chunk_smi = smis[i:i + BATCH]
            out = enc.encode_smiles(inputs[i:i + BATCH])
            for s, f in zip(chunk_smi, out):
                feats[s] = f.detach().cpu().numpy().astype(np.float32)
        t_f = time.time() - t1
        pickle.dump({"features": feats, "foundation_model": "unimol",
                     "dataset_name": dsname, "drugood_subset": sub},
                    open(fpath, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[{key}] {len(feats)}/{len(smis)} feats | 3D {t_g:.0f}s fwd {t_f:.0f}s -> {fcache}", flush=True)
    print("PRECOMPUTE_DONE " + ",".join(cells), flush=True)


if __name__ == "__main__":
    main()
