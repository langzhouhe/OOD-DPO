#!/usr/bin/env python3
"""Encode every molecule used by the MePOE catalogs with Uni-Mol, so the batch-level
detector-selection experiment can be repeated on a second backbone.

Reuses the size-sorted dynamic batching and OOM-halving from ablate_unimol.py: the
Gaussian basis materialises a (B, N, N, K=128) tensor, so a single large molecule at a
fixed batch of 256 needs tens of GB.

Merges into the existing {base}_unimol_features.pkl (creating it when the endpoint is
new, e.g. Ki), so nothing already encoded is recomputed.

Usage: CUDA_VISIBLE_DEVICES=0 python precompute_unimol_mepoe.py ec50_assay ki_assay
"""
import os, json, pickle, sys, time, warnings, logging, argparse
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import concurrent.futures as cf
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from utils import smiles2graph

CACHE = Path("cache/ood_dpo_cache"); MP = Path("cache/mepoe")
BASE = {"ec50_assay": "lbap_general_ec50_assay", "ic50_assay": "lbap_general_ic50_assay",
        "ki_assay": "lbap_general_ki_assay",
        "ec50_scaffold": "lbap_general_ec50_scaffold",
        "ic50_scaffold": "lbap_general_ic50_scaffold"}
SEEDS = (21, 22, 23)
WORKERS = int(os.environ.get("GRAPH_WORKERS", "32"))
BATCH = int(os.environ.get("UNIMOL_BATCH", "256"))
BUDGET = int(os.environ.get("UNIMOL_BUDGET", "600000"))


def build_graph(s):
    try:
        g = smiles2graph(s)
        if isinstance(g, dict) and "mol" in g:
            g = {k: v for k, v in g.items() if k != "mol"}
        return s, g
    except Exception:
        return s, None


def make_batches(smis, graphs, budget, cap=256):
    def natoms(s):
        g = graphs.get(s)
        return (len(g["atoms"]) if isinstance(g, dict) and "atoms" in g else 128) + 2
    out, cur, nmax = [], [], 0
    for i in sorted(range(len(smis)), key=lambda j: natoms(smis[j])):
        n = max(nmax, natoms(smis[i]))
        if cur and ((len(cur) + 1) * n * n > budget or len(cur) >= cap):
            out.append(cur); cur, nmax = [i], natoms(smis[i])
        else:
            cur.append(i); nmax = n
    if cur: out.append(cur)
    return out


def encode_batch(enc, items, keys, feats, depth=0):
    import torch
    try:
        for k, f in zip(keys, enc.encode_smiles(items)):
            feats[k] = f.detach().cpu().numpy().astype(np.float32)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "out of memory" not in str(e).lower() or len(items) == 1 or depth > 6:
            raise
        torch.cuda.empty_cache()
        h = len(items) // 2
        encode_batch(enc, items[:h], keys[:h], feats, depth + 1)
        encode_batch(enc, items[h:], keys[h:], feats, depth + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+")
    a = ap.parse_args()
    import torch
    from model import UniMolEncoder
    enc = UniMolEncoder()
    print(f"UniMol on {next(enc.model.parameters()).device}", flush=True)

    for cell in a.cells:
        b = BASE[cell]
        want = set()
        for s in SEEDS:
            f = MP / f"{b}_mepoe_seed{s}.json"
            if not f.exists(): continue
            r = json.load(open(f))
            want |= {x for n in r["catalog"] for ss in r["catalog"][n].values() for x in ss}
            want |= set(r["id_train"]) | set(r["id_test"])
        fpath = CACHE / f"{b}_unimol_features.pkl"
        if fpath.exists():
            blob = pickle.load(open(fpath, "rb")); feats = blob["features"]
        else:
            blob = {"features": {}, "foundation_model": "unimol",
                    "dataset_name": b, "drugood_subset": b}
            feats = blob["features"]
        need = sorted(want - set(feats))
        print(f"[{cell}] {len(want)} wanted, {len(need)} to encode", flush=True)
        if not need:
            print(f"UNIMOL_MEPOE_DONE {cell}", flush=True); continue

        t0 = time.time(); graphs = {}
        with cf.ProcessPoolExecutor(max_workers=WORKERS) as ex:
            for s, g in ex.map(build_graph, need, chunksize=16):
                if g is not None: graphs[s] = g
        inputs = [graphs.get(s, s) for s in need]
        batches = make_batches(need, graphs, BUDGET, BATCH)
        print(f"[{cell}] 3D built in {time.time()-t0:.0f}s, {len(batches)} batches", flush=True)

        t1 = time.time(); done = 0
        torch.cuda.empty_cache()
        for bi, bidx in enumerate(batches):
            try:
                encode_batch(enc, [inputs[i] for i in bidx], [need[i] for i in bidx], feats)
                done += len(bidx)
            except Exception as e:
                print(f"  [{cell}] batch {bi} failed: {type(e).__name__}", flush=True)
            if (bi + 1) % 25 == 0:
                print(f"  [{cell}] {done}/{len(need)} ({time.time()-t1:.0f}s)", flush=True)
        blob["features"] = feats
        pickle.dump(blob, open(fpath, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[{cell}] encoded {done}/{len(need)} in {time.time()-t1:.0f}s -> {fpath}", flush=True)
        print(f"UNIMOL_MEPOE_DONE {cell}", flush=True)
    print("ALL_UNIMOL_DONE", flush=True)


if __name__ == "__main__":
    main()
