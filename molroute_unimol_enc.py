#!/usr/bin/env python3
"""Encode the official ood_test molecules used by MolRoute with Uni-Mol.

Same molecules as the MiniMol run -- the catalog file is read, not regenerated, so the
400 mechanisms and their 8 molecules per cell are identical across backbones.

Reuses ablate_unimol.py's size-sorted dynamic batching and OOM halving: the Gaussian
basis materialises a (B, N, N, 128) tensor, so one large molecule at a fixed batch of 256
needs tens of GB.

Usage: CUDA_VISIBLE_DEVICES=0 python molroute_unimol_enc.py ec50_assay ki_assay
"""
import os, json, pickle, sys, time, warnings, logging, argparse
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import concurrent.futures as cf
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from utils import smiles2graph

CACHE = Path("cache/ood_dpo_cache"); MR = Path("cache/molroute")
BASE = {"ec50_assay": "lbap_general_ec50_assay", "ic50_assay": "lbap_general_ic50_assay",
        "ki_assay": "lbap_general_ki_assay",
        "ec50_scaffold": "lbap_general_ec50_scaffold",
        "ic50_scaffold": "lbap_general_ic50_scaffold"}
WORKERS = int(os.environ.get("GRAPH_WORKERS", "24"))
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
    ap = argparse.ArgumentParser(); ap.add_argument("cells", nargs="+")
    a = ap.parse_args()
    import torch
    from model import UniMolEncoder
    enc = UniMolEncoder()
    print(f"UniMol on {next(enc.model.parameters()).device}", flush=True)

    for cell in a.cells:
        b = BASE[cell]
        rec = json.load(open(MR / f"{b}_molroute_test.json"))
        want = {s for ss in rec["catalog"].values() for s in ss}
        fp = CACHE / f"{b}_unimol_features.pkl"
        blob = pickle.load(open(fp, "rb")) if fp.exists() else {
            "features": {}, "foundation_model": "unimol", "dataset_name": b, "drugood_subset": b}
        feats = blob["features"]
        need = sorted(want - set(feats))
        print(f"[{cell}] {len(want)} ood_test molecules, {len(need)} to encode", flush=True)
        if not need:
            print(f"UMENC_DONE {cell}", flush=True); continue
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
        blob["features"] = feats
        pickle.dump(blob, open(fp, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        cov = len(want & set(feats)) / len(want)
        print(f"[{cell}] encoded {done}/{len(need)} in {time.time()-t1:.0f}s | "
              f"Uni-Mol coverage of ood_test = {cov*100:.2f}%", flush=True)
        print(f"UMENC_DONE {cell}", flush=True)
    print("ALL_UMENC_DONE", flush=True)


if __name__ == "__main__":
    main()
