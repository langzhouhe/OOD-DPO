#!/usr/bin/env python3
"""Encoder-information ablation for Uni-Mol: does the detector actually read the
pretrained representation, or would an untrained encoder do just as well?

This is the experiment E15 is often *misread* as providing. E15 compared the FIXED
reproduction against the PAPER's reported numbers (0.8103 vs 0.8080) -- that is a
reproduction-agreement delta, not an ablation. The pre-fix features were overwritten,
so no pre/post comparison exists. This script creates the missing arms.

What the [MASK] bug actually did (weights/dict.txt is one token short -> 30 vs 31):
  embed_tokens.weight  (31,512) shape-mismatched -> stayed at init_bert_params,
                       i.e. N(0, 0.02) random ATOM-TYPE embeddings.
  gbf.mul/bias.weight  (961,1) shape-mismatched -> stayed random N(0, 0.02).
                       GaussianLayer's constructor sets mul=1 / bias=0
                       (unimol.py:409-411), but UniMolModel.__init__ then runs
                       self.apply(init_bert_params) (unimol.py:177), which
                       re-initialises EVERY nn.Embedding -- including gbf.mul and
                       gbf.bias -- to N(0, 0.02). So the per-edge-type scale and
                       offset applied to interatomic distances really were noise,
                       as E15 states.

Arms (everything except the named parameters stays pretrained):
  pretrained  current cache -- all 193 parameters loaded
  pre         faithful pre-fix state: embed_tokens + gbf.mul + gbf.bias re-initialised
  gbf_id      only gbf.mul / gbf.bias re-init -> isolates the 3D distance pathway
  emb_rand    only embed_tokens re-init       -> isolates atom-element identity
  rand        the whole encoder re-initialised -> untrained-encoder control

Writes each arm to its own cache directory holding the standard filenames, so the
Phase-0 trainer can consume it by pointing revision_run.CACHE at that directory and
nothing else about the protocol changes.

Usage:  CUDA_VISIBLE_DEVICES=0 python ablate_unimol.py [cell ...]
"""
import os, json, pickle, sys, time, warnings, logging, argparse
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import concurrent.futures as cf
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from utils import smiles2graph

CACHE = Path("cache/ood_dpo_cache")
WORKERS = int(os.environ.get("GRAPH_WORKERS", "48"))
BATCH = int(os.environ.get("UNIMOL_BATCH", "256"))
BUDGET = int(os.environ.get("UNIMOL_BUDGET", "800000"))   # bound on batch_size * n_atoms^2
ARMS = ["pre", "gbf_id", "emb_rand", "rand"]
BASE = {"ec50_scaffold": "lbap_general_ec50_scaffold", "ec50_assay": "lbap_general_ec50_assay",
        "ic50_scaffold": "lbap_general_ic50_scaffold", "ic50_assay": "lbap_general_ic50_assay"}

EMB = "embed_tokens.weight"
GBF = ["gbf.mul.weight", "gbf.bias.weight"]


def build_graph(s):
    try:
        g = smiles2graph(s)
        if isinstance(g, dict) and "mol" in g:
            g = {k: v for k, v in g.items() if k != "mol"}
        return s, g
    except Exception:
        return s, None


def make_batches(smis, graphs, budget, cap=256):
    """Size-sorted dynamic batching. The Gaussian basis materialises a
    (B, N, N, K=128) tensor, so a single 500-atom molecule at a fixed batch of 256
    needs >30 GB. Bound B*N^2 instead of B."""
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
    """Encode one batch, halving it on CUDA OOM rather than losing the molecules."""
    import torch
    try:
        out = enc.encode_smiles(items)
        for k, f in zip(keys, out):
            feats[k] = f.detach().cpu().numpy().astype(np.float32)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "out of memory" not in str(e).lower() or len(items) == 1 or depth > 6:
            raise
        torch.cuda.empty_cache()
        h = len(items) // 2
        encode_batch(enc, items[:h], keys[:h], feats, depth + 1)
        encode_batch(enc, items[h:], keys[h:], feats, depth + 1)


def fresh_state(dictionary):
    """A second UniMolModel built with identical args but NO checkpoint load, so its
    parameters are exactly what the shape-mismatched ones fell back to."""
    import argparse as ap, torch
    from unimol.models import UniMolModel
    a = ap.Namespace(arch="unimol_base", encoder_layers=15, encoder_attention_heads=64,
                     encoder_embed_dim=512, encoder_ffn_embed_dim=2048, dropout=0.1,
                     attention_dropout=0.1, activation_dropout=0.0, pooler_dropout=0.0,
                     max_seq_len=512, post_ln=False, mode="infer", remove_hydrogen=False,
                     no_token_positional_embeddings=False, encoder_normalize_before=True,
                     masked_token_loss=-1.0, masked_coord_loss=-1.0, masked_dist_loss=-1.0,
                     x_norm_loss=-1.0, delta_pair_repr_norm_loss=-1.0, activation_fn="gelu",
                     pooler_activation_fn="tanh", emb_dropout=0.1)
    torch.manual_seed(0)
    return {k: v.clone() for k, v in UniMolModel(a, dictionary).state_dict().items()}


def apply_arm(model, pre_sd, fresh_sd, arm):
    import torch
    sd = {k: v.clone() for k, v in pre_sd.items()}
    if arm == "rand":
        sd = {k: v.clone() for k, v in fresh_sd.items()}
    else:
        names = {"pre": [EMB] + GBF, "gbf_id": GBF, "emb_rand": [EMB]}[arm]
        for n in names:
            hit = [k for k in sd if k.endswith(n)]
            assert hit, f"parameter {n} not found"
            for k in hit:
                sd[k] = fresh_sd[k].clone()
    model.load_state_dict(sd)
    return sorted(k for k in sd if not torch.equal(sd[k], pre_sd[k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="*", default=list(BASE))
    ap.add_argument("--arms", nargs="+", default=ARMS)
    a = ap.parse_args()

    import torch
    from model import UniMolEncoder
    enc = UniMolEncoder()
    dev = next(enc.model.parameters()).device
    print(f"UniMol on {dev}", flush=True)
    pre_sd = {k: v.detach().clone() for k, v in enc.model.state_dict().items()}
    fresh_sd = fresh_state(enc.dictionary)
    fresh_sd = {k: v.to(dev) for k, v in fresh_sd.items()}
    # sanity: init_bert_params overrides GaussianLayer's constant init, so the
    # fallback state of gbf.mul/bias should look like N(0, 0.02), not 1 / 0.
    for n in ("gbf.mul.weight", "gbf.bias.weight"):
        k = [x for x in fresh_sd if x.endswith(n)][0]
        w = fresh_sd[k]
        print(f"  fallback {n}: mean={w.mean():+.4f} std={w.std():.4f} "
              f"(expect ~0.000 / ~0.020)", flush=True)

    for cell in a.cells:
        b = BASE[cell]
        sp = json.load(open(CACHE / f"{b}_seed42_splits.json"))["splits"]
        smis = sorted(set(s for v in sp.values() for s in v))
        t0 = time.time(); graphs = {}
        with cf.ProcessPoolExecutor(max_workers=WORKERS) as ex:
            for s, g in ex.map(build_graph, smis, chunksize=16):
                if g is not None: graphs[s] = g
        inputs = [graphs.get(s, s) for s in smis]
        batches = make_batches(smis, graphs, BUDGET, BATCH)
        print(f"[{cell}] {len(smis)} mols, 3D built in {time.time()-t0:.0f}s, "
              f"{len(batches)} batches (max {max(len(b) for b in batches)})", flush=True)

        for arm in a.arms:
            changed = apply_arm(enc.model, pre_sd, fresh_sd, arm)
            outdir = Path(f"cache/abl_{arm}"); outdir.mkdir(parents=True, exist_ok=True)
            t1 = time.time(); feats = {}
            torch.cuda.empty_cache()
            for bidx in batches:
                encode_batch(enc, [inputs[i] for i in bidx], [smis[i] for i in bidx], feats)
            pickle.dump({"features": feats, "foundation_model": "unimol",
                         "dataset_name": b, "drugood_subset": b},
                        open(outdir / f"{b}_unimol_features.pkl", "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)
            for f in (f"{b}_minimol_features.pkl", f"{b}_seed42_splits.json"):
                lk = outdir / f
                if not lk.exists(): lk.symlink_to((CACHE / f).resolve())
            V = np.stack(list(feats.values()))
            print(f"[{cell}] {arm:9} {len(feats)} feats in {time.time()-t1:.0f}s | "
                  f"changed={len(changed)} params | |z| mean={np.abs(V).mean():.3f} "
                  f"std={V.std():.3f}", flush=True)
        enc.model.load_state_dict(pre_sd)          # restore pretrained
    print("ABLATE_DONE", flush=True)


if __name__ == "__main__":
    main()
