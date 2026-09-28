#!/usr/bin/env python3
"""Build the MePOE Gate-0 substrate: a source catalog plus two disjoint pseudo-target
mechanism sets.

Gate 0 asks whether there is ANY stable, learnable signal in *which* auxiliary
mechanisms you expose -- before writing a single line of preference-learning code. If
mechanism choice does not transfer from one held-out mechanism population to another,
there is nothing for an acquisition policy to learn and line B stops immediately.

  S    source catalog       N_S mechanisms -- the acquisition pool
  H1   pseudo-target A      N_H mechanisms -- used to rank subsets
  H2   pseudo-target B      N_H mechanisms -- used to test whether that ranking holds

All three are mutually disjoint at the domain_id level, and all exclude the domains
consumed by the MRPO build so that experiment stays independent. Molecules are also
checked for cross-set overlap on RDKit canonical SMILES, not just raw strings, because
the same compound can appear in several assays under different SMILES spellings.

ID molecules come unchanged from the existing cache.

Usage: python mepoe_data.py --cell ec50_assay --seed 21
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json, pickle, argparse, warnings, logging, collections
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

CACHE = Path("cache/ood_dpo_cache"); OUT = Path("cache/mepoe")
BASE = {"ec50_assay": "lbap_general_ec50_assay", "ic50_assay": "lbap_general_ic50_assay",
        "ki_assay": "lbap_general_ki_assay",
        "ec50_scaffold": "lbap_general_ec50_scaffold",
        "ic50_scaffold": "lbap_general_ic50_scaffold"}
M = 8
N_S, N_H = 200, 100


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    b = BASE[a.cell]
    raw = json.load(open(f"data/raw/{b}.json"))["split"]

    used = set()
    p = Path(f"cache/mrpo/{b}_mrpo_seed7_splits.json")
    if p.exists():
        r = json.load(open(p))
        for k in r["domains"]: used |= set(r["domains"][k])

    by = collections.defaultdict(set)
    for it in raw["ood_val"]:
        s, d = it.get("smiles"), it.get("domain_id")
        if s and d is not None and d not in used: by[d].add(s)
    elig = sorted([d for d, v in by.items() if len(v) >= M])
    need = N_S + 2 * N_H
    assert len(elig) >= need, f"only {len(elig)} eligible domains, need {need}"

    rng = np.random.default_rng(a.seed)
    rng.shuffle(elig)
    picks = {"S": elig[:N_S], "H1": elig[N_S:N_S + N_H], "H2": elig[N_S + N_H:need]}
    sets = {}
    for name, doms in picks.items():
        sets[name] = {}
        for d in doms:
            u = sorted(by[d])
            sets[name][d] = [u[i] for i in rng.choice(len(u), size=M, replace=False)]

    # disjointness: domain level, then canonical-SMILES level
    for x in picks:
        for y in picks:
            if x < y: assert not (set(picks[x]) & set(picks[y])), f"domain overlap {x}/{y}"
    can = {n: {c for ss in sets[n].values() for c in (canon(s) for s in ss) if c} for n in sets}
    sp_path = CACHE / f"{b}_seed42_splits.json"
    if sp_path.exists():
        old = json.load(open(sp_path))["splits"]
    else:
        # endpoint never used before (e.g. the sealed Ki): build ID splits from the
        # official ID partitions, same sizes as every other cell
        rid = np.random.default_rng(12345)
        def take(part, n):
            u = sorted({it["smiles"] for it in raw.get(part, []) if it.get("smiles")})
            return [u[i] for i in rid.choice(len(u), size=min(n, len(u)), replace=False)]
        old = {"train_id": take("train", 2000), "test_id": take("iid_test", 1000)}
        print(f"[{a.cell}] built fresh ID splits: {len(old['train_id'])} train / "
              f"{len(old['test_id'])} test", flush=True)
    can["ID_train"] = {c for c in (canon(s) for s in old["train_id"]) if c}
    can["ID_test"] = {c for c in (canon(s) for s in old["test_id"]) if c}
    ov = {f"{x}&{y}": len(can[x] & can[y]) for i, x in enumerate(can) for y in list(can)[i + 1:]}
    print(f"[{a.cell}] canonical-SMILES overlaps: {ov}", flush=True)

    rec = {"catalog": {n: {str(d): v for d, v in sets[n].items()} for n in sets},
           "id_train": old["train_id"], "id_test": old["test_id"],
           "seed": a.seed, "M": M, "n_mech": {n: len(sets[n]) for n in sets},
           "canon_overlap": ov}
    json.dump(rec, open(OUT / f"{b}_mepoe_seed{a.seed}.json", "w"))
    print(f"[{a.cell}] mechanisms S/H1/H2 = {len(sets['S'])}/{len(sets['H1'])}/{len(sets['H2'])}", flush=True)

    fpath = CACHE / f"{b}_minimol_features.pkl"
    if fpath.exists():
        blob = pickle.load(open(fpath, "rb")); feats = blob["features"]
    else:                       # first time this endpoint is used at all (e.g. sealed Ki)
        blob = {"features": {}, "foundation_model": "minimol", "dataset_name": b,
                "drugood_subset": b}
        feats = blob["features"]
    # ID molecules need encoding too when the endpoint is new
    want = {s for n in sets for ss in sets[n].values() for s in ss}
    want |= set(old["train_id"]) | set(old["test_id"])
    need_enc = sorted(want - set(feats))
    print(f"[{a.cell}] {len(need_enc)} molecules need encoding", flush=True)
    if need_enc:
        from model import MinimolEncoder
        enc = MinimolEncoder(); done = 0
        for i in range(0, len(need_enc), 256):
            ch = need_enc[i:i + 256]
            try:
                for s, f in zip(ch, enc.encode_smiles(ch)):
                    feats[s] = np.asarray(f, dtype=np.float32); done += 1
            except Exception:
                for s in ch:
                    try:
                        feats[s] = np.asarray(enc.encode_smiles([s])[0], dtype=np.float32); done += 1
                    except Exception: pass
            print(f"  [{a.cell}] encoded {done}/{len(need_enc)}", flush=True)
        blob["features"] = feats
        pickle.dump(blob, open(fpath, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print(f"MEPOE_DATA_DONE {a.cell}", flush=True)


if __name__ == "__main__":
    main()
