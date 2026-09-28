#!/usr/bin/env python3
"""Build the official ood_test mechanism substrate for MolRoute.

Development mechanisms keep coming from the existing ood_val catalog (cache/mepoe seed 21).
This adds a disjoint set drawn from official `ood_test`, which the benchmark guarantees
shares no domain_id with `ood_val` (verified, 0 overlap on all five cells).

400 mechanisms per cell at seed 777, both fixed in PROTOCOL_MOLROUTE.md before any score
was computed -- using all 14,964 eligible mechanisms would need ~120k new encodings.

Usage: python molroute_data.py --cell ec50_assay
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json, pickle, argparse, warnings, logging, collections
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

CACHE = Path("cache/ood_dpo_cache"); MP = Path("cache/mepoe"); OUT = Path("cache/molroute")
BASE = {"ec50_assay": "lbap_general_ec50_assay", "ic50_assay": "lbap_general_ic50_assay",
        "ki_assay": "lbap_general_ki_assay",
        "ec50_scaffold": "lbap_general_ec50_scaffold",
        "ic50_scaffold": "lbap_general_ic50_scaffold"}
M = 8
N_TEST_MECH = 400
TEST_SEED = 777
DEV_SEED = 21          # the ood_val catalog the detectors were developed on


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    b = BASE[a.cell]
    raw = json.load(open(f"data/raw/{b}.json"))["split"]

    by = collections.defaultdict(set)
    for it in raw["ood_test"]:
        s, d = it.get("smiles"), it.get("domain_id")
        if s and d is not None: by[d].add(s)
    elig = sorted([d for d, v in by.items() if len(v) >= M])
    rng = np.random.default_rng(TEST_SEED)
    rng.shuffle(elig)
    take = elig[:N_TEST_MECH]
    sets = {}
    for d in take:
        u = sorted(by[d])
        sets[d] = [u[i] for i in rng.choice(len(u), size=M, replace=False)]
    print(f"[{a.cell}] {len(elig)} eligible ood_test mechanisms -> using {len(take)}", flush=True)

    dev = json.load(open(MP / f"{b}_mepoe_seed{DEV_SEED}.json"))
    dev_dom = {int(x) if str(x).lstrip('-').isdigit() else x
               for n in dev["catalog"] for x in dev["catalog"][n]}
    dev_dom |= {str(x) for n in dev["catalog"] for x in dev["catalog"][n]}
    ov_dom = len({str(d) for d in take} & {str(d) for d in dev_dom})

    can_test = {c for ss in sets.values() for c in (canon(s) for s in ss) if c}
    can_dev = {c for n in dev["catalog"] for ss in dev["catalog"][n].values()
               for c in (canon(s) for s in ss) if c}
    can_idtr = {c for c in (canon(s) for s in dev["id_train"]) if c}
    can_idte = {c for c in (canon(s) for s in dev["id_test"]) if c}
    audit = {"domain_overlap_test_vs_dev": ov_dom,
             "canon_overlap_test_vs_devOOD": len(can_test & can_dev),
             "canon_overlap_test_vs_ID_train": len(can_test & can_idtr),
             "canon_overlap_test_vs_ID_test": len(can_test & can_idte),
             "n_eligible": len(elig), "n_used": len(take)}
    print(f"[{a.cell}] audit {audit}", flush=True)

    rec = {"catalog": {str(d): v for d, v in sets.items()}, "seed": TEST_SEED, "M": M,
           "dev_seed": DEV_SEED, "audit": audit,
           "id_train": dev["id_train"], "id_test": dev["id_test"]}
    json.dump(rec, open(OUT / f"{b}_molroute_test.json", "w"))

    fpath = CACHE / f"{b}_minimol_features.pkl"
    blob = pickle.load(open(fpath, "rb")); feats = blob["features"]
    need = sorted({s for ss in sets.values() for s in ss} - set(feats))
    print(f"[{a.cell}] {len(need)} molecules need MiniMol encoding", flush=True)
    if need:
        from model import MinimolEncoder
        enc = MinimolEncoder(); done = 0
        for i in range(0, len(need), 256):
            ch = need[i:i + 256]
            try:
                for s, f in zip(ch, enc.encode_smiles(ch)):
                    feats[s] = np.asarray(f, dtype=np.float32); done += 1
            except Exception:
                for s in ch:
                    try:
                        feats[s] = np.asarray(enc.encode_smiles([s])[0], dtype=np.float32); done += 1
                    except Exception: pass
            print(f"  [{a.cell}] encoded {done}/{len(need)}", flush=True)
        blob["features"] = feats
        pickle.dump(blob, open(fpath, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print(f"MOLROUTE_DATA_DONE {a.cell}", flush=True)


if __name__ == "__main__":
    main()
