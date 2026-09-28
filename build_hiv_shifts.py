"""
Build DrugOOD-format OOD-detection JSONs from the real MoleculeNet HIV set.

Produces two files under ./data/raw/ that plug directly into
process_drugood_data() (utils.py) -> the RPO ("Energy-DPO") pipeline:

  - lbap_general_hiv_size.json     : covariate shift on molecular SIZE
  - lbap_general_hiv_scaffold.json : covariate shift on Bemis-Murcko SCAFFOLD

These reproduce the paper's *shift design* (Size / Scaffold) on the exact HIV
molecules used by the GOOD-HIV benchmark. ID and OOD pools are disjoint by
construction, and train/val/test within each pool are disjoint, matching what
the data_loader's cross-split overlap check requires.

JSON schema expected by utils.process_drugood_data:
  {"split": {"train":[{smiles,cls_label}], "iid_val":[...], "iid_test":[...],
             "ood_val":[...], "ood_test":[...]}}
"""
import csv, json, os, random
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
random.seed(42)

CSV = "data/raw/HIV.csv"
OUT_DIR = "data/raw"


def load_molecules():
    mols = []
    with open(CSV) as f:
        for row in csv.DictReader(f):
            smi = (row.get("smiles") or "").strip()
            if not smi:
                continue
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            try:
                scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
            except Exception:
                scaf = ""
            mols.append({
                "smiles": smi,
                "cls_label": int(float(row.get("HIV_active", 0) or 0)),
                "n_heavy": m.GetNumHeavyAtoms(),
                "scaffold": scaf,
            })
    # dedup by SMILES
    seen, uniq = set(), []
    for r in mols:
        if r["smiles"] in seen:
            continue
        seen.add(r["smiles"])
        uniq.append(r)
    return uniq


def rec(r):
    return {"smiles": r["smiles"], "cls_label": r["cls_label"]}


def partition_id_pool(id_recs):
    """Split a disjoint ID pool into train / iid_val / iid_test."""
    random.shuffle(id_recs)
    n = len(id_recs)
    n_val = min(1200, n // 6)
    n_test = min(1500, n // 5)
    iid_val = id_recs[:n_val]
    iid_test = id_recs[n_val:n_val + n_test]
    train = id_recs[n_val + n_test:]
    return train, iid_val, iid_test


def partition_ood_pool(ood_recs):
    """Split a disjoint OOD pool into ood_val (training negatives) / ood_test."""
    random.shuffle(ood_recs)
    n = len(ood_recs)
    n_test = min(1500, n // 3)
    ood_test = ood_recs[:n_test]
    ood_val = ood_recs[n_test:]
    return ood_val, ood_test


def write_split(name, train, iid_val, iid_test, ood_val, ood_test):
    obj = {"split": {
        "train":    [rec(r) for r in train],
        "iid_val":  [rec(r) for r in iid_val],
        "iid_test": [rec(r) for r in iid_test],
        "ood_val":  [rec(r) for r in ood_val],
        "ood_test": [rec(r) for r in ood_test],
    }}
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f"[{name}] train={len(train)} iid_val={len(iid_val)} iid_test={len(iid_test)} "
          f"ood_val={len(ood_val)} ood_test={len(ood_test)}  -> {path}")


def build_size(mols):
    """ID = smaller molecules (bottom band), OOD = larger molecules (top band),
    with a gap band left out to make the covariate shift clear."""
    s = sorted(mols, key=lambda r: r["n_heavy"])
    n = len(s)
    id_pool = s[: int(0.60 * n)]          # smallest 60%
    ood_pool = s[int(0.80 * n):]          # largest 20% (gap band 60-80% unused)
    id_sizes = [r["n_heavy"] for r in id_pool]
    ood_sizes = [r["n_heavy"] for r in ood_pool]
    print(f"[size] ID heavy-atoms {min(id_sizes)}-{max(id_sizes)} (n={len(id_pool)}); "
          f"OOD heavy-atoms {min(ood_sizes)}-{max(ood_sizes)} (n={len(ood_pool)})")
    tr, iv, it = partition_id_pool(list(id_pool))
    ov, ot = partition_ood_pool(list(ood_pool))
    write_split("lbap_general_hiv_size", tr, iv, it, ov, ot)


def build_scaffold(mols):
    """ID and OOD get disjoint Bemis-Murcko scaffolds (standard scaffold split)."""
    by_scaf = defaultdict(list)
    for r in mols:
        by_scaf[r["scaffold"]].append(r)
    scafs = [sc for sc in by_scaf if sc]          # drop empty-scaffold bucket
    random.shuffle(scafs)
    # assign ~78% of scaffolds to ID, rest to OOD (disjoint scaffold sets)
    cut = int(0.78 * len(scafs))
    id_scafs, ood_scafs = set(scafs[:cut]), set(scafs[cut:])
    id_pool = [r for sc in id_scafs for r in by_scaf[sc]]
    ood_pool = [r for sc in ood_scafs for r in by_scaf[sc]]
    print(f"[scaffold] {len(id_scafs)} ID scaffolds (n={len(id_pool)} mols); "
          f"{len(ood_scafs)} OOD scaffolds (n={len(ood_pool)} mols)")
    tr, iv, it = partition_id_pool(list(id_pool))
    ov, ot = partition_ood_pool(list(ood_pool))
    write_split("lbap_general_hiv_scaffold", tr, iv, it, ov, ot)


if __name__ == "__main__":
    mols = load_molecules()
    print(f"Loaded {len(mols)} unique valid HIV molecules")
    build_size(mols)
    build_scaffold(mols)
