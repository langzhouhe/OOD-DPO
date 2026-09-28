"""Prefetch + process all GOOD datasets (HIV/PCBA/ZINC, scaffold+size covariate).
First access per dataset downloads the official processed data via gdown; results
are cached under ./data so later training runs load instantly."""
import warnings, logging, sys
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from utils import process_good_data

for ds in ["good_hiv", "good_pcba", "good_zinc"]:
    for dom in ["scaffold", "size"]:
        try:
            d = process_good_data(ds, domain=dom, shift="covariate",
                                  data_path="./data", validate_smiles_flag=False)
            n = {k: len(v) for k, v in d.items() if isinstance(v, list)}
            print(f"OK {ds} {dom}: {n}", flush=True)
        except Exception as e:
            print(f"FAIL {ds} {dom}: {repr(e)[:200]}", flush=True)
print("GOOD_PREFETCH_DONE", flush=True)
