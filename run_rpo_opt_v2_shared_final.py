#!/usr/bin/env python3
"""Parallel launcher for the objective-blind shared-recipe final comparison."""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rpo_opt_v2_final_one import FINAL_SEEDS


SETTINGS = tuple((c, b) for c in ("ec50_assay", "ic50_assay")
                 for b in ("minimol", "unimol"))


def run_one(python: str, cell: str, backbone: str, seed: int) -> str:
    target = Path("repro") / f"rpo_opt_v2_shared_final_{cell}_{backbone}_s{seed}.json"
    if target.exists():
        return f"SKIP {target}"
    log = Path("logs") / f"rpo_opt_v2_shared_final_{cell}_{backbone}_s{seed}.log"
    command = [python, "rpo_opt_v2_shared_final_one.py", "--cell", cell,
               "--backbone", backbone, "--seed", str(seed), "--device", "cpu"]
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"shared final failed ({result.returncode}): {log}")
    return f"DONE {target}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    jobs = [(c, b, s) for c, b in SETTINGS for s in FINAL_SEEDS]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, args.python, c, b, s): (c, b, s)
                   for c, b, s in jobs}
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
