#!/usr/bin/env python3
"""Run all frozen final paired seeds after every selection file exists."""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rpo_opt_v2_final_one import FINAL_SEEDS


CELLS = tuple(
    (cell, backbone)
    for cell in ("ec50_assay", "ic50_assay")
    for backbone in ("minimol", "unimol")
)


def run_one(python: str, cell: str, backbone: str, seed: int) -> str:
    target = Path("repro") / f"rpo_opt_v2_final_{cell}_{backbone}_s{seed}.json"
    if target.exists():
        return f"SKIP {target}"
    log = Path("logs") / f"rpo_opt_v2_final_{cell}_{backbone}_s{seed}.log"
    cmd = [
        python,
        "rpo_opt_v2_final_one.py",
        "--cell",
        cell,
        "--backbone",
        backbone,
        "--seed",
        str(seed),
        "--device",
        "cpu",
    ]
    with log.open("w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"final run failed ({result.returncode}): {log}")
    return f"DONE {target}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    missing = [
        Path("repro") / f"rpo_opt_v2_selection_{cell}_{backbone}.json"
        for cell, backbone in CELLS
        if not (Path("repro") / f"rpo_opt_v2_selection_{cell}_{backbone}.json").exists()
    ]
    if missing:
        raise RuntimeError(f"selection is not frozen: {missing}")
    jobs = [(cell, backbone, seed) for cell, backbone in CELLS for seed in FINAL_SEEDS]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, args.python, cell, backbone, seed): (cell, backbone, seed)
            for cell, backbone, seed in jobs
        }
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()

