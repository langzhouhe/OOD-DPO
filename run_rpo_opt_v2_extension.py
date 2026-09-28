#!/usr/bin/env python3
"""Parallel CPU launcher for the frozen v2 boundary extension."""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rpo_opt_v2_screen import SELECTION_SEEDS


CELLS = tuple(
    (cell, backbone)
    for cell in ("ec50_assay", "ic50_assay")
    for backbone in ("minimol", "unimol")
)


def run_one(python: str, cell: str, backbone: str, seed: int) -> str:
    target = Path("repro") / f"rpo_opt_v2_extension_{cell}_{backbone}_s{seed}.json"
    if target.exists():
        return f"SKIP {target}"
    log = Path("logs") / f"rpo_opt_v2_extension_{cell}_{backbone}_s{seed}.log"
    cmd = [
        python,
        "rpo_opt_v2_extension.py",
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
        raise RuntimeError(f"extension failed ({result.returncode}): {log}")
    return f"DONE {target}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    jobs = [(c, b, s) for c, b in CELLS for s in SELECTION_SEEDS]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, args.python, c, b, s): (c, b, s) for c, b, s in jobs
        }
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()

