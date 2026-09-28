#!/usr/bin/env python3
"""Run the four frozen matched-OE v2 baseline settings in parallel."""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SETTINGS = tuple(
    (cell, backbone)
    for cell in ("ec50_assay", "ic50_assay")
    for backbone in ("minimol", "unimol")
)


def run_one(python: str, cell: str, backbone: str) -> str:
    target = Path("repro") / f"oe_baselines_v2_{cell}_{backbone}.json"
    if target.exists():
        return f"SKIP {target}"
    log = Path("logs") / f"oe_baselines_v2_{cell}_{backbone}.log"
    command = [
        python,
        "oe_baselines_v2.py",
        "--cell",
        cell,
        "--backbone",
        backbone,
    ]
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"OE baseline run failed ({result.returncode}): {log}")
    return f"DONE {target} log={log}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, args.python, cell, backbone): (cell, backbone)
            for cell, backbone in SETTINGS
        }
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
