#!/usr/bin/env python3
"""Bounded-parallel launcher for the equal-48 OE pipeline.

Phases must be invoked explicitly.  In particular, this launcher never chains selection
to final evaluation in one command; this leaves a reviewable frozen-selection boundary.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import oe_equal48 as E


def _slug(method: str) -> str:
    return method.lower().replace("-", "_")


def _target(phase: str, cell: str, backbone: str, method: str, seed: int | None) -> Path:
    artifact_phase = "selection" if phase == "select" else phase
    stem = f"oe_equal48_{artifact_phase}_{cell}_{backbone}_{_slug(method)}"
    if seed is not None:
        stem += f"_s{seed}"
    return Path("repro") / f"{stem}.json"


def _run_one(
    python: str,
    phase: str,
    cell: str,
    backbone: str,
    method: str,
    seed: int | None,
    device: str,
) -> str:
    target = _target(phase, cell, backbone, method, seed)
    if target.exists():
        return f"SKIP {target}"
    log = Path("logs") / (target.stem + ".log")
    log.parent.mkdir(exist_ok=True)
    command = [
        python,
        "oe_equal48.py",
        phase,
        "--cell",
        cell,
        "--backbone",
        backbone,
        "--method",
        method,
        "--device",
        device,
    ]
    if seed is not None:
        command += ["--seed", str(seed)]
    with log.open("w") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"{phase} failed ({completed.returncode}): {log}")
    return f"DONE {target} log={log}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("screen", "select", "final"))
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--method", action="append", choices=E.METHODS)
    parser.add_argument(
        "--scope",
        choices=("primary", "table1"),
        default="primary",
        help="primary: two assay cells; table1: all 12 manuscript cells",
    )
    parser.add_argument(
        "--cell",
        action="append",
        choices=E.TABLE1_CELLS,
        help="optional explicit cell subset (overrides --scope)",
    )
    args = parser.parse_args()
    methods = tuple(args.method) if args.method else E.METHODS
    cells = (
        tuple(args.cell)
        if args.cell
        else (E.TABLE1_CELLS if args.scope == "table1" else E.CELLS)
    )
    if args.phase == "screen":
        jobs = [
            (cell, backbone, method, seed)
            for cell in cells
            for backbone in E.BACKBONES
            for method in methods
            for seed in E.SELECTION_SEEDS
        ]
    elif args.phase == "select":
        jobs = [
            (cell, backbone, method, None)
            for cell in cells
            for backbone in E.BACKBONES
            for method in methods
        ]
    else:
        jobs = [
            (cell, backbone, method, seed)
            for cell in cells
            for backbone in E.BACKBONES
            for method in methods
            for seed in E.FINAL_SEEDS
        ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _run_one,
                args.python,
                args.phase,
                cell,
                backbone,
                method,
                seed,
                args.device,
            )
            for cell, backbone, method, seed in jobs
        ]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
