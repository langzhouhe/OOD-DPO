#!/usr/bin/env python3
"""Build the independent artifact manifest for the equal-48 Table 1 extension.

The original RPO-OE manifest describes the four-setting primary experiment and must
not silently expand when the twelve-column classifier-score table is generated.  This
script therefore validates and records only the three requested OE rows over the exact
Table 1 cells/backbones.  The manifest intentionally excludes its own output so that it
does not contain an impossible self-referential hash.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import oe_equal48 as E
import oe_equal48_table1_aggregate as A


ROOT = Path(__file__).resolve().parent
REPRO = ROOT / "repro"
SUMMARY = REPRO / "oe_equal48_table1_summary.json"
ROWS = REPRO / "oe_equal48_table1_rows.tex"
MARKDOWN = REPRO / "oe_equal48_table1_summary.md"
PROTOCOL = ROOT / "OE_EQUAL48_TABLE1_PROTOCOL.md"
TARGET = REPRO / "MANIFEST_oe_equal48_table1.json"


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _require_exact_json(path: Path, expected: dict, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = json.loads(path.read_text())
    if actual != expected:
        raise RuntimeError(f"{description} is stale or disagrees with raw artifacts: {path}")


def main() -> None:
    # E's canonical artifact path helpers are intentionally repository-relative.
    # Normalize the working directory here so this command is safe from any caller.
    os.chdir(ROOT)
    expected_summary = A.aggregate()
    _require_exact_json(SUMMARY, expected_summary, "Table 1 aggregate")
    if not ROWS.is_file() or ROWS.read_text() != A.latex_rows(expected_summary):
        raise RuntimeError(f"generated LaTeX rows are stale: {ROWS}")
    if not MARKDOWN.is_file() or MARKDOWN.read_text() != A.markdown(expected_summary):
        raise RuntimeError(f"generated Markdown summary is stale: {MARKDOWN}")
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)

    screen_files = [
        ROOT / E._screen_path(cell, backbone, method, seed)
        for cell in A.TABLE_ORDER
        for backbone in E.BACKBONES
        for method in A.METHODS
        for seed in E.SELECTION_SEEDS
    ]
    selection_files = [
        ROOT / E._selection_path(cell, backbone, method)
        for cell in A.TABLE_ORDER
        for backbone in E.BACKBONES
        for method in A.METHODS
    ]
    final_files = [
        ROOT / E._final_path(cell, backbone, method, seed)
        for cell in A.TABLE_ORDER
        for backbone in E.BACKBONES
        for method in A.METHODS
        for seed in E.FINAL_SEEDS
    ]
    expected_counts = {"screen": 360, "selection": 72, "final": 1440}
    actual_counts = {
        "screen": len(screen_files),
        "selection": len(selection_files),
        "final": len(final_files),
    }
    if actual_counts != expected_counts:
        raise RuntimeError(f"internal artifact cardinality error: {actual_counts}")
    missing = [path for path in screen_files + selection_files + final_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} Table 1 artifacts; first: {missing[0]}")

    code_and_outputs = [
        ROOT / "oe_equal48.py",
        ROOT / "run_oe_equal48.py",
        ROOT / "oe_equal48_aggregate.py",
        ROOT / "oe_equal48_table1_aggregate.py",
        ROOT / "build_oe_equal48_table1_manifest.py",
        ROOT / "good_labels.py",
        ROOT / "cache/good_labels_hiv.json",
        ROOT / "cache/good_labels_pcba.json",
        ROOT / "cache/good_labels_zinc.json",
        PROTOCOL,
        SUMMARY,
        ROWS,
        MARKDOWN,
    ]
    all_files = screen_files + selection_files + final_files + code_and_outputs
    if len(all_files) != len(set(all_files)):
        raise RuntimeError("duplicate file in Table 1 manifest input set")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest = {
        "protocol": "oe-equal48-table1-artifact-manifest-v1",
        "source_protocol": E.PROTOCOL,
        "scope_protocol": expected_summary["scope_protocol"],
        "repository_base_commit": commit,
        "self_hash_excluded": True,
        "cells_in_table_order": list(A.TABLE_ORDER),
        "backbones": list(E.BACKBONES),
        "methods": list(A.METHODS),
        "selection_seeds": list(E.SELECTION_SEEDS),
        "final_seeds": list(E.FINAL_SEEDS),
        "grid_protocol": E.GRID_PROTOCOL,
        "all_grid_hash": E.all_grid_hash(),
        "method_grid_hashes": {
            method: E.grid_hash(method) for method in A.METHODS
        },
        "good_classifier_label_artifacts": {
            tag: {
                "path": f"cache/good_labels_{tag}.json",
                "meta": json.loads(
                    (ROOT / f"cache/good_labels_{tag}.json").read_text()
                )["meta"],
            }
            for tag in ("hiv", "pcba", "zinc")
        },
        "counts": actual_counts,
        "files": {
            _relative(path): {
                "bytes": path.stat().st_size,
                "sha256": E.file_sha256(path),
            }
            for path in sorted(all_files)
        },
    }
    TARGET.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"WROTE {TARGET} ({len(all_files)} hashed files; self excluded)")


if __name__ == "__main__":
    main()
