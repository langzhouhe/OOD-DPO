#!/usr/bin/env python3
"""Build a SHA256 manifest for the frozen RPO-OE v2 paper artifacts."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from rpo_opt_v2_paper_aggregate import (
    EQUAL48_SUMMARY,
    LEGACY_BASELINE_PROTOCOL,
    load_and_validate_equal48_summary,
)


ROOT = Path(__file__).resolve().parent
REPRO = ROOT / "repro"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    equal48_path = REPRO / EQUAL48_SUMMARY.name
    equal48 = equal48_path.is_file()
    baseline_patterns = (
        (
            "oe_equal48_screen_*.json",
            "oe_equal48_selection_*.json",
            "oe_equal48_final_*.json",
            "oe_equal48_summary.json",
        )
        if equal48
        else ("oe_baselines_v2_*.json",)
    )
    patterns = (
        "rpo_opt_v2_screen_*.json",
        "rpo_opt_v2_extension_*.json",
        "rpo_opt_v2_selection_*.json",
        "rpo_opt_v2_final_*.json",
        "rpo_opt_v2_shared_selection_*.json",
        "rpo_opt_v2_shared_final_*.json",
        "rpo_opt_v2_shared_summary.*",
        "pairwise_hinge_v2_screen_*.json",
        "pairwise_hinge_v2_selection_*.json",
        "pairwise_hinge_v2_final_*.json",
        "pairwise_hinge_v2_final_summary.*",
        "rpo_opt_v2_paper_summary.*",
        "rpo_opt_v2_paper_tables.tex",
        "rpo_opt_v2_compact_table.tex",
        "rpo_opt_v2_main_table.tex",
        "rpo_opt_v2_paper_text.tex",
    ) + baseline_patterns
    files = set()
    for pattern in patterns:
        files.update(REPRO.glob(pattern))
    protocols = [ROOT / "RPO_OE_TUNING_V2_PROTOCOL.md", ROOT / "PAIRWISE_HINGE_V2_PROTOCOL.md"]
    if equal48:
        protocols.append(ROOT / "OE_EQUAL48_PROTOCOL.md")
    files.update(path for path in protocols if path.exists())
    required_counts = {
        "rpo_opt_v2_screen": len(list(REPRO.glob("rpo_opt_v2_screen_*.json"))),
        "rpo_opt_v2_extension": len(list(REPRO.glob("rpo_opt_v2_extension_*.json"))),
        "rpo_opt_v2_final": len(list(REPRO.glob("rpo_opt_v2_final_*_s*.json"))),
        "rpo_opt_v2_shared_final": len(list(REPRO.glob("rpo_opt_v2_shared_final_*.json"))),
        "pairwise_hinge_v2_screen": len(list(REPRO.glob("pairwise_hinge_v2_screen_*.json"))),
        "pairwise_hinge_v2_final": len(list(REPRO.glob("pairwise_hinge_v2_final_*_s*.json"))),
    }
    expected = {
        "rpo_opt_v2_screen": 20,
        "rpo_opt_v2_extension": 20,
        "rpo_opt_v2_final": 80,
        "rpo_opt_v2_shared_final": 80,
        "pairwise_hinge_v2_screen": 20,
        "pairwise_hinge_v2_final": 80,
    }
    if equal48:
        # This also verifies the frozen protocol/grid hashes, exact final-seed order,
        # and every selection/final artifact SHA256 before the manifest is written.
        equal48_doc = load_and_validate_equal48_summary(equal48_path)
        required_counts.update(
            {
                "oe_equal48_screen": len(list(REPRO.glob("oe_equal48_screen_*.json"))),
                "oe_equal48_selection": len(list(REPRO.glob("oe_equal48_selection_*.json"))),
                "oe_equal48_final": len(list(REPRO.glob("oe_equal48_final_*.json"))),
                "oe_equal48_summary": len(list(REPRO.glob("oe_equal48_summary.json"))),
            }
        )
        expected.update(
            {
                "oe_equal48_screen": 120,
                "oe_equal48_selection": 24,
                "oe_equal48_final": 480,
                "oe_equal48_summary": 1,
            }
        )
        baseline_source = {
            "kind": "equal48",
            "protocol": equal48_doc["protocol"],
            "grid_protocol": equal48_doc["grid_protocol"],
            "all_grid_hash": equal48_doc["all_grid_hash"],
            "method_grid_hashes": equal48_doc["method_grid_hashes"],
            "n_validation_trials_per_method": 48,
            "n_final_seeds": 20,
        }
    else:
        required_counts["oe_baselines_v2"] = len(
            list(REPRO.glob("oe_baselines_v2_*.json"))
        )
        expected["oe_baselines_v2"] = 4
        baseline_source = {
            "kind": "legacy9-fallback",
            "protocol": LEGACY_BASELINE_PROTOCOL,
            "n_validation_trials_per_method": 9,
            "n_final_seeds": 20,
        }
    if required_counts != expected:
        raise RuntimeError(f"incomplete artifact set: {required_counts}, expected {expected}")
    paper_summary_path = REPRO / "rpo_opt_v2_paper_summary.json"
    paper_summary = json.loads(paper_summary_path.read_text())
    if paper_summary.get("baseline_source", {}).get("kind") != baseline_source["kind"]:
        raise RuntimeError(
            "paper summary was generated from a different OE baseline source; "
            "rerun rpo_opt_v2_paper_aggregate.py and rpo_opt_v2_compact_table.py"
        )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "protocol": "rpo-oe-v2-paper-artifact-manifest",
        "repository_base_commit": commit,
        "baseline_source": baseline_source,
        "counts": required_counts,
        "files": {
            str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(files)
        },
    }
    target = REPRO / "MANIFEST_rpo_oe_v2.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"WROTE {target} ({len(files)} files)")


if __name__ == "__main__":
    main()
