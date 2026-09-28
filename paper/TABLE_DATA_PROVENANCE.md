# Table data provenance

- Original attachment: `/root/.codex/attachments/12b25b3b-05e2-4831-bd0e-6c3ae6cf9184/pasted-text.txt`
- Original SHA256: `801f3cb1e19b1159629fb1e7bc5bd02ea925f604cc3786131976637fe922030f`
- Workspace snapshot: `paper/main.original.tex` (same SHA256)
- Updated manuscript: `paper/main.tex`
- Updated SHA256: `e73ea5e75f18d1d416221d758c6e2242f2cbb62bdf0ffa265fa50fbce3a1af61`

## Updated tables

1. `AUROC` (main table): canonical matched-OE means from `repro/moe_*.json`; same 13-column layout and same number of method rows. The obsolete confounded Improvement row was replaced by the capacity/data-matched Balanced OOD Head.
2. `tab:rpo_bce_matched` (new compact table): optimized RPO versus optimized balanced BCE from `repro/rpo_opt_final_*.json` and `repro/rpo_opt_final_summary.md`; this is kept separate because its 32-recipe/10-seed protocol is not interchangeable with the legacy equal-9-trial/5-seed matched-OE table.
3. `tab:drugood_results`: objective-only shared-recipe thinning results from `repro/rpo_opt_thinning_*` and `repro/rpo_opt_thinning_summary.md`; original 5-column/3-row layout retained.
4. `tab:fpr95_results`: canonical matched-OE FPR95 means from `repro/moe_*.json`; same 13-column layout and same number of rows.
5. `Table:overall_performance`: canonical AUROC/AUPR/FPR95 mean plus sample SD from `repro/moe_*.json`; original 16-column layout retained. The redundant ODIN-OE full-metric column was used for Balanced BCE so the table width did not change; ODIN-OE remains reported in the main AUROC and FPR95 tables.

## Deliberately unchanged

Dataset statistics, cross-shift transfer, external SOTA, uncertainty, efficiency, and fine-tuning tables were not overwritten because the new canonical artifacts do not provide protocol-compatible replacement values for all of their cells.

## Verification

- 384/384 rounded AUROC/FPR95 cells in the two 12-cell tables match canonical JSON.
- 504 means and 504 sample standard deviations in the full three-metric table match canonical JSON.
- Modified-table column counts pass.
- Active LaTeX table/tabular environments are balanced.
- A full PDF compile was not possible in this workspace because the attachment did not include `neurips_2026.sty`, `reference.bib`, or the referenced figure assets.
