# RPO-OE tuning v2: frozen protocol

Frozen before creating any `rpo_opt_v2_final_*.json` test result.

The boundary-extension protocol was frozen at repository base commit
`c0a1b77e5f0e8f78fbf72d7fab5b9ac59f1c7f52`; its exact recipe list is stored and
hashed in every extension artifact.

## Scope

- Cells: EC50-Assay and IC50-Assay.
- Backbones: MiniMol and Uni-Mol.
- Encoder: frozen cached representation.
- Head: the manuscript's unchanged `d -> 256 -> 128 -> 1` scalar MLP.
- RPO objective: unchanged all-pairs pairwise logistic loss.
- Exact matched control: class-balanced BCE with the same molecules, head, optimizer
  family, preprocessing candidates, recipe count, presentation budget and checkpoint count.

## Validation-only selection

- Use the fixed 32-recipe Sobol grid and recipe hash from `rpo_opt_screen.py`.
- Unlike v1, evaluate all 32 recipes independently for every cell/backbone/objective;
  there is no global top-eight truncation.
- Selection seeds: 21, 22, 23, 24, 25.
- Per-run budget: 1,024,000 total ID+OOD molecule presentations.
- Validation checkpoints: 100 evenly spaced checkpoints.
- Selection rule, independently per cell/backbone/objective:
  1. highest mean validation AUROC over the five seeds;
  2. highest worst-seed validation AUROC;
  3. smallest recipe ID.
- Target test scores are not computed by the screen or selector.

### Pre-final boundary extension

The complete 32-point validation screen selected the same three basins for both
objectives: recipe 13 for EC50-Assay/MiniMol, recipe 24 for IC50-Assay/MiniMol, and
recipe 3 for both Uni-Mol settings.  Recipe 13 lies at both original continuous search
boundaries (`lr=2.93e-3`, `gamma=0.995`).  Before any v2 final test score is computed,
run one fixed 16-point boundary extension for **both RPO and BCE in all four settings**:

- six `batch=512`, z-score, dropout-0.1 recipes spanning the high-lr/high-gamma edge;
- six `batch=64`, z-score, dropout-0.2 recipes spanning the low-lr edge observed on
  IC50-Assay/MiniMol;
- four `batch=128`, Ledoit--Wolf-whitened recipes surrounding recipe 3, which is the
  validation optimum for both Uni-Mol settings;
- all other training code, five selection seeds, presentation budget and 100 validation
  checkpoints remain unchanged.

Final recipe selection is from the union of the original 32 and the extension 16, using
the same mean/worst-seed/recipe-ID rule.  This extension was triggered and fixed from
validation geometry only; no v2 final test result existed when it was added.

## Frozen final evaluation

- Final paired seeds: 301 through 320.
- All twenty seeds are reported; no seed selection.
- Metrics: AUROC, AUPR and conventional FPR95.
- Primary comparison: macro AUROC across the four assay/backbone settings.
- Existing official test splits have been observed in older project experiments, so the
  result is described as a frozen re-evaluation, not as a sealed test.

## Reporting guardrails

- A result may be called the highest mean only if it is compared with baselines evaluated
  on the same split and final seeds.
- Pairwise superiority over BCE requires the paired comparison; differences caused by
  independently selected recipes are additionally labelled as a best-system comparison.
- No test score may be used to alter the recipe grid, selection seeds, checkpoint count,
  cell set, metric, or final seeds.
