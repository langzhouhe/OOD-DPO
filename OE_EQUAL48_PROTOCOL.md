# Supervision-matched OE baselines: frozen equal-48 protocol

Status: implementation frozen for review; no equal-48 training or final evaluation has
been launched as of 2026-08-13.

## Scope and information boundary

- Primary settings: EC50-Assay and IC50-Assay with MiniMol and Uni-Mol (four cells).
- Methods: MSP-OE, ODIN-OE, Energy-OE, and the project-defined two-sample adaptations
  OE-Mahalanobis, OE-KNN, and OE-LOF.
- Each method has exactly 48 unique, method-specific recipe trials per selection seed.
  Each neural recipe evaluates exactly 100 EMA checkpoints on validation, matching the
  RPO-v2 within-recipe checkpoint protocol; a closed-form recipe has no checkpoint axis.
- Selection seeds are 21--25.  The selected recipe maximizes mean validation AUROC; ties
  are resolved by worst-seed validation AUROC and then lower recipe ID.
- Final seeds are 301--320.  Final evaluation cannot run until the five validation files
  have produced a frozen selection artifact.
- Every method and seed uses the RPO-OE-v2 molecule draw:
  `numpy.random.default_rng(seed).permutation`, with at most 1,500 train-ID and 2,000
  auxiliary-OOD molecules.  The exact index hashes and observed labelled-ID count are
  recorded per artifact.
- A screen process materializes only train-ID, train-OOD, val-ID, and val-OOD feature
  arrays.  Test arrays are first materialized by the final process after selection is
  frozen.
- The official benchmark test partitions have been evaluated in earlier project phases;
  these runs are therefore a frozen re-evaluation, not a newly sealed test.  No earlier
  test score was read by `oe_equal48.py`, and no equal-48 recipe can change after final
  execution begins.

The classifier methods use a d-to-256-to-128-to-C MLP with ReLU and dropout 0.1.  Its
hidden trunk matches the paper's RPO/BCE head; only the required C-way output differs.
Each recipe receives a 1,024,000-presentation budget, 5% linear warmup followed by cosine
decay, EMA decay 0.995, and exactly 100 validation checkpoints.  The checkpoint with the
best validation AUROC is retained.  A final-seed run also loads validation, repeats this
checkpoint rule, and evaluates the test partition exactly once after checkpoint choice.
All higher scores mean more OOD.

The matched BCE head remains the decisive objective-isolation control because it shares
RPO's scalar output and exact loss-only architecture.  MSP/ODIN/Energy are detector-family
baselines rather than loss-only ablations, but matching their hidden widths removes an
avoidable capacity disadvantage relative to RPO.

## Fixed grids

Every grid includes both train-ID z-score normalization and Ledoit--Wolf whitening.  The
Cartesian products below include normalization and have exactly 48 members.

| Method | Fixed Cartesian product |
|---|---|
| MSP-OE | normalization (2) x lr {0.003, 0.01, 0.03} x wd {0.0005, 0.005} x OE weight {0.03, 0.1, 0.3, 1.0} |
| ODIN-OE | normalization (2) x lr {0.003, 0.01, 0.03} x OE weight {0.1, 0.5} x temperature {1, 10, 100, 1000}; wd=0.0005 |
| Energy-OE | normalization (2) x lr {0.003, 0.01} x energy weight {0.03, 0.1, 0.3} x margins {(-9,-5), (-7,-5), (-5,-3), (-3,-1)}; wd=0.0005 |
| OE-Mahalanobis | normalization (2) x covariance {shared, separate, diagonal} x epsilon {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100} |
| OE-KNN | normalization (2) x ID k {1, 5, 20, 100} x OOD k {1, 5, 20, 50, 100, 200} |
| OE-LOF | normalization (2) x ID neighbors {5, 20, 50, 200} x OOD neighbors {5, 20, 50, 100, 200, 500} |

ODIN-OE corrects the older protocol's confounding: the OE classifier learning rate and
OE weight are now selected jointly with temperature.  Energy-OE likewise selects its
optimizer learning rate, penalty weight, and ordered in/out energy margins jointly.
For a binary classifier, `max softmax(z/T) = sigmoid(|z1-z0|/T)`, a strictly monotone
transform of the absolute logit margin for every positive T.  Hence all ranking metrics
and the best-AUROC checkpoint are temperature invariant, and the four temperatures may
reuse one training trajectory.  The implementation checks `C==2`; for any multiclass
cell, temperature remains in the cache key and no trajectory is reused.

Frozen hashes generated from canonical sorted JSON:

| Grid | SHA-256 |
|---|---|
| all six grids | `fcecbddefa33b23589cf0c032711eea509e15a207325a6b799519b863182b541` |
| MSP-OE | `40dcd87dc24d2133b8ac5ec05be1f9261a9d52132201e619e4cfba1ca745d0f5` |
| ODIN-OE | `f004377825d4d639d164385c1defce163f4efd5d7d7e3588857c899db1489feb` |
| Energy-OE | `00ae540e772b38810c2f7af42ce3d2568bf21c4038e8a68a370fd74e4755f1bd` |
| OE-Mahalanobis | `b6a6fcb692b1f61561092e853c92759e48d6e26e7e29ad19441d7aaf35f5dbab` |
| OE-KNN | `7a4675feee63134029b79511606974a9a94c2376898a68942d2ff0222273912d` |
| OE-LOF | `cde8fadd4a447b9af759af86d77b6813f39bb3690d38dcbf1a501f154f626823` |

The three `OE-*` feature-space methods are project-defined two-sample adaptations, not
standard named literature methods.  They must remain marked as such in the manuscript.

## Artifacts and commands (do not chain across the frozen boundary)

```bash
/root/miniconda3/envs/ood/bin/python run_oe_equal48.py screen --workers 20
/root/miniconda3/envs/ood/bin/python run_oe_equal48.py select --workers 20
# Review and freeze all selection JSONs before invoking the next line.
/root/miniconda3/envs/ood/bin/python run_oe_equal48.py final --workers 20
/root/miniconda3/envs/ood/bin/python oe_equal48_aggregate.py
```

The screen phase produces 120 JSON files (4 settings x 6 methods x 5 seeds), each with
all 48 recipes and validation AUROC/AUPR/FPR95.  Selection produces 24 JSON files with
screen SHA-256 provenance.  Final produces 480 per-seed JSON files (4 x 6 x 20), each
with AUROC/AUPR/FPR95, frozen-selection SHA-256, recipe, and molecule-index hashes.

## Static cost accounting

- Recipe-level validation trials: 4 x 6 x 5 x 48 = 5,760.
- Neural screen fits: 1,920 MSP/Energy fits plus 240 ODIN classifier fits.  ODIN has 48
  validation recipes but reuses each fitted classifier across its four temperatures.
- Neural final fits: 4 x 3 x 20 = 240.
- Post-hoc screen evaluations: 4 x 3 x 5 x 48 = 2,880; post-hoc final evaluations: 240.
- At full 1,500/2,000 coverage a neural fit has 292 full-batch optimizer steps and
  1,022,000 actual presentations.  Across the 2,400 unique screen/final fits this is
  700,800 optimizer steps and about 2.45 billion molecule presentations.
- Every unique neural fit evaluates exactly 100 EMA checkpoints.  This adds about
  240,000 validation checkpoint evaluations and will dominate wall time on large
  validation partitions.
- Wall time is hardware-dependent; budget substantial CPU-hours with 20 isolated
  workers, or schedule a small number of workers per free GPU.  Do not point 20 workers
  at one GPU.
