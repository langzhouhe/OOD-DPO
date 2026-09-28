# Pairwise-Hinge / RankSVM OE v2: frozen protocol

This protocol is matched to `RPO_OE_TUNING_V2_PROTOCOL.md`.  The validation
screen and selector have no target-test code path.  The final evaluator refuses
to run until its cell/backbone-specific selection artifact exists.

## Estimator and data

- Cells: EC50-Assay and IC50-Assay.
- Backbones: MiniMol and Uni-Mol.
- Encoder: the same frozen cached representation used by RPO-Opt v2.
- Head: the unchanged manuscript MLP, `d -> 256 -> 128 -> 1`.
- OE molecules: at most 1,500 ID and 2,000 OOD molecules, selected by the same
  seeded NumPy permutations as RPO-Opt v2.
- Pairwise-Hinge objective (complete within-minibatch U-statistic):

  `mean_{i,o} max(0, 1 - (s_o - s_i))`.

The RankSVM margin is fixed to 1.0.  This is the conventional unit-margin loss
and matches RPO's fixed unit inverse-temperature.  It is deliberately not
tuned: adding a margin grid would give Hinge more validation trials than RPO
and BCE.  Score-scale regularization `gamma`, normalization, batch size,
learning rate, weight decay, and dropout use the exact same 48 recipes.

## Validation-only selection

- Recipes: the frozen union of the original 32 Sobol recipes and 16 boundary
  extension recipes used by RPO-Opt v2.
- Expected recipe-union SHA256:
  `20d3841c1d3800f2bf8693c36d34482f7e59aefd8c2c7bbd01b64a0ac28f28de`.
- Selection seeds: 21, 22, 23, 24, 25.
- Per-run budget: 1,024,000 total ID+OOD molecule presentations.
- Validation checkpoints: 100, with the same EMA and validation-AUROC
  checkpoint rule as RPO-Opt v2.
- Per cell/backbone selection order: highest five-seed mean validation AUROC,
  then highest worst-seed AUROC, then smallest recipe ID.

## Frozen final evaluation

- Final seeds: 301 through 320; every seed is reported.
- Metrics: AUROC, AUPR, and conventional FPR95.
- The final evaluator verifies both the frozen recipe hash and the SHA256 of the
  selection artifact before training.
- No test result may alter the recipe set, margin, seed lists, checkpoint count,
  or selection rule.

