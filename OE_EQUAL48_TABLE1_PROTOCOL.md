# Equal-48 Table 1 extension (frozen before the additional runs)

This extension applies the already frozen `oe-equal48-supervision-matched-v2`
training and selection protocol to every column of the manuscript's Table 1.
It does not change the four-setting primary estimand or any recipe grid.

## Scope

- Cells, in manuscript order: EC50 Scaffold/Size/Assay; IC50
  Scaffold/Size/Assay; HIV Scaffold/Size; PCBA Scaffold/Size; ZINC
  Scaffold/Size.
- Backbones: MiniMol and Uni-Mol.
- Table 1 methods: MSP-OE, ODIN-OE (temperature-only frozen-feature
  adaptation), and Energy-OE.
- The four already completed assay/backbone settings are reused byte for byte.
- Remaining work: 20 settings x 3 methods x 5 selection seeds, followed by
  frozen selection and 20 final seeds.

## Frozen fairness contract

Every method retains its existing 48-point method-specific recipe grid, the same
1,500 ID and 2,000 auxiliary-OOD draw per paired seed, 1,024,000 molecule
presentations for trainable classifiers, 100 validation checkpoints, five
selection seeds (21--25), and twenty final seeds (301--320). Model selection uses
validation AUROC only. All twelve columns are reported; no cell is selected or
removed based on its final score.

Balanced BCE and pairwise hinge are objective-isolation controls and remain in a
separate ablation table rather than being relabeled as Table 1 scoring baselines.

No additional recipe, split, seed, score orientation, or reporting rule may be
introduced after any new Table 1 final score is produced.
