#!/bin/bash
# Reference-Anchored RPO, frozen method: 3 assay endpoints x 2 backbones.
# ki_assay is included as the post-freeze confirmation endpoint (NOT sealed -- it was
# opened during the batchsel work; see report section 13).
cd /root/autodl-tmp/OOD-DPO
PY=/root/miniconda3/envs/ood/bin/python
mkdir -p logs repro
for bb in minimol unimol; do
  for c in ec50_assay ic50_assay ki_assay; do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup $PY reference_rpo.py --cell $c --backbone $bb \
      > logs/refrpo_${c}_${bb}.log 2>&1 &
  done
done
wait
echo "REFRPO_ALL_FINISHED"
