#!/bin/bash
# Full matched-OE table (12 cells x 2 backbones) + RPO-Lite capacity sweep
# (7 non-size cells x 2 backbones).  One process per config, OMP pinned to 1 thread.
cd /root/autodl-tmp/OOD-DPO
PY=/root/miniconda3/envs/ood/bin/python
mkdir -p logs repro

ALL="ec50_assay ec50_scaffold ec50_size ic50_assay ic50_scaffold ic50_size \
     hiv_scaffold hiv_size pcba_scaffold pcba_size zinc_scaffold zinc_size"
CAP="ec50_assay ic50_assay ec50_scaffold ic50_scaffold hiv_scaffold pcba_scaffold zinc_scaffold"

for bb in minimol unimol; do
  for c in $ALL; do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup $PY matched_oe.py --cell $c --backbone $bb \
      > logs/moe_${c}_${bb}.log 2>&1 &
  done
done

for bb in minimol unimol; do
  for c in $CAP; do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup $PY capacity_sweep.py --cell $c --backbone $bb \
      > logs/cap_${c}_${bb}.log 2>&1 &
  done
done

wait
echo "ALL_JOBS_FINISHED"
