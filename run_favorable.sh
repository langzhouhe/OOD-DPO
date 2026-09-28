#!/bin/bash
# Axis A (head capacity) x Axis B (OE difficulty) x Axis C (selection rule)
# on the two clean assay cells, MiniMol.  Concurrency capped so this coexists with the
# matched-OE / capacity jobs already running.
cd /root/autodl-tmp/OOD-DPO
PY=/root/miniconda3/envs/ood/bin/python
mkdir -p logs repro

for c in ec50_assay ic50_assay; do
  for w in 4 8 16 32 128 full; do
    for t in near medium far random all; do
      echo "$c $w $t"
    done
  done
done | xargs -P 40 -L 1 bash -c '
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/miniconda3/envs/ood/bin/python favorable.py \
    --cell $0 --backbone minimol --width $1 --tier $2 > logs/fav_$0_h$1_$2.log 2>&1'

echo "FAVORABLE_ALL_FINISHED"
