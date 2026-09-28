#!/bin/bash
# Parallel RPO train+eval over the 6 GOOD cells (hiv/pcba/zinc x scaffold/size covariate) for one backbone.
# Usage: bash run_good_backbone.sh <minimol|unimol> <seed> <epochs>
BB=${1:-minimol}; SEED=${2:-1}; EPOCHS=${3:-500}
BATCH=512; [ "$BB" = "unimol" ] && BATCH=256
PY=/root/miniconda3/envs/ood/bin/python
export HF_ENDPOINT=https://hf-mirror.com TQDM_DISABLE=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd /root/autodl-tmp/OOD-DPO

run_cell () {
  local DS=$1; local DOM=$2
  local NAME=${DS}_${DOM}
  local OUT=./repro/$BB/good_$NAME/seed$SEED
  mkdir -p $OUT
  $PY main.py --mode train --dataset good_$DS --good_domain $DOM --good_shift covariate \
    --data_path ./data --foundation_model $BB \
    --dpo_beta 0.1 --lambda_reg 0.01 --lr 1e-4 --epochs $EPOCHS \
    --batch_size $BATCH --eval_batch_size 256 --seed $SEED --data_seed 42 \
    --cache_root ./cache --output_dir $OUT --num_workers 4 > $OUT/train.log 2>&1
  $PY main.py --mode eval --dataset good_$DS --good_domain $DOM --good_shift covariate \
    --data_path ./data --foundation_model $BB \
    --model_path $OUT/best_model.pth --seed $SEED --data_seed 42 \
    --cache_root ./cache --output_dir $OUT --eval_batch_size 256 > $OUT/eval.log 2>&1
  local A=$($PY -c "import json;d=json.load(open('$OUT/ood_evaluation_results.json'));print(f\"{d['auroc']:.4f} {d['aupr']:.4f} {d['fpr95']:.4f}\")" 2>/dev/null)
  echo "DONE $BB good_$NAME seed$SEED : AUROC/AUPR/FPR95 = $A"
}

for DS in hiv pcba zinc; do for DOM in scaffold size; do
  run_cell $DS $DOM &
done; done
wait
echo "ALL_GOOD_CELLS_DONE $BB seed$SEED"
