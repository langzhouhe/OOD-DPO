#!/bin/bash
# P0 rebuttal control: BCE binary ID/OOD classifier vs RPO (pairwise DPO),
# MATCHED — same frozen encoder, same head, same ID/OOD splits, same tuning
# budget (lr/lambda/epochs/optimizer/val-based selection). Only the loss differs.
# Reuses cached MiniMol features, so each cell just retrains the head.
BB=${1:-minimol}; SEED=${2:-1}; EPOCHS=${3:-500}
BATCH=512; [ "$BB" = "unimol" ] && BATCH=256
PY=/root/miniconda3/envs/ood/bin/python
export HF_ENDPOINT=https://hf-mirror.com TQDM_DISABLE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /root/autodl-tmp/OOD-DPO

run_drug () {
  local SUBSET=$1; local OUT=./repro_bce/$BB/$SUBSET/seed$SEED; mkdir -p $OUT
  $PY main.py --mode train --dataset $SUBSET --drugood_subset $SUBSET --data_file ./data/raw/$SUBSET.json \
    --foundation_model $BB --loss_type bce --lambda_reg 0.01 --lr 1e-4 --epochs $EPOCHS \
    --batch_size $BATCH --eval_batch_size 256 --seed $SEED --data_seed 42 \
    --cache_root ./cache --output_dir $OUT --num_workers 4 > $OUT/train.log 2>&1
  $PY main.py --mode eval --dataset $SUBSET --drugood_subset $SUBSET --data_file ./data/raw/$SUBSET.json \
    --foundation_model $BB --model_path $OUT/best_model.pth --seed $SEED --data_seed 42 \
    --cache_root ./cache --output_dir $OUT --eval_batch_size 256 > $OUT/eval.log 2>&1
  local A=$($PY -c "import json;d=json.load(open('$OUT/ood_evaluation_results.json'));print(f\"{d['auroc']:.4f}\")" 2>/dev/null)
  echo "DONE $BB $SUBSET seed$SEED : BCE_AUROC = $A"
}
run_good () {
  local DS=$1; local DOM=$2; local OUT=./repro_bce/$BB/good_${DS}_${DOM}/seed$SEED; mkdir -p $OUT
  $PY main.py --mode train --dataset good_$DS --good_domain $DOM --good_shift covariate --data_path ./data \
    --foundation_model $BB --loss_type bce --lambda_reg 0.01 --lr 1e-4 --epochs $EPOCHS \
    --batch_size $BATCH --eval_batch_size 256 --seed $SEED --data_seed 42 \
    --cache_root ./cache --output_dir $OUT --num_workers 4 > $OUT/train.log 2>&1
  $PY main.py --mode eval --dataset good_$DS --good_domain $DOM --good_shift covariate --data_path ./data \
    --foundation_model $BB --model_path $OUT/best_model.pth --seed $SEED --data_seed 42 \
    --cache_root ./cache --output_dir $OUT --eval_batch_size 256 > $OUT/eval.log 2>&1
  local A=$($PY -c "import json;d=json.load(open('$OUT/ood_evaluation_results.json'));print(f\"{d['auroc']:.4f}\")" 2>/dev/null)
  echo "DONE $BB good_${DS}_${DOM} seed$SEED : BCE_AUROC = $A"
}

for S in ec50 ic50; do for SH in scaffold size assay; do run_drug lbap_general_${S}_${SH} & done; done
for DS in hiv pcba zinc; do for DOM in scaffold size; do run_good $DS $DOM & done; done
wait
echo "ALL_BCE_DONE $BB seed$SEED"
