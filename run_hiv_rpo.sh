#!/bin/bash
# Full RPO (Energy-DPO) train+eval on the real-HIV size/scaffold shifts.
set -e
PY=/root/miniconda3/envs/ood/bin/python
export HF_ENDPOINT=https://hf-mirror.com TQDM_DISABLE=1
cd /root/autodl-tmp/OOD-DPO

for SHIFT in size scaffold; do
  DS=lbap_general_hiv_${SHIFT}
  OUT=./runs/hiv_${SHIFT}
  echo "############### TRAIN  $DS ###############"
  $PY main.py --mode train \
    --dataset $DS --data_file ./data/raw/${DS}.json \
    --foundation_model minimol \
    --dpo_beta 0.1 --lambda_reg 0.01 \
    --epochs 300 --batch_size 512 --eval_batch_size 256 \
    --lr 1e-4 --eval_steps 25 --early_stopping_patience 50 \
    --cache_root ./cache --output_dir $OUT --num_workers 4 \
    2>&1 | grep -iE "Val-AUROC|Best Val|Final Dataset|train_id|Computing features|Feature encoding progress|Training completed|Error|Traceback" | tail -8
  echo "############### EVAL   $DS ###############"
  $PY main.py --mode eval \
    --dataset $DS --data_file ./data/raw/${DS}.json \
    --foundation_model minimol \
    --model_path $OUT/best_model.pth \
    --cache_root ./cache --output_dir $OUT --eval_batch_size 256 \
    2>&1 | grep -iE "AUROC|AUPR|FPR95|Separation|Mean Energy|Test data|Error|Traceback" | tail -12
  echo
done
echo "ALL DONE"
