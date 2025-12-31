#!/bin/bash

# Quick test script to verify finetuning implementation works
# This runs a minimal test (1 epoch) to check everything is functional

set -e

echo "=========================================="
echo "Testing Uni-Mol Finetuning Implementation"
echo "=========================================="

DATASET="lbap_general_ec50_scaffold"
DATA_FILE="./data/raw/${DATASET}.json"

# Check if data file exists
if [ ! -f "${DATA_FILE}" ]; then
    echo "Error: Data file not found: ${DATA_FILE}"
    echo "Please ensure the dataset is available before running this test."
    exit 1
fi

echo ""
echo "Test 1: Frozen encoder (baseline, 1 epoch)..."
python main.py \
    --mode train \
    --dataset ${DATASET} \
    --data_file ${DATA_FILE} \
    --foundation_model unimol \
    --epochs 1 \
    --batch_size 512 \
    --eval_batch_size 256 \
    --lr 1e-4 \
    --seed 42 \
    --data_seed 42 \
    --device cuda \
    --output_dir ./outputs/test_frozen \
    --encoding_batch_size 2000 \
    --log_level INFO \
    --eval_steps 50

echo ""
echo "✓ Test 1 passed: Frozen encoder works"
echo ""

echo "Test 2: Finetunable encoder (1 epoch)..."
python main.py \
    --mode train \
    --dataset ${DATASET} \
    --data_file ${DATA_FILE} \
    --foundation_model unimol \
    --finetune_encoder \
    --encoder_lr 5e-6 \
    --freeze_layers "0-12" \
    --epochs 1 \
    --batch_size 128 \
    --eval_batch_size 256 \
    --lr 1e-4 \
    --seed 42 \
    --data_seed 42 \
    --device cuda \
    --output_dir ./outputs/test_finetuned \
    --encoding_batch_size 2000 \
    --log_level INFO \
    --eval_steps 50

echo ""
echo "✓ Test 2 passed: Finetunable encoder works"
echo ""

echo "=========================================="
echo "✓✓✓ All tests passed! ✓✓✓"
echo "=========================================="
echo ""
echo "Your finetuning implementation is working correctly!"
echo "You can now run the full experiment:"
echo "  bash run_finetune_comparison.sh"
