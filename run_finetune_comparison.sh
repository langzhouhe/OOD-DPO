#!/bin/bash

# Uni-Mol Full Finetuning + Energy-DPO
# 运行“微调 Uni-Mol 全部层（不冻结） + Energy-DPO”，并自动评估

set -e  # Exit on error

# ============================================
# 实验配置
# ============================================

# 默认同时跑 scaffold / assay 两个数据集；也可通过环境变量覆盖：
#   DATASETS="lbap_general_ec50_scaffold lbap_general_ec50_assay"
#   或兼容旧的 DATASET 单一配置
DATASETS_DEFAULT="lbap_general_ec50_assay"
DATASETS_STR="${DATASETS:-${DATASET:-$DATASETS_DEFAULT}}"
read -r -a DATASETS <<< "${DATASETS_STR}"

FOUNDATION_MODEL="unimol"
LOSS_TYPE="dpo"

# 训练参数
EPOCHS=100
LR=1e-4
ENCODER_LR=5e-6
DPO_BETA=0.1
HIDDEN_DIM=256
WEIGHT_DECAY=1e-5
LAMBDA_REG=0.01

# Batch sizes（全量微调更占显存，默认更小，可按需调大）
FINETUNE_BATCH_SIZE=${FINETUNE_BATCH_SIZE:-4}
FINETUNE_EVAL_BATCH_SIZE=${FINETUNE_EVAL_BATCH_SIZE:-8}

# Encoding batch size（大分子/全量微调用小一点；有余量可升到 200/500）
ENCODING_BATCH_SIZE=${ENCODING_BATCH_SIZE:-100}
NUM_WORKERS=${NUM_WORKERS:-16}

# Reduce CUDA fragmentation when retrying after large jobs
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}

# Seeds（默认 10 个种子，可通过 SEEDS 环境变量覆盖，例如：SEEDS="0 1 2 ... 9"）
SEED_LIST="${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}"
read -r -a SEEDS <<< "${SEED_LIST}"
DATA_SEED=${DATA_SEED:-42}

# 输出目录
OUTPUT_ROOT="./outputs/finetune_comparison"
mkdir -p ${OUTPUT_ROOT}

echo "=========================================="
echo "全量微调 Uni-Mol + Energy-DPO"
echo "=========================================="
echo "Datasets: ${DATASETS[*]}"
echo "Seeds:    ${SEEDS[*]}"
echo "Output:   ${OUTPUT_ROOT}"
echo "=========================================="

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "Dataset: ${DATASET}"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    DATA_FILE="./data/raw/${DATASET}.json"
    if [ ! -f "${DATA_FILE}" ]; then
        echo "[WARN] Data file not found: ${DATA_FILE} -- skipping"
        continue
    fi

    DATASET_OUTPUT_ROOT="${OUTPUT_ROOT}/${DATASET}"
    mkdir -p "${DATASET_OUTPUT_ROOT}"

    for SEED in "${SEEDS[@]}"; do
        echo "Running seed=${SEED} on ${DATASET}..."

        OUTPUT_DIR="${DATASET_OUTPUT_ROOT}/finetuned_seed${SEED}"

        # ============ 训练（全量微调） ============
        # 说明：main.py 会在 finetune + train 场景自动关闭特征缓存，避免重复 compute features。
        python main.py \
            --mode train \
            --dataset ${DATASET} \
            --data_file ${DATA_FILE} \
            --foundation_model ${FOUNDATION_MODEL} \
            --loss_type ${LOSS_TYPE} \
            --finetune_encoder \
            --encoder_lr ${ENCODER_LR} \
            --freeze_layers "none" \
            --epochs ${EPOCHS} \
            --batch_size ${FINETUNE_BATCH_SIZE} \
            --eval_batch_size ${FINETUNE_EVAL_BATCH_SIZE} \
            --lr ${LR} \
            --dpo_beta ${DPO_BETA} \
            --hidden_dim ${HIDDEN_DIM} \
            --weight_decay ${WEIGHT_DECAY} \
            --lambda_reg ${LAMBDA_REG} \
            --seed ${SEED} \
            --data_seed ${DATA_SEED} \
            --device cuda \
            --output_dir ${OUTPUT_DIR} \
            --early_stopping_patience 10 \
            --eval_steps 100 \
            --cache_root /home/ubuntu/projects \
            --encoding_batch_size ${ENCODING_BATCH_SIZE} \
            --log_level INFO

        echo "Seed ${SEED} training completed. Results saved to ${OUTPUT_DIR}"
        echo "Running evaluation for seed=${SEED}..."

        # ============ 评估（使用 best_model，预计算特征一次） ============
        python main.py \
            --mode eval \
            --dataset ${DATASET} \
            --data_file ${DATA_FILE} \
            --foundation_model ${FOUNDATION_MODEL} \
            --loss_type ${LOSS_TYPE} \
            --finetune_encoder \
            --freeze_layers "none" \
            --model_path ${OUTPUT_DIR}/best_model.pth \
            --output_dir ${OUTPUT_DIR} \
            --seed ${SEED} \
            --data_seed ${DATA_SEED} \
            --device cuda \
            --precompute_features \
            --encoding_batch_size ${ENCODING_BATCH_SIZE} \
            --eval_batch_size ${FINETUNE_EVAL_BATCH_SIZE} \
            --num_workers ${NUM_WORKERS} \
            --graph_cache_workers ${NUM_WORKERS} \
            --cache_root /home/ubuntu/projects \
            --log_level INFO

        echo ""
    done
done

echo "实验跑完，开始汇总并计算均值/STD..."
SUMMARY_FILE="${OUTPUT_ROOT}/comparison_summary.txt"

OUTPUT_ROOT="${OUTPUT_ROOT}" \
SUMMARY_FILE="${SUMMARY_FILE}" \
DATASETS_STR="${DATASETS_STR}" \
SEED_LIST="${SEED_LIST}" \
python - <<'PY'
import datetime
import glob
import json
import os
import statistics

output_root = os.environ["OUTPUT_ROOT"]
summary_file = os.environ["SUMMARY_FILE"]
datasets = os.environ["DATASETS_STR"].split()
seed_list = os.environ.get("SEED_LIST", "")

def fmt(val):
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.4f}"
    except Exception:
        return "N/A"

with open(summary_file, "w") as out:
    out.write("Uni-Mol Full Finetuning (Energy-DPO)\n")
    out.write(f"Generated at: {datetime.datetime.now()}\n")
    out.write(f"Datasets: {', '.join(datasets)}\n")
    out.write(f"Seeds: {seed_list}\n")
    out.write("=" * 40 + "\n\n")

    for dataset in datasets:
        dataset_root = os.path.join(output_root, dataset)
        pattern = os.path.join(dataset_root, "finetuned_seed*/test_metrics.json")
        metric_files = sorted(glob.glob(pattern))

        out.write(f"Dataset: {dataset}\n")

        if not metric_files:
            out.write("  No metrics found.\n\n")
            continue

        records = []
        for path in metric_files:
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                seed = os.path.basename(os.path.dirname(path)).replace("finetuned_seed", "")
                records.append((seed, data))
            except Exception:
                continue

        if not records:
            out.write("  No valid metrics parsed.\n\n")
            continue

        # Sort seeds numerically when possible
        def sort_key(item):
            seed, _ = item
            return int(seed) if seed.isdigit() else seed

        records = sorted(records, key=sort_key)
        out.write(f"  Seeds ({len(records)}): {', '.join(seed for seed, _ in records)}\n")

        def collect(key):
            vals = []
            for _, data in records:
                val = data.get(key)
                if val is not None:
                    vals.append(float(val))
            return vals

        def stats(vals):
            if not vals:
                return None, None
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return mean, std

        for seed, data in records:
            out.write(
                f"    Seed {seed}: AUROC={fmt(data.get('auroc'))}, "
                f"AUPR={fmt(data.get('aupr'))}, FPR95={fmt(data.get('fpr95'))}\n"
            )

        auroc_mean, auroc_std = stats(collect("auroc"))
        aupr_mean, aupr_std = stats(collect("aupr"))
        fpr95_mean, fpr95_std = stats(collect("fpr95"))

        out.write("  Aggregate:\n")
        out.write(f"    AUROC: {fmt(auroc_mean)} ± {fmt(auroc_std)}\n")
        out.write(f"    AUPR : {fmt(aupr_mean)} ± {fmt(aupr_std)}\n")
        out.write(f"    FPR95: {fmt(fpr95_mean)} ± {fmt(fpr95_std)}\n")
        out.write("\n")

print(f"Summary saved to {summary_file}")
PY

cat ${SUMMARY_FILE}

echo ""
echo "=========================================="
echo "所有实验完成！"
echo "=========================================="
echo "结果目录: ${OUTPUT_ROOT}"
echo "汇总文件: ${SUMMARY_FILE}"
