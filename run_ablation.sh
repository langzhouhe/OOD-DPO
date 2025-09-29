#!/bin/bash

# Energy-DPO 消融实验运行脚本
# 使用方法: ./run_ablation_fixed.sh [选项]

set -e  # 遇到错误立即退出

# 默认参数
# DATASET="drugood"  # DrugOOD类型占位名（非GOOD_*即视作DrugOOD）
# DRUGOOD_SUBSET="lbap_general_ic50_scaffold"  # DrugOOD: IC50 + Scaffold 移位
DATASET="good_zinc"
GOOD_DOMAIN="scaffold"  # 对于GOOD数据集
GOOD_SHIFT="covariate"  # 对于GOOD数据集
FOUNDATION_MODEL="minimol"
EPOCHS=500
BATCH_SIZE=512
DEVICE="cuda"
SEEDS="1 2 3 4 5 6 7 8 9 10"  # 默认较少种子，快速测试
BASE_OUTPUT_DIR="./ablation_results"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --drugood_subset)
            DRUGOOD_SUBSET="$2"
            shift 2
            ;;
        --good_domain)
            GOOD_DOMAIN="$2"
            shift 2
            ;;
        --good_shift)
            GOOD_SHIFT="$2"
            shift 2
            ;;
        --foundation_model)
            FOUNDATION_MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --output_dir)
            BASE_OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip_training)
            SKIP_TRAINING="--skip_training"
            shift
            ;;
        --only_dpo)
            ONLY_LOSS_TYPES="--only_loss_types dpo"
            shift
            ;;
        --only_baselines)
            ONLY_LOSS_TYPES="--only_loss_types bce mse hinge"
            shift
            ;;
        --help|-h)
            echo "Energy-DPO 消融实验运行脚本"
            echo ""
            echo "用法: $0 --dataset DATASET [其他选项]"
            echo ""
            echo "必需参数:"
            echo "  --dataset DATASET           数据集名称 (必需)"
            echo ""
            echo "可选参数:"
            echo "  --drugood_subset SUBSET     DrugOOD子集 (用于DrugOOD数据集)"
            echo "  --good_domain DOMAIN        GOOD域选择 (默认: scaffold, 可选: size)"
            echo "  --good_shift SHIFT          GOOD偏移类型 (默认: covariate)"
            echo "  --foundation_model MODEL    基础模型 (默认: minimol, 可选: unimol)"
            echo "  --epochs EPOCHS             训练轮数 (默认: 50)"
            echo "  --batch_size BATCH_SIZE     批次大小 (默认: 512)"
            echo "  --device DEVICE             设备 (默认: cuda)"
            echo "  --seeds \"SEED1 SEED2 ...\"   随机种子列表 (默认: \"42 123 456\")"
            echo "  --output_dir DIR            输出目录 (默认: ./ablation_results)"
            echo "  --skip_training             跳过训练，只运行评估"
            echo "  --only_dpo                  只运行DPO损失函数"
            echo "  --only_baselines            只运行基线损失函数 (BCE, MSE, Hinge)"
            echo "  --help, -h                  显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  # DrugOOD数据集"
            echo "  $0 --dataset drugood --drugood_subset lbap_general_ic50_scaffold"
            echo "  $0 --dataset drugood --drugood_subset lbap_general_ec50_scaffold --only_baselines"
            echo ""
            echo "  # GOOD数据集"  
            echo "  $0 --dataset good_hiv --good_domain scaffold --good_shift covariate"
            echo ""
            echo "  # 使用Uni-Mol，300轮训练"
            echo "  $0 --dataset drugood --drugood_subset lbap_general_ic50_scaffold --foundation_model unimol --epochs 300"
            echo ""
            echo "  # 只运行基线，使用2个种子"
            echo "  $0 --dataset drugood --drugood_subset lbap_general_ic50_scaffold --only_baselines --seeds \"42 123\""
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$DATASET" ]; then
    echo "❌ 错误: --dataset 参数是必需的"
    echo ""
    echo "常见的数据集选项:"
    echo "  # DrugOOD数据集:"
    echo "  --dataset lbap_general_ec50_scaffold"
    echo "  --dataset assay_general_ec50_assay"
    echo "  --dataset size_general_ec50_size"
    echo ""
    echo "  # GOOD数据集:"
    echo "  --dataset good_hiv"
    echo "  --dataset good_bace"
    echo ""
    echo "使用 --help 查看完整帮助信息"
    exit 1
fi

# 显示配置信息
echo "🎯 Energy-DPO 消融实验配置"
echo "================================"
echo "📊 数据集: $DATASET"
if [ -n "$DRUGOOD_SUBSET" ]; then
    echo "📊 DrugOOD子集: $DRUGOOD_SUBSET"
fi
if [[ "$DATASET" == good_* ]]; then
    echo "📊 GOOD域: $GOOD_DOMAIN"
    echo "📊 GOOD偏移: $GOOD_SHIFT"
fi
echo "🏗️  基础模型: $FOUNDATION_MODEL"
echo "⏰ 训练轮数: $EPOCHS"
echo "📦 批次大小: $BATCH_SIZE"
echo "💻 设备: $DEVICE"
echo "🎲 随机种子: $SEEDS"
echo "📂 输出目录: $BASE_OUTPUT_DIR"
if [ -n "$SKIP_TRAINING" ]; then
    echo "⚠️  跳过训练，只进行评估"
fi
if [ -n "$ONLY_LOSS_TYPES" ]; then
    echo "🎯 损失函数限制: $ONLY_LOSS_TYPES"
fi
echo "================================"
echo ""

# 检查Python环境
echo "🔍 检查Python环境..."
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到Python"
    exit 1
fi

# 检查必要文件
if [ ! -f "run_ablation_study.py" ]; then
    echo "❌ 错误: 未找到 run_ablation_study.py"
    echo "请确保脚本在正确的目录中运行"
    exit 1
fi

if [ ! -f "main.py" ]; then
    echo "❌ 错误: 未找到 main.py"
    echo "请确保脚本在正确的目录中运行"
    exit 1
fi

# 创建输出目录
mkdir -p "$BASE_OUTPUT_DIR"

# 构建Python命令 - 🔥 修复版本
PYTHON_CMD="python run_ablation_study.py"
PYTHON_CMD="$PYTHON_CMD --dataset $DATASET"
PYTHON_CMD="$PYTHON_CMD --data_path ./data/raw"

# 只在有DrugOOD子集时添加
if [ -n "$DRUGOOD_SUBSET" ]; then
    PYTHON_CMD="$PYTHON_CMD --drugood_subset $DRUGOOD_SUBSET"
fi

# 🔥 关键修复: 只在GOOD数据集时添加相关参数
if [[ "$DATASET" == good_* ]]; then
    PYTHON_CMD="$PYTHON_CMD --good_domain $GOOD_DOMAIN"
    PYTHON_CMD="$PYTHON_CMD --good_shift $GOOD_SHIFT"
fi

PYTHON_CMD="$PYTHON_CMD --foundation_model $FOUNDATION_MODEL"
PYTHON_CMD="$PYTHON_CMD --epochs $EPOCHS"
PYTHON_CMD="$PYTHON_CMD --batch_size $BATCH_SIZE"
PYTHON_CMD="$PYTHON_CMD --device $DEVICE"
PYTHON_CMD="$PYTHON_CMD --base_output_dir $BASE_OUTPUT_DIR"
PYTHON_CMD="$PYTHON_CMD --seeds $SEEDS"

if [ -n "$SKIP_TRAINING" ]; then
    PYTHON_CMD="$PYTHON_CMD $SKIP_TRAINING"
fi

if [ -n "$ONLY_LOSS_TYPES" ]; then
    PYTHON_CMD="$PYTHON_CMD $ONLY_LOSS_TYPES"
fi

# 记录开始时间
START_TIME=$(date)
echo "🚀 开始消融实验..."
echo "⏰ 开始时间: $START_TIME"
echo ""
echo "🔧 执行命令: $PYTHON_CMD"
echo ""

# 运行Python脚本
if eval $PYTHON_CMD; then
    END_TIME=$(date)
    echo ""
    echo "🎉 消融实验完成!"
    echo "⏰ 开始时间: $START_TIME"
    echo "⏰ 结束时间: $END_TIME"
    echo "📊 结果保存在: $BASE_OUTPUT_DIR"
    echo ""
    echo "📋 快速查看结果:"
    echo "   find $BASE_OUTPUT_DIR -name 'ablation_summary.json' -exec cat {} \;"
    echo ""
else
    echo "❌ 消融实验失败!"
    exit 1
fi
