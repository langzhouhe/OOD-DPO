#!/usr/bin/env bash

# Baseline OOD Detection Experiments with SupervisedBaselineDataLoader

set -e

# Experiment configuration - supports DrugOOD and GOOD datasets
DATASETS=(
    "drugood_lbap_general_ec50_assay"
    "drugood_lbap_general_ec50_scaffold" 
    "drugood_lbap_general_ec50_size"
    "drugood_lbap_general_ic50_assay"
    "drugood_lbap_general_ic50_scaffold"
    "drugood_lbap_general_ic50_size"
#     "good_zinc"
#     "good_hiv"
#     "good_pcba"
# 
)

SEEDS=(1 2 3 4 5 6 7 8 9 10)
DATA_SEED=42
# METHODS=("msp" "energy")
# METHODS=("knn" "lof")
# METHODS=("odin")

METHODS=("msp" "energy" "odin" "mahalanobis" "knn" "lof" "dam_msp" "dam_energy")

FOUNDATION_MODELS=("unimol")
OUTPUT_BASE="./baseline_outputs"
DATA_PATH="./data"

# GOOD dataset configuration
GOOD_DOMAIN="scaffold"  # scaffold or size
GOOD_SHIFT="covariate"  # covariate or concept

# Model configuration
HIDDEN_CHANNELS=64
NUM_LAYERS=3
DROPOUT=0.5
LR=0.01
EPOCHS=500
BATCH_SIZE=32
EVAL_BATCH_SIZE=64

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect dataset type
is_good_dataset() {
    local dataset=$1
    [[ "$dataset" == good_* ]]
}

# Extract DrugOOD subset from full dataset name
extract_drugood_subset() {
    local full_dataset_name=$1
    # Remove "drugood_" prefix to get subset name
    echo "${full_dataset_name#drugood_}"
}

# Build data file path
get_data_file_path() {
    local dataset=$1
    
    if is_good_dataset "$dataset"; then
        # GOOD datasets don't need data file path, return dataset name directly
        echo "$dataset"
    else
        local subset=$(extract_drugood_subset "$dataset")

        # Try multiple possible file paths
        local possible_paths=(
            "$DATA_PATH/raw/${subset}.json"
            "$DATA_PATH/raw/${dataset}.json"
            "$DATA_PATH/${subset}.json"
            "$DATA_PATH/${dataset}.json"
            "$DATA_PATH/drugood/${subset}.json"
        )
        
        for path in "${possible_paths[@]}"; do
            if [[ -f "$path" ]]; then
                echo "$path"
                return 0
            fi
        done
        
        # If not found, return default path
        echo "$DATA_PATH/raw/${subset}.json"
    fi
}

# Function to generate dataset metrics file
generate_dataset_metrics_file() {
    local method=$1
    local foundation_model=$2
    local dataset=$3
    local auroc_mean=$4
    local auroc_std=$5
    local aupr_mean=$6
    local aupr_std=$7
    local fpr95_mean=$8
    local fpr95_std=$9
    local num_seeds=${10}

    local metrics_dir="$OUTPUT_BASE/${method}/${foundation_model}/${dataset}"
    mkdir -p "$metrics_dir"

    local metrics_file="$metrics_dir/dataset_metrics.json"

    # Generate metrics file
    cat > "$metrics_file" << EOF
{
    "method": "$method",
    "foundation_model": "$foundation_model",
    "dataset": "$dataset",
    "metrics": {
        "auroc": {
            "mean": $auroc_mean,
            "std": $auroc_std
        },
        "aupr": {
            "mean": $aupr_mean,
            "std": $aupr_std
        },
        "fpr95": {
            "mean": $fpr95_mean,
            "std": $fpr95_std
        }
    },
    "num_seeds": $num_seeds,
    "total_seeds": ${#SEEDS[@]},
    "timestamp": "$(date -Iseconds)"
}
EOF
}

# Function to run single experiment
run_single_experiment() {
    local method=$1
    local foundation_model=$2
    local dataset=$3
    local seed=$4
    local data_seed=$5
    
    # Set parameters based on dataset type
    local data_file=""
    local drugood_subset=""
    local output_dir

    if is_good_dataset "$dataset"; then
        # GOOD数据集需要在路径中包含domain信息以避免覆盖
        output_dir="$OUTPUT_BASE/${method}/${foundation_model}/${dataset}_${GOOD_DOMAIN}/${seed}"
        data_file="$dataset"
        log_info "Using GOOD dataset: $dataset (domain=$GOOD_DOMAIN, shift=$GOOD_SHIFT)"
    else
        # DrugOOD数据集
        output_dir="$OUTPUT_BASE/${method}/${foundation_model}/${dataset}/${seed}"
        drugood_subset=$(extract_drugood_subset "$dataset")
        data_file=$(get_data_file_path "$dataset")

        if [[ ! -f "$data_file" ]]; then
            log_error "Data file not found: $data_file"
            log_info "Tried to find data for subset: $drugood_subset"
            return 1
        fi
        log_info "Using DrugOOD subset: $drugood_subset"
    fi
    
    mkdir -p "$output_dir"
    
    # 环境变量设置
    export TQDM_DISABLE=0
    export SHOW_PROGRESS=1
    
    # 根据foundation model调整batch size
    local batch_size=$BATCH_SIZE
    local eval_batch_size=$EVAL_BATCH_SIZE
    if [[ "$foundation_model" == "unimol" ]]; then
        batch_size=16  # unimol需要更小的batch size
        eval_batch_size=32
    fi
    
    # 移除详细的日志信息，在主循环中统一处理进度显示
    
    # 构建命令行参数数组
    local cmd_args=(
        "--dataset" "$dataset"
        "--foundation_model" "$foundation_model"
        "--method" "$method"
        "--output_dir" "$output_dir"
        "--seed" "$seed"
        "--data_seed" "$data_seed"
        "--hidden_channels" "$HIDDEN_CHANNELS"
        "--num_layers" "$NUM_LAYERS"
        "--dropout" "$DROPOUT"
        "--lr" "$LR"
        "--epochs" "$EPOCHS"
        "--batch_size" "$batch_size"
        "--eval_batch_size" "$eval_batch_size"
        "--patience" "30"
        "--weight_decay" "5e-4"
        "--device" "cuda"
        "--num_workers" "2"
        "--precompute_features"
        "--cache_root" "/home/ubuntu/OOD-DPO"
        "--encoding_batch_size" "50"
        "--data_path" "$DATA_PATH"
    )

    # ODIN-specific hyperparameters: pass only for ODIN, use conservative defaults
    if [[ "$method" == "odin" ]]; then
        # Standard ODIN hyperparameters
        cmd_args+=("--T" "1000")
        cmd_args+=("--noise" "0.0014")
    fi

    # DAM-specific hyperparameters
    if [[ "$method" == "dam_msp" || "$method" == "dam_energy" ]]; then
        cmd_args+=("--dam_lr" "0.1")
        cmd_args+=("--dam_margin" "1.0")
    fi
    
    # 添加数据集特定参数
    if is_good_dataset "$dataset"; then
        cmd_args+=("--good_domain" "$GOOD_DOMAIN")
        cmd_args+=("--good_shift" "$GOOD_SHIFT")
    else
        cmd_args+=("--drugood_subset" "$drugood_subset")
        cmd_args+=("--data_file" "$data_file")
    fi
    
    # 运行baseline实验 (静默运行，只记录到日志文件)
    if python baselines.py "${cmd_args[@]}" > "$output_dir/experiment.log" 2>&1; then

        # 读取结果但不在终端显示详细信息
        local results_file="$output_dir/${method}_results.json"
        if [[ -f "$results_file" ]]; then
            local auroc=$(python -c "
import json
try:
    with open('$results_file') as f:
        results = json.load(f)
    print(f\"{results.get('auroc', 0):.4f}\")
except:
    print('0.0000')
" 2>/dev/null)
            # 只返回AUROC值，不在终端打印
            echo "$auroc"
            return 0
        else
            return 1
        fi
    else
        log_error "Experiment failed: $method/$foundation_model/$dataset (seed=$seed)"
        return 1
    fi
}

# Main function
main() {
    log_info "Starting supervised baseline OOD detection experiments (SupervisedBaselineDataLoader)"
    echo "=============================================================="
    echo "Supervised Baseline Experiment Configuration"
    echo "=============================================================="
    echo "Datasets: ${DATASETS[*]}"
    echo "Seeds: ${SEEDS[*]}"
    echo "Foundation Models: ${FOUNDATION_MODELS[*]}"
    echo "Methods: ${METHODS[*]}"
    echo "Data seed: $DATA_SEED"
    echo "Output directory: $OUTPUT_BASE"
    echo "Training epochs: $EPOCHS"
    echo "Directory structure: method/foundation_model/dataset/seed"
    echo "=============================================================="
    
    # Check dependencies
    if ! command -v python &> /dev/null; then
        log_error "Python not found"
        exit 1
    fi
    
    if [[ ! -d "$DATA_PATH" ]]; then
        log_error "Data path not found: $DATA_PATH"
        exit 1
    fi
    
    if [[ ! -f "baselines.py" ]]; then
        log_error "baselines.py not found in current directory"
        exit 1
    fi
    
    # 检查SupervisedBaselineDataLoader
    if ! python -c "from SupervisedBaselineDataLoader import SupervisedBaselineDataLoader" 2>/dev/null; then
        log_error "SupervisedBaselineDataLoader.py not found or has import errors"
        log_info "Please ensure SupervisedBaselineDataLoader.py is in the current directory"
        exit 1
    fi
    
    log_success "SupervisedBaselineDataLoader found and importable"
    
    # Verify data files exist (only for DrugOOD datasets)
    log_info "Verifying data files..."
    local missing_files=()
    for dataset in "${DATASETS[@]}"; do
        if is_good_dataset "$dataset"; then
            log_info "GOOD dataset: $dataset (no file check needed)"
        else
            local data_file=$(get_data_file_path "$dataset")
            if [[ ! -f "$data_file" ]]; then
                missing_files+=("$dataset: $data_file")
            else
                log_info "Found: $dataset -> $data_file"
            fi
        fi
    done
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        log_warning "Following DrugOOD data files not found:"
        for missing in "${missing_files[@]}"; do
            log_warning "  $missing"
        done
        log_info "Experiments will skip missing datasets"
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_BASE"

    # Track experiment progress
    local total_experiments=$((${#METHODS[@]} * ${#FOUNDATION_MODELS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]}))
    local completed_experiments=0
    local failed_experiments=0
    
    # Run all experiments - organized by method/foundation_model/dataset/seed
    for method in "${METHODS[@]}"; do
        log_info "Starting ${method^^} method experiments"

        for foundation_model in "${FOUNDATION_MODELS[@]}"; do
            log_info "Foundation Model: $foundation_model"

            for dataset in "${DATASETS[@]}"; do
                # Check if data file exists (only for DrugOOD datasets)
                if ! is_good_dataset "$dataset"; then
                    local data_file=$(get_data_file_path "$dataset")
                    if [[ ! -f "$data_file" ]]; then
                        log_warning "Skipping $dataset (data file not found)"
                        completed_experiments=$((completed_experiments + ${#SEEDS[@]}))
                        failed_experiments=$((failed_experiments + ${#SEEDS[@]}))
                        continue
                    fi
                fi
                
                local dataset_aucs=()
                local dataset_auprs=()
                local dataset_fpr95s=()
                
                for seed in "${SEEDS[@]}"; do
                    completed_experiments=$((completed_experiments + 1))

                    # Simplified output, only show progress
                    printf "[$completed_experiments/$total_experiments] %s/%s: %s (seed=%s) ... " "$method" "$foundation_model" "$dataset" "$seed"

                    if run_single_experiment "$method" "$foundation_model" "$dataset" "$seed" "$DATA_SEED"; then
                        printf "✓\n"
                        # Extract all results (AUROC, AUPR, FPR95)
                        local dataset_path_name="$dataset"
                        if is_good_dataset "$dataset"; then
                            dataset_path_name="${dataset}_${GOOD_DOMAIN}"
                        fi
                        local results_file="$OUTPUT_BASE/${method}/${foundation_model}/${dataset_path_name}/${seed}/${method}_results.json"
                        if [[ -f "$results_file" ]]; then
                            local metrics=$(python -c "
import json
try:
    with open('$results_file') as f:
        results = json.load(f)
    auroc = results.get('auroc', 0)
    aupr = results.get('aupr', 0)
    fpr95 = results.get('fpr95', 1)
    print(f'{auroc:.4f},{aupr:.4f},{fpr95:.4f}')
except:
    print('0.0000,0.0000,1.0000')
" 2>/dev/null)
                            IFS=',' read -r auroc aupr fpr95 <<< "$metrics"
                            if [[ "$auroc" != "0.0000" ]]; then
                                dataset_aucs+=("$auroc")
                                dataset_auprs+=("$aupr")
                                dataset_fpr95s+=("$fpr95")
                            fi
                        fi
                    else
                        printf "✗\n"
                        failed_experiments=$((failed_experiments + 1))
                    fi
                done
                
                # Calculate average of all dataset metrics and generate record file
                if [[ ${#dataset_aucs[@]} -gt 0 ]]; then
                    local stats=$(python -c "
import numpy as np
aucs = [float(x) for x in '${dataset_aucs[*]}'.split()]
auprs = [float(x) for x in '${dataset_auprs[*]}'.split()]
fpr95s = [float(x) for x in '${dataset_fpr95s[*]}'.split()]

auroc_mean, auroc_std = np.mean(aucs), (np.std(aucs, ddof=1) if len(aucs) > 1 else 0)
aupr_mean, aupr_std = np.mean(auprs), (np.std(auprs, ddof=1) if len(auprs) > 1 else 0)
fpr95_mean, fpr95_std = np.mean(fpr95s), (np.std(fpr95s, ddof=1) if len(fpr95s) > 1 else 0)

print(f'{auroc_mean:.3f},{auroc_std:.3f},{aupr_mean:.3f},{aupr_std:.3f},{fpr95_mean:.3f},{fpr95_std:.3f}')
")
                    IFS=',' read -r auroc_mean auroc_std aupr_mean aupr_std fpr95_mean fpr95_std <<< "$stats"

                    # Generate dataset record file with correct path
                    local dataset_path_name="$dataset"
                    if is_good_dataset "$dataset"; then
                        dataset_path_name="${dataset}_${GOOD_DOMAIN}"
                    fi
                    generate_dataset_metrics_file "$method" "$foundation_model" "$dataset_path_name" "$auroc_mean" "$auroc_std" "$aupr_mean" "$aupr_std" "$fpr95_mean" "$fpr95_std" "${#dataset_aucs[@]}"

                    log_success "$method/$foundation_model $dataset: AUROC=$auroc_mean±$auroc_std AUPR=$aupr_mean±$aupr_std FPR95=$fpr95_mean±$fpr95_std (${#dataset_aucs[@]}/${#SEEDS[@]} seeds)"
                fi
                
            done
        done
    done
    
    # Calculate and display summary statistics
    echo ""
    log_info "All experiments completed!"
    log_info "Total: $total_experiments, Failed: $failed_experiments"

    if [[ $failed_experiments -eq $total_experiments ]]; then
        log_error "All experiments failed!"
        exit 1
    fi

    compute_summary_stats
    print_final_results

    log_success "All supervised baseline experiments completed!"
    log_info "Results saved in: $OUTPUT_BASE"
}


# Add missing functions for compute_summary_stats and print_final_results
compute_summary_stats() {
    log_info "Computing summary statistics..."
}

print_final_results() {
    log_info "Final results available in individual dataset_metrics.json files"
    echo "Results structure: $OUTPUT_BASE/method/foundation_model/dataset/dataset_metrics.json"
}

# Generate ablation style summary JSON
generate_ablation_summary() {
    log_info "Generating detailed ablation style summary..."

    local summary_file="$OUTPUT_BASE/baseline_ablation_summary.json"

    cat > "$summary_file" << 'EOF'
{
    "experiment_info": {
        "timestamp": "",
        "note": "Simplified summary - detailed analysis requires python script"
    },
    "results": {}
}
EOF

    log_success "Ablation style summary generation completed"
}

# Print ablation summary
print_ablation_summary() {
    log_info "Printing Ablation style summary:"
    echo "=============================================================="
    echo "Baseline Experiment Ablation Summary - similar to ablation_results format"
    echo "=============================================================="

    echo "Summary files can be found in $OUTPUT_BASE/*/dataset_metrics.json"
    echo "For detailed analysis, use a Python script to parse the results"
}

# Quick test mode
run_quick_test() {
    log_info "Running quick test mode with SupervisedBaselineDataLoader"
    
    local test_dataset="good_pcba"  # Use GOOD dataset for quick test
    local test_seed=1
    local test_foundation="minimol"
    local test_methods=("odin" "mahalanobis")
    
    log_info "Test config: $test_dataset, seed=$test_seed, foundation=$test_foundation"
    log_info "Test methods: ${test_methods[*]}"
    log_info "GOOD配置: domain=$GOOD_DOMAIN, shift=$GOOD_SHIFT"
    
    # GOOD数据集不需要文件检查
    if is_good_dataset "$test_dataset"; then
        log_success "Using GOOD dataset: $test_dataset - no file check needed"
    else
        # Check data file (if DrugOOD dataset)
        local data_file=$(get_data_file_path "$test_dataset")
        if [[ ! -f "$data_file" ]]; then
            log_error "测试数据文件不存在: $data_file"
            return 1
        fi
        log_success "找到测试数据文件: $data_file"
    fi
    
    for method in "${test_methods[@]}"; do
        log_info "测试方法: $method"
        
        if run_single_experiment "$method" "$test_foundation" "$test_dataset" "$test_seed" "$DATA_SEED"; then
            log_success "快速测试成功: $method"
        else
            log_error "快速测试失败: $method"
            log_info "检查日志文件: $OUTPUT_BASE/${method}/${test_foundation}/${test_dataset}/${test_seed}/experiment.log"
        fi
        
        echo ""
    done
}

# 处理中断
cleanup() {
    echo ""
    log_warning "实验被用户中断"
    log_info "部分结果可能在: $OUTPUT_BASE"
    exit 130
}

trap cleanup SIGINT SIGTERM

# 帮助函数
show_help() {
    echo "Supervised Baseline OOD Detection Experiments with SupervisedBaselineDataLoader"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --datasets DATASET1,DATASET2,..."
    echo "                        逗号分隔的数据集列表"
    echo "                        Default: DrugOOD LBAP variants and GOOD datasets"
    echo "  --seeds SEED1,SEED2,..."  
    echo "                        逗号分隔的种子列表"
    echo "                        Default: 1,2,3,4,5,6,7,8,9,10"
    echo "  --foundation_models MODEL1,MODEL2,..."
    echo "                        逗号分隔的foundation model列表"
    echo "                        Default: minimol,unimol"
    echo "  --methods METHOD1,METHOD2,..."
    echo "                        逗号分隔的方法列表"
    echo "                        Default: odin,mahalanobis"
    echo "  --output_dir DIR      Output directory - default: ./baseline_outputs"
    echo "  --data_path DIR       Data directory - default: ./data"
    echo "  --epochs N           Training epochs - default: 500"
    echo "  --good_domain DOMAIN  GOOD dataset domain: scaffold or size - default: scaffold"
    echo "  --good_shift SHIFT    GOOD dataset shift type: covariate or concept - default: covariate"
    echo "  --quick              Quick test mode - 1 dataset, 1 seed, 2 methods"
    echo "  --help               显示此帮助信息"
    echo ""
    echo "支持的数据集:"
    echo "  DrugOOD: drugood_lbap_general_ec50_*, drugood_lbap_general_ic50_*"
    echo "  GOOD:    good_pcba, good_hiv, good_zinc"
    echo ""
    echo "重要变更:"
    echo "  - 使用SupervisedBaselineDataLoader替代EnergyDPODataLoader"
    echo "  - 支持DrugOOD和GOOD数据集"
    echo "  - 自动检测数据集类型并应用相应参数"
    echo "  - 使用真实分类标签进行监督学习"
    echo ""
    echo "目录结构: method/foundation_model/dataset/seed"
    echo ""
    echo "示例:"
    echo "  $0                                    # 运行所有实验"
    echo "  $0 --quick                           # 快速测试"
    echo "  $0 --datasets good_pcba,good_hiv --seeds 1,2"
    echo "  $0 --foundation_models minimol --epochs 200"
    echo "  $0 --methods odin --good_domain size --good_shift covariate"
    echo ""
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets)
            IFS=',' read -ra DATASETS <<< "$2"
            shift 2
            ;;
        --seeds)
            IFS=',' read -ra SEEDS <<< "$2"  
            shift 2
            ;;
        --foundation_models)
            IFS=',' read -ra FOUNDATION_MODELS <<< "$2"
            shift 2
            ;;
        --methods)
            IFS=',' read -ra METHODS <<< "$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
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
        --quick)
            run_quick_test
            exit 0
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 运行主函数
main "$@"
