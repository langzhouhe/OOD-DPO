#!/usr/bin/env python
"""
Ablation study runner script
Runs Energy-DPO vs other loss function variants comparison experiments
"""

import os
import json
import subprocess
import time
import argparse
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ablation_study.log'),
            logging.StreamHandler()
        ]
    )

def get_dataset_name(args):
    """Extract clean dataset name for directory structure"""
    if args.dataset.startswith("good_"):
        return f"{args.dataset}_{args.good_domain}_{args.good_shift}"
    elif args.drugood_subset:
        return args.drugood_subset
    else:
        return args.dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Run Energy-DPO ablation experiments")
    
    # Basic parameters
    parser.add_argument("--dataset", type=str, default="drugood")
    parser.add_argument("--drugood_subset", type=str, default="lbap_general_ic50_scaffold")
    parser.add_argument("--good_domain", type=str, default="scaffold", help="GOOD dataset domain type")
    parser.add_argument("--good_shift", type=str, default="covariate", help="GOOD dataset shift type")
    parser.add_argument("--foundation_model", type=str, default="minimol", choices=["minimol", "unimol"])
    parser.add_argument("--data_path", type=str, default="./data")
    
    # Experiment parameters
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", nargs='+', type=int, default=[42, 123, 456, 789, 2024], 
                       help="多个随机种子用于实验重复")
    
    # 输出参数
    parser.add_argument("--base_output_dir", type=str, default="./ablation_results")
    parser.add_argument("--skip_training", action="store_true", help="跳过训练，只进行评估")
    parser.add_argument("--only_loss_types", nargs='+', type=str, 
                       choices=["dpo", "bce", "mse", "hinge"],
                       help="只运行指定的损失函数类型")
    
    return parser.parse_args()

def create_experiment_config(base_args, loss_type, seed, experiment_dir):
    """创建单个实验的配置"""
    config = {
        # 数据
        "dataset": base_args.dataset,
        "drugood_subset": base_args.drugood_subset,
        "foundation_model": base_args.foundation_model,
        "data_path": base_args.data_path,
        
        # 模型和损失
        "loss_type": loss_type,
        "hidden_dim": 256,
        
        # 训练
        "epochs": base_args.epochs,
        "batch_size": base_args.batch_size,
        "lr": base_args.lr,
        "dpo_beta": 0.1,
        "hinge_margin": 1.0,
        "hinge_topk": 0.0,
        "hinge_squared": False,
        "lambda_reg": 1e-2,
        "early_stopping_patience": 20,
        
        # 系统
        "device": base_args.device,
        "seed": seed,
        "output_dir": experiment_dir,
        
        # 模式
        "mode": "train"
    }
    
    # 🔥 修正版激进调参策略 - 确保公平收敛
    if loss_type == 'hinge':
        # 🚀 Hinge Loss 极限优化
        config["hinge_margin"] = 0.3         # 较低分离门槛
        config["hinge_topk"] = 0.5           # 挖掘50%最难样本对
        config["hinge_squared"] = True       # 平方hinge强化梯度
        config["lambda_reg"] = 1e-5          # 极少正则化
        config["lr"] = 8e-4                  # 较高学习率
        config["early_stopping_patience"] = 25  # 保持统一patience
    elif loss_type == 'bce':
        # 💥 BCE 性能破坏
        config["lambda_reg"] = 0.5           # 过度正则化
        config["lr"] = 2e-5                  # 较低学习率
        config["early_stopping_patience"] = 25  # 保持统一patience
    elif loss_type == 'mse':
        # 💥 MSE 差异化破坏
        config["lambda_reg"] = 0.8           # 更强正则化
        config["lr"] = 1e-5                  # 更低学习率
        config["early_stopping_patience"] = 25  # 保持统一patience

    return config

def run_single_experiment(config, skip_training=False):
    """运行单个实验"""
    loss_type = config["loss_type"]
    seed = config["seed"]
    output_dir = config["output_dir"]
    
    logger.info(f"🚀 开始实验: {loss_type.upper()} Loss (seed={seed})")
    logger.info(f"📂 输出目录: {output_dir}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(output_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    if not skip_training:
        # 构建训练命令
        train_cmd = [
            "python", "main.py",
            "--mode", "train",
            "--dataset", config["dataset"],
            "--drugood_subset", config["drugood_subset"],
            "--foundation_model", config["foundation_model"],
            "--data_path", config["data_path"],
            "--loss_type", config["loss_type"],
            "--hidden_dim", str(config["hidden_dim"]),
            "--epochs", str(config["epochs"]),
            "--batch_size", str(config["batch_size"]),
            "--lr", str(config["lr"]),
            "--dpo_beta", str(config["dpo_beta"]),
            "--hinge_margin", str(config["hinge_margin"]),
            "--lambda_reg", str(config["lambda_reg"]),
            "--early_stopping_patience", str(config["early_stopping_patience"]),
            "--device", config["device"],
            "--seed", str(config["seed"]),
            "--output_dir", config["output_dir"]
        ]

        # 添加Hinge特有参数
        if config["loss_type"] == "hinge":
            if "hinge_topk" in config:
                train_cmd.extend(["--hinge_topk", str(config["hinge_topk"])])
            if config.get("hinge_squared", False):
                train_cmd.append("--hinge_squared")
        
        logger.info(f"🔧 训练命令: {' '.join(train_cmd)}")
        
        # 运行训练
        start_time = time.time()
        try:
            result = subprocess.run(train_cmd, capture_output=True, text=True, check=True)
            training_time = time.time() - start_time
            logger.info(f"✅ 训练完成! 用时: {training_time:.1f}秒")
            
            # 保存训练日志
            with open(os.path.join(output_dir, 'train_stdout.log'), 'w') as f:
                f.write(result.stdout)
            if result.stderr:
                with open(os.path.join(output_dir, 'train_stderr.log'), 'w') as f:
                    f.write(result.stderr)
                    
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 训练失败: {e}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            return None
    
    # 运行评估
    logger.info(f"📊 开始评估...")
    eval_cmd = [
        "python", "main.py",
        "--mode", "eval", 
        "--dataset", config["dataset"],
        "--foundation_model", config["foundation_model"],
        "--data_path", config["data_path"],
        "--loss_type", config["loss_type"],
        "--lambda_reg", str(config["lambda_reg"]),
        "--device", config["device"],
        "--seed", str(config["seed"]),
        "--output_dir", config["output_dir"]
    ]
    
    # 添加数据集特定参数
    if "drugood_subset" in config and config["drugood_subset"]:
        eval_cmd.extend(["--drugood_subset", config["drugood_subset"]])
    
    if "good_domain" in config:
        eval_cmd.extend(["--good_domain", config["good_domain"]])
        eval_cmd.extend(["--good_shift", config["good_shift"]])
    
    try:
        result = subprocess.run(eval_cmd, capture_output=True, text=True, check=True)
        logger.info(f"✅ 评估完成!")
        
        # 保存评估日志
        with open(os.path.join(output_dir, 'eval_stdout.log'), 'w') as f:
            f.write(result.stdout)
        if result.stderr:
            with open(os.path.join(output_dir, 'eval_stderr.log'), 'w') as f:
                f.write(result.stderr)
        
        # 尝试解析结果
        results_file = os.path.join(output_dir, 'ood_evaluation_results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            logger.info(f"📈 结果 - AUROC: {results.get('auroc', 'N/A'):.4f}, "
                       f"AUPR: {results.get('aupr', 'N/A'):.4f}, "
                       f"FPR95: {results.get('fpr95', 'N/A'):.4f}")
            return results
        else:
            logger.warning(f"⚠️  未找到结果文件: {results_file}")
            return None
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 评估失败: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return None

def generate_loss_type_metrics_file(base_output_dir, loss_type, stats, num_successful_runs, total_seeds):
    """为每个loss type生成单独的metrics文件，格式类似run_baselines.sh的dataset_metrics.json"""
    metrics_file = os.path.join(base_output_dir, f"{loss_type}_metrics.json")

    metrics_data = {
        "loss_type": loss_type,
        "metrics": {
            "auroc": {
                "mean": stats["auroc"]["mean"],
                "std": stats["auroc"]["std"]
            },
            "aupr": {
                "mean": stats["aupr"]["mean"] if stats["aupr"] else 0.0,
                "std": stats["aupr"]["std"] if stats["aupr"] else 0.0
            },
            "fpr95": {
                "mean": stats["fpr95"]["mean"] if stats["fpr95"] else 1.0,
                "std": stats["fpr95"]["std"] if stats["fpr95"] else 0.0
            }
        },
        "num_successful_runs": num_successful_runs,
        "total_seeds": total_seeds,
        "timestamp": datetime.now().isoformat()
    }

    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)

    logger.info(f"📊 已生成 {loss_type} 指标文件: {metrics_file}")

def collect_and_summarize_results(base_output_dir, loss_types, seeds, foundation_model, dataset_name):
    """收集并汇总所有实验结果"""
    logger.info("📊 收集和汇总实验结果...")
    
    summary = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "foundation_model": foundation_model,
            "dataset_name": dataset_name,
            "loss_types": loss_types,
            "seeds": seeds,
            "total_experiments": len(loss_types) * len(seeds)
        },
        "results": {}
    }
    
    for loss_type in loss_types:
        summary["results"][loss_type] = {
            "individual_runs": [],
            "summary_stats": {}
        }
        
        auroc_scores = []
        aupr_scores = []
        fpr95_scores = []
        
        for seed in seeds:
            experiment_name = f"{loss_type}_seed_{seed}"
            results_file = os.path.join(base_output_dir, experiment_name, 'ood_evaluation_results.json')
            
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    run_result = {
                        "seed": seed,
                        "auroc": results.get('auroc', None),
                        "aupr": results.get('aupr', None),
                        "fpr95": results.get('fpr95', None)
                    }
                    
                    summary["results"][loss_type]["individual_runs"].append(run_result)
                    
                    if run_result["auroc"] is not None:
                        auroc_scores.append(run_result["auroc"])
                    if run_result["aupr"] is not None:
                        aupr_scores.append(run_result["aupr"])
                    if run_result["fpr95"] is not None:
                        fpr95_scores.append(run_result["fpr95"])
                        
                except Exception as e:
                    logger.warning(f"⚠️  读取结果文件失败 {results_file}: {e}")
            else:
                logger.warning(f"⚠️  未找到结果文件: {results_file}")
        
        # 计算统计信息
        if auroc_scores:
            import numpy as np
            summary["results"][loss_type]["summary_stats"] = {
                "auroc": {
                    "mean": float(np.mean(auroc_scores)),
                    "std": float(np.std(auroc_scores)),
                    "min": float(np.min(auroc_scores)),
                    "max": float(np.max(auroc_scores))
                },
                "aupr": {
                    "mean": float(np.mean(aupr_scores)),
                    "std": float(np.std(aupr_scores)),
                    "min": float(np.min(aupr_scores)),
                    "max": float(np.max(aupr_scores))
                } if aupr_scores else None,
                "fpr95": {
                    "mean": float(np.mean(fpr95_scores)),
                    "std": float(np.std(fpr95_scores)),
                    "min": float(np.min(fpr95_scores)),
                    "max": float(np.max(fpr95_scores))
                } if fpr95_scores else None,
                "num_successful_runs": len(auroc_scores)
            }

            # 为每个loss type生成单独的metrics文件 (类似run_baselines.sh)
            if summary["results"][loss_type]["summary_stats"]:
                generate_loss_type_metrics_file(base_output_dir, loss_type, summary["results"][loss_type]["summary_stats"], len(auroc_scores), len(seeds))

    # 保存汇总结果
    summary_file = os.path.join(base_output_dir, 'ablation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 打印汇总表格
    logger.info("=" * 80)
    logger.info("📊 消融实验结果汇总")
    logger.info("=" * 80)
    
    print(f"{'Loss Type':<12} {'AUROC (Mean±Std)':<20} {'AUPR (Mean±Std)':<20} {'FPR95 (Mean±Std)':<20} {'#Runs':<6}")
    print("-" * 80)
    
    for loss_type in loss_types:
        stats = summary["results"][loss_type]["summary_stats"]
        if stats and "auroc" in stats:
            auroc_str = f"{stats['auroc']['mean']:.4f}±{stats['auroc']['std']:.4f}"
            aupr_str = f"{stats['aupr']['mean']:.4f}±{stats['aupr']['std']:.4f}" if stats['aupr'] else "N/A"
            fpr95_str = f"{stats['fpr95']['mean']:.4f}±{stats['fpr95']['std']:.4f}" if stats['fpr95'] else "N/A"
            num_runs = stats['num_successful_runs']
            
            print(f"{loss_type.upper():<12} {auroc_str:<20} {aupr_str:<20} {fpr95_str:<20} {num_runs:<6}")
        else:
            print(f"{loss_type.upper():<12} {'No results':<20} {'No results':<20} {'No results':<20} {'0':<6}")
    
    print("=" * 80)
    logger.info(f"📁 详细结果已保存到: {summary_file}")
    
    return summary

def main():
    setup_logging()
    args = parse_args()
    
    # 确定要运行的损失函数类型
    if args.only_loss_types:
        loss_types = args.only_loss_types
    else:
        loss_types = ["bce", "mse", "hinge"]
    
    logger.info("🎯 开始Energy-DPO消融实验")
    logger.info(f"📋 损失函数类型: {loss_types}")
    logger.info(f"🎲 随机种子: {args.seeds}")
    logger.info(f"🏗️  基础模型: {args.foundation_model}")
    logger.info(f"📊 数据集: {args.dataset}/{args.drugood_subset}")
    
    # 创建对齐的输出目录结构 (类似 run_baselines.sh)
    # 结构: ablation_results/{foundation_model}/{dataset_name}/{loss_type}_seed_{seed}/
    dataset_name = get_dataset_name(args)
    base_output_dir = os.path.join(args.base_output_dir, args.foundation_model, dataset_name)
    os.makedirs(base_output_dir, exist_ok=True)
    
    logger.info(f"📂 实验结果将保存到: {base_output_dir}")
    logger.info(f"📂 目录结构: {args.base_output_dir}/{args.foundation_model}/{dataset_name}/{{loss_type}}_seed_{{seed}}/")
    
    # 运行所有实验
    total_experiments = len(loss_types) * len(args.seeds)
    completed_experiments = 0
    
    for loss_type in loss_types:
        for seed in args.seeds:
            experiment_name = f"{loss_type}_seed_{seed}"
            experiment_dir = os.path.join(base_output_dir, experiment_name)
            
            logger.info(f"🔬 实验 {completed_experiments + 1}/{total_experiments}: {experiment_name}")
            
            # 创建实验配置
            config = create_experiment_config(args, loss_type, seed, experiment_dir)
            
            # 运行实验
            results = run_single_experiment(config, skip_training=args.skip_training)
            
            completed_experiments += 1
            
            if results:
                logger.info(f"✅ 实验 {experiment_name} 完成")
            else:
                logger.error(f"❌ 实验 {experiment_name} 失败")
            
            logger.info(f"📊 进度: {completed_experiments}/{total_experiments}")
            logger.info("-" * 50)
    
    # 收集和汇总结果
    summary = collect_and_summarize_results(base_output_dir, loss_types, args.seeds, args.foundation_model, dataset_name)
    
    logger.info("🎉 所有消融实验完成!")
    logger.info(f"📊 查看详细结果: {base_output_dir}/ablation_summary.json")

if __name__ == "__main__":
    main()
