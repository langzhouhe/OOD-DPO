#!/usr/bin/env python3

import sys
import subprocess
import os
import json
import numpy as np
import argparse
import threading
import pandas as pd

def filter_output(process):
    """过滤stderr中的pandas警告"""
    while True:
        line = process.stderr.readline()
        if line == '' and process.poll() is not None:
            break
        if line:
            # 过滤掉pandas警告
            if not any(keyword in line for keyword in [
                "Failed to find the pandas get_adjustment",
                "Failed to patch pandas", 
                "PandasTools will have limited functionality",
                "RDKit WARNING",
                "UserWarning"
            ]):
                print(line.rstrip(), file=sys.stderr)

def find_model_in_outputs(base_dir="./outputs"):
    """递归查找最新的模型文件"""
    best_model = None
    latest_time = 0
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if 'best' in file and file.endswith('.pth'):
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    file_time = os.path.getmtime(file_path)
                    if file_time > latest_time:
                        latest_time = file_time
                        best_model = file_path
    
    return best_model

def run_experiment(dataset, domain='size', shift='covariate', seed=42, epochs=500, 
                   batch_size=256, lr=1e-4, foundation_model='minimol', data_seed=42):
    """运行单次训练-评估实验，支持minimol和unimol"""
    
    print(f"🚀 开始实验: {dataset} (domain={domain}, shift={shift}, seed={seed}, model={foundation_model})")
    
    # 构建输出目录
    experiment_name = f"{dataset}_{domain}_{shift}"
    output_dir = f"./outputs/{foundation_model}/{experiment_name}/{seed}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 环境变量设置
    env = os.environ.copy()
    env['TQDM_DISABLE'] = '0'
    env['SHOW_PROGRESS'] = '1'
    env['PYTHONWARNINGS'] = 'ignore'
    env['RDK_QUIET'] = '1'
    
    # 根据foundation_model设置批大小（参考第一个脚本的逻辑）
    if foundation_model == "unimol":
        train_batch_size = "256"
        eval_batch_size = "128"
    else:  # minimol
        train_batch_size = "512"
        eval_batch_size = "256"
    
    # 如果用户指定了batch_size，使用用户指定的值
    if batch_size != 256:  # 256是默认值
        train_batch_size = str(batch_size)
        eval_batch_size = str(batch_size // 2)
    
    # 训练命令 - 添加foundation_model相关参数
    train_cmd = [
        sys.executable, "main.py",
        "--mode", "train",
        "--dataset", dataset,
        "--good_domain", domain,
        "--good_shift", shift,
        "--foundation_model", foundation_model,
        "--seed", str(seed),
        "--data_seed", str(data_seed),
        "--output_dir", output_dir,
        "--epochs", str(epochs),
        "--batch_size", train_batch_size,
        "--eval_batch_size", eval_batch_size,
        "--lr", str(lr),
        "--eval_steps", "25",
        "--precompute_features",
        "--cache_root", "/home/ubuntu/projects",
        "--encoding_batch_size", "50"
    ]
    
    try:
        print(f"  📚 开始训练 ({foundation_model.upper()}, batch={train_batch_size})...")
        
        # 使用Popen和过滤线程
        process = subprocess.Popen(
            train_cmd,
            env=env,
            stdout=sys.stdout,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # 启动过滤线程
        filter_thread = threading.Thread(target=filter_output, args=(process,))
        filter_thread.daemon = True
        filter_thread.start()
        
        # 等待完成
        return_code = process.wait()
        
        if return_code != 0:
            print(f"  ❌ 训练失败，退出码: {return_code}")
            return None
            
        print("  ✅ 训练完成")
        
    except Exception as e:
        print(f"  ❌ 训练失败: {e}")
        return None
    
    # 查找模型文件
    model_path = find_model_in_outputs(output_dir)
    
    if not model_path:
        print("  ❌ 未找到模型文件")
        return None
    
    print(f"  📁 使用模型: {os.path.basename(model_path)}")
    print("  🔍 开始评估...")
    
    # 评估命令 - 添加foundation_model相关参数
    eval_cmd = [
        sys.executable, "main.py",
        "--mode", "eval",
        "--dataset", dataset,
        "--good_domain", domain,
        "--good_shift", shift,
        "--foundation_model", foundation_model,
        "--seed", str(seed),
        "--data_seed", str(data_seed),
        "--model_path", model_path,
        "--output_dir", output_dir,
        "--eval_batch_size", eval_batch_size,
        "--precompute_features",
        "--cache_root", "/home/ubuntu/projects"
    ]
    
    try:
        # 评估也用相同的过滤方式
        process = subprocess.Popen(
            eval_cmd,
            env=env,
            stdout=sys.stdout,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # 启动过滤线程
        filter_thread = threading.Thread(target=filter_output, args=(process,))
        filter_thread.daemon = True
        filter_thread.start()
        
        # 等待完成
        return_code = process.wait()
        
        if return_code != 0:
            print(f"  ❌ 评估失败，退出码: {return_code}")
            return None
            
        print("  ✅ 评估完成")
        
    except Exception as e:
        print(f"  ❌ 评估失败: {e}")
        return None
    
    # 读取结果文件（优先读取evaluation_metrics.csv）
    eval_metrics_file = os.path.join(output_dir, "evaluation_metrics.csv")
    if os.path.exists(eval_metrics_file):
        try:
            metrics_df = pd.read_csv(eval_metrics_file)
            if len(metrics_df) > 0:
                row = metrics_df.iloc[0]
                
                metrics = {
                    'auroc': row.get('auroc', None),
                    'aupr': row.get('aupr', None), 
                    'fpr95': row.get('fpr95', None)
                }
                
                for metric_name, metric_value in metrics.items():
                    if metric_value is not None:
                        print(f"  📈 {metric_name.upper()}: {metric_value:.4f}")
                
                return metrics
                
        except Exception as e:
            print(f"  ❌ 读取evaluation_metrics.csv失败: {e}")
    
    # 备选：读取ood_evaluation_results.json（参考第一个脚本）
    ood_results_file = os.path.join(output_dir, "ood_evaluation_results.json")
    if os.path.exists(ood_results_file):
        try:
            with open(ood_results_file, 'r') as f:
                results = json.load(f)
            
            metrics = {
                'auroc': results.get("auroc"),
                'aupr': results.get("aupr"), 
                'fpr95': results.get("fpr95")
            }
            
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    print(f"  📈 {metric_name.upper()}: {metric_value:.4f}")
            
            return metrics
                
        except Exception as e:
            print(f"  ❌ 读取ood_evaluation_results.json失败: {e}")
    
    # 备选：读取eval_results.json
    eval_results_file = os.path.join(output_dir, "eval_results.json")
    if os.path.exists(eval_results_file):
        try:
            with open(eval_results_file, 'r') as f:
                results = json.load(f)
            
            metrics = {
                'auroc': results.get('auroc', results.get('auc', None)),
                'aupr': results.get('aupr', results.get('auprc', None)),
                'fpr95': results.get('fpr95', results.get('fpr_at_95_tpr', None))
            }
            
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    print(f"  📈 {metric_name.upper()}: {metric_value:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"  ❌ 读取eval_results.json失败: {e}")
    
    # 如果所有结果文件都不存在，列出输出目录中的文件
    print(f"  📋 输出目录中的文件:")
    try:
        for file in os.listdir(output_dir):
            print(f"    - {file}")
    except:
        pass
    
    print(f"  ❌ 未找到结果文件")
    return None

def calculate_stats(values):
    """计算平均值和标准差"""
    if not values:
        return None, None

    values = [v for v in values if v is not None]
    if not values:
        return None, None

    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return mean_val, std_val

def generate_dataset_metrics_file(foundation_model, dataset, domain, shift, auroc_mean, auroc_std, aupr_mean, aupr_std, fpr95_mean, fpr95_std, num_seeds):
    """Generate dataset metrics JSON file"""
    import datetime

    experiment_name = f"{dataset}_{domain}_{shift}"
    metrics_dir = f"./outputs/{foundation_model}/{experiment_name}"
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_file = os.path.join(metrics_dir, "dataset_metrics.json")

    metrics_data = {
        "foundation_model": foundation_model,
        "dataset": dataset,
        "domain": domain,
        "shift": shift,
        "experiment_name": experiment_name,
        "metrics": {
            "auroc": {
                "mean": float(auroc_mean) if auroc_mean is not None else None,
                "std": float(auroc_std) if auroc_std is not None else None
            },
            "aupr": {
                "mean": float(aupr_mean) if aupr_mean is not None else None,
                "std": float(aupr_std) if aupr_std is not None else None
            },
            "fpr95": {
                "mean": float(fpr95_mean) if fpr95_mean is not None else None,
                "std": float(fpr95_std) if fpr95_std is not None else None
            }
        },
        "num_seeds": num_seeds,
        "timestamp": datetime.datetime.now().isoformat()
    }

    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)

    print(f"📄 Dataset metrics saved to: {metrics_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="运行good_data数据集多种子实验（支持minimol和unimol）")
    
    parser.add_argument("--datasets", nargs='+', 
                        choices=['good_hiv', 'good_pcba', 'good_zinc'],
                        default=['good_hiv', 'good_pcba', 'good_zinc'],
                        help="要运行的数据集")
    
    parser.add_argument("--domains", nargs='+',
                        choices=['scaffold', 'size'], 
                        default=['size', 'scaffold'],
                        help="要测试的域")
    
    parser.add_argument("--shifts", nargs='+',
                        choices=['covariate', 'concept', 'no_shift'],
                        default=['covariate'],
                        help="要测试的shift类型")
    
    parser.add_argument("--seeds", nargs='+', type=int,
                        default=[1, 2, 3, 4, 5, 6,7,8,9,10],
                        help="要测试的随机种子")
    
    # 🔥 修复：添加foundation_models参数支持
    parser.add_argument("--foundation_models", nargs='+',
                        choices=['minimol', 'unimol'],
                        default=['minimol', 'unimol'],
                        help="要测试的基础模型")
    
    # 🔥 兼容性：同时支持单数形式（向后兼容）
    parser.add_argument("--foundation_model", type=str,
                        choices=['minimol', 'unimol'],
                        default=None,
                        help="单个基础模型（向后兼容）")
    
    parser.add_argument("--data_seed", type=int, default=42,
                        help="数据划分随机种子")
    
    parser.add_argument("--epochs", type=int, default=500, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=256, help="批大小（可选，会被模型默认值覆盖）")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 🔥 修复：处理参数兼容性
    if args.foundation_model is not None:
        # 如果指定了单数形式，转换为列表形式
        args.foundation_models = [args.foundation_model]
        print(f"⚠️  检测到单模型参数，自动转换: {args.foundation_model}")
    
    print("🎯 Good Data 多种子多模型实验运行器")
    print("🔇 已自动过滤RDKit pandas警告")
    print("=" * 80)
    print(f"数据集: {args.datasets}")
    print(f"域: {args.domains}")
    print(f"Shift: {args.shifts}")
    print(f"基础模型: {args.foundation_models}")
    print(f"种子: {args.seeds}")
    print(f"数据种子: {args.data_seed}")
    print(f"训练参数: epochs={args.epochs}, lr={args.lr}")
    print(f"🔧 模型特定批大小: unimol(256/128), minimol(512/256)")
    print("=" * 80)
    
    all_results = {}
    total_experiments = len(args.datasets) * len(args.domains) * len(args.shifts) * len(args.foundation_models) * len(args.seeds)
    completed_experiments = 0
    
    for foundation_model in args.foundation_models:
        print(f"\n{'🤖 基础模型: ' + foundation_model.upper():<80}")
        
        for dataset in args.datasets:
            for domain in args.domains:
                for shift in args.shifts:
                    experiment_name = f"{foundation_model}_{dataset}_{domain}_{shift}"
                    print(f"\n{'='*60}")
                    print(f"🎯 实验配置: {experiment_name}")
                    print(f"{'='*60}")
                    
                    # 收集每个指标的所有种子结果
                    experiment_metrics = {
                        'auroc': [],
                        'aupr': [],
                        'fpr95': []
                    }
                    
                    successful_runs = 0
                    for seed in args.seeds:
                        completed_experiments += 1
                        print(f"\n🌱 种子 {seed} [{completed_experiments}/{total_experiments}]:")
                        
                        metrics = run_experiment(
                            dataset=dataset,
                            domain=domain,
                            shift=shift,
                            seed=seed,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            foundation_model=foundation_model,
                            data_seed=args.data_seed
                        )
                        
                        if metrics is not None:
                            successful_runs += 1
                            for metric_name, metric_value in metrics.items():
                                if metric_value is not None:
                                    experiment_metrics[metric_name].append(metric_value)
                        else:
                            print(f"  💥 种子 {seed} 实验失败")
                    
                    # 计算每个指标的统计量
                    if successful_runs > 0:
                        experiment_stats = {}
                        for metric_name, metric_values in experiment_metrics.items():
                            mean_val, std_val = calculate_stats(metric_values)
                            experiment_stats[metric_name] = {
                                'mean': mean_val,
                                'std': std_val,
                                'count': len(metric_values)
                            }

                        all_results[experiment_name] = experiment_stats

                        # Generate individual dataset metrics JSON file
                        auroc_mean = experiment_stats['auroc']['mean']
                        auroc_std = experiment_stats['auroc']['std']
                        aupr_mean = experiment_stats['aupr']['mean']
                        aupr_std = experiment_stats['aupr']['std']
                        fpr95_mean = experiment_stats['fpr95']['mean']
                        fpr95_std = experiment_stats['fpr95']['std']

                        generate_dataset_metrics_file(
                            foundation_model, dataset, domain, shift,
                            auroc_mean, auroc_std, aupr_mean, aupr_std, fpr95_mean, fpr95_std,
                            successful_runs
                        )

                        # 打印当前实验总结
                        print(f"\n📊 {experiment_name} 总结 (成功运行: {successful_runs}/{len(args.seeds)}):")
                        for metric_name, stats in experiment_stats.items():
                            if stats['mean'] is not None:
                                print(f"  {metric_name.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
                            else:
                                print(f"  {metric_name.upper()}: 无有效数据")
                    else:
                        print(f"\n💥 {experiment_name}: 所有实验都失败了")
    
    # 最终完整报告
    print(f"\n{'='*100}")
    print("🚀 Good Data 多模型实验最终总结报告 🚀".center(100))
    print(f"{'='*100}")
    
    if all_results:
        # 按模型分组显示
        for foundation_model in args.foundation_models:
            print(f"\n🤖 {foundation_model.upper()} 模型结果:")
            print(f"{'实验配置':<50} {'AUROC':<15} {'AUPR':<15} {'FPR95':<15}")
            print(f"{'-'*50} {'-'*15} {'-'*15} {'-'*15}")
            
            model_results = {k: v for k, v in all_results.items() if k.startswith(foundation_model)}
            for experiment_name, experiment_stats in model_results.items():
                # 移除模型名前缀显示
                display_name = experiment_name[len(foundation_model)+1:]
                
                auroc_str = f"{experiment_stats['auroc']['mean']:.3f}±{experiment_stats['auroc']['std']:.3f}" if experiment_stats['auroc']['mean'] is not None else "N/A"
                aupr_str = f"{experiment_stats['aupr']['mean']:.3f}±{experiment_stats['aupr']['std']:.3f}" if experiment_stats['aupr']['mean'] is not None else "N/A"
                fpr95_str = f"{experiment_stats['fpr95']['mean']:.3f}±{experiment_stats['fpr95']['std']:.3f}" if experiment_stats['fpr95']['mean'] is not None else "N/A"
                
                print(f"{display_name:<50} {auroc_str:<15} {aupr_str:<15} {fpr95_str:<15}")
        
        # 保存详细结果到CSV文件
        results_file = "./good_experiment_results_multi_model.csv"
        summary_data = []
        
        for experiment_name, experiment_stats in all_results.items():
            # 解析实验名称: foundation_model_dataset_domain_shift
            parts = experiment_name.split('_')
            if len(parts) >= 4:
                foundation_model = parts[0]
                dataset = parts[1] + '_' + parts[2]  # good_hiv, good_pcba, good_zinc
                domain = parts[3]
                shift = parts[4] if len(parts) > 4 else 'unknown'
            else:
                foundation_model, dataset, domain, shift = 'unknown', experiment_name, 'unknown', 'unknown'
            
            row = {
                'experiment': experiment_name,
                'foundation_model': foundation_model,
                'dataset': dataset,
                'domain': domain, 
                'shift': shift
            }
            
            for metric_name, stats in experiment_stats.items():
                if stats['mean'] is not None:
                    row[f'{metric_name}_mean'] = stats['mean']
                    row[f'{metric_name}_std'] = stats['std']
                    row[f'{metric_name}_count'] = stats['count']
                else:
                    row[f'{metric_name}_mean'] = None
                    row[f'{metric_name}_std'] = None
                    row[f'{metric_name}_count'] = 0
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(results_file, index=False)
        print(f"\n💾 详细结果已保存到: {results_file}")
        
        # 保存JSON格式的完整结果（参考第一个脚本）
        json_file = "./good_experiment_results_complete.json"
        complete_results = {
            'foundation_models': args.foundation_models,
            'experimental_setup': {
                'datasets': args.datasets,
                'domains': args.domains,
                'shifts': args.shifts,
                'train_seeds': args.seeds,
                'data_seed': args.data_seed,
                'epochs': args.epochs,
                'lr': args.lr
            },
            'results': all_results,
            'total_experiments': total_experiments,
            'completed_experiments': completed_experiments
        }
        
        with open(json_file, 'w') as f:
            json.dump(complete_results, f, indent=2)
        print(f"💾 完整结果已保存到: {json_file}")
        
    else:
        print("💥 没有成功的实验结果")
    
    print(f"{'='*100}")

if __name__ == "__main__":
    main()