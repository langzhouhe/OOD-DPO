#!/usr/bin/env python3
import sys
import subprocess
import os
import json
import numpy as np

def find_model_in_outputs(base_dir="./outputs"):
    """Recursively find the latest model file"""
    best_model = None
    latest_time = 0
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if 'best' in file and file.endswith('.pth'):
                file_path = os.path.join(root, file)
                file_time = os.path.getmtime(file_path)
                if file_time > latest_time:
                    latest_time = file_time
                    best_model = file_path
    
    return best_model

def generate_dataset_metrics_file(foundation_model, dataset, auroc_mean, auroc_std, aupr_mean, aupr_std, fpr95_mean, fpr95_std, num_seeds):
    """Generate dataset metrics JSON file"""
    import datetime

    metrics_dir = f"./outputs/{foundation_model}/{dataset}"
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_file = os.path.join(metrics_dir, "dataset_metrics.json")

    metrics_data = {
        "foundation_model": foundation_model,
        "dataset": dataset,
        "metrics": {
            "auroc": {
                "mean": float(auroc_mean),
                "std": float(auroc_std)
            },
            "aupr": {
                "mean": float(aupr_mean),
                "std": float(aupr_std)
            },
            "fpr95": {
                "mean": float(fpr95_mean),
                "std": float(fpr95_std)
            }
        },
        "num_seeds": num_seeds,
        "timestamp": datetime.datetime.now().isoformat()
    }

    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)

    print(f"Dataset metrics saved to: {metrics_file}")

def run_experiment(dataset, seed, foundation_model="minimol", data_seed=42):
    """Run single training-evaluation experiment"""
    data_file = f"./data/raw/{dataset}.json"
    output_dir = f"./outputs/{foundation_model}/{dataset}/{seed}"
    
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Environment setup
    env = os.environ.copy()
    env['TQDM_DISABLE'] = '0'
    env['SHOW_PROGRESS'] = '1'
    
    # Model-specific parameters - OPTIMIZED FOR 80GB A100
    if foundation_model == "unimol":
        batch_size = "4096"  # Increased from 256 -> 1024 -> 4096 for better GPU utilization
        eval_batch_size = "4096"  # Increased from 128 -> 1024 -> 4096
    else:
        batch_size = "8192"  # Increased from 512 -> 2048 -> 8192 for better GPU utilization
        eval_batch_size = "4096"  # Increased from 256 -> 1024 -> 4096
    
    # Training command - UPDATED CACHE PATH
    train_cmd = [
        sys.executable, "main.py",
        "--mode", "train",
        "--dataset", dataset,
        "--data_file", data_file,
        "--foundation_model", foundation_model,
        "--output_dir", output_dir,
        "--seed", str(seed),
        "--data_seed", str(data_seed),
        "--epochs", "500",
        "--batch_size", batch_size,
        "--eval_batch_size", eval_batch_size,
        "--lr", "1e-4",
        "--eval_steps", "25",
        "--precompute_features",
        "--cache_root", "/home/ubuntu/projects",
        "--encoding_batch_size", "500"  # Increased from 50 for better GPU utilization
    ]
    
    print(f"Training {dataset} (seed={seed}, model={foundation_model})")
    
    try:
        result = subprocess.run(train_cmd, env=env, check=True)
        print("Training completed")
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        return None
    
    # Find model
    model_path = find_model_in_outputs(output_dir)
    if not model_path:
        print("Model file not found")
        return None
    
    # Evaluation command - UPDATED CACHE PATH
    eval_cmd = [
        sys.executable, "main.py",
        "--mode", "eval",
        "--dataset", dataset,
        "--data_file", data_file,
        "--foundation_model", foundation_model,
        "--model_path", model_path,
        "--output_dir", output_dir,
        "--seed", str(seed),
        "--data_seed", str(data_seed),
        "--eval_batch_size", eval_batch_size,
        "--precompute_features",
        "--cache_root", "/home/ubuntu/projects"
    ]
    
    print("Starting evaluation...")
    
    try:
        result = subprocess.run(eval_cmd, env=env, check=True)
        print("Evaluation completed")
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed: {e}")
        return None
    
    # Read results
    eval_results_file = os.path.join(output_dir, "ood_evaluation_results.json")
    training_stats_file = os.path.join(output_dir, "training_stats.json")
    try:
        # Read evaluation results
        with open(eval_results_file, 'r') as f:
            results = json.load(f)
        auroc = results.get("auroc", 0)
        aupr = results.get("aupr", 0)
        fpr95 = results.get("fpr95", 1)
        eval_time = results.get("eval_time_seconds", 0)
        eval_gpu = results.get("peak_gpu_memory_eval_gb", 0)

        # Read training stats
        train_time = 0
        avg_epoch_time = 0
        train_gpu = 0
        if os.path.exists(training_stats_file):
            with open(training_stats_file, 'r') as f:
                train_stats = json.load(f)
            train_time = train_stats.get("train_time_seconds", 0)
            avg_epoch_time = train_stats.get("avg_epoch_time_seconds", 0)
            train_gpu = train_stats.get("peak_gpu_memory_train_gb", 0)

        print(f"AUROC: {auroc:.2f}, AUPR: {aupr:.2f}, FPR95: {fpr95:.2f}, Train: {train_time:.1f}s, Eval: {eval_time:.1f}s")
        return {
            "auroc": auroc, "aupr": aupr, "fpr95": fpr95,
            "train_time_seconds": train_time,
            "eval_time_seconds": eval_time,
            "avg_epoch_time_seconds": avg_epoch_time,
            "peak_gpu_memory_train_gb": train_gpu,
            "peak_gpu_memory_eval_gb": eval_gpu
        }
    except Exception as e:
        print(f"Failed to read results: {e}")
        return None

def main():
    datasets = [
        "lbap_general_ec50_assay",
        # "lbap_general_ec50_scaffold", 
        # "lbap_general_ec50_size",
        # "lbap_general_ic50_assay",
        # "lbap_general_ic50_scaffold",
        # "lbap_general_ic50_size"
    ]
    
    seeds = [1]
    foundation_model = "minimol"  
    data_seed = 42
    
    results = {}
    total_experiments = len(datasets) * len(seeds)
    completed_experiments = 0
    
    print(f"Running {foundation_model.upper()} experiments")
    print(f"Total: {len(datasets)} datasets × {len(seeds)} seeds = {total_experiments} experiments")
    print(f"Train seeds: {seeds}, Data seed: {data_seed}")
    print("=" * 60)
    
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        print("-" * 40)
        dataset_aucs = []
        dataset_auprs = []
        dataset_fpr95s = []
        dataset_train_times = []
        dataset_eval_times = []
        dataset_epoch_times = []
        dataset_train_gpu = []
        dataset_eval_gpu = []

        for seed in seeds:
            completed_experiments += 1
            print(f"[{completed_experiments}/{total_experiments}] Seed {seed}")

            result = run_experiment(dataset, seed, foundation_model, data_seed)
            if result is not None:
                dataset_aucs.append(result["auroc"])
                dataset_auprs.append(result["aupr"])
                dataset_fpr95s.append(result["fpr95"])
                dataset_train_times.append(result.get("train_time_seconds", 0))
                dataset_eval_times.append(result.get("eval_time_seconds", 0))
                dataset_epoch_times.append(result.get("avg_epoch_time_seconds", 0))
                dataset_train_gpu.append(result.get("peak_gpu_memory_train_gb", 0))
                dataset_eval_gpu.append(result.get("peak_gpu_memory_eval_gb", 0))
                print(f"Success, AUROC = {result['auroc']:.2f}, AUPR = {result['aupr']:.2f}, FPR95 = {result['fpr95']:.2f}")
            else:
                print("Failed")

        if dataset_aucs:
            auroc_mean = np.mean(dataset_aucs)
            auroc_std = np.std(dataset_aucs, ddof=1) if len(dataset_aucs) > 1 else 0
            aupr_mean = np.mean(dataset_auprs)
            aupr_std = np.std(dataset_auprs, ddof=1) if len(dataset_auprs) > 1 else 0
            fpr95_mean = np.mean(dataset_fpr95s)
            fpr95_std = np.std(dataset_fpr95s, ddof=1) if len(dataset_fpr95s) > 1 else 0

            # Calculate timing and memory stats
            train_time_mean = np.mean(dataset_train_times) if dataset_train_times else 0
            train_time_std = np.std(dataset_train_times, ddof=1) if len(dataset_train_times) > 1 else 0
            eval_time_mean = np.mean(dataset_eval_times) if dataset_eval_times else 0
            eval_time_std = np.std(dataset_eval_times, ddof=1) if len(dataset_eval_times) > 1 else 0
            epoch_time_mean = np.mean(dataset_epoch_times) if dataset_epoch_times else 0
            epoch_time_std = np.std(dataset_epoch_times, ddof=1) if len(dataset_epoch_times) > 1 else 0
            train_gpu_mean = np.mean(dataset_train_gpu) if dataset_train_gpu else 0
            train_gpu_std = np.std(dataset_train_gpu, ddof=1) if len(dataset_train_gpu) > 1 else 0
            eval_gpu_mean = np.mean(dataset_eval_gpu) if dataset_eval_gpu else 0
            eval_gpu_std = np.std(dataset_eval_gpu, ddof=1) if len(dataset_eval_gpu) > 1 else 0

            results[dataset] = {
                'auroc_mean': auroc_mean,
                'auroc_std': auroc_std,
                'aupr_mean': aupr_mean,
                'aupr_std': aupr_std,
                'fpr95_mean': fpr95_mean,
                'fpr95_std': fpr95_std,
                'train_time_mean': train_time_mean,
                'train_time_std': train_time_std,
                'eval_time_mean': eval_time_mean,
                'eval_time_std': eval_time_std,
                'epoch_time_mean': epoch_time_mean,
                'epoch_time_std': epoch_time_std,
                'train_gpu_mean': train_gpu_mean,
                'train_gpu_std': train_gpu_std,
                'eval_gpu_mean': eval_gpu_mean,
                'eval_gpu_std': eval_gpu_std,
                'aucs': dataset_aucs,
                'auprs': dataset_auprs,
                'fpr95s': dataset_fpr95s,
                'n_success': len(dataset_aucs)
            }

            # Generate dataset metrics file
            generate_dataset_metrics_file(foundation_model, dataset, auroc_mean, auroc_std, aupr_mean, aupr_std, fpr95_mean, fpr95_std, len(dataset_aucs))

            print(f"Summary: AUROC={auroc_mean:.3f}±{auroc_std:.3f}, AUPR={aupr_mean:.3f}±{aupr_std:.3f}, FPR95={fpr95_mean:.3f}±{fpr95_std:.3f} (success {len(dataset_aucs)}/{len(seeds)})")
            print(f"  Train Time: {train_time_mean:.2f}±{train_time_std:.2f}s | Eval Time: {eval_time_mean:.2f}±{eval_time_std:.2f}s | Avg Epoch: {epoch_time_mean:.2f}±{epoch_time_std:.2f}s")
            print(f"  Peak GPU (Train): {train_gpu_mean:.2f}±{train_gpu_std:.2f}GB | Peak GPU (Eval): {eval_gpu_mean:.2f}±{eval_gpu_std:.2f}GB")
        else:
            print("All experiments failed")
    
    # Final report
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    if results:
        overall_auroc_means = []
        overall_aupr_means = []
        overall_fpr95_means = []

        for dataset, data in results.items():
            auroc_mean = data['auroc_mean']
            auroc_std = data['auroc_std']
            aupr_mean = data['aupr_mean']
            aupr_std = data['aupr_std']
            fpr95_mean = data['fpr95_mean']
            fpr95_std = data['fpr95_std']
            n_success = data['n_success']

            overall_auroc_means.append(auroc_mean)
            overall_aupr_means.append(aupr_mean)
            overall_fpr95_means.append(fpr95_mean)

            train_time_mean = data.get('train_time_mean', 0)
            train_time_std = data.get('train_time_std', 0)
            eval_time_mean = data.get('eval_time_mean', 0)
            eval_time_std = data.get('eval_time_std', 0)
            epoch_time_mean = data.get('epoch_time_mean', 0)
            epoch_time_std = data.get('epoch_time_std', 0)
            train_gpu_mean = data.get('train_gpu_mean', 0)
            train_gpu_std = data.get('train_gpu_std', 0)
            eval_gpu_mean = data.get('eval_gpu_mean', 0)
            eval_gpu_std = data.get('eval_gpu_std', 0)

            print(f"{dataset:30s}: AUROC={auroc_mean:.3f}±{auroc_std:.3f}, AUPR={aupr_mean:.3f}±{aupr_std:.3f}, FPR95={fpr95_mean:.3f}±{fpr95_std:.3f} ({n_success}/{len(seeds)})")
            print(f"{'':{30}}  Train: {train_time_mean:.2f}±{train_time_std:.2f}s | Eval: {eval_time_mean:.2f}±{eval_time_std:.2f}s | Epoch: {epoch_time_mean:.2f}±{epoch_time_std:.2f}s")
            print(f"{'':{30}}  GPU Train: {train_gpu_mean:.2f}±{train_gpu_std:.2f}GB | GPU Eval: {eval_gpu_mean:.2f}±{eval_gpu_std:.2f}GB")

        print("-" * 60)
        overall_auroc_mean = np.mean(overall_auroc_means)
        overall_auroc_std = np.std(overall_auroc_means, ddof=1) if len(overall_auroc_means) > 1 else 0
        overall_aupr_mean = np.mean(overall_aupr_means)
        overall_aupr_std = np.std(overall_aupr_means, ddof=1) if len(overall_aupr_means) > 1 else 0
        overall_fpr95_mean = np.mean(overall_fpr95_means)
        overall_fpr95_std = np.std(overall_fpr95_means, ddof=1) if len(overall_fpr95_means) > 1 else 0

        print(f"{'Overall AUROC':30s}: {overall_auroc_mean:.3f} ± {overall_auroc_std:.3f}")
        print(f"{'Overall AUPR':30s}: {overall_aupr_mean:.3f} ± {overall_aupr_std:.3f}")
        print(f"{'Overall FPR95':30s}: {overall_fpr95_mean:.3f} ± {overall_fpr95_std:.3f}")

        # Save results
        summary_file = f"./outputs/{foundation_model}_experiment_results.json"
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        with open(summary_file, 'w') as f:
            json.dump({
                'foundation_model': foundation_model,
                'experimental_setup': {
                    'train_seeds': seeds,
                    'data_seed': data_seed,
                    'epochs': 500,
                    'eval_steps': 25
                },
                'results': results,
                'overall_metrics': {
                    'auroc_mean': overall_auroc_mean,
                    'auroc_std': overall_auroc_std,
                    'aupr_mean': overall_aupr_mean,
                    'aupr_std': overall_aupr_std,
                    'fpr95_mean': overall_fpr95_mean,
                    'fpr95_std': overall_fpr95_std
                },
                'total_experiments': total_experiments,
                'datasets': datasets
            }, f, indent=2)
        print(f"Results saved to: {summary_file}")
    else:
        print("No successful experiments")
    
    print("=" * 60)

if __name__ == "__main__":
    main()