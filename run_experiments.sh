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
    
    # Model-specific parameters
    if foundation_model == "unimol":
        batch_size = "256"
        eval_batch_size = "128"
    else:
        batch_size = "512"
        eval_batch_size = "256"
    
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
        "--encoding_batch_size", "50"
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
    try:
        with open(eval_results_file, 'r') as f:
            results = json.load(f)
        auroc = results.get("auroc", 0)
        aupr = results.get("aupr", 0)
        fpr95 = results.get("fpr95", 1)
        print(f"AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, FPR95: {fpr95:.4f}")
        return {"auroc": auroc, "aupr": aupr, "fpr95": fpr95}
    except Exception as e:
        print(f"Failed to read results: {e}")
        return None

def main():
    datasets = [
        "lbap_general_ec50_assay",
        "lbap_general_ec50_scaffold", 
        "lbap_general_ec50_size",
        "lbap_general_ic50_assay",
        "lbap_general_ic50_scaffold",
        "lbap_general_ic50_size"
    ]
    
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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

        for seed in seeds:
            completed_experiments += 1
            print(f"[{completed_experiments}/{total_experiments}] Seed {seed}")

            result = run_experiment(dataset, seed, foundation_model, data_seed)
            if result is not None:
                dataset_aucs.append(result["auroc"])
                dataset_auprs.append(result["aupr"])
                dataset_fpr95s.append(result["fpr95"])
                print(f"Success, AUROC = {result['auroc']:.4f}, AUPR = {result['aupr']:.4f}, FPR95 = {result['fpr95']:.4f}")
            else:
                print("Failed")

        if dataset_aucs:
            auroc_mean = np.mean(dataset_aucs)
            auroc_std = np.std(dataset_aucs, ddof=1) if len(dataset_aucs) > 1 else 0
            aupr_mean = np.mean(dataset_auprs)
            aupr_std = np.std(dataset_auprs, ddof=1) if len(dataset_auprs) > 1 else 0
            fpr95_mean = np.mean(dataset_fpr95s)
            fpr95_std = np.std(dataset_fpr95s, ddof=1) if len(dataset_fpr95s) > 1 else 0

            results[dataset] = {
                'auroc_mean': auroc_mean,
                'auroc_std': auroc_std,
                'aupr_mean': aupr_mean,
                'aupr_std': aupr_std,
                'fpr95_mean': fpr95_mean,
                'fpr95_std': fpr95_std,
                'aucs': dataset_aucs,
                'auprs': dataset_auprs,
                'fpr95s': dataset_fpr95s,
                'n_success': len(dataset_aucs)
            }

            # Generate dataset metrics file
            generate_dataset_metrics_file(foundation_model, dataset, auroc_mean, auroc_std, aupr_mean, aupr_std, fpr95_mean, fpr95_std, len(dataset_aucs))

            print(f"Summary: AUROC={auroc_mean:.3f}±{auroc_std:.3f}, AUPR={aupr_mean:.3f}±{aupr_std:.3f}, FPR95={fpr95_mean:.3f}±{fpr95_std:.3f} (success {len(dataset_aucs)}/{len(seeds)})")
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

            print(f"{dataset:30s}: AUROC={auroc_mean:.3f}±{auroc_std:.3f}, AUPR={aupr_mean:.3f}±{aupr_std:.3f}, FPR95={fpr95_mean:.3f}±{fpr95_std:.3f} ({n_success}/{len(seeds)})")

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