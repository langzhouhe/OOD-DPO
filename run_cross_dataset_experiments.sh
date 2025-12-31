#!/usr/bin/env python3
"""
Cross-Dataset Generalization Experiments
Train on one dataset (e.g., EC50 Scaffold) and test on another (e.g., EC50 Size)
"""
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

def generate_cross_dataset_metrics_file(foundation_model, train_dataset, test_dataset,
                                       auroc_mean, auroc_std, aupr_mean, aupr_std,
                                       fpr95_mean, fpr95_std, num_seeds):
    """Generate cross-dataset metrics JSON file"""
    import datetime

    metrics_dir = f"./outputs/{foundation_model}/cross_dataset/{train_dataset}_to_{test_dataset}"
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_file = os.path.join(metrics_dir, "cross_dataset_metrics.json")

    metrics_data = {
        "foundation_model": foundation_model,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "experiment_type": "cross_dataset_generalization",
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

    print(f"Cross-dataset metrics saved to: {metrics_file}")

def run_cross_dataset_experiment(train_dataset, test_dataset, seed, foundation_model="minimol", data_seed=42):
    """Run single cross-dataset training-evaluation experiment"""
    train_data_file = f"./data/raw/{train_dataset}.json"
    test_data_file = f"./data/raw/{test_dataset}.json"
    output_dir = f"./outputs/{foundation_model}/cross_dataset/{train_dataset}_to_{test_dataset}/{seed}"

    # Check data files exist
    if not os.path.exists(train_data_file):
        print(f"Training data file not found: {train_data_file}")
        return None
    if not os.path.exists(test_data_file):
        print(f"Test data file not found: {test_data_file}")
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

    # Training command (on train_dataset)
    train_cmd = [
        sys.executable, "main_generalization.py",
        "--mode", "train",
        "--dataset", train_dataset,
        "--data_file", train_data_file,
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

    print(f"Training on {train_dataset} (seed={seed}, model={foundation_model})")

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

    # Cross-dataset evaluation command (test on test_dataset)
    eval_cmd = [
        sys.executable, "main_generalization.py",
        "--mode", "eval",
        "--dataset", train_dataset,  # Keep train dataset for loading model config
        "--data_file", train_data_file,  # Keep train data file for validation set
        "--test_data_file", test_data_file,  # NEW: Test on different dataset
        "--test_drugood_subset", test_dataset,  # NEW: Specify test dataset name
        "--foundation_model", foundation_model,
        "--model_path", model_path,
        "--output_dir", output_dir,
        "--seed", str(seed),
        "--data_seed", str(data_seed),
        "--eval_batch_size", eval_batch_size,
        "--precompute_features",
        "--cache_root", "/home/ubuntu/projects"
    ]

    print(f"Evaluating on {test_dataset} (cross-dataset)...")

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
        print(f"Cross-dataset results - AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, FPR95: {fpr95:.4f}")
        return {"auroc": auroc, "aupr": aupr, "fpr95": fpr95}
    except Exception as e:
        print(f"Failed to read results: {e}")
        return None

def main():
    # Define cross-dataset experiment pairs
    # Format: (train_dataset, test_dataset, description)
    experiment_pairs = [
        # EC50 experiments
        ("lbap_general_ec50_size", "lbap_general_ic50_scaffold", "Size → Scaffold"),
        ("lbap_general_ic50_size", "lbap_general_ec50_scaffold", "Size → Scaffold"),
        # ("lbap_general_ec50_size", "lbap_general_ic50_scaffold", "Size → Scaffold"),
        # ("lbap_general_ec50_size", "lbap_general_ic50_scaffold", "Size → Scaffold"),
        # ("lbap_general_ic50_scaffold", "lbap_general_ec50_size", "Scaffold → Size"),
        # ("lbap_general_ec50_assay", "lbap_general_ec50_scaffold", "Assay → Scaffold"),
        # ("lbap_general_ec50_assay", "lbap_general_ec50_size", "Assay → Size"),

        # IC50 experiments (uncomment if needed)
        # ("lbap_general_ic50_scaffold", "lbap_general_ic50_size", "IC50 Scaffold → Size"),
        # ("lbap_general_ic50_scaffold", "lbap_general_ic50_assay", "IC50 Scaffold → Assay"),
    ]

    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    foundation_model = "minimol"
    data_seed = 42

    total_experiments = len(experiment_pairs) * len(seeds)
    completed_experiments = 0

    print("=" * 80)
    print(f"CROSS-DATASET GENERALIZATION EXPERIMENTS - {foundation_model.upper()}")
    print("=" * 80)
    print(f"Total: {len(experiment_pairs)} dataset pairs × {len(seeds)} seeds = {total_experiments} experiments")
    print(f"Train seeds: {seeds}, Data seed: {data_seed}")
    print()
    print("Experiment pairs:")
    for i, (train_ds, test_ds, desc) in enumerate(experiment_pairs, 1):
        print(f"  {i}. {desc}: Train on {train_ds}, Test on {test_ds}")
    print("=" * 80)

    all_results = {}

    for train_dataset, test_dataset, description in experiment_pairs:
        pair_key = f"{train_dataset}_to_{test_dataset}"
        print(f"\n{'=' * 80}")
        print(f"Experiment: {description}")
        print(f"Train: {train_dataset}")
        print(f"Test:  {test_dataset}")
        print("=" * 80)

        pair_aucs = []
        pair_auprs = []
        pair_fpr95s = []

        for seed in seeds:
            completed_experiments += 1
            print(f"\n[{completed_experiments}/{total_experiments}] Seed {seed}")
            print("-" * 40)

            result = run_cross_dataset_experiment(train_dataset, test_dataset, seed,
                                                 foundation_model, data_seed)
            if result is not None:
                pair_aucs.append(result["auroc"])
                pair_auprs.append(result["aupr"])
                pair_fpr95s.append(result["fpr95"])
                print(f"✓ Success: AUROC={result['auroc']:.4f}, AUPR={result['aupr']:.4f}, FPR95={result['fpr95']:.4f}")
            else:
                print("✗ Failed")

        # Calculate statistics for this pair
        if pair_aucs:
            auroc_mean = np.mean(pair_aucs)
            auroc_std = np.std(pair_aucs, ddof=1) if len(pair_aucs) > 1 else 0
            aupr_mean = np.mean(pair_auprs)
            aupr_std = np.std(pair_auprs, ddof=1) if len(pair_auprs) > 1 else 0
            fpr95_mean = np.mean(pair_fpr95s)
            fpr95_std = np.std(pair_fpr95s, ddof=1) if len(pair_fpr95s) > 1 else 0

            all_results[pair_key] = {
                "train_dataset": train_dataset,
                "test_dataset": test_dataset,
                "description": description,
                "auroc_mean": auroc_mean,
                "auroc_std": auroc_std,
                "aupr_mean": aupr_mean,
                "aupr_std": aupr_std,
                "fpr95_mean": fpr95_mean,
                "fpr95_std": fpr95_std,
                "num_seeds": len(pair_aucs)
            }

            # Save metrics
            generate_cross_dataset_metrics_file(
                foundation_model, train_dataset, test_dataset,
                auroc_mean, auroc_std, aupr_mean, aupr_std, fpr95_mean, fpr95_std,
                len(pair_aucs)
            )

            print(f"\n{description} Summary:")
            print(f"  AUROC = {auroc_mean:.3f} ± {auroc_std:.3f}")
            print(f"  AUPR  = {aupr_mean:.3f} ± {aupr_std:.3f}")
            print(f"  FPR95 = {fpr95_mean:.3f} ± {fpr95_std:.3f}")
            print(f"  (success {len(pair_aucs)}/{len(seeds)})")
        else:
            print(f"\n{description}: No successful experiments")

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL CROSS-DATASET RESULTS SUMMARY")
    print("=" * 80)

    for pair_key, stats in all_results.items():
        print(f"\n{stats['description']}")
        print(f"  Train: {stats['train_dataset']}")
        print(f"  Test:  {stats['test_dataset']}")
        print(f"  AUROC: {stats['auroc_mean']:.3f} ± {stats['auroc_std']:.3f}")
        print(f"  AUPR:  {stats['aupr_mean']:.3f} ± {stats['aupr_std']:.3f}")
        print(f"  FPR95: {stats['fpr95_mean']:.3f} ± {stats['fpr95_std']:.3f}")
        print(f"  Seeds: {stats['num_seeds']}/{len(seeds)}")

    # Save overall summary
    summary_file = f"./outputs/{foundation_model}/cross_dataset_summary.json"
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nOverall summary saved to: {summary_file}")

    print("\n" + "=" * 80)
    print("All cross-dataset experiments completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
