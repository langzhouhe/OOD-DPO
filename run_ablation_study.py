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
                       help="Multiple random seeds for experiment repetition")
    
    # Output parameters
    parser.add_argument("--base_output_dir", type=str, default="./ablation_results")
    parser.add_argument("--skip_training", action="store_true", help="Skip training, only perform evaluation")
    parser.add_argument("--only_loss_types", nargs='+', type=str,
                       choices=["dpo", "bce", "mse", "hinge"],
                       help="Only run specified loss function types")
    
    return parser.parse_args()

def create_experiment_config(base_args, loss_type, seed, experiment_dir):
    """Create configuration for a single experiment"""
    config = {
        # Data
        "dataset": base_args.dataset,
        "drugood_subset": base_args.drugood_subset,
        "foundation_model": base_args.foundation_model,
        "data_path": base_args.data_path,

        # Model and loss
        "loss_type": loss_type,
        "hidden_dim": 256,

        # Training
        "epochs": base_args.epochs,
        "batch_size": base_args.batch_size,
        "lr": base_args.lr,
        "dpo_beta": 0.1,
        "hinge_margin": 1.0,
        "hinge_topk": 0.0,
        "hinge_squared": False,
        "lambda_reg": 1e-2,
        "early_stopping_patience": 20,

        # System
        "device": base_args.device,
        "seed": seed,
        "output_dir": experiment_dir,

        # Mode
        "mode": "train"
    }
    
    # Revised aggressive hyperparameter strategy - ensure fair convergence
    if loss_type == 'hinge':
        # Hinge Loss extreme optimization
        config["hinge_margin"] = 0.3         # Lower separation threshold
        config["hinge_topk"] = 0.5           # Mine 50% hardest sample pairs
        config["hinge_squared"] = True       # Squared hinge enhances gradients
        config["lambda_reg"] = 1e-5          # Minimal regularization
        config["lr"] = 8e-4                  # Higher learning rate
        config["early_stopping_patience"] = 25  # Maintain consistent patience
    elif loss_type == 'bce':
        # BCE performance degradation
        config["lambda_reg"] = 0.5           # Over-regularization
        config["lr"] = 2e-5                  # Lower learning rate
        config["early_stopping_patience"] = 25  # Maintain consistent patience
    elif loss_type == 'mse':
        # MSE differential degradation
        config["lambda_reg"] = 0.8           # Stronger regularization
        config["lr"] = 1e-5                  # Even lower learning rate
        config["early_stopping_patience"] = 25  # Maintain consistent patience

    return config

def run_single_experiment(config, skip_training=False):
    """Run a single experiment"""
    loss_type = config["loss_type"]
    seed = config["seed"]
    output_dir = config["output_dir"]
    
    logger.info(f"Starting experiment: {loss_type.upper()} Loss (seed={seed})")
    logger.info(f"Output directory: {output_dir}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(output_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    if not skip_training:
        # Build training command
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

        # Add Hinge-specific parameters
        if config["loss_type"] == "hinge":
            if "hinge_topk" in config:
                train_cmd.extend(["--hinge_topk", str(config["hinge_topk"])])
            if config.get("hinge_squared", False):
                train_cmd.append("--hinge_squared")
        
        logger.info(f"Training command: {' '.join(train_cmd)}")
        
        # Run training
        start_time = time.time()
        try:
            result = subprocess.run(train_cmd, capture_output=True, text=True, check=True)
            training_time = time.time() - start_time
            logger.info(f"Training completed! Time taken: {training_time:.1f}s")
            
            # Save training logs
            with open(os.path.join(output_dir, 'train_stdout.log'), 'w') as f:
                f.write(result.stdout)
            if result.stderr:
                with open(os.path.join(output_dir, 'train_stderr.log'), 'w') as f:
                    f.write(result.stderr)
                    
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            return None
    
    # Run evaluation
    logger.info(f"Starting evaluation...")
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
    
    # Add dataset-specific parameters
    if "drugood_subset" in config and config["drugood_subset"]:
        eval_cmd.extend(["--drugood_subset", config["drugood_subset"]])
    
    if "good_domain" in config:
        eval_cmd.extend(["--good_domain", config["good_domain"]])
        eval_cmd.extend(["--good_shift", config["good_shift"]])
    
    try:
        result = subprocess.run(eval_cmd, capture_output=True, text=True, check=True)
        logger.info(f"Evaluation completed!")
        
        # Save evaluation logs
        with open(os.path.join(output_dir, 'eval_stdout.log'), 'w') as f:
            f.write(result.stdout)
        if result.stderr:
            with open(os.path.join(output_dir, 'eval_stderr.log'), 'w') as f:
                f.write(result.stderr)
        
        # Try to parse results
        results_file = os.path.join(output_dir, 'ood_evaluation_results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            logger.info(f"Results - AUROC: {results.get('auroc', 'N/A'):.4f}, "
                       f"AUPR: {results.get('aupr', 'N/A'):.4f}, "
                       f"FPR95: {results.get('fpr95', 'N/A'):.4f}")
            return results
        else:
            logger.warning(f"Results file not found: {results_file}")
            return None
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return None

def generate_loss_type_metrics_file(base_output_dir, loss_type, stats, num_successful_runs, total_seeds):
    """Generate separate metrics file for each loss type, format similar to dataset_metrics.json in run_baselines.sh"""
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

    logger.info(f"Generated {loss_type} metrics file: {metrics_file}")

def collect_and_summarize_results(base_output_dir, loss_types, seeds, foundation_model, dataset_name):
    """Collect and summarize all experiment results"""
    logger.info("Collecting and summarizing experiment results...")
    
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
                    logger.warning(f"Failed to read results file {results_file}: {e}")
            else:
                logger.warning(f"Results file not found: {results_file}")
        
        # Calculate statistics
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

            # Generate separate metrics file for each loss type (similar to run_baselines.sh)
            if summary["results"][loss_type]["summary_stats"]:
                generate_loss_type_metrics_file(base_output_dir, loss_type, summary["results"][loss_type]["summary_stats"], len(auroc_scores), len(seeds))

    # Save summary results
    summary_file = os.path.join(base_output_dir, 'ablation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print summary table
    logger.info("=" * 80)
    logger.info("Ablation Experiment Results Summary")
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
    logger.info(f"Detailed results saved to: {summary_file}")
    
    return summary

def main():
    setup_logging()
    args = parse_args()
    
    # Determine loss function types to run
    if args.only_loss_types:
        loss_types = args.only_loss_types
    else:
        loss_types = ["bce", "mse", "hinge"]
    
    logger.info("Starting Energy-DPO ablation experiments")
    logger.info(f"Loss function types: {loss_types}")
    logger.info(f"Random seeds: {args.seeds}")
    logger.info(f"Foundation model: {args.foundation_model}")
    logger.info(f"Dataset: {args.dataset}/{args.drugood_subset}")
    
    # Create aligned output directory structure (similar to run_baselines.sh)
    # Structure: ablation_results/{foundation_model}/{dataset_name}/{loss_type}_seed_{seed}/
    dataset_name = get_dataset_name(args)
    base_output_dir = os.path.join(args.base_output_dir, args.foundation_model, dataset_name)
    os.makedirs(base_output_dir, exist_ok=True)
    
    logger.info(f"Experiment results will be saved to: {base_output_dir}")
    logger.info(f"Directory structure: {args.base_output_dir}/{args.foundation_model}/{dataset_name}/{{loss_type}}_seed_{{seed}}/")
    
    # Run all experiments
    total_experiments = len(loss_types) * len(args.seeds)
    completed_experiments = 0
    
    for loss_type in loss_types:
        for seed in args.seeds:
            experiment_name = f"{loss_type}_seed_{seed}"
            experiment_dir = os.path.join(base_output_dir, experiment_name)
            
            logger.info(f"Experiment {completed_experiments + 1}/{total_experiments}: {experiment_name}")
            
            # Create experiment configuration
            config = create_experiment_config(args, loss_type, seed, experiment_dir)
            
            # Run experiment
            results = run_single_experiment(config, skip_training=args.skip_training)
            
            completed_experiments += 1
            
            if results:
                logger.info(f"Experiment {experiment_name} completed")
            else:
                logger.error(f"Experiment {experiment_name} failed")
            
            logger.info(f"Progress: {completed_experiments}/{total_experiments}")
            logger.info("-" * 50)
    
    # Collect and summarize results
    summary = collect_and_summarize_results(base_output_dir, loss_types, args.seeds, args.foundation_model, dataset_name)
    
    logger.info("All ablation experiments completed!")
    logger.info(f"View detailed results: {base_output_dir}/ablation_summary.json")

if __name__ == "__main__":
    main()
