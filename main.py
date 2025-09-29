import argparse
import logging
import os
import sys
import torch
import random
import numpy as np
import datetime
import warnings

warnings.filterwarnings("ignore", message="Failed to find the pandas get_adjustment.*")
warnings.filterwarnings("ignore", message="Failed to patch pandas.*")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Energy-DPO for OOD Detection")

    # Basic parameters
    parser.add_argument("--mode", type=str, required=True, choices=['train', 'eval', 'predict'],
                        help="Run mode: train, eval, or predict")
    parser.add_argument("--seed", type=int, default=38, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--data_seed", type=int, default=42, 
                        help="Data sampling seed (fixed to ensure consistent data across experiments)")

    # Foundation model selection
    parser.add_argument("--foundation_model", type=str, default="minimol", 
                        choices=['minimol', 'unimol'],
                        help="Foundation model encoder: minimol or unimol")

    # Dataset parameters
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., drugood, good_hiv, etc.)")
    parser.add_argument("--data_path", type=str, default="./data", help="Data root directory")
    parser.add_argument("--data_file", type=str, default=None, help="Specific data file path (mainly for DrugOOD)")

    # DrugOOD specific parameters
    parser.add_argument("--drugood_subset", type=str, default=None,
                        help="DrugOOD subset. If not provided, will use --dataset value")

    # GOOD data specific parameters
    parser.add_argument("--good_domain", type=str, default="scaffold", choices=['scaffold', 'size'],
                        help="GOOD data domain selection")
    parser.add_argument("--good_shift", type=str, default="covariate",
                        choices=['covariate', 'concept', 'no_shift'],
                        help="GOOD data shift selection")

    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--loss_type", type=str, default="dpo", 
                   choices=["dpo", "bce", "mse", "hinge"],
                   help="Loss function type")
    parser.add_argument("--hinge_margin", type=float, default=1.0, help="Margin parameter for Hinge Loss")
    parser.add_argument("--hinge_topk", type=float, default=0.0, help="Hinge in-batch hard mining ratio (0 to disable, 0.25 for hardest 25%)")
    parser.add_argument("--hinge_squared", action='store_true', help="Use squared Hinge loss: relu(m-Δ)^2")
    parser.add_argument("--lambda_reg", type=float, default=1e-2, help="L2 regularization coefficient λ for output energy")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=256, help="Evaluation batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation step interval")
    parser.add_argument("--save_steps", type=int, default=1000, help="Checkpoint saving step interval")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    
    # Cache related parameters - UPDATED CACHE PATH
    parser.add_argument("--precompute_features", action='store_true', default=True, help="Precompute feature cache")
    parser.add_argument("--force_recompute_cache", action='store_true', help="Force recompute cache")
    parser.add_argument("--cache_root", type=str, default="/home/ubuntu/projects", help="Cache root directory")
    parser.add_argument("--encoding_batch_size", type=int, default=50, help="Encoding batch size")
    # Optional external caches
    parser.add_argument("--feature_cache_file", type=str, default=None,
                        help="Path to a precomputed feature cache pkl to use directly")
    parser.add_argument("--splits_cache_file", type=str, default=None,
                        help="Path to a precomputed splits json to use directly")

    

    # Output and logging parameters
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # Evaluation and prediction parameters
    parser.add_argument("--model_path", type=str, default=None, help="Pre-trained model path (for eval and predict)")
    parser.add_argument("--test_smiles", type=str, nargs='+', help="SMILES list for prediction")
    parser.add_argument("--test_file", type=str, help="File containing test SMILES")

    # Debug parameters
    parser.add_argument("--debug", action='store_true', help="Debug mode")
    parser.add_argument("--debug_dataset_size", type=int, default=None, help="Dataset size for debugging")
    
    # DataLoader parameters
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loader workers")
    
    # Dataset type auto-inference
    parser.add_argument("--dataset_type", type=str, default=None, 
                        choices=['drugood', 'good'], 
                        help="Dataset type (usually auto-inferred)")

    return parser.parse_args()

def get_exp_name(args):
    """Generate experiment name"""
    if args.exp_name:
        return args.exp_name
    
    base_name = f"{args.foundation_model}_{args.dataset}"
    
    if args.dataset_type == 'drugood' and args.drugood_subset:
        base_name = f"{args.foundation_model}_{args.drugood_subset}"
    elif args.dataset_type == 'good':
        base_name = f"{args.foundation_model}_{args.dataset}_{args.good_domain}_{args.good_shift}"
    
    param_str = f"beta{args.dpo_beta}_lr{args.lr}_hdim{args.hidden_dim}"
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    
    return f"{base_name}_{param_str}_{timestamp}"

def setup_output_dir(args):
    """Setup output directory"""
    if args.output_dir is None:
        exp_name = get_exp_name(args)
        args.output_dir = os.path.join("./runs", exp_name)
        args.exp_name = exp_name
    
    os.makedirs(args.output_dir, exist_ok=True)
    return args

def setup_logging(log_level, output_dir):
    """Setup logging"""
    log_file = os.path.join(output_dir, "training.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    return logger

def set_seed(seed):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_required_modules():
    """Check if required module files exist"""
    logger = logging.getLogger(__name__)
    
    try:
        from train import EnergyDPOTrainer
        from evaluation import EnergyDPOEvaluator
        logger.info("All required modules loaded")
        return True
    except ImportError as e:
        logger.warning(f"Some modules unavailable: {e}")
        return False

def validate_args(args):
    """Validate arguments"""
    logger = logging.getLogger(__name__)
    
    # Auto-infer dataset type
    if args.dataset_type is None:
        if args.dataset.lower().startswith('good_'):
            args.dataset_type = 'good'
        else:
            args.dataset_type = 'drugood'
    
    # Auto-find model path for eval and predict modes if not provided
    if args.mode in ['eval', 'predict'] and not args.model_path:
        # Try to find model files in output directory
        if args.output_dir and os.path.exists(args.output_dir):
            # Prioritize using best_model.pth
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                args.model_path = best_model_path
                logger.info(f"Using best model for evaluation: {args.model_path}")
            else:
                # If no best_model.pth, use the latest model file
                model_files = [f for f in os.listdir(args.output_dir) if f.endswith('.pth')]
                if model_files:
                    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
                    args.model_path = os.path.join(args.output_dir, latest_model)
                    logger.warning(f"best_model.pth not found, using latest model: {args.model_path}")
                else:
                    raise ValueError(f"{args.mode} mode requires --model_path argument. No model files found in {args.output_dir}")
        else:
            raise ValueError(f"{args.mode} mode requires --model_path argument")
    
    # Validate prediction input
    if args.mode == 'predict' and not args.test_smiles and not args.test_file:
        raise ValueError("Predict mode requires --test_smiles or --test_file argument")
    
    logger.info("Argument validation passed")

def find_model_file(args):
    """Auto-find model file (simplified version, main logic is in validate_args)"""
    logger = logging.getLogger(__name__)
    
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"Using specified model file: {args.model_path}")
        return args
    
    # Log warning if model path is still problematic
    if args.mode in ['eval', 'predict']:
        logger.warning(f"Model path verification: {args.model_path}")
    
    return args

def run_training(args):
    """Run training"""
    from train import EnergyDPOTrainer
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting training (Foundation Model: {args.foundation_model})")
    
    trainer = EnergyDPOTrainer(args)
    trainer.train()
    
    logger.info("Training completed")

def run_evaluation(args):
    """Run evaluation"""
    from evaluation import EnergyDPOEvaluator
    from data_loader import EnergyDPODataLoader
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting evaluation (Foundation Model: {args.foundation_model})")
    logger.info(f"Model path: {args.model_path}")
    
    evaluator = EnergyDPOEvaluator(args.model_path, args)
    
    # Load data and prepare features
    logger.info("Creating data loader...")
    data_loader = EnergyDPODataLoader(args)
    
    logger.info("Getting test data...")
    test_data = data_loader.get_final_test_data()
    id_smiles = test_data['id_smiles']
    ood_smiles = test_data['ood_smiles']
    
    logger.info(f"Test data - ID: {len(id_smiles)}, OOD: {len(ood_smiles)}")
    
    # Try to use precomputed features if available
    try:
        if hasattr(data_loader, 'feature_cache') and data_loader.feature_cache:
            id_features = data_loader._get_features_for_smiles(id_smiles)
            ood_features = data_loader._get_features_for_smiles(ood_smiles)
            logger.info(f"Using precomputed features - ID: {id_features.shape}, OOD: {ood_features.shape}")
            
            results = evaluator.evaluate_ood_detection_from_features(
                id_features, ood_features, output_dir=args.output_dir
            )
        else:
            logger.info("Using SMILES-based evaluation")
            results = evaluator.evaluate_ood_detection(
                id_smiles, ood_smiles, output_dir=args.output_dir
            )
    except Exception as e:
        logger.warning(f"Precomputed features failed: {e}, falling back to SMILES")
        results = evaluator.evaluate_ood_detection(
            id_smiles, ood_smiles, output_dir=args.output_dir
        )
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("Evaluation Summary")
    logger.info("="*50)
    logger.info(f"AUROC: {results['auroc']:.4f}")
    logger.info(f"AUPR: {results['aupr']:.4f}")
    logger.info(f"FPR95: {results['fpr95']:.4f}")
    logger.info(f"Energy Separation: {results['energy_separation']:.4f}")
    logger.info(f"ID Mean Energy: {results['id_mean_energy']:.4f}")
    logger.info(f"OOD Mean Energy: {results['ood_mean_energy']:.4f}")
    logger.info("="*50)
    
    logger.info("Evaluation completed")

def run_prediction(args):
    """Run prediction"""
    from evaluation import EnergyDPOEvaluator
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting prediction (Foundation Model: {args.foundation_model})")
    logger.info(f"Model path: {args.model_path}")
    
    # Prepare test SMILES
    if args.test_file:
        with open(args.test_file, 'r') as f:
            test_smiles = [line.strip() for line in f if line.strip()]
    else:
        test_smiles = args.test_smiles
    
    logger.info(f"Number of molecules to predict: {len(test_smiles)}")
    
    # Create evaluator and predict
    evaluator = EnergyDPOEvaluator(args.model_path, args)
    scores = evaluator.predict_batch(test_smiles)
    
    # Output results
    print("\n" + "="*50)
    print("Prediction Results")
    print("="*50)
    print(f"Foundation Model: {args.foundation_model}")
    for smiles, score in zip(test_smiles, scores):
        print(f"{smiles}: {score:.4f}")
    print("="*50)
    
    # Save results
    output_file = os.path.join(args.output_dir, "prediction_results.txt")
    with open(output_file, 'w') as f:
        f.write(f"Foundation Model: {args.foundation_model}\n")
        f.write("SMILES\tOOD_Score\n")
        for smiles, score in zip(test_smiles, scores):
            f.write(f"{smiles}\t{score:.4f}\n")
    
    logger.info(f"Prediction completed, results saved to: {output_file}")

def main():
    # Parse arguments
    args = parse_args()

    # Post-process arguments
    if args.dataset_type == 'drugood' and args.drugood_subset is None:
        args.drugood_subset = args.dataset

    # Setup output directory
    args = setup_output_dir(args)

    # Setup logging
    logger = setup_logging(args.log_level, args.output_dir)

    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")
    logger.info(f"Data seed set to: {args.data_seed}")

    # Print core parameters
    logger.info(f"Run mode: {args.mode}")
    logger.info(f"Foundation model: {args.foundation_model}")
    logger.info(f"Dataset: {args.dataset} (type: {args.dataset_type})")
    if args.dataset_type == 'drugood':
        logger.info(f"DrugOOD subset: {args.drugood_subset}")
    else:
        logger.info(f"GOOD Domain: {args.good_domain}, Shift: {args.good_shift}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {args.device}")

    # Validate arguments
    validate_args(args)

    # Check required modules
    modules_available = check_required_modules()
    if not modules_available and args.mode in ['train', 'eval']:
        logger.warning("Required modules unavailable, but will attempt to run...")

    # Find model file if needed
    if args.mode in ['eval', 'predict']:
        args = find_model_file(args)

    # Run corresponding functionality based on mode
    try:
        if args.mode == 'train':
            run_training(args)
        elif args.mode == 'eval':
            run_evaluation(args)
        elif args.mode == 'predict':
            run_prediction(args)
    except Exception as e:
        logger.error(f"Program execution failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
