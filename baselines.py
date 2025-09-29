#!/usr/bin/env python3
"""
Baseline OOD detection methods for molecular graphs.
Seamlessly integrates with SupervisedBaselineDataLoader to ensure proper data splitting and foundation model usage.

Usage:
    python baselines.py --dataset drugood_lbap_core_ic50_assay --foundation_model unimol --method all
    python baselines.py --dataset good_hiv --foundation_model minimol --method msp --good_domain scaffold
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path
import numpy as np

import torch

# Import baseline training functions
from baseline_trainer import run_all_baselines, run_baseline_training, run_baseline_evaluation


def parse_args():
    """Parse command line arguments compatible with SupervisedBaselineDataLoader."""
    parser = argparse.ArgumentParser(description="Run baseline OOD detection methods for molecular graphs")
    
    # Dataset arguments (matching SupervisedBaselineDataLoader exactly)
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name (e.g., 'drugood_lbap_core_ic50_assay', 'good_hiv', 'good_pcba', 'good_zinc')")
    
    parser.add_argument("--foundation_model", type=str, default="unimol",
                       choices=['unimol', 'minimol'], 
                       help="Foundation model for molecular representation")
    
    # Data paths and files
    parser.add_argument("--data_path", type=str, default="./data/raw",
                       help="Path to data directory")
    
    parser.add_argument("--data_file", type=str, default=None,
                       help="Specific data file path (for DrugOOD datasets)")
    
    parser.add_argument("--cache_root", type=str, default="./cache",
                       help="Root directory for feature caching")
    
    # Data sampling parameters (matching SupervisedBaselineDataLoader)
    parser.add_argument("--debug_dataset_size", type=int, default=None,
                       help="Limit dataset size for debugging")
    
    parser.add_argument("--data_seed", type=int, default=42,
                       help="Seed for data splitting")
    
    # GOOD dataset specific parameters
    parser.add_argument("--good_domain", type=str, default="scaffold",
                       choices=['scaffold', 'size'],
                       help="Domain for GOOD datasets")
    
    parser.add_argument("--good_shift", type=str, default="covariate", 
                       choices=['covariate', 'concept', 'no_shift'],
                       help="Shift type for GOOD datasets")
    
    # DrugOOD specific parameters  
    parser.add_argument("--drugood_subset", type=str, default=None,
                       help="DrugOOD subset name (auto-detected if not provided)")
    
    # Feature computation parameters (matching SupervisedBaselineDataLoader)
    parser.add_argument("--precompute_features", action="store_true", default=True,
                       help="Enable feature pre-computation and caching")
    
    parser.add_argument("--force_recompute_cache", action="store_true", default=False,
                       help="Force recompute cached features")
    
    parser.add_argument("--encoding_batch_size", type=int, default=128,
                       help="Batch size for feature encoding")
    
    # Model architecture
    parser.add_argument("--hidden_channels", type=int, default=64,
                       help="Hidden dimension for classifier")
    
    parser.add_argument("--num_layers", type=int, default=3,
                       help="Number of layers in classifier")
    
    parser.add_argument("--dropout", type=float, default=0.5,
                       help="Dropout rate")
    
    # Training parameters
    parser.add_argument("--lr", type=float, default=0.01,
                       help="Learning rate")
    
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                       help="Weight decay")
    
    parser.add_argument("--epochs", type=int, default=200,
                       help="Maximum number of training epochs")
    
    parser.add_argument("--patience", type=int, default=20,
                       help="Early stopping patience")
    
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Training batch size")
    
    parser.add_argument("--eval_batch_size", type=int, default=64,
                       help="Evaluation batch size")
    
    # Baseline method selection
    parser.add_argument("--method", type=str, default="all",
                       choices=['all', 'msp', 'odin', 'energy', 'mahalanobis', 'knn', 'lof', 'dam_msp', 'dam_energy'],
                       help="Which baseline method(s) to run")
    
    # ODIN-specific parameters
    parser.add_argument("--T", type=float, default=10.0,
                       help="Temperature parameter for ODIN method")
    
    parser.add_argument("--noise", type=float, default=0.0014,
                       help="Noise magnitude for ODIN method")

    # KNN-specific parameters
    parser.add_argument("--knn_k", type=int, default=50,
                       help="K parameter for KNN method")

    # LOF-specific parameters
    parser.add_argument("--lof_neighbors", type=int, default=20,
                       help="Number of neighbors for LOF method")

    # DAM-specific parameters
    parser.add_argument("--dam_lr", type=float, default=0.1,
                       help="PESG learning rate for DAM")

    parser.add_argument("--dam_margin", type=float, default=1.0,
                       help="AUCMLoss margin for DAM")
    
    # Output and system
    parser.add_argument("--output_dir", type=str, default="./baseline_results",
                       help="Output directory for results and checkpoints")
    
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of workers for data loading")
    
    # General settings
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    parser.add_argument("--verbose", action="store_true", default=False,
                       help="Enable verbose logging")
    
    return parser.parse_args()


def setup_logging(args):
    """Set up logging configuration."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create log filename
    dataset_name = args.dataset.replace('/', '_').replace('-', '_')
    foundation_model = args.foundation_model
    method_name = args.method
    log_filename = f"baseline_{dataset_name}_{foundation_model}_{method_name}.log"
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, log_filename)),
            logging.StreamHandler(sys.stdout)
        ]
    )


def set_random_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_args(args):
    """Validate arguments and set defaults."""
    logger = logging.getLogger(__name__)
    
    # Auto-detect DrugOOD subset if not provided
    if not hasattr(args, 'drugood_subset') or args.drugood_subset is None:
        if 'drugood' in args.dataset.lower() or 'lbap' in args.dataset.lower():
            args.drugood_subset = args.dataset.replace('drugood_', '').replace('drugood-', '').replace('-', '_')
            logger.info(f"Auto-detected DrugOOD subset: {args.drugood_subset}")
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    # Validate data paths
    if not os.path.exists(args.data_path):
        logger.warning(f"Data path does not exist: {args.data_path}")
        logger.info("SupervisedBaselineDataLoader will handle data loading internally")
    
    # Set reasonable defaults based on dataset
    if 'good' in args.dataset.lower():
        # GOOD datasets are typically larger, ensure caching is enabled
        if not args.precompute_features:
            logger.info("Enabling feature caching for GOOD dataset")
            args.precompute_features = True
    
    # Validate foundation model
    if args.foundation_model not in ['unimol', 'minimol']:
        logger.warning(f"Unknown foundation model: {args.foundation_model}, using 'unimol'")
        args.foundation_model = 'unimol'
    
    return args


def check_dependencies():
    """Check if required dependencies are available."""
    logger = logging.getLogger(__name__)
    
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
    except ImportError:
        logger.error("PyTorch not found! Please install PyTorch")
        return False
    
    try:
        from SupervisedBaselineDataLoader import SupervisedBaselineDataLoader
        logger.info("SupervisedBaselineDataLoader found")
    except ImportError:
        logger.error("Cannot import SupervisedBaselineDataLoader! Please check SupervisedBaselineDataLoader.py")
        return False
    
    try:
        from baseline_trainer import BaselineTrainer, BaselineEvaluator
        logger.info("Baseline trainer and evaluator found")
    except ImportError:
        logger.error("Cannot import baseline_trainer! Please check baseline_trainer.py")
        return False
    
    try:
        from baselinemodel import GCN_Classifier, BaselineOODModel
        logger.info("Baseline models found")
    except ImportError:
        logger.error("Cannot import baselinemodel! Please check baselinemodel.py")
        return False
    
    try:
        import sklearn
        logger.info(f"scikit-learn version: {sklearn.__version__}")
    except ImportError:
        logger.warning("scikit-learn not found, Mahalanobis method may not work")
    
    try:
        import rdkit
        logger.info("RDKit found for molecular property computation")
    except ImportError:
        logger.warning("RDKit not found, using alternative label generation")
    
    # Test if model.py is available for feature encoding
    try:
        from model import MinimolEncoder, UniMolEncoder
        logger.info("Foundation model encoders found")
    except ImportError:
        logger.warning("Foundation model encoders not found - ensure model.py is available")
        logger.warning("Feature caching may not work without foundation model encoders")
    
    return True


def print_startup_info(args):
    """Print detailed startup information."""
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("SUPERVISED BASELINE OOD DETECTION FOR MOLECULAR GRAPHS")
    logger.info("="*80)
    logger.info(f"Dataset:           {args.dataset}")
    logger.info(f"Foundation Model:  {args.foundation_model}")
    logger.info(f"Method:            {args.method}")
    logger.info(f"Data Path:         {args.data_path}")
    logger.info(f"Cache Root:        {args.cache_root}")
    logger.info(f"Output Dir:        {args.output_dir}")
    logger.info(f"Device:            {args.device}")
    logger.info(f"Random Seed:       {args.seed}")
    logger.info(f"Data Seed:         {args.data_seed}")
    logger.info(f"Feature Caching:   {args.precompute_features}")
    
    if 'good' in args.dataset.lower():
        logger.info(f"GOOD Domain:       {args.good_domain}")
        logger.info(f"GOOD Shift:        {args.good_shift}")
    
    if hasattr(args, 'drugood_subset') and args.drugood_subset:
        logger.info(f"DrugOOD Subset:    {args.drugood_subset}")
    
    logger.info("-"*80)
    logger.info("Model Configuration:")
    logger.info(f"  Hidden Channels:  {args.hidden_channels}")
    logger.info(f"  Num Layers:       {args.num_layers}")
    logger.info(f"  Dropout:          {args.dropout}")
    logger.info(f"  Learning Rate:    {args.lr}")
    logger.info(f"  Batch Size:       {args.batch_size}")
    logger.info(f"  Max Epochs:       {args.epochs}")
    
    if args.method == 'odin' or args.method == 'all':
        logger.info(f"ODIN Parameters:")
        logger.info(f"  Temperature:      {args.T}")
        logger.info(f"  Noise:            {args.noise}")
    
    logger.info("="*80)


def print_data_loading_info():
    """Print information about the data loading approach."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("DATA LOADING WITH SUPERVISED APPROACH")
    logger.info("="*80)
    logger.info("Key improvements over EnergyDPODataLoader:")
    logger.info("✅ Uses SupervisedBaselineDataLoader for real classification labels")
    logger.info("✅ Reuses existing data splits and feature caches for fair comparison")
    logger.info("✅ Provides proper supervised training with real target labels")
    logger.info("✅ Seamless integration with foundation model features")
    logger.info("✅ Comprehensive label caching system for efficiency")
    logger.info("-"*80)
    logger.info("Training Process:")
    logger.info("1. Load data with SupervisedBaselineDataLoader (inherits from EnergyDPODataLoader)")
    logger.info("2. Extract real classification labels for ID data")
    logger.info("3. Train supervised classifier on ID data with real labels")
    logger.info("4. Evaluate OOD detection using baseline methods (MSP, Energy, ODIN, Mahalanobis)")
    logger.info("5. Test on proper ID/OOD test splits")
    logger.info("="*80)


def main():
    """Main function to run baseline evaluation."""
    # Parse arguments and setup
    args = parse_args()
    args = validate_args(args)
    setup_logging(args)
    set_random_seed(args.seed)
    
    logger = logging.getLogger(__name__)
    
    # Print startup information
    print_startup_info(args)
    print_data_loading_info()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed!")
        sys.exit(1)
    
    try:
        logger.info("Starting supervised baseline evaluation...")
        
        if args.method == 'all':
            # Run comprehensive evaluation with all methods
            logger.info("Running comprehensive baseline evaluation...")
            logger.info("This will:")
            logger.info("1. Load data using SupervisedBaselineDataLoader with proper splitting")
            logger.info("2. Use foundation model features and real classification labels")
            logger.info("3. Train a single supervised classifier backbone")
            logger.info("4. Evaluate all baseline methods (MSP, Energy, ODIN, Mahalanobis)")
            
            results = run_all_baselines(args)
            
            # Print comprehensive summary
            logger.info("\n" + "="*80)
            logger.info("COMPREHENSIVE SUPERVISED BASELINE RESULTS SUMMARY")
            logger.info("="*80)
            logger.info(f"Dataset: {args.dataset}")
            logger.info(f"Foundation Model: {args.foundation_model}")
            logger.info("-"*80)
            
            header = f"{'Method':<12} | {'AUROC':>8} | {'AUPR':>8} | {'FPR95':>8} | {'Status':>10}"
            logger.info(header)
            logger.info("-" * len(header))
            
            for method, result in results.items():
                if result is not None:
                    logger.info(f"{method.upper():<12} | {result['auroc']:>8.4f} | "
                               f"{result['aupr']:>8.4f} | {result['fpr95']:>8.4f} | {'SUCCESS':>10}")
                else:
                    logger.info(f"{method.upper():<12} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'FAILED':>10}")
            
            # Find best method
            valid_results = {k: v for k, v in results.items() if v is not None}
            if valid_results:
                best_method, best_result = max(valid_results.items(), key=lambda x: x[1]['auroc'])
                logger.info("-"*80)
                logger.info(f"Best Method: {best_method.upper()} (AUROC: {best_result['auroc']:.4f})")
            
            logger.info("="*80)
            
        else:
            # Run single method
            logger.info(f"Running single baseline method: {args.method.upper()}")
            logger.info("This will:")
            logger.info("1. Load data using SupervisedBaselineDataLoader")
            logger.info("2. Train supervised classifier backbone with real labels")
            logger.info(f"3. Evaluate {args.method.upper()} baseline method")
            
            # Train the backbone
            logger.info("\nTraining supervised classifier backbone...")
            checkpoint_path = run_baseline_training(args)
            
            # Evaluate the specific method
            logger.info(f"\nEvaluating {args.method.upper()}...")
            results = run_baseline_evaluation(checkpoint_path, args.method, args)
            
            logger.info(f"\n{args.method.upper()} evaluation completed successfully!")
            logger.info(f"AUROC: {results['auroc']:.4f}")
            logger.info(f"Results saved in: {args.output_dir}")
        
        logger.info("\nSupervised baseline evaluation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Baseline evaluation failed: {str(e)}")
        logger.error("\nTroubleshooting tips:")
        logger.error("1. Check that SupervisedBaselineDataLoader.py exists and is properly configured")
        logger.error("2. Ensure utils.py contains process_drugood_data and process_good_data functions")
        logger.error("3. Verify that baselinemodel.py contains GCN_Classifier and BaselineOODModel classes")
        logger.error("4. Check that model.py contains foundation model encoders (MinimolEncoder, UniMolEncoder)")
        logger.error("5. Ensure dataset files exist and are accessible")
        logger.error("6. Try with --verbose flag for more detailed logging")
        logger.error("7. Try with --debug_dataset_size 1000 for faster testing")
        
        if args.verbose:
            import traceback
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
        
        raise


if __name__ == "__main__":
    main()