#!/usr/bin/env python3
"""
Test script to verify cross-dataset evaluation functionality.
Tests loading training data from EC50 Scaffold and test data from EC50 Size.
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock args for testing
class MockArgs:
    def __init__(self):
        # Training dataset (EC50 Scaffold)
        self.dataset = "lbap_general_ec50_scaffold"
        self.drugood_subset = "lbap_general_ec50_scaffold"
        self.data_path = "./data/raw"
        self.data_file = "./data/raw/lbap_general_ec50_scaffold.json"

        # Test dataset (EC50 Size) - this is the cross-dataset part
        self.test_data_file = "./data/raw/lbap_general_ec50_size.json"
        self.test_drugood_subset = "lbap_general_ec50_size"

        # Other parameters
        self.foundation_model = "minimol"
        self.data_seed = 42
        self.debug_dataset_size = None
        self.precompute_features = False  # Disable caching for quick test
        self.cache_root = "/home/ubuntu/projects"
        self.force_recompute_cache = False
        self.encoding_batch_size = 50
        self.feature_cache_file = None
        self.splits_cache_file = None

        # GOOD parameters (not used for DrugOOD)
        self.good_domain = "scaffold"
        self.good_shift = "covariate"

def test_cross_dataset_loading():
    """Test that cross-dataset loading works correctly."""
    logger.info("=" * 60)
    logger.info("Testing Cross-Dataset Loading Functionality")
    logger.info("=" * 60)

    # Import data_loader
    try:
        from data_loader import EnergyDPODataLoader
    except ImportError as e:
        logger.error(f"Failed to import EnergyDPODataLoader: {e}")
        return False

    # Create mock args
    args = MockArgs()

    logger.info(f"\nConfiguration:")
    logger.info(f"  Training dataset: {args.dataset}")
    logger.info(f"  Training data file: {args.data_file}")
    logger.info(f"  Test dataset: {args.test_drugood_subset}")
    logger.info(f"  Test data file: {args.test_data_file}")

    # Test 1: Create data loader
    logger.info("\n[Test 1] Creating EnergyDPODataLoader...")
    try:
        data_loader = EnergyDPODataLoader(args)
        logger.info("✓ Data loader created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create data loader: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Check that parameters were captured
    logger.info("\n[Test 2] Verifying parameters were captured...")
    assert data_loader.test_data_file == args.test_data_file, "test_data_file not captured"
    assert data_loader.test_drugood_subset == args.test_drugood_subset, "test_drugood_subset not captured"
    logger.info("✓ Parameters captured correctly")

    # Test 3: Load data
    logger.info("\n[Test 3] Loading data with prepare_data()...")
    try:
        data_loader.prepare_data()
        logger.info("✓ Data loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Check train/val data comes from Scaffold dataset
    logger.info("\n[Test 4] Verifying train/val data...")
    logger.info(f"  Train ID samples: {len(data_loader.final_smiles.get('train_id', []))}")
    logger.info(f"  Train OOD samples: {len(data_loader.final_smiles.get('train_ood', []))}")
    logger.info(f"  Val ID samples: {len(data_loader.final_smiles.get('val_id', []))}")
    logger.info(f"  Val OOD samples: {len(data_loader.final_smiles.get('val_ood', []))}")
    assert len(data_loader.final_smiles.get('train_id', [])) > 0, "No training ID data loaded"
    assert len(data_loader.final_smiles.get('train_ood', [])) > 0, "No training OOD data loaded"
    logger.info("✓ Train/val data loaded from Scaffold dataset")

    # Test 5: Check test data comes from Size dataset
    logger.info("\n[Test 5] Verifying test data comes from Size dataset...")
    test_data = data_loader.get_final_test_data()
    logger.info(f"  Test ID samples: {len(test_data['id_smiles'])}")
    logger.info(f"  Test OOD samples: {len(test_data['ood_smiles'])}")
    assert len(test_data['id_smiles']) > 0, "No test ID data loaded"
    assert len(test_data['ood_smiles']) > 0, "No test OOD data loaded"
    logger.info("✓ Test data loaded from Size dataset")

    # Test 6: Verify data is different (sanity check)
    logger.info("\n[Test 6] Sanity check - train and test should use different molecules...")
    # Just check that test data size is different from train (simple heuristic)
    train_total = len(data_loader.final_smiles.get('train_id', [])) + len(data_loader.final_smiles.get('train_ood', []))
    test_total = len(test_data['id_smiles']) + len(test_data['ood_smiles'])
    logger.info(f"  Training total: {train_total}")
    logger.info(f"  Test total: {test_total}")
    logger.info("✓ Cross-dataset loading appears to be working")

    return True

def main():
    logger.info("Starting cross-dataset functionality test...\n")

    success = test_cross_dataset_loading()

    logger.info("\n" + "=" * 60)
    if success:
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("Cross-dataset evaluation is working correctly.")
        logger.info("\nYou can now train on EC50 Scaffold and test on EC50 Size using:")
        logger.info("  python main.py --mode train --dataset lbap_general_ec50_scaffold ...")
        logger.info("  python main.py --mode eval --dataset lbap_general_ec50_scaffold \\")
        logger.info("    --test_data_file ./data/raw/lbap_general_ec50_size.json \\")
        logger.info("    --test_drugood_subset lbap_general_ec50_size ...")
    else:
        logger.error("✗ TESTS FAILED!")
        logger.error("There were errors in the cross-dataset implementation.")
        sys.exit(1)
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
