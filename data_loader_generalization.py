import logging
import os
import pickle
import random
import json
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Disable pandas/RDKit warnings
warnings.filterwarnings("ignore", message="Failed to find the pandas get_adjustment.*")
warnings.filterwarnings("ignore", message="Failed to patch pandas.*")

# Import utility functions from utils.py
from utils import process_drugood_data, process_good_data

# Set up a logger for this module.
logger = logging.getLogger(__name__)


class EnergyDPODataset(Dataset):
    """Dataset providing ID/OOD SMILES pairs for Energy-DPO training."""

    def __init__(self, id_smiles_list, ood_smiles_list, mode='train', seed=None):
        self.id_smiles_list = id_smiles_list
        self.ood_smiles_list = ood_smiles_list
        self.mode = mode
        self.length = min(len(self.id_smiles_list), len(self.ood_smiles_list))
                # Use a dedicated RNG for reproducible sampling
        self.rng = random.Random(seed)
        logger.info(
            f"EnergyDPODataset ({mode}): ID={len(id_smiles_list)}, OOD={len(ood_smiles_list)}, effective length={self.length}"
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode == 'train':
            # For training, randomly sample to increase diversity.
            ood_idx = self.rng.randint(0, len(self.ood_smiles_list) - 1)
            id_idx = self.rng.randint(0, len(self.id_smiles_list) - 1)
        else:
            # For evaluation, use deterministic indexing.
            ood_idx = idx % len(self.ood_smiles_list)
            id_idx = idx % len(self.id_smiles_list)

        return {
            'ood_smiles': self.ood_smiles_list[ood_idx],
            'id_smiles': self.id_smiles_list[id_idx],
        }


class PrecomputedEnergyDPODataset(Dataset):
    """Dataset using pre-computed feature tensors for ID and OOD SMILES."""

    def __init__(self, id_features, ood_features, id_smiles_list, ood_smiles_list, mode='train', seed=None):
        self.id_features = id_features
        self.ood_features = ood_features
        self.id_smiles_list = id_smiles_list
        self.ood_smiles_list = ood_smiles_list
        self.mode = mode
        self.length = min(len(self.id_smiles_list), len(self.ood_smiles_list))
        self.rng = random.Random(seed)
        logger.info(
            f"Precomputed Dataset ({mode}): ID={len(id_smiles_list)}, OOD={len(ood_smiles_list)}, effective length={self.length}"
        )
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode == 'train':
            ood_idx = self.rng.randint(0, len(self.ood_smiles_list) - 1)
            id_idx = self.rng.randint(0, len(self.id_smiles_list) - 1)
        else:
            ood_idx = idx % len(self.ood_smiles_list)
            id_idx = idx % len(self.id_smiles_list)

        return {
            'id_features': self.id_features[id_idx],
            'ood_features': self.ood_features[ood_idx],
            'id_smiles': self.id_smiles_list[id_idx],
            'ood_smiles': self.ood_smiles_list[ood_idx],
        }


def energy_dpo_collate_fn(batch):
    """Collates a batch of SMILES strings."""
    return {
        'ood_smiles': [item['ood_smiles'] for item in batch],
        'id_smiles': [item['id_smiles'] for item in batch],
    }


def precomputed_energy_dpo_collate_fn(batch):
    """Collates a batch of precomputed features and SMILES strings."""
    return {
        'id_features': torch.stack([item['id_features'] for item in batch]),
        'ood_features': torch.stack([item['ood_features'] for item in batch]),
        'id_smiles': [item['id_smiles'] for item in batch],
        'ood_smiles': [item['ood_smiles'] for item in batch],
    }


class EnergyDPODataLoader:
    """
    Handles loading, processing, and caching of ID/OOD datasets for Energy-based DPO.

    This class orchestrates the entire data pipeline:
    1. Loads raw SMILES data from specified dataset files (DrugOOD, GOOD).
    2. Sub-samples the data to target sizes for train, validation, and test sets.
    3. Manages a feature cache to avoid recomputing expensive molecular representations.
    4. Computes features for any molecules not found in the cache using a specified foundation model.
    5. Provides PyTorch DataLoader instances for training and evaluation.
    """

    def __init__(self, args):
        """Initializes the data loader with configuration from args."""
        self.args = args
        self.data_path = args.data_path
        self.data_file = getattr(args, 'data_file', None)
        self.dataset_name = args.dataset
        self.foundation_model = getattr(args, 'foundation_model', 'minimol')

        # Dataset-specific configurations
        self.drugood_subset = getattr(args, 'drugood_subset', 'lbap_general_ic50_scaffold')
        self.good_domain = getattr(args, 'good_domain', 'scaffold')
        self.good_shift = getattr(args, 'good_shift', 'covariate')

        # Test dataset configuration (for cross-dataset evaluation)
        self.test_data_file = getattr(args, 'test_data_file', None)
        self.test_drugood_subset = getattr(args, 'test_drugood_subset', None)

        # Control seeds and sampling
        self.data_seed = getattr(args, 'data_seed', 42)
        self.max_samples = getattr(args, 'debug_dataset_size', None)

        # Caching configuration - UPDATED CACHE PATH
        self.enable_cache = getattr(args, 'precompute_features', True)
        self.cache_dir = Path(getattr(args, 'cache_root', '/home/ubuntu/projects')) / 'ood_dpo_cache'
        self.force_recompute = getattr(args, 'force_recompute_cache', False)
        self.encoding_batch_size = getattr(args, 'encoding_batch_size', 128)
        # Optional external cache overrides
        self.external_feature_cache_file = Path(getattr(args, 'feature_cache_file', '')) if getattr(args, 'feature_cache_file', None) else None
        self.external_splits_cache_file = Path(getattr(args, 'splits_cache_file', '')) if getattr(args, 'splits_cache_file', None) else None

        # Data containers - maintain compatibility with original attribute names
        self._raw_smiles = {}
        self.final_smiles = {}
        self.feature_cache = {}
        
        # Additional attributes for backward compatibility
        self.train_id_smiles = []
        self.val_id_smiles = []
        self.train_ood_smiles = []
        self.val_ood_smiles = []
        self.final_test_id_smiles = []
        self.final_test_ood_smiles = []

        # CRITICAL FIX: Automatically prepare data in __init__ to maintain compatibility
        logger.info("=== Starting OOD Data Preparation ===")
        logger.info(f"Dataset: {self.dataset_name} | Foundation Model: {self.foundation_model} | Data Seed: {self.data_seed}")
        self.prepare_data()
        logger.info("Data preparation complete.")

    def prepare_data(self):
        """Main method to orchestrate data loading, processing, and feature computation."""
        # Fast path: try external splits cache, then internal splits cache
        if self.external_splits_cache_file and self._load_splits_cache(self.external_splits_cache_file):
            logger.info(f"Loaded data splits from external cache: {self.external_splits_cache_file}")
        elif self._load_splits_cache(self._get_splits_cache_path()):
            logger.info("Loaded data splits from cache early - skipping raw data loading")
        else:
            self._load_raw_data()
            self._select_final_smiles()

        # CRITICAL FIX: Always load separate test data when specified (even if cache was used)
        # This ensures cross-dataset evaluation uses the correct test set
        if self.test_data_file:
            logger.info(f"Loading separate test data from: {self.test_data_file}")
            self._load_test_data()
            # Update final_smiles with cross-dataset test data
            self.final_smiles['test_id'] = self._raw_smiles['test_id']
            self.final_smiles['test_ood'] = self._raw_smiles['test_ood']
            logger.info(f"Updated test data: {len(self.final_smiles['test_id'])} ID, {len(self.final_smiles['test_ood'])} OOD")

        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._prepare_features()

        # CRITICAL FIX: Update compatibility attributes
        self._update_compatibility_attributes()
        self._print_summary()

    def _update_compatibility_attributes(self):
        """Update attributes to maintain compatibility with existing code."""
        self.train_id_smiles = self.final_smiles.get('train_id', [])
        self.val_id_smiles = self.final_smiles.get('val_id', [])
        self.train_ood_smiles = self.final_smiles.get('train_ood', [])
        self.val_ood_smiles = self.final_smiles.get('val_ood', [])
        self.final_test_id_smiles = self.final_smiles.get('test_id', [])
        self.final_test_ood_smiles = self.final_smiles.get('test_ood', [])

    def _load_raw_data(self):
        """Loads the raw SMILES data from the appropriate source file."""
        dataset_lower = self.dataset_name.lower()
        if 'drugood' in dataset_lower or 'lbap' in dataset_lower:
            if not self.data_file:
                self.data_file = os.path.join(self.data_path, f"{self.drugood_subset}.json")
            if not os.path.exists(self.data_file):
                raise FileNotFoundError(f"Could not find DrugOOD data file: {self.data_file}")
            # Assume process_drugood_data is a function that reads the file and returns a dict of lists
            data_dict = process_drugood_data(self.data_file, self.max_samples)
            # Handle missing val_ood by splitting from train_ood
            if 'val_ood_smiles' not in data_dict or not data_dict['val_ood_smiles']:
                logger.warning("Validation OOD not found. Splitting from training OOD set.")
                train_ood = data_dict['train_ood_smiles']
                random.Random(self.data_seed).shuffle(train_ood)
                val_size = min(3000, len(train_ood) // 5)  # Use 20% or 3k for validation
                data_dict['val_ood_smiles'] = train_ood[:val_size]
                data_dict['train_ood_smiles'] = train_ood[val_size:]

        elif 'good' in dataset_lower:
            # Use GOOD processed splits, skipping RDKit validation for speed
            data_dict = process_good_data(
                dataset_name=self.dataset_name, domain=self.good_domain,
                shift=self.good_shift, data_path=self.data_path, max_samples=self.max_samples,
                validate_smiles_flag=False
            )
            # Cache GOOD data for potential reuse by subclasses
            self._good_data_dict = data_dict
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self._raw_smiles = {
            'train_id': data_dict.get('train_id_smiles', []),
            'train_ood': data_dict.get('train_ood_smiles', []),
            'val_id': data_dict.get('val_id_smiles', []),
            'val_ood': data_dict.get('val_ood_smiles', []),
            'test_id': data_dict.get('test_id_smiles', data_dict.get('val_id_smiles', [])),  # Fallback to val if test not present
            'test_ood': data_dict.get('test_ood_smiles', data_dict.get('val_ood_smiles', []))
        }
        logger.info("Raw data loaded. Found %d train_id, %d train_ood, %d val_id, %d val_ood smiles.",
                    len(self._raw_smiles['train_id']), len(self._raw_smiles['train_ood']),
                    len(self._raw_smiles['val_id']), len(self._raw_smiles['val_ood']))

        # Load separate test data if specified (for cross-dataset evaluation)
        if self.test_data_file:
            logger.info(f"Loading separate test data from: {self.test_data_file}")
            self._load_test_data()

    def _load_test_data(self):
        """Loads test data from a separate file for cross-dataset evaluation."""
        if not os.path.exists(self.test_data_file):
            raise FileNotFoundError(f"Could not find test data file: {self.test_data_file}")

        # Determine test dataset type
        test_dataset_lower = (self.test_drugood_subset or os.path.basename(self.test_data_file)).lower()

        if 'drugood' in test_dataset_lower or 'lbap' in test_dataset_lower:
            # Load DrugOOD test data
            test_data_dict = process_drugood_data(self.test_data_file, None)  # No sampling for test
        elif 'good' in test_dataset_lower:
            # Load GOOD test data
            test_data_dict = process_good_data(
                dataset_name=self.test_drugood_subset or 'good_hiv',
                domain=self.good_domain,
                shift=self.good_shift,
                data_path=os.path.dirname(self.test_data_file),
                max_samples=None,
                validate_smiles_flag=False
            )
        else:
            raise ValueError(f"Unsupported test dataset type: {test_dataset_lower}")

        # Override test splits with data from separate file
        self._raw_smiles['test_id'] = test_data_dict.get('test_id_smiles', test_data_dict.get('val_id_smiles', []))
        self._raw_smiles['test_ood'] = test_data_dict.get('test_ood_smiles', test_data_dict.get('val_ood_smiles', []))

        logger.info(f"Loaded separate test data: {len(self._raw_smiles['test_id'])} test_id, {len(self._raw_smiles['test_ood'])} test_ood")

    def _select_final_smiles(self):
        """Sub-samples the raw data to meet the target sizes for each split with caching support."""
        # Try to load from cache first
        splits_cache_file = self._get_splits_cache_path()
        if self._load_splits_cache(splits_cache_file):
            logger.info("Loaded data splits from cache - skipping sampling process")
            return

        # If cache miss, perform the sampling
        logger.info("Cache miss - performing data sampling...")
        
        is_good = self.dataset_name.lower().startswith('good_')
        default_sizes = {
            'train_id': 5000 if is_good else 2000, 'train_ood': 5000 if is_good else 2000,
            'val_id': 1500 if is_good else 600, 'val_ood': 1500 if is_good else 600,
            'test_id': 2000 if is_good else 1000, 'test_ood': 2000 if is_good else 1000
        }

        target_sizes = {}
        if self.max_samples:
            debug_factor = self.max_samples / default_sizes['train_id']
            for key, val in default_sizes.items():
                target_sizes[key] = int(val * debug_factor)
        else:
            target_sizes = default_sizes

        logger.info(f"Target dataset sizes: {target_sizes}")

        # FIXED: Use deterministic split mapping instead of hash()
        split_offsets = {
            'train_id': 0, 'train_ood': 1, 'val_id': 2, 
            'val_ood': 3, 'test_id': 4, 'test_ood': 5
        }

        for split in ['train_id', 'train_ood', 'val_id', 'val_ood', 'test_id', 'test_ood']:
            raw_data = self._raw_smiles[split]
            target_size = min(target_sizes[split], len(raw_data))

            if not raw_data:
                logger.warning(f"No raw data available for split: {split}")
                self.final_smiles[split] = []
                continue

            # FIXED: Use deterministic seed calculation
            seed = abs(self.data_seed + split_offsets[split]) % (2**31)
            rng = np.random.default_rng(seed)
            self.final_smiles[split] = list(rng.choice(raw_data, size=target_size, replace=False))

            if len(self.final_smiles[split]) < target_sizes[split]:
                logger.warning(f"Split {split}: available data ({len(raw_data)}) is less than target ({target_sizes[split]}). Using all available.")

        # Verify and report any cross-split overlap
        self._check_cross_split_overlap()
        
        # Save the splits cache after successful sampling
        self._save_splits_cache(splits_cache_file, target_sizes)
        logger.info("Saved data splits to cache for future runs")

    def _get_splits_cache_path(self):
        """Generates a unique path for the data splits cache file."""
        if 'drugood' in self.dataset_name.lower() or 'lbap' in self.dataset_name.lower():
            base_name = self.drugood_subset if self.drugood_subset else self.dataset_name
        else:
            base_name = self.dataset_name

        # Ensure base_name is not None
        if not base_name:
            base_name = 'unknown_dataset'

        clean_name = base_name.replace('/', '_').replace('-', '_')

        # Include relevant parameters in filename for cache invalidation
        if 'good' in clean_name:
            filename = f"{clean_name}_{self.good_domain}_{self.good_shift}_seed{self.data_seed}"
        else:
            filename = f"{clean_name}_seed{self.data_seed}"

        # Add debug size info if applicable
        if self.max_samples:
            filename += f"_debug{self.max_samples}"
        
        filename += "_splits.json"
        return self.cache_dir / filename

    def _load_splits_cache(self, cache_file):
        """Attempts to load data splits from cache. Returns True if successful."""
        if not cache_file.exists() or self.force_recompute:
            return False

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Validate cache compatibility
            cache_metadata = cache_data.get('metadata', {})
            current_metadata = self._get_splits_cache_metadata()
            
            # Check all critical parameters (relax if user supplied external cache)
            relax_check = (self.external_splits_cache_file is not None and cache_file == self.external_splits_cache_file)
            for key, expected_value in current_metadata.items():
                if cache_metadata.get(key) != expected_value:
                    if relax_check:
                        logger.warning(f"External splits cache parameter mismatch: {key} = {cache_metadata.get(key)} != {expected_value}. Proceeding anyway.")
                        continue
                    logger.info(f"Cache parameter mismatch: {key} = {cache_metadata.get(key)} != {expected_value}")
                    return False

            # Load the splits data
            self.final_smiles = cache_data['splits']
            
            # Validate data integrity
            total_molecules = sum(len(smiles_list) for smiles_list in self.final_smiles.values())
            logger.info(f"Loaded splits cache with {total_molecules} total molecules from: {cache_file}")
            return True

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load splits cache {cache_file}: {e}")
            return False

    def _save_splits_cache(self, cache_file, target_sizes):
        """Saves the current data splits to cache with metadata."""
        cache_data = {
            'metadata': self._get_splits_cache_metadata(),
            'target_sizes': target_sizes,
            'splits': self.final_smiles,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            file_size_kb = cache_file.stat().st_size / 1024
            total_molecules = sum(len(smiles_list) for smiles_list in self.final_smiles.values())
            logger.info(f"Splits cache saved: {cache_file} ({file_size_kb:.1f} KB, {total_molecules} molecules)")
            
        except Exception as e:
            logger.error(f"Failed to save splits cache {cache_file}: {e}")

    def _get_splits_cache_metadata(self):
        """Returns metadata for cache validation."""
        metadata = {
            'dataset_name': self.dataset_name,
            'data_seed': self.data_seed,
            'max_samples': self.max_samples,
        }
        
        # Add dataset-specific metadata
        if 'drugood' in self.dataset_name.lower() or 'lbap' in self.dataset_name.lower():
            metadata['drugood_subset'] = self.drugood_subset
            if self.data_file:
                # Add file modification time for cache invalidation
                try:
                    metadata['data_file_mtime'] = os.path.getmtime(self.data_file)
                except OSError:
                    pass
        elif 'good' in self.dataset_name.lower():
            metadata['good_domain'] = self.good_domain
            metadata['good_shift'] = self.good_shift
        
        return metadata

    def _check_cross_split_overlap(self):
        """Check and report any cross-split overlap in SMILES data."""
        # For cross-dataset evaluation, test data comes from a different dataset,
        # so overlap between train/val and test is expected and acceptable
        if self.test_data_file:
            logger.info("Cross-dataset evaluation mode: skipping train-test overlap checks")
            splits_to_check = [
                ('train_id', 'val_id'),
                ('train_ood', 'val_ood'),
                ('train_id', 'train_ood'),
                ('val_id', 'val_ood'),
                ('test_id', 'test_ood')  # Still check test ID vs OOD
            ]
        else:
            # Normal mode: check all overlaps including train-test
            splits_to_check = [
                ('train_id', 'val_id'), ('train_id', 'test_id'),
                ('train_ood', 'val_ood'), ('train_ood', 'test_ood'),
                ('train_id', 'train_ood'), ('val_id', 'val_ood'), ('test_id', 'test_ood')
            ]

        overlap_found = False
        for split1, split2 in splits_to_check:
            smiles1 = set(self.final_smiles.get(split1, []))
            smiles2 = set(self.final_smiles.get(split2, []))
            overlap = smiles1.intersection(smiles2)

            if overlap:
                overlap_found = True
                logger.error(f"DATA LEAKAGE DETECTED: {len(overlap)} molecules overlap between {split1} and {split2}")
                if len(overlap) <= 5:
                    logger.error(f"Overlapping molecules: {list(overlap)}")
                else:
                    logger.error(f"First 5 overlapping molecules: {list(overlap)[:5]}")

        if overlap_found:
            error_msg = ("Cross-split overlap detected! This will cause data leakage in evaluation. "
                        "Training stopped to prevent invalid results.")
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            logger.info("✅ No cross-split overlap detected - data splits are clean.")

    def _prepare_features(self):
        """Manages the feature caching and computation process."""
        # Prefer user-provided feature cache if available
        cache_file = self.external_feature_cache_file if self.external_feature_cache_file else self._get_cache_path()
        
        # Try legacy cache file format first (backward compatibility)
        legacy_cache_file = None
        if not cache_file.exists():
            legacy_name = f"{self.dataset_name}_features.pkl"
            legacy_cache_file = self.cache_dir / legacy_name
            if legacy_cache_file.exists():
                logger.info(f"Found legacy cache file: {legacy_cache_file}")
                cache_file = legacy_cache_file

        # Load existing cache if available and not forced to recompute
        if cache_file.exists() and not self.force_recompute:
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                # Support multiple cache formats: {'features': {...}, 'foundation_model': ...} or {smiles: array}
                features_obj = None
                cached_foundation_model = 'unknown'
                if isinstance(cached_data, dict) and 'features' in cached_data:
                    features_obj = cached_data['features']
                    cached_foundation_model = cached_data.get('foundation_model', 'unknown')
                elif isinstance(cached_data, dict):
                    features_obj = cached_data  # assume {smiles: array}
                else:
                    raise ValueError("Unsupported feature cache format; expected dict.")

                current_foundation_model = self.foundation_model
                logger.info(f"Cache check: cached='{cached_foundation_model}' vs current='{current_foundation_model}'")

                models_match = (
                    cached_foundation_model == current_foundation_model or
                    (cached_foundation_model in ['minimol', 'Minimol'] and current_foundation_model in ['minimol', 'Minimol']) or
                    (cached_foundation_model in ['unimol', 'UniMol', 'unimol_base'] and current_foundation_model in ['unimol', 'UniMol', 'unimol_base']) or
                    cached_foundation_model == 'unknown'
                )

                if models_match and features_obj is not None:
                    self.feature_cache = {}
                    for s, f in features_obj.items():
                        if isinstance(f, torch.Tensor):
                            self.feature_cache[s] = f.cpu()
                        else:
                            try:
                                self.feature_cache[s] = torch.from_numpy(f)
                            except Exception:
                                self.feature_cache[s] = torch.tensor(f)
                    logger.info(f"Loaded {len(self.feature_cache)} features from cache: {cache_file}")
                else:
                    logger.warning(f"Cache foundation model mismatch: '{cached_foundation_model}' != '{current_foundation_model}'. Ignoring cache.")
            except Exception as e:
                logger.error(f"Could not load cache file {cache_file}. Error: {e}")

        # Determine which SMILES need new features
        all_needed_smiles = set()
        for smiles_list in self.final_smiles.values():
            all_needed_smiles.update(smiles_list)

        smiles_to_compute = [s for s in all_needed_smiles if s not in self.feature_cache]

        if not smiles_to_compute:
            logger.info("All required features were found in the cache.")
            return

        # Compute features for the missing SMILES
        logger.info(f"Computing features for {len(smiles_to_compute)} new molecules...")
        new_features = self._compute_features_batch(smiles_to_compute)
        self.feature_cache.update(new_features)

        # Save the updated cache
        if new_features:
            logger.info(f"Saving updated cache with {len(self.feature_cache)} total features.")
            self._save_cache(cache_file)

    def _get_cache_path(self):
        """Generates a descriptive and unique path for the feature cache file."""
        if 'drugood' in self.dataset_name.lower() or 'lbap' in self.dataset_name.lower():
            base_name = self.drugood_subset if self.drugood_subset else self.dataset_name
        else:
            base_name = self.dataset_name

        # Ensure base_name is not None
        if not base_name:
            base_name = 'unknown_dataset'

        clean_name = base_name.replace('/', '_').replace('-', '_')

        if 'good' in clean_name:
            filename = f"{clean_name}_{self.good_domain}_{self.good_shift}_{self.foundation_model}_features.pkl"
        else:
            filename = f"{clean_name}_{self.foundation_model}_features.pkl"

        return self.cache_dir / filename

    def _save_cache(self, cache_path):
        """Saves the current feature cache to a file."""
        # FIXED: Safe conversion - handle both PyTorch tensors and numpy arrays
        numpy_features = {}
        for s, f in self.feature_cache.items():
            if isinstance(f, torch.Tensor):
                numpy_features[s] = f.cpu().numpy()
            elif isinstance(f, np.ndarray):
                numpy_features[s] = f
            else:
                logger.warning(f"Unexpected feature type for {s}: {type(f)}")
                numpy_features[s] = np.array(f)
        
        cache_data = {
            'features': numpy_features,
            'foundation_model': self.foundation_model,
            'dataset_name': self.dataset_name,
            'drugood_subset': self.drugood_subset,
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_size_mb = cache_path.stat().st_size / (1024 * 1024)
        logger.info(f"Cache saved to {cache_path} ({file_size_mb:.2f} MB)")

    def _compute_features_batch(self, smiles_list):
        """
        Computes molecular features in batches for a list of SMILES strings.
        Handles model loading and GPU memory errors gracefully.
        """
        if not smiles_list:
            return {}

        # Dynamically import and instantiate the encoder to keep dependencies local
        if self.foundation_model == 'minimol':
            from model import MinimolEncoder
            encoder = MinimolEncoder()
        elif self.foundation_model == 'unimol':
            from model import UniMolEncoder
            encoder = UniMolEncoder()
        else:
            raise ValueError(f"Unsupported foundation_model: {self.foundation_model}")

        computed_features = {}
        failed_smiles = set()

        for i in range(0, len(smiles_list), self.encoding_batch_size):
            batch_smiles = smiles_list[i:i + self.encoding_batch_size]
            try:
                features = encoder.encode_smiles(batch_smiles)
                for smiles, feature in zip(batch_smiles, features):
                    computed_features[smiles] = feature.cpu()
            except Exception as e:
                logger.warning(f"Batch encoding failed: {e}. Retrying individually.")
                for smiles in batch_smiles:
                    try:
                        feature = encoder.encode_smiles([smiles])[0]
                        computed_features[smiles] = feature.cpu()
                    except Exception as single_e:
                        logger.error(f"Failed to encode SMILES: {smiles}. Error: {single_e}")
                        failed_smiles.add(smiles)

            if i % (self.encoding_batch_size * 10) == 0:
                logger.info(f"Computed features for {len(computed_features)}/{len(smiles_list)} molecules...")

        if failed_smiles:
            logger.warning(f"Total failed SMILES encodings: {len(failed_smiles)}")

        return computed_features

    def _get_features_for_smiles(self, smiles_list):
        """Get features for SMILES list from cache, with error checking"""
        features = []
        missing_smiles = []
        
        for smiles in smiles_list:
            if smiles in self.feature_cache:
                feature = self.feature_cache[smiles]
                # FIXED: Ensure feature is a PyTorch tensor
                if isinstance(feature, np.ndarray):
                    feature = torch.from_numpy(feature)
                elif not isinstance(feature, torch.Tensor):
                    logger.warning(f"Unexpected feature type for {smiles}: {type(feature)}")
                    feature = torch.tensor(feature)
                features.append(feature)
            else:
                missing_smiles.append(smiles)
        
        if missing_smiles:
            logger.error(f"Missing features for {len(missing_smiles)} molecules: {missing_smiles[:5]}...")
            raise ValueError(f"Missing {len(missing_smiles)} molecule features")
        
        return torch.stack(features)

    def get_dataloaders(self):
        """
        Constructs and returns the training and evaluation DataLoaders.
        Automatically uses pre-computed features if caching is enabled.
        """
        if not self.final_smiles.get('train_id') or not self.final_smiles.get('train_ood'):
            raise ValueError("Training data is not prepared. Call `prepare_data()` first.")
        if not self.final_smiles.get('val_id') or not self.final_smiles.get('val_ood'):
            raise ValueError("Validation data is not prepared. Call `prepare_data()` first.")

        train_id_smiles, train_ood_smiles = self.final_smiles['train_id'], self.final_smiles['train_ood']
        val_id_smiles, val_ood_smiles = self.final_smiles['val_id'], self.final_smiles['val_ood']

        # Determine optimal number of workers for DataLoader
        num_workers = getattr(self.args, 'num_workers', 4)
        seed = getattr(self.args, 'seed', 0)

        # Worker initialization to ensure deterministic behavior
        def worker_init_fn(worker_id):
            worker_seed = seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        # Optional generator for DataLoader
        generator = torch.Generator()
        generator.manual_seed(seed)

        if self.enable_cache:
            logger.info("Creating DataLoaders with pre-computed features.")

            # Helper to safely retrieve features from cache with type conversion
            def get_feats(smiles):
                feats = []
                for s in smiles:
                    f = self.feature_cache.get(s)
                    if f is None:
                        raise RuntimeError(f"Missing feature for SMILES: {s}. Caching may have failed.")
                    # Ensure feature is a PyTorch tensor
                    if isinstance(f, np.ndarray):
                        f = torch.from_numpy(f)
                    elif not isinstance(f, torch.Tensor):
                        logger.warning(f"Unexpected feature type for {s}: {type(f)}")
                        f = torch.tensor(f)
                    feats.append(f)
                return torch.stack(feats)

            train_dataset = PrecomputedEnergyDPODataset(
                get_feats(train_id_smiles), get_feats(train_ood_smiles),
                train_id_smiles, train_ood_smiles, mode='train', seed=seed
            )
            eval_dataset = PrecomputedEnergyDPODataset(
                get_feats(val_id_smiles), get_feats(val_ood_smiles),
                val_id_smiles, val_ood_smiles, mode='eval', seed=seed
            )
            collate_fn = precomputed_energy_dpo_collate_fn
        else:
            logger.info("Creating DataLoaders for real-time feature encoding.")
            train_dataset = EnergyDPODataset(train_id_smiles, train_ood_smiles, mode='train', seed=seed)
            eval_dataset = EnergyDPODataset(val_id_smiles, val_ood_smiles, mode='eval', seed=seed)
            collate_fn = energy_dpo_collate_fn
            num_workers = min(num_workers, 2)  # Reduce workers for real-time GPU encoding to avoid contention

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
        return train_loader, eval_loader

    def get_final_test_data(self):
        """Returns the final test set SMILES for OOD detection evaluation."""
        test_id = self.final_smiles.get('test_id', [])
        test_ood = self.final_smiles.get('test_ood', [])

        if not test_id or not test_ood:
            logger.warning("Final test data not available, falling back to validation data.")
            return {'id_smiles': self.final_smiles.get('val_id', []), 'ood_smiles': self.final_smiles.get('val_ood', [])}

        logger.info(f"Providing final test set: {len(test_id)} ID samples, {len(test_ood)} OOD samples.")
        return {'id_smiles': test_id, 'ood_smiles': test_ood}

    def _print_summary(self):
        """Prints a summary of the final dataset sizes."""
        logger.info("=" * 50)
        logger.info("Final Dataset Summary:")
        for split, smiles_list in self.final_smiles.items():
            logger.info(f"  - {split:<10}: {len(smiles_list):>5} samples")
        logger.info("=" * 50)


# Backward compatibility alias
FastEnergyDPODataLoader = EnergyDPODataLoader
