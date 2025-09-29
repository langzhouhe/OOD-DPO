import json
import pandas as pd
import numpy as np
import random
import logging
import os
import torch

# Suppress RDKit pandas warnings
import warnings
warnings.filterwarnings("ignore", message="Failed to find the pandas get_adjustment.*")
warnings.filterwarnings("ignore", message="Failed to patch pandas.*")
warnings.filterwarnings("ignore", message=".*PandasTools will have limited functionality.*")

# Set RDKit to silent mode
os.environ['RDK_QUIET'] = '1'
try:
    from rdkit import rdBase
    rdBase.DisableLog('rdApp.*')
except ImportError:
    pass

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from munch import Munch

logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# UniMol related utility functions
# =============================================================================

def smiles_to_3d_coords(smiles, seed=42, max_attempts=5000):
    """
    Convert SMILES to 3D molecular coordinates
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Add hydrogen atoms
    mol = AllChem.AddHs(mol)
    
    # Generate 3D coordinates
    embed_result = AllChem.EmbedMolecule(mol, randomSeed=seed)
    if embed_result == -1:
        # Try more attempts
        embed_result = AllChem.EmbedMolecule(mol, maxAttempts=max_attempts, randomSeed=seed)
        if embed_result == -1:
            # If still fails, use 2D coordinates and extend to 3D
            AllChem.Compute2DCoords(mol)
            conformer = mol.GetConformer()
            for i in range(mol.GetNumAtoms()):
                pos = conformer.GetAtomPosition(i)
                conformer.SetAtomPosition(i, [pos.x, pos.y, 0.0])
        else:
            # Optimize 3D structure
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                pass
    else:
        # Optimize 3D structure
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass
    
    return mol

def smiles2graph(smiles):
    """
    Convert single SMILES to Uni-Mol graph data format
    """
    try:
        mol = smiles_to_3d_coords(smiles)
        
        # Get atom information
        atoms = []
        coordinates = []
        for atom in mol.GetAtoms():
            atoms.append(atom.GetSymbol())
        
        # Get coordinates
        conformer = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            pos = conformer.GetAtomPosition(i)
            coordinates.append([pos.x, pos.y, pos.z])
        
        coordinates = np.array(coordinates, dtype=np.float32)
        
        # Calculate distance matrix
        num_atoms = len(atoms)
        distance_matrix = np.zeros((num_atoms, num_atoms), dtype=np.float32)
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    distance_matrix[i][j] = np.linalg.norm(coordinates[i] - coordinates[j])
        
        # Get edge information
        edge_types = np.zeros((num_atoms, num_atoms), dtype=np.int64)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            if bond_type == Chem.rdchem.BondType.SINGLE:
                edge_type = 1
            elif bond_type == Chem.rdchem.BondType.DOUBLE:
                edge_type = 2
            elif bond_type == Chem.rdchem.BondType.TRIPLE:
                edge_type = 3
            elif bond_type == Chem.rdchem.BondType.AROMATIC:
                edge_type = 4
            else:
                edge_type = 1
            
            edge_types[i][j] = edge_type
            edge_types[j][i] = edge_type
        
        return {
            'atoms': atoms,
            'coordinates': coordinates,
            'distance_matrix': distance_matrix,
            'edge_types': edge_types,
            'mol': mol,
            'smiles': smiles
        }
    
    except Exception as e:
        logger.warning(f"Failed to process SMILES {smiles}: {e}")
        # Return empty graph data
        return {
            'atoms': ['C'],  # Minimal placeholder
            'coordinates': np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            'distance_matrix': np.array([[0.0]], dtype=np.float32),
            'edge_types': np.array([[0]], dtype=np.int64),
            'mol': None,
            'smiles': smiles
        }

def create_atom_token_mapping():
    """Create mapping from atom symbols to token IDs"""
    # Based on Uni-Mol atom vocabulary
    atom_vocab = [
        '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]',
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
        'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
        'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
    ]
    
    return {atom: idx for idx, atom in enumerate(atom_vocab)}

def unimol_collate_fn(samples, atom_token_mapping=None):
    """
    Uni-Mol batch collate function
    Organize multiple molecular graph data into model input format
    """
    if atom_token_mapping is None:
        atom_token_mapping = create_atom_token_mapping()
    
    batch_size = len(samples)
    if batch_size == 0:
        return {}
    
    # Find maximum number of atoms
    max_atoms = max(len(sample['atoms']) for sample in samples)
    max_atoms = max(max_atoms, 1)  # At least 1
    
    # Initialize batch tensors
    batch_tokens = torch.zeros(batch_size, max_atoms + 2, dtype=torch.long)  # +2 for CLS and SEP
    batch_coordinates = torch.zeros(batch_size, max_atoms + 2, 3, dtype=torch.float32)
    batch_distance = torch.zeros(batch_size, max_atoms + 2, max_atoms + 2, dtype=torch.float32)
    batch_edge_type = torch.zeros(batch_size, max_atoms + 2, max_atoms + 2, dtype=torch.long)
    
    for i, sample in enumerate(samples):
        atoms = sample['atoms']
        coordinates = sample['coordinates']
        distance_matrix = sample['distance_matrix']
        edge_types = sample['edge_types']
        
        num_atoms = len(atoms)
        
        # Set tokens (CLS + atoms + SEP)
        batch_tokens[i, 0] = atom_token_mapping.get('[CLS]', 2)  # CLS token
        for j, atom in enumerate(atoms):
            batch_tokens[i, j + 1] = atom_token_mapping.get(atom, atom_token_mapping.get('[UNK]', 1))
        if num_atoms + 1 < max_atoms + 2:
            batch_tokens[i, num_atoms + 1] = atom_token_mapping.get('[SEP]', 3)  # SEP token
        
        # Set coordinates (CLS coordinates as origin, SEP coordinates also as origin)
        batch_coordinates[i, 0] = torch.zeros(3)  # CLS coordinates
        batch_coordinates[i, 1:num_atoms + 1] = torch.from_numpy(coordinates)
        if num_atoms + 1 < max_atoms + 2:
            batch_coordinates[i, num_atoms + 1] = torch.zeros(3)  # SEP coordinates
        
        # Set distance matrix
        batch_distance[i, 1:num_atoms + 1, 1:num_atoms + 1] = torch.from_numpy(distance_matrix)
        
        # Set edge types
        batch_edge_type[i, 1:num_atoms + 1, 1:num_atoms + 1] = torch.from_numpy(edge_types)
    
    # Create padding mask
    padding_mask = (batch_tokens == atom_token_mapping.get('[PAD]', 0))
    
    return {
        'net_input': {
            'mol_src_tokens': batch_tokens,
            'mol_src_distance': batch_distance,
            'mol_src_edge_type': batch_edge_type,
            'mol_src_coord': batch_coordinates,
        },
        'batched_data': {
            'tokens': batch_tokens,
            'coordinates': batch_coordinates,
            'distance': batch_distance,
            'edge_type': batch_edge_type,
            'padding_mask': padding_mask,
        }
    }

# =============================================================================
# Data processing functions
# =============================================================================

def get_dataset_name_from_file(file_path):
    """Extract dataset name from file path"""
    if not file_path:
        return "unknown"
    
    filename = os.path.basename(file_path)
    
    # good_data datasets
    if "good_hiv" in filename.lower():
        return "good_hiv"
    elif "good_pcba" in filename.lower():
        return "good_pcba"
    elif "good_zinc" in filename.lower():
        return "good_zinc"
    
    # DrugOOD datasets (maintain backward compatibility)
    if "lbap" in filename.lower():
        parts = filename.replace('.json', '').split('_')
        return '_'.join(parts[:4]) if len(parts) >= 4 else filename.replace('.json', '')
    
    return filename.replace('.json', '').replace('.pt', '')

def validate_smiles(smiles_list):
    """Validate and clean SMILES list - lenient version"""
    valid_smiles = []
    invalid_count = 0
    
    for smiles in smiles_list:
        # Basic check: non-empty string
        if smiles and isinstance(smiles, str) and len(smiles.strip()) > 0:
            smiles_clean = smiles.strip()
            
            try:
                # Try RDKit parsing, but don't force standardization
                mol = Chem.MolFromSmiles(smiles_clean)
                if mol is not None:
                    # Use original SMILES, avoid data loss from standardization
                    valid_smiles.append(smiles_clean)
                else:
                    # RDKit cannot parse, but might still be valid chemical structure
                    # Perform basic character check
                    if _basic_smiles_check(smiles_clean):
                        valid_smiles.append(smiles_clean)
                        logger.debug(f"Using SMILES that passed basic check: {smiles_clean}")
                    else:
                        invalid_count += 1
            except Exception as e:
                # RDKit parsing error, try basic check
                if _basic_smiles_check(smiles_clean):
                    valid_smiles.append(smiles_clean)
                    logger.debug(f"RDKit parsing failed but basic check passed: {smiles_clean}")
                else:
                    invalid_count += 1
                    logger.debug(f"SMILES validation failed: {smiles_clean}, error: {e}")
        else:
            invalid_count += 1
    
    if invalid_count > 0:
        logger.warning(f"Filtered out {invalid_count} invalid SMILES")
    
    return valid_smiles

def _basic_smiles_check(smiles):
    """Basic SMILES format check"""
    if not smiles or len(smiles) < 1:
        return False
    
    # Basic SMILES character set check
    valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]{}=#@+-.\\/|')
    smiles_chars = set(smiles)
    
    # If contains too many non-SMILES characters, consider invalid
    invalid_chars = smiles_chars - valid_chars
    if len(invalid_chars) > 0:
        # Allow small amount of special characters (might be extended SMILES syntax)
        if len(invalid_chars) <= 2:
            return True
        return False
    
    # Basic bracket matching check
    paren_count = smiles.count('(') - smiles.count(')')
    bracket_count = smiles.count('[') - smiles.count(']')
    
    if paren_count != 0 or bracket_count != 0:
        return False
    
    return True

def process_drugood_data(data_file, max_samples=None):
    """
    Process DrugOOD dataset for OOD Detection format

    Args:
        data_file: Data file path
        max_samples: Maximum sample number limit

    Returns:
        dict: Dictionary containing training and validation data
    """
    logger.info(f"OOD Detection mode loading: {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, dict) or 'split' not in data:
        raise ValueError(f"Not a valid DrugOOD format: missing 'split' key")
    
    split_data = data['split']
    
    # Use DrugOOD data according to correct OOD Detection principles
    train_raw = split_data.get('train', [])           # ID training data
    ood_val_raw = split_data.get('ood_val', [])       # OOD training data (for DPO negative examples)
    ood_test_raw = split_data.get('ood_test', [])     # OOD test data (for final testing)
    iid_val_raw = split_data.get('iid_val', [])       # ID validation data
    iid_test_raw = split_data.get('iid_test', [])     # ID test data (for final testing)
    
    logger.info(f"OOD Detection raw data:")
    logger.info(f"  ID training(train): {len(train_raw)}")
    logger.info(f"  ID validation(iid_val): {len(iid_val_raw)}")
    logger.info(f"  ID test(iid_test): {len(iid_test_raw)}")
    logger.info(f"  OOD training(ood_val): {len(ood_val_raw)}")
    logger.info(f"  OOD test(ood_test): {len(ood_test_raw)}")
    
    # If maximum sample number is specified, perform sampling
    if max_samples:
        if len(train_raw) > max_samples:
            train_raw = random.sample(train_raw, max_samples)
        if len(ood_val_raw) > max_samples:
            ood_val_raw = random.sample(ood_val_raw, max_samples)
        if len(iid_test_raw) > max_samples:
            iid_test_raw = random.sample(iid_test_raw, max_samples)
        if len(ood_test_raw) > max_samples:
            ood_test_raw = random.sample(ood_test_raw, max_samples)
    
    # Process validation data
    if iid_val_raw:
        if max_samples and len(iid_val_raw) > max_samples:
            iid_val_raw = random.sample(iid_val_raw, max_samples)
        val_raw = iid_val_raw
    else:
        # If no iid_val, split validation data from training
        if len(train_raw) >= 100:
            val_size = min(len(train_raw) // 4, 1000)
            val_raw = random.sample(train_raw, val_size)
            # Remove selected validation data from train_raw
            val_smiles_set = set(item.get('smiles', '') for item in val_raw)
            train_raw = [item for item in train_raw if item.get('smiles', '') not in val_smiles_set]
        else:
            # Not enough data to split, random split
            train_copy = train_raw.copy()
            random.shuffle(train_copy)
            split_point = len(train_copy) // 2
            val_raw = train_copy[:split_point]
            train_raw = train_copy[split_point:]
    
    # Extract SMILES
    train_id_smiles = [item.get('smiles') for item in train_raw if item.get('smiles')]
    val_id_smiles = [item.get('smiles') for item in val_raw if item.get('smiles')]
    train_ood_smiles = [item.get('smiles') for item in ood_val_raw if item.get('smiles')]
    test_id_smiles = [item.get('smiles') for item in iid_test_raw if item.get('smiles')]
    test_ood_smiles = [item.get('smiles') for item in ood_test_raw if item.get('smiles')]
    
    # Validate SMILES
    train_id_smiles = validate_smiles(train_id_smiles)
    val_id_smiles = validate_smiles(val_id_smiles)
    train_ood_smiles = validate_smiles(train_ood_smiles)
    test_id_smiles = validate_smiles(test_id_smiles)
    test_ood_smiles = validate_smiles(test_ood_smiles)
    
    logger.info(f"OOD Detection data processing completed:")
    logger.info(f"  Training ID: {len(train_id_smiles)}")
    logger.info(f"  Validation ID: {len(val_id_smiles)}")
    logger.info(f"  Training OOD: {len(train_ood_smiles)}")
    logger.info(f"  Test ID: {len(test_id_smiles)}")
    logger.info(f"  Test OOD: {len(test_ood_smiles)}")
    
    return {
        'train_id_smiles': train_id_smiles,
        'val_id_smiles': val_id_smiles,
        'train_ood_smiles': train_ood_smiles,
        'test_id_smiles': test_id_smiles,
        'test_ood_smiles': test_ood_smiles
    }

def process_good_data(dataset_name, domain='scaffold', shift='covariate',
                      data_path='./data', max_samples=None, seed=42,
                      validate_smiles_flag=True):
    """Process GOOD series molecular datasets and return SMILES and their labels

    Args:
        validate_smiles_flag: Whether to perform RDKit validation on extracted SMILES. For GOOD official preprocessed products, it is recommended to disable this to improve speed.
    """

    logger.info(f"Processing {dataset_name} dataset (domain={domain}, shift={shift})")

    try:
        # Dynamically import corresponding dataset class
        if dataset_name == 'good_hiv':
            from data.good_data.good_datasets.good_hiv import GOODHIV as DatasetClass
        elif dataset_name == 'good_pcba':
            from data.good_data.good_datasets.good_pcba import GOODPCBA as DatasetClass
        elif dataset_name == 'good_zinc':
            from data.good_data.good_datasets.good_zinc import GOODZINC as DatasetClass
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        logger.info(f"Loading dataset from {data_path}...")

        # Load different splits
        train_dataset = DatasetClass(root=data_path, domain=domain, shift=shift, subset='train')
        val_dataset = DatasetClass(root=data_path, domain=domain, shift=shift, subset='val')
        test_dataset = DatasetClass(root=data_path, domain=domain, shift=shift, subset='test')

        # If there is shift, also load id splits
        if shift != 'no_shift':
            try:
                id_val_dataset = DatasetClass(root=data_path, domain=domain, shift=shift, subset='id_val')
                id_test_dataset = DatasetClass(root=data_path, domain=domain, shift=shift, subset='id_test')
            except Exception:
                logger.warning("Unable to load id_val/id_test, will split from other data")
                id_val_dataset = None
                id_test_dataset = None
        else:
            id_val_dataset = None
            id_test_dataset = None

        # Extract SMILES and labels - support multi-task learning
        def extract_smiles_labels(dataset):
            smiles_list, label_list = [], []
            for i in range(len(dataset)):
                try:
                    data = dataset[i]
                    if hasattr(data, 'smiles') and data.smiles:
                        smiles_list.append(data.smiles)
                        if hasattr(data, 'y') and data.y is not None:
                            y = data.y
                            if isinstance(y, torch.Tensor):
                                # Convert to JSON-serializable Python native types
                                if y.dim() > 1:
                                    # Multi-task case: [1, num_tasks] -> [num_tasks], convert to list
                                    label = y.view(-1).tolist()
                                else:
                                    # Single-task case, convert to Python scalar or list
                                    label = y.item() if y.numel() == 1 else y.tolist()
                            else:
                                label = y
                        else:
                            label = None
                        label_list.append(label)
                except Exception:
                    continue
            return smiles_list, label_list

        train_smiles, train_labels = extract_smiles_labels(train_dataset)
        val_ood_all_smiles, _ = extract_smiles_labels(val_dataset)
        test_ood_smiles, _ = extract_smiles_labels(test_dataset)

        if id_val_dataset is not None:
            val_id_smiles, val_id_labels = extract_smiles_labels(id_val_dataset)
            test_id_smiles, test_id_labels = extract_smiles_labels(id_test_dataset)
        else:
            # Split validation set from train
            if len(train_smiles) > 100:
                random.seed(seed)
                val_size = min(len(train_smiles) // 4, 1000)
                idx = random.sample(range(len(train_smiles)), val_size)
                val_id_smiles = [train_smiles[i] for i in idx]
                val_id_labels = [train_labels[i] for i in idx]
                mask = set(idx)
                train_smiles = [s for i, s in enumerate(train_smiles) if i not in mask]
                train_labels = [l for i, l in enumerate(train_labels) if i not in mask]
            else:
                random.seed(seed)
                combined = list(zip(train_smiles, train_labels))
                random.shuffle(combined)
                split_idx = len(combined) // 2
                val_part, train_part = combined[:split_idx], combined[split_idx:]
                val_id_smiles, val_id_labels = zip(*val_part)
                train_smiles, train_labels = zip(*train_part)
                val_id_smiles, val_id_labels = list(val_id_smiles), list(val_id_labels)
                train_smiles, train_labels = list(train_smiles), list(train_labels)

            # Instead of using OOD test set as ID test, create separate ID test from train
            if len(train_smiles) > 200:
                random.seed(seed + 2)  # Different seed for test split
                test_size = min(len(train_smiles) // 10, 500)
                test_idx = random.sample(range(len(train_smiles)), test_size)
                test_id_smiles = [train_smiles[i] for i in test_idx]
                test_id_labels = [train_labels[i] for i in test_idx]
                # Remove test samples from train
                train_mask = set(test_idx)
                train_smiles = [s for i, s in enumerate(train_smiles) if i not in train_mask]
                train_labels = [l for i, l in enumerate(train_labels) if i not in train_mask]
            else:
                # For very small datasets, use a portion of validation as test
                test_size = len(val_id_smiles) // 2
                test_id_smiles = val_id_smiles[:test_size]
                test_id_labels = val_id_labels[:test_size] if val_id_labels else [None] * test_size
                val_id_smiles = val_id_smiles[test_size:]
                val_id_labels = val_id_labels[test_size:] if val_id_labels else [None] * len(val_id_smiles)

        # Split OOD data into training and validation parts
        if val_ood_all_smiles:
            random.seed(seed)
            random.shuffle(val_ood_all_smiles)
            split_idx = int(len(val_ood_all_smiles) * 0.7)
            train_ood_smiles = val_ood_all_smiles[:split_idx]
            val_ood_smiles = val_ood_all_smiles[split_idx:]
            if len(val_ood_smiles) == 0 and len(train_ood_smiles) > 0:
                min_val_size = min(len(train_ood_smiles) // 4, 500)
                val_ood_smiles = train_ood_smiles[-min_val_size:]
                train_ood_smiles = train_ood_smiles[:-min_val_size]
        else:
            logger.warning("No OOD validation data found, using part of test data")
            if test_ood_smiles:
                random.seed(seed)
                random.shuffle(test_ood_smiles)
                # Three-way split: train_ood, val_ood, test_ood (ensure no overlap)
                total_len = len(test_ood_smiles)
                train_split = total_len // 3
                val_split = (total_len * 2) // 3

                train_ood_smiles = test_ood_smiles[:train_split]
                val_ood_smiles = test_ood_smiles[train_split:val_split]
                test_ood_smiles = test_ood_smiles[val_split:]

                logger.info(f"OOD data three-way split: train_ood={len(train_ood_smiles)}, val_ood={len(val_ood_smiles)}, test_ood={len(test_ood_smiles)}")
            else:
                raise ValueError(
                    f"Dataset {dataset_name} does not contain proper OOD splits. "
                    f"For OOD detection training, the dataset must have separate OOD validation/test data "
                    f"that comes from a different distribution than the training data. "
                    f"Available splits: domain={domain}, shift={shift}. "
                    f"Consider using a different domain/shift combination that provides OOD splits."
                )

        # If maximum sample number is specified, perform sampling
        if max_samples:
            random.seed(seed)
            if len(train_smiles) > max_samples:
                idx = random.sample(range(len(train_smiles)), max_samples)
                train_smiles = [train_smiles[i] for i in idx]
                train_labels = [train_labels[i] for i in idx]
            if len(train_ood_smiles) > max_samples:
                train_ood_smiles = random.sample(train_ood_smiles, max_samples)
            if len(val_ood_smiles) > max_samples:
                val_ood_smiles = random.sample(val_ood_smiles, max_samples)
            if len(test_ood_smiles) > max_samples:
                test_ood_smiles = random.sample(test_ood_smiles, max_samples)
            if len(val_id_smiles) > max_samples:
                idx = random.sample(range(len(val_id_smiles)), max_samples)
                val_id_smiles = [val_id_smiles[i] for i in idx]
                val_id_labels = [val_id_labels[i] for i in idx]
            if len(test_id_smiles) > max_samples:
                test_id_smiles = random.sample(test_id_smiles, max_samples)
                if test_id_labels[0] is not None:
                    test_id_labels = test_id_labels[: max_samples]

        # Validate SMILES (optional)
        if validate_smiles_flag:
            train_smiles = validate_smiles(train_smiles)
            val_id_smiles = validate_smiles(val_id_smiles)
            train_ood_smiles = validate_smiles(train_ood_smiles)
            val_ood_smiles = validate_smiles(val_ood_smiles)
            test_id_smiles = validate_smiles(test_id_smiles)
            test_ood_smiles = validate_smiles(test_ood_smiles)

        logger.info(f"{dataset_name} data loading completed:")
        logger.info(f"  Training ID: {len(train_smiles)}")
        logger.info(f"  Validation ID: {len(val_id_smiles)}")
        logger.info(f"  Training OOD: {len(train_ood_smiles)}")
        logger.info(f"  Validation OOD: {len(val_ood_smiles)}")
        logger.info(f"  Test ID: {len(test_id_smiles)}")
        logger.info(f"  Test OOD: {len(test_ood_smiles)}")

        return {
            'train_id_smiles': train_smiles,
            'train_id_labels': train_labels,
            'val_id_smiles': val_id_smiles,
            'val_id_labels': val_id_labels,
            'train_ood_smiles': train_ood_smiles,
            'val_ood_smiles': val_ood_smiles,
            'test_id_smiles': test_id_smiles,
            'test_id_labels': test_id_labels,
            'test_ood_smiles': test_ood_smiles,
        }

    except Exception as e:
        logger.error(f"Loading {dataset_name} failed: {e}")
        raise
