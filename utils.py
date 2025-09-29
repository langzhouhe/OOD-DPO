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
    """创建原子符号到 token ID 的映射"""
    # 基于 Uni-Mol 的原子词汇表
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
    Uni-Mol 的批处理整理函数
    将多个分子图数据整理成模型输入格式
    """
    if atom_token_mapping is None:
        atom_token_mapping = create_atom_token_mapping()
    
    batch_size = len(samples)
    if batch_size == 0:
        return {}
    
    # 找到最大原子数
    max_atoms = max(len(sample['atoms']) for sample in samples)
    max_atoms = max(max_atoms, 1)  # 至少为 1
    
    # 初始化批处理张量
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
        
        # 设置 tokens (CLS + atoms + SEP)
        batch_tokens[i, 0] = atom_token_mapping.get('[CLS]', 2)  # CLS token
        for j, atom in enumerate(atoms):
            batch_tokens[i, j + 1] = atom_token_mapping.get(atom, atom_token_mapping.get('[UNK]', 1))
        if num_atoms + 1 < max_atoms + 2:
            batch_tokens[i, num_atoms + 1] = atom_token_mapping.get('[SEP]', 3)  # SEP token
        
        # 设置坐标 (CLS 坐标为原点，SEP 坐标也为原点)
        batch_coordinates[i, 0] = torch.zeros(3)  # CLS coordinates
        batch_coordinates[i, 1:num_atoms + 1] = torch.from_numpy(coordinates)
        if num_atoms + 1 < max_atoms + 2:
            batch_coordinates[i, num_atoms + 1] = torch.zeros(3)  # SEP coordinates
        
        # 设置距离矩阵
        batch_distance[i, 1:num_atoms + 1, 1:num_atoms + 1] = torch.from_numpy(distance_matrix)
        
        # 设置边类型
        batch_edge_type[i, 1:num_atoms + 1, 1:num_atoms + 1] = torch.from_numpy(edge_types)
    
    # 创建 padding mask
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
# 原有的数据处理函数
# =============================================================================

def get_dataset_name_from_file(file_path):
    """从文件路径中提取数据集名称"""
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
    
    # DrugOOD datasets (保持向后兼容)
    if "lbap" in filename.lower():
        parts = filename.replace('.json', '').split('_')
        return '_'.join(parts[:4]) if len(parts) >= 4 else filename.replace('.json', '')
    
    return filename.replace('.json', '').replace('.pt', '')

def validate_smiles(smiles_list):
    """验证和清理SMILES列表 - 宽松版本"""
    valid_smiles = []
    invalid_count = 0
    
    for smiles in smiles_list:
        # 基本检查：非空字符串
        if smiles and isinstance(smiles, str) and len(smiles.strip()) > 0:
            smiles_clean = smiles.strip()
            
            try:
                # 尝试RDKit解析，但不强制标准化
                mol = Chem.MolFromSmiles(smiles_clean)
                if mol is not None:
                    # 使用原始SMILES，避免标准化导致的数据丢失
                    valid_smiles.append(smiles_clean)
                else:
                    # RDKit无法解析，但可能仍然是有效的化学结构
                    # 进行基本的字符检查
                    if _basic_smiles_check(smiles_clean):
                        valid_smiles.append(smiles_clean)
                        logger.debug(f"使用基本检查通过的SMILES: {smiles_clean}")
                    else:
                        invalid_count += 1
            except Exception as e:
                # RDKit解析出错，尝试基本检查
                if _basic_smiles_check(smiles_clean):
                    valid_smiles.append(smiles_clean)
                    logger.debug(f"RDKit解析失败但基本检查通过: {smiles_clean}")
                else:
                    invalid_count += 1
                    logger.debug(f"SMILES验证失败: {smiles_clean}, 错误: {e}")
        else:
            invalid_count += 1
    
    if invalid_count > 0:
        logger.warning(f"过滤掉 {invalid_count} 个无效SMILES")
    
    return valid_smiles

def _basic_smiles_check(smiles):
    """基本的SMILES格式检查"""
    if not smiles or len(smiles) < 1:
        return False
    
    # 基本的SMILES字符集检查
    valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]{}=#@+-.\\/|')
    smiles_chars = set(smiles)
    
    # 如果包含太多非SMILES字符，则认为无效
    invalid_chars = smiles_chars - valid_chars
    if len(invalid_chars) > 0:
        # 允许少量特殊字符（可能是扩展的SMILES语法）
        if len(invalid_chars) <= 2:
            return True
        return False
    
    # 基本的括号匹配检查
    paren_count = smiles.count('(') - smiles.count(')')
    bracket_count = smiles.count('[') - smiles.count(']')
    
    if paren_count != 0 or bracket_count != 0:
        return False
    
    return True

def process_drugood_data(data_file, max_samples=None):
    """
    处理DrugOOD数据集为OOD Detection格式
    
    Args:
        data_file: 数据文件路径
        max_samples: 最大样本数限制
    
    Returns:
        dict: 包含训练和验证数据的字典
    """
    logger.info(f"🎯 OOD Detection模式加载: {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, dict) or 'split' not in data:
        raise ValueError(f"不是有效的DrugOOD格式：缺少'split'键")
    
    split_data = data['split']
    
    # 按照OOD Detection的正确理念使用DrugOOD数据
    train_raw = split_data.get('train', [])           # ID训练数据
    ood_val_raw = split_data.get('ood_val', [])       # OOD训练数据（用于DPO负例）
    ood_test_raw = split_data.get('ood_test', [])     # OOD测试数据（最终测试用）
    iid_val_raw = split_data.get('iid_val', [])       # ID验证数据
    iid_test_raw = split_data.get('iid_test', [])     # ID测试数据（最终测试用）
    
    logger.info(f"📊 OOD Detection原始数据:")
    logger.info(f"  ID训练(train): {len(train_raw)}")
    logger.info(f"  ID验证(iid_val): {len(iid_val_raw)}")
    logger.info(f"  ID测试(iid_test): {len(iid_test_raw)}")
    logger.info(f"  OOD训练(ood_val): {len(ood_val_raw)}")
    logger.info(f"  OOD测试(ood_test): {len(ood_test_raw)}")
    
    # 如果指定了最大样本数，进行采样
    if max_samples:
        if len(train_raw) > max_samples:
            train_raw = random.sample(train_raw, max_samples)
        if len(ood_val_raw) > max_samples:
            ood_val_raw = random.sample(ood_val_raw, max_samples)
        if len(iid_test_raw) > max_samples:
            iid_test_raw = random.sample(iid_test_raw, max_samples)
        if len(ood_test_raw) > max_samples:
            ood_test_raw = random.sample(ood_test_raw, max_samples)
    
    # 处理验证数据
    if iid_val_raw:
        if max_samples and len(iid_val_raw) > max_samples:
            iid_val_raw = random.sample(iid_val_raw, max_samples)
        val_raw = iid_val_raw
    else:
        # 如果没有iid_val，从train中分出验证数据
        if len(train_raw) >= 100:
            val_size = min(len(train_raw) // 4, 1000)
            val_raw = random.sample(train_raw, val_size)
            # 从train_raw中移除已选的验证数据
            val_smiles_set = set(item.get('smiles', '') for item in val_raw)
            train_raw = [item for item in train_raw if item.get('smiles', '') not in val_smiles_set]
        else:
            # 数据不够分，随机分割
            train_copy = train_raw.copy()
            random.shuffle(train_copy)
            split_point = len(train_copy) // 2
            val_raw = train_copy[:split_point]
            train_raw = train_copy[split_point:]
    
    # 提取SMILES
    train_id_smiles = [item.get('smiles') for item in train_raw if item.get('smiles')]
    val_id_smiles = [item.get('smiles') for item in val_raw if item.get('smiles')]
    train_ood_smiles = [item.get('smiles') for item in ood_val_raw if item.get('smiles')]
    test_id_smiles = [item.get('smiles') for item in iid_test_raw if item.get('smiles')]
    test_ood_smiles = [item.get('smiles') for item in ood_test_raw if item.get('smiles')]
    
    # 验证SMILES
    train_id_smiles = validate_smiles(train_id_smiles)
    val_id_smiles = validate_smiles(val_id_smiles)
    train_ood_smiles = validate_smiles(train_ood_smiles)
    test_id_smiles = validate_smiles(test_id_smiles)
    test_ood_smiles = validate_smiles(test_ood_smiles)
    
    logger.info(f"✅ OOD Detection数据处理完成:")
    logger.info(f"  训练ID: {len(train_id_smiles)}")
    logger.info(f"  验证ID: {len(val_id_smiles)}")
    logger.info(f"  训练OOD: {len(train_ood_smiles)}")
    logger.info(f"  测试ID: {len(test_id_smiles)}")
    logger.info(f"  测试OOD: {len(test_ood_smiles)}")
    
    return {
        'train_id_smiles': train_id_smiles,
        'val_id_smiles': val_id_smiles,
        'train_ood_smiles': train_ood_smiles,
        'test_id_smiles': test_id_smiles,
        'test_ood_smiles': test_ood_smiles
    }

def process_drugood_data(data_file, max_samples=None):
    """
    处理DrugOOD数据集为OOD Detection格式
    
    Args:
        data_file: 数据文件路径
        max_samples: 最大样本数限制
    
    Returns:
        dict: 包含训练和验证数据的字典
    """
    logger.info(f"🎯 OOD Detection模式加载: {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, dict) or 'split' not in data:
        raise ValueError(f"不是有效的DrugOOD格式：缺少'split'键")
    
    split_data = data['split']
    
    # 按照OOD Detection的正确理念使用DrugOOD数据
    train_raw = split_data.get('train', [])           # ID训练数据
    ood_val_raw = split_data.get('ood_val', [])       # OOD训练数据（用于DPO负例）
    ood_test_raw = split_data.get('ood_test', [])     # OOD测试数据（最终测试用）
    iid_val_raw = split_data.get('iid_val', [])       # ID验证数据
    iid_test_raw = split_data.get('iid_test', [])     # ID测试数据（最终测试用）
    
    logger.info(f"📊 OOD Detection原始数据:")
    logger.info(f"  ID训练(train): {len(train_raw)}")
    logger.info(f"  ID验证(iid_val): {len(iid_val_raw)}")
    logger.info(f"  ID测试(iid_test): {len(iid_test_raw)}")
    logger.info(f"  OOD训练(ood_val): {len(ood_val_raw)}")
    logger.info(f"  OOD测试(ood_test): {len(ood_test_raw)}")
    
    # 如果指定了最大样本数，进行采样
    if max_samples:
        if len(train_raw) > max_samples:
            train_raw = random.sample(train_raw, max_samples)
        if len(ood_val_raw) > max_samples:
            ood_val_raw = random.sample(ood_val_raw, max_samples)
        if len(iid_test_raw) > max_samples:
            iid_test_raw = random.sample(iid_test_raw, max_samples)
        if len(ood_test_raw) > max_samples:
            ood_test_raw = random.sample(ood_test_raw, max_samples)
    
    # 处理验证数据
    if iid_val_raw:
        if max_samples and len(iid_val_raw) > max_samples:
            iid_val_raw = random.sample(iid_val_raw, max_samples)
        val_raw = iid_val_raw
    else:
        # 如果没有iid_val，从train中分出验证数据
        if len(train_raw) >= 100:
            val_size = min(len(train_raw) // 4, 1000)
            val_raw = random.sample(train_raw, val_size)
            # 从train_raw中移除已选的验证数据
            val_smiles_set = set(item.get('smiles', '') for item in val_raw)
            train_raw = [item for item in train_raw if item.get('smiles', '') not in val_smiles_set]
        else:
            # 数据不够分，随机分割
            train_copy = train_raw.copy()
            random.shuffle(train_copy)
            split_point = len(train_copy) // 2
            val_raw = train_copy[:split_point]
            train_raw = train_copy[split_point:]
    
    # 提取SMILES
    train_id_smiles = [item.get('smiles') for item in train_raw if item.get('smiles')]
    val_id_smiles = [item.get('smiles') for item in val_raw if item.get('smiles')]
    train_ood_smiles = [item.get('smiles') for item in ood_val_raw if item.get('smiles')]
    test_id_smiles = [item.get('smiles') for item in iid_test_raw if item.get('smiles')]
    test_ood_smiles = [item.get('smiles') for item in ood_test_raw if item.get('smiles')]
    
    # 验证SMILES
    train_id_smiles = validate_smiles(train_id_smiles)
    val_id_smiles = validate_smiles(val_id_smiles)
    train_ood_smiles = validate_smiles(train_ood_smiles)
    test_id_smiles = validate_smiles(test_id_smiles)
    test_ood_smiles = validate_smiles(test_ood_smiles)
    
    logger.info(f"✅ OOD Detection数据处理完成:")
    logger.info(f"  训练ID: {len(train_id_smiles)}")
    logger.info(f"  验证ID: {len(val_id_smiles)}")
    logger.info(f"  训练OOD: {len(train_ood_smiles)}")
    logger.info(f"  测试ID: {len(test_id_smiles)}")
    logger.info(f"  测试OOD: {len(test_ood_smiles)}")
    
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
    """处理 GOOD 系列分子数据集并返回 SMILES 及其标签

    Args:
        validate_smiles_flag: 是否对提取到的 SMILES 进行 RDKit 校验。对 GOOD 官方预处理产物，建议关闭以提升速度。
    """

    logger.info(f"🎯 处理 {dataset_name} 数据集 (domain={domain}, shift={shift})")

    try:
        # 动态导入相应的数据集类
        if dataset_name == 'good_hiv':
            from data.good_data.good_datasets.good_hiv import GOODHIV as DatasetClass
        elif dataset_name == 'good_pcba':
            from data.good_data.good_datasets.good_pcba import GOODPCBA as DatasetClass
        elif dataset_name == 'good_zinc':
            from data.good_data.good_datasets.good_zinc import GOODZINC as DatasetClass
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")

        logger.info(f"📂 从 {data_path} 加载数据集...")

        # 加载不同的 splits
        train_dataset = DatasetClass(root=data_path, domain=domain, shift=shift, subset='train')
        val_dataset = DatasetClass(root=data_path, domain=domain, shift=shift, subset='val')
        test_dataset = DatasetClass(root=data_path, domain=domain, shift=shift, subset='test')

        # 如果有 shift，也加载 id splits
        if shift != 'no_shift':
            try:
                id_val_dataset = DatasetClass(root=data_path, domain=domain, shift=shift, subset='id_val')
                id_test_dataset = DatasetClass(root=data_path, domain=domain, shift=shift, subset='id_test')
            except Exception:
                logger.warning("无法加载id_val/id_test，将从其他数据中分割")
                id_val_dataset = None
                id_test_dataset = None
        else:
            id_val_dataset = None
            id_test_dataset = None

        # 提取 SMILES 和标签 - 支持多任务学习
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
                                # 转换为JSON可序列化的Python原生类型
                                if y.dim() > 1:
                                    # 多任务情况: [1, num_tasks] -> [num_tasks]，转为list
                                    label = y.view(-1).tolist()
                                else:
                                    # 单任务情况，转为Python标量或list
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
            # 从 train 中分出验证集
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

        # 将 OOD 数据分成训练和验证两部分
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
            logger.warning("没有找到OOD验证数据，使用测试数据的一部分")
            if test_ood_smiles:
                random.seed(seed)
                random.shuffle(test_ood_smiles)
                # 三等分：train_ood, val_ood, test_ood (确保无重叠)
                total_len = len(test_ood_smiles)
                train_split = total_len // 3
                val_split = (total_len * 2) // 3

                train_ood_smiles = test_ood_smiles[:train_split]
                val_ood_smiles = test_ood_smiles[train_split:val_split]
                test_ood_smiles = test_ood_smiles[val_split:]

                logger.info(f"OOD数据三等分: train_ood={len(train_ood_smiles)}, val_ood={len(val_ood_smiles)}, test_ood={len(test_ood_smiles)}")
            else:
                raise ValueError(
                    f"Dataset {dataset_name} does not contain proper OOD splits. "
                    f"For OOD detection training, the dataset must have separate OOD validation/test data "
                    f"that comes from a different distribution than the training data. "
                    f"Available splits: domain={domain}, shift={shift}. "
                    f"Consider using a different domain/shift combination that provides OOD splits."
                )

        # 如果指定最大样本数，进行采样
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

        # 验证 SMILES（可选）
        if validate_smiles_flag:
            train_smiles = validate_smiles(train_smiles)
            val_id_smiles = validate_smiles(val_id_smiles)
            train_ood_smiles = validate_smiles(train_ood_smiles)
            val_ood_smiles = validate_smiles(val_ood_smiles)
            test_id_smiles = validate_smiles(test_id_smiles)
            test_ood_smiles = validate_smiles(test_ood_smiles)

        logger.info(f"✅ {dataset_name} 数据加载完成:")
        logger.info(f"  训练ID: {len(train_smiles)}")
        logger.info(f"  验证ID: {len(val_id_smiles)}")
        logger.info(f"  训练OOD: {len(train_ood_smiles)}")
        logger.info(f"  验证OOD: {len(val_ood_smiles)}")
        logger.info(f"  测试ID: {len(test_id_smiles)}")
        logger.info(f"  测试OOD: {len(test_ood_smiles)}")

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
        logger.error(f"加载 {dataset_name} 失败: {e}")
        raise
