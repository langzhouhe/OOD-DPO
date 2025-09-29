import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import os
from utils import smiles2graph, unimol_collate_fn, create_atom_token_mapping
import warnings

warnings.filterwarnings("ignore", message="Failed to find the pandas get_adjustment.*")
warnings.filterwarnings("ignore", message="Failed to patch pandas.*")

logger = logging.getLogger(__name__)

# Global instances for efficiency
_minimol_instance = None
_unimol_instance = None
_unimol_dictionary = None

def _initialize_unimol():
    """Initialize UniMol model and dictionary"""
    global _unimol_instance, _unimol_dictionary
    
    if _unimol_instance is None:
        try:
            logger.info("Loading UniMol model...")
            
            # Check required dependencies
            try:
                from unicore import checkpoint_utils
                from unicore.data import Dictionary
                from unimol.models import UniMolModel
            except ImportError as e:
                raise ImportError(f"UniMol dependencies missing: {e}")
            
            # Load or create dictionary
            dict_path = "./weights/dict.txt"
            if os.path.exists(dict_path):
                try:
                    _unimol_dictionary = Dictionary.load(dict_path)
                    logger.info(f"Dictionary loaded from {dict_path}")
                except Exception:
                    _unimol_dictionary = None
            
            if _unimol_dictionary is None:
                logger.info("Creating new Dictionary")
                _unimol_dictionary = Dictionary()
                
                # Add special tokens
                _unimol_dictionary.add_symbol('[PAD]', is_special=True)
                _unimol_dictionary.add_symbol('[UNK]', is_special=True)
                _unimol_dictionary.add_symbol('[CLS]', is_special=True)
                _unimol_dictionary.add_symbol('[SEP]', is_special=True)
                _unimol_dictionary.add_symbol('[MASK]', is_special=True)
                
                # Add common atom symbols
                common_atoms = [
                    'H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I',
                    'P', 'B', 'Si', 'Na', 'Mg', 'K', 'Ca',
                    'Fe', 'Zn', 'Cu', 'Mn', 'Al', 'Ag', 'Au', 'Pt',
                    'As', 'Se', 'Sn'
                ]
                
                for symbol in common_atoms[:26]:  # Total 31 tokens (5 special + 26 atoms)
                    _unimol_dictionary.add_symbol(symbol)
                
                # Save dictionary
                try:
                    os.makedirs("./weights", exist_ok=True)
                    if hasattr(_unimol_dictionary, 'save'):
                        _unimol_dictionary.save(dict_path)
                except Exception:
                    pass
            
            logger.info(f"Dictionary initialized with {len(_unimol_dictionary)} tokens")
            
            # Create model with basic configuration
            import argparse
            args = argparse.Namespace()
            args.arch = 'unimol_base'
            args.encoder_layers = 15
            args.encoder_attention_heads = 64
            args.encoder_embed_dim = 512
            args.encoder_ffn_embed_dim = 2048
            args.dropout = 0.1
            args.attention_dropout = 0.1
            args.activation_dropout = 0.0
            args.pooler_dropout = 0.0
            args.max_seq_len = 512
            args.post_ln = False
            args.mode = 'infer'
            args.remove_hydrogen = False
            args.no_token_positional_embeddings = False
            args.encoder_normalize_before = True
            args.masked_token_loss = -1.0
            args.masked_coord_loss = -1.0
            args.masked_dist_loss = -1.0
            args.x_norm_loss = -1.0
            args.delta_pair_repr_norm_loss = -1.0
            args.activation_fn = 'gelu'
            args.pooler_activation_fn = 'tanh'
            args.emb_dropout = 0.1
            
            _unimol_instance = UniMolModel(args, _unimol_dictionary)
            
            # Load pretrained weights if available
            pretrained_path = "./weights/mol_pre_no_h_220816.pt"
            if os.path.exists(pretrained_path):
                try:
                    logger.info("Loading pretrained weights...")
                    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
                    
                    model_state_dict = _unimol_instance.state_dict()
                    pretrained_state_dict = checkpoint['model']
                    
                    filtered_state_dict = {}
                    for name, param in pretrained_state_dict.items():
                        if name in model_state_dict and param.shape == model_state_dict[name].shape:
                            filtered_state_dict[name] = param
                    
                    _unimol_instance.load_state_dict(filtered_state_dict, strict=False)
                    logger.info(f"Loaded {len(filtered_state_dict)} parameters")
                    
                except Exception as e:
                    logger.warning(f"Failed to load pretrained weights: {e}")
            else:
                logger.warning(f"Pretrained weights not found: {pretrained_path}")
            
            logger.info("UniMol model initialization complete")
            
        except Exception as e:
            logger.error(f"UniMol initialization failed: {e}")
            raise
            
    return _unimol_instance, _unimol_dictionary

def _initialize_minimol():
    """Initialize Minimol model"""
    global _minimol_instance
    if _minimol_instance is None:
        try:
            from minimol import Minimol
            logger.info("Loading Minimol model...")
            # Correct initialization based on actual Minimol API
            _minimol_instance = Minimol()  # Default batch_size=100
            logger.info("Minimol model loaded successfully")
        except Exception as e:
            logger.error(f"Minimol loading failed: {e}")
            raise
    return _minimol_instance

class MinimolEncodingError(Exception):
    """Minimol encoding exception"""
    def __init__(self, smiles, original_error):
        self.smiles = smiles
        self.original_error = original_error
        super().__init__(f"Minimol encoding failed: {smiles} -> {original_error}")

class UniMolEncodingError(Exception):
    """UniMol encoding exception"""
    def __init__(self, smiles, original_error):
        self.smiles = smiles
        self.original_error = original_error
        super().__init__(f"UniMol encoding failed: {smiles} -> {original_error}")

class MinimolEncoder:
    def __init__(self):
        self.model = _initialize_minimol()
    
    def encode_smiles(self, smiles_list):
        """Encode SMILES list to 512-dim feature vectors"""
        if not smiles_list:
            return torch.empty(0, 512)
        
        try:
            # Use the correct Minimol API: model(smiles_list)
            features = self.model(smiles_list)
            
            if isinstance(features, list):
                # Convert list of tensors to a single tensor
                features = torch.stack(features)
            elif isinstance(features, np.ndarray):
                features = torch.from_numpy(features).float()
            elif not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)
            
            if features.dim() == 1 and len(smiles_list) == 1:
                features = features.unsqueeze(0)
            
            return features
        
        except Exception as e:
            failed_smiles = smiles_list if len(smiles_list) <= 3 else f"{smiles_list[:3]}... (total {len(smiles_list)})"
            logger.error(f"Minimol encoding failed: {e}")
            raise MinimolEncodingError(failed_smiles, e)

class UniMolEncoder:
    def __init__(self):
        self.model, self.dictionary = _initialize_unimol()
        self.atom_token_mapping = self._create_token_mapping()
    
    def _create_token_mapping(self):
        """Create token mapping from dictionary"""
        token_mapping = {}
        
        # Special tokens
        try:
            token_mapping['[PAD]'] = self.dictionary.pad()
            token_mapping['[UNK]'] = self.dictionary.unk()
            token_mapping['[CLS]'] = getattr(self.dictionary, 'bos', lambda: 2)()
            token_mapping['[SEP]'] = getattr(self.dictionary, 'eos', lambda: 3)()
            token_mapping['[MASK]'] = 4
        except Exception:
            token_mapping.update({'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4})
        
        # Compatibility mappings
        token_mapping.update({
            '<pad>': token_mapping['[PAD]'],
            '<unk>': token_mapping['[UNK]'],
            '<s>': token_mapping['[CLS]'],
            '</s>': token_mapping['[SEP]'],
            '<mask>': token_mapping['[MASK]']
        })
        
        # Atom symbols
        if hasattr(self.dictionary, 'symbols'):
            for i, symbol in enumerate(self.dictionary.symbols):
                if symbol not in token_mapping:
                    token_mapping[symbol] = i
        else:
            # Fallback atom mapping
            atom_symbols = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'P']
            for i, symbol in enumerate(atom_symbols, 5):
                token_mapping[symbol] = i
        
        return token_mapping
    
    def encode_smiles(self, smiles_list):
        """Encode SMILES list using UniMol pipeline"""
        if not smiles_list:
            return torch.empty(0, 512)
        
        try:
            # SMILES to graph data
            graph_samples = []
            for smiles in smiles_list:
                try:
                    graph_data = smiles2graph(smiles)
                    graph_samples.append(graph_data)
                except Exception:
                    # Minimal placeholder
                    graph_samples.append({
                        'atoms': ['C'],
                        'coordinates': np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                        'distance_matrix': np.array([[0.0]], dtype=np.float32),
                        'edge_types': np.array([[0]], dtype=np.int64),
                        'mol': None,
                        'smiles': smiles
                    })
            
            # Batch collation
            try:
                batch_data = unimol_collate_fn(graph_samples, self.atom_token_mapping)
            except Exception:
                batch_data = self._fallback_collate_fn(graph_samples)
            
            # Handle nested data structure
            if isinstance(batch_data, dict):
                if 'net_input' in batch_data:
                    actual_data = batch_data['net_input']
                elif 'batched_data' in batch_data:
                    actual_data = batch_data['batched_data']
                else:
                    actual_data = batch_data
                
                # Standardize keys
                key_mappings = {
                    'src_tokens': 'tokens',
                    'src_coord': 'coordinates',
                    'src_distance': 'distance_matrix',
                    'src_edge_type': 'edge_types'
                }
                
                standardized_data = {}
                for old_key, new_key in key_mappings.items():
                    if old_key in actual_data:
                        standardized_data[new_key] = actual_data[old_key]
                
                batch_data = standardized_data if standardized_data else self._fallback_collate_fn(graph_samples)
            
            # Move to device
            device = next(self.model.parameters()).device
            for key in ['tokens', 'coordinates', 'distance_matrix', 'edge_types']:
                if key in batch_data and isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(device)
            
            # UniMol inference
            self.model.eval()
            with torch.no_grad():
                try:
                    model_output = self.model.forward(
                        src_tokens=batch_data['tokens'],
                        src_coord=batch_data['coordinates'],
                        src_distance=batch_data['distance_matrix'],
                        src_edge_type=batch_data['edge_types'],
                        features_only=True
                    )
                    
                    if isinstance(model_output, tuple):
                        encoder_out = model_output[0]
                    elif isinstance(model_output, dict) and 'encoder_out' in model_output:
                        encoder_out = model_output['encoder_out']
                    else:
                        encoder_out = model_output
                        
                except Exception:
                    # Simplified fallback
                    encoder_out = self.model(
                        batch_data['tokens'],
                        batch_data['distance_matrix'],
                        batch_data['coordinates'],
                        batch_data['edge_types']
                    )
                    if isinstance(encoder_out, tuple):
                        encoder_out = encoder_out[0]
                
                # Extract molecular features from [CLS] token
                if isinstance(encoder_out, torch.Tensor):
                    molecule_features = encoder_out[:, 0, :]
                elif isinstance(encoder_out, tuple):
                    molecule_features = encoder_out[0][:, 0, :]
                elif isinstance(encoder_out, dict):
                    key = 'last_hidden_state' if 'last_hidden_state' in encoder_out else list(encoder_out.keys())[0]
                    molecule_features = encoder_out[key][:, 0, :]
                else:
                    raise ValueError(f"Unsupported model output type: {type(encoder_out)}")
                
                # Project to 512 dimensions if needed
                if molecule_features.size(-1) != 512:
                    if not hasattr(self, 'projection'):
                        self.projection = nn.Linear(molecule_features.size(-1), 512).to(device)
                    molecule_features = self.projection(molecule_features)
                
                return molecule_features
                
        except Exception as e:
            failed_smiles = smiles_list if len(smiles_list) <= 3 else f"{smiles_list[:3]}... (total {len(smiles_list)})"
            logger.error(f"UniMol encoding failed: {e}")
            raise UniMolEncodingError(failed_smiles, e)
    
    def _fallback_collate_fn(self, samples):
        """Fallback batch collation function"""
        batch_size = len(samples)
        if batch_size == 0:
            return {}
        
        max_atoms = max(len(sample['atoms']) for sample in samples)
        max_atoms = max(max_atoms, 1)
        
        batch_tokens = torch.zeros(batch_size, max_atoms + 2, dtype=torch.long)
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
            batch_tokens[i, 0] = self.atom_token_mapping.get('[CLS]', 2)
            for j, atom in enumerate(atoms):
                batch_tokens[i, j + 1] = self.atom_token_mapping.get(atom, self.atom_token_mapping.get('[UNK]', 1))
            if num_atoms + 1 < max_atoms + 2:
                batch_tokens[i, num_atoms + 1] = self.atom_token_mapping.get('[SEP]', 3)
            
            # Set coordinates
            batch_coordinates[i, 0] = torch.zeros(3)  # CLS
            batch_coordinates[i, 1:num_atoms + 1] = torch.from_numpy(coordinates)
            if num_atoms + 1 < max_atoms + 2:
                batch_coordinates[i, num_atoms + 1] = torch.zeros(3)  # SEP
            
            # Set distance matrix and edge types
            ext_distance_matrix = torch.zeros(max_atoms + 2, max_atoms + 2)
            ext_distance_matrix[1:num_atoms + 1, 1:num_atoms + 1] = torch.from_numpy(distance_matrix)
            batch_distance[i] = ext_distance_matrix
            
            ext_edge_types = torch.zeros(max_atoms + 2, max_atoms + 2, dtype=torch.long)
            ext_edge_types[1:num_atoms + 1, 1:num_atoms + 1] = torch.from_numpy(edge_types)
            batch_edge_type[i] = ext_edge_types
        
        return {
            'tokens': batch_tokens,
            'coordinates': batch_coordinates,
            'distance_matrix': batch_distance,
            'edge_types': batch_edge_type
        }

class EnergyHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        energy = self.fc3(x)
        return energy

class EnergyDPOModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embedding_dim = 512
        self.hidden_dim = getattr(args, 'hidden_dim', 256)
        self.beta = getattr(args, 'dpo_beta', 0.1)
        self.lambda_reg = getattr(args, 'lambda_reg', 0.01)  # L2 regularization strength
        self.foundation_model = getattr(args, 'foundation_model', 'minimol')
        
        logger.info(f"Initializing EnergyDPOModel (using {self.foundation_model})")
        
        # Select encoder
        if self.foundation_model == 'minimol':
            self.encoder = MinimolEncoder()
        elif self.foundation_model == 'unimol':
            self.encoder = UniMolEncoder()
        else:
            raise ValueError(f"Unsupported foundation_model: {self.foundation_model}")
        
        # Energy head
        self.energy_head = EnergyHead(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )
    
    def encode_smiles(self, smiles_list):
        if not smiles_list:
            return torch.empty(0, self.embedding_dim)
        
        device = next(self.parameters()).device
        embeddings = self.encoder.encode_smiles(smiles_list)
        return embeddings.to(device)
    

    def compute_energy_dpo_loss_from_features(self, id_features, ood_features):
        device = next(self.parameters()).device
        id_features = id_features.to(device)
        ood_features = ood_features.to(device)

        # 1) compute energies
        id_E  = self.energy_head(id_features).squeeze(-1)   # [n]
        ood_E = self.energy_head(ood_features).squeeze(-1)  # [m]

        # 3) full pairwise margins: diff[i, j] = ood_E[j] - id_E[i], shape [n, m]
        diff = ood_E.unsqueeze(0) - id_E.unsqueeze(1)       # [n, m]

        # 4) DPO main term: -log σ(β * diff) = softplus(-β * diff)
        dpo_loss = torch.nn.functional.softplus(-self.beta * diff).mean()

        # 5) same quadratic regularization as before
        id_energy_reg  = id_E.pow(2).mean()
        ood_energy_reg = ood_E.pow(2).mean()
        total_loss = dpo_loss + self.lambda_reg * (id_energy_reg + ood_energy_reg)

        # 6) Training dynamics indicators for "hard pairs corrected first"
        with torch.no_grad():
            # Pr(ΔEφ < 0): proportion of misranked pairs (OOD energy < ID energy)
            misranked_ratio = (diff < 0).float().mean().item()

            # Pr(|ΔEφ| < ε): proportion of pairs near decision boundary (ε=0.05)
            boundary_ratio = (diff.abs() < 0.05).float().mean().item()

            # E[ΔEφ]: average margin (energy difference)
            avg_margin = diff.mean().item()

        loss_dict = {
            'dpo_loss': dpo_loss.item(),
            'id_energy': id_E.mean().item(),
            'ood_energy': ood_E.mean().item(),
            'energy_separation': (ood_E.mean() - id_E.mean()).item(),
            'total_loss': total_loss.item(),
            # Training dynamics indicators
            'misranked_ratio': misranked_ratio,    # Pr(ΔEφ < 0)
            'boundary_ratio': boundary_ratio,      # Pr(|ΔEφ| < 0.05)
            'avg_margin': avg_margin               # E[ΔEφ]
        }
        return total_loss, loss_dict

    
    def compute_energy_dpo_loss(self, id_smiles, ood_smiles):
        id_features = self.encode_smiles(id_smiles)
        ood_features = self.encode_smiles(ood_smiles)
        return self.compute_energy_dpo_loss_from_features(id_features, ood_features)
    
    def forward(self, batch_data):
        if 'id_features' in batch_data and 'ood_features' in batch_data:
            loss, loss_dict = self.compute_energy_dpo_loss_from_features(
                batch_data['id_features'], 
                batch_data['ood_features']
            )
        else:
            # Fallback to SMILES encoding
            id_smiles = batch_data['id_smiles']
            ood_smiles = batch_data['ood_smiles']
            loss, loss_dict = self.compute_energy_dpo_loss(id_smiles, ood_smiles)
        
        return loss, loss_dict
    
    def _compute_energy_scores_from_features(self, features):
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            features = features.to(device)
            energy_scores = self.energy_head(features)
            return energy_scores.squeeze().cpu().numpy()
    
    def predict_ood_score(self, smiles_list):
        features = self.encode_smiles(smiles_list)
        return self._compute_energy_scores_from_features(features)
    
    def predict_ood_score_from_features(self, features):
        return self._compute_energy_scores_from_features(features)


class EnergyBCEModel(EnergyDPOModel):
    """Binary Cross Entropy Loss variant"""

    def __init__(self, args):
        super().__init__(args)
        logger.info("Initializing EnergyBCEModel (BCE Loss, logits, no smoothing)")

    def compute_energy_dpo_loss_from_features(self, id_features, ood_features):
        device = next(self.parameters()).device
        id_features = id_features.to(device)
        ood_features = ood_features.to(device)

        # Compute energy values (used as logits)
        id_logits = self.energy_head(id_features).squeeze(-1)
        ood_logits = self.energy_head(ood_features).squeeze(-1)

        # Labels: ID=0, OOD=1
        id_targets = torch.zeros_like(id_logits)
        ood_targets = torch.ones_like(ood_logits)

        all_logits = torch.cat([id_logits, ood_logits], dim=0)
        all_targets = torch.cat([id_targets, ood_targets], dim=0)

        # Logits-based BCE, consistent with standard practice
        bce_loss = F.binary_cross_entropy_with_logits(all_logits, all_targets)

        # Unified L2 output regularization λ·E[E(x)^2]
        id_energy_reg = torch.mean(id_logits**2)
        ood_energy_reg = torch.mean(ood_logits**2)
        total_loss = bce_loss + self.lambda_reg * (id_energy_reg + ood_energy_reg)

        # Energy separation measure (for evaluation)
        energy_diff = ood_logits.mean() - id_logits.mean()

        loss_dict = {
            'bce_loss': bce_loss.item(),
            'id_energy': id_logits.mean().item(),
            'ood_energy': ood_logits.mean().item(),
            'energy_separation': energy_diff.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict


class EnergyMSEModel(EnergyDPOModel):
    """Mean Squared Error Loss variant"""

    def __init__(self, args):
        super().__init__(args)
        logger.info("Initializing EnergyMSEModel (MSE on sigmoid probs)")

    def compute_energy_dpo_loss_from_features(self, id_features, ood_features):
        device = next(self.parameters()).device
        id_features = id_features.to(device)
        ood_features = ood_features.to(device)

        # Compute energy values -> probabilities (sigmoid)
        id_logits = self.energy_head(id_features).squeeze(-1)
        ood_logits = self.energy_head(ood_features).squeeze(-1)
        id_probs = torch.sigmoid(id_logits)
        ood_probs = torch.sigmoid(ood_logits)

        # Targets: ID=0, OOD=1
        id_targets = torch.zeros_like(id_probs)
        ood_targets = torch.ones_like(ood_probs)

        # Pointwise probability MSE
        id_mse = F.mse_loss(id_probs, id_targets)
        ood_mse = F.mse_loss(ood_probs, ood_targets)
        mse_loss = id_mse + ood_mse

        # Unified L2 output regularization λ·E[E(x)^2]
        id_energy_reg = torch.mean(id_logits**2)
        ood_energy_reg = torch.mean(ood_logits**2)
        total_loss = mse_loss + self.lambda_reg * (id_energy_reg + ood_energy_reg)

        # Energy separation measure (for evaluation)
        energy_diff = ood_logits.mean() - id_logits.mean()

        loss_dict = {
            'mse_loss': mse_loss.item(),
            'id_energy': id_logits.mean().item(),
            'ood_energy': ood_logits.mean().item(),
            'energy_separation': energy_diff.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict


class EnergyHingeModel(EnergyDPOModel):
    """Hinge Loss variant"""

    def __init__(self, args):
        super().__init__(args)
        self.margin = getattr(args, 'hinge_margin', 1.0)
        # in-batch hard mining fraction (0 disables)
        self.topk_frac = float(getattr(args, 'hinge_topk', 0.0) or 0.0)
        # squared hinge option
        self.squared = bool(getattr(args, 'hinge_squared', False))
        logger.info(f"Initializing EnergyHingeModel (Hinge Loss, margin={self.margin})")

    def compute_energy_dpo_loss_from_features(self, id_features, ood_features):
        device = next(self.parameters()).device
        id_features = id_features.to(device)
        ood_features = ood_features.to(device)

        # Compute energy values
        id_energies = self.energy_head(id_features)
        ood_energies = self.energy_head(ood_features)

        # Hinge Loss: ensure ood_energy - id_energy > margin
        # Compute hinge loss for all ID-OOD pairs
        id_energies_expanded = id_energies.unsqueeze(1)  # [batch_id, 1]
        ood_energies_expanded = ood_energies.unsqueeze(0)  # [1, batch_ood]

        # Compute energy differences for all pairs
        energy_diffs = ood_energies_expanded - id_energies_expanded  # [batch_id, batch_ood]

        # Standard hinge or squared hinge
        raw = torch.clamp(self.margin - energy_diffs, min=0)
        hinge_losses = raw.pow(2) if self.squared else raw

        # In-batch hard mining: only take hardest top-k fraction pairs
        if self.topk_frac and self.topk_frac > 0.0:
            flat = hinge_losses.view(-1)
            k = max(1, int(flat.numel() * min(self.topk_frac, 1.0)))
            topk_vals, _ = torch.topk(flat, k, largest=True, sorted=False)
            hinge_loss = topk_vals.mean()
        else:
            hinge_loss = hinge_losses.mean()

        # L2 regularization term
        id_energy_reg = torch.mean(id_energies**2)
        ood_energy_reg = torch.mean(ood_energies**2)

        # Total loss: Hinge + unified L2 regularization
        total_loss = hinge_loss + self.lambda_reg * (id_energy_reg + ood_energy_reg)

        # Energy separation measure (for evaluation)
        energy_diff = ood_energies.mean() - id_energies.mean()

        loss_dict = {
            'hinge_loss': hinge_loss.item(),
            'id_energy': id_energies.mean().item(),
            'ood_energy': ood_energies.mean().item(),
            'energy_separation': energy_diff.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict

def create_model(args):
    """Create model - supports ablation experiment variants"""
    loss_type = getattr(args, 'loss_type', 'dpo')  # Default to DPO
    
    if loss_type == 'dpo':
        return EnergyDPOModel(args)
    elif loss_type == 'bce':
        return EnergyBCEModel(args)
    elif loss_type == 'mse':
        return EnergyMSEModel(args)
    elif loss_type == 'hinge':
        return EnergyHingeModel(args)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Choose from ['dpo', 'bce', 'mse', 'hinge']")


def load_pretrained_model(model_path, args):
    """Load pretrained model - supports ablation experiment variants"""
    model = create_model(args)
    
    if model_path and model_path != "null":
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}")
            logger.info("Using randomly initialized model")
    
    return model

    
def reset_minimol():
    global _minimol_instance
    _minimol_instance = None
    logger.info("Minimol instance reset")

def reset_unimol():
    global _unimol_instance, _unimol_dictionary
    _unimol_instance = None
    _unimol_dictionary = None
    logger.info("UniMol instance reset")
