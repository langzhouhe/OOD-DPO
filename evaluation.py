import torch
import numpy as np
import pandas as pd
import logging
import os
import argparse
import json  
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve

from model import load_pretrained_model
from data_loader import EnergyDPODataLoader
from utils import set_seed, validate_smiles

logger = logging.getLogger(__name__)

class EnergyDPOEvaluator:
    def __init__(self, model_path, args):
        self.model_path = model_path
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        set_seed(args.seed)
        
        self.model = load_pretrained_model(model_path, args).to(self.device)
        self.model.eval()
        
        logger.info(f"Evaluator initialized on device: {self.device}")

    def predict_batch_from_features(self, features):
        """Batch prediction directly from pre-computed features."""
        if features.shape[0] == 0:
            return np.array([])
        
        self.model.eval()
        batch_size = getattr(self.args, 'eval_batch_size', 64)
        all_scores = []
        
        logger.info(f"Predicting from {features.shape[0]} features, batch size: {batch_size}")
        
        with torch.no_grad():
            for i in range(0, features.shape[0], batch_size):
                batch_features = features[i:i + batch_size].to(self.device)
                batch_scores = self.model.predict_ood_score_from_features(batch_features)

                # FIXED: Handle both tensor and numpy array returns
                if isinstance(batch_scores, torch.Tensor):
                    all_scores.append(batch_scores.cpu().numpy())
                elif isinstance(batch_scores, np.ndarray):
                    all_scores.append(batch_scores)
                else:
                    # Handle other types by converting to numpy
                    all_scores.append(np.array(batch_scores))
                
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"Processed {i + len(batch_features)}/{features.shape[0]} samples")
        
        return np.concatenate(all_scores)

    def predict_batch(self, smiles_list, batch_size=None):
        """Batch prediction from SMILES strings."""
        if batch_size is None:
            batch_size = getattr(self.args, 'eval_batch_size', 64)
            
        all_scores = []
        self.model.eval()
        
        logger.info(f"Predicting from {len(smiles_list)} SMILES, batch size: {batch_size}")
        
        with torch.no_grad():
            for i in range(0, len(smiles_list), batch_size):
                batch_smiles = smiles_list[i:i+batch_size]
                try:
                    scores = self.model.predict_ood_score(batch_smiles)
                    if isinstance(scores, np.ndarray):
                        all_scores.extend(scores.tolist())
                    else:
                        all_scores.extend([scores])
                except Exception as e:
                    logger.warning(f"Prediction batch {i//batch_size} failed: {e}")
                    all_scores.extend([0.0] * len(batch_smiles))
                
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"Processed {i + len(batch_smiles)}/{len(smiles_list)} samples")
        
        return np.array(all_scores)

    def _compute_ood_metrics(self, id_scores, ood_scores):
        """Compute standard OOD detection metrics."""
        all_scores = np.concatenate([id_scores, ood_scores])
        all_labels = np.concatenate([
            np.zeros(len(id_scores)),  # ID = 0
            np.ones(len(ood_scores))   # OOD = 1
        ])
        
        auroc = roc_auc_score(all_labels, all_scores)
        precision, recall, _ = precision_recall_curve(all_labels, all_scores)
        aupr = auc(recall, precision)
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)

        fpr95_idx = np.where(tpr >= 0.95)[0]
        fpr95 = fpr[fpr95_idx[0]] if len(fpr95_idx) > 0 else 1.0
        
        id_mean_energy = np.mean(id_scores)
        ood_mean_energy = np.mean(ood_scores)
        energy_separation = ood_mean_energy - id_mean_energy
        
        return {
            'auroc': auroc,
            'aupr': aupr,
            'fpr95': fpr95,
            'id_mean_energy': id_mean_energy,
            'ood_mean_energy': ood_mean_energy,
            'energy_separation': energy_separation,
            'id_count': len(id_scores),
            'ood_count': len(ood_scores),
            'id_std_energy': np.std(id_scores),
            'ood_std_energy': np.std(ood_scores)
        }

    def _save_results(self, results, id_scores, ood_scores, output_dir):
        """Save evaluation results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics to JSON
        results_file = os.path.join(output_dir, 'ood_evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump({k: v for k, v in results.items() if isinstance(v, (int, float))}, f, indent=2)
        
        # Save detailed predictions
        predictions_df = pd.DataFrame({
            'score': np.concatenate([id_scores, ood_scores]),
            'true_label': np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))]),
            'sample_type': ['ID'] * len(id_scores) + ['OOD'] * len(ood_scores)
        })
        predictions_df.to_csv(os.path.join(output_dir, 'detailed_predictions.csv'), index=False)
        
        logger.info(f"Results saved to: {results_file}")

    def evaluate_ood_detection_from_features(self, id_features, ood_features, output_dir=None):
        """Evaluate OOD detection using pre-computed features."""
        if id_features.shape[0] == 0 or ood_features.shape[0] == 0:
            raise ValueError("Empty feature tensors, cannot evaluate")
        
        logger.info(f"Evaluating on {id_features.shape[0]} ID and {ood_features.shape[0]} OOD samples (from features)")
        
        logger.info("Computing ID energy scores from features...")
        id_scores = self.predict_batch_from_features(id_features)
        
        logger.info("Computing OOD energy scores from features...")
        ood_scores = self.predict_batch_from_features(ood_features)

        results = self._compute_ood_metrics(id_scores, ood_scores)
        
        logger.info("Evaluation Results:")
        logger.info(f"  AUROC: {results['auroc']:.4f}")
        logger.info(f"  AUPR: {results['aupr']:.4f}")
        logger.info(f"  FPR95: {results['fpr95']:.4f}")
        logger.info(f"  Energy Separation: {results['energy_separation']:.4f}")
        logger.info(f"  ID Energy: {results['id_mean_energy']:.4f} ± {results['id_std_energy']:.4f}")
        logger.info(f"  OOD Energy: {results['ood_mean_energy']:.4f} ± {results['ood_std_energy']:.4f}")
        
        if output_dir:
            self._save_results(results, id_scores, ood_scores, output_dir)
        
        return results

    def evaluate_ood_detection(self, id_smiles, ood_smiles, output_dir=None):
        """Evaluate OOD detection from SMILES strings."""
        valid_id_smiles = [s for s in validate_smiles(id_smiles) if s is not None]
        valid_ood_smiles = [s for s in validate_smiles(ood_smiles) if s is not None]
        
        if len(valid_id_smiles) == 0 or len(valid_ood_smiles) == 0:
            raise ValueError("Insufficient valid samples for evaluation")
        
        logger.info(f"Evaluating on {len(valid_id_smiles)} ID and {len(valid_ood_smiles)} OOD samples")
        
        logger.info("Computing ID energy scores...")
        id_scores = self.predict_batch(valid_id_smiles)
        logger.info("Computing OOD energy scores...")
        ood_scores = self.predict_batch(valid_ood_smiles)

        results = self._compute_ood_metrics(id_scores, ood_scores)
        
        logger.info("Evaluation Results:")
        logger.info(f"  AUROC: {results['auroc']:.4f}")
        logger.info(f"  AUPR: {results['aupr']:.4f}")
        logger.info(f"  FPR95: {results['fpr95']:.4f}")
        logger.info(f"  Energy Separation: {results['energy_separation']:.4f}")
        logger.info(f"  ID Energy: {results['id_mean_energy']:.4f} ± {results['id_std_energy']:.4f}")
        logger.info(f"  OOD Energy: {results['ood_mean_energy']:.4f} ± {results['ood_std_energy']:.4f}")
        
        if output_dir:
            self._save_results(results, id_scores, ood_scores, output_dir)
        
        return results

def parse_args():
    parser = argparse.ArgumentParser(description='Energy-DPO Evaluation')
    
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    
    parser.add_argument("--dataset", type=str, default="drugood")
    parser.add_argument("--drugood_subset", type=str, default="lbap_general_ec50_scaffold")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--data_file", type=str, help="Specific data file path")
    
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="./evaluation_results")
    
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO)
    
    args = parse_args()
    
    # Create evaluator
    evaluator = EnergyDPOEvaluator(args.model_path, args)
    
    # Load test data
    data_loader = EnergyDPODataLoader(args)
    test_data = data_loader.get_final_test_data()
    
    # Run evaluation
    results = evaluator.evaluate_ood_detection(
        test_data['id_smiles'],
        test_data['ood_smiles'],
        output_dir=args.output_dir
    )
    
    # Print summary
    print("\n" + "="*50)
    print("Evaluation Summary")
    print("="*50)
    print(f"AUROC: {results['auroc']:.4f}")
    print(f"AUPR: {results['aupr']:.4f}")
    print(f"Energy Separation: {results['energy_separation']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
