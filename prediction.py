import torch
import numpy as np
import pandas as pd
import argparse
import logging
import os
from typing import List, Union

from model import load_pretrained_model
from utils import set_seed, validate_smiles

logger = logging.getLogger(__name__)

class EnergyDPOPredictor:
    def __init__(self, model_path, args, threshold=None):
        self.model_path = model_path
        self.args = args
        self.threshold = threshold
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        set_seed(args.seed)
    
        self.model = load_pretrained_model(model_path, args).to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Using device: {self.device}")
        if threshold is not None:
            logger.info(f"OOD threshold: {threshold}")
    
    def predict_single(self, smiles):
        """Predict OOD score for single SMILES"""
        return self.predict_batch([smiles])[0]
    
    def predict_batch(self, smiles_list, batch_size=64):
        """Batch predict OOD scores for SMILES"""
        # Validate SMILES
        valid_smiles = validate_smiles(smiles_list)
        
        if len(valid_smiles) == 0:
            logger.warning("No valid SMILES found")
            return np.array([])
        
        all_scores = []
        self.model.eval()
        
        with torch.no_grad():
            for i in range(0, len(valid_smiles), batch_size):
                batch_smiles = valid_smiles[i:i+batch_size]
                try:
                    scores = self.model.predict_ood_score(batch_smiles)
                    if isinstance(scores, np.ndarray):
                        all_scores.extend(scores.tolist())
                    else:
                        all_scores.extend([scores])
                except Exception as e:
                    logger.warning(f"Prediction batch {i//batch_size} error: {e}")
                    # Fill neutral scores for failed predictions
                    all_scores.extend([0.0] * len(batch_smiles))
        
        return np.array(all_scores)
    
    def predict_with_threshold(self, smiles_list, batch_size=64):
        """
        Predict OOD labels using threshold

        Returns:
            scores: Energy scores
            predictions: Binary predictions (1=OOD, 0=ID)
        """
        scores = self.predict_batch(smiles_list, batch_size)
        
        if self.threshold is None:
            logger.warning("Threshold not set, only returning scores")
            return scores, None
        
        predictions = (scores > self.threshold).astype(int)
        return scores, predictions
    
    def predict_from_file(self, input_file, output_file=None, smiles_column='smiles'):
        """
        Predict OOD scores from CSV file

        Args:
            input_file: Input CSV file path
            output_file: Output CSV file path
            smiles_column: SMILES column name
        """
        # Load data
        df = pd.read_csv(input_file)
        
        if smiles_column not in df.columns:
            raise ValueError(f"Column '{smiles_column}' not found in {input_file}")
        
        smiles_list = df[smiles_column].tolist()
        logger.info(f"Loaded {len(smiles_list)} molecules from {input_file}")
        
        # Make predictions
        scores = self.predict_batch(smiles_list)
        
        # Add results to dataframe
        df['energy_score'] = scores
        
        if self.threshold is not None:
            df['ood_prediction'] = (scores > self.threshold).astype(int)
            df['ood_probability'] = 1 / (1 + np.exp(-scores))  # Sigmoid transformation
        
        # Save results
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        
        return df
    
    def set_threshold(self, threshold):
        """Set OOD detection threshold"""
        self.threshold = threshold
        logger.info(f"Threshold set to {threshold}")

def parse_args():
    parser = argparse.ArgumentParser(description='Energy-DPO Prediction')
    
    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dpo_beta", type=float, default=0.1)

    # Input/Output
    parser.add_argument("--input_file", type=str, help="Input CSV file containing SMILES")
    parser.add_argument("--output_file", type=str, help="Output CSV file")
    parser.add_argument("--smiles_column", type=str, default="smiles", help="SMILES column name")
    parser.add_argument("--smiles", type=str, nargs='+', help="SMILES strings to predict")

    # Prediction
    parser.add_argument("--threshold", type=float, help="OOD detection threshold")
    parser.add_argument("--batch_size", type=int, default=64)

    # Others
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO)
    
    args = parse_args()
    
    # Create predictor
    predictor = EnergyDPOPredictor(args.model_path, args, args.threshold)

    if args.input_file:
        # Predict from file
        results_df = predictor.predict_from_file(
            args.input_file, 
            args.output_file, 
            args.smiles_column
        )
        
        # Print summary
        print("\nPrediction Summary:")
        print(f"Total molecules: {len(results_df)}")
        print(f"Average energy score: {results_df['energy_score'].mean():.4f}")
        print(f"Energy score std dev: {results_df['energy_score'].std():.4f}")
        
        if args.threshold is not None:
            ood_count = results_df['ood_prediction'].sum()
            print(f"OOD molecules: {ood_count} ({ood_count/len(results_df)*100:.1f}%)")
    
    elif args.smiles:
        # Predict individual SMILES
        scores = predictor.predict_batch(args.smiles)
        
        print("\nPrediction Results:")
        print("-" * 80)
        for smiles, score in zip(args.smiles, scores):
            if args.threshold is not None:
                prediction = "OOD" if score > args.threshold else "ID"
                print(f"SMILES: {smiles:<40} Score: {score:6.4f} Prediction: {prediction}")
            else:
                print(f"SMILES: {smiles:<40} Score: {score:6.4f}")
        print("-" * 80)
    
    else:
        print("Please provide --input_file or --smiles arguments")

if __name__ == "__main__":
    main()