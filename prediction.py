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
        使用阈值预测OOD标签
        
        Returns:
            scores: 能量分数
            predictions: 二进制预测(1=OOD, 0=ID)
        """
        scores = self.predict_batch(smiles_list, batch_size)
        
        if self.threshold is None:
            logger.warning("未设置阈值，只返回分数")
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
        # 加载数据
        df = pd.read_csv(input_file)
        
        if smiles_column not in df.columns:
            raise ValueError(f"列 '{smiles_column}' 在 {input_file} 中未找到")
        
        smiles_list = df[smiles_column].tolist()
        logger.info(f"从 {input_file} 加载了 {len(smiles_list)} 个分子")
        
        # 做预测
        scores = self.predict_batch(smiles_list)
        
        # 将结果添加到数据框
        df['energy_score'] = scores
        
        if self.threshold is not None:
            df['ood_prediction'] = (scores > self.threshold).astype(int)
            df['ood_probability'] = 1 / (1 + np.exp(-scores))  # Sigmoid变换
        
        # 保存结果
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"结果已保存到 {output_file}")
        
        return df
    
    def set_threshold(self, threshold):
        """设置OOD检测阈值"""
        self.threshold = threshold
        logger.info(f"阈值设置为 {threshold}")

def parse_args():
    parser = argparse.ArgumentParser(description='Energy-DPO预测')
    
    # 模型
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    
    # 输入/输出
    parser.add_argument("--input_file", type=str, help="包含SMILES的输入CSV文件")
    parser.add_argument("--output_file", type=str, help="输出CSV文件")
    parser.add_argument("--smiles_column", type=str, default="smiles", help="SMILES列名")
    parser.add_argument("--smiles", type=str, nargs='+', help="要预测的SMILES字符串")
    
    # 预测
    parser.add_argument("--threshold", type=float, help="OOD检测阈值")
    parser.add_argument("--batch_size", type=int, default=64)
    
    # 其他
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO)
    
    args = parse_args()
    
    # 创建预测器
    predictor = EnergyDPOPredictor(args.model_path, args, args.threshold)
    
    if args.input_file:
        # 从文件预测
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