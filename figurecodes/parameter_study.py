#!/usr/bin/env python3
"""
Energy-DPO 参数研究脚本
用于系统研究 beta 和 lambda 参数的影响，展示过小/过大引发的饱和或梯度弱化现象
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import argparse
from itertools import product
from tqdm import tqdm
import logging
from datetime import datetime

from model import create_model
try:
    # 提供从修正版脚本复用beta敏感性分析的能力
    from beta_sensitivity_corrected import BetaSensitivityAnalysis
except Exception:
    BetaSensitivityAnalysis = None
from data_loader import EnergyDPODataLoader
from train import EnergyDPOTrainer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParameterStudy:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # 参数网格定义
        self.beta_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        self.lambda_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

        # 存储结果
        self.results = {}
        self.training_curves = {}
        self.gradient_stats = {}

        # 设置输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.study_dir = os.path.join(args.output_dir, f"parameter_study_{timestamp}")
        os.makedirs(self.study_dir, exist_ok=True)

        logger.info(f"Parameter study output directory: {self.study_dir}")
        logger.info(f"Beta values: {self.beta_values}")
        logger.info(f"Lambda values: {self.lambda_values}")

    def prepare_data(self):
        """准备训练和测试数据"""
        logger.info("Preparing data...")

        # 使用小数据集进行快速实验
        data_args = argparse.Namespace()
        data_args.dataset = self.args.dataset
        data_args.drugood_subset = getattr(self.args, 'drugood_subset', 'lbap_general_ec50_scaffold')

        # 设置正确的数据文件路径
        if hasattr(self.args, 'data_file') and self.args.data_file:
            data_args.data_file = self.args.data_file
        else:
            # 使用与 run_experiments.sh 相同的路径格式
            dataset_name = data_args.drugood_subset if data_args.dataset == 'drugood' else data_args.dataset
            data_args.data_file = f"./data/raw/{dataset_name}.json"

        data_args.data_path = self.args.data_path
        data_args.batch_size = self.args.batch_size
        data_args.eval_batch_size = self.args.eval_batch_size

        self.data_loader = EnergyDPODataLoader(data_args)
        train_loader, valid_loader = self.data_loader.get_dataloaders()
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        logger.info(f"Training batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.valid_loader)}")

    def single_experiment(self, beta, lambda_reg, max_epochs=10):
        """运行单个参数组合的实验"""
        experiment_name = f"beta_{beta}_lambda_{lambda_reg}"
        logger.info(f"Running experiment: {experiment_name}")

        # 创建模型参数
        model_args = argparse.Namespace()
        model_args.hidden_dim = self.args.hidden_dim
        model_args.dpo_beta = beta
        model_args.lambda_reg = lambda_reg
        model_args.foundation_model = getattr(self.args, 'foundation_model', 'minimol')
        model_args.loss_type = 'dpo'

        # 创建模型
        model = create_model(model_args)
        model.to(self.device)

        # 设置优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=0.0)

        # 存储训练曲线和梯度统计
        training_history = {
            'epoch': [],
            'train_loss': [],
            'train_dpo_loss': [],
            'train_energy_sep': [],
            'grad_norm': [],
            'grad_max': [],
            'grad_std': [],
            'id_energy_mean': [],
            'ood_energy_mean': []
        }

        model.train()
        for epoch in range(max_epochs):
            epoch_losses = []
            epoch_dpo_losses = []
            epoch_energy_seps = []
            epoch_grad_norms = []
            epoch_grad_maxs = []
            epoch_grad_stds = []
            epoch_id_energies = []
            epoch_ood_energies = []

            for batch_idx, batch_data in enumerate(self.train_loader):
                if batch_idx >= 50:  # 限制每个epoch的batch数量以加速实验
                    break

                optimizer.zero_grad()

                # 前向传播
                loss, loss_dict = model(batch_data)

                # 反向传播
                loss.backward()

                # 计算梯度统计
                total_norm = 0
                max_grad = 0
                grad_values = []

                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        max_grad = max(max_grad, p.grad.data.abs().max().item())
                        grad_values.extend(p.grad.data.flatten().abs().cpu().numpy())

                total_norm = total_norm ** (1. / 2)
                grad_std = np.std(grad_values) if grad_values else 0

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # 记录统计信息
                epoch_losses.append(loss.item())
                epoch_dpo_losses.append(loss_dict['dpo_loss'])
                epoch_energy_seps.append(loss_dict['energy_separation'])
                epoch_grad_norms.append(total_norm)
                epoch_grad_maxs.append(max_grad)
                epoch_grad_stds.append(grad_std)
                epoch_id_energies.append(loss_dict['id_energy'])
                epoch_ood_energies.append(loss_dict['ood_energy'])

            # 记录epoch统计
            training_history['epoch'].append(epoch)
            training_history['train_loss'].append(np.mean(epoch_losses))
            training_history['train_dpo_loss'].append(np.mean(epoch_dpo_losses))
            training_history['train_energy_sep'].append(np.mean(epoch_energy_seps))
            training_history['grad_norm'].append(np.mean(epoch_grad_norms))
            training_history['grad_max'].append(np.mean(epoch_grad_maxs))
            training_history['grad_std'].append(np.mean(epoch_grad_stds))
            training_history['id_energy_mean'].append(np.mean(epoch_id_energies))
            training_history['ood_energy_mean'].append(np.mean(epoch_ood_energies))

        # 评估最终性能
        model.eval()
        eval_losses = []
        eval_energy_seps = []

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.valid_loader):
                if batch_idx >= 20:  # 限制评估batch数量
                    break

                loss, loss_dict = model(batch_data)
                eval_losses.append(loss.item())
                eval_energy_seps.append(loss_dict['energy_separation'])

        final_metrics = {
            'beta': beta,
            'lambda_reg': lambda_reg,
            'final_train_loss': training_history['train_loss'][-1],
            'final_eval_loss': np.mean(eval_losses),
            'final_energy_separation': np.mean(eval_energy_seps),
            'final_grad_norm': training_history['grad_norm'][-1],
            'max_grad_norm': max(training_history['grad_norm']),
            'min_grad_norm': min(training_history['grad_norm']),
            'convergence_rate': self._compute_convergence_rate(training_history['train_loss']),
            'stability_score': self._compute_stability_score(training_history['train_loss']),
            'gradient_explosion': max(training_history['grad_norm']) > 10.0,
            'gradient_vanishing': min(training_history['grad_norm']) < 1e-6
        }

        return final_metrics, training_history

    def _compute_convergence_rate(self, losses):
        """计算收敛速度"""
        if len(losses) < 3:
            return 0.0

        # 计算后半段的损失下降率
        mid_point = len(losses) // 2
        early_loss = np.mean(losses[:mid_point])
        late_loss = np.mean(losses[mid_point:])

        if early_loss == 0:
            return 0.0

        return (early_loss - late_loss) / early_loss

    def _compute_stability_score(self, losses):
        """计算训练稳定性分数"""
        if len(losses) < 2:
            return 1.0

        # 计算损失的变异系数（标准差/均值）
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)

        if mean_loss == 0:
            return 0.0

        cv = std_loss / mean_loss
        return 1.0 / (1.0 + cv)  # 转换为稳定性分数

    def run_grid_search(self):
        """运行网格搜索"""
        logger.info("Starting grid search...")

        total_experiments = len(self.beta_values) * len(self.lambda_values)

        with tqdm(total=total_experiments, desc="Parameter Study") as pbar:
            for beta, lambda_reg in product(self.beta_values, self.lambda_values):
                try:
                    experiment_key = f"beta_{beta}_lambda_{lambda_reg}"

                    # 运行实验
                    metrics, history = self.single_experiment(beta, lambda_reg, max_epochs=15)

                    # 存储结果
                    self.results[experiment_key] = metrics
                    self.training_curves[experiment_key] = history

                    pbar.set_postfix({
                        'beta': beta,
                        'lambda': lambda_reg,
                        'loss': f"{metrics['final_eval_loss']:.4f}",
                        'grad': f"{metrics['final_grad_norm']:.2e}"
                    })

                except Exception as e:
                    logger.warning(f"Experiment {experiment_key} failed: {e}")
                    # 记录失败的实验
                    self.results[experiment_key] = {
                        'beta': beta,
                        'lambda_reg': lambda_reg,
                        'failed': True,
                        'error': str(e)
                    }

                pbar.update(1)

        logger.info("Grid search completed!")

    def analyze_and_visualize(self):
        """分析结果并生成可视化"""
        logger.info("Analyzing results and generating visualizations...")

        # 准备数据
        successful_results = [r for r in self.results.values() if not r.get('failed', False)]

        if not successful_results:
            logger.error("No successful experiments found!")
            return

        df = pd.DataFrame(successful_results)

        # 1. 热力图: 最终损失
        self._plot_heatmap(df, 'final_eval_loss', 'Final Evaluation Loss', 'Blues_r')

        # 2. 热力图: 能量分离度
        self._plot_heatmap(df, 'final_energy_separation', 'Energy Separation', 'RdYlBu')

        # 3. 热力图: 梯度范数
        self._plot_heatmap(df, 'final_grad_norm', 'Final Gradient Norm', 'viridis')

        # 4. 热力图: 收敛速度
        self._plot_heatmap(df, 'convergence_rate', 'Convergence Rate', 'Greens')

        # 5. 热力图: 训练稳定性
        self._plot_heatmap(df, 'stability_score', 'Training Stability', 'plasma')

        # 6. 训练曲线比较
        self._plot_training_curves()

        # 7. 梯度行为分析
        self._plot_gradient_analysis(df)

        # 8. 参数敏感性分析
        self._plot_parameter_sensitivity(df)

        # 保存结果
        self._save_results(df)

        logger.info(f"Analysis complete! Results saved in {self.study_dir}")

    def _plot_heatmap(self, df, metric, title, cmap):
        """绘制参数热力图"""
        pivot_table = df.pivot(index='lambda_reg', columns='beta', values=metric)

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap=cmap,
                   cbar_kws={'label': title})
        plt.title(f'{title} vs Beta and Lambda')
        plt.xlabel('Beta (DPO Strength)')
        plt.ylabel('Lambda (L2 Regularization)')
        plt.tight_layout()

        filename = f"{title.lower().replace(' ', '_')}_heatmap.png"
        plt.savefig(os.path.join(self.study_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_training_curves(self):
        """绘制关键参数组合的训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 选择几个有代表性的参数组合
        key_combinations = [
            ('beta_0.01_lambda_0.01', 'Small Beta (Weak Signal)'),
            ('beta_0.1_lambda_0.01', 'Medium Beta (Baseline)'),
            ('beta_2.0_lambda_0.01', 'Large Beta (Saturation Risk)'),
            ('beta_0.1_lambda_0.2', 'Strong Regularization')
        ]

        for idx, (key, label) in enumerate(key_combinations):
            if key not in self.training_curves:
                continue

            history = self.training_curves[key]
            ax = axes[idx // 2, idx % 2]

            # 绘制损失和梯度范数
            ax2 = ax.twinx()

            line1 = ax.plot(history['epoch'], history['train_loss'], 'b-', label='Training Loss')
            line2 = ax2.plot(history['epoch'], history['grad_norm'], 'r--', label='Gradient Norm')

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Training Loss', color='b')
            ax2.set_ylabel('Gradient Norm', color='r')
            ax.set_title(label)

            # 组合图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper right')

            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.study_dir, 'training_curves_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_gradient_analysis(self, df):
        """梯度行为分析"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. Beta vs 梯度范数
        axes[0].scatter(df['beta'], df['final_grad_norm'],
                       c=df['lambda_reg'], cmap='viridis', s=60, alpha=0.7)
        axes[0].set_xlabel('Beta')
        axes[0].set_ylabel('Final Gradient Norm')
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_title('Beta vs Gradient Norm')
        axes[0].grid(True, alpha=0.3)

        # 2. Lambda vs 梯度范数
        axes[1].scatter(df['lambda_reg'], df['final_grad_norm'],
                       c=df['beta'], cmap='plasma', s=60, alpha=0.7)
        axes[1].set_xlabel('Lambda (L2 Regularization)')
        axes[1].set_ylabel('Final Gradient Norm')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_title('Lambda vs Gradient Norm')
        axes[1].grid(True, alpha=0.3)

        # 3. 梯度问题检测
        problem_detection = df.copy()
        problem_detection['gradient_status'] = 'Normal'
        problem_detection.loc[problem_detection['gradient_explosion'], 'gradient_status'] = 'Explosion'
        problem_detection.loc[problem_detection['gradient_vanishing'], 'gradient_status'] = 'Vanishing'

        status_counts = problem_detection['gradient_status'].value_counts()
        axes[2].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
        axes[2].set_title('Gradient Behavior Distribution')

        plt.tight_layout()
        plt.savefig(os.path.join(self.study_dir, 'gradient_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_parameter_sensitivity(self, df):
        """参数敏感性分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Beta 敏感性
        beta_sensitivity = df.groupby('beta').agg({
            'final_eval_loss': ['mean', 'std'],
            'final_energy_separation': ['mean', 'std']
        })

        axes[0, 0].errorbar(beta_sensitivity.index,
                           beta_sensitivity[('final_eval_loss', 'mean')],
                           yerr=beta_sensitivity[('final_eval_loss', 'std')],
                           marker='o', capsize=5)
        axes[0, 0].set_xlabel('Beta')
        axes[0, 0].set_ylabel('Evaluation Loss')
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_title('Beta Sensitivity - Loss')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].errorbar(beta_sensitivity.index,
                           beta_sensitivity[('final_energy_separation', 'mean')],
                           yerr=beta_sensitivity[('final_energy_separation', 'std')],
                           marker='s', capsize=5, color='orange')
        axes[0, 1].set_xlabel('Beta')
        axes[0, 1].set_ylabel('Energy Separation')
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_title('Beta Sensitivity - Energy Separation')
        axes[0, 1].grid(True, alpha=0.3)

        # Lambda 敏感性
        lambda_sensitivity = df.groupby('lambda_reg').agg({
            'final_eval_loss': ['mean', 'std'],
            'stability_score': ['mean', 'std']
        })

        axes[1, 0].errorbar(lambda_sensitivity.index,
                           lambda_sensitivity[('final_eval_loss', 'mean')],
                           yerr=lambda_sensitivity[('final_eval_loss', 'std')],
                           marker='^', capsize=5, color='green')
        axes[1, 0].set_xlabel('Lambda (L2 Regularization)')
        axes[1, 0].set_ylabel('Evaluation Loss')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_title('Lambda Sensitivity - Loss')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].errorbar(lambda_sensitivity.index,
                           lambda_sensitivity[('stability_score', 'mean')],
                           yerr=lambda_sensitivity[('stability_score', 'std')],
                           marker='D', capsize=5, color='red')
        axes[1, 1].set_xlabel('Lambda (L2 Regularization)')
        axes[1, 1].set_ylabel('Training Stability Score')
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_title('Lambda Sensitivity - Stability')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.study_dir, 'parameter_sensitivity.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _save_results(self, df):
        """保存结果到文件"""
        # 保存CSV格式的结果
        df.to_csv(os.path.join(self.study_dir, 'parameter_study_results.csv'), index=False)

        # 保存JSON格式的完整结果
        with open(os.path.join(self.study_dir, 'complete_results.json'), 'w') as f:
            json.dump({
                'results': self.results,
                'training_curves': self.training_curves,
                'study_config': {
                    'beta_values': self.beta_values,
                    'lambda_values': self.lambda_values,
                    'dataset': self.args.dataset,
                    'study_timestamp': datetime.now().isoformat()
                }
            }, f, indent=2)

        # 生成总结报告
        self._generate_summary_report(df)

    def _generate_summary_report(self, df):
        """生成总结报告"""
        best_overall = df.loc[df['final_eval_loss'].idxmin()]
        best_stability = df.loc[df['stability_score'].idxmax()]

        report = f"""
# Energy-DPO Parameter Study Report

## Study Configuration
- Dataset: {self.args.dataset}
- Beta values tested: {self.beta_values}
- Lambda values tested: {self.lambda_values}
- Total experiments: {len(df)}

## Key Findings

### Best Overall Performance
- Beta: {best_overall['beta']}
- Lambda: {best_overall['lambda_reg']}
- Final Loss: {best_overall['final_eval_loss']:.4f}
- Energy Separation: {best_overall['final_energy_separation']:.4f}

### Most Stable Training
- Beta: {best_stability['beta']}
- Lambda: {best_stability['lambda_reg']}
- Stability Score: {best_stability['stability_score']:.4f}
- Final Loss: {best_stability['final_eval_loss']:.4f}

### Gradient Behavior Analysis
- Experiments with gradient explosion: {df['gradient_explosion'].sum()}
- Experiments with gradient vanishing: {df['gradient_vanishing'].sum()}
- Normal gradient behavior: {(~df['gradient_explosion'] & ~df['gradient_vanishing']).sum()}

### Parameter Sensitivity
- Beta range with good performance: {df[df['final_eval_loss'] <= df['final_eval_loss'].quantile(0.25)]['beta'].min():.3f} - {df[df['final_eval_loss'] <= df['final_eval_loss'].quantile(0.25)]['beta'].max():.3f}
- Lambda range with good stability: {df[df['stability_score'] >= df['stability_score'].quantile(0.75)]['lambda_reg'].min():.3f} - {df[df['stability_score'] >= df['stability_score'].quantile(0.75)]['lambda_reg'].max():.3f}

## Recommendations

1. **For optimal performance**: Use beta={best_overall['beta']}, lambda={best_overall['lambda_reg']}
2. **For stable training**: Use beta={best_stability['beta']}, lambda={best_stability['lambda_reg']}
3. **Avoid**: Very small beta (<0.05) may cause weak gradients, very large beta (>2.0) may cause saturation
4. **Regularization**: Lambda in range [0.01, 0.1] provides good balance between performance and stability

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

        with open(os.path.join(self.study_dir, 'summary_report.md'), 'w') as f:
            f.write(report)

def parse_args():
    parser = argparse.ArgumentParser(description='Energy-DPO Parameter Study')

    # 数据相关
    parser.add_argument('--dataset', type=str, default='drugood', help='Dataset name')
    parser.add_argument('--drugood_subset', type=str, default='lbap_general_ec50_scaffold')
    parser.add_argument('--data_path', type=str, default='./data', help='Data directory')
    parser.add_argument('--data_file', type=str, help='Specific data file path (overrides auto-detection)')

    # 模型相关
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--foundation_model', type=str, default='minimol', help='Foundation model')

    # 训练相关
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Evaluation batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    # （仅用于 --beta_sensitivity 模式）
    parser.add_argument('--epochs', type=int, default=10, help='Beta sensitivity: epochs per beta')
    parser.add_argument('--max_train_batches', type=int, default=30, help='Beta sensitivity: max train batches per epoch (None for all)')
    parser.add_argument('--max_valid_batches', type=int, default=10, help='Beta sensitivity: max valid batches per eval (None for all)')

    # 其他
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output_dir', type=str, default='./parameter_study_results',
                       help='Output directory')
    parser.add_argument('--quick_mode', action='store_true',
                       help='Run with reduced parameter ranges for quick testing')
    parser.add_argument('--beta_sensitivity', action='store_true',
                       help='Run beta-only sensitivity analysis with fixed lambda=0.01 on EC50 splits')

    return parser.parse_args()

def main():
    args = parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 可选：进入beta敏感性分析模式（固定lambda）
    if args.beta_sensitivity:
        if BetaSensitivityAnalysis is None:
            logger.error('BetaSensitivityAnalysis not available. Please run beta_sensitivity_corrected.py directly.')
            return
        # 直接重用修正版分析，确保使用真实train/test数据与TEST AUC
        bs_args = argparse.Namespace(
            data_path=args.data_path,
            hidden_dim=args.hidden_dim,
            foundation_model=args.foundation_model,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            lr=args.lr,
            device=args.device,
            output_dir=os.path.join(args.output_dir, 'beta_sensitivity'),
            epochs=args.epochs,
            max_train_batches=args.max_train_batches,
            max_valid_batches=args.max_valid_batches
        )
        analysis = BetaSensitivityAnalysis(bs_args)
        analysis.run_beta_sensitivity()
        analysis.create_corrected_plots()
        analysis.save_results()
        logger.info('Beta-only sensitivity analysis completed (via corrected TEST pipeline).')
        return

    # 默认：运行原有参数网格研究
    study = ParameterStudy(args)
    if args.quick_mode:
        study.beta_values = [0.05, 0.1, 0.5, 1.0]
        study.lambda_values = [0.01, 0.05, 0.1]
        logger.info("Quick mode: Using reduced parameter ranges")
    study.prepare_data()
    study.run_grid_search()
    study.analyze_and_visualize()
    logger.info("Parameter study completed successfully!")

if __name__ == '__main__':
    main()
