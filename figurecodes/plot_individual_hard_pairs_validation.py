#!/usr/bin/env python3
"""
生成单独的EC50难对验证图表 - 每个分析一张图
同时输出PNG和SVG格式
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import torch
import torch.nn.functional as F
from scipy.special import expit as sigmoid

# 导入项目模块
sys.path.append('/home/ubuntu/OOD-DPO')
from model import EnergyDPOModel
from data_loader import EnergyDPODataLoader

# 设置专业绘图风格
plt.style.use('default')
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['axes.linewidth'] = 1.2
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

# 数据集颜色配置（参考beta plots）
DATASET_COLORS = {
    'lbap_general_ec50_assay': '#2E86AB',      # 明亮的蓝色
    'lbap_general_ec50_scaffold': '#F24236',   # 明亮的红色
    'lbap_general_ec50_size': '#2E8B57'        # 绿色
}

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_device():
    """设置计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    return device

class IndividualHardPairsValidator:
    """生成单独图表的EC50难对验证分析器"""

    def __init__(self, device):
        self.device = device
        self.base_model_path = '/home/ubuntu/OOD-DPO/outputs/minimol'
        self.data_path = './data/raw'

    def load_model_and_data(self, dataset_name, seed=1):
        """加载模型和数据"""
        # 加载模型
        model_path = f"{self.base_model_path}/{dataset_name}/{seed}/best_model.pth"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading model from {model_path}")

        # 创建args对象以初始化模型
        class Args:
            def __init__(self):
                self.foundation_model = 'minimol'
                self.dpo_beta = 0.1
                self.hidden_dim = 256

        args = Args()
        model = EnergyDPOModel(args)

        # 加载模型状态
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        # 提取实际的beta值
        if hasattr(model, 'beta'):
            if torch.is_tensor(model.beta):
                actual_beta = float(model.beta.cpu().detach().numpy())
            else:
                actual_beta = float(model.beta)
        else:
            actual_beta = 0.1
        logger.info(f"Extracted actual beta value: {actual_beta}")

        # 加载数据
        class DataArgs:
            def __init__(self, data_path):
                self.dataset = dataset_name
                self.foundation_model = 'minimol'
                self.data_path = data_path
                self.data_seed = 42

        data_args = DataArgs(self.data_path)
        data_loader = EnergyDPODataLoader(data_args)
        test_id, test_ood = data_loader.get_final_test_data()

        logger.info(f"Loaded test data: {len(test_id)} ID samples, {len(test_ood)} OOD samples")

        return model, actual_beta, test_id, test_ood

    def compute_analysis_data(self, model, actual_beta, test_id, test_ood, max_samples=500):
        """计算分析所需的数据"""
        if max_samples:
            n_samples = min(max_samples, len(test_id), len(test_ood))
            test_id = test_id[:n_samples]
            test_ood = test_ood[:n_samples]

        # 计算能量
        batch_size = 100
        all_energy_id = []
        all_energy_ood = []

        with torch.no_grad():
            # 计算ID能量
            for i in tqdm(range(0, len(test_id), batch_size), desc="Computing ID energies"):
                batch_id = test_id[i:i+batch_size]
                if isinstance(batch_id[0], dict):
                    features_id = torch.stack([sample['features'] for sample in batch_id]).to(self.device)
                else:
                    features_id = torch.stack(batch_id).to(self.device)
                energy_id = model.forward_energy(features_id).cpu().numpy()
                all_energy_id.extend(energy_id)

            # 计算OOD能量
            for i in tqdm(range(0, len(test_ood), batch_size), desc="Computing OOD energies"):
                batch_ood = test_ood[i:i+batch_size]
                if isinstance(batch_ood[0], dict):
                    features_ood = torch.stack([sample['features'] for sample in batch_ood]).to(self.device)
                else:
                    features_ood = torch.stack(batch_ood).to(self.device)
                energy_ood = model.forward_energy(features_ood).cpu().numpy()
                all_energy_ood.extend(energy_ood)

        all_energy_id = np.array(all_energy_id)
        all_energy_ood = np.array(all_energy_ood)

        # 生成pairs
        max_pairs = min(50000, len(all_energy_id) * len(all_energy_ood))
        id_indices = np.random.choice(len(all_energy_id), size=max_pairs, replace=True)
        ood_indices = np.random.choice(len(all_energy_ood), size=max_pairs, replace=True)

        energy_id_pairs = all_energy_id[id_indices]
        energy_ood_pairs = all_energy_ood[ood_indices]

        # 计算能量差和梯度权重
        delta_values = energy_ood_pairs - energy_id_pairs
        weights = actual_beta * sigmoid(-actual_beta * delta_values)

        # 创建分箱分析
        n_bins = 20
        valid_mask = np.isfinite(delta_values) & np.isfinite(weights)
        delta_clean = delta_values[valid_mask]
        weights_clean = weights[valid_mask]

        bin_edges = np.linspace(np.percentile(delta_clean, 1),
                               np.percentile(delta_clean, 99), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        mean_weights = np.zeros(n_bins)
        std_weights = np.zeros(n_bins)
        counts = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (delta_clean >= bin_edges[i]) & (delta_clean < bin_edges[i + 1])
            if mask.sum() > 0:
                mean_weights[i] = weights_clean[mask].mean()
                std_weights[i] = weights_clean[mask].std()
                counts[i] = mask.sum()
            else:
                mean_weights[i] = np.nan
                std_weights[i] = np.nan
                counts[i] = 0

        binned_data = {
            'bin_centers': bin_centers,
            'mean_weights': mean_weights,
            'std_weights': std_weights,
            'counts': counts
        }

        return {
            'delta_values': delta_values,
            'weights': weights,
            'binned_data': binned_data,
            'actual_beta': actual_beta
        }

def save_both_formats(fig, filepath_base):
    """保存PNG和SVG格式"""
    plt.savefig(f"{filepath_base}.png", format='png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.savefig(f"{filepath_base}.svg", format='svg', bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"✅ Saved both formats: {filepath_base}.png/.svg")

def create_individual_plots(dataset_name, analysis_data, output_dir):
    """创建4个单独的图表"""
    os.makedirs(output_dir, exist_ok=True)

    primary_color = DATASET_COLORS.get(dataset_name, '#F24236')
    dataset_display_name = dataset_name.replace('lbap_general_ec50_', '').title()

    delta_values = analysis_data['delta_values']
    weights = analysis_data['weights']
    binned_data = analysis_data['binned_data']
    actual_beta = analysis_data['actual_beta']

    hard_pairs_mask = delta_values < 0
    easy_pairs_mask = delta_values > 0
    boundary_mask = np.abs(delta_values) < 0.05

    # 图1: 核心验证图 - 经验vs理论曲线
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # 采样数据点
    sample_idx = np.random.choice(len(delta_values), size=min(3000, len(delta_values)), replace=False)
    ax.scatter(delta_values[sample_idx], weights[sample_idx],
               alpha=0.3, s=2, color='lightgray', label='Individual pairs', zorder=1)

    # 经验曲线
    valid_bins = ~np.isnan(binned_data['mean_weights'])
    ax.errorbar(binned_data['bin_centers'][valid_bins],
                binned_data['mean_weights'][valid_bins],
                yerr=binned_data['std_weights'][valid_bins] / np.sqrt(binned_data['counts'][valid_bins]),
                fmt='o-', color=primary_color, markersize=8, linewidth=3, capsize=5,
                label='Empirical curve', zorder=3)

    # 理论曲线
    t_theory = np.linspace(delta_values.min(), delta_values.max(), 1000)
    w_theory = actual_beta * sigmoid(-actual_beta * t_theory)
    ax.plot(t_theory, w_theory, '--', color='black', linewidth=3,
            label=f'Theory: β·σ(-βt), β={actual_beta:.3f}', zorder=2)

    ax.axvline(x=0, color='gray', linestyle=':', linewidth=2, alpha=0.7,
               label='Decision boundary')
    ax.set_xlabel('Energy Difference ΔE = E_ood - E_id', fontsize=16)
    ax.set_ylabel('Gradient Weight w_β(ΔE)', fontsize=16)
    ax.set_title(f'Core Validation: Empirical vs Theoretical\n{dataset_display_name} Dataset',
                fontsize=18, fontweight='bold')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    save_both_formats(fig, f"{output_dir}/figure_a3_1_core_validation_{dataset_name}")

    # 图2: 权重对比柱状图
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    categories = ['Hard Pairs\n(ΔE<0)', 'Easy Pairs\n(ΔE>0)', 'Boundary\n(|ΔE|<0.05)']
    mean_weights = [
        weights[hard_pairs_mask].mean() if hard_pairs_mask.any() else 0,
        weights[easy_pairs_mask].mean() if easy_pairs_mask.any() else 0,
        weights[boundary_mask].mean() if boundary_mask.any() else 0
    ]
    colors = [primary_color, 'lightgray', 'orange']

    bars = ax.bar(categories, mean_weights, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.5)

    for bar, weight in zip(bars, mean_weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                f'{weight:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=14)

    ax.set_ylabel('Average Gradient Weight', fontsize=16)
    advantage = ((mean_weights[0]/mean_weights[1]-1)*100) if mean_weights[1] > 0 else 0
    ax.set_title(f'Weight Comparison - {dataset_display_name}\nHard pairs get {advantage:.1f}% higher weights',
                fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    save_both_formats(fig, f"{output_dir}/figure_a3_2_weight_comparison_{dataset_name}")

    # 图3: 权重分布直方图
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    ax.hist(weights, bins=50, alpha=0.7, density=True, color=primary_color,
            edgecolor='black', linewidth=0.5, label='All pairs')

    # 添加均值线
    ax.axvline(x=weights.mean(), color='black', linestyle='--', linewidth=2,
               label=f'Overall mean: {weights.mean():.4f}')

    if hard_pairs_mask.any():
        ax.axvline(x=weights[hard_pairs_mask].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Hard pairs: {weights[hard_pairs_mask].mean():.4f}')

    if easy_pairs_mask.any():
        ax.axvline(x=weights[easy_pairs_mask].mean(), color='green', linestyle='--', linewidth=2,
                   label=f'Easy pairs: {weights[easy_pairs_mask].mean():.4f}')

    ax.set_xlabel('Gradient Weight w_β(ΔE)', fontsize=16)
    ax.set_ylabel('Probability Density', fontsize=16)
    ax.set_title(f'Weight Distribution Analysis - {dataset_display_name}',
                fontsize=18, fontweight='bold')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    save_both_formats(fig, f"{output_dir}/figure_a3_3_weight_distribution_{dataset_name}")

    # 图4: 能量差分布图
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    ax.hist(delta_values, bins=50, alpha=0.6, color='lightsteelblue',
            edgecolor='black', linewidth=0.5, label='Energy differences')

    ax.axvline(x=0, color='red', linestyle='--', linewidth=3, label='Decision boundary')
    ax.axvline(x=delta_values.mean(), color='blue', linestyle='--', linewidth=2,
               label=f'Mean ΔE: {delta_values.mean():.2f}')

    # 填充区域
    ylim = ax.get_ylim()
    if hard_pairs_mask.any():
        ax.fill_between([delta_values.min(), 0], 0, ylim[1], alpha=0.2, color='red',
                       label=f'Hard pairs ({hard_pairs_mask.mean():.1%})')

    if easy_pairs_mask.any():
        ax.fill_between([0, delta_values.max()], 0, ylim[1], alpha=0.2, color='green',
                       label=f'Easy pairs ({easy_pairs_mask.mean():.1%})')

    ax.set_xlabel('Energy Difference ΔE = E_ood - E_id', fontsize=16)
    ax.set_ylabel('Count', fontsize=16)
    ax.set_title(f'Energy Difference Distribution - {dataset_display_name}',
                fontsize=18, fontweight='bold')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    save_both_formats(fig, f"{output_dir}/figure_a3_4_energy_distribution_{dataset_name}")

    logger.info(f"✅ All 4 individual plots created for {dataset_name}")

def main():
    parser = argparse.ArgumentParser(description='生成单独的EC50难对验证图表')
    parser.add_argument('--datasets', nargs='+',
                       default=['lbap_general_ec50_scaffold', 'lbap_general_ec50_size', 'lbap_general_ec50_assay'],
                       help='要分析的数据集')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1],
                       help='要分析的随机种子')
    parser.add_argument('--max_samples', type=int, default=500,
                       help='每个数据集最大样本数')
    parser.add_argument('--output_dir', type=str, default='individual_hard_pairs_plots',
                       help='输出目录')

    args = parser.parse_args()

    # 设置设备
    device = setup_device()

    # 初始化分析器
    validator = IndividualHardPairsValidator(device)

    for dataset_name in args.datasets:
        for seed in args.seeds:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing {dataset_name} with seed {seed}")
                logger.info(f"{'='*60}")

                # 加载模型和数据
                model, actual_beta, test_id, test_ood = validator.load_model_and_data(dataset_name, seed)

                # 计算分析数据
                analysis_data = validator.compute_analysis_data(model, actual_beta, test_id, test_ood, args.max_samples)

                # 创建单独图表
                create_individual_plots(dataset_name, analysis_data, args.output_dir)

                print(f"\n✅ {dataset_name} 分析完成！生成了4个单独图表（PNG + SVG）")

            except Exception as e:
                logger.error(f"Failed to process {dataset_name} seed {seed}: {e}")
                continue

    logger.info(f"\n🎉 所有分析完成！图表保存至: {args.output_dir}")

if __name__ == '__main__':
    main()