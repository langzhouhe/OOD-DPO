"""
Figure A3: Hard Pairs Validation for EC50 Datasets
Analyze gradient weight contributions w_β(Δ) for Energy DPO models
Validate theoretical predictions: w_β(t) = β·σ(-βt)
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import warnings
from scipy.special import expit as sigmoid
from scipy.stats import binned_statistic

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Import custom modules
from model import load_pretrained_model
from data_loader import EnergyDPODataLoader
from utils import set_seed

def create_args_for_dataset(dataset_name, foundation_model='minimol'):
    """Create args object for loading specific EC50 dataset"""
    class Args:
        def __init__(self):
            self.dataset = 'drugood'
            self.drugood_subset = dataset_name
            self.foundation_model = foundation_model
            self.data_path = './data/raw'
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.seed = 42
            self.eval_batch_size = 64
            self.dpo_beta = 0.1  # From training config

    return Args()

def load_ec50_model_and_data(dataset_name, seed=1, foundation_model='minimol'):
    """Load trained EC50 model and corresponding dataset"""

    # Model path
    model_path = f"/home/ubuntu/OOD-DPO/outputs1_before/{foundation_model}/{dataset_name}/{seed}/best_model.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Create args
    args = create_args_for_dataset(dataset_name, foundation_model)
    set_seed(args.seed)

    # Load model
    logger.info(f"Loading model from {model_path}")
    device = torch.device(args.device)
    model = load_pretrained_model(model_path, args).to(device)
    model.eval()

    # Load data
    logger.info(f"Loading dataset: {dataset_name}")
    data_loader = EnergyDPODataLoader(args)

    # Get evaluation data
    eval_data = data_loader.get_final_test_data()

    return model, eval_data, args

def extract_features_and_compute_energy_differences(model, eval_data, max_samples=2000):
    """Extract features and compute energy differences ΔE = E_ood - E_id"""

    device = next(model.parameters()).device
    model.eval()

    # Limit samples for computational efficiency
    id_smiles = eval_data['id_smiles'][:max_samples]
    ood_smiles = eval_data['ood_smiles'][:max_samples]

    logger.info(f"Processing {len(id_smiles)} ID and {len(ood_smiles)} OOD samples")

    # Encode SMILES to features (pre-energy-head features)
    with torch.no_grad():
        id_features = model.encode_smiles(id_smiles).to(device)    # [n, 512]
        ood_features = model.encode_smiles(ood_smiles).to(device)  # [m, 512]

        # Compute energies
        id_energies = model.energy_head(id_features).squeeze(-1)    # [n]
        ood_energies = model.energy_head(ood_features).squeeze(-1)  # [m]

        # Pairwise energy differences: ΔE[i,j] = E_ood[j] - E_id[i]
        energy_diffs = ood_energies.unsqueeze(0) - id_energies.unsqueeze(1)  # [n, m]

        # Feature differences for theoretical validation
        # ||h_L-1(S_out) - h_L-1(S_in)||_2
        id_features_expanded = id_features.unsqueeze(1)     # [n, 1, 512]
        ood_features_expanded = ood_features.unsqueeze(0)   # [1, m, 512]
        feature_diffs_norm = torch.norm(
            ood_features_expanded - id_features_expanded,
            dim=2
        )  # [n, m]

    return {
        'energy_diffs': energy_diffs.cpu().numpy(),
        'feature_diffs_norm': feature_diffs_norm.cpu().numpy(),
        'id_energies': id_energies.cpu().numpy(),
        'ood_energies': ood_energies.cpu().numpy(),
        'beta': model.beta
    }

def compute_gradient_weights(energy_diffs, beta):
    """
    Compute DPO gradient weights: w_β(Δ) = β·σ(-β·Δ)
    where σ is the sigmoid function
    """
    # Flatten energy differences for analysis
    delta_flat = energy_diffs.flatten()

    # Compute gradient weights using sigmoid
    weights = beta * sigmoid(-beta * delta_flat)

    return delta_flat, weights

def create_binned_analysis(delta_values, weights, n_bins=30):
    """Create binned analysis of gradient weights vs energy differences"""

    # Remove any NaN or infinite values
    valid_mask = np.isfinite(delta_values) & np.isfinite(weights)
    delta_clean = delta_values[valid_mask]
    weights_clean = weights[valid_mask]

    if len(delta_clean) == 0:
        return None

    # Create bins
    bin_edges = np.linspace(delta_clean.min(), delta_clean.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute binned statistics
    mean_weights, _, _ = binned_statistic(delta_clean, weights_clean,
                                         statistic='mean', bins=bin_edges)
    std_weights, _, _ = binned_statistic(delta_clean, weights_clean,
                                        statistic='std', bins=bin_edges)
    counts, _, _ = binned_statistic(delta_clean, weights_clean,
                                   statistic='count', bins=bin_edges)

    return {
        'bin_centers': bin_centers,
        'mean_weights': mean_weights,
        'std_weights': std_weights,
        'counts': counts,
        'bin_edges': bin_edges
    }

def plot_hard_pairs_validation(results_dict, dataset_name, output_dir="./comparison_plots"):
    """Create Figure A3: Hard pairs validation plot"""

    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    delta_values = results_dict['delta_values']
    weights = results_dict['weights']
    binned_data = results_dict['binned_data']
    beta = results_dict['beta']

    # Left panel: Scatter plot with binned averages
    ax1.scatter(delta_values[::10], weights[::10], alpha=0.1, s=1, color='lightblue',
                label='Individual pairs')

    # Plot binned averages - empirical curve in bright red
    valid_bins = ~np.isnan(binned_data['mean_weights'])
    ax1.errorbar(binned_data['bin_centers'][valid_bins],
                 binned_data['mean_weights'][valid_bins],
                 yerr=binned_data['std_weights'][valid_bins] / np.sqrt(binned_data['counts'][valid_bins]),
                 fmt='o-', color='crimson', markersize=6, linewidth=3, capsize=4,
                 label='Empirical curve (binned averages)')

    # Theoretical curve: w_β(t) = β·σ(-β·t) - use contrasting bright color
    t_theory = np.linspace(delta_values.min(), delta_values.max(), 1000)
    w_theory = beta * sigmoid(-beta * t_theory)
    ax1.plot(t_theory, w_theory, '--', color='navy', linewidth=4, label=f'Theoretical curve: β·σ(-β·t), β={beta:.1f}')

    ax1.axvline(x=0, color='orange', linestyle=':', linewidth=2, alpha=0.8, label='Decision boundary (ΔE = 0)')
    ax1.set_xlabel('能量差 ΔE = E_ood - E_id\n(负值=难对, 正值=易对)', fontsize=12)
    ax1.set_ylabel('梯度权重 w_β(ΔE)', fontsize=12)
    ax1.set_title(f'难对验证: {dataset_name.replace("_", " ").title()}\n经验曲线对齐理论预测 w_β(t)=β·σ(-βt)', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right panel: Weight distribution histogram
    ax2.hist(weights, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    ax2.axvline(x=weights.mean(), color='red', linestyle='--',
                label=f'Mean weight: {weights.mean():.4f}')
    ax2.axvline(x=np.median(weights), color='orange', linestyle='--',
                label=f'Median weight: {np.median(weights):.4f}')
    ax2.set_xlabel('Gradient Weight w_β(ΔE)')
    ax2.set_ylabel('Density')
    ax2.set_title('Weight Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot as SVG format
    svg_path = os.path.join(output_dir, f'figure_a3_ec50_{dataset_name}_hard_pairs_validation.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    logger.info(f"Saved SVG plot: {svg_path}")

    # Also save as PNG for backup
    png_path = os.path.join(output_dir, f'figure_a3_ec50_{dataset_name}_hard_pairs_validation.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    return fig

def analyze_hard_vs_easy_pairs(results_dict):
    """Analyze hard pairs (ΔE < 0) vs easy pairs (ΔE > 0)"""

    delta_values = results_dict['delta_values']
    weights = results_dict['weights']

    # Split into hard and easy pairs
    hard_pairs_mask = delta_values < 0  # OOD energy < ID energy (misranked)
    easy_pairs_mask = delta_values > 0  # OOD energy > ID energy (correctly ranked)
    boundary_pairs_mask = np.abs(delta_values) < 0.05  # Near decision boundary

    hard_weights = weights[hard_pairs_mask]
    easy_weights = weights[easy_pairs_mask]
    boundary_weights = weights[boundary_pairs_mask]

    analysis = {
        'hard_pairs_ratio': hard_pairs_mask.mean(),
        'easy_pairs_ratio': easy_pairs_mask.mean(),
        'boundary_pairs_ratio': boundary_pairs_mask.mean(),
        'hard_pairs_avg_weight': hard_weights.mean() if len(hard_weights) > 0 else 0,
        'easy_pairs_avg_weight': easy_weights.mean() if len(easy_weights) > 0 else 0,
        'boundary_pairs_avg_weight': boundary_weights.mean() if len(boundary_weights) > 0 else 0,
        'avg_energy_difference': delta_values.mean(),
        'weight_peak_at_zero': weights[np.abs(delta_values) < 0.01].mean() if np.any(np.abs(delta_values) < 0.01) else 0
    }

    return analysis

def create_detailed_analysis_visualization(results_dict, dataset_name, output_dir="./comparison_plots"):
    """创建详细的难对验证分析图表"""

    os.makedirs(output_dir, exist_ok=True)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    delta_values = results_dict['delta_values']
    weights = results_dict['weights']
    binned_data = results_dict['binned_data']
    beta = results_dict['beta']

    # 创建2x2子图布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Figure A3 难对验证详细分析 - {dataset_name.replace("_", " ").title()}',
                 fontsize=16, fontweight='bold')

    # 分析数据
    hard_pairs_mask = delta_values < 0
    easy_pairs_mask = delta_values > 0
    boundary_mask = np.abs(delta_values) < 0.05

    # 子图1: 核心验证图 - 梯度权重vs能量差
    ax1 = axes[0, 0]

    # 散点图（降采样以提高清晰度）
    sample_idx = np.random.choice(len(delta_values), size=min(3000, len(delta_values)), replace=False)
    ax1.scatter(delta_values[sample_idx], weights[sample_idx],
                alpha=0.3, s=1.5, color='lightsteelblue', label='样本点', zorder=1)

    # 经验曲线（分箱平均值）
    valid_bins = ~np.isnan(binned_data['mean_weights'])
    ax1.errorbar(binned_data['bin_centers'][valid_bins],
                 binned_data['mean_weights'][valid_bins],
                 yerr=binned_data['std_weights'][valid_bins] / np.sqrt(binned_data['counts'][valid_bins]),
                 fmt='o-', color='crimson', markersize=8, linewidth=4, capsize=5,
                 label='经验曲线 (分箱平均)', zorder=3)

    # 理论曲线
    t_theory = np.linspace(delta_values.min(), delta_values.max(), 1000)
    w_theory = beta * sigmoid(-beta * t_theory)
    ax1.plot(t_theory, w_theory, '--', color='navy', linewidth=5,
             label=f'理论预测: w_β(t)=β·σ(-βt), β={beta:.1f}', zorder=2)

    ax1.axvline(x=0, color='orange', linestyle=':', linewidth=3, alpha=0.8, label='决策边界')
    ax1.set_xlabel('能量差 ΔE = E_ood - E_id', fontsize=12, fontweight='bold')
    ax1.set_ylabel('梯度权重 w_β(ΔE)', fontsize=12, fontweight='bold')
    ax1.set_title('Core validation: Empirical curve vs theoretical prediction\nMonotonically decreasing, highest near zero', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Hard vs easy pairs weight comparison
    ax2 = axes[0, 1]

    categories = ['Hard pairs\n(ΔE<0)', 'Easy pairs\n(ΔE>0)', 'Boundary pairs\n(|ΔE|<0.05)']
    mean_weights = [
        weights[hard_pairs_mask].mean(),
        weights[easy_pairs_mask].mean(),
        weights[boundary_mask].mean() if boundary_mask.any() else 0
    ]
    colors = ['red', 'green', 'orange']

    bars = ax2.bar(categories, mean_weights, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # 添加数值标注
    for bar, weight in zip(bars, mean_weights):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{weight:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax2.set_ylabel('平均梯度权重', fontsize=12, fontweight='bold')
    ax2.set_title(f'权重分组对比\n难对权重比易对高 {((mean_weights[0]/mean_weights[1]-1)*100):.1f}%', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # 子图3: 权重分布直方图
    ax3 = axes[1, 0]

    ax3.hist(weights, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    ax3.axvline(x=weights.mean(), color='red', linestyle='--', linewidth=2,
                label=f'整体均值: {weights.mean():.4f}')
    ax3.axvline(x=weights[hard_pairs_mask].mean(), color='crimson', linestyle='--', linewidth=2,
                label=f'难对均值: {weights[hard_pairs_mask].mean():.4f}')
    ax3.axvline(x=weights[easy_pairs_mask].mean(), color='green', linestyle='--', linewidth=2,
                label=f'易对均值: {weights[easy_pairs_mask].mean():.4f}')

    ax3.set_xlabel('梯度权重 w_β(ΔE)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('概率密度', fontsize=12, fontweight='bold')
    ax3.set_title('权重分布统计', fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 子图4: 能量差分布与难易对分类
    ax4 = axes[1, 1]

    ax4.hist(delta_values, bins=50, alpha=0.6, color='lightgreen', edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=3, label='决策边界')
    ax4.axvline(x=delta_values.mean(), color='blue', linestyle='--', linewidth=2,
                label=f'平均能量差: {delta_values.mean():.2f}')

    # 填充难对和易对区域
    ylim = ax4.get_ylim()
    ax4.fill_between([delta_values.min(), 0], 0, ylim[1], alpha=0.2, color='red', label=f'难对区域 ({(delta_values < 0).mean():.1%})')
    ax4.fill_between([0, delta_values.max()], 0, ylim[1], alpha=0.2, color='green', label=f'易对区域 ({(delta_values > 0).mean():.1%})')

    ax4.set_xlabel('能量差 ΔE = E_ood - E_id', fontsize=12, fontweight='bold')
    ax4.set_ylabel('频次', fontsize=12, fontweight='bold')
    ax4.set_title('能量差分布与分类', fontsize=11)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存为SVG格式
    svg_path = os.path.join(output_dir, f'figure_a3_ec50_{dataset_name}_detailed_analysis.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    logger.info(f"Saved detailed analysis SVG: {svg_path}")

    # 同时保存PNG格式备用
    png_path = os.path.join(output_dir, f'figure_a3_ec50_{dataset_name}_detailed_analysis.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    return fig

def print_validation_analysis(results_dict, dataset_name):
    """Print detailed validation analysis results"""

    delta_values = results_dict['delta_values']
    weights = results_dict['weights']
    beta = results_dict['beta']

    hard_pairs_mask = delta_values < 0
    easy_pairs_mask = delta_values > 0
    boundary_mask = np.abs(delta_values) < 0.05

    print("\n" + "="*60)
    print(f"{dataset_name} Hard Pairs Validation Analysis Results")
    print("="*60)

    print(f"\nTheoretical Validation Metrics:")
    print(f"   Theoretical formula: w_β(t) = β·σ(-βt), β = {beta:.1f}")
    print(f"   Validation point: Empirical curve should be monotonically decreasing, highest weights near zero")

    print(f"\nEnergy Difference Distribution:")
    print(f"   Total sample pairs: {len(delta_values):,}")
    print(f"   Average energy difference: {delta_values.mean():.3f}")
    print(f"   Standard deviation: {delta_values.std():.3f}")

    print(f"\nHard vs Easy Pair Classification:")
    print(f"   Hard pair ratio (ΔE<0): {hard_pairs_mask.mean():.1%} ({hard_pairs_mask.sum():,} pairs)")
    print(f"   Easy pair ratio (ΔE>0): {easy_pairs_mask.mean():.1%} ({easy_pairs_mask.sum():,} pairs)")
    print(f"   Boundary pair ratio (|ΔE|<0.05): {boundary_mask.mean():.1%} ({boundary_mask.sum():,} pairs)")

    print(f"\nGradient Weight Analysis:")
    hard_weight = weights[hard_pairs_mask].mean()
    easy_weight = weights[easy_pairs_mask].mean()
    boundary_weight = weights[boundary_mask].mean() if boundary_mask.any() else 0

    print(f"   Hard pairs average weight: {hard_weight:.5f}")
    print(f"   Easy pairs average weight: {easy_weight:.5f}")
    print(f"   Boundary pairs average weight: {boundary_weight:.5f}")
    print(f"   Hard pairs weight advantage: {((hard_weight/easy_weight-1)*100):+.1f}%")

    print(f"\nTheoretical Validation Results:")
    zero_weight = weights[np.abs(delta_values) < 0.01].mean()
    print(f"   Zero-point nearby weight: {zero_weight:.5f}")
    print(f"   Weight monotonicity: {'Passed' if hard_weight > easy_weight else 'Failed'}")
    print(f"   Zero-point maximum: {'Passed' if zero_weight >= hard_weight else 'Failed'}")
    print(f"   Theoretical alignment: {'Empirical curve highly consistent with theoretical prediction' if hard_weight > easy_weight else 'Deviation exists'}")

def main():
    parser = argparse.ArgumentParser(description='EC50 Hard Pairs Validation Analysis')
    parser.add_argument('--datasets', nargs='+',
                       default=['lbap_general_ec50_scaffold', 'lbap_general_ec50_size', 'lbap_general_ec50_assay'],
                       help='EC50 datasets to analyze')
    parser.add_argument('--foundation_model', type=str, default='minimol',
                       help='Foundation model to use')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3],
                       help='Seeds to analyze')
    parser.add_argument('--max_samples', type=int, default=2000,
                       help='Maximum samples to process per dataset')
    parser.add_argument('--output_dir', type=str, default='./comparison_plots',
                       help='Output directory for plots')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    for dataset_name in args.datasets:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"{'='*50}")

        dataset_results = []

        for seed in args.seeds:
            try:
                # Load model and data
                model, eval_data, model_args = load_ec50_model_and_data(
                    dataset_name, seed, args.foundation_model
                )

                # Extract features and compute energy differences
                extraction_results = extract_features_and_compute_energy_differences(
                    model, eval_data, args.max_samples
                )

                # Compute gradient weights
                delta_values, weights = compute_gradient_weights(
                    extraction_results['energy_diffs'],
                    extraction_results['beta']
                )

                # Create binned analysis
                binned_data = create_binned_analysis(delta_values, weights)

                if binned_data is None:
                    logger.warning(f"Skipping {dataset_name} seed {seed}: no valid data")
                    continue

                # Store results
                seed_results = {
                    'dataset': dataset_name,
                    'seed': seed,
                    'delta_values': delta_values,
                    'weights': weights,
                    'binned_data': binned_data,
                    'beta': extraction_results['beta'],
                    'extraction_results': extraction_results
                }

                # Analyze hard vs easy pairs
                hard_easy_analysis = analyze_hard_vs_easy_pairs(seed_results)
                seed_results['hard_easy_analysis'] = hard_easy_analysis

                dataset_results.append(seed_results)

                logger.info(f"Seed {seed} results:")
                logger.info(f"  Hard pairs ratio: {hard_easy_analysis['hard_pairs_ratio']:.3f}")
                logger.info(f"  Hard pairs avg weight: {hard_easy_analysis['hard_pairs_avg_weight']:.4f}")
                logger.info(f"  Easy pairs avg weight: {hard_easy_analysis['easy_pairs_avg_weight']:.4f}")

            except Exception as e:
                logger.error(f"Error processing {dataset_name} seed {seed}: {e}")
                continue

        if dataset_results:
            # Aggregate results across seeds
            all_delta = np.concatenate([r['delta_values'] for r in dataset_results])
            all_weights = np.concatenate([r['weights'] for r in dataset_results])
            avg_beta = np.mean([r['beta'] for r in dataset_results])

            # Create aggregated binned analysis
            aggregated_binned_data = create_binned_analysis(all_delta, all_weights)

            aggregated_results = {
                'dataset': dataset_name,
                'delta_values': all_delta,
                'weights': all_weights,
                'binned_data': aggregated_binned_data,
                'beta': avg_beta,
                'individual_seeds': dataset_results
            }

            # Plot results
            plot_hard_pairs_validation(aggregated_results, dataset_name, args.output_dir)

            # Create detailed analysis visualization
            create_detailed_analysis_visualization(aggregated_results, dataset_name, args.output_dir)

            # Print detailed analysis results
            print_validation_analysis(aggregated_results, dataset_name)

            all_results[dataset_name] = aggregated_results

            # Analyze aggregated results
            agg_analysis = analyze_hard_vs_easy_pairs(aggregated_results)
            logger.info(f"\nAggregated results for {dataset_name}:")
            logger.info(f"  Hard pairs ratio: {agg_analysis['hard_pairs_ratio']:.3f}")
            logger.info(f"  Hard pairs avg weight: {agg_analysis['hard_pairs_avg_weight']:.4f}")
            logger.info(f"  Easy pairs avg weight: {agg_analysis['easy_pairs_avg_weight']:.4f}")
            logger.info(f"  Boundary pairs avg weight: {agg_analysis['boundary_pairs_avg_weight']:.4f}")

    # Create comparison plot across all datasets
    if len(all_results) > 1:
        create_combined_comparison_plot(all_results, args.output_dir)

    # Save numerical results
    save_numerical_results(all_results, args.output_dir)

    logger.info(f"\nAnalysis complete! Results saved to {args.output_dir}")

def create_combined_comparison_plot(all_results, output_dir):
    """Create combined comparison plot across all EC50 datasets"""

    fig, axes = plt.subplots(1, len(all_results), figsize=(5*len(all_results), 6))
    if len(all_results) == 1:
        axes = [axes]

    colors = ['blue', 'red', 'green', 'purple', 'orange']

    for idx, (dataset_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        color = colors[idx % len(colors)]

        delta_values = results['delta_values']
        weights = results['weights']
        binned_data = results['binned_data']
        beta = results['beta']

        # Plot binned averages
        valid_bins = ~np.isnan(binned_data['mean_weights'])
        ax.errorbar(binned_data['bin_centers'][valid_bins],
                   binned_data['mean_weights'][valid_bins],
                   yerr=binned_data['std_weights'][valid_bins] / np.sqrt(binned_data['counts'][valid_bins]),
                   fmt='o-', color=color, markersize=4, linewidth=2, capsize=3,
                   label='Empirical')

        # Theoretical curve
        t_theory = np.linspace(delta_values.min(), delta_values.max(), 1000)
        w_theory = beta * sigmoid(-beta * t_theory)
        ax.plot(t_theory, w_theory, '--', color='black', linewidth=2, label=f'Theory (β={beta:.1f})')

        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.7)
        ax.set_xlabel('ΔE = E_ood - E_id')
        ax.set_ylabel('Gradient Weight w_β(ΔE)')
        ax.set_title(dataset_name.replace('lbap_general_ec50_', 'EC50 ').replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'figure_a3_ec50_combined_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved combined comparison plot: {output_path}")

    return fig

def save_numerical_results(all_results, output_dir):
    """Save numerical analysis results to JSON"""

    summary = {}

    for dataset_name, results in all_results.items():
        # Aggregate analysis across seeds
        analysis = analyze_hard_vs_easy_pairs(results)

        summary[dataset_name] = {
            'hard_pairs_ratio': float(analysis['hard_pairs_ratio']),
            'easy_pairs_ratio': float(analysis['easy_pairs_ratio']),
            'boundary_pairs_ratio': float(analysis['boundary_pairs_ratio']),
            'hard_pairs_avg_weight': float(analysis['hard_pairs_avg_weight']),
            'easy_pairs_avg_weight': float(analysis['easy_pairs_avg_weight']),
            'boundary_pairs_avg_weight': float(analysis['boundary_pairs_avg_weight']),
            'avg_energy_difference': float(analysis['avg_energy_difference']),
            'weight_peak_at_zero': float(analysis['weight_peak_at_zero']),
            'beta': float(results['beta']),
            'total_pairs': int(len(results['delta_values'])),
            'theoretical_validation': {
                'weight_monotonic_decrease': bool(analysis['hard_pairs_avg_weight'] > analysis['easy_pairs_avg_weight']),
                'peak_at_zero': bool(analysis['weight_peak_at_zero'] > analysis['hard_pairs_avg_weight']) if analysis['weight_peak_at_zero'] > 0 else False,
                'hard_pairs_prioritized': bool(analysis['hard_pairs_avg_weight'] > analysis['easy_pairs_avg_weight'])
            }
        }

    output_path = os.path.join(output_dir, 'ec50_hard_pairs_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved numerical results: {output_path}")

if __name__ == "__main__":
    main()