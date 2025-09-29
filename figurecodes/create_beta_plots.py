#!/usr/bin/env python3
"""
基于真实实验输出的beta敏感性可视化

本脚本读取由 beta_sensitivity_corrected.py 生成的CSV结果，
绘制以下图表：
- 折线图：各数据集 Test ROC-AUC vs Beta（对数x轴）
- 热力图：数据集 × Beta 的 Test ROC-AUC

注意：此前版本使用了“模拟数据”，这会误导分析。本版本完全移除模拟，
仅使用真实的训练/测试数据产生的结果CSV。
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

def _find_latest_results_csv(base_dir: str) -> str:
    """自动查找最近一次 beta_sensitivity_corrected 的结果CSV"""
    patterns = [
        os.path.join(base_dir, 'beta_sensitivity_corrected_*', 'beta_sensitivity_TEST_results.csv'),
        os.path.join(base_dir, '*', 'beta_sensitivity_TEST_results.csv'),
        os.path.join('./beta_sensitivity_results', 'beta_sensitivity_corrected_*', 'beta_sensitivity_TEST_results.csv'),
    ]
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(p))
    if not candidates:
        return ''
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def load_real_results(results_csv: str) -> pd.DataFrame:
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f'Results CSV not found: {results_csv}')
    df = pd.read_csv(results_csv)
    # 兼容列名：优先使用 test_auc
    if 'test_auc' not in df.columns:
        raise ValueError('CSV missing required column test_auc')
    if 'beta' not in df.columns or 'dataset' not in df.columns:
        raise ValueError('CSV must contain beta and dataset columns')
    return df

def create_line_plots(df: pd.DataFrame, output_dir: str):
    """创建ROC-AUC vs Beta的折线图"""
    print("Creating line plots...")

    # 设置绘图风格
    plt.style.use('default')
    sns.set_palette("husl")

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. 主要折线图：每个数据集的AUC vs Beta
    dataset_display_names = {
        'lbap_general_ec50_scaffold': 'EC50 Scaffold',
        'lbap_general_ec50_size': 'EC50 Size',
        'lbap_general_ec50_assay': 'EC50 Assay'
    }

    for dataset_name in ['lbap_general_ec50_scaffold', 'lbap_general_ec50_size', 'lbap_general_ec50_assay']:
        if 'failed' in df.columns:
            dsub = df[(df['dataset'] == dataset_name) & (~df['failed'])].copy()
        else:
            dsub = df[df['dataset'] == dataset_name].copy()
        if not dsub.empty:
            dsub = dsub.sort_values('beta')
            display_name = dataset_display_names.get(dataset_name, dataset_name)
            ax1.plot(dsub['beta'], dsub['test_auc'],
                     marker='o', linewidth=2.5, markersize=8,
                     label=display_name)

    ax1.set_xlabel('Beta Values', fontsize=12)
    ax1.set_ylabel('ROC-AUC Performance', fontsize=12)
    ax1.set_title('Beta Sensitivity Analysis\n(Fixed λ=0.01)', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_ylim(0.7, 1.0)

    # 添加参考线
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Guess')

    # 2. 热力图：显示所有数据集和beta值的组合
    if not df.empty:
        pivot_df = df.copy()
        pivot_df['Dataset'] = pivot_df['dataset'].map(dataset_display_names).fillna(pivot_df['dataset'])
        pivot_table = pivot_df.pivot(index='Dataset', columns='beta', values='test_auc')
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlBu_r',
                    center=0.75, ax=ax2, cbar_kws={'label': 'Test ROC-AUC'})
        ax2.set_title('Test AUC Heatmap\n(Dataset × Beta)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Beta Values', fontsize=12)
        ax2.set_ylabel('Datasets', fontsize=12)

    plt.tight_layout()

    # 保存图像
    output_path = os.path.join(output_dir, 'beta_sensitivity_TEST_AUC_from_csv.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Line plots saved to: {output_path}")
    plt.close()
    # 去除训练曲线的模拟生成：由 beta_sensitivity_corrected.py 自行输出训练相关图

def summarize_to_report(df: pd.DataFrame, output_dir: str):
    """基于真实CSV结果生成简要报告"""
    ds_map = {
        'lbap_general_ec50_scaffold': 'EC50 Scaffold',
        'lbap_general_ec50_size': 'EC50 Size',
        'lbap_general_ec50_assay': 'EC50 Assay'
    }
    successful_df = df if 'failed' not in df.columns else df[~df['failed']]
    lines = [
        '# Beta Sensitivity Analysis Report (基于真实CSV)',
        '',
        f'- 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'- 结果来源: {os.path.abspath(output_dir)}',
        '',
        '## 每个数据集最佳Beta (按Test AUC)',
    ]
    for ds in successful_df['dataset'].unique():
        sub = successful_df[successful_df['dataset'] == ds]
        if not sub.empty:
            best = sub.loc[sub['test_auc'].idxmax()]
            lines.append(f"- {ds_map.get(ds, ds)}: beta={best['beta']} | TEST AUC={best['test_auc']:.4f}")
    lines.append('')
    lines.append('## 总体统计')
    lines.append(f"- 平均TEST AUC: {successful_df['test_auc'].mean():.4f}")
    lines.append(f"- 标准差: {successful_df['test_auc'].std():.4f}")
    report_path = os.path.join(output_dir, 'csv_based_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'Report saved to: {report_path}')

def parse_args():
    parser = argparse.ArgumentParser(description='Create beta sensitivity plots from real CSV results')
    parser.add_argument('--results_csv', type=str, default='', help='Path to beta_sensitivity_TEST_results.csv')
    parser.add_argument('--search_dir', type=str, default='./beta_sensitivity_results', help='Where to search for latest CSV if not provided')
    parser.add_argument('--output_dir', type=str, default='', help='Output directory (defaults to CSV directory)')
    return parser.parse_args()

def main():
    args = parse_args()

    results_csv = args.results_csv or _find_latest_results_csv(args.search_dir)
    if not results_csv:
        print('❌ 未找到结果CSV。请先运行: python beta_sensitivity_corrected.py')
        return

    df = load_real_results(results_csv)

    # 输出目录默认与CSV所在目录一致
    output_dir = args.output_dir or os.path.dirname(results_csv)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Using results CSV: {results_csv}")
    print(f"Output directory: {output_dir}")

    # 创建可视化
    create_line_plots(df, output_dir)
    summarize_to_report(df, output_dir)

    print(f"Beta sensitivity plots created from real CSV.")

if __name__ == '__main__':
    main()
