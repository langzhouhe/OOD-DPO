#!/usr/bin/env python3
"""
生成专业的Lambda sensitivity分析图表
三列横排布局，共享x轴，统一使用紫色配色
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.ticker import LogLocator, LogFormatter
import seaborn as sns
from scipy.interpolate import UnivariateSpline, interp1d, PchipInterpolator

# 设置绘图风格和字体
plt.style.use('default')
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 22
matplotlib.rcParams['axes.linewidth'] = 1.2
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

def calculate_ci_and_best(df_subset, epsilon=0.005):
    """计算置信区间、最佳点和平台区间"""
    grouped = df_subset.groupby('lambda')['test_auc']

    means = grouped.mean()
    stds = grouped.std()
    counts = grouped.count()

    # 计算95% CI (使用标准误)
    ci_lower = means - 1.96 * stds / np.sqrt(counts)
    ci_upper = means + 1.96 * stds / np.sqrt(counts)

    # 处理单次运行的情况（标准差为0或NaN）
    ci_lower = ci_lower.fillna(means)
    ci_upper = ci_upper.fillna(means)

    # 找到最佳lambda
    best_lambda = means.idxmax()
    best_auc = means.max()

    # 计算平台区间（与最优差值 < epsilon）
    platform_mask = (best_auc - means) < epsilon
    platform_lambdas = means[platform_mask].index.values

    return {
        'lambdas': means.index.values,
        'means': means.values,
        'ci_lower': ci_lower.values,
        'ci_upper': ci_upper.values,
        'best_lambda': best_lambda,
        'best_auc': best_auc,
        'platform_lambdas': platform_lambdas
    }

def generate_comprehensive_lambda_plot():
    # 读取CSV结果
    csv_path = '/home/ubuntu/OOD-DPO/lambda_sensitivity_results/lambda_sensitivity_20250920_232010/lambda_sensitivity_TEST_results.csv'
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 过滤成功的结果
    if 'failed' in df.columns:
        df = df[~df['failed']]

    # 数据集配置 - 统一使用scaffold的颜色
    datasets_config = {
        'lbap_general_ec50_assay': {
            'name': 'Assay',
            'color': '#A569BD',  # 中紫色（scaffold颜色）
            'marker': 'o'
        },
        'lbap_general_ec50_scaffold': {
            'name': 'Scaffold',
            'color': '#A569BD',  # 中紫色（scaffold颜色）
            'marker': 's'
        },
        'lbap_general_ec50_size': {
            'name': 'Size',
            'color': '#A569BD',  # 中紫色（scaffold颜色）
            'marker': '^'
        }
    }

    # 创建图表 - 三列横排
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # 共享x轴配置
    lambda_range = [0.01, 0.1, 1.0, 5.0]

    for i, (dataset_key, config) in enumerate(datasets_config.items()):
        ax = axes[i]

        # 筛选数据
        dataset_df = df[df['dataset'] == dataset_key].copy()
        if dataset_df.empty:
            continue

        # 计算统计信息
        stats = calculate_ci_and_best(dataset_df)

        # 绘制原始数据点
        ax.plot(stats['lambdas'], stats['means'],
               marker=config['marker'], linewidth=0, markersize=12,
               color=config['color'], markerfacecolor=config['color'],
               markeredgecolor='white', markeredgewidth=1.5,
               label='Data Points')

        # 绘制平滑曲线
        if len(stats['lambdas']) >= 3:  # 至少需要3个点
            # 使用PCHIP保形插值，避免过冲
            try:
                # 创建更密集的lambda点用于插值
                lambdas_smooth = np.logspace(np.log10(stats['lambdas'].min()),
                                           np.log10(stats['lambdas'].max()), 100)

                # 使用PCHIP插值器，保证不超过原始数据的局部极值
                pchip = PchipInterpolator(stats['lambdas'], stats['means'])
                means_smooth = pchip(lambdas_smooth)

                # PCHIP保形特性已经确保不会过冲，无需额外限制
                ax.plot(lambdas_smooth, means_smooth,
                       linewidth=2.5, color=config['color'], alpha=0.8,
                       label='Smooth Curve')

            except Exception as e:
                # 失败时画直线连接
                ax.plot(stats['lambdas'], stats['means'],
                       linewidth=2.5, color=config['color'], alpha=0.8)
        else:
            # 数据点太少，只画直线连接
            ax.plot(stats['lambdas'], stats['means'],
                   linewidth=2.5, color=config['color'], alpha=0.8)

        # 标记最佳点
        ax.axvline(stats['best_lambda'], color=config['color'], linestyle='--',
                  alpha=0.7, linewidth=1.5, label='Best λ')
        ax.plot(stats['best_lambda'], stats['best_auc'],
               marker='D', markersize=10, color=config['color'],
               markerfacecolor='white', markeredgecolor=config['color'],
               markeredgewidth=2)

        # 设置x轴 - 对数刻度
        ax.set_xscale('log')
        ax.set_xlim(0.008, 6)

        # 主刻度
        ax.set_xticks(lambda_range)
        ax.set_xticklabels([f'{x:.2f}' if x < 1 else f'{x:.0f}' for x in lambda_range])

        # 次刻度
        minor_locator = LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.tick_params(axis='x', which='minor', length=3)
        ax.tick_params(axis='x', which='major', length=5)

        # y轴设置 - 使用与lambda相关的数据范围
        dataset_ylims = {
            'Assay': (0.5, 0.7),
            'Scaffold': (0.7, 1.0),
            'Size': (0.55, 1.015),  # 拉高Y轴上限到1.015
        }

        if config['name'] == 'Assay':
            # Assay使用动态范围（基于实际数据）
            dmin, dmax = float(stats['means'].min()), float(stats['means'].max())
            y_low_adj = dmin - 0.01
            y_high_adj = dmax + 0.01
            y_ticks = np.linspace(y_low_adj, y_high_adj, 5)
            ax.set_ylim(y_low_adj, y_high_adj)
            ax.set_yticks(y_ticks)
        elif config['name'] in ['Scaffold', 'Size']:
            # Scaffold和Size使用固定范围
            ylims = dataset_ylims[config['name']]
            y_low, y_high = float(ylims[0]), float(ylims[1])
            y_ticks = np.linspace(y_low, y_high, 5)
            ax.set_ylim(y_low, y_high)
            ax.set_yticks(y_ticks)
        else:
            # 备用：动态设置
            y_range = stats['means'].max() - stats['means'].min()
            y_margin = max(y_range * 0.1, 0.002)
            ax.set_ylim(stats['means'].min() - y_margin,
                       stats['means'].max() + y_margin)

        # 格式化y轴标签
        from matplotlib.ticker import FuncFormatter
        def format_y_ticks(x, pos):
            return f'{x:.3f}'
        ax.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))

        # 网格线 - 只有y方向的细网格
        ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)

        # 标题和标签
        ax.set_title(f'{config["name"]}', fontsize=22, fontweight='bold')
        if i == 0:
            ax.set_ylabel('Test AUC', fontsize=22, fontweight='bold')

        # 只在中间的子图添加x轴标签
        if i == 1:
            ax.set_xlabel(r'Regularization $\lambda$', fontsize=22, fontweight='bold')

        # 显示图例在底部中间
        ax.legend(loc='lower center', frameon=True, fancybox=True, shadow=True,
                 fontsize=12)

    # 保存综合图片
    output_dir = os.path.dirname(csv_path)
    output_path = os.path.join(output_dir, 'lambda_sensitivity_comprehensive.svg')

    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✅ Comprehensive lambda sensitivity plot saved: {output_path}")

    # 生成3个分别的单独图片
    print("\n🎨 Generating individual lambda plots...")
    for dataset_key, config in datasets_config.items():
        dataset_df = df[df['dataset'] == dataset_key].copy()
        if dataset_df.empty:
            continue

        # 创建单独的图表
        fig_single, ax_single = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)

        # 计算统计信息
        stats = calculate_ci_and_best(dataset_df)

        # 绘制原始数据点
        ax_single.plot(stats['lambdas'], stats['means'],
                      marker=config['marker'], linewidth=0, markersize=12,
                      color=config['color'], markerfacecolor=config['color'],
                      markeredgecolor='white', markeredgewidth=1.5,
                      label='Data Points')

        # 绘制平滑曲线
        if len(stats['lambdas']) >= 3:
            try:
                lambdas_smooth = np.logspace(np.log10(stats['lambdas'].min()),
                                           np.log10(stats['lambdas'].max()), 100)
                pchip = PchipInterpolator(stats['lambdas'], stats['means'])
                means_smooth = pchip(lambdas_smooth)
                ax_single.plot(lambdas_smooth, means_smooth,
                              linewidth=2.5, color=config['color'], alpha=0.8,
                              label='Smooth Curve')
            except Exception:
                ax_single.plot(stats['lambdas'], stats['means'],
                              linewidth=2.5, color=config['color'], alpha=0.8)
        else:
            ax_single.plot(stats['lambdas'], stats['means'],
                          linewidth=2.5, color=config['color'], alpha=0.8)

        # 标记最佳点
        ax_single.axvline(stats['best_lambda'], color=config['color'], linestyle='--',
                         alpha=0.7, linewidth=1.5, label='Best λ')
        ax_single.plot(stats['best_lambda'], stats['best_auc'],
                      marker='D', markersize=10, color=config['color'],
                      markerfacecolor='white', markeredgecolor=config['color'],
                      markeredgewidth=2)

        # 设置x轴
        ax_single.set_xscale('log')
        ax_single.set_xlim(0.008, 6)
        lambda_range = [0.01, 0.1, 1.0, 5.0]
        ax_single.set_xticks(lambda_range)
        ax_single.set_xticklabels([f'{x:.2f}' if x < 1 else f'{x:.0f}' for x in lambda_range])

        # 次刻度
        minor_locator = LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20)
        ax_single.xaxis.set_minor_locator(minor_locator)
        ax_single.tick_params(axis='x', which='minor', length=3)
        ax_single.tick_params(axis='x', which='major', length=5)

        # y轴设置
        dataset_ylims = {
            'Assay': (0.5, 0.7),
            'Scaffold': (0.7, 1.0),
            'Size': (0.55, 1.015),  # 拉高Y轴上限到1.015
        }

        if config['name'] == 'Assay':
            dmin, dmax = float(stats['means'].min()), float(stats['means'].max())
            y_low_adj = dmin - 0.01
            y_high_adj = dmax + 0.01
            y_ticks = np.linspace(y_low_adj, y_high_adj, 5)
            ax_single.set_ylim(y_low_adj, y_high_adj)
            ax_single.set_yticks(y_ticks)
        elif config['name'] in ['Scaffold', 'Size']:
            ylims = dataset_ylims[config['name']]
            y_low, y_high = float(ylims[0]), float(ylims[1])
            y_ticks = np.linspace(y_low, y_high, 5)
            ax_single.set_ylim(y_low, y_high)
            ax_single.set_yticks(y_ticks)
        else:
            y_range = stats['means'].max() - stats['means'].min()
            y_margin = max(y_range * 0.1, 0.002)
            ax_single.set_ylim(stats['means'].min() - y_margin,
                              stats['means'].max() + y_margin)

        # 格式化y轴标签
        from matplotlib.ticker import FuncFormatter
        def format_y_ticks(x, pos):
            return f'{x:.3f}'
        ax_single.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))

        # 网格线
        ax_single.grid(True, axis='y', alpha=0.3, linewidth=0.5)

        # 标题和标签
        ax_single.set_title(f'{config["name"]}', fontsize=22, fontweight='bold')
        ax_single.set_xlabel(r'Regularization $\lambda$', fontsize=22, fontweight='bold')
        ax_single.set_ylabel('Test AUC', fontsize=22, fontweight='bold')

        # 图例
        ax_single.legend(loc='lower center', frameon=True, fancybox=True, shadow=True,
                        fontsize=12)

        # 保存单独图片
        individual_output_path = os.path.join(output_dir, f'lambda_sensitivity_{config["name"].lower()}_individual.svg')
        plt.savefig(individual_output_path, format='svg', bbox_inches='tight', dpi=300)
        plt.close()

        print(f"✅ Individual {config['name']} lambda plot saved: {individual_output_path}")

    # 打印统计信息
    print("\n📊 Lambda Sensitivity Analysis Summary:")
    for dataset_key, config in datasets_config.items():
        dataset_df = df[df['dataset'] == dataset_key].copy()
        if not dataset_df.empty:
            stats = calculate_ci_and_best(dataset_df)
            platform_range = f"[{stats['platform_lambdas'].min():.2f}, {stats['platform_lambdas'].max():.2f}]" if len(stats['platform_lambdas']) > 1 else "N/A"
            print(f"  {config['name']:8}: Best λ = {stats['best_lambda']:5.2f}, AUC = {stats['best_auc']:.4f}, Platform = {platform_range}")

if __name__ == '__main__':
    generate_comprehensive_lambda_plot()