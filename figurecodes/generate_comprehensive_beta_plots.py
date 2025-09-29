#!/usr/bin/env python3
"""
生成专业的Beta sensitivity分析图表
三列横排布局，共享x轴，添加CI阴影、最佳点标记、平台区间等
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
    grouped = df_subset.groupby('beta')['test_auc']

    means = grouped.mean()
    stds = grouped.std()
    counts = grouped.count()

    # 计算95% CI (使用标准误)
    ci_lower = means - 1.96 * stds / np.sqrt(counts)
    ci_upper = means + 1.96 * stds / np.sqrt(counts)

    # 处理单次运行的情况（标准差为0或NaN）
    ci_lower = ci_lower.fillna(means)
    ci_upper = ci_upper.fillna(means)

    # 找到最佳beta
    best_beta = means.idxmax()
    best_auc = means.max()

    # 计算平台区间（与最优差值 < epsilon）
    platform_mask = (best_auc - means) < epsilon
    platform_betas = means[platform_mask].index.values

    return {
        'betas': means.index.values,
        'means': means.values,
        'ci_lower': ci_lower.values,
        'ci_upper': ci_upper.values,
        'best_beta': best_beta,
        'best_auc': best_auc,
        'platform_betas': platform_betas
    }

def generate_comprehensive_plot():
    # 读取CSV结果
    csv_path = '/home/ubuntu/OOD-DPO/beta_sensitivity_results/beta_sensitivity_corrected_20250920_195300/beta_sensitivity_TEST_AUC_results.csv'
    if not os.path.exists(csv_path):
        csv_path = '/home/ubuntu/OOD-DPO/beta_sensitivity_results/beta_sensitivity_corrected_20250920_195300/beta_sensitivity_TEST_results.csv'

    df = pd.read_csv(csv_path)

    # 过滤成功的结果
    if 'failed' in df.columns:
        df = df[~df['failed']]

    # 数据集配置
    datasets_config = {
        'lbap_general_ec50_assay': {
            'name': 'Assay',
            'color': '#2E86AB',  # 明亮的蓝色
            'marker': 'o'
        },
        'lbap_general_ec50_scaffold': {
            'name': 'Scaffold',
            'color': '#F24236',  # 明亮的红色
            'marker': 's'
        },
        'lbap_general_ec50_size': {
            'name': 'Size',
            'color': '#2E8B57',  # 绿色
            'marker': '^'
        }
    }

    # 创建图表 - 三列横排
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # 共享x轴配置
    beta_range = [0.01, 0.1, 1.0, 10.0]

    for i, (dataset_key, config) in enumerate(datasets_config.items()):
        ax = axes[i]

        # 筛选数据
        dataset_df = df[df['dataset'] == dataset_key].copy()
        if dataset_df.empty:
            continue

        # 计算统计信息
        stats = calculate_ci_and_best(dataset_df)

        # 绘制原始数据点
        ax.plot(stats['betas'], stats['means'],
               marker=config['marker'], linewidth=0, markersize=12,
               color=config['color'], markerfacecolor=config['color'],
               markeredgecolor='white', markeredgewidth=1.5,
               label='Data Points')

        # 绘制平滑曲线
        if len(stats['betas']) >= 3:  # 至少需要3个点
            # 使用PCHIP保形插值，避免过冲
            try:
                # 创建更密集的beta点用于插值
                betas_smooth = np.logspace(np.log10(stats['betas'].min()),
                                         np.log10(stats['betas'].max()), 100)

                # 使用PCHIP插值器，保证不超过原始数据的局部极值
                pchip = PchipInterpolator(stats['betas'], stats['means'])
                means_smooth = pchip(betas_smooth)

                # PCHIP保形特性已经确保不会过冲，无需额外限制
                ax.plot(betas_smooth, means_smooth,
                       linewidth=2.5, color=config['color'], alpha=0.8,
                       label='Smooth Curve')

            except Exception as e:
                # 失败时画直线连接
                ax.plot(stats['betas'], stats['means'],
                       linewidth=2.5, color=config['color'], alpha=0.8)
        else:
            # 数据点太少，只画直线连接
            ax.plot(stats['betas'], stats['means'],
                   linewidth=2.5, color=config['color'], alpha=0.8)

        # 标记最佳点
        ax.axvline(stats['best_beta'], color=config['color'], linestyle='--',
                  alpha=0.7, linewidth=1.5, label='Best β')
        ax.plot(stats['best_beta'], stats['best_auc'],
               marker='D', markersize=10, color=config['color'],
               markerfacecolor='white', markeredgecolor=config['color'],
               markeredgewidth=2)

        # 设置x轴 - 对数刻度
        ax.set_xscale('log')
        ax.set_xlim(0.008, 12)

        # 主刻度
        ax.set_xticks(beta_range)
        ax.set_xticklabels([f'{x:.2f}' if x < 1 else f'{x:.0f}' for x in beta_range])

        # 次刻度
        minor_locator = LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.tick_params(axis='x', which='minor', length=3)
        ax.tick_params(axis='x', which='major', length=5)

        # y轴设置 - 使用与单独图表相同的逻辑
        dataset_ylims = {
            'Assay': (0.72, 0.77),
            'Scaffold': (0.93, 0.96),
            'Size': (0.9975, 1.0005),
        }

        if config['name'] == 'Assay':
            # Assay使用动态范围（基于实际数据）
            dmin, dmax = float(stats['means'].min()), float(stats['means'].max())
            y_low_adj = dmin - 0.005
            y_high_adj = dmax + 0.005
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
            ax.set_xlabel(r'Temperature $\beta$', fontsize=22, fontweight='bold')

        # 显示图例在左上角
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True,
                 fontsize=12)

    # 保存综合图片
    output_dir = os.path.dirname(csv_path)
    output_path = os.path.join(output_dir, 'beta_sensitivity_comprehensive.svg')

    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✅ Comprehensive beta sensitivity plot saved: {output_path}")

    # 生成3个分别的单独图片
    print("\n🎨 Generating individual plots...")
    for dataset_key, config in datasets_config.items():
        dataset_df = df[df['dataset'] == dataset_key].copy()
        if dataset_df.empty:
            continue

        # 创建单独的图表
        fig_single, ax_single = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)

        # 计算统计信息
        stats = calculate_ci_and_best(dataset_df)

        # 绘制原始数据点
        ax_single.plot(stats['betas'], stats['means'],
                      marker=config['marker'], linewidth=0, markersize=12,
                      color=config['color'], markerfacecolor=config['color'],
                      markeredgecolor='white', markeredgewidth=1.5,
                      label='Data Points')

        # 绘制平滑曲线
        if len(stats['betas']) >= 3:
            try:
                betas_smooth = np.logspace(np.log10(stats['betas'].min()),
                                         np.log10(stats['betas'].max()), 100)
                pchip = PchipInterpolator(stats['betas'], stats['means'])
                means_smooth = pchip(betas_smooth)
                ax_single.plot(betas_smooth, means_smooth,
                              linewidth=2.5, color=config['color'], alpha=0.8,
                              label='Smooth Curve')
            except Exception:
                ax_single.plot(stats['betas'], stats['means'],
                              linewidth=2.5, color=config['color'], alpha=0.8)
        else:
            ax_single.plot(stats['betas'], stats['means'],
                          linewidth=2.5, color=config['color'], alpha=0.8)

        # 标记最佳点
        ax_single.axvline(stats['best_beta'], color=config['color'], linestyle='--',
                         alpha=0.7, linewidth=1.5, label='Best β')
        ax_single.plot(stats['best_beta'], stats['best_auc'],
                      marker='D', markersize=10, color=config['color'],
                      markerfacecolor='white', markeredgecolor=config['color'],
                      markeredgewidth=2)

        # 设置x轴
        ax_single.set_xscale('log')
        ax_single.set_xlim(0.008, 12)
        beta_range = [0.01, 0.1, 1.0, 10.0]
        ax_single.set_xticks(beta_range)
        ax_single.set_xticklabels([f'{x:.2f}' if x < 1 else f'{x:.0f}' for x in beta_range])

        # 次刻度
        minor_locator = LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20)
        ax_single.xaxis.set_minor_locator(minor_locator)
        ax_single.tick_params(axis='x', which='minor', length=3)
        ax_single.tick_params(axis='x', which='major', length=5)

        # y轴设置
        dataset_ylims = {
            'Assay': (0.72, 0.77),
            'Scaffold': (0.93, 0.96),
            'Size': (0.9975, 1.0005),
        }

        if config['name'] == 'Assay':
            dmin, dmax = float(stats['means'].min()), float(stats['means'].max())
            y_low_adj = dmin - 0.005
            y_high_adj = dmax + 0.005
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
        ax_single.set_xlabel(r'Temperature $\beta$', fontsize=22, fontweight='bold')
        ax_single.set_ylabel('Test AUC', fontsize=22, fontweight='bold')

        # 图例
        ax_single.legend(loc='upper left', frameon=True, fancybox=True, shadow=True,
                        fontsize=12)

        # 保存单独图片
        individual_output_path = os.path.join(output_dir, f'beta_sensitivity_{config["name"].lower()}_individual.svg')
        plt.savefig(individual_output_path, format='svg', bbox_inches='tight', dpi=300)
        plt.close()

        print(f"✅ Individual {config['name']} plot saved: {individual_output_path}")

    # 打印统计信息
    print("\n📊 Beta Sensitivity Analysis Summary:")
    for dataset_key, config in datasets_config.items():
        dataset_df = df[df['dataset'] == dataset_key].copy()
        if not dataset_df.empty:
            stats = calculate_ci_and_best(dataset_df)
            platform_range = f"[{stats['platform_betas'].min():.2f}, {stats['platform_betas'].max():.2f}]" if len(stats['platform_betas']) > 1 else "N/A"
            print(f"  {config['name']:8}: Best β = {stats['best_beta']:5.2f}, AUC = {stats['best_auc']:.4f}, Platform = {platform_range}")

if __name__ == '__main__':
    generate_comprehensive_plot()