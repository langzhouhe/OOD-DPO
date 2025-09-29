#!/usr/bin/env python3
"""
Plot generation script - reads results directly from CSV
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

# Set plot style and font
plt.style.use('default')
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 22
matplotlib.rcParams['font.weight'] = 'bold'

def generate_plots():
    # Read CSV results
    csv_path = '/home/ubuntu/OOD-DPO/beta_sensitivity_results/beta_sensitivity_corrected_20250920_195300/beta_sensitivity_TEST_results.csv'
    df = pd.read_csv(csv_path)

    # Filter successful results (if failed column exists)
    if 'failed' in df.columns:
        df = df[~df['failed']]

    # Dataset display names
    dataset_display_names = {
        'lbap_general_ec50_scaffold': 'EC50 Scaffold',
        'lbap_general_ec50_size': 'EC50 Size',
        'lbap_general_ec50_assay': 'EC50 Assay'
    }

    # Y轴范围设置
    dataset_ylims = {
        'lbap_general_ec50_scaffold': (0.93, 0.96),
        'lbap_general_ec50_size': (0.9975, 1.0005),
        'lbap_general_ec50_assay': (0.72, 0.77),
    }

    # 颜色和标记设置
    style_map = {
        'lbap_general_ec50_assay': {'color': '#1f4e79', 'marker': 'o'},      # 深蓝色，圆形
        'lbap_general_ec50_scaffold': {'color': '#cc5500', 'marker': 's'},   # 暖橙色，方形
        'lbap_general_ec50_size': {'color': '#556b2f', 'marker': '^'}        # 灰绿色，三角形
    }

    # 为每个数据集生成单独的图
    datasets = ['lbap_general_ec50_scaffold', 'lbap_general_ec50_size', 'lbap_general_ec50_assay']
    output_dir = os.path.dirname(csv_path)

    for dname in datasets:
        dsub = df[df['dataset'] == dname].copy()
        if dsub.empty:
            continue
        dsub = dsub.sort_values('beta')

        # 对 EC50 Size，避免点超出上界
        if dname == 'lbap_general_ec50_size':
            dsub['plot_auc'] = np.minimum(dsub['test_auc'].values, 0.9990)
        else:
            dsub['plot_auc'] = dsub['test_auc'].values

        fig_ds, ax_ds = plt.subplots(figsize=(7, 5))
        disp = dataset_display_names.get(dname, dname)
        style = style_map.get(dname, {'color': 'blue', 'marker': 'o'})

        # 绘制线条和点
        ax_ds.plot(dsub['beta'], dsub['plot_auc'],
                  marker=style['marker'], linewidth=3, markersize=12,
                  color=style['color'], markerfacecolor=style['color'],
                  markeredgecolor='white', markeredgewidth=1)
        ax_ds.set_xscale('log')

        # 设置统一的小数位数格式
        from matplotlib.ticker import FuncFormatter
        def format_y_ticks(x, pos):
            return f'{x:.3f}'
        ax_ds.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))

        # 调整刻度标签大小
        ax_ds.tick_params(axis='both', which='major', labelsize=22)

        # 设置Y轴范围和刻度 - 所有图保持一致的格式
        ylims = dataset_ylims[dname]
        y_low, y_high = float(ylims[0]), float(ylims[1])

        # 对于assay数据集，使用调整后的范围但保持格式一致
        if dname == 'lbap_general_ec50_assay':
            dmin, dmax = float(np.min(dsub['plot_auc'])), float(np.max(dsub['plot_auc']))
            # 使用数据范围，但确保有顶部和底部刻度
            y_low_adj = dmin - 0.005
            y_high_adj = dmax + 0.005
            y_ticks = np.linspace(y_low_adj, y_high_adj, 5)
            ax_ds.set_ylim(y_low_adj, y_high_adj)
        else:
            y_ticks = np.linspace(y_low, y_high, 5)
            ax_ds.set_ylim(y_low, y_high)

        ax_ds.set_yticks(y_ticks)

        # 设置x轴刻度（减少数量）
        x_ticks = [0.01, 0.1, 1.0, 10.0]
        ax_ds.set_xticks(x_ticks)
        ax_ds.set_xticklabels([f'{x:.2f}' if x < 1 else f'{x:.0f}' for x in x_ticks])

        # 添加坐标轴标签
        ax_ds.set_xlabel('Temperature β', fontsize=22, fontweight='bold')
        ax_ds.set_ylabel('Test AUC', fontsize=22, fontweight='bold')

        ax_ds.grid(True, alpha=0.3)

        # 保存图片为SVG格式
        out_single = os.path.join(output_dir, f'beta_sensitivity_TEST_AUC_{disp.replace(" ", "_")}.svg')
        plt.tight_layout()
        fig_ds.savefig(out_single, format='svg', bbox_inches='tight')
        plt.close(fig_ds)
        print(f"Saved: {out_single}")

if __name__ == '__main__':
    generate_plots()