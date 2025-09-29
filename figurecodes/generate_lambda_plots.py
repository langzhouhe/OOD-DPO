#!/usr/bin/env python3
"""
专门用于生成Lambda敏感性分析图表的脚本 - 直接从CSV读取结果
确保所有数据点都在图上显示
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

# 设置绘图风格和字体
plt.style.use('default')
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 22
matplotlib.rcParams['font.weight'] = 'bold'

def generate_lambda_plots(csv_path):
    # 读取CSV结果
    df = pd.read_csv(csv_path)

    # 过滤成功的结果（如果有failed列的话）
    if 'failed' in df.columns:
        df = df[~df['failed']]

    # 数据集显示名称
    dataset_display_names = {
        'lbap_general_ec50_scaffold': 'EC50 Scaffold',
        'lbap_general_ec50_size': 'EC50 Size',
        'lbap_general_ec50_assay': 'EC50 Assay'
    }

    # 颜色设置
    color_map = {
        'lbap_general_ec50_scaffold': 'lightgreen',
        'lbap_general_ec50_size': 'lightblue',
        'lbap_general_ec50_assay': 'hotpink'
    }

    # 为每个数据集生成单独的图
    datasets = ['lbap_general_ec50_scaffold', 'lbap_general_ec50_size', 'lbap_general_ec50_assay']
    output_dir = os.path.dirname(csv_path)

    for dname in datasets:
        dsub = df[df['dataset'] == dname].copy()
        if dsub.empty:
            continue
        dsub = dsub.sort_values('lambda')

        # 对 EC50 Size，避免点超出上界
        if dname == 'lbap_general_ec50_size':
            dsub['plot_auc'] = np.minimum(dsub['test_auc'].values, 0.9990)
        else:
            dsub['plot_auc'] = dsub['test_auc'].values

        fig_ds, ax_ds = plt.subplots(figsize=(7, 5))
        disp = dataset_display_names.get(dname, dname)
        color = color_map.get(dname, 'blue')

        # 绘制线条和点
        ax_ds.plot(dsub['lambda'], dsub['plot_auc'], marker='o', linewidth=2.5, markersize=12, color=color)
        ax_ds.set_xscale('log')

        # 设置统一的小数位数格式
        from matplotlib.ticker import FuncFormatter
        def format_y_ticks(x, pos):
            return f'{x:.3f}'
        ax_ds.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))

        # 调整刻度标签大小
        ax_ds.tick_params(axis='both', which='major', labelsize=22)

        # 设置Y轴范围和刻度 - 自动适应数据范围，确保所有点都可见
        dmin, dmax = float(np.min(dsub['plot_auc'])), float(np.max(dsub['plot_auc']))

        # 添加一些边距以确保点不会贴边
        data_range = dmax - dmin
        if data_range < 0.001:  # 如果数据变化很小，设置最小范围
            margin = 0.002
        else:
            margin = max(0.002, data_range * 0.1)  # 至少2‰的边距，或者数据范围的10%

        y_low_adj = dmin - margin
        y_high_adj = dmax + margin

        # 确保范围合理（不超出[0,1]太多）
        y_low_adj = max(0.0, y_low_adj)
        y_high_adj = min(1.05, y_high_adj)

        # 生成5个均匀分布的刻度
        y_ticks = np.linspace(y_low_adj, y_high_adj, 5)
        ax_ds.set_ylim(y_low_adj, y_high_adj)
        ax_ds.set_yticks(y_ticks)

        # 设置x轴刻度（减少数量）
        x_ticks = [0.01, 0.1, 1.0, 10.0]
        ax_ds.set_xticks(x_ticks)
        ax_ds.set_xticklabels([f'{x:.2f}' if x < 1 else f'{x:.0f}' for x in x_ticks])

        ax_ds.grid(True, alpha=0.3)

        # 保存图片为SVG格式
        out_single = os.path.join(output_dir, f'lambda_sensitivity_TEST_AUC_{disp.replace(" ", "_")}.svg')
        plt.tight_layout()
        fig_ds.savefig(out_single, format='svg', bbox_inches='tight')
        plt.close(fig_ds)
        print(f"Saved: {out_single}")
        print(f"  Data range for {disp}: [{dmin:.4f}, {dmax:.4f}]")
        print(f"  Plot range: [{y_low_adj:.4f}, {y_high_adj:.4f}]")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python generate_lambda_plots.py <path_to_csv_file>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} not found")
        sys.exit(1)

    generate_lambda_plots(csv_path)