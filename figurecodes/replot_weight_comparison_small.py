#!/usr/bin/env python3
"""
Re-plot the individual hard-pairs weight comparison charts at a smaller size.
Reads the existing summary JSON and writes output to individual_hard_pairs_plots1.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Smaller, cleaner style
plt.style.use('default')
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['axes.linewidth'] = 1.0
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

DATASET_COLORS = {
    'lbap_general_ec50_assay': '#2E86AB',
    'lbap_general_ec50_scaffold': '#F24236',
    'lbap_general_ec50_size': '#2E8B57',
}

def save_both(fig, base):
    fig.savefig(f"{base}.png", format='png', bbox_inches='tight', dpi=200, facecolor='white')
    fig.savefig(f"{base}.svg", format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)

def main():
    json_path = 'comparison_plots/ec50_professional_hard_pairs_analysis.json'
    out_dir = 'individual_hard_pairs_plots1'

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing JSON: {json_path}")

    os.makedirs(out_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    for dataset_name, dataset in data.items():
        primary = DATASET_COLORS.get(dataset_name, '#F24236')
        disp = dataset_name.replace('lbap_general_ec50_', '').title()

        hard_w = dataset['hard_pairs_avg_weight']
        easy_w = dataset['easy_pairs_avg_weight']
        boundary_w = dataset['boundary_pairs_avg_weight']

        # Smaller figure and fonts; slightly narrower bars
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        categories = ['Hard Pairs\n(ΔE<0)', 'Easy Pairs\n(ΔE>0)', 'Boundary\n(|ΔE|<0.05)']
        means = [hard_w, easy_w, boundary_w]
        colors = [primary, 'lightgray', 'orange']

        bars = ax.bar(categories, means, color=colors, alpha=0.75,
                      edgecolor='black', linewidth=1.0)

        # Add value labels (larger font)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='semibold')

        ax.set_ylabel('Average Gradient Weight', fontsize=18)
        if easy_w > 0:
            advantage = (hard_w / easy_w - 1) * 100.0
        else:
            advantage = 0.0
        ax.set_title(
            f'Weight Comparison - {disp}\nHard pairs get {advantage:.1f}% higher weights',
            fontsize=18,
            fontweight='bold',
            pad=8,
        )
        ax.grid(True, alpha=0.25, axis='y')

        # Larger tick labels
        ax.tick_params(axis='both', which='major', labelsize=14)

        # Slightly tighter y-limits to avoid exaggerated look
        top = max(means) * 1.15
        ax.set_ylim(0, top)

        fig.tight_layout()
        save_both(fig, os.path.join(out_dir, f'figure_a3_1_weight_comparison_{dataset_name}'))

    print(f"Saved resized charts to: {out_dir}")

if __name__ == '__main__':
    main()
