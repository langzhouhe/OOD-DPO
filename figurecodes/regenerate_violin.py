#!/usr/bin/env python3
"""
Quick script to regenerate just the combined violin plot without title
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import os
import json
import torch
from sklearn.metrics import roc_curve, auc
import pandas as pd

# Configure matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 12
plt.style.use('default')

def load_model_predictions():
    """Load predictions from saved results"""
    results = {}

    # Define the experiments to load
    experiments = [
        ('hinge', 'ablation_results/minimol/lbap_general_ic50_scaffold/hinge_seed_1.json'),
        ('bce', 'ablation_results/minimol/lbap_general_ic50_scaffold/bce_seed_1.json'),
        ('mse', 'ablation_results/minimol/lbap_general_ic50_scaffold/mse_seed_1.json')
    ]

    for loss_type, result_path in experiments:
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r') as f:
                    data = json.load(f)

                if 'test_predictions' in data:
                    results[loss_type] = {
                        'id_scores': data['test_predictions']['id_scores'],
                        'ood_scores': data['test_predictions']['ood_scores'],
                        'auroc': data['test_predictions']['auroc']
                    }
                    print(f"Loaded {loss_type}: AUROC = {data['test_predictions']['auroc']:.4f}")
            except Exception as e:
                print(f"Error loading {result_path}: {e}")

    return results

def create_combined_violin_plot(results, output_path):
    """Create combined violin plot showing ID vs OOD score distribution"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    method_names = {'hinge': 'Hinge Loss', 'bce': 'BCE Loss', 'mse': 'MSE Loss'}
    colors = {'hinge': '#2E8B57', 'bce': '#4682B4', 'mse': '#CD853F'}

    for i, (method, data) in enumerate(results.items()):
        ax = axes[i]

        id_scores = np.array(data['id_scores'])
        ood_scores = np.array(data['ood_scores'])

        # Prepare data for violin plot
        scores_data = []
        labels_data = []

        scores_data.extend(id_scores)
        labels_data.extend(['ID'] * len(id_scores))

        scores_data.extend(ood_scores)
        labels_data.extend(['OOD'] * len(ood_scores))

        df = pd.DataFrame({
            'Score': scores_data,
            'Type': labels_data
        })

        # Create violin plot
        parts = ax.violinplot([id_scores, ood_scores], positions=[0, 1],
                             showmeans=True, showmedians=True, widths=0.7)

        # Set colors
        for pc in parts['bodies']:
            pc.set_facecolor(colors[method])
            pc.set_alpha(0.7)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['ID', 'OOD'])
        ax.set_title(f'{method_names[method]}\n(AUROC: {data["auroc"]:.4f})')

        if i == 0:
            ax.set_ylabel('Energy Score')

        # Calculate separation
        separation = np.mean(ood_scores) - np.mean(id_scores)
        ax.text(0.5, 0.95, f'Sep: {separation:.3f}',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Combined violin plot saved to: {output_path}")

def main():
    # Create output directory
    os.makedirs('comparison_plots', exist_ok=True)

    # Load predictions
    results = load_model_predictions()

    if len(results) == 3:
        # Generate combined violin plot
        violin_path = os.path.join('comparison_plots', 'combined_violin_plot.svg')
        create_combined_violin_plot(results, violin_path)
        print("Violin plot regenerated successfully!")
    else:
        print(f"Could only load {len(results)} methods, need 3")

if __name__ == "__main__":
    main()