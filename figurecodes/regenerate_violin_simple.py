#!/usr/bin/env python3
"""
Simple script to regenerate just the combined violin plot from existing data
"""

import sys
import os
sys.path.append('/home/ubuntu/OOD-DPO')

# Import the functions from the main extraction script
from extract_real_results import extract_results_from_experiments, create_combined_violin_plot

def main():
    print("Regenerating combined violin plot without title...")

    # Create output directory
    os.makedirs('comparison_plots', exist_ok=True)

    # Extract results (this will use cached/existing model outputs)
    results = extract_results_from_experiments()

    if not results:
        print("❌ No results found")
        return

    if len(results) >= 3:
        # Generate just the combined violin plot
        violin_path = os.path.join('comparison_plots', 'combined_violin_plot.svg')
        create_combined_violin_plot(results, violin_path)
        print("✅ Violin plot regenerated successfully without title!")
    else:
        print(f"❌ Only found {len(results)} methods, need at least 3")

if __name__ == "__main__":
    main()