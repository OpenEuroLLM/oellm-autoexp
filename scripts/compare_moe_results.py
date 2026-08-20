#!/usr/bin/env python3
"""Compare MoE metrics across different models.

Usage:
    python compare_moe_results.py \\
      --inputs mixtral_results.json dbrx_results.json qwen_results.json \\
      --output comparison_report.md \\
      --plot comparison_plots/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def load_results(file_path: str) -> dict:
    """Load results JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_metrics(results: dict, model_name: str) -> List[Dict]:
    """Extract per-layer metrics into flat list."""
    metrics = []
    
    layers = results.get("layers", {})
    for layer_name, layer_data in layers.items():
        metrics.append({
            "model": model_name,
            "layer": layer_name,
            "saturation": layer_data.get("saturation", layer_data.get("expert_saturation", 0)),
            "coactivation": layer_data.get("expert_coactivation", layer_data.get("coactivation", 0))
        })
    
    return metrics


def create_comparison_plots(df: pd.DataFrame, output_dir: Path):
    """Generate comparison visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Saturation comparison
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="model", y="saturation")
    plt.title("Expert Saturation by Model", fontsize=14, fontweight="bold")
    plt.ylabel("Saturation", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / "saturation_comparison.png", dpi=200)
    plt.close()
    
    # 2. Co-activation comparison
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="model", y="coactivation")
    plt.title("Expert Co-activation by Model", fontsize=14, fontweight="bold")
    plt.ylabel("Co-activation Rate", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "coactivation_comparison.png", dpi=200)
    plt.close()
    
    # 3. Violin plot for distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.violinplot(data=df, x="model", y="saturation", ax=axes[0])
    axes[0].set_title("Saturation Distribution", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Saturation")
    axes[0].set_ylim(0, 1.05)
    
    sns.violinplot(data=df, x="model", y="coactivation", ax=axes[1])
    axes[1].set_title("Co-activation Distribution", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Co-activation Rate")
    
    plt.tight_layout()
    plt.savefig(output_dir / "distribution_comparison.png", dpi=200)
    plt.close()
    
    # 4. Layer-by-layer comparison (if same number of layers)
    models = df["model"].unique()
    if len(models) == 2:
        model1_df = df[df["model"] == models[0]].sort_values("layer")
        model2_df = df[df["model"] == models[1]].sort_values("layer")
        
        if len(model1_df) == len(model2_df):
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            layers = range(len(model1_df))
            axes[0].plot(layers, model1_df["saturation"].values, marker='o', label=models[0])
            axes[0].plot(layers, model2_df["saturation"].values, marker='s', label=models[1])
            axes[0].set_xlabel("Layer")
            axes[0].set_ylabel("Saturation")
            axes[0].set_title("Saturation by Layer")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(layers, model1_df["coactivation"].values, marker='o', label=models[0])
            axes[1].plot(layers, model2_df["coactivation"].values, marker='s', label=models[1])
            axes[1].set_xlabel("Layer")
            axes[1].set_ylabel("Co-activation")
            axes[1].set_title("Co-activation by Layer")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "layer_by_layer_comparison.png", dpi=200)
            plt.close()


def generate_markdown_report(df: pd.DataFrame, output_path: Path):
    """Generate a markdown comparison report."""
    
    report = ["# MoE Metrics Comparison Report\n"]
    report.append(f"Comparing {len(df['model'].unique())} models\n")
    report.append("---\n\n")
    
    # Summary statistics
    report.append("## Summary Statistics\n\n")
    
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        report.append(f"### {model}\n\n")
        report.append(f"- **Layers analyzed**: {len(model_df)}\n")
        report.append(f"- **Average Saturation**: {model_df['saturation'].mean():.4f} (±{model_df['saturation'].std():.4f})\n")
        report.append(f"- **Average Co-activation**: {model_df['coactivation'].mean():.4f} (±{model_df['coactivation'].std():.4f})\n")
        report.append(f"- **Min Saturation**: {model_df['saturation'].min():.4f}\n")
        report.append(f"- **Max Saturation**: {model_df['saturation'].max():.4f}\n")
        report.append("\n")
    
    # Comparison table
    report.append("## Model Comparison\n\n")
    report.append("| Model | Avg Saturation | Avg Co-activation | # Layers |\n")
    report.append("|-------|----------------|-------------------|----------|\n")
    
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        avg_sat = model_df["saturation"].mean()
        avg_coact = model_df["coactivation"].mean()
        n_layers = len(model_df)
        report.append(f"| {model} | {avg_sat:.4f} | {avg_coact:.4f} | {n_layers} |\n")
    
    report.append("\n")
    
    # Interpretation
    report.append("## Interpretation\n\n")
    report.append("### Saturation\n")
    report.append("- **High saturation (>0.85)**: Good expert coverage, most experts are utilized\n")
    report.append("- **Medium saturation (0.6-0.85)**: Moderate coverage, some expert specialization\n")
    report.append("- **Low saturation (<0.6)**: Strong specialization, many experts underutilized\n\n")
    
    report.append("### Co-activation\n")
    report.append("- **High co-activation (>0.5)**: Experts frequently work together\n")
    report.append("- **Medium co-activation (0.3-0.5)**: Balanced expert collaboration\n")
    report.append("- **Low co-activation (<0.3)**: Strong expert specialization\n\n")
    
    # Rankings
    report.append("## Rankings\n\n")
    
    model_stats = []
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        model_stats.append({
            "model": model,
            "avg_sat": model_df["saturation"].mean(),
            "avg_coact": model_df["coactivation"].mean()
        })
    
    report.append("### By Saturation (High to Low)\n")
    sorted_by_sat = sorted(model_stats, key=lambda x: x["avg_sat"], reverse=True)
    for i, stat in enumerate(sorted_by_sat, 1):
        report.append(f"{i}. **{stat['model']}**: {stat['avg_sat']:.4f}\n")
    report.append("\n")
    
    report.append("### By Co-activation (High to Low)\n")
    sorted_by_coact = sorted(model_stats, key=lambda x: x["avg_coact"], reverse=True)
    for i, stat in enumerate(sorted_by_coact, 1):
        report.append(f"{i}. **{stat['model']}**: {stat['avg_coact']:.4f}\n")
    report.append("\n")
    
    # Visualizations reference
    report.append("## Visualizations\n\n")
    report.append("See the following plots for detailed comparisons:\n\n")
    report.append("1. `saturation_comparison.png` - Box plot of saturation by model\n")
    report.append("2. `coactivation_comparison.png` - Box plot of co-activation by model\n")
    report.append("3. `distribution_comparison.png` - Distribution violin plots\n")
    report.append("4. `layer_by_layer_comparison.png` - Layer-by-layer comparison (if applicable)\n")
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(report))


def main():
    parser = argparse.ArgumentParser(description="Compare MoE metrics across models")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSON files with MoE metrics"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_report.md",
        help="Output markdown report path"
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="comparison_plots",
        help="Directory for comparison plots"
    )
    
    args = parser.parse_args()
    
    # Load all results
    all_metrics = []
    for input_file in args.inputs:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"Warning: {input_file} not found, skipping...")
            continue
        
        print(f"Loading {input_file}...")
        results = load_results(input_file)
        
        # Extract model name from results or filename
        model_name = results.get("model", results.get("model_name", input_path.stem))
        
        metrics = extract_metrics(results, model_name)
        all_metrics.extend(metrics)
    
    if not all_metrics:
        print("Error: No valid metrics found!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)
    
    print(f"\nLoaded metrics from {len(df['model'].unique())} models")
    print(f"Total layers analyzed: {len(df)}")
    
    # Generate plots
    print(f"\nGenerating plots in {args.plot_dir}...")
    plot_dir = Path(args.plot_dir)
    create_comparison_plots(df, plot_dir)
    
    # Generate report
    print(f"Generating report: {args.output}...")
    output_path = Path(args.output)
    generate_markdown_report(df, output_path)
    
    print("\n✅ Comparison complete!")
    print(f"   Report: {output_path}")
    print(f"   Plots: {plot_dir}/")
    
    # Print quick summary
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        print(f"\n{model}:")
        print(f"  Saturation: {model_df['saturation'].mean():.4f} (±{model_df['saturation'].std():.4f})")
        print(f"  Co-activation: {model_df['coactivation'].mean():.4f} (±{model_df['coactivation'].std():.4f})")
    print("="*60)


if __name__ == "__main__":
    main()
