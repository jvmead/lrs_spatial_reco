#!/usr/bin/env python3
"""
Model comparison tool for DUNE LRS spatial reconstruction.

Loads 1D residual statistics from multiple reconstruction runs (analytic, GNN, etc.)
and generates comparison plots showing relative performance.

Example usage:
    python compare_models.py processed_outputs_*/

Or specify model directories directly:
    python compare_models.py \
        processed_outputs_*/analytic_reco_outputs_pde_*/ \
        processed_outputs_*/gnn_reco_outputs_*/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================== DATA LOADING ====================

def find_model_runs(base_dirs):
    """
    Find all reconstruction output directories in base directories.

    Groups by model type (analytic_reco, gnn_reco, etc.)

    Args:
        base_dirs: List of directories to search (e.g., processed_outputs_*)

    Returns:
        Dictionary mapping model type to list of output directories
    """
    model_runs = {}

    for base_dir in base_dirs:
        if not base_dir.is_dir():
            continue

        # Look for reconstruction output directories
        for subdir in base_dir.iterdir():
            if not subdir.is_dir():
                continue

            # Check if it's a reconstruction output directory
            has_residuals = (subdir / "residuals_1d").exists()
            has_manifest = (subdir / "manifest.json").exists() or (subdir / "config.json").exists()

            if has_residuals and has_manifest:
                # Extract model type from directory name
                name = subdir.name
                if "analytic_reco" in name:
                    model_type = "analytic"
                elif "gnn_reco" in name:
                    model_type = "gnn"
                else:
                    model_type = "unknown"

                if model_type not in model_runs:
                    model_runs[model_type] = []
                model_runs[model_type].append(subdir)

    return model_runs


def load_manifest(model_dir):
    """Load manifest.json or config.json from model directory."""
    manifest_path = model_dir / "manifest.json"
    config_path = model_dir / "config.json"

    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            return json.load(f)
    elif config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        return {}


def load_1d_residuals(model_dir, variable):
    """
    Load 1D residual statistics for a given variable.

    Args:
        model_dir: Path to reconstruction output directory
        variable: Variable name (truex, truey, truez, total_signal, dr)

    Returns:
        DataFrame with columns: bin_x, dx_mu, dx_sig, dy_mu, dy_sig, dz_mu, dz_sig, dr_mu, dr_sig
    """
    resid_file = model_dir / "residuals_1d" / f"resids_f_{variable}.csv.gz"

    if not resid_file.exists():
        return None

    return pd.read_csv(resid_file)


def get_model_label(model_dir, manifest):
    """
    Generate a human-readable label for the model.

    Args:
        model_dir: Path to reconstruction output directory
        manifest: Loaded manifest dictionary

    Returns:
        Label string (e.g., "Analytic (PDE)", "GNN (grid8, 64h)")
    """
    name = model_dir.name

    if "analytic_reco" in name:
        if manifest.get("uw", False):
            return "Analytic (unweighted)"
        else:
            return "Analytic (PDE)"
    elif "gnn_reco" in name:
        # Extract architecture details from config (nested under "model" key)
        model_cfg = manifest.get("model", {})
        adj_type = model_cfg.get("adj_type", "unknown")
        hidden = model_cfg.get("hidden", "?")
        n_layers = model_cfg.get("n_layers", "?")
        return f"GNN ({adj_type}, {n_layers}×{hidden}h)"
    else:
        return name


# ==================== PLOTTING FUNCTIONS ====================

def plot_sigma_comparison(
    data,
    variable,
    outdir,
    residual_cols=None,
):
    """
    Plot σ (standard deviation) comparison for multiple models.

    Args:
        data: Dictionary mapping model_key to (dataframe, label)
        variable: Independent variable name
        outdir: Output directory
        residual_cols: List of residual sigma columns to plot
    """
    if residual_cols is None:
        residual_cols = ["dx_sig", "dy_sig", "dz_sig", "dr_sig"]

    fig, axes = plt.subplots(1, len(residual_cols), figsize=(5*len(residual_cols), 4))
    if len(residual_cols) == 1:
        axes = [axes]

    # Color palette
    colors = plt.cm.tab10(range(len(data)))

    xscale = "log" if variable == "total_signal" else "linear"

    for ax_idx, resid_col in enumerate(residual_cols):
        ax = axes[ax_idx]
        resid_name = resid_col.replace("_sig", "")  # dx_sig -> dx

        for (model_key, (df, label)), color in zip(data.items(), colors):
            if resid_col in df.columns:
                # Plot with error bars if available
                err_col = resid_col + "_err"
                if err_col in df.columns:
                    ax.errorbar(
                        df["bin_x"], df[resid_col], yerr=df[err_col],
                        label=label, alpha=0.7, linewidth=2, capsize=3, color=color
                    )
                else:
                    ax.plot(df["bin_x"], df[resid_col], label=label,
                           linewidth=2, alpha=0.8, color=color)

        ax.set_xlabel(variable)
        ax.set_ylabel(f"σ({resid_name}) [cm]")
        ax.set_title(f"{resid_name.upper()} Resolution vs {variable}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if xscale == "log":
            ax.set_xscale("log")

        # Set y-axis minimum to 0 for dr sigma
        if resid_name == "dr":
            ax.set_ylim(bottom=0)

    plt.tight_layout()
    outfile = outdir / f"compare_sigma_vs_{variable}.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outfile}")


def plot_mu_comparison(
    data: Dict[str, Tuple[pd.DataFrame, str]],
    variable: str,
    outdir: Path,
    residual_cols: List[str] = ["dx_mu", "dy_mu", "dz_mu", "dr_mu"],
):
    """
    Plot σ (standard deviation) comparison for multiple models.

    Args:
        data: Dictionary mapping model_key to (dataframe, label)
        variable: Independent variable name
        outdir: Output directory
        residual_cols: List of residual sigma columns to plot
    """
    if residual_cols is None:
        residual_cols = ["dx_sig", "dy_sig", "dz_sig", "dr_sig"]

    fig, axes = plt.subplots(1, len(residual_cols), figsize=(5*len(residual_cols), 4))
    if len(residual_cols) == 1:
        axes = [axes]

    colors = plt.cm.tab10(range(len(data)))
    xscale = "log" if variable == "total_signal" else "linear"

    for ax_idx, resid_col in enumerate(residual_cols):
        ax = axes[ax_idx]
        resid_name = resid_col.replace("_mu", "")  # dx_mu -> dx

        for (model_key, (df, label)), color in zip(data.items(), colors):
            if resid_col in df.columns:
                # Plot with error bars if available
                err_col = resid_col + "_err"
                if err_col in df.columns:
                    ax.errorbar(
                        df["bin_x"], df[resid_col], yerr=df[err_col],
                        label=label, alpha=0.7, linewidth=2, capsize=3, color=color
                    )
                else:
                    ax.plot(df["bin_x"], df[resid_col], label=label,
                           linewidth=2, alpha=0.8, color=color)

        # Add zero reference line
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xlabel(variable)
        ax.set_ylabel(f"μ({resid_name}) [cm]")
        ax.set_title(f"{resid_name.upper()} Bias vs {variable}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if xscale == "log":
            ax.set_xscale("log")

    plt.tight_layout()
    outfile = outdir / f"compare_mu_vs_{variable}.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outfile}")


def plot_summary_table(
    data,
    variable,
    outdir,
):
    """
    Create a summary table showing median/mean performance metrics.

    Args:
        data: Dictionary mapping model_key to (dataframe, label)
        variable: Independent variable name
        outdir: Output directory
    """
    # Compute summary statistics for each model
    summary = []

    for model_key, (df, label) in data.items():
        stats = {"Model": label}

        # Median σ values
        for col in ["dx_sig", "dy_sig", "dz_sig", "dr_sig"]:
            if col in df.columns:
                stats[f"median_{col}"] = df[col].median()

        # Mean |μ| values (absolute bias)
        for col in ["dx_mu", "dy_mu", "dz_mu", "dr_mu"]:
            if col in df.columns:
                stats[f"mean_abs_{col}"] = df[col].abs().mean()

        summary.append(stats)

    df_summary = pd.DataFrame(summary)

    # Save to CSV
    outfile = outdir / f"summary_stats_{variable}.csv"
    df_summary.to_csv(outfile, index=False, float_format="%.3f")
    print(f"Saved: {outfile}")

    # Create text table visualization
    fig, ax = plt.subplots(figsize=(12, len(summary) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')

    # Format table
    table_data = []
    headers = ["Model", "σ(dx)", "σ(dy)", "σ(dz)", "σ(dr)", "|μ(dx)|", "|μ(dy)|", "|μ(dz)|", "|μ(dr)|"]

    for _, row in df_summary.iterrows():
        table_data.append([
            row["Model"],
            f"{row.get('median_dx_sig', np.nan):.2f}",
            f"{row.get('median_dy_sig', np.nan):.2f}",
            f"{row.get('median_dz_sig', np.nan):.2f}",
            f"{row.get('median_dr_sig', np.nan):.2f}",
            f"{row.get('mean_abs_dx_mu', np.nan):.2f}",
            f"{row.get('mean_abs_dy_mu', np.nan):.2f}",
            f"{row.get('mean_abs_dz_mu', np.nan):.2f}",
            f"{row.get('mean_abs_dr_mu', np.nan):.2f}",
        ])

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title(f"Summary Statistics vs {variable} (cm)", fontsize=12, pad=20)

    outfile = outdir / f"summary_table_{variable}.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outfile}")


def plot_combined_sigma(
    data,
    variable,
    outdir,
):
    """
    Plot all σ values (dx, dy, dz, dr) for each model on a single plot.

    Args:
        data: Dictionary mapping model_key to (dataframe, label)
        variable: Independent variable name
        outdir: Output directory
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(range(len(data)))
    linestyles = ['-', '--', '-.', ':']
    residual_cols = ["dx_sig", "dy_sig", "dz_sig", "dr_sig"]
    residual_labels = ["σ(dx)", "σ(dy)", "σ(dz)", "σ(dr)"]

    xscale = "log" if variable == "total_signal" else "linear"

    for model_idx, ((model_key, (df, label)), color) in enumerate(zip(data.items(), colors)):
        for resid_idx, (resid_col, resid_label) in enumerate(zip(residual_cols, residual_labels)):
            if resid_col in df.columns:
                linestyle = linestyles[resid_idx % len(linestyles)]
                plot_label = f"{label} - {resid_label}"
                ax.plot(df["bin_x"], df[resid_col], label=plot_label,
                       linewidth=2, alpha=0.7, color=color, linestyle=linestyle)

    ax.set_xlabel(variable)
    ax.set_ylabel("Resolution σ [cm]")
    ax.set_title(f"All Resolution Components vs {variable}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    if xscale == "log":
        ax.set_xscale("log")

    plt.tight_layout()
    outfile = outdir / f"compare_all_sigma_vs_{variable}.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outfile}")


# ==================== MAIN COMPARISON FUNCTION ====================

def compare_models_for_variable(
    model_runs,
    variable,
    outdir,
):
    """
    Generate all comparison plots for a given variable.

    Args:
        model_runs: Dictionary mapping model type to list of output directories
        variable: Independent variable name
        outdir: Output directory
    """
    print(f"\n=== Comparing models for variable: {variable} ===")

    # Load data from all models
    data = {}
    for model_type, dirs in model_runs.items():
        for model_dir in dirs:
            manifest = load_manifest(model_dir)
            label = get_model_label(model_dir, manifest)
            df = load_1d_residuals(model_dir, variable)

            if df is not None:
                key = f"{model_type}_{model_dir.name}"
                data[key] = (df, label)
                print(f"  Loaded: {label} ({len(df)} bins)")

    if not data:
        print(f"  No data found for variable: {variable}")
        return

    # Generate plots
    plot_sigma_comparison(data, variable, outdir)
    plot_mu_comparison(data, variable, outdir)
    plot_combined_sigma(data, variable, outdir)
    plot_summary_table(data, variable, outdir)


# ==================== CLI ====================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare reconstruction model performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument(
        "directories",
        nargs="+",
        type=Path,
        help="Directories to search for model runs (e.g., processed_outputs_*/ or specific model output dirs)"
    )

    p.add_argument(
        "--variables",
        nargs="+",
        default=["truex", "truey", "truez", "total_signal", "dr"],
        help="Variables to compare"
    )

    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: comparison_TIMESTAMP/ in first input directory)"
    )

    p.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag for output directory name"
    )

    return p.parse_args()


def main():
    args = parse_args()

    # Expand globs if needed
    directories = []
    for d in args.directories:
        if d.exists():
            directories.append(d)

    if not directories:
        print("Error: No valid directories found")
        return 1

    # Find model runs
    print(f"Searching {len(directories)} directories for model runs...")
    model_runs = find_model_runs(directories)

    if not model_runs:
        print("Error: No reconstruction output directories found")
        print("Looking for directories with residuals_1d/ and manifest.json")
        return 1

    print(f"\nFound {sum(len(v) for v in model_runs.values())} model runs:")
    for model_type, dirs in model_runs.items():
        print(f"  {model_type}: {len(dirs)} runs")

    # Create output directory
    if args.outdir is None:
        from datetime import datetime
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        tag_str = f"_{args.tag}" if args.tag else ""
        outdir = directories[0].parent / f"comparison{tag_str}_{timestamp}"
    else:
        outdir = args.outdir

    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {outdir}")

    # Generate comparisons for each variable
    for variable in args.variables:
        compare_models_for_variable(model_runs, variable, outdir)

    print(f"\n=== Comparison complete ===")
    print(f"Results saved to: {outdir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
