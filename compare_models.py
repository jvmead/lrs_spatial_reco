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

def plot_dr_vs_spatial(data, variable, outdir):
    """Plot dr sigma and mu as function of true x, y, z."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.tab10(range(len(data)))
    xscale = "log" if variable == "total_signal" else "linear"

    # Sigma plot
    for (model_key, (df, label)), color in zip(data.items(), colors):
        axes[0].plot(df["bin_x"], df["dr_sig"], marker="o", label=label,
                    alpha=0.7, linewidth=2, color=color)
    axes[0].set_xlabel(variable)
    axes[0].set_ylabel("σ(dr) [cm]")
    axes[0].set_title(f"3D Resolution vs {variable}")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    if xscale == "log":
        axes[0].set_xscale("log")
    axes[0].set_ylim(bottom=0)

    # Mu plot
    for (model_key, (df, label)), color in zip(data.items(), colors):
        axes[1].plot(df["bin_x"], df["dr_mu"], marker="o", label=label,
                    alpha=0.7, linewidth=2, color=color)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel(variable)
    axes[1].set_ylabel("μ(dr) [cm]")
    axes[1].set_title(f"3D Bias vs {variable}")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    if xscale == "log":
        axes[1].set_xscale("log")

    plt.tight_layout()
    outfile = outdir / f"dr_vs_{variable}.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")


def plot_residual_vs_true(data, variable, component, outdir):
    """Plot d{component} sigma and mu as function of true{component}.

    Args:
        component: 'x', 'y', or 'z'
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.tab10(range(len(data)))
    xscale = "log" if variable == "total_signal" else "linear"

    sigma_col = f"d{component}_sig"
    mu_col = f"d{component}_mu"

    # Sigma plot
    for (model_key, (df, label)), color in zip(data.items(), colors):
        if sigma_col in df.columns:
            axes[0].plot(df["bin_x"], df[sigma_col], marker="o", label=label,
                        alpha=0.7, linewidth=2, color=color)
    axes[0].set_xlabel(variable)
    axes[0].set_ylabel(f"σ(d{component}) [cm]")
    axes[0].set_title(f"{component.upper()}-Resolution vs {variable}")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    if xscale == "log":
        axes[0].set_xscale("log")
    axes[0].set_ylim(bottom=0)

    # Mu plot
    for (model_key, (df, label)), color in zip(data.items(), colors):
        if mu_col in df.columns:
            axes[1].plot(df["bin_x"], df[mu_col], marker="o", label=label,
                        alpha=0.7, linewidth=2, color=color)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel(variable)
    axes[1].set_ylabel(f"μ(d{component}) [cm]")
    axes[1].set_title(f"{component.upper()}-Bias vs {variable}")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    if xscale == "log":
        axes[1].set_xscale("log")

    plt.tight_layout()
    outfile = outdir / f"d{component}_vs_{variable}.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
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


def plot_pred_vs_true(data, variable, component, outdir):
    """Plot pred{component} as function of true{variable}.

    Shows predicted coordinate vs true binned coordinate.

    Args:
        component: 'x', 'y', or 'z'
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.tab10(range(len(data)))
    xscale = "log" if variable == "total_signal" else "linear"

    mu_col = f"d{component}_mu"

    # For pred vs true, we calculate pred from true + mu (since mu = pred - true)
    for (model_key, (df, label)), color in zip(data.items(), colors):
        if mu_col in df.columns:
            pred_vals = df["bin_x"] + df[mu_col]
            ax.plot(df["bin_x"], pred_vals, marker="o", label=label,
                   alpha=0.7, linewidth=2, color=color)

    # Add perfect prediction line (y=x)
    xlim = ax.get_xlim()
    ax.plot(xlim, xlim, 'k--', linewidth=1, alpha=0.5, label='Perfect')
    ax.set_xlim(xlim)

    ax.set_xlabel(variable)
    ax.set_ylabel(f"pred{component} [cm]")
    ax.set_title(f"Predicted vs True: {component.upper()} coordinate (binned by {variable})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if xscale == "log":
        ax.set_xscale("log")

    plt.tight_layout()
    outfile = outdir / f"pred{component}_vs_{variable}.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
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
    # 1. dr sigma & mu vs the variable (always generated)
    plot_dr_vs_spatial(data, variable, outdir)

    # 2. d{component} vs true{component} (only if variable matches a spatial coord)
    if variable in ["truex", "x"]:
        plot_residual_vs_true(data, variable, "x", outdir)
        plot_pred_vs_true(data, variable, "x", outdir)
    elif variable in ["truey", "y"]:
        plot_residual_vs_true(data, variable, "y", outdir)
        plot_pred_vs_true(data, variable, "y", outdir)
    elif variable in ["truez", "z"]:
        plot_residual_vs_true(data, variable, "z", outdir)
        plot_pred_vs_true(data, variable, "z", outdir)

    # 3. Summary table (always generated)
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
