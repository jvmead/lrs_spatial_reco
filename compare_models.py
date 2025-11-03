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


def load_predictions_with_quantiles(model_dir, variable, component):
    """
    Load raw predictions and compute quantile bands for pred vs true plot.

    Args:
        model_dir: Path to reconstruction output directory
        variable: Variable name (truex, truey, truez, total_signal)
        component: Component to predict ('x', 'y', or 'z')

    Returns:
        DataFrame with columns: bin_center, pred_median, pred_q16, pred_q84
        or None if data not available
    """
    pred_file = model_dir / "predictions" / "predictions.csv.gz"

    if not pred_file.exists():
        return None

    try:
        # Load raw predictions
        df = pd.read_csv(pred_file)

        # Check required columns exist
        true_col = f"true{component}"
        pred_col = f"pred{component}"

        # Check if truth columns are already in predictions file (GNN format)
        has_truth = true_col in df.columns and variable in df.columns

        if not has_truth:
            # Need to load and merge truth data (Analytic format)
            manifest = load_manifest(model_dir)
            input_csv = manifest.get("input_csv") or manifest.get("processed_csv")

            if not input_csv:
                return None

            # Resolve input CSV path
            input_path = Path(input_csv)
            if not input_path.is_absolute():
                if input_path.exists():
                    pass
                elif (model_dir.parent / input_csv).exists():
                    input_path = model_dir.parent / input_csv
                elif (model_dir.parent / input_path.name).exists():
                    input_path = model_dir.parent / input_path.name
                else:
                    parts = input_path.parts
                    if len(parts) > 1 and parts[0] == model_dir.parent.name:
                        input_path = model_dir.parent / parts[-1]

            if not input_path.exists():
                return None

            # Load input truth data
            needed_cols = [f"true{component}", variable]
            if variable != "total_signal":
                needed_cols.append("total_signal")
            df_truth = pd.read_csv(input_path, usecols=needed_cols)

            # Apply same filtering as reconstruction
            df_truth = df_truth[df_truth["total_signal"] > 0].reset_index(drop=True)

            # Merge
            if len(df) != len(df_truth):
                if len(df) < len(df_truth):
                    df_truth = df_truth.iloc[:len(df)].reset_index(drop=True)
                else:
                    return None

            df = pd.concat([df_truth.reset_index(drop=True),
                           df.reset_index(drop=True)], axis=1)

        # Verify columns
        if true_col not in df.columns or pred_col not in df.columns or variable not in df.columns:
            return None

        # Determine binning strategy
        if variable == "total_signal":
            var_data = df[variable].values
            log_var = np.log10(var_data + 1e-10)
            bins = np.linspace(log_var.min(), log_var.max(), 25)
            df['bin_idx'] = pd.cut(log_var, bins=bins, labels=False)
            bin_centers = 10 ** ((bins[:-1] + bins[1:]) / 2)
        else:
            var_data = df[variable].values
            bins = np.linspace(var_data.min(), var_data.max(), 25)
            df['bin_idx'] = pd.cut(var_data, bins=bins, labels=False)
            bin_centers = (bins[:-1] + bins[1:]) / 2

        # Compute quantiles per bin
        quantiles = df.groupby('bin_idx')[pred_col].quantile([0.16, 0.5, 0.84]).unstack()
        quantiles.columns = ['pred_q16', 'pred_median', 'pred_q84']
        quantiles['bin_center'] = bin_centers[:len(quantiles)]

        return quantiles.reset_index(drop=True)

    except Exception as e:
        return None


def load_residuals_with_quantiles(model_dir, variable, component):
    """
    Load raw predictions and compute residual quantile bands.

    Args:
        model_dir: Path to reconstruction output directory
        variable: Variable name (truex, truey, truez, total_signal)
        component: Component to compute residuals for ('x', 'y', or 'z')

    Returns:
        DataFrame with columns: bin_center, resid_median, resid_q16, resid_q84, sigma, mu
        or None if data not available
    """
    pred_file = model_dir / "predictions" / "predictions.csv.gz"

    if not pred_file.exists():
        return None

    try:
        # Load raw predictions
        df = pd.read_csv(pred_file)

        # Check required columns exist
        true_col = f"true{component}"
        pred_col = f"pred{component}"

        # Check if truth columns are already in predictions file (GNN format)
        has_truth = true_col in df.columns and variable in df.columns

        if not has_truth:
            # Need to load and merge truth data (Analytic format)
            manifest = load_manifest(model_dir)
            input_csv = manifest.get("input_csv") or manifest.get("processed_csv")

            if not input_csv:
                return None

            # Resolve input CSV path
            input_path = Path(input_csv)
            if not input_path.is_absolute():
                if input_path.exists():
                    pass
                elif (model_dir.parent / input_csv).exists():
                    input_path = model_dir.parent / input_csv
                elif (model_dir.parent / input_path.name).exists():
                    input_path = model_dir.parent / input_path.name
                else:
                    parts = input_path.parts
                    if len(parts) > 1 and parts[0] == model_dir.parent.name:
                        input_path = model_dir.parent / parts[-1]

            if not input_path.exists():
                return None

            # Load input truth data
            needed_cols = [f"true{component}", variable]
            if variable != "total_signal":
                needed_cols.append("total_signal")
            df_truth = pd.read_csv(input_path, usecols=needed_cols)

            # Apply same filtering as reconstruction
            df_truth = df_truth[df_truth["total_signal"] > 0].reset_index(drop=True)

            # Merge
            if len(df) != len(df_truth):
                if len(df) < len(df_truth):
                    df_truth = df_truth.iloc[:len(df)].reset_index(drop=True)
                else:
                    return None

            df = pd.concat([df_truth.reset_index(drop=True),
                           df.reset_index(drop=True)], axis=1)

        # Verify columns
        if true_col not in df.columns or pred_col not in df.columns or variable not in df.columns:
            return None

        # Compute residuals
        df[f'd{component}'] = df[pred_col] - df[true_col]

        # Determine binning strategy
        if variable == "total_signal":
            var_data = df[variable].values
            log_var = np.log10(var_data + 1e-10)
            bins = np.linspace(log_var.min(), log_var.max(), 25)
            df['bin_idx'] = pd.cut(log_var, bins=bins, labels=False)
            bin_centers = 10 ** ((bins[:-1] + bins[1:]) / 2)
        else:
            var_data = df[variable].values
            bins = np.linspace(var_data.min(), var_data.max(), 25)
            df['bin_idx'] = pd.cut(var_data, bins=bins, labels=False)
            bin_centers = (bins[:-1] + bins[1:]) / 2

        # Compute quantiles AND mean/std per bin
        resid_col = f'd{component}'
        grouped = df.groupby('bin_idx')[resid_col]

        quantiles = grouped.quantile([0.16, 0.5, 0.84]).unstack()
        quantiles.columns = ['resid_q16', 'resid_median', 'resid_q84']

        # Add mean and std for comparison with existing plots
        quantiles['mu'] = grouped.mean().values
        quantiles['sigma'] = grouped.std().values
        quantiles['count'] = grouped.count().values  # Number of samples per bin
        quantiles['bin_center'] = bin_centers[:len(quantiles)]

        return quantiles.reset_index(drop=True)

    except Exception as e:
        return None


def load_dr_residuals_with_quantiles(model_dir, variable):
    """
    Load raw predictions and compute 3D residual (dr) quantile bands.

    Args:
        model_dir: Path to reconstruction output directory
        variable: Variable name (truex, truey, truez, total_signal)

    Returns:
        DataFrame with columns: bin_center, dr_median, dr_q16, dr_q84, sigma, mu
        or None if data not available
    """
    pred_file = model_dir / "predictions" / "predictions.csv.gz"

    if not pred_file.exists():
        return None

    try:
        # Load raw predictions
        df = pd.read_csv(pred_file)

        # Check if truth columns are already in predictions file (GNN format)
        has_truth = all(col in df.columns for col in ['truex', 'truey', 'truez', variable])

        if not has_truth:
            # Need to load and merge truth data (Analytic format)
            manifest = load_manifest(model_dir)
            input_csv = manifest.get("input_csv") or manifest.get("processed_csv")

            if not input_csv:
                return None

            # Resolve input CSV path
            input_path = Path(input_csv)
            if not input_path.is_absolute():
                if input_path.exists():
                    pass
                elif (model_dir.parent / input_csv).exists():
                    input_path = model_dir.parent / input_csv
                elif (model_dir.parent / input_path.name).exists():
                    input_path = model_dir.parent / input_path.name
                else:
                    parts = input_path.parts
                    if len(parts) > 1 and parts[0] == model_dir.parent.name:
                        input_path = model_dir.parent / parts[-1]

            if not input_path.exists():
                return None

            # Load input truth data
            needed_cols = ['truex', 'truey', 'truez', variable]
            if variable != "total_signal":
                needed_cols.append("total_signal")
            df_truth = pd.read_csv(input_path, usecols=needed_cols)

            # Apply same filtering as reconstruction
            df_truth = df_truth[df_truth["total_signal"] > 0].reset_index(drop=True)

            # Merge
            if len(df) != len(df_truth):
                if len(df) < len(df_truth):
                    df_truth = df_truth.iloc[:len(df)].reset_index(drop=True)
                else:
                    return None

            df = pd.concat([df_truth.reset_index(drop=True),
                           df.reset_index(drop=True)], axis=1)

        # Verify columns
        required = ['truex', 'truey', 'truez', 'predx', 'predy', 'predz', variable]
        if not all(col in df.columns for col in required):
            return None

        # Compute 3D residuals
        df['dx'] = df['predx'] - df['truex']
        df['dy'] = df['predy'] - df['truey']
        df['dz'] = df['predz'] - df['truez']
        df['dr'] = np.sqrt(df['dx']**2 + df['dy']**2 + df['dz']**2)

        # Determine binning strategy
        if variable == "total_signal":
            var_data = df[variable].values
            log_var = np.log10(var_data + 1e-10)
            bins = np.linspace(log_var.min(), log_var.max(), 25)
            df['bin_idx'] = pd.cut(log_var, bins=bins, labels=False)
            bin_centers = 10 ** ((bins[:-1] + bins[1:]) / 2)
        else:
            var_data = df[variable].values
            bins = np.linspace(var_data.min(), var_data.max(), 25)
            df['bin_idx'] = pd.cut(var_data, bins=bins, labels=False)
            bin_centers = (bins[:-1] + bins[1:]) / 2

        # Compute quantiles AND mean/std per bin for dr
        grouped = df.groupby('bin_idx')['dr']

        quantiles = grouped.quantile([0.16, 0.5, 0.84]).unstack()
        quantiles.columns = ['dr_q16', 'dr_median', 'dr_q84']

        # Add mean and std
        quantiles['mu'] = grouped.mean().values
        quantiles['sigma'] = grouped.std().values
        quantiles['count'] = grouped.count().values  # Number of samples per bin
        quantiles['bin_center'] = bin_centers[:len(quantiles)]

        return quantiles.reset_index(drop=True)

    except Exception as e:
        return None


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

def plot_dr_vs_spatial(data, variable, outdir, model_runs):
    """Plot dr sigma and mu as function of true x, y, z with quantile bands."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.tab10(range(len(data)))
    xscale = "log" if variable == "total_signal" else "linear"

    # Sigma plot
    for (model_key, (df, label)), color in zip(data.items(), colors):
        # Load raw data statistics first to get uncertainty bands
        quantile_df = load_dr_residuals_with_quantiles(model_runs[model_key], variable)
        if quantile_df is not None:
            x = quantile_df['bin_center']
            sigma = quantile_df['sigma']
            count = quantile_df['count']
            # Standard error of standard deviation: sigma / sqrt(2*N)
            sigma_error = sigma / np.sqrt(2 * count)
            # Plot sigma with uncertainty band centered on it
            axes[0].plot(x, sigma, marker="o", label=label,
                        alpha=0.7, linewidth=2, color=color)
            axes[0].fill_between(x, sigma - sigma_error, sigma + sigma_error,
                               alpha=0.2, color=color)
        else:
            # Fallback to binned statistics if raw data unavailable
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

        # Add quantile bands if available
        quantile_df = load_dr_residuals_with_quantiles(model_runs[model_key], variable)
        if quantile_df is not None:
            x = quantile_df['bin_center']
            median = quantile_df['dr_median']
            q16 = quantile_df['dr_q16']
            q84 = quantile_df['dr_q84']
            axes[1].fill_between(x, q16, q84, alpha=0.2, color=color)

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


def plot_residual_vs_true(data, variable, component, outdir, model_runs):
    """Plot d{component} sigma and mu as function of true{component} with quantile bands.

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
        # Load raw data statistics first to get uncertainty bands
        quantile_df = load_residuals_with_quantiles(model_runs[model_key], variable, component)
        if quantile_df is not None:
            x = quantile_df['bin_center']
            sigma = quantile_df['sigma']
            count = quantile_df['count']
            # Standard error of standard deviation: sigma / sqrt(2*N)
            sigma_error = sigma / np.sqrt(2 * count)
            # Plot sigma with uncertainty band centered on it
            axes[0].plot(x, sigma, marker="o", label=label,
                        alpha=0.7, linewidth=2, color=color)
            axes[0].fill_between(x, sigma - sigma_error, sigma + sigma_error,
                               alpha=0.2, color=color)
        elif sigma_col in df.columns:
            # Fallback to binned statistics if raw data unavailable
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

            # Add quantile bands if available
            quantile_df = load_residuals_with_quantiles(model_runs[model_key], variable, component)
            if quantile_df is not None:
                x = quantile_df['bin_center']
                q16 = quantile_df['resid_q16']
                q84 = quantile_df['resid_q84']
                axes[1].fill_between(x, q16, q84, alpha=0.2, color=color)

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

    # Print summary to console
    print(f"\nSummary Statistics for {variable}:")
    print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Saved: {outfile}")


def plot_pred_vs_true(data, variable, component, outdir, model_runs):
    """Plot pred{component} as function of true{variable} with quantile bands.

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

            # Add quantile bands if available
            quantile_df = load_predictions_with_quantiles(model_runs[model_key], variable, component)
            if quantile_df is not None:
                x = quantile_df['bin_center']
                q16 = quantile_df['pred_q16']
                q84 = quantile_df['pred_q84']
                ax.fill_between(x, q16, q84, alpha=0.2, color=color)

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
        model_runs: Dictionary mapping model key to output directory Path
        variable: Independent variable name
        outdir: Output directory
    """
    print(f"\n=== Comparing models for variable: {variable} ===")

    # Load data from all models
    data = {}
    for model_key, model_dir in model_runs.items():
        manifest = load_manifest(model_dir)
        label = get_model_label(model_dir, manifest)
        df = load_1d_residuals(model_dir, variable)

        if df is not None:
            data[model_key] = (df, label)
            print(f"  Loaded: {label} ({len(df)} bins)")

    if not data:
        print(f"  No data found for variable: {variable}")
        return

    # Generate plots
    # 1. dr sigma & mu vs the variable (always generated)
    plot_dr_vs_spatial(data, variable, outdir, model_runs)

    # 2. d{component} sigma & mu vs variable for ALL components (always generated)
    for component in ['x', 'y', 'z']:
        plot_residual_vs_true(data, variable, component, outdir, model_runs)

    # 3. pred{component} vs true{component} (only if variable matches a spatial coord)
    if variable in ["truex", "x"]:
        plot_pred_vs_true(data, variable, "x", outdir, model_runs)
    elif variable in ["truey", "y"]:
        plot_pred_vs_true(data, variable, "y", outdir, model_runs)
    elif variable in ["truez", "z"]:
        plot_pred_vs_true(data, variable, "z", outdir, model_runs)

    # 4. Summary table (always generated)
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
        default=["truex", "truey", "truez", "total_signal"],
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
    model_runs_grouped = find_model_runs(directories)

    if not model_runs_grouped:
        print("Error: No reconstruction output directories found")
        print("Looking for directories with residuals_1d/ and manifest.json")
        return 1

    # Flatten to single dict with unique keys for each model run
    model_runs = {}
    for model_type, dirs in model_runs_grouped.items():
        for model_dir in dirs:
            key = f"{model_type}_{model_dir.name}"
            model_runs[key] = model_dir

    print(f"\nFound {len(model_runs)} model runs:")
    for key, model_dir in model_runs.items():
        manifest = load_manifest(model_dir)
        label = get_model_label(model_dir, manifest)
        print(f"  {key}: {label}")

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
