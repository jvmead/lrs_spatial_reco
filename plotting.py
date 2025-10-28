"""
plotting.py - Shared plotting and diagnostics for DUNE LRS spatial reco

Contains all diagnostic/statistics/plotting functions previously in utils.py.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from utils import transform_arrays

def plot_spatial_distributions(
    df: pd.DataFrame,
    x_true_col: str,
    y_true_col: str,
    z_true_col: str,
    x_pred_col: Optional[str] = None,
    y_pred_col: Optional[str] = None,
    z_pred_col: Optional[str] = None,
    tpc_bounds_mm: Optional[np.ndarray] = None,
    include_log_signal: bool = True,
    out_file: Optional[Path] = None,
    title_prefix: str = "",
    label_true: str = "true",
    label_pred: str = "pred",
) -> None:
    """
    Plot spatial (x, y, z) distributions + optional log(total_signal).

    Args:
        df: DataFrame with position columns
        x_true_col, y_true_col, z_true_col: Column names for true positions
        x_pred_col, y_pred_col, z_pred_col: Optional column names for predicted positions
        tpc_bounds_mm: Optional TPC bounds array (n_tpc, 2, 3) for vertical lines
        include_log_signal: If True, add 4th subplot with log10(total_signal)
        out_file: If provided, save figure to this path
        title_prefix: Prefix for subplot titles (e.g., "True" or "Pred vs True")
        label_true: Label for first histogram (default "true", use "raw" for input distributions)
        label_pred: Label for second histogram (default "pred", use "tsfm" for input distributions)
    """
    # Check if total_signal exists before deciding number of subplots
    has_total_signal = "total_signal" in df.columns
    n_plots = 4 if (include_log_signal and has_total_signal) else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 3))

    # X histogram
    xbins = np.linspace(-75, 75, 150)
    axes[0].hist(df[x_true_col], bins=xbins, alpha=0.5, color="red", label=label_true)
    if x_pred_col:
        axes[0].hist(df[x_pred_col], bins=xbins, alpha=0.5, color="orange", label=label_pred)
    axes[0].set_xlabel("x [cm]")
    if tpc_bounds_mm is not None:
        for bounds in tpc_bounds_mm:
            axes[0].axvline(bounds[0][0], color="k", linestyle="--", alpha=0.5)
            axes[0].axvline(bounds[1][0], color="k", linestyle="--", alpha=0.5)
    axes[0].legend()

    # Y histogram
    ybins = np.linspace(-75, 75, 150)
    axes[1].hist(df[y_true_col], bins=ybins, alpha=0.5, color="green", label=label_true)
    if y_pred_col:
        axes[1].hist(df[y_pred_col], bins=ybins, alpha=0.5, color="lightgreen", label=label_pred)
    axes[1].set_xlabel("y [cm]")
    if tpc_bounds_mm is not None:
        for bounds in tpc_bounds_mm:
            axes[1].axvline(bounds[0][1], color="k", linestyle="--", alpha=0.5)
            axes[1].axvline(bounds[1][1], color="k", linestyle="--", alpha=0.5)
    axes[1].legend()

    # Z histogram
    zbins = np.linspace(-75, 75, 150)
    axes[2].hist(df[z_true_col], bins=zbins, alpha=0.5, color="blue", label=label_true)
    if z_pred_col:
        axes[2].hist(df[z_pred_col], bins=zbins, alpha=0.5, color="cyan", label=label_pred)
    axes[2].set_xlabel("z [cm]")
    if tpc_bounds_mm is not None:
        for bounds in tpc_bounds_mm:
            axes[2].axvline(bounds[0][2], color="k", linestyle="--", alpha=0.5)
            axes[2].axvline(bounds[1][2], color="k", linestyle="--", alpha=0.5)
    axes[2].legend()

    # Optional log(total_signal) histogram (only if column exists and requested)
    if n_plots == 4:
        log_signal = np.log10(df["total_signal"].values[df["total_signal"] > 0])
        axes[3].hist(log_signal, bins=50, alpha=0.5, color="purple", label="total_signal")
        axes[3].set_xlabel("log10(total_signal) [PE/sample]")
        axes[3].legend()

    plt.tight_layout()
    if out_file:
        fig.savefig(out_file)
        plt.close(fig)
    else:
        plt.show()

def _edges_log10(indep: np.ndarray, bins: int) -> np.ndarray:
    finite_pos = np.isfinite(indep) & (indep > 0)
    if finite_pos.sum() == 0:
        finite = np.isfinite(indep)
        if finite.sum() == 0:
            return np.array([1.0, 10.0])
        mn = float(np.nanmin(indep[finite])); mx = float(np.nanmax(indep[finite]))
        return np.linspace(mn, mx, bins + 1)
    lo = float(indep[finite_pos].min()); hi = float(indep[finite_pos].max())
    if lo == hi:
        return np.array([lo, hi])
    return np.logspace(np.log10(lo), np.log10(hi), bins + 1)

def _edges_linear(indep: np.ndarray, bins: int) -> np.ndarray:
    finite = np.isfinite(indep)
    if finite.sum() == 0:
        return np.array([0.0, 1.0])
    lo = float(np.nanmin(indep[finite])); hi = float(np.nanmax(indep[finite]))
    if lo == hi:
        return np.array([lo, hi])
    return np.linspace(lo, hi, bins + 1)

def compute_differential_stats_1d_minimal(
    df_pred: pd.DataFrame,
    var: str,
    bins: int = 24,
) -> pd.DataFrame:
    """
    Returns:
      bin_x, dx_mu, dx_sig, dy_mu, dy_sig, dz_mu, dz_sig, dr_mu, dr_sig,
      dx_mu_err, dy_mu_err, dz_mu_err, dr_mu_err,
      dx_sig_err, dy_sig_err, dz_sig_err, dr_sig_err, n_per_bin
    Uses LOG10-SPACED bins when var == 'total_signal'.
    Errors: mu_err = sig/sqrt(n), sig_err = sig/sqrt(2n)
    """
    dx = (df_pred["predx"] - df_pred["truex"]).to_numpy(dtype=float)
    dy = (df_pred["predy"] - df_pred["truey"]).to_numpy(dtype=float)
    dz = (df_pred["predz"] - df_pred["truez"]).to_numpy(dtype=float)
    dr = np.sqrt(dx**2 + dy**2 + dz**2)
    if var == "dr":
        indep = dr
    elif var == "r":
        indep = np.sqrt(df_pred["truex"]**2 + df_pred["truey"]**2 + df_pred["truez"]**2).to_numpy(dtype=float)
    elif var in df_pred.columns:
        indep = df_pred[var].to_numpy(dtype=float)
    elif var == "tot_signal":
        indep = df_pred["total_signal"].to_numpy(dtype=float)
        var = "total_signal"
    else:
        raise ValueError(f"Unknown independent variable '{var}'")
    edges = _edges_log10(indep, bins) if var == "total_signal" else _edges_linear(indep, bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ib = np.clip(np.digitize(indep, edges) - 1, 0, len(edges) - 2)
    def moments_per_bin(vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = np.zeros(len(edges) - 1); s2 = np.zeros(len(edges) - 1); n = np.zeros(len(edges) - 1, dtype=int)
        for k, v in enumerate(vals):
            if not np.isfinite(v): continue
            i = ib[k]; s[i] += v; s2[i] += v*v; n[i] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(n > 0, s / n, np.nan)
            varr = np.where(n > 0, s2 / n - mu * mu, np.nan)
            sig = np.sqrt(np.where(varr >= 0, varr, np.nan))
            # Standard error of mean: sig/sqrt(n)
            mu_err = np.where(n > 0, sig / np.sqrt(n), np.nan)
            # Standard error of std dev: sig/sqrt(2n)
            sig_err = np.where(n > 0, sig / np.sqrt(2.0 * n), np.nan)
        return mu, sig, mu_err, sig_err, n
    dx_mu, dx_sig, dx_mu_err, dx_sig_err, n_dx = moments_per_bin(dx)
    dy_mu, dy_sig, dy_mu_err, dy_sig_err, n_dy = moments_per_bin(dy)
    dz_mu, dz_sig, dz_mu_err, dz_sig_err, n_dz = moments_per_bin(dz)
    dr_mu, dr_sig, dr_mu_err, dr_sig_err, n_dr = moments_per_bin(dr)
    return pd.DataFrame({
        "bin_x": centers.astype(float),
        "dx_mu": dx_mu, "dx_sig": dx_sig,
        "dy_mu": dy_mu, "dy_sig": dy_sig,
        "dz_mu": dz_mu, "dz_sig": dz_sig,
        "dr_mu": dr_mu, "dr_sig": dr_sig,
        "dx_mu_err": dx_mu_err, "dx_sig_err": dx_sig_err,
        "dy_mu_err": dy_mu_err, "dy_sig_err": dy_sig_err,
        "dz_mu_err": dz_mu_err, "dz_sig_err": dz_sig_err,
        "dr_mu_err": dr_mu_err, "dr_sig_err": dr_sig_err,
        "n_per_bin": n_dx,  # all should be the same
    })

def plot_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    xrange: np.ndarray,
    yrange: np.ndarray,
    figsize=(6, 5),
    cmap="viridis",
    x_log: bool = False,
    show=False,
    return_fig=False,
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    h = ax.hist2d(df[x_col], df[y_col], bins=[xrange, yrange], cmap=cmap)
    plt.colorbar(h[3], ax=ax, label="Counts")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if x_log:
        ax.set_xscale("log")
    x_left_edges = xrange[:-1]
    y_upper, y_centre, y_lower = [], [], []
    for j in range(len(xrange) - 1):
        mask = (df[x_col] >= xrange[j]) & (df[x_col] < xrange[j + 1])
        yvals = df.loc[mask, y_col]
        if len(yvals) > 0:
            y_upper.append(np.nanquantile(yvals, 0.84))
            y_centre.append(np.nanquantile(yvals, 0.5))
            y_lower.append(np.nanquantile(yvals, 0.16))
        else:
            y_upper.append(np.nan); y_centre.append(np.nan); y_lower.append(np.nan)
    ax.step(x_left_edges, y_upper, color="red", linestyle="--", linewidth=1.2, where="post", label="68% band")
    ax.step(x_left_edges, y_centre, color="red", linestyle=":",  linewidth=1.2, where="post", label="Median")
    ax.step(x_left_edges, y_lower, color="red", linestyle="--", linewidth=1.2, where="post")
    plt.tight_layout()
    if show:
        plt.show()
    if return_fig:
        return fig, ax
    return None

def plot_heatmaps_resid_vs_vars(
    df_pred: pd.DataFrame, outdir: Path, vars_list: list, bins_map: dict
) -> None:
    dx = (df_pred["predx"] - df_pred["truex"]).astype(float)
    dy = (df_pred["predy"] - df_pred["truey"]).astype(float)
    dz = (df_pred["predz"] - df_pred["truez"]).astype(float)
    dr = np.sqrt(dx**2 + dy**2 + dz**2)
    r_true = np.sqrt(df_pred["truex"]**2 + df_pred["truey"]**2 + df_pred["truez"]**2).astype(float)
    df_diag = pd.DataFrame({"dx": dx, "dy": dy, "dz": dz, "dr": dr, "r": r_true})
    for var in vars_list:
        if var == "r":
            x_vals = r_true.values
        elif var == "total_signal":
            x_vals = df_pred["total_signal"].to_numpy(dtype=float)
        else:
            x_vals = df_pred[var].to_numpy(dtype=float)
        bins = bins_map.get(var, 24)
        if var == "total_signal":
            x_edges = _edges_log10(x_vals, bins) if np.any(x_vals>0) else np.linspace(1,10,bins+1)
            x_log = True
        else:
            x_edges = _edges_linear(x_vals, bins) if len(x_vals)>0 else np.linspace(0,1,bins+1)
            x_log = False
        for resid in ["dx", "dy", "dz", "dr"]:
            yy = df_diag[resid].to_numpy(dtype=float)
            finite = np.isfinite(yy)
            if finite.sum() == 0:
                y_edges = np.linspace(-1.0, 1.0, 60)
            else:
                if resid == "dr":
                    # dr is always non-negative, set minimum to 0
                    q84 = np.nanquantile(yy[finite], 0.84)
                    hi = 1.5 * q84
                    if hi == 0:
                        hi = 1.0
                    y_edges = np.linspace(0, hi, 60)
                else:
                    q16, q84 = np.nanquantile(yy[finite], [0.16, 0.84])
                    span = max(abs(q16), abs(q84))
                    lo, hi = -1.5 * span, 1.5 * span
                    if lo == hi:
                        lo, hi = lo - 1.0, hi + 1.0
                    y_edges = np.linspace(lo, hi, 60)
            tmp = pd.DataFrame({var: x_vals, resid: df_diag[resid]})
            fig_ax = plot_heatmap(
                tmp,
                x_col=var,
                y_col=resid,
                title=f"{resid} vs {var}",
                xlabel=var,
                ylabel=resid,
                xrange=x_edges,
                yrange=y_edges,
                figsize=(7, 5),
                cmap="viridis",
                x_log=x_log,
                show=False,
                return_fig=True,
            )
            fig, ax = fig_ax if fig_ax is not None else (None, None)
            if fig is not None:
                fig.savefig(outdir / f"heatmap_{resid}_vs_{var}.png", dpi=150)
                plt.close(fig)

def plot_pred_vs_true_xyz(df_pred: pd.DataFrame, outdir: Path, bins_map: dict) -> None:
    pairs = [
        ("truex", "predx", "x"),
        ("truey", "predy", "y"),
        ("truez", "predz", "z"),
    ]
    for tcol, pcol, label in pairs:
        bins = bins_map.get(f"true{label}", 24)
        x_true = df_pred[tcol].to_numpy(dtype=float)
        x_pred = df_pred[pcol].to_numpy(dtype=float)
        edges = _edges_linear(x_true, bins) if len(x_true)>0 else np.linspace(0,1,bins+1)
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))
        h = ax.hist2d(x_true, x_pred, bins=[edges, edges], cmap="viridis")
        plt.colorbar(h[3], ax=ax, label="Counts")
        ax.set_title(f"pred{label} vs true{label}")
        ax.set_xlabel(f"true{label}")
        ax.set_ylabel(f"pred{label}")
        lo = float(min(edges[0], edges[-1])); hi = float(max(edges[0], edges[-1]))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="white", linewidth=1.2, label="y = x")
        ax.legend(loc="best", framealpha=0.6)
        plt.tight_layout()
        fig.savefig(outdir / f"heatmap_pred{label}_vs_true{label}.png", dpi=150)
        plt.close(fig)

def plot_1d_curves(df1d: pd.DataFrame, outdir: Path, var: str, error_style: str = "band") -> None:
    """
    Plot 1D residual curves (mean and sigma) vs an independent variable.

    Args:
        df1d: DataFrame with binned statistics
        outdir: Output directory for plots
        var: Independent variable name
        error_style: "band" for fill_between (default), "errorbar" for error bars, "none" for no errors
    """
    xscale = "log" if var in ("total_signal", "log_total_signal") else "linear"
    if var in ("total_signal", "log_total_signal"):
        print(f"[plot_1d_curves] xscale for {var}: log")
    else:
        print(f"[plot_1d_curves] xscale for {var}: linear")

    # Check if error columns exist
    has_errors = all(col in df1d.columns for col in ["dx_mu_err", "dy_mu_err", "dz_mu_err", "dr_mu_err"])

    # Plot mu (mean residuals)
    fig = plt.figure(figsize=(7, 4))
    if has_errors and error_style == "errorbar":
        plt.errorbar(df1d["bin_x"], df1d["dx_mu"], yerr=df1d["dx_mu_err"], label="dx", fmt='o-', capsize=3, markersize=4)
        plt.errorbar(df1d["bin_x"], df1d["dy_mu"], yerr=df1d["dy_mu_err"], label="dy", fmt='s-', capsize=3, markersize=4)
        plt.errorbar(df1d["bin_x"], df1d["dz_mu"], yerr=df1d["dz_mu_err"], label="dz", fmt='^-', capsize=3, markersize=4)
        plt.errorbar(df1d["bin_x"], df1d["dr_mu"], yerr=df1d["dr_mu_err"], label="dr", fmt='d-', capsize=3, markersize=4)
    elif has_errors and error_style == "band":
        # dx
        plt.plot(df1d["bin_x"], df1d["dx_mu"], label="dx", linewidth=2)
        plt.fill_between(df1d["bin_x"], df1d["dx_mu"] - df1d["dx_mu_err"], df1d["dx_mu"] + df1d["dx_mu_err"], alpha=0.3)
        # dy
        plt.plot(df1d["bin_x"], df1d["dy_mu"], label="dy", linewidth=2)
        plt.fill_between(df1d["bin_x"], df1d["dy_mu"] - df1d["dy_mu_err"], df1d["dy_mu"] + df1d["dy_mu_err"], alpha=0.3)
        # dz
        plt.plot(df1d["bin_x"], df1d["dz_mu"], label="dz", linewidth=2)
        plt.fill_between(df1d["bin_x"], df1d["dz_mu"] - df1d["dz_mu_err"], df1d["dz_mu"] + df1d["dz_mu_err"], alpha=0.3)
        # dr
        plt.plot(df1d["bin_x"], df1d["dr_mu"], label="dr", linewidth=2)
        plt.fill_between(df1d["bin_x"], df1d["dr_mu"] - df1d["dr_mu_err"], df1d["dr_mu"] + df1d["dr_mu_err"], alpha=0.3)
    else:
        plt.plot(df1d["bin_x"], df1d["dx_mu"], label="dx")
        plt.plot(df1d["bin_x"], df1d["dy_mu"], label="dy")
        plt.plot(df1d["bin_x"], df1d["dz_mu"], label="dz")
        plt.plot(df1d["bin_x"], df1d["dr_mu"], label="dr")
    plt.xlabel(var); plt.ylabel("μ (cm)")
    plt.title(f"Residual μ vs {var}")
    plt.legend(); plt.grid(True, alpha=0.3); fig.tight_layout()
    if xscale == "log":
        plt.xscale("log")
    plt.savefig(outdir / f"plot_1d_{var}_mu.png", dpi=150); plt.close(fig)

    # Plot sigma (standard deviations)
    fig = plt.figure(figsize=(7, 4))
    if has_errors and error_style == "errorbar":
        plt.errorbar(df1d["bin_x"], df1d["dx_sig"], yerr=df1d["dx_sig_err"], label="dx", fmt='o-', capsize=3, markersize=4)
        plt.errorbar(df1d["bin_x"], df1d["dy_sig"], yerr=df1d["dy_sig_err"], label="dy", fmt='s-', capsize=3, markersize=4)
        plt.errorbar(df1d["bin_x"], df1d["dz_sig"], yerr=df1d["dz_sig_err"], label="dz", fmt='^-', capsize=3, markersize=4)
        plt.errorbar(df1d["bin_x"], df1d["dr_sig"], yerr=df1d["dr_sig_err"], label="dr", fmt='d-', capsize=3, markersize=4)
    elif has_errors and error_style == "band":
        # dx
        plt.plot(df1d["bin_x"], df1d["dx_sig"], label="dx", linewidth=2)
        plt.fill_between(df1d["bin_x"], df1d["dx_sig"] - df1d["dx_sig_err"], df1d["dx_sig"] + df1d["dx_sig_err"], alpha=0.3)
        # dy
        plt.plot(df1d["bin_x"], df1d["dy_sig"], label="dy", linewidth=2)
        plt.fill_between(df1d["bin_x"], df1d["dy_sig"] - df1d["dy_sig_err"], df1d["dy_sig"] + df1d["dy_sig_err"], alpha=0.3)
        # dz
        plt.plot(df1d["bin_x"], df1d["dz_sig"], label="dz", linewidth=2)
        plt.fill_between(df1d["bin_x"], df1d["dz_sig"] - df1d["dz_sig_err"], df1d["dz_sig"] + df1d["dz_sig_err"], alpha=0.3)
        # dr
        plt.plot(df1d["bin_x"], df1d["dr_sig"], label="dr", linewidth=2)
        plt.fill_between(df1d["bin_x"], df1d["dr_sig"] - df1d["dr_sig_err"], df1d["dr_sig"] + df1d["dr_sig_err"], alpha=0.3)
    else:
        plt.plot(df1d["bin_x"], df1d["dx_sig"], label="dx")
        plt.plot(df1d["bin_x"], df1d["dy_sig"], label="dy")
        plt.plot(df1d["bin_x"], df1d["dz_sig"], label="dz")
        plt.plot(df1d["bin_x"], df1d["dr_sig"], label="dr")
    plt.xlabel(var); plt.ylabel("σ (cm)")
    plt.title(f"Residual σ vs {var}")
    # Set y-axis minimum to 0 for dr sigma (dr is always >= 0)
    current_ylim = plt.gca().get_ylim()
    plt.ylim(0, current_ylim[1])
    plt.legend(); plt.grid(True, alpha=0.3); fig.tight_layout()
    if xscale == "log":
        plt.xscale("log")
    plt.savefig(outdir / f"plot_1d_{var}_sig.png", dpi=150); plt.close(fig)


def plot_3d_event_display(
    ax: plt.Axes,
    event: pd.Series,
    df_geom: pd.DataFrame,
    tpc_bounds_mm: np.ndarray,
    align_params,
    show_pred: bool = True,
    title: str = "",
) -> None:
    """
    Plot 3D event display with detectors, TPC boxes, and true/predicted positions.

    Args:
        ax: matplotlib 3D axis
        event: Single event row with det_#_max, truex/truey/truez, predx/predy/predz
        df_geom: Detector geometry DataFrame with transformed positions
        tpc_bounds_mm: TPC bounds array (n_tpc, 2, 3)
        align_params: AlignParams for coordinate transformation
        show_pred: Whether to show predicted position (False for preprocessing)
        title: Subplot title
    """
    # Color palette for TPCs
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Get event's TPC number to filter detectors
    event_tpc = int(event.get('tpc_num', -1)) if 'tpc_num' in event.index else -1

    # Filter detector geometry to only show detectors from this event's TPC
    if event_tpc >= 0:
        df_geom_filtered = df_geom[df_geom['TPC'] == event_tpc]
    else:
        df_geom_filtered = df_geom

    # Get detector signals for this event and scale marker sizes (only from event's TPC)
    det_signals = []
    det_positions = []
    for det_id in range(16):
        col = f"det_{det_id}_max"
        if col in event.index:
            signal = float(event[col])
            # Get transformed detector position from geometry (filtered to event's TPC)
            det_row = df_geom_filtered[df_geom_filtered['Detector'] == det_id]
            if not det_row.empty:
                det_signals.append(signal)
                det_positions.append([
                    float(det_row['x_offset'].iloc[0]),
                    float(det_row['y_offset'].iloc[0]),
                    float(det_row['z_offset'].iloc[0])
                ])
    det_signals = np.array(det_signals)
    det_positions = np.array(det_positions)

    # Scale marker sizes: min size 1 for zero signal, others scaled accordingly
    size_min, size_max = 1, 300
    if len(det_signals) > 0:
        if det_signals.max() > 0:
            sizes = size_min + (det_signals / det_signals.max()) * (size_max - size_min)
            sizes[det_signals == 0] = size_min
        else:
            sizes = np.full_like(det_signals, size_min)
    else:
        sizes = np.array([])

    # Plot detectors
    if len(det_positions) > 0:
        # Plot actual detectors (variable size)
        scatter_det = ax.scatter(det_positions[:, 0], det_positions[:, 1], det_positions[:, 2],
                   s=sizes, c='lightgray', alpha=0.7, edgecolors='black', linewidths=0.5,
                   label='_nolegend_', depthshade=False)
        # Add a fixed-size legend marker for detectors
        ax.scatter([], [], [], s=60, c='lightgray', edgecolors='black', linewidths=0.5, label='Detectors')

    # Draw TPC boxes with transformed corners
    edges_idx = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]

    for tpc_idx, bounds in enumerate(tpc_bounds_mm):
        color = colors[tpc_idx % len(colors)]
        x_min, y_min, z_min = bounds[0]
        x_max, y_max, z_max = bounds[1]

        # Define 8 corners
        corners = np.array([
            [x_min, y_min, z_min], [x_min, y_min, z_max],
            [x_min, y_max, z_min], [x_min, y_max, z_max],
            [x_max, y_min, z_min], [x_max, y_min, z_max],
            [x_max, y_max, z_min], [x_max, y_max, z_max],
        ])

        # Transform corners
        x_transformed, y_transformed, z_transformed = transform_arrays(
            corners[:, 0], corners[:, 1], corners[:, 2], align_params
        )
        transformed_corners = np.column_stack([x_transformed, y_transformed, z_transformed])

        # Draw edges
        for i, j in edges_idx:
            ax.plot([transformed_corners[i, 0], transformed_corners[j, 0]],
                   [transformed_corners[i, 1], transformed_corners[j, 1]],
                   [transformed_corners[i, 2], transformed_corners[j, 2]],
                   color=color, alpha=0.6, linewidth=1.5)

    # Compute axis limits from ALL transformed TPC corners (not just min/max)
    all_transformed_x = []
    all_transformed_y = []
    all_transformed_z = []
    for bounds in tpc_bounds_mm:
        x_min, y_min, z_min = bounds[0]
        x_max, y_max, z_max = bounds[1]
        # Transform all 8 corners
        corners = np.array([
            [x_min, y_min, z_min], [x_min, y_min, z_max],
            [x_min, y_max, z_min], [x_min, y_max, z_max],
            [x_max, y_min, z_min], [x_max, y_min, z_max],
            [x_max, y_max, z_min], [x_max, y_max, z_max],
        ])
        x_t, y_t, z_t = transform_arrays(
            corners[:, 0], corners[:, 1], corners[:, 2], align_params
        )
        all_transformed_x.extend(x_t)
        all_transformed_y.extend(y_t)
        all_transformed_z.extend(z_t)
    # Also include the transformed true position in axis limits
    truex, truey, truez = float(event['truex']), float(event['truey']), float(event['truez'])
    truex_t, truey_t, truez_t = transform_arrays(np.array([truex]), np.array([truey]), np.array([truez]), align_params)
    all_transformed_x.append(truex_t[0])
    all_transformed_y.append(truey_t[0])
    all_transformed_z.append(truez_t[0])
    xlim = [min(all_transformed_x), max(all_transformed_x)]
    ylim = [min(all_transformed_y), max(all_transformed_y)]
    zlim = [min(all_transformed_z), max(all_transformed_z)]

    # Get true position for axis lines
    true_pos = [float(event['truex']), float(event['truey']), float(event['truez'])]

    # True position axis lines to boundaries (plot BEFORE markers)
    ax.plot([true_pos[0], xlim[0]], [true_pos[1], true_pos[1]], [true_pos[2], true_pos[2]],
           'b:', linewidth=1, alpha=0.5)
    ax.plot([true_pos[0], xlim[1]], [true_pos[1], true_pos[1]], [true_pos[2], true_pos[2]],
           'b:', linewidth=1, alpha=0.5)
    ax.plot([true_pos[0], true_pos[0]], [true_pos[1], ylim[0]], [true_pos[2], true_pos[2]],
           'b:', linewidth=1, alpha=0.5)
    ax.plot([true_pos[0], true_pos[0]], [true_pos[1], ylim[1]], [true_pos[2], true_pos[2]],
           'b:', linewidth=1, alpha=0.5)
    ax.plot([true_pos[0], true_pos[0]], [true_pos[1], true_pos[1]], [true_pos[2], zlim[0]],
           'b:', linewidth=1, alpha=0.5)
    ax.plot([true_pos[0], true_pos[0]], [true_pos[1], true_pos[1]], [true_pos[2], zlim[1]],
           'b:', linewidth=1, alpha=0.5)

    # Plot predicted position axis lines if requested (BEFORE markers)
    if show_pred and 'predx' in event.index:
        pred_pos = [float(event['predx']), float(event['predy']), float(event['predz'])]

        # Predicted position axis lines
        ax.plot([pred_pos[0], xlim[0]], [pred_pos[1], pred_pos[1]], [pred_pos[2], pred_pos[2]],
               'r:', linewidth=1, alpha=0.5)
        ax.plot([pred_pos[0], xlim[1]], [pred_pos[1], pred_pos[1]], [pred_pos[2], pred_pos[2]],
               'r:', linewidth=1, alpha=0.5)
        ax.plot([pred_pos[0], pred_pos[0]], [pred_pos[1], ylim[0]], [pred_pos[2], pred_pos[2]],
               'r:', linewidth=1, alpha=0.5)
        ax.plot([pred_pos[0], pred_pos[0]], [pred_pos[1], ylim[1]], [pred_pos[2], pred_pos[2]],
               'r:', linewidth=1, alpha=0.5)
        ax.plot([pred_pos[0], pred_pos[0]], [pred_pos[1], pred_pos[1]], [pred_pos[2], zlim[0]],
               'r:', linewidth=1, alpha=0.5)
        ax.plot([pred_pos[0], pred_pos[0]], [pred_pos[1], pred_pos[1]], [pred_pos[2], zlim[1]],
               'r:', linewidth=1, alpha=0.5)

    # Plot true position marker LAST (on top, larger, no depthshade)
    # Ensure true position is transformed with align_params for both marker and axis lines
    truex, truey, truez = float(event['truex']), float(event['truey']), float(event['truez'])
    truex_t, truey_t, truez_t = transform_arrays(np.array([truex]), np.array([truey]), np.array([truez]), align_params)
    # Axis lines from true position to plot limits (use transformed coordinates)
    ax.plot([truex_t[0], xlim[0]], [truey_t[0], truey_t[0]], [truez_t[0], truez_t[0]], 'b:', linewidth=1, alpha=0.5)
    ax.plot([truex_t[0], xlim[1]], [truey_t[0], truey_t[0]], [truez_t[0], truez_t[0]], 'b:', linewidth=1, alpha=0.5)
    ax.plot([truex_t[0], truex_t[0]], [truey_t[0], ylim[0]], [truez_t[0], truez_t[0]], 'b:', linewidth=1, alpha=0.5)
    ax.plot([truex_t[0], truex_t[0]], [truey_t[0], ylim[1]], [truez_t[0], truez_t[0]], 'b:', linewidth=1, alpha=0.5)
    ax.plot([truex_t[0], truex_t[0]], [truey_t[0], truey_t[0]], [truez_t[0], zlim[0]], 'b:', linewidth=1, alpha=0.5)
    ax.plot([truex_t[0], truex_t[0]], [truey_t[0], truey_t[0]], [truez_t[0], zlim[1]], 'b:', linewidth=1, alpha=0.5)
    ax.scatter(truex_t[0], truey_t[0], truez_t[0], c='blue', s=200, marker='o', label='True', 
               edgecolors='black', linewidths=2, depthshade=False, zorder=1000)
    # Plot predicted position marker LAST (on top) if requested
    if show_pred and 'predx' in event.index:
        ax.scatter(*pred_pos, c='red', s=200, marker='o', label='Pred',
                   edgecolors='black', linewidths=2, depthshade=False, zorder=1000)

    # Set 1:1:1 aspect ratio
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    z_range = zlim[1] - zlim[0]
    max_range = max(x_range, y_range, z_range)
    ax.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])

    # Set labels and limits
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    ax.set_zlabel('Z [cm]')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_title(title, fontsize=10)
    ax.legend(loc='upper right', fontsize=8)


def plot_3d_event_displays(
    df: pd.DataFrame,
    df_geom: pd.DataFrame,
    tpc_bounds_mm: np.ndarray,
    align_params,
    events_by_metric: Dict[str, List[int]],
    outdir: Path,
    show_pred: bool = True,
) -> None:
    """
    Create 3D event display plots for selected events.

    Args:
        df: Full dataframe with events
        df_geom: Detector geometry with transformed positions
        tpc_bounds_mm: TPC bounds
        align_params: AlignParams for transforms
        events_by_metric: Dict mapping metric name to list of 3 event indices [min, median, max]
        outdir: Output directory
        show_pred: Whether to show predicted positions
    """
    for metric_name, event_indices in events_by_metric.items():
        fig = plt.figure(figsize=(18, 6))

        labels = ['Min', 'Median', 'Max']
        for i, (event_idx, label) in enumerate(zip(event_indices, labels)):
            ax = fig.add_subplot(1, 3, i+1, projection='3d')
            event = df.loc[event_idx]

            # Build title with metric value
            if metric_name == 'total_signal':
                metric_val = event['total_signal']
                title = f"{label} {metric_name}\n({metric_val:.1f} PE/sample)"
            elif metric_name == 'dr':
                metric_val = event.get('dr', np.nan)
                title = f"{label} residual r\n({metric_val:.2f} cm)"
            else:
                title = f"{label} {metric_name}"

            plot_3d_event_display(
                ax, event, df_geom, tpc_bounds_mm, align_params,
                show_pred=show_pred, title=title
            )

        plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space at top for titles
        fig.savefig(outdir / f"event_display_3d_{metric_name}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
