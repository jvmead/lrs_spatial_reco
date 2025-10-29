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

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False
    print("Warning: corner package not available. Corner plots will be skipped.")

def plot_distributions(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    w_col: Optional[str] = None,
    x_col2: Optional[str] = None,
    y_col2: Optional[str] = None,
    z_col2: Optional[str] = None,
    w_col2: Optional[str] = None,
    tpc_bounds_mm: Optional[np.ndarray] = None,
    out_file: Optional[Path] = None,
    label1: str = "data1",
    label2: Optional[str] = None,
    titles: Optional[list[str]] = None,
    colors: Optional[list[str]] = None,
    colors2: Optional[list[str]] = None,
    reference_lines: Optional[list[tuple[float, str, str]]] = None,
    xlabels: Optional[list[str]] = None,
) -> None:
    """
    Plot distributions for up to 4 variables, optionally with overlaid second dataset.

    Args:
        df: DataFrame with data columns
        x_col, y_col, z_col: Column names for first 3 variables
        w_col: Optional 4th variable column (e.g., "total_signal" or "dr")
        x_col2, y_col2, z_col2, w_col2: Optional columns for second dataset (overlay)
        tpc_bounds_mm: Optional TPC bounds array (n_tpc, 2, 3) for vertical lines on x/y/z
        out_file: If provided, save figure to this path
        label1: Label for first dataset
        label2: Label for second dataset (if provided)
        titles: List of subplot titles (default: None)
        colors: List of colors for first dataset [x, y, z, w] (default: red, green, blue, purple)
        colors2: List of colors for second dataset [x, y, z, w] (default: orange, lightgreen, cyan, pink)
        reference_lines: List of (value, color, label) tuples for vertical reference lines on each subplot
        xlabels: List of x-axis labels (default: "x [cm]", "y [cm]", "z [cm]", "w")
    """
    # Determine number of subplots
    n_plots = 4 if w_col is not None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 3))

    # Default colors (matching old plot_spatial_distributions)
    if colors is None:
        colors = ["red", "green", "blue", "purple"]
    if colors2 is None:
        colors2 = ["orange", "lightgreen", "cyan", "pink"]

    # Default labels
    if xlabels is None:
        xlabels = ["x [cm]", "y [cm]", "z [cm]", "w"]

    # Column lists for iteration
    cols1 = [x_col, y_col, z_col]
    cols2 = [x_col2, y_col2, z_col2]
    if w_col:
        cols1.append(w_col)
        cols2.append(w_col2)

    # Plot each variable
    for i in range(len(cols1)):
        col1 = cols1[i]
        col2 = cols2[i] if i < len(cols2) else None
        color1 = colors[i] if i < len(colors) else "gray"
        color2 = colors2[i] if i < len(colors2) else "lightgray"
        xlabel = xlabels[i] if i < len(xlabels) else f"var{i}"

        # Compute bins based on data range
        if col2:
            data_min = min(df[col1].min(), df[col2].min())
            data_max = max(df[col1].max(), df[col2].max())
        else:
            data_min, data_max = df[col1].min(), df[col1].max()

        bins = np.linspace(data_min, data_max, 50)

        # Plot first dataset (no edgecolor like original)
        hist_kwargs1 = {'bins': bins, 'alpha': 0.5, 'color': color1}
        if label1 is not None:
            hist_kwargs1['label'] = label1
        axes[i].hist(df[col1], **hist_kwargs1)

        # Plot second dataset if provided
        if col2:
            hist_kwargs2 = {'bins': bins, 'alpha': 0.5, 'color': color2}
            if label2 is not None:
                hist_kwargs2['label'] = label2
            axes[i].hist(df[col2], **hist_kwargs2)

        # Set labels and limits
        axes[i].set_xlabel(xlabel)
        axes[i].set_xlim(data_min, data_max)

        # Add title if provided
        if titles and i < len(titles):
            axes[i].set_title(titles[i])

        # Add TPC bounds for x, y, z (not w)
        if tpc_bounds_mm is not None and i < 3:
            for bounds in tpc_bounds_mm:
                axes[i].axvline(bounds[0][i], color="k", linestyle="--", alpha=1)
                axes[i].axvline(bounds[1][i], color="k", linestyle="--", alpha=1)

        # Add reference lines if provided (list of lists or single list for all)
        if reference_lines is not None and i < len(reference_lines):
            ref_lines = reference_lines[i]
            if ref_lines:  # Only if this subplot has reference lines
                for value, color, lbl in ref_lines:
                    axes[i].axvline(value, color=color, linestyle='--', linewidth=1, label=lbl)

        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    if out_file:
        fig.savefig(out_file)
        plt.close(fig)
    else:
        plt.show()


# Backwards compatibility alias
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
    """Backwards compatibility wrapper for plot_distributions."""
    w_col = "total_signal" if (include_log_signal and "total_signal" in df.columns) else None
    plot_distributions(
        df=df,
        x_col=x_true_col,
        y_col=y_true_col,
        z_col=z_true_col,
        w_col=w_col,
        x_col2=x_pred_col,
        y_col2=y_pred_col,
        z_col2=z_pred_col,
        w_col2=None,
        tpc_bounds_mm=tpc_bounds_mm,
        out_file=out_file,
        label1=label_true,
        label2=label_pred if x_pred_col else None,
    )


def plot_residual_distributions(
    df: pd.DataFrame,
    dx_col: str = "dx",
    dy_col: str = "dy",
    dz_col: str = "dz",
    dr_col: str = "dr",
    out_file: Optional[Path] = None,
    tpc_bounds_mm: Optional[np.ndarray] = None,
    truex_col: str = "truex",
    truey_col: str = "truey",
    truez_col: str = "truez",
) -> None:
    """
    Plot residual distributions using the generalized plot_distributions function.
    Adds statistics: std dev for dx/dy/dz, and median/1.538 for dr.
    """
    # Compute statistics
    dx_std = df[dx_col].std()
    dy_std = df[dy_col].std()
    dz_std = df[dz_col].std()
    dr_median = df[dr_col].median()
    dr_scaled = dr_median / 1.538

    # Compute center-guess reference for dr if TPC bounds provided
    reference_lines = []

    # dx, dy, dz get perfect line + std dev annotation
    for i, (col, std) in enumerate([(dx_col, dx_std), (dy_col, dy_std), (dz_col, dz_std)]):
        reference_lines.append([
            (0, 'red', ''),
            (std, 'k', f'σ={std:.2f} cm'),
            (-std, 'k', '')  # Mirror line without duplicate label
        ])

    # For dr, add perfect, center-guess, and median/1.538 lines
    dr_refs = [(0, 'red', '')]
    if tpc_bounds_mm is not None and all(col in df.columns for col in [truex_col, truey_col, truez_col]):
        # Compute TPC center (average across all modules)
        tpc_centers = []
        for mod_bounds in tpc_bounds_mm:
            center_x = (mod_bounds[0, 0] + mod_bounds[1, 0]) / 2
            center_y = (mod_bounds[0, 1] + mod_bounds[1, 1]) / 2
            center_z = (mod_bounds[0, 2] + mod_bounds[1, 2]) / 2
            tpc_centers.append([center_x, center_y, center_z])

        avg_center = np.mean(tpc_centers, axis=0)

        # Compute distance from each true position to center
        dr_center = np.sqrt(
            (df[truex_col] - avg_center[0])**2 +
            (df[truey_col] - avg_center[1])**2 +
            (df[truez_col] - avg_center[2])**2
        )
        # Use median/1.538 as std dev estimator for center guess (consistent with actual reco)
        center_guess_stddev = np.median(dr_center) / 1.538
        dr_refs.append((center_guess_stddev, 'orange', f'Center guess: {center_guess_stddev:.2f} cm'))

    # Add median/1.538 line for actual reconstruction
    dr_refs.append((dr_scaled, 'green', f'Reco median/1.538: {dr_scaled:.2f} cm'))
    reference_lines.append(dr_refs)

    # Use the generalized plot_distributions function
    plot_distributions(
        df=df,
        x_col=dx_col,
        y_col=dy_col,
        z_col=dz_col,
        w_col=dr_col,
        tpc_bounds_mm=None,  # Don't show TPC bounds for residuals
        out_file=out_file,
        label1=None,  # No legend entry for the histograms
        titles=None,  # No titles on subplots
        xlabels=["dx [cm]", "dy [cm]", "dz [cm]", "dr [cm]"],
        reference_lines=reference_lines,
    )


def plot_residual_corner(
    df: pd.DataFrame,
    dx_col: str = "dx",
    dy_col: str = "dy",
    dz_col: str = "dz",
    dr_col: str = "dr",
    out_file: Optional[Path] = None,
) -> None:
    """
    Create a corner plot showing correlations between residuals.
    Requires the 'corner' package.

    Args:
        df: DataFrame with residual columns
        dx_col, dy_col, dz_col, dr_col: Column names for residuals
        out_file: If provided, save figure to this path
    """
    if not HAS_CORNER:
        print("Corner plot skipped: 'corner' package not installed. Install with: pip install corner")
        return

    # Prepare data
    labels = ['dx (cm)', 'dy (cm)', 'dz (cm)', 'dr (cm)']
    data = np.vstack([
        df[dx_col].to_numpy(dtype=float),
        df[dy_col].to_numpy(dtype=float),
        df[dz_col].to_numpy(dtype=float),
        df[dr_col].to_numpy(dtype=float)
    ]).T

    # Remove any rows with NaN or Inf
    finite_mask = np.all(np.isfinite(data), axis=1)
    data = data[finite_mask]

    if len(data) == 0:
        print("Warning: No finite data for corner plot")
        return

    # Create corner plot
    figure = corner.corner(
        data,
        labels=labels,
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12},
        quantiles=[0.16, 0.5, 0.84],
        levels=(0.68, 0.95),
        plot_datapoints=False,
        fill_contours=True,
    )

    if out_file:
        figure.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close(figure)
    else:
        plt.show()


def plot_detector_signals(
    df: pd.DataFrame,
    out_file: Optional[Path] = None,
    log_scale: bool = False,
    bins: int = 50,
    log_bins: bool = False,
    log_xscale: bool = False,
    pde_values: Optional[np.ndarray] = None,
) -> None:
    """
    Plot detector signal distributions in a 2-column layout.

    Layout: Detectors 0-7 in left column (bottom to top), 8-15 in right column (bottom to top).
    Histograms are colored by PDE (photon detection efficiency) if provided.

    Args:
        df: DataFrame with det_0_max through det_15_max columns
        out_file: Output file path (if None, displays interactively)
        log_scale: If True, use log scale on y-axis
        bins: Number of histogram bins (if log_bins=False)
        log_bins: If True, use logarithmic binning
        log_xscale: If True, use log scale on x-axis
        pde_values: Optional array of PDE values (length 16) for color mapping
    """
    det_cols = [f"det_{i}_max" for i in range(16)]

    # Check which columns exist
    available_cols = [c for c in det_cols if c in df.columns]
    if not available_cols:
        print("Warning: No detector columns found (det_#_max)")
        return

    # Create 2-column layout: 8 rows x 2 columns
    fig, axes = plt.subplots(8, 2, figsize=(10, 20))
    fig.suptitle("Detector Signal Distributions (colored by PDE)", fontsize=14, y=0.995)

    # Color mapping based on PDE values
    if pde_values is not None and len(pde_values) == 16:
        # Normalize PDE for color mapping (0.002 to 0.006 range typically)
        pde_min, pde_max = pde_values.min(), pde_values.max()
        if pde_max > pde_min:
            cmap = plt.cm.viridis
            norm = plt.Normalize(vmin=pde_min, vmax=pde_max)
            colors = [cmap(norm(pde)) for pde in pde_values]
        else:
            colors = ['steelblue'] * 16
    else:
        colors = ['steelblue'] * 16

    for i in range(16):
        # Layout: det 0-7 in left column (row 7 to 0), det 8-15 in right column (row 7 to 0)
        if i < 8:
            col = 0
            row = 7 - i  # det 0 at bottom (row 7), det 7 at top (row 0)
        else:
            col = 1
            row = 7 - (i - 8)  # det 8 at bottom (row 7), det 15 at top (row 0)

        ax = axes[row, col]

        col_name = f"det_{i}_max"
        if col_name in df.columns:
            data = df[col_name].values
            # Remove NaN and inf
            data = data[np.isfinite(data)]
            # Remove zeros for log binning/scale
            if log_bins or log_xscale:
                data = data[data > 0]

            if len(data) > 0:
                # Compute bins
                if log_bins:
                    # Create log-spaced bins
                    if data.min() > 0:
                        bin_edges = np.logspace(np.log10(data.min()), np.log10(data.max()), bins + 1)
                    else:
                        bin_edges = bins
                else:
                    bin_edges = bins

                # Use color based on PDE
                hist_color = colors[i]
                ax.hist(data, bins=bin_edges, alpha=0.7, color=hist_color, edgecolor='none')

                # Add statistics text
                mean_val = np.mean(data)
                median_val = np.median(data)
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'μ={mean_val:.1f}')
                ax.axvline(median_val, color='orange', linestyle=':', linewidth=1, alpha=0.7, label=f'med={median_val:.1f}')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Missing', ha='center', va='center', transform=ax.transAxes)

        # Title with PDE info if available
        if pde_values is not None and len(pde_values) == 16:
            ax.set_title(f"Det {i} (PDE={pde_values[i]:.4f})", fontsize=9)
        else:
            ax.set_title(f"Det {i}", fontsize=9)
        ax.tick_params(labelsize=7)

        if log_scale:
            ax.set_yscale('log')

        if log_xscale:
            ax.set_xscale('log')

        # Only show x-label on bottom row
        if row == 7:
            ax.set_xlabel("Signal", fontsize=8)

        # Only show y-label on leftmost column
        if col == 0:
            ax.set_ylabel("Count", fontsize=8)

        # Add small legend
        if col_name in df.columns and len(data) > 0:
            ax.legend(fontsize=6, loc='upper right', framealpha=0.7)

    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_file}")
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
                # Create subdirectory for each independent variable
                heatmaps_var_dir = outdir / "heatmaps" / f"heatmaps_f_{var}"
                heatmaps_var_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(heatmaps_var_dir / f"heatmap_{resid}_vs_{var}.png", dpi=150)
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
        # Create subdirectory for pred vs true heatmaps
        pred_vs_true_dir = outdir / "heatmaps" / "pred_vs_true"
        pred_vs_true_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(pred_vs_true_dir / f"heatmap_pred{label}_vs_true{label}.png", dpi=150)
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
    # Create subdirectory for 1D plots
    plots_1d_dir = outdir / "plots_1d"
    plots_1d_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_1d_dir / f"plot_1d_{var}_mu.png", dpi=150); plt.close(fig)

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
    # Create subdirectory for 1D plots
    plots_1d_dir = outdir / "plots_1d"
    plots_1d_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_1d_dir / f"plot_1d_{var}_sig.png", dpi=150); plt.close(fig)


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
    det_signals = np.nan_to_num(det_signals, nan=0.0)  # Treat NaN as zero signal
    det_positions = np.array(det_positions)

    # Scale marker sizes: linear scaling from 10 (min non-zero) to 300 (max)
    size_min, size_max = 10, 300
    if len(det_signals) > 0:
        if det_signals.max() > 0:
            # Linear scaling based on detector signal values
            sizes = size_min + (det_signals / det_signals.max()) * (size_max - size_min)
            sizes[det_signals == 0] = 1  # Zero signal gets size 1 for array, but plotted separately
        else:
            sizes = np.full_like(det_signals, 1)  # All zeros
    else:
        sizes = np.array([])

    # Plot detectors (with separate markers for signal vs no-signal)
    if len(det_positions) > 0:
        has_signal = det_signals > 0
        no_signal = (det_signals == 0) | (det_signals == np.nan)

        # Plot detectors with signal (circles, variable size)
        if has_signal.any():
            ax.scatter(det_positions[has_signal, 0], det_positions[has_signal, 2], det_positions[has_signal, 1],
                   s=sizes[has_signal], c='lightgray', alpha=0.7, edgecolors='black', linewidths=0.5,
                   label='_nolegend_', depthshade=False, marker='o')

        # Plot detectors without signal (crosses, larger for visibility)
        if no_signal.any():
            ax.scatter(det_positions[no_signal, 0], det_positions[no_signal, 2], det_positions[no_signal, 1],
                   s=50, c='red', alpha=0.3,linewidths=0.5,
                   label='_nolegend_', depthshade=False, marker='x')

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

        # Draw edges (X, Z, Y order for Y-up orientation)
        for i, j in edges_idx:
            ax.plot([transformed_corners[i, 0], transformed_corners[j, 0]],
                   [transformed_corners[i, 2], transformed_corners[j, 2]],
                   [transformed_corners[i, 1], transformed_corners[j, 1]],
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

    # True position axis lines to boundaries (X, Z, Y order for Y-up orientation)
    ax.plot([true_pos[0], xlim[0]], [true_pos[2], true_pos[2]], [true_pos[1], true_pos[1]],
           'b:', linewidth=1, alpha=0.5)
    ax.plot([true_pos[0], xlim[1]], [true_pos[2], true_pos[2]], [true_pos[1], true_pos[1]],
           'b:', linewidth=1, alpha=0.5)
    ax.plot([true_pos[0], true_pos[0]], [true_pos[2], zlim[0]], [true_pos[1], true_pos[1]],
           'b:', linewidth=1, alpha=0.5)
    ax.plot([true_pos[0], true_pos[0]], [true_pos[2], zlim[1]], [true_pos[1], true_pos[1]],
           'b:', linewidth=1, alpha=0.5)
    ax.plot([true_pos[0], true_pos[0]], [true_pos[2], true_pos[2]], [true_pos[1], ylim[0]],
           'b:', linewidth=1, alpha=0.5)
    ax.plot([true_pos[0], true_pos[0]], [true_pos[2], true_pos[2]], [true_pos[1], ylim[1]],
           'b:', linewidth=1, alpha=0.5)

    # Plot predicted position axis lines if requested (X, Z, Y order for Y-up orientation)
    if show_pred and 'predx' in event.index:
        pred_pos = [float(event['predx']), float(event['predy']), float(event['predz'])]

        # Predicted position axis lines
        ax.plot([pred_pos[0], xlim[0]], [pred_pos[2], pred_pos[2]], [pred_pos[1], pred_pos[1]],
               'r:', linewidth=1, alpha=0.5)
        ax.plot([pred_pos[0], xlim[1]], [pred_pos[2], pred_pos[2]], [pred_pos[1], pred_pos[1]],
               'r:', linewidth=1, alpha=0.5)
        ax.plot([pred_pos[0], pred_pos[0]], [pred_pos[2], zlim[0]], [pred_pos[1], pred_pos[1]],
               'r:', linewidth=1, alpha=0.5)
        ax.plot([pred_pos[0], pred_pos[0]], [pred_pos[2], zlim[1]], [pred_pos[1], pred_pos[1]],
               'r:', linewidth=1, alpha=0.5)
        ax.plot([pred_pos[0], pred_pos[0]], [pred_pos[2], pred_pos[2]], [pred_pos[1], ylim[0]],
               'r:', linewidth=1, alpha=0.5)
        ax.plot([pred_pos[0], pred_pos[0]], [pred_pos[2], pred_pos[2]], [pred_pos[1], ylim[1]],
               'r:', linewidth=1, alpha=0.5)

    # Plot true position marker (X, Z, Y order for Y-up orientation)
    truex_t, truey_t, truez_t = float(event['truex']), float(event['truey']), float(event['truez'])
    ax.scatter(truex_t, truez_t, truey_t, c='blue', s=60, marker='o', label='True',
               edgecolors='black', linewidths=2, depthshade=False, zorder=1000)

    # Plot predicted position marker if requested
    if show_pred and 'predx' in event.index:
        ax.scatter(pred_pos[0], pred_pos[2], pred_pos[1], c='red', s=60, marker='o', label='Pred',
                   edgecolors='black', linewidths=2, depthshade=False, zorder=1000)

    # Set 1:1:1 aspect ratio
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    z_range = zlim[1] - zlim[0]
    max_range = max(x_range, y_range, z_range)
    ax.set_box_aspect([x_range/max_range, z_range/max_range, y_range/max_range])

    # Set labels and limits (X, Z, Y order for Y-up orientation)
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Z [cm]')
    ax.set_zlabel('Y [cm]')
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_zlim(ylim)
    ax.set_title(title, fontsize=10, pad=20)  # Added pad=20 to offset title
    ax.legend(loc='lower right', fontsize=8)


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
    Create 3D event display plots for selected events with 2D projections below.

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
        # Create figure with 2 rows: 3D plots on top, 2D projections on bottom
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.05, wspace=0.15)

        labels = ['Min', 'Median', 'Max']
        for i, (event_idx, label) in enumerate(zip(event_indices, labels)):
            # 3D plot on top row
            ax_3d = fig.add_subplot(gs[0, i], projection='3d')
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
                ax_3d, event, df_geom, tpc_bounds_mm, align_params,
                show_pred=show_pred, title=title
            )

            # Add 2D projections in bottom row
            # Get positions
            truex, truey, truez = float(event['truex']), float(event['truey']), float(event['truez'])

            # Filter detector geometry to only show detectors from this event's TPC
            event_tpc_val = event.get('int_tpc_num', -1)
            if pd.notna(event_tpc_val):
                event_tpc = int(event_tpc_val)
            else:
                event_tpc = -1
            if event_tpc >= 0:
                df_geom_filtered = df_geom[df_geom['TPC'] == event_tpc]
            else:
                df_geom_filtered = df_geom

            # Get detector signals and positions
            det_signals = []
            det_positions = []
            for det_id in range(16):
                col = f"det_{det_id}_max"
                if col in event.index:
                    signal = float(event[col])
                    det_row = df_geom_filtered[df_geom_filtered['Detector'] == det_id]
                    if not det_row.empty:
                        det_signals.append(signal)
                        det_positions.append([
                            float(det_row['x_offset'].iloc[0]),
                            float(det_row['y_offset'].iloc[0]),
                            float(det_row['z_offset'].iloc[0])
                        ])
            det_signals = np.array(det_signals)
            det_signals = np.nan_to_num(det_signals, nan=0.0)
            det_positions = np.array(det_positions)

            # Scale marker sizes: linear scaling from 10 (min non-zero) to 300 (max)
            size_min, size_max = 10, 300
            if len(det_signals) > 0 and det_signals.max() > 0:
                # Linear scaling based on detector signal values
                sizes = size_min + (det_signals / det_signals.max()) * (size_max - size_min)
                sizes[det_signals == 0] = 1  # Zero signal gets very small size
            else:
                sizes = np.full_like(det_signals, 1)

            # Create 2D projection plot (ZY view: Z horizontal, Y vertical)
            ax_zy = fig.add_subplot(gs[1, i])

            # Plot detectors with signal (circles)
            if len(det_positions) > 0:
                has_signal = det_signals > 0
                if has_signal.any():
                    ax_zy.scatter(det_positions[has_signal, 2], det_positions[has_signal, 1],
                                 s=sizes[has_signal], c='lightgray', alpha=0.7,
                                 edgecolors='black', linewidths=0.5, marker='o')

                # Plot detectors without signal (crosses, MUCH larger for visibility)
                no_signal = (det_signals == 0) | (det_signals == np.nan)
                if no_signal.any():
                    ax_zy.scatter(det_positions[no_signal, 2], det_positions[no_signal, 1],
                                 s=50, c='red', alpha=0.3, linewidths=0.5, marker='x')

            # Plot true position
            ax_zy.scatter(truez, truey, c='blue', s=50, marker='o',
                         edgecolors='black', linewidths=1)

            # Plot predicted position if available
            if show_pred and 'predx' in event.index:
                ax_zy.scatter(float(event['predz']), float(event['predy']),
                             c='red', s=50, marker='o', edgecolors='black',
                             linewidths=1)

            # Compute axis limits from ALL transformed TPC bounds (not just event's TPC)
            # This ensures all three 2D plots have the same axis limits
            from utils import transform_arrays
            all_z_transformed = []
            all_y_transformed = []

            for bounds in tpc_bounds_mm:
                z_min_orig, y_min_orig = bounds[0][2], bounds[0][1]
                z_max_orig, y_max_orig = bounds[1][2], bounds[1][1]
                x_min_orig, x_max_orig = bounds[0][0], bounds[1][0]

                # Transform all 8 corners
                corners_x = np.array([x_min_orig, x_min_orig, x_min_orig, x_min_orig,
                                     x_max_orig, x_max_orig, x_max_orig, x_max_orig])
                corners_y = np.array([y_min_orig, y_min_orig, y_max_orig, y_max_orig,
                                     y_min_orig, y_min_orig, y_max_orig, y_max_orig])
                corners_z = np.array([z_min_orig, z_max_orig, z_min_orig, z_max_orig,
                                     z_min_orig, z_max_orig, z_min_orig, z_max_orig])

                # Transform returns (x_t, y_t, z_t) in that order
                x_t, y_t, z_t = transform_arrays(corners_x, corners_y, corners_z, align_params)
                all_z_transformed.extend(z_t)
                all_y_transformed.extend(y_t)

            # Set limits to encompass all TPCs
            z_min, z_max = min(all_z_transformed), max(all_z_transformed)
            y_min, y_max = min(all_y_transformed), max(all_y_transformed)
            ax_zy.set_xlim([z_min, z_max])
            ax_zy.set_ylim([y_min, y_max])

            ax_zy.set_xlabel('Z [cm]', fontsize=9)
            ax_zy.set_ylabel('Y [cm]', fontsize=9)
            # Set title with true position coordinates
            ax_zy.set_title(f'True: ({truex:.1f}, {truey:.1f}, {truez:.1f}) cm', fontsize=10)
            ax_zy.tick_params(labelsize=8)
            # Enforce 1:1 aspect ratio
            ax_zy.set_aspect('equal')
            ax_zy.grid(True, alpha=0.3)

        event_displays_dir = outdir / "event_displays_3d"
        event_displays_dir.mkdir(exist_ok=True)
        fig.savefig(event_displays_dir / f"event_display_3d_{metric_name}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

