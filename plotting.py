"""
plotting.py - Shared plotting and diagnostics for DUNE LRS spatial reco

Contains all diagnostic/statistics/plotting functions previously in utils.py.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

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
      bin_x, dx_mu, dx_sig, dy_mu, dy_sig, dz_mu, dz_sig, dr_mu, dr_sig
    Uses LOG10-SPACED bins when var == 'total_signal'.
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
    def moments_per_bin(vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = np.zeros(len(edges) - 1); s2 = np.zeros(len(edges) - 1); n = np.zeros(len(edges) - 1, dtype=int)
        for k, v in enumerate(vals):
            if not np.isfinite(v): continue
            i = ib[k]; s[i] += v; s2[i] += v*v; n[i] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(n > 0, s / n, np.nan)
            varr = np.where(n > 0, s2 / n - mu * mu, np.nan)
            sig = np.sqrt(np.where(varr >= 0, varr, np.nan))
        return mu, sig
    dx_mu, dx_sig = moments_per_bin(dx)
    dy_mu, dy_sig = moments_per_bin(dy)
    dz_mu, dz_sig = moments_per_bin(dz)
    dr_mu, dr_sig = moments_per_bin(dr)
    return pd.DataFrame({
        "bin_x": centers.astype(float),
        "dx_mu": dx_mu, "dx_sig": dx_sig,
        "dy_mu": dy_mu, "dy_sig": dy_sig,
        "dz_mu": dz_mu, "dz_sig": dz_sig,
        "dr_mu": dr_mu, "dr_sig": dr_sig,
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

def plot_1d_curves(df1d: pd.DataFrame, outdir: Path, var: str) -> None:
    xscale = "log" if var in ("total_signal", "log_total_signal") else "linear"
    if var in ("total_signal", "log_total_signal"):
        print(f"[plot_1d_curves] xscale for {var}: log")
    else:
        print(f"[plot_1d_curves] xscale for {var}: linear")
    fig = plt.figure(figsize=(7, 4))
    plt.plot(df1d["bin_x"], df1d["dx_mu"], label="dx")
    plt.plot(df1d["bin_x"], df1d["dy_mu"], label="dy")
    plt.plot(df1d["bin_x"], df1d["dz_mu"], label="dz")
    plt.plot(df1d["bin_x"], df1d["dr_mu"], label="dr")
    plt.xlabel(var); plt.ylabel("μ (mm)")
    plt.title(f"Residual μ vs {var}")
    plt.legend(); plt.grid(True, alpha=0.3); fig.tight_layout()
    if xscale == "log":
        plt.xscale("log")
    plt.savefig(outdir / f"plot_1d_{var}_mu.png", dpi=150); plt.close(fig)
    fig = plt.figure(figsize=(7, 4))
    plt.plot(df1d["bin_x"], df1d["dx_sig"], label="dx")
    plt.plot(df1d["bin_x"], df1d["dy_sig"], label="dy")
    plt.plot(df1d["bin_x"], df1d["dz_sig"], label="dz")
    plt.plot(df1d["bin_x"], df1d["dr_sig"], label="dr")
    plt.xlabel(var); plt.ylabel("σ (mm)")
    plt.title(f"Residual σ vs {var}")
    plt.legend(); plt.grid(True, alpha=0.3); fig.tight_layout()
    if xscale == "log":
        plt.xscale("log")
    plt.savefig(outdir / f"plot_1d_{var}_sig.png", dpi=150); plt.close(fig)
