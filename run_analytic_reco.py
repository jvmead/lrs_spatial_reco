#!/usr/bin/env python3
"""
Analytic reconstruction with:
- predx/predy/predz computation (names unchanged; --uw only affects directory name)
- Minimalist 1D residual CSV exports (bin_x, *_mu, *_sig)
- 1D μ/σ line plots vs chosen variables
- Heatmaps: residual (y) vs independent variable (x) with RED 16/50/84% quantile overlays
  * residuals: {dx, dy, dz, dr}
  * independent vars: {truex, truey, truez, r(true), total_signal}
  * total_signal uses LOG10-SPACED BINS and x-axis is set to log
- Additionally, when --plot-2d-heatmaps is used, also plot:
  * predx vs truex (with white dashed y=x)
  * predy vs truey (with white dashed y=x)
  * predz vs truez (with white dashed y=x)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import h5py
import numpy as np
import pandas as pd
import yaml

# Import geometry utilities from utils module
from utils import (
    compute_align_params,
    transform_geom,
    detector_means_by_id,
    load_hdf_geometry,
    load_geom_csv,
    load_geom_yaml,
    compute_module_centres,
    compute_detector_offsets,
    extract_pde_per_detector,
)


DEFAULT_1D_VARS = ["truex", "truey", "truez", "dr", "total_signal"]
HEATMAP_INDEP_VARS = ["truex", "truey", "truez", "r", "total_signal"]  # r = true radius


# ---------- small utils ----------

def timestamped_outdir(base_dir: Path, mode_tag: str, tag: Optional[str]) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    parts = [f"outputs_{mode_tag}"]
    if tag:
        parts.append(tag)
    parts.append(ts)
    out_name = "_".join(parts)
    outdir = base_dir / out_name
    outdir.mkdir(parents=True, exist_ok=False)
    print("Saving outputs to:", outdir)
    return outdir


def ensure_outdir(args, mode_tag: str) -> Path:
    base = Path(args.outdir) if args.outdir else args.input_csv.parent
    prefixed_base = base.parent / f"analytic_reco_{base.name}"
    return timestamped_outdir(prefixed_base, mode_tag, args.tag)


def sanitize_and_load_csv(input_csv: Path, x_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(input_csv)
    df = df.replace([np.inf, -np.inf], np.nan)

    required_truth = {"truex", "truey", "truez"}
    missing_truth = required_truth - set(df.columns)
    if missing_truth:
        raise ValueError(f"Missing required truth columns: {sorted(missing_truth)}")

    n_before = len(df)
    df = df.dropna(subset=["truex", "truey", "truez"], how="any").reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} rows missing true positions")

    present_x_cols = [c for c in x_cols if c in df.columns]
    if not present_x_cols:
        raise ValueError("No detector amplitude columns like det_#_max found.")
    df[present_x_cols] = df[present_x_cols].fillna(0.0)

    df["total_signal"] = df[present_x_cols].sum(axis=1)
    n_before = len(df)
    df = df[df["total_signal"] > 0.0].reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} rows with zero total signal")
    df["log_total_signal"] = np.log10(df["total_signal"])

    return df, present_x_cols


# Note: load_hdf_geometry and build_detector_positions now imported from utils


def compute_predictions(
    df: pd.DataFrame,
    det_cols: List[str],
    det_pos_x: np.ndarray,
    det_pos_y: np.ndarray,
    det_pos_z: np.ndarray,
    uw: bool,
    eff_per_det: Optional[np.ndarray],
) -> pd.DataFrame:
    full = np.zeros((len(df), 16), dtype=float)
    for i in range(16):
        col = f"det_{i}_max"
        if col in det_cols:
            full[:, i] = df[col].to_numpy(dtype=float, na_value=0.0)

    if uw:
        weights = full
    else:
        if eff_per_det is None:
            eff_per_det = np.ones((16,), dtype=float)
        eff = eff_per_det.reshape(1, 16)
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.divide(full, eff, out=np.zeros_like(full), where=eff != 0)

    totals = weights.sum(axis=1, keepdims=True)
    zero_mask = (totals.squeeze() == 0.0)
    totals[zero_mask, :] = 1.0
    wnorm = weights / totals

    px = wnorm.dot(det_pos_x)
    py = wnorm.dot(det_pos_y)
    pz = wnorm.dot(det_pos_z)

    px[zero_mask] = np.nan
    py[zero_mask] = np.nan
    pz[zero_mask] = np.nan

    out = df.copy()
    out["predx"] = px
    out["predy"] = py
    out["predz"] = pz
    return out


def _residual_arrays(df_pred: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx = (df_pred["predx"] - df_pred["truex"]).to_numpy(dtype=float)
    dy = (df_pred["predy"] - df_pred["truey"]).to_numpy(dtype=float)
    dz = (df_pred["predz"] - df_pred["truez"]).to_numpy(dtype=float)
    dr = np.sqrt(dx**2 + dy**2 + dz**2)
    return dx, dy, dz, dr


# ---------- 1D differentials (for CSVs and 1D plots) ----------

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
    dx, dy, dz, dr = _residual_arrays(df_pred)

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


# ---------- Heatmaps (correct axes + RED quantiles) ----------

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
    """
    Plot a single 2D histogram (heatmap) of df[x_col] vs df[y_col].
    xrange and yrange are arrays of bin edges.
    If x_log is True, set a logarithmic x-axis (edges should already be log-spaced).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    h = ax.hist2d(df[x_col], df[y_col], bins=[xrange, yrange], cmap=cmap)
    plt.colorbar(h[3], ax=ax, label="Counts")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if x_log:
        ax.set_xscale("log")

    # Quantiles per x-bin (left edges for step alignment)
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

    # RED quantile overlays
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
    df_pred: pd.DataFrame, outdir: Path, vars_list: List[str], bins_map: Dict[str, int]
) -> None:
    """
    For each residual in {dx,dy,dz,dr} and each var in vars_list,
    draw heatmap: x = var, y = residual.
    - total_signal uses log10-spaced bins + log x-axis
    """
    import matplotlib.pyplot as plt

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
            x_edges = _edges_log10(x_vals, bins)
            x_log = True
        else:
            x_edges = _edges_linear(x_vals, bins)
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


def plot_pred_vs_true_xyz(df_pred: pd.DataFrame, outdir: Path, bins_map: Dict[str, int]) -> None:
    """
    Plot predx vs truex, predy vs truey, predz vs truez with white dashed y=x.
    Uses linear, shared edges based on true#.
    """
    import matplotlib.pyplot as plt

    pairs = [
        ("truex", "predx", "x"),
        ("truey", "predy", "y"),
        ("truez", "predz", "z"),
    ]

    for tcol, pcol, label in pairs:
        bins = bins_map.get(f"true{label}", 24)
        x_true = df_pred[tcol].to_numpy(dtype=float)
        x_pred = df_pred[pcol].to_numpy(dtype=float)
        edges = _edges_linear(x_true, bins)

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


# ---------- 1D line plots ----------

def plot_1d_curves(df1d: pd.DataFrame, outdir: Path, var: str) -> None:
    import matplotlib.pyplot as plt
    # Determine xscale for total_signal
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


# ------------------------------ CLI ------------------------------

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analytic reconstruction with correct heatmaps + 1D exports/plots")
    p.add_argument("input_csv", type=Path, help="Processed CSV with truth and det_*_max columns")
    p.add_argument("--geom-csv", type=Path, default=Path("../lrs_sanity_check/geom_files/light_module_desc-4.0.0.csv"),
                   help="Geometry CSV describing detectors")
    p.add_argument("--geom-yaml", type=Path, default=Path("../lrs_sanity_check/geom_files/light_module_desc-4.0.0.yaml"),
                   help="Geometry YAML with tpc_center_offset and det_center")
    p.add_argument("--hdf5", type=Path, default=Path("/global/cfs/cdirs/dune/www/data/2x2/simulation/productions/MiniRun6.5_1E19_RHC/MiniRun6.5_1E19_RHC.flow/FLOW/0000000/MiniRun6.5_1E19_RHC.flow.0000000.FLOW.hdf5"),
                   help="Reference HDF5 to obtain module bounds and max drift")

    # I/O behavior
    p.add_argument("--save", action="store_true", help="Save ONLY predx/predy/predz to a timestamped directory")
    p.add_argument("--tag", type=str, default=None, help="Optional tag appended to output directory name")
    p.add_argument("--outdir", type=Path, default=None, help="Explicit output base directory (overrides input dir use)")

    # Weighting toggle
    p.add_argument("--uw", action="store_true",
                   help="Use UNWEIGHTED amplitudes (do NOT scale by 1/PDE). If omitted, amplitudes are scaled by 1/PDE.")

    # 1D exports
    p.add_argument("--export-1d-over", nargs="*", default=[],
                   help="Export 1D residual stats over listed variables "
                        "(choose from: truex truey truez predx predy predz total_signal log_total_signal dr tot_signal r)")
    p.add_argument("--export-1d-all", action="store_true",
                   help="Export 1D residuals for: truex truey truez dr total_signal")

    # Bin counts
    p.add_argument("--bins-x", type=int, default=24, help="Bins for x-like variables (truex/predx)")
    p.add_argument("--bins-y", type=int, default=24, help="Bins for y-like variables (truey/predy)")
    p.add_argument("--bins-z", type=int, default=24, help="Bins for z-like variables (truez/predz)")
    p.add_argument("--bins-r", type=int, default=24, help="Bins for radius r and residual radius dr")
    p.add_argument("--bins-signal", type=int, default=24, help="Bins for total_signal/log_total_signal (log10-spaced for total_signal)")

    # Plotting (AUTO)
    p.add_argument("--plot-2d-heatmaps", action="store_true",
                   help="Auto-generate heatmaps: residual (y) vs each of truex,truey,truez,r,total_signal (log10 bins for total_signal) and also predx/predy/predz vs truex/truey/truez with y=x overlay.")
    p.add_argument("--plot-1d-over", nargs="*", default=[],
                   help="Save 1D line plots of μ and σ vs the listed variables (same options as --export-1d-over).")
    p.add_argument("--plot-1d-all", action="store_true",
                   help="Save 1D plots for: truex truey truez dr total_signal")

    return p.parse_args(argv)


# ------------------------------ Main ------------------------------

def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    x_cols = [f"det_{i}_max" for i in range(16)]

    df, det_cols_present = sanitize_and_load_csv(args.input_csv, x_cols)

    # Load geometry files
    df_geom = load_geom_csv(args.geom_csv)
    df_geom["mod"] = df_geom["TPC"] // 2
    geom_data = load_geom_yaml(args.geom_yaml)

    # Extract module bounds and build TPC bounds for transforms
    mod_bounds_mm, max_drift_distance = load_hdf_geometry(args.hdf5)
    module_centres = compute_module_centres(mod_bounds_mm)

    # Build TPC bounds array (needed for coordinate transform)
    # Each module has 2 TPCs (drift regions)
    tpc_bounds = []
    for mod in mod_bounds_mm:
        x_min, x_max = float(mod[0][0]), float(mod[1][0])
        y_min, y_max = float(mod[0][1]), float(mod[1][1])
        z_min, z_max = float(mod[0][2]), float(mod[1][2])
        x_min_adj = x_max - max_drift_distance
        x_max_adj = x_min + max_drift_distance
        tpc_bounds.append(((x_min_adj, y_min, z_min), (x_max, y_max, z_max)))
        tpc_bounds.append(((x_min, y_min, z_min), (x_max_adj, y_max, z_max)))
    tpc_bounds_mm = np.array(tpc_bounds, dtype=float)

    # Compute alignment parameters (SAME as used for event data in process_light_outputs.py)
    print("Computing coordinate transform parameters...")
    params = compute_align_params(tpc_bounds_mm)
    print(f"  x_diff={params.x_diff:.2f}, x_midway={params.x_midway:.2f}, z_diff={params.z_diff:.2f}")

    # Compute detector offsets in UNTRANSFORMED coordinates
    df_geom = compute_detector_offsets(df_geom, module_centres, geom_data)

    # CRITICAL: Transform detector geometry to common frame (same transform as events!)
    print("Transforming detector geometry to common frame...")
    df_geom_transformed = transform_geom(
        df_geom,
        params,
        x_col="x_offset",
        y_col="y_offset",
        z_col="z_offset"
    )

    # Aggregate transformed detector positions
    det_pos_x, det_pos_y, det_pos_z = detector_means_by_id(df_geom_transformed)
    print(f"Detector positions (transformed): x=[{det_pos_x.min():.1f}, {det_pos_x.max():.1f}], "
          f"y=[{det_pos_y.min():.1f}, {det_pos_y.max():.1f}], "
          f"z=[{det_pos_z.min():.1f}, {det_pos_z.max():.1f}]")

    # Extract PDE values systematically
    eff_per_det = extract_pde_per_detector(df_geom)

    df_pred = compute_predictions(
        df=df,
        det_cols=det_cols_present,
        det_pos_x=det_pos_x,
        det_pos_y=det_pos_y,
        det_pos_z=det_pos_z,
        uw=args.uw,
        eff_per_det=eff_per_det,
    )

    need_outdir = bool(
        args.save or args.export_1d_over or args.export_1d_all or
        args.plot_2d_heatmaps or args.plot_1d_over or args.plot_1d_all
    )
    outdir: Optional[Path] = None
    if need_outdir:
        mode_tag = "uw" if args.uw else "pde"
        outdir = ensure_outdir(args, mode_tag)

    if args.save and outdir is not None:
        out_df = df_pred[["predx", "predy", "predz"]].copy()
        out_path = outdir / "predictions.csv.gz"
        out_df.to_csv(out_path, index=False, compression="gzip")
        print("Wrote:", out_path)

        meta = {
            "input_csv": str(args.input_csv),
            "geom_csv": str(args.geom_csv),
            "geom_yaml": str(args.geom_yaml),
            "hdf5": str(args.hdf5),
            "uw": bool(args.uw),
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "n_rows": int(len(out_df)),
            "columns": ["predx", "predy", "predz"],
        }
        with open(outdir / "manifest.json", "w") as f:
            json.dump(meta, f, indent=2)

    export_vars = list(args.export_1d_over)
    plot1d_vars = list(args.plot_1d_over)
    if args.export_1d_all:
        export_vars = sorted(set(export_vars + DEFAULT_1D_VARS), key=lambda v: DEFAULT_1D_VARS.index(v) if v in DEFAULT_1D_VARS else 99)
    if args.plot_1d_all:
        plot1d_vars = sorted(set(plot1d_vars + DEFAULT_1D_VARS), key=lambda v: DEFAULT_1D_VARS.index(v) if v in DEFAULT_1D_VARS else 99)

    def choose_bins(var: str) -> int:
        m = {
            "truex": args.bins_x, "predx": args.bins_x,
            "truey": args.bins_y, "predy": args.bins_y,
            "truez": args.bins_z, "predz": args.bins_z,
            "r": args.bins_r, "dr": args.bins_r,
            "total_signal": args.bins_signal, "log_total_signal": args.bins_signal,
            "tot_signal": args.bins_signal,
        }
        return m.get(var, args.bins_x)

    if export_vars:
        if outdir is None:
            mode_tag = "uw" if args.uw else "pde"
            outdir = ensure_outdir(args, mode_tag)
        for var in export_vars:
            if var == "tot_signal":
                var = "total_signal"
            bins = choose_bins(var)
            df1d = compute_differential_stats_1d_minimal(df_pred, var=var, bins=bins)
            fname = f"resids_f_{var}.csv.gz"
            path = outdir / fname
            df1d.to_csv(path, index=False, compression="gzip")
            print("Wrote:", path)

    if args.plot_2d_heatmaps:
        if outdir is None:
            mode_tag = "uw" if args.uw else "pde"
            outdir = ensure_outdir(args, mode_tag)
        bins_map = {
            "truex": args.bins_x,
            "truey": args.bins_y,
            "truez": args.bins_z,
            "r": args.bins_r,
            "total_signal": args.bins_signal,  # log10-spaced inside plotting
        }
        # residual vs variable heatmaps
        plot_heatmaps_resid_vs_vars(df_pred, outdir=outdir, vars_list=HEATMAP_INDEP_VARS, bins_map=bins_map)
        # additional pred vs true for x,y,z
        bins_map_xyz = {"truex": args.bins_x, "truey": args.bins_y, "truez": args.bins_z}
        plot_pred_vs_true_xyz(df_pred, outdir=outdir, bins_map=bins_map_xyz)

    if plot1d_vars:
        if outdir is None:
            mode_tag = "uw" if args.uw else "pde"
            outdir = ensure_outdir(args, mode_tag)
        for var in plot1d_vars:
            if var == "tot_signal":
                var = "total_signal"
            bins = choose_bins(var)
            df1d = compute_differential_stats_1d_minimal(df_pred, var=var, bins=bins)
            plot_1d_curves(df1d, outdir=outdir, var=var)

    print("Sample predictions (first 5 rows):")
    print(df_pred[["predx", "predy", "predz"]].head(5).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
