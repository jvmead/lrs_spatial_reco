#!/usr/bin/env python3
"""
Refactored analytic reconstruction script.

Features:
- Cleaner structure with functions.
- Optional saving of all plots and metrics to a new timestamped directory
  inheriting the name of the input CSV directory (with optional tag).
- Attempts to import general utilities from process_light_outputs.py and
  falls back to local implementations if not available.
- Vectorized prediction computation (faster than per-row apply).
- Saves mu and sigma of residuals (dx,dy,dz,dr) for both PDE-corrected
  and uncorrected reconstructions as JSON for later comparisons.

Example usage:
  python run_analytic_reco.py processed_outputs_n100_nint1_mod123_dEweight_20251028-033432/truth_reco_processed.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Try to import commonly reused geometry helpers from a user's module.
# If not present, define minimal fallbacks used by this script.
try:
    from process_light_outputs import (
        load_geom_csv,
        load_geom_yaml,
        load_or_build_tpc_bounds,
        compute_module_centres,
        transform_detector_coords,
    )
    _HAS_PLO_MODULE = True
except Exception:
    _HAS_PLO_MODULE = False

    def load_geom_csv(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def load_geom_yaml(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def compute_module_centres(mod_bounds_mm: np.ndarray) -> List[List[float]]:
        centres = []
        for mod in mod_bounds_mm:
            centres.append([
                float((mod[0][0] + mod[1][0]) / 2.0),
                float((mod[0][1] + mod[1][1]) / 2.0),
                float((mod[0][2] + mod[1][2]) / 2.0),
            ])
        return centres

    def transform_detector_coords(x: float, z: float, x_diff: float, x_midway: float, z_diff: float) -> Tuple[float, float]:
        x_t = x + x_diff if x < 0 else x
        x_t = 2 * x_midway - x_t if x_t > x_midway else x_t
        z_t = z + z_diff if z < 0 else z
        return x_t, z_t


def timestamped_outdir(base_dir: Path, tag: Optional[str]) -> Path:
    base_name = base_dir.name or "out"
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_name = f"outputs_{ts}" if tag is None else f"outputs_{tag}_{ts}"
    outdir = base_dir / out_name
    outdir.mkdir(parents=True, exist_ok=False)
    return outdir


def sanitize_and_load_csv(input_csv: Path, x_cols: List[str], y_cols_candidates: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(input_csv)
    # Replace inf with NaN, then drop rows missing true positions
    df = df.replace([np.inf, -np.inf], np.nan)

    # determine target y column names
    target_y = ["truex", "truey", "truez"] if {"truex", "truey", "truez"}.issubset(df.columns) else y_cols_candidates

    n_before = len(df)
    df = df.dropna(subset=target_y, how="any").reset_index(drop=True)
    n_after = len(df)
    print(f"Dropped {n_before - n_after} rows missing {target_y}")

    # set detector NaNs to zero for present columns
    det_cols_present = [c for c in x_cols if c in df.columns]
    df[det_cols_present] = df[det_cols_present].fillna(0.0)

    y_cols = ["truex", "truey", "truez"]
    # total signal and log
    df["total_signal"] = df[det_cols_present].sum(axis=1)
    n_before = len(df)
    df = df[df["total_signal"] > 0.0].reset_index(drop=True)
    n_after = len(df)
    print(f"Dropped {n_before - n_after} rows with zero total signal")
    df["log_total_signal"] = np.log10(df["total_signal"].replace(0.0, np.nan))

    return df, det_cols_present


def load_hdf_geometry(hdf5_path: Path) -> Tuple[np.ndarray, float]:
    with h5py.File(hdf5_path, "r") as f:
        try:
            mod_bounds_mm = np.array(f["geometry_info"].attrs["module_RO_bounds"])
        except Exception as exc:
            raise RuntimeError(f"Unable to read module_RO_bounds from {hdf5_path}: {exc}")
        max_drift_distance = float(f["geometry_info"].attrs["max_drift_distance"])
    return mod_bounds_mm, max_drift_distance


def compute_transform_helpers(mod_bounds_mm: np.ndarray) -> Tuple[np.ndarray, float, float]:
    # Build tpc_bounds_mm equivalent and compute x_diff, x_midway, z_diff as before
    tpc_bounds = []
    # we need max_drift_distance; input mod_bounds_mm contains module bounds pairs --- assume arranged same as earlier
    # If mod_bounds_mm shape doesn't provide max drift, caller should provide it. Here we assume mod_bounds contains pairs already adjusted.
    # For compatibility with previous script, compute xneg/xmin/x_midway as previously:
    tpc_bounds_mm = np.array(mod_bounds_mm)
    tpc_bounds_mm_x = tpc_bounds_mm[:, :, 0]
    xneg = tpc_bounds_mm_x[tpc_bounds_mm_x <= 0].min()
    xmin = tpc_bounds_mm_x[tpc_bounds_mm_x >= 0].min()
    x_diff = float(xmin - xneg)

    flat_x = np.unique(tpc_bounds_mm_x.flatten())
    positives = np.sort(flat_x[flat_x >= 0])
    if positives.size < 3:
        x_midway = float(positives.mean())
    else:
        x_second_min = float(positives[1])
        x_third_min = float(positives[2])
        x_midway = (x_second_min + x_third_min) / 2.0

    tpc_bounds_mm_z = tpc_bounds_mm[:, :, 2]
    zneg = tpc_bounds_mm_z[tpc_bounds_mm_z <= 0].min()
    zmin = tpc_bounds_mm_z[tpc_bounds_mm_z >= 0].min()
    z_diff = float(zmin - zneg)

    return tpc_bounds_mm, x_diff, x_midway, z_diff


def build_detector_positions(df_geom_transformed: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # get representative mean positions per detector (0..15)
    def mean_for(det_col: str, det_idx: int) -> float:
        vals = df_geom_transformed.loc[df_geom_transformed["Detector"] == det_idx, det_col]
        if vals.empty:
            return 0.0
        return float(vals.mean())

    det_pos_x = np.array([mean_for("x_transformed", d) for d in range(16)], dtype=float)
    det_pos_y = np.array([mean_for("y_offset", d) for d in range(16)], dtype=float)
    det_pos_z = np.array([mean_for("z_translated", d) for d in range(16)], dtype=float)

    # if any NaNs, turn to 0
    if np.isnan(det_pos_x).any() or np.isnan(det_pos_y).any() or np.isnan(det_pos_z).any():
        print("Warning: filling missing detector position means with 0.0")
        det_pos_x = np.nan_to_num(det_pos_x, nan=0.0)
        det_pos_y = np.nan_to_num(det_pos_y, nan=0.0)
        det_pos_z = np.nan_to_num(det_pos_z, nan=0.0)

    return det_pos_x, det_pos_y, det_pos_z


def compute_predictions_vectorized(
    df: pd.DataFrame,
    det_cols: List[str],
    det_pos_x: np.ndarray,
    det_pos_y: np.ndarray,
    det_pos_z: np.ndarray,
) -> pd.DataFrame:
    n = len(df)
    det_matrix = df[[f"det_{i}_max" for i in range(16)]].fillna(0.0).values.astype(float)
    # PDE-corrected: compute efficiency per detector using df_geom if available in df (we assume scaled columns already added if necessary)
    scaled_cols = [f"det_{i}_max_scaled" for i in range(16)]
    if all(c in df.columns for c in scaled_cols):
        det_matrix_scaled = df[scaled_cols].fillna(0.0).values.astype(float)
    else:
        # if scaled not present, fallback to det_matrix (i.e. uncorrected)
        det_matrix_scaled = det_matrix.copy()

    def weighted_mean_positions(mat):
        totals = mat.sum(axis=1, keepdims=True)  # shape (n,1)
        zero_mask = (totals.squeeze() == 0.0)
        totals[zero_mask, :] = 1.0  # avoid division by zero
        weights = mat / totals
        px = weights.dot(det_pos_x)
        py = weights.dot(det_pos_y)
        pz = weights.dot(det_pos_z)
        # set rows that originally had zero total to NaN
        px[zero_mask] = np.nan
        py[zero_mask] = np.nan
        pz[zero_mask] = np.nan
        return px, py, pz

    px_s, py_s, pz_s = weighted_mean_positions(det_matrix_scaled)
    px_u, py_u, pz_u = weighted_mean_positions(det_matrix)

    df = df.copy()
    df["predx"] = px_s
    df["predy"] = py_s
    df["predz"] = pz_s
    df["predx_uw"] = px_u
    df["predy_uw"] = py_u
    df["predz_uw"] = pz_u

    return df


def compute_residuals_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    # residuals (PDE corrected)
    dx = (df["predx"] - df["truex"]).to_numpy(dtype=float)
    dy = (df["predy"] - df["truey"]).to_numpy(dtype=float)
    dz = (df["predz"] - df["truez"]).to_numpy(dtype=float)
    dr = np.sqrt(dx**2 + dy**2 + dz**2)

    dx_uw = (df["predx_uw"] - df["truex"]).to_numpy(dtype=float)
    dy_uw = (df["predy_uw"] - df["truey"]).to_numpy(dtype=float)
    dz_uw = (df["predz_uw"] - df["truez"]).to_numpy(dtype=float)
    dr_uw = np.sqrt(dx_uw**2 + dy_uw**2 + dz_uw**2)

    def moments(arr: np.ndarray) -> Dict[str, float]:
        # ignore NaNs
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return {"mu": float("nan"), "sigma": float("nan")}
        return {"mu": float(np.mean(arr)), "sigma": float(np.std(arr, ddof=0))}

    metrics = {
        "dx": moments(dx),
        "dy": moments(dy),
        "dz": moments(dz),
        "dr": moments(dr),
        "dx_uw": moments(dx_uw),
        "dy_uw": moments(dy_uw),
        "dz_uw": moments(dz_uw),
        "dr_uw": moments(dr_uw),
    }
    return metrics


def save_metrics(metrics: Dict, outdir: Path, filename: str = "residuals_metrics.json") -> None:
    with open(outdir / filename, "w") as f:
        json.dump(metrics, f, indent=2)


def plot_and_optionally_save_all(df: pd.DataFrame, det_pos: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                 xlim: Tuple[float, float], ylim: Tuple[float, float], zlim: Tuple[float, float],
                                 outdir: Optional[Path] = None, event_idx: int = 1) -> None:
    det_pos_x, det_pos_y, det_pos_z = det_pos

    # 1) Histograms of true positions + log_total_signal
    fig1, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].hist(df["truex"].dropna(), bins=50, color="C0", density=True)
    axs[0, 0].set(title="True X Position Distribution", xlabel="True X (mm)", ylabel="Density")
    axs[0, 1].hist(df["truey"].dropna(), bins=50, color="C0", density=True)
    axs[0, 1].set(title="True Y Position Distribution", xlabel="True Y (mm)", ylabel="Density")
    axs[1, 0].hist(df["truez"].dropna(), bins=50, color="C0", density=True)
    axs[1, 0].set(title="True Z Position Distribution", xlabel="True Z (mm)", ylabel="Density")
    axs[1, 1].hist(df["log_total_signal"].dropna(), bins=50, color="C0", density=True)
    axs[1, 1].set(title="Log Total Signal", xlabel="log10(total signal)", ylabel="Density")
    fig1.tight_layout()
    if outdir:
        fig1.savefig(outdir / "true_positions_and_signal_hist.png", dpi=150)
    plt.close(fig1)

    # 2) X/Y/Z distribution comparing true/pred/pred_uw
    fig2, axs = plt.subplots(1, 3, figsize=(15, 5))
    xbins = np.linspace(-10, 40, 50)
    axs[0].hist(df["truex"].dropna(), bins=xbins, alpha=0.5, label="True", color="C0", density=True)
    axs[0].hist(df["predx"].dropna(), bins=xbins, alpha=0.5, label="Pred (scaled)", color="C1", density=True)
    axs[0].hist(df["predx_uw"].dropna(), bins=xbins, alpha=0.4, label="Pred (uncorr)", color="C2", density=True)
    axs[0].set(title="X Position", xlabel="X (mm)")
    axs[0].legend()
    ybins = np.linspace(-70, 70, 50)
    axs[1].hist(df["truey"].dropna(), bins=ybins, alpha=0.5, label="True", color="C0", density=True)
    axs[1].hist(df["predy"].dropna(), bins=ybins, alpha=0.5, label="Pred (scaled)", color="C1", density=True)
    axs[1].hist(df["predy_uw"].dropna(), bins=ybins, alpha=0.4, label="Pred (uncorr)", color="C2", density=True)
    axs[1].set(title="Y Position", xlabel="Y (mm)")
    axs[1].legend()
    zbins = np.linspace(-10, 80, 50)
    axs[2].hist(df["truez"].dropna(), bins=zbins, alpha=0.5, label="True", color="C0", density=True)
    axs[2].hist(df["predz"].dropna(), bins=zbins, alpha=0.5, label="Pred (scaled)", color="C1", density=True)
    axs[2].hist(df["predz_uw"].dropna(), bins=zbins, alpha=0.4, label="Pred (uncorr)", color="C2", density=True)
    axs[2].set(title="Z Position", xlabel="Z (mm)")
    axs[2].legend()
    fig2.tight_layout()
    if outdir:
        fig2.savefig(outdir / "xyz_distributions.png", dpi=150)
    plt.close(fig2)

    # 3) Residual histograms and r
    residuals = df[["predx", "predy", "predz"]].values - df[["truex", "truey", "truez"]].values
    residuals_uw = df[["predx_uw", "predy_uw", "predz_uw"]].values - df[["truex", "truey", "truez"]].values
    dx, dy, dz = residuals.T
    dx_uw, dy_uw, dz_uw = residuals_uw.T
    dr = np.sqrt(dx**2 + dy**2 + dz**2)
    dr_uw = np.sqrt(dx_uw**2 + dy_uw**2 + dz_uw**2)

    fig3, axs = plt.subplots(1, 3, figsize=(15, 5))
    bins = 100
    axs[0].hist(dx[~np.isnan(dx)], bins=bins, alpha=0.7, color="C0", label=f"PDE corr std={np.nanstd(dx):.2f}")
    axs[0].hist(dx_uw[~np.isnan(dx_uw)], bins=bins, alpha=0.5, color="C1", label=f"uncorr std={np.nanstd(dx_uw):.2f}")
    axs[0].set(xlabel="dx (mm)", ylabel="Count")
    axs[0].legend()
    axs[1].hist(dy[~np.isnan(dy)], bins=bins, alpha=0.7, color="C0", label=f"PDE corr std={np.nanstd(dy):.2f}")
    axs[1].hist(dy_uw[~np.isnan(dy_uw)], bins=bins, alpha=0.5, color="C1", label=f"uncorr std={np.nanstd(dy_uw):.2f}")
    axs[1].set(xlabel="dy (mm)")
    axs[1].legend()
    axs[2].hist(dz[~np.isnan(dz)], bins=bins, alpha=0.7, color="C0", label=f"PDE corr std={np.nanstd(dz):.2f}")
    axs[2].hist(dz_uw[~np.isnan(dz_uw)], bins=bins, alpha=0.5, color="C1", label=f"uncorr std={np.nanstd(dz_uw):.2f}")
    axs[2].set(xlabel="dz (mm)")
    axs[2].legend()
    fig3.tight_layout()
    if outdir:
        fig3.savefig(outdir / "residuals_xyz_hist.png", dpi=150)
    plt.close(fig3)

    fig4 = plt.figure(figsize=(6, 5))
    plt.hist(dr[~np.isnan(dr)], bins=100, alpha=0.7, color="C0", label=f"PDE corr std={np.nanstd(dr):.2f}")
    plt.hist(dr_uw[~np.isnan(dr_uw)], bins=100, alpha=0.5, color="C1", label=f"uncorr std={np.nanstd(dr_uw):.2f}")
    plt.xlabel("r (mm)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    if outdir:
        fig4.savefig(outdir / "residuals_r_hist.png", dpi=150)
    plt.close(fig4)

    # 4) Optional interactive 3D plot for a chosen event using plotly; save as HTML if outdir is given.
    try:
        import plotly.graph_objects as go  # local import since it's optional
    except Exception:
        print("plotly not available; skipping interactive 3D event plots")
        return

    def plot_event_3d_html(df_t: pd.DataFrame, event_row: pd.Series, tpc_bounds_mm: np.ndarray,
                           x_diff: float, x_midway: float, z_diff: float, outpath: Optional[Path] = None):
        fig = go.Figure()
        # detectors (df_t contains detectors for TPC 1 ordering as used earlier)
        detectors_per_row = df_t["Detector"].astype(int).tolist()
        det_vals = {i: float(event_row.get(f"det_{i}_max_scaled", 0.0)) for i in range(16)}
        raw_sizes = np.array([det_vals.get(d, 0.0) for d in detectors_per_row], dtype=float)
        if raw_sizes.size == 0:
            sizes_scaled = np.array([], dtype=float)
        else:
            if raw_sizes.max() > raw_sizes.min():
                sizes_scaled = 2.0 + (raw_sizes - raw_sizes.min()) / (raw_sizes.max() - raw_sizes.min()) * (20.0 - 2.0)
            else:
                sizes_scaled = np.full_like(raw_sizes, 10.0)

        fig.add_trace(go.Scatter3d(
            x=df_t["x_transformed"],
            y=df_t["y_offset"],
            z=df_t["z_translated"],
            mode="markers",
            marker=dict(size=sizes_scaled, color="lightgray", opacity=0.9, line=dict(width=0.5, color="black")),
            name="Detectors"
        ))

        # Add TPC boxes (similar to earlier implementation)
        edges_idx = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        for tpc_idx, bounds in enumerate(tpc_bounds_mm):
            c = colors[tpc_idx % len(colors)]
            (x_min, y_min, z_min), (x_max, y_max, z_max) = bounds
            corners = [
                (x_min, y_min, z_min), (x_min, y_min, z_max),
                (x_min, y_max, z_min), (x_min, y_max, z_max),
                (x_max, y_min, z_min), (x_max, y_min, z_max),
                (x_max, y_max, z_min), (x_max, y_max, z_max),
            ]
            transformed = [(*transform_detector_coords(xc, zc, x_diff, x_midway, z_diff), yc) for (xc, yc, zc) in corners]
            transformed = [(tx, ty, tz) for (tx, tz, ty) in transformed]  # adjust tuple order
            first = True
            for i, j in edges_idx:
                xi, yi, zi = transformed[i]
                xj, yj, zj = transformed[j]
                fig.add_trace(go.Scatter3d(x=[xi, xj], y=[yi, yj], z=[zi, zj], mode="lines",
                                           line=dict(color=c, width=3), opacity=0.6, name=f"TPC {tpc_idx}" if first else None,
                                           showlegend=first))
                first = False

        # True and reconstructed points
        fig.add_trace(go.Scatter3d(
            x=[float(event_row["truex"])], y=[float(event_row["truey"])], z=[float(event_row["truez"])],
            mode="markers", marker=dict(size=8, color="blue"), name="True"
        ))
        fig.add_trace(go.Scatter3d(
            x=[float(event_row["predx"])], y=[float(event_row["predy"])], z=[float(event_row["predz"])],
            mode="markers", marker=dict(size=8, color="red"), name="Predicted"
        ))

        fig.update_layout(scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)", aspectmode="data"),
                          margin=dict(l=0, r=0, b=0, t=30))
        if outpath:
            fig.write_html(str(outpath))
        return fig

    # Select df_t for TPC 1 for plotting detectors (consistent with earlier code)
    # If geometry DataFrame was not kept, attempt to reconstruct from df by merging; else skip interactive
    # Here we attempt to reconstruct df_t from columns expected in the df (if present)
    geom_cols_needed = {"Detector", "x_transformed", "y_offset", "z_translated", "TPC"}
    df_t_candidates = None
    # If those cols exist in the original df, use them; else user should have loaded df_geom and df_geom_transformed previously
    # For portability, we check the global namespace for a df_geom_transformed variable (not ideal but pragmatic)
    if "df_geom_transformed" in globals() and isinstance(globals()["df_geom_transformed"], pd.DataFrame):
        df_t_candidates = globals()["df_geom_transformed"]
    else:
        # Try to derive a minimal df_t from detector positions we used
        # As long as det_pos arrays are provided, create a simple df_t with Detector 0..15
        det_idx = np.arange(16)
        df_t_candidates = pd.DataFrame({
            "Detector": det_idx,
            "x_transformed": det_pos_x,
            "y_offset": det_pos_y,
            "z_translated": det_pos_z,
            "TPC": [1] * 16
        })

    # pick event row and produce html
    if event_idx < 0 or event_idx >= len(df):
        event_idx = 0
    event_row = df.iloc[event_idx]
    html_out = None
    if outdir:
        html_out = outdir / f"event_{event_idx:04d}_3d.html"

    tpc_bounds_mm = load_or_build_tpc_bounds(out_dir / "geom", args.fallback_hdf5)
    logging.info("Loaded TPC bounds: %s", (out_dir / "geom" / "tpc_bounds_mm.npy"))
    plot_event_3d_html(df_t_candidates[df_t_candidates["TPC"] == 1].reset_index(drop=True), event_row,
                       tpc_bounds_mm, x_diff, x_midway, z_diff, outpath=html_out)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Analytic spatial reconstruction diagnostics")
  p.add_argument("input_csv", type=Path, help="Processed CSV with truth and det_*_max columns")
  p.add_argument("--geom-csv", type=Path, default=Path("../lrs_sanity_check/geom_files/light_module_desc-4.0.0.csv"),
           help="Geometry CSV describing detectors (default: repository relative)")
  p.add_argument("--geom-yaml", type=Path, default=Path("../lrs_sanity_check/geom_files/light_module_desc-4.0.0.yaml"),
           help="Geometry YAML with tpc_center_offset and det_center")
  p.add_argument("--hdf5", type=Path, default=Path("/global/cfs/cdirs/dune/www/data/2x2/simulation/productions/MiniRun6.5_1E19_RHC/MiniRun6.5_1E19_RHC.flow/FLOW/0000000/MiniRun6.5_1E19_RHC.flow.0000000.FLOW.hdf5"),
           help="Reference HDF5 to obtain module bounds and max drift (default: cluster path)")
  p.add_argument("--save", action="store_true", help="(legacy) Save both figures and metrics to timestamped directory")
  p.add_argument("--tag", type=str, default=None, help="Optional tag appended to output directory name")
  p.add_argument("--event-idx", type=int, default=1, help="Event index to visualize in 3D")
  p.add_argument("--outdir", type=Path, default=None, help="Explicit output base directory (overrides input dir use)")

  # PDE use flag: mutually exclusive; default = use PDE corrections
  group = p.add_mutually_exclusive_group()
  group.add_argument("--use-pde", dest="use_pde", action="store_true", help="Use PDE corrections (scaled det columns) (default)")
  group.add_argument("--no-pde", dest="use_pde", action="store_false", help="Disable PDE corrections and use uncorrected detector weights")
  p.set_defaults(use_pde=True)

  return p.parse_args(argv)

def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    # basic settings
    x_cols = [f"det_{i}_max" for i in range(16)]
    y_cols_candidates = ["truex", "truey", "truez"]

    # Load and sanitize CSV
    df, det_cols_present = sanitize_and_load_csv(args.input_csv, x_cols, y_cols_candidates)

    # Load geometry CSV and HDF5 geometry info
    df_geom = load_geom_csv(str(args.geom_csv))
    df_geom["mod"] = df_geom["TPC"] // 2

    mod_bounds_mm, max_drift_distance = load_hdf_geometry(args.hdf5)

    # Derive tpc bounds & transform helpers
    tpc_bounds_mm, x_diff, x_midway, z_diff = compute_transform_helpers(mod_bounds_mm)

    # Compute module centres
    module_centres = compute_module_centres(mod_bounds_mm)

    # Annotate geometry DataFrame with offsets (use YAML offsets)
    geom_data = load_geom_yaml(str(args.geom_yaml))
    for idx, row in df_geom.iterrows():
        det_num = int(row["Detector"])
        tpc_num = int(row["TPC"])
        mod_num = int(row["mod"])
        tpc_centre_offset = geom_data["tpc_center_offset"][tpc_num]
        det_centre = geom_data["det_center"][det_num]
        x_offset = det_centre[0] + tpc_centre_offset[0] + module_centres[mod_num][0]
        y_offset = det_centre[1] + tpc_centre_offset[1] + module_centres[mod_num][1]
        z_offset = det_centre[2] + tpc_centre_offset[2] + module_centres[mod_num][2]
        df_geom.at[idx, "x_offset"] = x_offset
        df_geom.at[idx, "y_offset"] = y_offset
        df_geom.at[idx, "z_offset"] = z_offset

    # Transform geometry coordinates similar to earlier script
    df_geom_transformed = df_geom.copy()
    df_geom_transformed["x_translated"] = df_geom_transformed["x_offset"].apply(lambda x: x + x_diff if x < 0 else x)
    df_geom_transformed["x_transformed"] = df_geom_transformed["x_translated"].apply(
        lambda x: 2 * x_midway - x if x > x_midway else x
    )
    df_geom_transformed["z_translated"] = df_geom_transformed["z_offset"].apply(lambda z: z + z_diff if z < 0 else z)

    # Compute EFF_16 per detector based on TrapType if present in df_geom_transformed
    eff_list = []
    for d in range(16):
        subset = df_geom_transformed.loc[df_geom_transformed["Detector"] == d]
        if not subset.empty and "TrapType" in subset.columns:
            eff = 0.006 if int(subset["TrapType"].values[0]) != 0 else 0.002
        else:
            eff = 0.006
        eff_list.append(eff)
    EFF_16 = np.array(eff_list, dtype=float).reshape(-1, 1)

    # Add scaled detector columns to df
    for i in range(16):
        col = f"det_{i}_max"
        scaled_col = f"det_{i}_max_scaled"
        if col in df.columns:
            df[scaled_col] = df[col].astype(float) / float(EFF_16[i][0])
        else:
            df[scaled_col] = 0.0

    # Build detector position arrays (det_pos_x/y/z)
    det_pos_x, det_pos_y, det_pos_z = build_detector_positions(df_geom_transformed)

    # Vectorized predictions
    df = compute_predictions_vectorized(df, det_cols_present, det_pos_x, det_pos_y, det_pos_z)

    # Compute metrics and save if requested
    metrics = compute_residuals_metrics(df)

    # Determine output directory if saving requested
    outdir: Optional[Path] = None
    if args.save:
        if args.outdir:
            base = Path(args.outdir)
        else:
            base = args.input_csv.parent
        # Pass a Path to timestamped_outdir; include optional analytic_reco_ prefix by
        # placing the timestamped output directory under a prefixed directory inside base.
        prefixed_base = base.parent / f"analytic_reco_{base.name}"
        outdir = timestamped_outdir(prefixed_base, args.tag)
        print("Saving outputs to:", outdir)
        # Save a copy of the processed dataframe for traceability
        df.to_csv(outdir / "processed_df.csv.gz", index=False, compression="gzip")
        # Save metrics JSON
        save_metrics(metrics, outdir)

    # prepare bounds for plotting (these are from original script variables)
    xlim = (2.91024995, 33.34124995)
    ylim = (-62.07600021, 62.07600021)
    zlim = (2.46200013, 64.53800201)

    # Plot and optionally save figures
    plot_and_optionally_save_all(df, (det_pos_x, det_pos_y, det_pos_z), xlim, ylim, zlim, outdir=outdir, event_idx=args.event_idx)

    # Always print metrics summary
    print("Residual metrics (mu, sigma):")
    for k, v in metrics.items():
        print(f"  {k}: mu={v['mu']:.4g}, sigma={v['sigma']:.4g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
