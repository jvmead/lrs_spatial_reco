#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

Example usage:
python run_analytic_reco.py /path/to/input.csv --outdir /path/to/output --tag mytag
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
    ensure_outdir,
    sanitize_and_load_csv,
)

# Import plotting and diagnostics from plotting module
from plotting import (
    compute_differential_stats_1d_minimal,
    plot_heatmaps_resid_vs_vars,
    plot_pred_vs_true_xyz,
    plot_1d_curves,
    plot_spatial_distributions,
    plot_residual_distributions,
    plot_residual_corner,
    plot_detector_signals,
    plot_3d_event_displays,
)

DEFAULT_1D_VARS = ["truex", "truey", "truez", "dr", "total_signal"]
HEATMAP_INDEP_VARS = ["truex", "truey", "truez", "total_signal"]

# ---------- Analytic-specific reconstruction ----------

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
    p.add_argument("--all", action="store_true",
                   help="Enable all plotting and export options (distributions, heatmaps, 1D plots/exports, 3D displays)")
    p.add_argument("--plot-2d-heatmaps", action="store_true",
                   help="Auto-generate heatmaps: residual (y) vs each of truex,truey,truez,r,total_signal (log10 bins for total_signal) and also predx/predy/predz vs truex/truey/truez with y=x overlay.")
    p.add_argument("--plot-1d-over", nargs="*", default=[],
                   help="Save 1D line plots of μ and σ vs the listed variables (same options as --export-1d-over).")
    p.add_argument("--plot-1d-all", action="store_true",
                   help="Save 1D plots for: truex truey truez dr total_signal")
    p.add_argument("--plot-distributions", action="store_true",
                   help="Plot output distributions (true vs pred for x/y/z) and residual histograms (dx/dy/dz/dr)")
    p.add_argument("--plot-3d-displays", action="store_true",
                   help="Generate 3D event displays for events with min/median/max total_signal and dr")

    return p.parse_args(argv)


# ------------------------------ Main ------------------------------

def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    # If --all is specified, enable all plotting and export options
    if args.all:
        args.plot_distributions = True
        args.plot_2d_heatmaps = True
        args.plot_1d_all = True
        args.plot_3d_displays = True
        args.export_1d_all = True

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
        z_col="z_offset",
        out_x="x_offset",
        out_y="y_offset",
        out_z="z_offset"
    )

    # Aggregate transformed detector positions (using overwritten column names)
    det_pos_x, det_pos_y, det_pos_z = detector_means_by_id(
        df_geom_transformed,
        x_col="x_offset",
        y_col="y_offset",
        z_col="z_offset"
    )
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

    # Create output directory if needed
    outdir: Optional[Path] = None
    need_outdir = bool(
        args.save or args.export_1d_over or args.export_1d_all or
        args.plot_2d_heatmaps or args.plot_1d_over or args.plot_1d_all or
        args.plot_distributions or args.plot_3d_displays
    )
    if need_outdir:
        mode_tag = "uw" if args.uw else "pde"
        outdir = ensure_outdir(args, "analytic_reco", mode_tag)

    # Create output distribution plots
    if args.plot_distributions and outdir is not None:
        # Compute residuals first
        dx = (df_pred["predx"] - df_pred["truex"]).to_numpy(dtype=float)
        dy = (df_pred["predy"] - df_pred["truey"]).to_numpy(dtype=float)
        dz = (df_pred["predz"] - df_pred["truez"]).to_numpy(dtype=float)
        dr = np.sqrt(dx**2 + dy**2 + dz**2)

        # Add to dataframe temporarily for plotting
        df_pred_temp = df_pred.copy()
        df_pred_temp['dx'] = dx
        df_pred_temp['dy'] = dy
        df_pred_temp['dz'] = dz
        df_pred_temp['dr'] = dr

        # Plot detector signal distributions (2 columns, colored by PDE)
        plot_detector_signals(
            df=df_pred_temp,
            out_file=outdir / "detector_signals.png",
            log_scale=False,
            bins=50,
            log_bins=True,
            log_xscale=True,
            pde_values=eff_per_det,  # Pass PDE values for color mapping
        )

        # Plot overlaid true/pred distributions for x, y, z
        plot_spatial_distributions(
            df=df_pred_temp,
            x_true_col="truex",
            y_true_col="truey",
            z_true_col="truez",
            x_pred_col="predx",
            y_pred_col="predy",
            z_pred_col="predz",
            tpc_bounds_mm=tpc_bounds_mm,
            out_file=outdir / "output_distributions.png",
            title_prefix="",
            label_true="true",
            label_pred="pred",
        )
        print(f"Saved: {outdir / 'output_distributions.png'}")

        # Plot residual histograms using consistent styling
        plot_residual_distributions(
            df=df_pred_temp,
            dx_col="dx",
            dy_col="dy",
            dz_col="dz",
            dr_col="dr",
            out_file=outdir / "residual_distributions.png",
            tpc_bounds_mm=tpc_bounds_mm,
            truex_col="truex",
            truey_col="truey",
            truez_col="truez",
        )
        print(f"Saved: {outdir / 'residual_distributions.png'}")

        # Plot corner plot to show residual correlations
        plot_residual_corner(
            df=df_pred_temp,
            dx_col="dx",
            dy_col="dy",
            dz_col="dz",
            dr_col="dr",
            out_file=outdir / "residual_corner.png",
        )
        print(f"Saved: {outdir / 'residual_corner.png'}")

    if args.save and outdir is not None:
        out_df = df_pred[["predx", "predy", "predz"]].copy()
        predictions_dir = outdir / "predictions"
        predictions_dir.mkdir(exist_ok=True)
        out_path = predictions_dir / "predictions.csv.gz"
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
        residuals_dir = outdir / "residuals_1d"
        residuals_dir.mkdir(exist_ok=True)
        for var in export_vars:
            if var == "tot_signal":
                var = "total_signal"
            bins = choose_bins(var)
            df1d = compute_differential_stats_1d_minimal(df_pred, var=var, bins=bins)
            fname = f"resids_f_{var}.csv.gz"
            path = residuals_dir / fname
            df1d.to_csv(path, index=False, compression="gzip")
            print("Wrote:", path)

    if args.plot_2d_heatmaps:
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
        for var in plot1d_vars:
            if var == "tot_signal":
                var = "total_signal"
            bins = choose_bins(var)
            df1d = compute_differential_stats_1d_minimal(df_pred, var=var, bins=bins)
            plot_1d_curves(df1d, outdir=outdir, var=var)

    # Generate 3D event displays if requested
    if args.plot_3d_displays:
        print("Generating 3D event displays...")
        # Compute residual r for sorting
        dx = df_pred['predx'] - df_pred['truex']
        dy = df_pred['predy'] - df_pred['truey']
        dz = df_pred['predz'] - df_pred['truez']
        df_pred['dr'] = np.sqrt(dx**2 + dy**2 + dz**2)

        # Select events by total_signal (min, median, max)
        sorted_by_signal = df_pred.sort_values('total_signal')
        signal_indices = [
            sorted_by_signal.index[0],
            sorted_by_signal.index[len(sorted_by_signal) // 2],
            sorted_by_signal.index[-1],
        ]

        # Select events by residual dr (min, median, max)
        sorted_by_dr = df_pred.sort_values('dr')
        dr_indices = [
            sorted_by_dr.index[0],
            sorted_by_dr.index[len(sorted_by_dr) // 2],
            sorted_by_dr.index[-1],
        ]

        events_by_metric = {
            'total_signal': signal_indices,
            'dr': dr_indices,
        }

        # Load TPC bounds for transforms
        # Try to extract from CSV's parent directory (default location from process_light_outputs.py)
        csv_path = Path(args.input_csv)
        tpc_bounds_path = csv_path.parent / "geom" / "tpc_bounds_mm.npy"
        if tpc_bounds_path.exists():
            tpc_bounds_mm = np.load(tpc_bounds_path)
        else:
            print(f"Warning: TPC bounds not found at {tpc_bounds_path}, skipping 3D event displays")
            tpc_bounds_mm = None

        if tpc_bounds_mm is not None:
            align_params = compute_align_params(tpc_bounds_mm)

            # Use UNTRANSFORMED geometry - plotting function will apply transforms
            plot_3d_event_displays(
                df=df_pred,
                df_geom=df_geom,  # Pass untransformed geometry
                tpc_bounds_mm=tpc_bounds_mm,
                align_params=align_params,
                events_by_metric=events_by_metric,
                outdir=outdir,
                show_pred=True
            )
            print(f"Wrote 3D event displays to: {outdir}/event_displays_3d/")

    print("Sample predictions (first 5 rows):")
    print(df_pred[["predx", "predy", "predz"]].head(5).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
