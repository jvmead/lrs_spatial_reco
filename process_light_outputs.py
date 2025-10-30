#!/usr/bin/env python3
"""
Process and transform truth/reco match CSVs for visualization and export.

Features:
 - CLI args for common toggles
 - Robust filesystem handling with pathlib
 - Vectorized coordinate transforms (no DataFrame.apply)
 - Optional regeneration of TPC bounds from an example HDF5
 - Logging instead of prints

 Example usage:
 python process_light_outputs.py --n 100 --nint1 --mod123 --dE-weight

 """

from pathlib import Path
import argparse
import logging
import time
import json

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

# Import transform and plotting utilities
from utils import (
    transform_positions as utils_transform_positions,
    compute_align_params,
    transform_geom,
    load_geom_csv,
    compute_detector_offsets,
    load_hdf_geometry,
    compute_module_centres,
)
from plotting import plot_spatial_distributions, plot_3d_event_displays, plot_detector_log_signals


def load_or_build_tpc_bounds(out_geom_dir, fallback_hdf5):
    # type: (Path, Path) -> np.ndarray
    """Load or build TPC bounds array [cm units despite 'mm' in filename]."""
    out_geom_dir.mkdir(parents=True, exist_ok=True)
    tpc_path = out_geom_dir / "tpc_bounds_mm.npy"
    if tpc_path.exists():
        return np.load(tpc_path)
    if not fallback_hdf5.exists():
        raise FileNotFoundError(f"No tpc bounds and fallback HDF5 not found: {fallback_hdf5}")
    with h5py.File(fallback_hdf5, "r") as f:
        mod_bounds_mm = np.array(f["geometry_info"].attrs["module_RO_bounds"])
        max_drift_distance = float(f["geometry_info"].attrs["max_drift_distance"])
        tpc_bounds = []
        for mod in mod_bounds_mm:
            x_min, x_max = float(mod[0][0]), float(mod[1][0])
            y_min, y_max = float(mod[0][1]), float(mod[1][1])
            z_min, z_max = float(mod[0][2]), float(mod[1][2])
            # two TPC regions per module (original logic)
            x_min_adj = x_max - max_drift_distance
            x_max_adj = x_min + max_drift_distance
            tpc_bounds.append(((x_min_adj, y_min, z_min), (x_max, y_max, z_max)))
            tpc_bounds.append(((x_min, y_min, z_min), (x_max_adj, y_max, z_max)))
        tpc_bounds = np.array(tpc_bounds, dtype=float)
        np.save(tpc_path, tpc_bounds)
        return tpc_bounds


def transform_positions(df: pd.DataFrame,
                        tpc_bounds_mm: np.ndarray,
                        x_col: str,
                        y_col: str,
                        z_col: str) -> pd.DataFrame:
    """
    LEGACY WRAPPER: Calls utils.transform_positions() for consistency.
    Kept for backward compatibility but delegates to the canonical implementation.
    """
    return utils_transform_positions(
        df=df,
        tpc_bounds_mm=tpc_bounds_mm,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        out_x="truex",
        out_y="truey",
        out_z="truez",
        return_params=False,
    )


def build_output_dir(base: Path, n: int, nint1: bool, mod123: bool, dE_weight: bool) -> Path:
    name = f"processed_outputs_n{n}"
    if nint1:
        name += "_nint1"
    if mod123:
        name += "_mod123"
    if dE_weight:
        name += "_dEweight"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out = base / f"{name}_{timestamp}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "geom").mkdir(parents=True, exist_ok=True)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--input-path", type=Path,
                   default=Path("/global/homes/j/jvmead/dune/light_file_processing/outputs/gnn_training/"))
    p.add_argument("--match-filename", type=str, default=None)
    p.add_argument("--out-base", type=Path,
                   default=Path("/global/homes/j/jvmead/dune/lrs_spatial_reco/"))
    p.add_argument("--fallback-hdf5", type=Path,
                   default=Path("/global/cfs/cdirs/dune/www/data/2x2/simulation/productions/MiniRun6.5_1E19_RHC/MiniRun6.5_1E19_RHC.flow/FLOW/0000000/MiniRun6.5_1E19_RHC.flow.0000000.FLOW.hdf5"))
    p.add_argument("--nint1", action="store_true")
    p.add_argument("--mod123", action="store_true")
    p.add_argument("--dE-weight", dest="dE_weight", action="store_true")
    p.add_argument("--no-plots", dest="print_plots", action="store_false")
    p.add_argument("--geom-csv", type=Path,
                   default=Path("/global/homes/j/jvmead/dune/lrs_sanity_check/geom_files/light_module_desc-4.0.0.csv"),
                   help="Path to detector geometry CSV")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    n = args.n
    input_path = args.input_path
    match_filename = args.match_filename or str(input_path / f"truth_reco_match_n{n}.csv")
    df = pd.read_csv(match_filename)
    if df.empty:
        logging.warning("Input CSV is empty: %s", match_filename)
        return
    logging.info("Loaded %d rows from %s", len(df), match_filename)

    out_dir = build_output_dir(args.out_base, n, args.nint1, args.mod123, args.dE_weight)
    logging.info("Output directory: %s", out_dir)

    # load or build tpc bounds
    tpc_bounds_mm = load_or_build_tpc_bounds(out_dir / "geom", args.fallback_hdf5)
    logging.info("Loaded TPC bounds: %s", (out_dir / "geom" / "tpc_bounds_mm.npy"))

    # filtering
    if args.nint1:
        df = df[df["n_int_per_tpc"] == 1]
        logging.info("Filtered to n_int_per_tpc == 1 -> %d rows", len(df))
    if args.mod123:
        df = df[df["tpc_num"] > 1]
        logging.info("Filtered to tpc_num > 1 -> %d rows", len(df))

    # choose coordinate columns
    if args.dE_weight:
        x_col, y_col, z_col = "x_wmean_int", "y_wmean_int", "z_wmean_int"
    else:
        x_col, y_col, z_col = "x_mean_int", "y_mean_int", "z_mean_int"

    # ensure columns exist
    for col in (x_col, y_col, z_col):
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in CSV")

    df_new = transform_positions(df, tpc_bounds_mm, x_col, y_col, z_col)

    # Compute total_signal from detector amplitudes if det_#_max columns exist
    det_cols = [f"det_{i}_max" for i in range(16)]
    if all(col in df_new.columns for col in det_cols):
        df_new["total_signal"] = df_new[det_cols].sum(axis=1)
        logging.info("Computed total_signal from detector amplitudes")
    else:
        logging.warning("Not all det_#_max columns found; skipping total_signal computation")

    # save processed CSV
    out_csv = out_dir / "truth_reco_processed.csv"
    df_new.to_csv(out_csv, index=False)
    logging.info("Wrote processed CSV: %s", out_csv)

    if args.print_plots:
        # Determine if total_signal exists for appropriate filename
        has_total_signal = "total_signal" in df_new.columns
        plot_name = "input_distributions.png"
        out_plot = out_dir / plot_name

        # Use generalized plotting function with original x/y/z columns and transformed truex/truey/truez
        plot_spatial_distributions(
            df=df_new,
            x_true_col=x_col,
            y_true_col=y_col,
            z_true_col=z_col,
            x_pred_col="truex",
            y_pred_col="truey",
            z_pred_col="truez",
            tpc_bounds_mm=tpc_bounds_mm,
            out_file=out_plot,
            title_prefix="",
            label_true="raw",
            label_pred="tsfm",
        )
        logging.info("Wrote histogram: %s", out_plot)

        # Plot detector log signals (8x2 layout) if detector columns exist
        det_cols = [f"det_{i}_max" for i in range(16)]
        if all(col in df_new.columns for col in det_cols):
            out_det_plot = out_dir / "detector_log_signals.png"
            plot_detector_log_signals(
                df=df_new,
                out_file=out_det_plot,
                bins=50,
            )
            logging.info("Wrote detector log signal plot: %s", out_det_plot)

        # Generate 3D event displays for min, median, max total_signal
        if has_total_signal:
            logging.info("Generating 3D event displays...")
            # Load and transform detector geometry
            df_geom = load_geom_csv(args.geom_csv)
            # Add 'mod' column (module number = TPC // 2)
            df_geom["mod"] = df_geom["TPC"] // 2

            mod_bounds_mm, _ = load_hdf_geometry(args.fallback_hdf5)
            module_centres = compute_module_centres(mod_bounds_mm)

            # Load geometry data for offsets
            import yaml
            geom_yaml_path = args.geom_csv.parent / (args.geom_csv.stem + ".yaml")
            if geom_yaml_path.exists():
                with open(geom_yaml_path, 'r') as f:
                    geom_data = yaml.safe_load(f)
            else:
                logging.warning(f"YAML geometry file not found: {geom_yaml_path}, using empty dict")
                geom_data = {}

            df_geom = compute_detector_offsets(df_geom, module_centres, geom_data)

            # Transform geometry to common frame (overwrite x/y/z_offset columns)
            align_params = compute_align_params(tpc_bounds_mm)
            df_geom = transform_geom(df_geom, align_params, "x_offset", "y_offset", "z_offset",
                                     out_x="x_offset", out_y="y_offset", out_z="z_offset")

            # Select 3 events by total_signal (min, median, max)
            # First filter: remove events with NaN true positions
            df_valid = df_new[
                df_new['truex'].notna() &
                df_new['truey'].notna() &
                df_new['truez'].notna()
            ].copy()
            frac_valid_coords = len(df_valid) / len(df_new) * 100
            logging.info(f"Filtered to events with valid true positions: {len(df_valid)} / {len(df_new)} ({frac_valid_coords:.1f}%)")

            # Second filter: remove events with zero total signal
            df_valid = df_valid[df_valid['total_signal'] > 0].copy()
            frac_nonzero_signal = len(df_valid) / len(df_new) * 100
            logging.info(f"Filtered to events with non-zero signal: {len(df_valid)} / {len(df_new)} ({frac_nonzero_signal:.1f}%)")

            sorted_by_signal = df_valid.sort_values('total_signal')
            event_indices = [
                sorted_by_signal.index[0],  # min
                sorted_by_signal.index[len(sorted_by_signal) // 2],  # median
                sorted_by_signal.index[-1],  # max
            ]

            events_by_metric = {'total_signal': event_indices}

            plot_3d_event_displays(
                df=df_new,
                df_geom=df_geom,
                tpc_bounds_mm=tpc_bounds_mm,
                align_params=align_params,
                events_by_metric=events_by_metric,
                outdir=out_dir,
                show_pred=False  # No predictions in preprocessing
            )
            logging.info("Wrote 3D event displays to: %s", out_dir)

    # save inputs to json for record-keeping
    inputs_record = {
        "n": n,
        "input_path": str(input_path),
        "match_filename": str(match_filename),
        "out_base": str(args.out_base),
        "fallback_hdf5": str(args.fallback_hdf5),
        "nint1": args.nint1,
        "mod123": args.mod123,
        "dE_weight": args.dE_weight,
        "no_plots": not args.print_plots,
    }
    with open(out_dir / "processing_inputs.json", "w") as f:
        json.dump(inputs_record, f, indent=4)
    logging.info("Wrote processing inputs record.")


if __name__ == "__main__":
    main()


