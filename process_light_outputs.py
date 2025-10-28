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
 python process_light_outputs.py --n 100 --nint1 --mod123 --dE-weight --no-plots

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

# Import transform utilities
from utils import transform_positions as utils_transform_positions


def load_or_build_tpc_bounds(out_geom_dir: Path, fallback_hdf5: Path) -> np.ndarray:
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


def make_histograms(df: pd.DataFrame,
                    tpc_bounds_mm: np.ndarray,
                    x_col: str,
                    y_col: str,
                    z_col: str,
                    out_file: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    xbins = np.linspace(-75, 75, 150)
    axes[0].hist(df[x_col], bins=xbins, alpha=0.5, color="red", label="orig")
    axes[0].hist(df["truex"], bins=xbins, alpha=0.5, color="orange", label="transformed")
    axes[0].set_title("X")
    for bounds in tpc_bounds_mm:
        axes[0].axvline(bounds[0][0], color="k", linestyle="--", alpha=0.5)
        axes[0].axvline(bounds[1][0], color="k", linestyle="--", alpha=0.5)

    ybins = np.linspace(-75, 75, 150)
    axes[1].hist(df[y_col], bins=ybins, alpha=0.5, color="green")
    axes[1].set_title("Y")
    for bounds in tpc_bounds_mm:
        axes[1].axvline(bounds[0][1], color="k", linestyle="--", alpha=0.5)
        axes[1].axvline(bounds[1][1], color="k", linestyle="--", alpha=0.5)

    zbins = np.linspace(-75, 75, 150)
    axes[2].hist(df[z_col], bins=zbins, alpha=0.5, color="blue", label="orig")
    axes[2].hist(df["truez"], bins=zbins, alpha=0.5, color="cyan", label="transformed")
    axes[2].set_title("Z")
    for bounds in tpc_bounds_mm:
        axes[2].axvline(bounds[0][2], color="k", linestyle="--", alpha=0.5)
        axes[2].axvline(bounds[1][2], color="k", linestyle="--", alpha=0.5)

    for ax in axes:
        ax.legend()
    plt.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


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

    # save processed CSV
    out_csv = out_dir / "truth_reco_processed.csv"
    df_new.to_csv(out_csv, index=False)
    logging.info("Wrote processed CSV: %s", out_csv)

    if args.print_plots:
        out_plot = out_dir / "true_position_histograms.png"
        make_histograms(df_new, tpc_bounds_mm, x_col, y_col, z_col, out_plot)
        logging.info("Wrote histogram: %s", out_plot)

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


