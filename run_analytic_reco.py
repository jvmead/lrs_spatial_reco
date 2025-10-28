#!/usr/bin/env python3
"""
Analytic reconstruction (cleaned) — saves only predx/predy/predz.
- `--uw` toggles whether amplitudes are unweighted (no 1/PDE scaling).
  * --uw omitted  -> scale amplitudes by 1/PDE (default: PDE-corrected)
  * --uw provided -> DO NOT scale by 1/PDE (unweighted)
- Variable names are always predx/predy/predz (no _uw suffix).
- If --save is used, only a CSV with predx/predy/predz is written.
- The output directory name reflects `_uw` vs `_pde`, but variable names do not.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import yaml

# Optional geometry helpers from a user's module.
try:
    from process_light_outputs import (
        load_geom_csv,
        load_geom_yaml,
        compute_module_centres,
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


def timestamped_outdir(base_dir: Path, mode_tag: str, tag: Optional[str]) -> Path:
    """
    Create outputs_<mode>_<opt_tag>_<UTC timestamp> under base_dir.
    Example: analytic_reco_<inputdir>/outputs_pde_mytag_20251028-101500
    """
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    parts = [f"outputs_{mode_tag}"]
    if tag:
        parts.append(tag)
    parts.append(ts)
    out_name = "_".join(parts)
    outdir = base_dir / out_name
    outdir.mkdir(parents=True, exist_ok=False)
    return outdir


def sanitize_and_load_csv(input_csv: Path, x_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(input_csv)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Ensure true coords exist
    required_truth = {"truex", "truey", "truez"}
    missing_truth = required_truth - set(df.columns)
    if missing_truth:
        raise ValueError(f"Missing required truth columns: {sorted(missing_truth)}")

    # Drop rows with missing truths
    n_before = len(df)
    df = df.dropna(subset=["truex", "truey", "truez"], how="any").reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} rows missing true positions")

    # Replace NaNs in detector channels with 0
    present_x_cols = [c for c in x_cols if c in df.columns]
    if not present_x_cols:
        raise ValueError("No detector amplitude columns like det_#_max found.")
    df[present_x_cols] = df[present_x_cols].fillna(0.0)

    # Remove zero-signal rows
    df["total_signal"] = df[present_x_cols].sum(axis=1)
    n_before = len(df)
    df = df[df["total_signal"] > 0.0].reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} rows with zero total signal")

    return df, present_x_cols


def load_hdf_geometry(hdf5_path: Path) -> Tuple[np.ndarray, float]:
    with h5py.File(hdf5_path, "r") as f:
        try:
            mod_bounds_mm = np.array(f["geometry_info"].attrs["module_RO_bounds"])
        except Exception as exc:
            raise RuntimeError(f"Unable to read module_RO_bounds from {hdf5_path}: {exc}")
        max_drift_distance = float(f["geometry_info"].attrs["max_drift_distance"])
    return mod_bounds_mm, max_drift_distance


def build_detector_positions(df_geom: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Use mean transformed positions per Detector index (0..15).
    Expects columns: Detector, x_offset, y_offset, z_offset already filled.
    """
    def mean_for(col: str, det_idx: int) -> float:
        vals = df_geom.loc[df_geom["Detector"] == det_idx, col]
        if vals.empty:
            return 0.0
        return float(vals.mean())

    det_pos_x = np.array([mean_for("x_offset", d) for d in range(16)], dtype=float)
    det_pos_y = np.array([mean_for("y_offset", d) for d in range(16)], dtype=float)
    det_pos_z = np.array([mean_for("z_offset", d) for d in range(16)], dtype=float)

    det_pos_x = np.nan_to_num(det_pos_x, nan=0.0)
    det_pos_y = np.nan_to_num(det_pos_y, nan=0.0)
    det_pos_z = np.nan_to_num(det_pos_z, nan=0.0)
    return det_pos_x, det_pos_y, det_pos_z


def compute_predictions(
    df: pd.DataFrame,
    det_cols: List[str],
    det_pos_x: np.ndarray,
    det_pos_y: np.ndarray,
    det_pos_z: np.ndarray,
    uw: bool,
    eff_per_det: Optional[np.ndarray],
) -> pd.DataFrame:
    """
    Weighted centroid using either:
    - PDE-corrected amplitudes (divide by eff): uw == False
    - Unweighted (raw amplitudes):              uw == True
    """
    det_mat = df[det_cols].values.astype(float)  # shape (n, <=16)

    # Build a fixed 16-column matrix in detector index order 0..15
    # Missing columns (not in det_cols) are treated as 0.
    n = len(df)
    full = np.zeros((n, 16), dtype=float)
    for i in range(16):
        col = f"det_{i}_max"
        if col in det_cols:
            full[:, i] = df[col].to_numpy(dtype=float, na_value=0.0)

    if uw:
        weights = full  # no scaling
    else:
        # PDE-corrected weights: divide by per-detector efficiencies
        if eff_per_det is None:
            # fallback to uniform non-zero efficiencies
            eff_per_det = np.full((16,), 1.0, dtype=float)
        eff = eff_per_det.reshape(1, 16)
        weights = np.divide(full, eff, out=np.zeros_like(full), where=eff != 0)

    totals = weights.sum(axis=1, keepdims=True)
    zero_mask = (totals.squeeze() == 0.0)
    totals[zero_mask, :] = 1.0  # avoid division by zero
    wnorm = weights / totals

    px = wnorm.dot(det_pos_x)
    py = wnorm.dot(det_pos_y)
    pz = wnorm.dot(det_pos_z)

    # set rows that originally had zero total to NaN (should be none after filtering)
    px[zero_mask] = np.nan
    py[zero_mask] = np.nan
    pz[zero_mask] = np.nan

    out = df.copy()
    out["predx"] = px
    out["predy"] = py
    out["predz"] = pz
    return out


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analytic spatial reconstruction (predx/predy/predz only)")
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

    # Unweighted (no PDE scaling) toggle
    p.add_argument("--uw", action="store_true",
                   help="Use UNWEIGHTED amplitudes (do NOT scale by 1/PDE). If omitted, amplitudes are scaled by 1/PDE.")

    return p.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    # columns expected from input
    x_cols = [f"det_{i}_max" for i in range(16)]

    # Load and sanitize CSV
    df, det_cols_present = sanitize_and_load_csv(args.input_csv, x_cols)

    # Load geometry CSV and HDF5 geometry info
    df_geom = load_geom_csv(str(args.geom_csv))
    df_geom["mod"] = df_geom["TPC"] // 2
    mod_bounds_mm, _max_drift_distance = load_hdf_geometry(args.hdf5)

    # Compute module centres & annotate offsets using YAML
    module_centres = compute_module_centres(mod_bounds_mm)
    geom_data = load_geom_yaml(str(args.geom_yaml))

    # Fill x_offset/y_offset/z_offset per geometry row
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

    # Build detector position arrays
    det_pos_x, det_pos_y, det_pos_z = build_detector_positions(df_geom)

    # Per-detector PDE efficiencies (if available, infer from TrapType like before; else default)
    eff_list = []
    for d in range(16):
        subset = df_geom.loc[df_geom["Detector"] == d]
        if not subset.empty and "TrapType" in subset.columns:
            eff = 0.006 if int(subset["TrapType"].values[0]) != 0 else 0.002
        else:
            eff = 0.006
        eff_list.append(eff)
    eff_per_det = np.array(eff_list, dtype=float)

    # Compute predictions (single set: predx/predy/predz) controlled by --uw
    df_pred = compute_predictions(
        df=df,
        det_cols=det_cols_present,
        det_pos_x=det_pos_x,
        det_pos_y=det_pos_y,
        det_pos_z=det_pos_z,
        uw=args.uw,
        eff_per_det=eff_per_det,
    )

    # Save ONLY predx/predy/predz if requested
    if args.save:
        # derive base dir
        base = Path(args.outdir) if args.outdir else args.input_csv.parent
        prefixed_base = base.parent / f"analytic_reco_{base.name}"
        mode_tag = "uw" if args.uw else "pde"
        outdir = timestamped_outdir(prefixed_base, mode_tag, args.tag)
        print("Saving outputs to:", outdir)

        # minimal artifact: just the three columns
        out_df = df_pred[["predx", "predy", "predz"]].copy()
        out_path = outdir / "predictions.csv.gz"
        out_df.to_csv(out_path, index=False, compression="gzip")
        print("Wrote:", out_path)

        # also store a tiny manifest with run metadata for provenance
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

    # Always print a small summary to stdout
    print("Sample predictions (first 5):")
    print(df_pred[["predx", "predy", "predz"]].head(5).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
