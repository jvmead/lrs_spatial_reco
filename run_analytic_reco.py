#!/usr/bin/env python3
"""
Analytic reconstruction with differential (binned) residual stats + binnings export.

Key points:
- Always computes predx/predy/predz. Variable names never get a *_uw suffix.
- `--uw` toggles amplitude weighting:
    * default (no --uw): divide by 1/PDE (PDE-corrected)
    * with    (--uw)   : do NOT divide by 1/PDE (unweighted)
- Saving predictions: ONLY the three columns predx/predy/predz.
- Differential residual μ/σ heatmap stats can be exported separately with `--export-stats`.
- When exporting heatmap stats, a `binnings.json` is also written with the bin edges for:
    truex/predx, truey/predy, truez/predz, dr, total_signal, log_total_signal.
- You can choose the heatmap binning axes (default: truex vs truez) and bin counts.

Examples:
  # PDE-corrected predictions only (no files written)
  python run_analytic_reco.py data.csv

  # Unweighted predictions, save ONLY predx/predy/predz
  python run_analytic_reco.py data.csv --uw --save

  # Export differential stats (and binnings.json) WITHOUT saving predictions
  python run_analytic_reco.py data.csv --export--stats

  # Control binning and also save heatmap images
  python run_analytic_reco.py data.csv --bins-x 30 --bins-z 40 --plots --save-plots --export-stats
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
import warnings

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
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    parts = [f"outputs_{mode_tag}"]
    if tag:
        parts.append(tag)
    parts.append(ts)
    out_name = "_".join(parts)
    outdir = base_dir / out_name
    outdir.mkdir(parents=True, exist_ok=False)
    return outdir


def ensure_outdir_for_artifacts(args, mode_tag: str) -> Path:
    """
    Create an output directory when we need to write *anything*:
    - predictions (args.save)
    - stats (args.export_stats)
    - plots (args.plots and args.save_plots)
    """
    base = Path(args.outdir) if args.outdir else args.input_csv.parent
    prefixed_base = base.parent / f"analytic_reco_{base.name}"
    outdir = timestamped_outdir(prefixed_base, mode_tag, args.tag)
    print("Saving outputs to:", outdir)
    return outdir


def sanitize_and_load_csv(input_csv: Path, x_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(input_csv)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Ensure truth exists
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

    # Compute total_signal & log
    df["total_signal"] = df[present_x_cols].sum(axis=1)
    n_before = len(df)
    df = df[df["total_signal"] > 0.0].reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} rows with zero total signal")
    df["log_total_signal"] = np.log10(df["total_signal"])

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
    full = np.zeros((len(df), 16), dtype=float)
    for i in range(16):
        col = f"det_{i}_max"
        if col in det_cols:
            full[:, i] = df[col].to_numpy(dtype=float, na_value=0.0)

    if uw:
        weights = full  # unweighted (no 1/PDE)
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


def compute_differential_stats(
    df_pred: pd.DataFrame,
    x_axis: str = "truex",
    z_axis: str = "truez",
    bins_x: int = 24,
    bins_z: int = 24,
) -> Dict[str, pd.DataFrame]:
    """
    Bin events over (x_axis, z_axis) and compute per-bin μ/σ of residuals.
    Returns tidy DataFrames with columns: bin_x_center, bin_z_center, mu, sigma
    for each of 'dx','dy','dz','dr'.
    """
    # residuals
    dx = (df_pred["predx"] - df_pred["truex"]).to_numpy(dtype=float)
    dy = (df_pred["predy"] - df_pred["truey"]).to_numpy(dtype=float)
    dz = (df_pred["predz"] - df_pred["truez"]).to_numpy(dtype=float)
    dr = np.sqrt(dx**2 + dy**2 + dz**2)

    bx = np.linspace(np.nanmin(df_pred[x_axis]), np.nanmax(df_pred[x_axis]), bins_x + 1)
    bz = np.linspace(np.nanmin(df_pred[z_axis]), np.nanmax(df_pred[z_axis]), bins_z + 1)

    # digitize once
    ix = np.digitize(df_pred[x_axis].to_numpy(dtype=float), bx) - 1
    iz = np.digitize(df_pred[z_axis].to_numpy(dtype=float), bz) - 1

    # clamp to [0, bins-1]
    ix = np.clip(ix, 0, bins_x - 1)
    iz = np.clip(iz, 0, bins_z - 1)

    # container for per-bin accumulations (indices -> list of values)
    def accum(vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = np.full((bins_x, bins_z), np.nan, dtype=float)
        sg = np.full((bins_x, bins_z), np.nan, dtype=float)
        buckets: Dict[Tuple[int, int], List[float]] = {}
        for i in range(vals.size):
            if not np.isfinite(vals[i]):
                continue
            key = (ix[i], iz[i])
            buckets.setdefault(key, []).append(float(vals[i]))
        for (i, j), arr in buckets.items():
            m = np.mean(arr)
            s = np.std(arr, ddof=0)
            mu[i, j] = m
            sg[i, j] = s
        return mu, sg

    mu_dx, sg_dx = accum(dx)
    mu_dy, sg_dy = accum(dy)
    mu_dz, sg_dz = accum(dz)
    mu_dr, sg_dr = accum(dr)

    # centers
    cx = 0.5 * (bx[:-1] + bx[1:])
    cz = 0.5 * (bz[:-1] + bz[1:])

    # melt into tidy DataFrames
    def to_df(mu: np.ndarray, sg: np.ndarray) -> pd.DataFrame:
        cx_grid, cz_grid = np.meshgrid(cx, cz, indexing="ij")
        flat = pd.DataFrame({
            "bin_x_center": cx_grid.ravel(),
            "bin_z_center": cz_grid.ravel(),
            "mu": mu.ravel(),
            "sigma": sg.ravel(),
        })
        return flat

    return {
        "dx": to_df(mu_dx, sg_dx),
        "dy": to_df(mu_dy, sg_dy),
        "dz": to_df(mu_dz, sg_dz),
        "dr": to_df(mu_dr, sg_dr),
        "edges": {"bx": bx.tolist(), "bz": bz.tolist()},  # include for convenience
    }


def build_binnings_json(
    df_pred: pd.DataFrame,
    bins_x: int,
    bins_y: int,
    bins_z: int,
    bins_r: int,
    bins_signal: int,
) -> Dict[str, List[float]]:
    """
    Build a dict of bin edges for true/pred (x,y,z), residual radius dr,
    total_signal and log_total_signal. Linear bins between min/max of each.
    """
    # residual radius
    dx = (df_pred["predx"] - df_pred["truex"]).to_numpy(dtype=float)
    dy = (df_pred["predy"] - df_pred["truey"]).to_numpy(dtype=float)
    dz = (df_pred["predz"] - df_pred["truez"]).to_numpy(dtype=float)
    dr = np.sqrt(dx**2 + dy**2 + dz**2)

    def edges(series: pd.Series, n: int) -> List[float]:
        s = series.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            return []
        mn, mx = float(s.min()), float(s.max())
        if not np.isfinite(mn) or not np.isfinite(mx) or mn == mx:
            return [mn, mx]
        return np.linspace(mn, mx, n + 1).tolist()

    binnings = {
        "truex": edges(df_pred["truex"], bins_x),
        "predx": edges(df_pred["predx"], bins_x),
        "truey": edges(df_pred["truey"], bins_y),
        "predy": edges(df_pred["predy"], bins_y),
        "truez": edges(df_pred["truez"], bins_z),
        "predz": edges(df_pred["predz"], bins_z),
        "dr": edges(pd.Series(dr), bins_r),
        "total_signal": edges(df_pred["total_signal"], bins_signal),
        "log_total_signal": edges(df_pred["log_total_signal"], bins_signal),
    }
    return binnings


def maybe_plot_heatmaps(
    stats: Dict[str, pd.DataFrame],
    title_prefix: str = "",
    save_dir: Optional[Path] = None,
    save_plots: bool = False,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        warnings.warn("matplotlib not available; skipping heatmap plots")
        return

    # 'edges' key is only metadata; skip it
    for key, df in stats.items():
        if key == "edges":
            continue
        for field in ("mu", "sigma"):
            pivot = df.pivot(index="bin_x_center", columns="bin_z_center", values=field)
            plt.figure(figsize=(6, 5))
            im = plt.imshow(
                pivot.values, aspect="auto", origin="lower",
                extent=[pivot.columns.min(), pivot.columns.max(),
                        pivot.index.min(), pivot.index.max()]
            )
            plt.xlabel("Z bin center (mm)")
            plt.ylabel("X bin center (mm)")
            plt.title(f"{title_prefix}{key} — {field}")
            plt.colorbar(im)
            if save_dir and save_plots:
                out = save_dir / f"heatmap_{key}_{field}.png"
                plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analytic spatial reconstruction with differential residual stats + binnings")
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

    # Differential stats controls
    p.add_argument("--bins-x", type=int, default=24, help="Bins for X(-axis and x-related exports)")
    p.add_argument("--bins-y", type=int, default=24, help="Bins for Y-related exports")
    p.add_argument("--bins-z", type=int, default=24, help="Bins for Z(-axis and z-related exports)")
    p.add_argument("--bins-r", type=int, default=24, help="Bins for residual radius dr")
    p.add_argument("--bins-signal", type=int, default=24, help="Bins for total_signal/log_total_signal")
    p.add_argument("--bin-x-axis", type=str, default="truex", choices=["truex", "predx"],
                   help="X-axis for heatmap binning (default: truex)")
    p.add_argument("--bin-z-axis", type=str, default="truez", choices=["truez", "predz"],
                   help="Z-axis for heatmap binning (default: truez)")
    p.add_argument("--export-stats", action="store_true",
                   help="Write CSV tables for binned μ/σ (dx,dy,dz,dr) AND a binnings.json with all bin edges.")
    p.add_argument("--plots", action="store_true", help="Render heatmap images of μ/σ (not saved unless --save-plots).")
    p.add_argument("--save-plots", action="store_true", help="Write heatmap images (requires --plots).")

    return p.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    # columns expected from input
    x_cols = [f"det_{i}_max" for i in range(16)]

    # Load and sanitize CSV
    df, det_cols_present = sanitize_and_load_csv(args.input_csv, x_cols)

    # Load geometry + build detector positions
    df_geom = load_geom_csv(str(args.geom_csv))
    df_geom["mod"] = df_geom["TPC"] // 2
    mod_bounds_mm, _max_drift_distance = load_hdf_geometry(args.hdf5)
    module_centres = compute_module_centres(mod_bounds_mm)
    geom_data = load_geom_yaml(str(args.geom_yaml))

    # Annotate offsets
    for idx, row in df_geom.iterrows():
        det_num = int(row["Detector"]); tpc_num = int(row["TPC"]); mod_num = int(row["mod"])
        tco = geom_data["tpc_center_offset"][tpc_num]
        dc = geom_data["det_center"][det_num]
        df_geom.at[idx, "x_offset"] = dc[0] + tco[0] + module_centres[mod_num][0]
        df_geom.at[idx, "y_offset"] = dc[1] + tco[1] + module_centres[mod_num][1]
        df_geom.at[idx, "z_offset"] = dc[2] + tco[2] + module_centres[mod_num][2]

    det_pos_x, det_pos_y, det_pos_z = build_detector_positions(df_geom)

    # Per-detector PDE efficiencies (TrapType heuristic fallback)
    eff_list = []
    for d in range(16):
        subset = df_geom.loc[df_geom["Detector"] == d]
        if not subset.empty and "TrapType" in subset.columns:
            eff = 0.006 if int(subset["TrapType"].values[0]) != 0 else 0.002
        else:
            eff = 0.006
        eff_list.append(eff)
    eff_per_det = np.array(eff_list, dtype=float)

    # Predictions (names are predx/predy/predz, regardless of weighting)
    df_pred = compute_predictions(
        df=df,
        det_cols=det_cols_present,
        det_pos_x=det_pos_x,
        det_pos_y=det_pos_y,
        det_pos_z=det_pos_z,
        uw=args.uw,
        eff_per_det=eff_per_det,
    )

    # Differential (binned) residual stats — always computed in-memory
    stats = compute_differential_stats(
        df_pred=df_pred,
        x_axis=args.bin_x_axis,
        z_axis=args.bin_z_axis,
        bins_x=args.bins_x,
        bins_z=args.bins_z,
    )

    # Optionally plot heatmaps
    if args.plots:
        # If saving plots is requested, ensure we have an outdir
        save_dir = None
        if args.save_plots or args.export_stats or args.save:
            mode_tag = "uw" if args.uw else "pde"
            save_dir = ensure_outdir_for_artifacts(args, mode_tag)
        maybe_plot_heatmaps(
            stats,
            title_prefix=("UW " if args.uw else "PDE ") + f"({args.bin_x_axis} vs {args.bin_z_axis}) ",
            save_dir=save_dir,
            save_plots=args.save_plots,
        )

    outdir: Optional[Path] = None
    need_outdir = bool(args.save or args.export_stats or (args.plots and args.save_plots))
    if need_outdir:
        mode_tag = "uw" if args.uw else "pde"
        outdir = ensure_outdir_for_artifacts(args, mode_tag)

    # Save ONLY predictions if requested
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

    # Export heatmap stats + binnings
    if args.export_stats and outdir is not None:
        # 1) heatmap stats (dx,dy,dz,dr)
        for key, df_stat in stats.items():
            if key == "edges":
                continue
            path = outdir / f"differential_{key}.csv.gz"
            df_stat.to_csv(path, index=False, compression="gzip")
            print("Wrote:", path)

        # 2) binnings.json — bin edges for true/pred (x,y,z), dr, total_signal, log_total_signal
        binnings = build_binnings_json(
            df_pred=df_pred,
            bins_x=args.bins_x,
            bins_y=args.bins_y,
            bins_z=args.bins_z,
            bins_r=args.bins_r,
            bins_signal=args.bins_signal,
        )

        # include the actual heatmap (x,z) edges that were used
        binnings["_heatmap_axes"] = {
            "x_axis": args.bin_x_axis,
            "z_axis": args.bin_z_axis,
            "edges_x_used": stats["edges"]["bx"],
            "edges_z_used": stats["edges"]["bz"],
        }

        with open(outdir / "binnings.json", "w") as f:
            json.dump(binnings, f, indent=2)
        print("Wrote:", outdir / "binnings.json")

    # Always print a tiny preview
    print("Sample predictions (first 5):")
    print(df_pred[["predx", "predy", "predz"]].head(5).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
