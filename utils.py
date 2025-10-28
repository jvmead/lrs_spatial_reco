# utils.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Dict, Optional, List

import numpy as np
import pandas as pd
import h5py
import yaml


@dataclass(frozen=True)
class AlignParams:
    """
    Parameters defining the detector->common-frame transform.

    x_diff   : shift applied to negative-x side before reflection
    x_midway : midpoint used to reflect x > x_midway back into the common frame
    z_diff   : shift applied to negative-z side
    """
    x_diff: float
    x_midway: float
    z_diff: float


# --------------------------- param computation ---------------------------

def compute_align_params(tpc_bounds_mm: np.ndarray) -> AlignParams:
    """
    Derive alignment parameters (x_diff, x_midway, z_diff) from TPC module bounds.

    Expected shape: (n_tpc, 2, 3) where each TPC has (min_corner, max_corner) in (x,y,z) [cm].
    Robust to extra duplicates; uses unique coordinate values.

    x logic:
      - Find minimal negative x (xneg) and minimal non-negative x (xmin)
        -> x_diff = xmin - xneg (shifts negative side onto positive side).
      - For the midpoint used for reflection:
          * if >= 3 non-negative unique x values exist, use mean of 2nd and 3rd smallest
            (this preserves the legacy behavior from your script);
          * otherwise use the mean of all non-negative unique x values.

    z logic:
      - z_diff shifts negative z to align with the positive side:
        z_diff = min(z >= 0) - min(z <= 0)
    """
    if tpc_bounds_mm.ndim != 3 or tpc_bounds_mm.shape[-1] != 3:
        raise ValueError("tpc_bounds_mm must have shape (n_tpc, 2, 3)")

    # X parameters
    tpc_x = tpc_bounds_mm[:, :, 0]
    unique_x = np.unique(tpc_x.flatten())
    neg_x = unique_x[unique_x <= 0]
    pos_x = unique_x[unique_x >= 0]
    if neg_x.size == 0 or pos_x.size == 0:
        raise RuntimeError("Unexpected TPC x bounds; cannot compute x alignment.")

    xneg = float(neg_x.min())
    xmin = float(pos_x.min())
    x_diff = xmin - xneg

    x_sorted_pos = np.sort(pos_x)
    if x_sorted_pos.size >= 3:
        x_midway = float((x_sorted_pos[1] + x_sorted_pos[2]) / 2.0)
    else:
        x_midway = float(x_sorted_pos.mean())

    # Z parameter
    tpc_z = tpc_bounds_mm[:, :, 2]
    unique_z = np.unique(tpc_z.flatten())
    neg_z = unique_z[unique_z <= 0]
    pos_z = unique_z[unique_z >= 0]
    if neg_z.size == 0 or pos_z.size == 0:
        raise RuntimeError("Unexpected TPC z bounds; cannot compute z alignment.")
    zneg = float(neg_z.min())
    zmin = float(pos_z.min())
    z_diff = zmin - zneg

    return AlignParams(x_diff=x_diff, x_midway=float(x_midway), z_diff=float(z_diff))


# --------------------------- core transforms ---------------------------

def transform_arrays(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    params: AlignParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized transform for numpy arrays. Implements the same logic as transform_positions():
      x' = reflect( shift_negatives(x, x_diff), x_midway )
      z' = shift_negatives(z, z_diff)
      y' = y  (pass-through)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    # shift negatives, then reflect over x_midway
    x_shifted = np.where(x < 0.0, x + params.x_diff, x)
    x_true = np.where(x_shifted > params.x_midway, 2.0 * params.x_midway - x_shifted, x_shifted)

    # z shift
    z_true = np.where(z < 0.0, z + params.z_diff, z)

    return x_true, y, z_true


def transform_positions(
    df: pd.DataFrame,
    tpc_bounds_mm: np.ndarray,
    x_col: str,
    y_col: str,
    z_col: str,
    out_x: str = "truex",
    out_y: str = "truey",
    out_z: str = "truez",
    return_params: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, AlignParams]:
    """
    Generalized DataFrame transform:
      - derives AlignParams from tpc_bounds_mm
      - applies transform to df[x_col], df[y_col], df[z_col]
      - writes to new columns (out_x/out_y/out_z)
      - returns df (and optionally the AlignParams)

    Use the returned AlignParams to transform geometry consistently (see transform_geom()).
    """
    params = compute_align_params(tpc_bounds_mm)
    x_true, y_true, z_true = transform_arrays(df[x_col].to_numpy(), df[y_col].to_numpy(), df[z_col].to_numpy(), params)

    df_out = df.copy()
    df_out[out_x] = x_true
    df_out[out_y] = y_true
    df_out[out_z] = z_true

    return (df_out, params) if return_params else df_out


def apply_transform(
    df: pd.DataFrame,
    params: AlignParams,
    x_col: str,
    y_col: str,
    z_col: str,
    out_x: str,
    out_y: str,
    out_z: str,
) -> pd.DataFrame:
    """
    Apply a PRE-COMPUTED AlignParams to a DataFrame (no recomputation from bounds).
    This is the preferred way to ensure consistency between event data and detector geometry.
    """
    x_true, y_true, z_true = transform_arrays(df[x_col].to_numpy(), df[y_col].to_numpy(), df[z_col].to_numpy(), params)
    df_out = df.copy()
    df_out[out_x] = x_true
    df_out[out_y] = y_true
    df_out[out_z] = z_true
    return df_out


# --------------------------- detector geometry helpers ---------------------------

def transform_geom(
    df_geom: pd.DataFrame,
    params: AlignParams,
    x_col: str = "x_offset",
    y_col: str = "y_offset",
    z_col: str = "z_offset",
    out_x: str = "x_transformed",
    out_y: str = "y_transformed",
    out_z: str = "z_transformed",
) -> pd.DataFrame:
    """
    Transform detector geometry positions with the SAME AlignParams used for events.
    Ensures the per-detector means you compute are in the identical frame as the true/pred coords.
    """
    return apply_transform(
        df_geom,
        params=params,
        x_col=x_col, y_col=y_col, z_col=z_col,
        out_x=out_x, out_y=out_y, out_z=out_z,
    )


def detector_means_by_id(
    df_geom_t: pd.DataFrame,
    detector_col: str = "Detector",
    x_col: str = "x_transformed",
    y_col: str = "y_transformed",
    z_col: str = "z_transformed",
    n_detectors: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean transformed positions per detector id (0..n_detectors-1).
    Fills missing detectors with zeros.
    """
    gx = df_geom_t.groupby(detector_col)[x_col].mean()
    gy = df_geom_t.groupby(detector_col)[y_col].mean()
    gz = df_geom_t.groupby(detector_col)[z_col].mean()

    det_pos_x = gx.reindex(range(n_detectors), fill_value=0.0).to_numpy(dtype=float)
    det_pos_y = gy.reindex(range(n_detectors), fill_value=0.0).to_numpy(dtype=float)
    det_pos_z = gz.reindex(range(n_detectors), fill_value=0.0).to_numpy(dtype=float)
    return det_pos_x, det_pos_y, det_pos_z


# --------------------------- convenience: true radius ---------------------------

def add_true_radius(
    df: pd.DataFrame,
    x_col: str = "truex",
    y_col: str = "truey",
    z_col: str = "truez",
    out_col: str = "r_true",
) -> pd.DataFrame:
    """
    Add ||(x,y,z)|| as a convenience column in the transformed frame.
    """
    df_out = df.copy()
    df_out[out_col] = np.sqrt(df_out[x_col] ** 2 + df_out[y_col] ** 2 + df_out[z_col] ** 2)
    return df_out


# --------------------------- HDF5 & geometry loading ---------------------------

def load_hdf_geometry(hdf5_path: Path) -> Tuple[np.ndarray, float]:
    """
    Extract module bounds and max drift distance from HDF5 geometry_info.

    Returns:
        mod_bounds_mm: array of shape (n_modules, 2, 3) with min/max corners
        max_drift_distance: float in mm
    """
    with h5py.File(hdf5_path, "r") as f:
        try:
            mod_bounds_mm = np.array(f["geometry_info"].attrs["module_RO_bounds"])
        except Exception as exc:
            raise RuntimeError(f"Unable to read module_RO_bounds from {hdf5_path}: {exc}")
        max_drift_distance = float(f["geometry_info"].attrs["max_drift_distance"])
    return mod_bounds_mm, max_drift_distance


def load_geom_csv(path: Path) -> pd.DataFrame:
    """Load detector geometry CSV."""
    return pd.read_csv(path)


def load_geom_yaml(path: Path) -> dict:
    """Load detector geometry YAML with tpc_center_offset and det_center."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_module_centres(mod_bounds_mm: np.ndarray) -> List[List[float]]:
    """
    Compute geometric center of each module from bounds array.

    Args:
        mod_bounds_mm: shape (n_modules, 2, 3) - min/max corners per module

    Returns:
        List of [x_center, y_center, z_center] for each module
    """
    centres = []
    for mod in mod_bounds_mm:
        centres.append([
            float((mod[0][0] + mod[1][0]) / 2.0),
            float((mod[0][1] + mod[1][1]) / 2.0),
            float((mod[0][2] + mod[1][2]) / 2.0),
        ])
    return centres


def compute_detector_offsets(
    df_geom: pd.DataFrame,
    module_centres: List[List[float]],
    geom_data: dict,
) -> pd.DataFrame:
    """
    Compute absolute x/y/z_offset for each detector in UNTRANSFORMED coordinates.

    Combines:
      - det_center (from YAML, detector-local coords)
      - tpc_center_offset (from YAML, TPC-relative offset)
      - module_centres (computed from HDF5 bounds)

    Adds columns: x_offset, y_offset, z_offset
    Requires columns: Detector, TPC, mod (where mod = TPC // 2)
    """
    df_out = df_geom.copy()
    for idx, row in df_out.iterrows():
        det_num = int(row["Detector"])
        tpc_num = int(row["TPC"])
        mod_num = int(row["mod"])

        tco = geom_data["tpc_center_offset"][tpc_num]
        dc = geom_data["det_center"][det_num]

        df_out.at[idx, "x_offset"] = dc[0] + tco[0] + module_centres[mod_num][0]
        df_out.at[idx, "y_offset"] = dc[1] + tco[1] + module_centres[mod_num][1]
        df_out.at[idx, "z_offset"] = dc[2] + tco[2] + module_centres[mod_num][2]

    return df_out


def extract_pde_per_detector(
    df_geom: pd.DataFrame,
    n_detectors: int = 16,
    default_pde: float = 0.006,
    traptype_col: str = "TrapType",
) -> np.ndarray:
    """
    Extract photon detection efficiency (PDE) for each detector based on TrapType.

    Convention (hardcoded from DUNE 2x2):
      - TrapType != 0 → PDE = 0.006
      - TrapType == 0 → PDE = 0.002

    Returns:
        Array of shape (n_detectors,) with PDE values
    """
    eff_list = []
    for d in range(n_detectors):
        subset = df_geom.loc[df_geom["Detector"] == d]
        if not subset.empty and traptype_col in subset.columns:
            trap_type = int(subset[traptype_col].values[0])
            eff = 0.006 if trap_type != 0 else 0.002
        else:
            eff = default_pde
        eff_list.append(eff)
    return np.array(eff_list, dtype=float)