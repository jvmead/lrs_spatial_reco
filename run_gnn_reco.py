#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN-based spatial reconstruction for DUNE 2x2 detector.

This is a wrapper script for training GNN models with configurable settings:
- Graph adjacency types (grid4, grid8, adjacent, fully connected, line)
- Training hyperparameters (epochs, patience, learning rate, batch size)
- Model architecture (layers, hidden size, dropout)
- Input features (log transform, detector positions, efficiencies)
- Loss functions (Huber, MSE, MAE, with/without radius regularization)
- Normalization options for inputs and targets

All ML-specific functionality is in ml_utils.py.

Example usage:
    # Basic training with defaults
    python run_gnn_reco.py /path/to/input.csv --plot

    # Custom configuration
    python run_gnn_reco.py input.csv --epochs 500 --hidden 128 --n-layers 3 \
        --adj-type fully --loss huber_radius --no-normalize-y --plot

    # Minimal features (no positions, no efficiency)
    python run_gnn_reco.py input.csv --no-xyz --no-eff --use-log --plot
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from spektral.data.loaders import DisjointLoader
import spektral

# Import utilities
from utils import (
    compute_align_params,
    transform_geom,
    load_hdf_geometry,
    load_geom_csv,
    load_geom_yaml,
    compute_module_centres,
    compute_detector_offsets,
    extract_pde_per_detector,
    timestamped_outdir,
    ensure_outdir,
    sanitize_and_load_csv,
)

from plotting import (
    plot_spatial_distributions,
    plot_residual_distributions,
    plot_residual_corner,
    plot_detector_signals,
    plot_heatmaps_resid_vs_vars,
    plot_pred_vs_true_xyz,
    plot_1d_curves,
    compute_differential_stats_1d_minimal,
    plot_3d_event_displays,
)

from ml_utils import (
    Det16Dataset,
    override_adj,
    GNNModel,
    get_loss_by_name,
    get_optimizer,
    infer_steps,
    plot_training_curves,
    plot_learning_rate,
    plot_metric_curves,
)

# Quieter TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Constants for diagnostic plots
HEATMAP_INDEP_VARS = ["truex", "truey", "truez", "total_signal"]
DEFAULT_1D_VARS = ["truex", "truey", "truez", "total_signal"]


# ==================== UTILITY FUNCTIONS ====================

def save_config(outdir: Path, args: argparse.Namespace, extra_config: Dict[str, Any]) -> None:
    """Save configuration to JSON file."""
    config = {
        "input_csv": str(args.input_csv),
        "geom_csv": str(args.geom_csv),
        "geom_yaml": str(args.geom_yaml),
        "hdf5": str(args.hdf5),
        "seed": SEED,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "val_split": args.val_split,
            "patience": args.patience,
            "reduce_lr_patience": args.reduce_lr_patience,
            "lr": args.lr,
            "optimizer": args.optimizer,
            "loss": args.loss,
            "loss_delta": args.loss_delta,
            "loss_lam": args.loss_lam,
        },
        "model": {
            "hidden": args.hidden,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "adj_type": args.adj_type,
            "k_hops": args.k_hops if args.adj_type == "line_k" else None,
        },
        "features": {
            "use_log": args.use_log,
            "use_log10": args.use_log10,
            "add_eff": args.add_eff,
            "add_xyz": args.add_xyz,
            "normalize_x": args.normalize_x,
            "normalize_y": args.normalize_y,
        },
        **extra_config,
    }

    config_path = outdir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {config_path}")


# ==================== MAIN ====================

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GNN spatial reconstruction for DUNE 2x2 detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # I/O
    p.add_argument("input_csv", type=Path, help="Input CSV with truth and det_*_max columns")
    p.add_argument("--outdir", type=Path, default=None, help="Output base directory")
    p.add_argument("--tag", type=str, default=None, help="Optional tag for output directory")

    # Geometry files
    p.add_argument(
        "--geom-csv",
        type=Path,
        default=Path("../lrs_sanity_check/geom_files/light_module_desc-4.0.0.csv"),
        help="Geometry CSV"
    )
    p.add_argument(
        "--geom-yaml",
        type=Path,
        default=Path("../lrs_sanity_check/geom_files/light_module_desc-4.0.0.yaml"),
        help="Geometry YAML"
    )
    p.add_argument(
        "--hdf5",
        type=Path,
        default=Path("/global/cfs/cdirs/dune/www/data/2x2/simulation/productions/MiniRun6.5_1E19_RHC/MiniRun6.5_1E19_RHC.flow/FLOW/0000000/MiniRun6.5_1E19_RHC.flow.0000000.FLOW.hdf5"),
        help="HDF5 file for geometry bounds"
    )

    # Training hyperparameters
    train_group = p.add_argument_group("Training hyperparameters")
    train_group.add_argument("--epochs", type=int, default=1000, help="Maximum epochs")
    train_group.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    train_group.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction")
    train_group.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    train_group.add_argument("--reduce-lr-patience", type=int, default=5, help="ReduceLROnPlateau patience")
    train_group.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    train_group.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "sgd"],
        help="Optimizer"
    )
    train_group.add_argument(
        "--loss",
        type=str,
        default="huber_radius",
        choices=["huber", "huber_radius", "mse", "mae"],
        help="Loss function"
    )
    train_group.add_argument("--loss-delta", type=float, default=1.0, help="Huber loss delta")
    train_group.add_argument("--loss-lam", type=float, default=0.1, help="Radius loss weight")

    # Model architecture
    model_group = p.add_argument_group("Model architecture")
    model_group.add_argument("--hidden", type=int, default=64, help="Hidden layer size")
    model_group.add_argument("--n-layers", type=int, default=2, help="Number of GCN layers")
    model_group.add_argument("--dropout", type=float, default=0.05, help="Dropout rate")
    model_group.add_argument("--no-norm", dest="use_norm", action="store_false", help="Disable normalization layers")
    model_group.add_argument("--norm-type", type=str, default="layer", choices=["layer", "batch"],
                            help="Normalization type: 'layer' for LayerNorm, 'batch' for BatchNorm")
    model_group.add_argument(
        "--adj-type",
        type=str,
        default="grid8",
        choices=["grid4", "grid8", "adjacent", "fully", "complete", "line", "line_k"],
        help="Graph adjacency type"
    )
    model_group.add_argument("--k-hops", type=int, default=2, help="Number of hops for line_k adjacency")

    # Feature engineering
    feat_group = p.add_argument_group("Feature engineering")
    feat_group.add_argument("--use-log", action="store_true", help="Use log1p(signal) features (natural log)")
    feat_group.add_argument("--use-log10", action="store_true", help="Use log10(signal + epsilon) features (base-10 log)")
    feat_group.add_argument("--no-eff", dest="add_eff", action="store_false", help="Don't add efficiency features")
    feat_group.add_argument("--no-xyz", dest="add_xyz", action="store_false", help="Don't add position features")
    feat_group.add_argument("--no-normalize-x", dest="normalize_x", action="store_false", help="Don't normalize input features")
    feat_group.add_argument("--no-normalize-y", dest="normalize_y", action="store_false", help="Don't normalize targets")

    # Plotting and exports
    plot_group = p.add_argument_group("Plotting and exports")
    plot_group.add_argument("--all", action="store_true", help="Enable all plotting and export options (diagnostics, metrics, distributions, heatmaps, 1D plots/exports, 3D displays)")
    plot_group.add_argument("--plot", action="store_true", help="Generate diagnostic plots")
    plot_group.add_argument("--plot-all-metrics", action="store_true", help="Plot all training metrics (MAE, MSE)")
    plot_group.add_argument("--plot-distributions", action="store_true", help="Plot output distributions (true vs pred for x/y/z) and residual histograms (dx/dy/dz/dr)")
    plot_group.add_argument("--plot-2d-heatmaps", action="store_true", help="Generate 2D heatmaps (residuals vs variables)")
    plot_group.add_argument("--plot-1d-all", action="store_true", help="Generate 1D residual curves for all default variables")
    plot_group.add_argument("--plot-3d-displays", action="store_true", help="Generate 3D event displays for events with min/median/max total_signal and dr")
    plot_group.add_argument("--export-1d-all", action="store_true", help="Export 1D residual statistics (mu, sigma) to CSV for all default variables")

    # Binning options for heatmaps
    bin_group = p.add_argument_group("Binning options")
    bin_group.add_argument("--bins-x", type=int, default=50, help="Number of bins for truex/predx")
    bin_group.add_argument("--bins-y", type=int, default=50, help="Number of bins for truey/predy")
    bin_group.add_argument("--bins-z", type=int, default=50, help="Number of bins for truez/predz")
    bin_group.add_argument("--bins-r", type=int, default=50, help="Number of bins for r/dr")
    bin_group.add_argument("--bins-signal", type=int, default=50, help="Number of bins for total_signal (log-spaced)")

    return p.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    # If --all is specified, enable all plotting and export options
    if args.all:
        args.plot = True
        args.plot_all_metrics = True
        args.plot_distributions = True
        args.plot_2d_heatmaps = True
        args.plot_1d_all = True
        args.plot_3d_displays = True
        args.export_1d_all = True

    print(f"TensorFlow: {tf.__version__} | Keras: {keras.__version__} | Spektral: {spektral.__version__}")
    print(f"\n=== Configuration ===")
    norm_desc = f"{args.norm_type}norm" if args.use_norm else "none"
    print(f"Model: {args.n_layers} GCN layers, hidden={args.hidden}, dropout={args.dropout}, norm={norm_desc}")
    print(f"Adjacency: {args.adj_type}")
    log_type = "log1p" if args.use_log else ("log10" if args.use_log10 else "none")
    print(f"Features: log={log_type}, eff={args.add_eff}, xyz={args.add_xyz}")
    print(f"Normalization: inputs={args.normalize_x}, targets={args.normalize_y}")
    print(f"Training: epochs={args.epochs}, lr={args.lr}, optimizer={args.optimizer}, loss={args.loss}")

    # ========== Load and sanitize data ==========
    print("\n=== Loading data ===")
    df, det_cols = sanitize_and_load_csv(args.input_csv)
    print(f"Final dataset: {len(df)} events")

    # ========== Load geometry ==========
    print("\n=== Loading geometry ===")
    df_geom = load_geom_csv(args.geom_csv)
    df_geom["mod"] = df_geom["TPC"] // 2
    geom_data = load_geom_yaml(args.geom_yaml)

    mod_bounds_mm, max_drift_distance = load_hdf_geometry(args.hdf5)
    module_centres = compute_module_centres(mod_bounds_mm)

    # Build TPC bounds
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

    # Transform geometry
    params = compute_align_params(tpc_bounds_mm)
    df_geom = compute_detector_offsets(df_geom, module_centres, geom_data)
    df_geom_transformed = transform_geom(
        df_geom, params, "x_offset", "y_offset", "z_offset"
    )

    # Extract detector positions (transformed)
    det_pos_x = df_geom_transformed.groupby("Detector")["x_offset"].mean().to_numpy(np.float32)
    det_pos_y = df_geom_transformed.groupby("Detector")["y_offset"].mean().to_numpy(np.float32)
    det_pos_z = df_geom_transformed.groupby("Detector")["z_offset"].mean().to_numpy(np.float32)

    # Stack into NODE_POS_16: shape (16, 3)
    NODE_POS_16 = np.column_stack([det_pos_x, det_pos_y, det_pos_z]).astype(np.float32)

    # Extract efficiency per detector
    EFF_16 = extract_pde_per_detector(df_geom).reshape(-1, 1).astype(np.float32)

    print(f"Detector positions: x=[{det_pos_x.min():.1f}, {det_pos_x.max():.1f}], "
          f"y=[{det_pos_y.min():.1f}, {det_pos_y.max():.1f}], "
          f"z=[{det_pos_z.min():.1f}, {det_pos_z.max():.1f}]")

    # ========== Train/val split ==========
    print("\n=== Splitting data ===")
    train_df, val_df = train_test_split(
        df, test_size=args.val_split, random_state=SEED
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # ========== Compute normalization stats on train only ==========
    if args.normalize_x or args.normalize_y:
        print("\n=== Computing normalization statistics ===")

    # Input normalization (optional)
    X_NORM = None
    if args.normalize_x:
        sig_train = train_df[det_cols].to_numpy(np.float32)
        sig_mu = float(np.mean(sig_train))
        sig_sd = float(np.std(sig_train)) or 1.0

        eff_mu = float(np.mean(EFF_16))
        eff_sd = float(np.std(EFF_16)) or 1.0

        xyz_mu = NODE_POS_16.mean(axis=0)
        xyz_sd = NODE_POS_16.std(axis=0)
        xyz_sd = np.where(xyz_sd == 0.0, 1.0, xyz_sd)

        # Channel order: [signal, eff, x, y, z]
        X_NORM = {
            "mean": np.array([sig_mu, eff_mu, *xyz_mu], dtype=np.float32),
            "std": np.array([sig_sd, eff_sd, *xyz_sd], dtype=np.float32),
        }
        print(f"Input normalization: signal μ={sig_mu:.2f}, σ={sig_sd:.2f}")
    else:
        print("Input normalization: DISABLED")

    # Target normalization (optional)
    Y_NORM = None
    if args.normalize_y:
        y_mean = train_df[["truex", "truey", "truez"]].mean().to_numpy(np.float32)
        y_std = train_df[["truex", "truey", "truez"]].std(ddof=0).to_numpy(np.float32)
        y_std = np.where(y_std == 0.0, 1.0, y_std)
        Y_NORM = {"mean": y_mean, "std": y_std}
        print(f"Target normalization: x μ={y_mean[0]:.2f}, y μ={y_mean[1]:.2f}, z μ={y_mean[2]:.2f}")
    else:
        print("Target normalization: DISABLED")

    # ========== Build datasets ==========
    print("\n=== Building datasets ===")
    train_ds = Det16Dataset(
        train_df,
        use_log=args.use_log,
        use_log10=args.use_log10,
        add_eff=args.add_eff,
        add_xyz=args.add_xyz,
        eff_vec=EFF_16,
        xyz=NODE_POS_16,
        x_channel_norm=X_NORM,
        y_norm=Y_NORM,
    )

    val_ds = Det16Dataset(
        val_df,
        use_log=args.use_log,
        use_log10=args.use_log10,
        add_eff=args.add_eff,
        add_xyz=args.add_xyz,
        eff_vec=EFF_16,
        xyz=NODE_POS_16,
        x_channel_norm=X_NORM,
        y_norm=Y_NORM,
    )

    # Override adjacency
    nx, ny = 2, 8
    override_adj(train_ds, args.adj_type, nx=nx, ny=ny, connect_k=args.k_hops)
    override_adj(val_ds, args.adj_type, nx=nx, ny=ny, connect_k=args.k_hops)

    # Create loaders
    train_loader = DisjointLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DisjointLoader(val_ds, batch_size=128, shuffle=False)

    train_ds_tf = train_loader.load()
    val_ds_tf = val_loader.load()

    train_steps = infer_steps(train_loader, train_ds)
    val_steps = infer_steps(val_loader, val_ds)

    n_node_features = train_ds[0].x.shape[-1]
    print(f"Node features: {n_node_features}, Train steps: {train_steps}, Val steps: {val_steps}")

    # ========== Build model ==========
    print("\n=== Building model ===")
    model = GNNModel(
        n_node_features=n_node_features,
        hidden=args.hidden,
        out_dim=3,
        dropout=args.dropout,
        n_layers=args.n_layers,
        use_norm=args.use_norm,
        norm_type=args.norm_type,
    )

    model.build([
        tf.TensorShape([None, n_node_features]),
        tf.TensorShape([None, None]),
        tf.TensorShape([None]),
    ])

    # Loss function
    loss_fn = get_loss_by_name(
        args.loss,
        delta=args.loss_delta,
        lam=args.loss_lam
    )
    print(f"Loss function: {args.loss}")

    # Optimizer
    opt = get_optimizer(args.optimizer, args.lr)
    print(f"Optimizer: {args.optimizer}, LR: {args.lr}")

    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=["mse", "mae"],
    )

    model.summary()

    # ========== Train model ==========
    print("\n=== Training ===")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=args.reduce_lr_patience,
            min_lr=1e-8,
            verbose=1
        ),
    ]

    history = model.fit(
        train_ds_tf,
        steps_per_epoch=train_steps,
        validation_data=val_ds_tf,
        validation_steps=val_steps,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ========== Evaluate ==========
    print("\n=== Evaluation ===")
    results = model.evaluate(val_ds_tf, steps=val_steps)
    metrics_dict = {n: float(v) for n, v in zip(model.metrics_names, results)}
    print(metrics_dict)

    # ========== Create output directory ==========
    outdir = ensure_outdir(args, "gnn_reco", mode_tag="")

    # ========== Save model ==========
    model_path = outdir / "gnn_model.keras"
    model.save(model_path, include_optimizer=False)
    print(f"Saved model: {model_path}")

    # ========== Save config ==========
    extra_config = {
        "model_architecture": {
            "n_node_features": int(n_node_features),
            "n_layers": args.n_layers,
            "hidden": args.hidden,
            "out_dim": 3,
            "dropout": args.dropout,
            "use_norm": args.use_norm,
            "norm_type": args.norm_type,
        },
        "normalization": {
            "X_mean": X_NORM["mean"].tolist() if X_NORM else None,
            "X_std": X_NORM["std"].tolist() if X_NORM else None,
            "Y_mean": Y_NORM["mean"].tolist() if Y_NORM else None,
            "Y_std": Y_NORM["std"].tolist() if Y_NORM else None,
        },
        "final_metrics": metrics_dict,
        "n_train": len(train_df),
        "n_val": len(val_df),
    }
    save_config(outdir, args, extra_config)

    # ========== Generate predictions ==========
    print("\n=== Generating predictions ===")
    val_predictions = model.predict(val_ds_tf, steps=val_steps)

    # Rescale to original units (if normalized)
    if Y_NORM is not None:
        val_predictions = val_predictions * Y_NORM["std"][None, :] + Y_NORM["mean"][None, :]
        val_targets = np.vstack([g.y for g in val_ds])
        val_targets = val_targets * Y_NORM["std"][None, :] + Y_NORM["mean"][None, :]
    else:
        val_targets = np.vstack([g.y for g in val_ds])

    # Save predictions
    pred_df = pd.DataFrame(
        {
            "truex": val_targets[:, 0],
            "truey": val_targets[:, 1],
            "truez": val_targets[:, 2],
            "predx": val_predictions[:, 0],
            "predy": val_predictions[:, 1],
            "predz": val_predictions[:, 2],
        }
    )

    # Add detector signals and total_signal from val_df for plotting
    det_cols = [f"det_{i}_max" for i in range(16)]
    for col in det_cols:
        if col in val_df.columns:
            pred_df[col] = val_df[col].values

    # Compute total_signal if not already present
    if "total_signal" in val_df.columns:
        pred_df["total_signal"] = val_df["total_signal"].values
    else:
        present_det_cols = [c for c in det_cols if c in pred_df.columns]
        if present_det_cols:
            pred_df["total_signal"] = pred_df[present_det_cols].sum(axis=1)
        else:
            pred_df["total_signal"] = 0.0

    predictions_dir = outdir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    pred_path = predictions_dir / "predictions.csv.gz"
    pred_df.to_csv(pred_path, index=False, compression="gzip")
    print(f"Saved predictions: {pred_path}")

    # ========== Plotting ==========
    if args.plot:
        print("\n=== Generating ML training diagnostics ===")

        # Create training diagnostics directory
        training_dir = outdir / "training"
        training_dir.mkdir(exist_ok=True)

        # ML-specific training diagnostics
        plot_training_curves(history, args.lr, args.patience, training_dir)
        plot_learning_rate(history, training_dir)

        # Plot additional metrics if requested
        if args.plot_all_metrics:
            for metric in ["mae", "mse"]:
                plot_metric_curves(history, metric, training_dir)

    # ========== Distribution Plots ==========
    if args.plot_distributions:
        print("\n=== Generating distribution plots ===")

        # Create diagnostics directory
        diagnostics_dir = outdir / "diagnostics"
        diagnostics_dir.mkdir(exist_ok=True)

        # Plot detector signal distributions (2 columns, colored by PDE)
        plot_detector_signals(
            df=df,
            out_file=diagnostics_dir / "detector_signals.png",
            log_scale=False,
            bins=50,
            log_bins=True,
            log_xscale=True,
            pde_values=EFF_16.flatten(),  # Pass PDE values for color mapping
        )

        # Add residuals to dataframe
        if "dx" not in pred_df.columns:
            pred_df["dx"] = pred_df["predx"] - pred_df["truex"]
            pred_df["dy"] = pred_df["predy"] - pred_df["truey"]
            pred_df["dz"] = pred_df["predz"] - pred_df["truez"]
            pred_df["dr"] = np.sqrt(pred_df["dx"]**2 + pred_df["dy"]**2 + pred_df["dz"]**2)

        # Output distributions (same as analytic approach)
        plot_spatial_distributions(
            df=pred_df,
            x_true_col="truex",
            y_true_col="truey",
            z_true_col="truez",
            x_pred_col="predx",
            y_pred_col="predy",
            z_pred_col="predz",
            tpc_bounds_mm=tpc_bounds_mm,
            out_file=diagnostics_dir / "output_distributions.png",
            label_true="true",
            label_pred="pred",
        )
        print(f"Saved: {diagnostics_dir / 'output_distributions.png'}")

        # Residual distributions (same as analytic approach)
        plot_residual_distributions(
            df=pred_df,
            dx_col="dx",
            dy_col="dy",
            dz_col="dz",
            dr_col="dr",
            out_file=diagnostics_dir / "residual_distributions.png",
            tpc_bounds_mm=tpc_bounds_mm,
            truex_col="truex",
            truey_col="truey",
            truez_col="truez",
        )
        print(f"Saved: {diagnostics_dir / 'residual_distributions.png'}")

        # Corner plot (same as analytic approach)
        plot_residual_corner(
            df=pred_df,
            dx_col="dx",
            dy_col="dy",
            dz_col="dz",
            dr_col="dr",
            out_file=diagnostics_dir / "residual_corner.png",
        )
        print(f"Saved: {diagnostics_dir / 'residual_corner.png'}")

    # ========== 2D Heatmaps ==========
    if args.plot_2d_heatmaps:
        print("\n=== Generating 2D heatmaps ===")

        # Ensure residuals are computed
        if "dx" not in pred_df.columns:
            pred_df["dx"] = pred_df["predx"] - pred_df["truex"]
            pred_df["dy"] = pred_df["predy"] - pred_df["truey"]
            pred_df["dz"] = pred_df["predz"] - pred_df["truez"]
            pred_df["dr"] = np.sqrt(pred_df["dx"]**2 + pred_df["dy"]**2 + pred_df["dz"]**2)

        bins_map = {
            "truex": args.bins_x,
            "truey": args.bins_y,
            "truez": args.bins_z,
            "r": args.bins_r,
            "total_signal": args.bins_signal,  # log10-spaced inside plotting
        }

        # Residual vs variable heatmaps
        plot_heatmaps_resid_vs_vars(pred_df, outdir=outdir, vars_list=HEATMAP_INDEP_VARS, bins_map=bins_map)

        # Additional pred vs true for x,y,z
        bins_map_xyz = {"truex": args.bins_x, "truey": args.bins_y, "truez": args.bins_z}
        plot_pred_vs_true_xyz(pred_df, outdir=outdir, bins_map=bins_map_xyz)
        print(f"Saved 2D heatmaps to {outdir}")

    # ========== 1D Residual Exports ==========
    def choose_bins(var: str) -> int:
        """Select appropriate number of bins for a variable."""
        m = {
            "truex": args.bins_x, "predx": args.bins_x,
            "truey": args.bins_y, "predy": args.bins_y,
            "truez": args.bins_z, "predz": args.bins_z,
            "r": args.bins_r, "dr": args.bins_r,
            "total_signal": args.bins_signal, "log_total_signal": args.bins_signal,
            "tot_signal": args.bins_signal,
        }
        return m.get(var, args.bins_x)

    export_vars = []
    plot1d_vars = []

    if args.export_1d_all:
        export_vars = DEFAULT_1D_VARS.copy()

    if args.plot_1d_all:
        plot1d_vars = DEFAULT_1D_VARS.copy()

    if export_vars:
        print("\n=== Exporting 1D residual statistics ===")

        # Ensure residuals are computed
        if "dx" not in pred_df.columns:
            pred_df["dx"] = pred_df["predx"] - pred_df["truex"]
            pred_df["dy"] = pred_df["predy"] - pred_df["truey"]
            pred_df["dz"] = pred_df["predz"] - pred_df["truez"]
            pred_df["dr"] = np.sqrt(pred_df["dx"]**2 + pred_df["dy"]**2 + pred_df["dz"]**2)

        residuals_dir = outdir / "residuals_1d"
        residuals_dir.mkdir(exist_ok=True)

        for var in export_vars:
            if var == "tot_signal":
                var = "total_signal"
            bins = choose_bins(var)
            df1d = compute_differential_stats_1d_minimal(pred_df, var=var, bins=bins)
            fname = f"resids_f_{var}.csv.gz"
            path = residuals_dir / fname
            df1d.to_csv(path, index=False, compression="gzip")
            print(f"Wrote: {path}")

    if plot1d_vars:
        print("\n=== Generating 1D residual curves ===")

        # Ensure residuals are computed
        if "dx" not in pred_df.columns:
            pred_df["dx"] = pred_df["predx"] - pred_df["truex"]
            pred_df["dy"] = pred_df["predy"] - pred_df["truey"]
            pred_df["dz"] = pred_df["predz"] - pred_df["truez"]
            pred_df["dr"] = np.sqrt(pred_df["dx"]**2 + pred_df["dy"]**2 + pred_df["dz"]**2)

        for var in plot1d_vars:
            if var == "tot_signal":
                var = "total_signal"
            bins = choose_bins(var)
            df1d = compute_differential_stats_1d_minimal(pred_df, var=var, bins=bins)
            plot_1d_curves(df1d, outdir=outdir, var=var)
            print(f"Saved: {outdir / f'resids_f_{var}.png'}")

    # ========== 3D Event Displays ==========
    if args.plot_3d_displays:
        print("\n=== Generating 3D event displays ===")

        # Ensure residuals are computed
        if "dx" not in pred_df.columns:
            pred_df["dx"] = pred_df["predx"] - pred_df["truex"]
            pred_df["dy"] = pred_df["predy"] - pred_df["truey"]
            pred_df["dz"] = pred_df["predz"] - pred_df["truez"]
            pred_df["dr"] = np.sqrt(pred_df["dx"]**2 + pred_df["dy"]**2 + pred_df["dz"]**2)

        # Select events by total_signal (min, median, max)
        sorted_by_signal = pred_df.sort_values('total_signal')
        signal_indices = [
            sorted_by_signal.index[0],
            sorted_by_signal.index[len(sorted_by_signal) // 2],
            sorted_by_signal.index[-1],
        ]

        # Select events by residual dr (min, median, max)
        sorted_by_dr = pred_df.sort_values('dr')
        dr_indices = [
            sorted_by_dr.index[0],
            sorted_by_dr.index[len(sorted_by_dr) // 2],
            sorted_by_dr.index[-1],
        ]

        events_by_metric = {
            'total_signal': signal_indices,
            'dr': dr_indices,
        }

        # Use the SAME TPC bounds and align_params computed earlier (line 360-361)
        # This ensures consistency with the detector geometry transformation
        plot_3d_event_displays(
            df=pred_df,
            df_geom=df_geom,  # Pass untransformed geometry
            tpc_bounds_mm=tpc_bounds_mm,  # Use the same bounds from earlier
            align_params=params,  # Use the same params from earlier
            events_by_metric=events_by_metric,
            outdir=outdir,
            show_pred=True
        )
        print(f"Wrote 3D event displays to: {outdir}/event_displays_3d/")

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
