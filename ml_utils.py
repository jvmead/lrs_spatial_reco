#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning utilities for GNN-based spatial reconstruction.

This module contains:
- Graph adjacency builders (grid, line, fully-connected)
- Spektral Dataset classes for detector data
- GNN model architectures
- Custom loss functions
- Training utilities and diagnostics
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

import tensorflow as tf
from tensorflow import keras
from keras import layers

import spektral
from spektral.layers import GCNConv, GlobalAttentionPool
from spektral.data import Dataset, Graph


# ==================== GRAPH ADJACENCY BUILDERS ====================

def build_complete_adj(n: int, include_self_loops: bool = False) -> sp.csr_matrix:
    """
    Build fully connected adjacency matrix.

    Args:
        n: Number of nodes
        include_self_loops: Whether to include self-connections

    Returns:
        Sparse adjacency matrix
    """
    A = np.ones((n, n), dtype=np.float32)
    if not include_self_loops:
        np.fill_diagonal(A, 0.0)
    return sp.csr_matrix(A)


def build_line_adj(n: int = 16, connect_k: int = 1) -> sp.csr_matrix:
    """
    Build line graph adjacency: each node connected to k nearest neighbors.

    Args:
        n: Number of nodes
        connect_k: Number of neighbors to connect (1 = nearest only)

    Returns:
        Sparse adjacency matrix
    """
    rows, cols = [], []
    for i in range(n):
        for d in range(1, connect_k + 1):
            j = i - d
            k = i + d
            if j >= 0:
                rows += [i, j]
                cols += [j, i]
            if k < n:
                rows += [i, k]
                cols += [k, i]
    data = np.ones(len(rows), dtype=np.float32)
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def build_grid_adj(nx: int = 2, ny: int = 8, connect_diag: bool = False) -> sp.csr_matrix:
    """
    Build 2D grid adjacency with optional diagonal connections.

    Args:
        nx: Grid width (number of columns)
        ny: Grid height (number of rows)
        connect_diag: If True, include diagonal (8-connectivity), else 4-connectivity

    Returns:
        Sparse adjacency matrix
    """
    n = nx * ny
    rows, cols = [], []

    def idx(x, y):
        return x * ny + y  # column-major ordering

    for y in range(ny):
        for x in range(nx):
            u = idx(x, y)
            # 4-connectivity (up, down, left, right)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                xx, yy = x + dx, y + dy
                if 0 <= xx < nx and 0 <= yy < ny:
                    v = idx(xx, yy)
                    rows += [u, v]
                    cols += [v, u]

            # 8-connectivity (diagonals)
            if connect_diag:
                for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < nx and 0 <= yy < ny:
                        v = idx(xx, yy)
                        rows += [u, v]
                        cols += [v, u]

    data = np.ones(len(rows), dtype=np.float32)
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def gcn_normalize(a: sp.spmatrix, add_self_loops: bool = True) -> sp.csr_matrix:
    """
    Apply GCN normalization: D^(-1/2) A D^(-1/2).

    Args:
        a: Adjacency matrix (scipy sparse)
        add_self_loops: Whether to add self-loops before normalization

    Returns:
        Normalized adjacency matrix
    """
    if not sp.isspmatrix(a):
        raise TypeError("Adjacency must be scipy sparse")

    a = a.tocsr().astype(np.float32)
    n = a.shape[0]

    if add_self_loops:
        a = a + sp.eye(n, dtype=np.float32, format="csr")

    deg = np.asarray(a.sum(axis=1)).ravel()
    deg[deg == 0.0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    D = sp.diags(d_inv_sqrt, dtype=np.float32, format="csr")

    return (D @ a @ D).tocsr()


def build_adj_by_tag(
    n: int,
    tag: str = "grid4",
    *,
    nx: int = 2,
    ny: int = 8,
    connect_k: int = 1
) -> sp.csr_matrix:
    """
    Build adjacency matrix based on string tag.

    Args:
        n: Number of nodes
        tag: Adjacency type ('grid4', 'grid8', 'adjacent', 'fully', 'complete', 'line', 'line_k')
        nx: Grid width (for grid types)
        ny: Grid height (for grid types)
        connect_k: Number of hops (for line_k)

    Returns:
        Sparse adjacency matrix
    """
    tag = tag.lower()

    if tag in {"fully", "complete"}:
        return build_complete_adj(n, False)
    elif tag in {"adjacent", "grid", "grid4"}:
        return build_grid_adj(nx, ny, False)
    elif tag == "grid8":
        return build_grid_adj(nx, ny, True)
    elif tag == "line":
        return build_line_adj(n, 1)
    elif tag == "line_k":
        return build_line_adj(n, connect_k)
    else:
        raise ValueError(f"Unknown adjacency tag: {tag}")


# ==================== SPEKTRAL DATASET ====================

class Det16Dataset(Dataset):
    """
    Spektral Dataset for 16-detector events.

    Converts tabular detector data into graph format suitable for GNN training.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        use_log: bool = False,
        use_log10: bool = False,
        add_eff: bool = True,
        add_xyz: bool = True,
        eff_vec: Optional[np.ndarray] = None,
        xyz: Optional[np.ndarray] = None,
        x_channel_norm: Optional[Dict[str, np.ndarray]] = None,
        y_norm: Optional[Dict[str, np.ndarray]] = None,
        **kwargs
    ):
        """
        Args:
            df: DataFrame with det_#_max columns and truex/truey/truez
            use_log: Apply log1p transform to detector signals (natural log)
            use_log10: Apply log10(x + epsilon) transform to detector signals
            add_eff: Include detector efficiency as node feature
            add_xyz: Include detector positions as node features
            eff_vec: Efficiency values per detector (shape: 16 or 16x1)
            xyz: Detector positions (shape: 16x3)
            x_channel_norm: Dict with 'mean' and 'std' for feature normalization
            y_norm: Dict with 'mean' and 'std' for target normalization
        """
        self.df = df.reset_index(drop=True)
        self.use_log = use_log
        self.use_log10 = use_log10
        self.add_eff = add_eff
        self.add_xyz = add_xyz
        self.eff_vec = eff_vec
        self.xyz = xyz
        self.x_channel_norm = x_channel_norm
        self.y_norm = y_norm
        super().__init__(**kwargs)

    def _coerce_eff(self, n_det: int) -> Optional[np.ndarray]:
        """Validate and reshape efficiency vector."""
        if not self.add_eff:
            return None
        if self.eff_vec is None:
            raise ValueError("add_eff=True but eff_vec is None")
        eff = np.asarray(self.eff_vec, dtype=np.float32)
        if eff.ndim == 1:
            eff = eff.reshape(-1, 1)
        if eff.shape != (n_det, 1):
            raise ValueError(f"eff_vec must have shape ({n_det},1); got {eff.shape}")
        return eff

    def _coerce_xyz(self, n_det: int) -> Optional[np.ndarray]:
        """Validate and reshape position array."""
        if not self.add_xyz:
            return None
        if self.xyz is None:
            raise ValueError("add_xyz=True but xyz is None")
        xyz = np.asarray(self.xyz, dtype=np.float32)
        if xyz.ndim == 1:
            if xyz.size != n_det * 3:
                raise ValueError(f"xyz flat size must be {n_det*3}; got {xyz.size}")
            xyz = xyz.reshape(n_det, 3)
        if xyz.shape != (n_det, 3):
            raise ValueError(f"xyz must have shape ({n_det},3); got {xyz.shape}")
        return xyz

    def read(self):
        """Build list of Graph objects from dataframe."""
        graphs = []
        n_det = 16
        sig_cols = [f"det_{i}_max" for i in range(n_det)]
        y_cols = ["truex", "truey", "truez"]

        eff_fixed = self._coerce_eff(n_det)
        xyz_fixed = self._coerce_xyz(n_det)

        for _, row in self.df.iterrows():
            # Node features: detector signals
            x_sig = row[sig_cols].to_numpy(np.float32).reshape(n_det, 1)

            # Apply log transform if requested
            if self.use_log:
                x_sig = np.log1p(np.clip(x_sig, a_min=0.0, a_max=None))
            elif self.use_log10:
                # log10(x + epsilon) to handle zeros and NaNs
                epsilon = 1e-10
                x_sig = np.log10(np.clip(x_sig, a_min=0.0, a_max=None) + epsilon)

            x_sig = np.nan_to_num(x_sig, 0.0, 0.0, 0.0)

            feats = [x_sig]
            if eff_fixed is not None:
                feats.append(eff_fixed)
            if xyz_fixed is not None:
                feats.append(xyz_fixed)

            X = np.concatenate(feats, axis=1).astype(np.float32)

            # Normalization
            if self.x_channel_norm is not None:
                mu = np.asarray(self.x_channel_norm["mean"], dtype=np.float32)
                sd = np.asarray(self.x_channel_norm["std"], dtype=np.float32)
                sd = np.where(sd == 0.0, 1.0, sd)
                if mu.shape[0] != X.shape[1] or sd.shape[0] != X.shape[1]:
                    raise ValueError(
                        f"x_channel_norm must match feature channels ({X.shape[1]})"
                    )
                X = (X - mu[None, :]) / sd[None, :]

            # Targets
            y = row[y_cols].to_numpy(np.float32)
            y = np.nan_to_num(y, 0.0, 0.0, 0.0)
            if self.y_norm is not None:
                y = (y - self.y_norm["mean"]) / self.y_norm["std"]

            # Placeholder adjacency (will override later)
            A = sp.eye(n_det, dtype=np.float32, format="csr")
            graphs.append(Graph(x=X, a=A, y=y))

        return graphs


def override_adj(
    ds: Det16Dataset,
    tag: str,
    *,
    nx: int = 2,
    ny: int = 8,
    connect_k: int = 1
):
    """
    Override adjacency matrices in dataset after construction.

    Args:
        ds: Det16Dataset instance
        tag: Adjacency type
        nx: Grid width
        ny: Grid height
        connect_k: Number of hops for line_k
    """
    for i in range(len(ds)):
        g = ds[i]
        n = g.x.shape[0]
        A = build_adj_by_tag(n, tag, nx=nx, ny=ny, connect_k=connect_k)
        g.a = gcn_normalize(A, add_self_loops=True)


# ==================== GNN MODELS ====================

class StripMask(layers.Layer):
    """Remove mask from layer inputs to prevent propagation."""
    def call(self, inputs, *args, **kwargs):
        return inputs

    def compute_mask(self, inputs, mask=None):
        return None


class NoMaskGCNConv(layers.Layer):
    """GCN convolution that doesn't propagate Keras masks."""
    def __init__(self, channels, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.activation = activation
        self.strip = StripMask()
        self.conv = GCNConv(channels, activation=activation)
        self.supports_masking = False

    def build(self, input_shape):
        self.conv.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        inputs = self.strip(inputs)
        return self.conv.call(inputs, mask=None)

    def compute_mask(self, inputs, mask=None):
        return None


class GNNModel(keras.Model):
    """
    Graph Neural Network for spatial reconstruction.

    Architecture:
    - Configurable number of GCN layers with residual connections
    - Optional layer/batch normalization after each GCN
    - Dropout for regularization
    - Global attention pooling
    - Signal-weighted position barycentre
    - Dense output layer
    """

    def __init__(
        self,
        n_node_features: int,
        hidden: int = 64,
        out_dim: int = 3,
        dropout: float = 0.05,
        n_layers: int = 2,
        use_norm: bool = True,
        norm_type: str = "layer"
    ):
        """
        Args:
            n_node_features: Number of input features per node
            hidden: Hidden layer size
            out_dim: Output dimension (3 for x,y,z)
            dropout: Dropout rate
            n_layers: Number of GCN layers
            use_norm: If True, add normalization layers after GCN layers
            norm_type: Type of normalization - "layer" or "batch"
        """
        super().__init__(name="gnn_model")
        self.n_node_features = n_node_features
        self.hidden = hidden
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.use_norm = use_norm
        self.norm_type = norm_type

        # GNN layers
        self.gcn_layers = []
        self.norm_layers = []
        self.dropout_layers = []

        for i in range(n_layers):
            self.gcn_layers.append(NoMaskGCNConv(hidden, activation="relu"))

            # Add normalization layer if requested
            if use_norm:
                if norm_type == "batch":
                    self.norm_layers.append(layers.BatchNormalization())
                else:  # default to layer norm
                    self.norm_layers.append(layers.LayerNormalization())

            if i < n_layers - 1:  # No dropout after last GCN
                self.dropout_layers.append(layers.Dropout(dropout))

        # Global pooling
        self.pool = GlobalAttentionPool(self.hidden)

        # Output regressor
        self.out = layers.Dense(out_dim, name="regressor")

    def build(self, input_shape):
        x_shape, a_shape, i_shape = input_shape

        # Build GCN layers
        current_shape = x_shape
        for i in range(self.n_layers):
            self.gcn_layers[i].build([current_shape, a_shape])
            current_shape = tf.TensorShape([None, self.hidden])

            # Build normalization layer if it exists
            if self.use_norm:
                self.norm_layers[i].build(current_shape)

            if i < self.n_layers - 1:
                self.dropout_layers[i].build(current_shape)

        # Build pooling and output
        self.pool.build([current_shape, i_shape])
        self.out.build(tf.TensorShape([None, self.hidden + 3]))

        super().build(input_shape)

    def call(self, inputs, training=False):
        X, A, I = inputs

        # Assertions for debugging
        tf.debugging.assert_equal(
            tf.shape(A)[0], tf.shape(A)[1], message="A must be square"
        )
        tf.debugging.assert_equal(
            tf.shape(X)[0], tf.shape(A)[0], message="X and A must have same N"
        )
        tf.debugging.assert_equal(
            tf.shape(X)[0], tf.shape(I)[0], message="X and I must have same N"
        )

        I = tf.cast(I, tf.int32)

        # GNN backbone with residual connections
        h = X
        for i in range(self.n_layers):
            h_prev = h
            h = self.gcn_layers[i]([h, A], training=training)

            # Apply normalization if enabled
            if self.use_norm:
                h = self.norm_layers[i](h, training=training)

            # Residual connection (if not first layer)
            if i > 0:
                h = h + h_prev

            # Dropout (except after last layer)
            if i < self.n_layers - 1:
                h = self.dropout_layers[i](h, training=training)

        # Global attention pooling
        g = self.pool([h, I])

        # Barycentre feature from signal-weighted position
        sig = tf.nn.relu(X[:, 0])
        z = tf.math.log1p(sig)

        # Robust per-graph softmax
        zmax = tf.math.unsorted_segment_max(
            z, I, num_segments=tf.reduce_max(I) + 1
        )
        zexp = tf.exp(z - tf.gather(zmax, I))
        zsum = tf.math.unsorted_segment_sum(
            zexp, I, num_segments=tf.shape(zmax)[0]
        )
        w = zexp / tf.gather(zsum, I)

        # Weighted position barycentre
        xyz = X[:, -3:]
        bary = tf.math.unsorted_segment_sum(
            w[:, None] * xyz, I, num_segments=tf.shape(zmax)[0]
        )

        # Concatenate pooled embedding with barycentre
        g = tf.concat([g, bary], axis=-1)

        return self.out(g)


# ==================== LOSS FUNCTIONS ====================

def huber_xyz_loss(delta: float = 1.0):
    """Huber loss on xyz coordinates."""
    return keras.losses.Huber(delta=delta)


def huber_xyz_plus_radius(delta: float = 1.0, lam: float = 0.1):
    """
    Huber loss on xyz + regularization on radius.

    Encourages the model to learn both individual coordinates and overall distance.

    Args:
        delta: Huber delta parameter
        lam: Weight for radius regularization term

    Returns:
        Loss function
    """
    base = keras.losses.Huber(delta=delta)

    def loss(y_true, y_pred):
        l_xyz = base(y_true, y_pred)
        r_true = tf.norm(y_true, axis=-1)
        r_pred = tf.norm(y_pred, axis=-1)
        l_r = base(r_true, r_pred)
        return l_xyz + lam * l_r

    return loss


def mse_loss():
    """Mean squared error loss."""
    return keras.losses.MeanSquaredError()


def mae_loss():
    """Mean absolute error loss."""
    return keras.losses.MeanAbsoluteError()


def get_loss_by_name(name: str, **kwargs):
    """
    Get loss function by name.

    Args:
        name: Loss function name ('huber', 'huber_radius', 'mse', 'mae')
        **kwargs: Additional arguments for loss function

    Returns:
        Loss function
    """
    name = name.lower()

    if name == "huber":
        return huber_xyz_loss(delta=kwargs.get("delta", 1.0))
    elif name in {"huber_radius", "huber_r"}:
        return huber_xyz_plus_radius(
            delta=kwargs.get("delta", 1.0),
            lam=kwargs.get("lam", 0.1)
        )
    elif name == "mse":
        return mse_loss()
    elif name == "mae":
        return mae_loss()
    else:
        raise ValueError(f"Unknown loss function: {name}")


# ==================== TRAINING UTILITIES ====================

def infer_steps(loader, dataset=None):
    """
    Infer number of steps per epoch from loader.

    Args:
        loader: Spektral data loader
        dataset: Optional dataset for fallback calculation

    Returns:
        Number of steps per epoch
    """
    for attr in ("steps_per_epoch", "steps"):
        if hasattr(loader, attr):
            v = getattr(loader, attr)
            try:
                if v is not None and float(v) > 0:
                    return int(math.ceil(float(v)))
            except:
                pass

    n = getattr(loader, "n_samples", None)
    b = getattr(loader, "batch_size", None)
    if n is not None and b:
        return int(math.ceil(n / b))
    if dataset is not None and b:
        return int(math.ceil(len(dataset) / b))

    return 100


def get_optimizer(name: str, lr: float, **kwargs):
    """
    Get optimizer by name.

    Args:
        name: Optimizer name ('adam', 'adamw', 'sgd')
        lr: Learning rate
        **kwargs: Additional optimizer arguments

    Returns:
        Keras optimizer
    """
    name = name.lower()

    if name == "adam":
        return keras.optimizers.Adam(
            learning_rate=lr,
            clipnorm=kwargs.get("clipnorm", 1.0)
        )
    elif name == "adamw":
        try:
            return keras.optimizers.AdamW(
                learning_rate=lr,
                weight_decay=kwargs.get("weight_decay", 1e-4),
                clipnorm=kwargs.get("clipnorm", 1.0)
            )
        except AttributeError:
            # Fallback if AdamW not available
            return keras.optimizers.Adam(
                learning_rate=lr,
                clipnorm=kwargs.get("clipnorm", 1.0)
            )
    elif name == "sgd":
        return keras.optimizers.SGD(
            learning_rate=lr,
            momentum=kwargs.get("momentum", 0.9),
            clipnorm=kwargs.get("clipnorm", 1.0)
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# ==================== ML-SPECIFIC PLOTTING ====================

def plot_training_curves(
    history: keras.callbacks.History,
    lr: float,
    patience: int,
    outdir: Path
):
    """
    Plot training and validation loss curves.

    Shows:
    - Training loss over epochs
    - Validation loss over epochs
    - Best epoch marker
    - Early stopping window shading

    Args:
        history: Keras History object from model.fit()
        lr: Learning rate used
        patience: Early stopping patience
        outdir: Output directory for plot
    """
    import matplotlib.pyplot as plt

    hist = history.history
    epochs_ran = np.arange(1, len(hist["loss"]) + 1)
    val_series = np.array(hist.get("val_loss", hist["loss"]), dtype=float)
    best_idx = int(np.nanargmin(val_series))
    best_epoch = best_idx + 1

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_ran, hist["loss"], label="train loss", linewidth=2)
    if "val_loss" in hist:
        plt.plot(epochs_ran, hist["val_loss"], label="val loss", linewidth=2)
    plt.scatter(
        [best_epoch], [val_series[best_idx]], s=80, marker="*",
        color="red", label="best epoch", zorder=10
    )

    # Shade early stopping window
    es_start, es_end = best_epoch + 1, min(best_epoch + patience, epochs_ran[-1])
    if es_start <= es_end:
        plt.axvspan(es_start, es_end, alpha=0.15, color="gray",
                    label=f"ES window (+{patience})")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Curves (LR={lr})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(outdir / "training_curves.png", dpi=150)
    plt.close()
    print(f"Saved: {outdir / 'training_curves.png'}")


def plot_learning_rate(
    history: keras.callbacks.History,
    outdir: Path
):
    """
    Plot learning rate schedule over training.

    Args:
        history: Keras History object from model.fit()
        outdir: Output directory for plot
    """
    import matplotlib.pyplot as plt

    hist = history.history
    if "lr" not in hist:
        print("No learning rate history available")
        return

    epochs_ran = np.arange(1, len(hist["lr"]) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs_ran, hist["lr"], linewidth=2, color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(outdir / "learning_rate.png", dpi=150)
    plt.close()
    print(f"Saved: {outdir / 'learning_rate.png'}")


def plot_metric_curves(
    history: keras.callbacks.History,
    metric_name: str,
    outdir: Path
):
    """
    Plot training metric curves (e.g., MAE, MSE).

    Args:
        history: Keras History object from model.fit()
        metric_name: Name of metric to plot
        outdir: Output directory for plot
    """
    import matplotlib.pyplot as plt

    hist = history.history
    if metric_name not in hist:
        print(f"Metric '{metric_name}' not found in history")
        return

    epochs_ran = np.arange(1, len(hist[metric_name]) + 1)
    val_metric = f"val_{metric_name}"

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_ran, hist[metric_name], label=f"train {metric_name}", linewidth=2)
    if val_metric in hist:
        plt.plot(epochs_ran, hist[val_metric], label=f"val {metric_name}", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel(metric_name.upper())
    plt.title(f"{metric_name.upper()} over Training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(outdir / f"{metric_name}_curves.png", dpi=150)
    plt.close()
    print(f"Saved: {outdir / f'{metric_name}_curves.png'}")
