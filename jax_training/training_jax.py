"""
Unified training functions for JAX models using Flax.
Supports classification and regression, *and now keeps track of the best‑performing
parameters even across resumed runs*.

Key features
------------
• Metrics + TrainState definitions for both tasks.  
• Simple MLP model.  
• Data utilities (`create_data`, `create_regression_data`, iterator).  
• **Best‑checkpoint logic**: `train_cls` and `train_reg` accept
  `initial_best_*` kwargs and return `(state, metrics, best_blob, best_loss, best_epoch)`.
• Optional on‑the‑fly checkpoint saving + MLflow artifact logging.
"""

# from __future__ import annotations

import pickle
from functools import partial
import os
import tempfile
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from clu import metrics
from flax import linen as nn, struct
from flax.training import train_state
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optax

# ================================================================
# Metrics and TrainState definitions
# ================================================================

@struct.dataclass
class ClassificationMetrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")

class ClassificationTrainState(train_state.TrainState):
    metrics: ClassificationMetrics

@struct.dataclass
class RegressionMetrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    mse: metrics.Average.from_output("mse")

class RegressionTrainState(train_state.TrainState):
    metrics: RegressionMetrics

# ================================================================
# Model definition (simple MLP as a placeholder)
# ================================================================

class SimpleModel(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x

# ================================================================
# Data utilities
# ================================================================

def create_data(n_samples: int = 1_000, noise: float = 0.2, random_state: int = 42):
    """Binary‑classification moons dataset."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler


def create_regression_data(n_samples: int = 1_000, noise: float = 0.1, random_state: int = 42):
    """1‑D synthetic regression y = 2.5x + 3 + ϵ."""
    rng = np.random.RandomState(random_state)
    X = rng.rand(n_samples, 1) * 10.0
    y = X.dot(np.array([2.5])) + 3.0 + rng.randn(n_samples, 1) * noise
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def data_iterator(features: np.ndarray, labels: np.ndarray, batch_size: int,
                  shuffle: bool = True, seed: int = 0):
    """Yield dict batches converted to `jnp` arrays."""
    size = len(features)
    indices = np.arange(size)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
    for start in range(0, size, batch_size):
        idx = indices[start : start + batch_size]
        yield {"input": jnp.array(features[idx]), "label": jnp.array(labels[idx])}

# ================================================================
# (De)serialisation utilities
# ================================================================

def save_state_to_bytes(params: Any, opt_state: Any) -> bytes:
    return pickle.dumps({"params": params, "opt_state": opt_state})


def load_state_from_bytes(blob: bytes):
    data = pickle.loads(blob)
    return data["params"], data["opt_state"]

# ================================================================
# Train‑state factories
# ================================================================

def _make_tx(learning_rate: float, weight_decay: float, clip: float | None):
    tx = optax.lion(learning_rate, weight_decay)
    if clip is not None:
        tx = optax.chain(optax.clip_by_global_norm(clip), tx)
    return tx


def create_train_state_cls(module: nn.Module, rng, learning_rate: float, weight_decay: float,
                           x_item, clip_by_global_norm: float | None = 0.01):
    params = module.init(rng, x_item)["params"]
    return ClassificationTrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=_make_tx(learning_rate, weight_decay, clip_by_global_norm),
        metrics=ClassificationMetrics.empty(),
    )


def create_train_state_reg(module: nn.Module, rng, learning_rate: float, weight_decay: float,
                           x_item, clip_by_global_norm: float | None = 0.01):
    params = module.init(rng, x_item)["params"]
    return RegressionTrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=_make_tx(learning_rate, weight_decay, clip_by_global_norm),
        metrics=RegressionMetrics.empty(),
    )

# ================================================================
# ── Classification helpers ───────────────────────────────────────
# ================================================================

def _cross_entropy(logits, labels, smoothing: float = 0.0):
    num_classes = logits.shape[-1]
    if smoothing:
        one_hot = jax.nn.one_hot(labels, num_classes)
        labels = (1 - smoothing) * one_hot + smoothing / num_classes
        loss = optax.softmax_cross_entropy(logits, labels)
    else:
        loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, num_classes))
    return loss.mean()


@partial(jax.jit, static_argnums=(2,))
def _train_step_cls(state: ClassificationTrainState, batch: Dict[str, jnp.ndarray],
                    smoothing: float = 0.0):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["input"])
        loss = _cross_entropy(logits, batch["label"], smoothing)
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    updates = state.metrics.single_from_model_output(
        loss=loss, logits=logits, labels=batch["label"]
    )
    return state.replace(metrics=state.metrics.merge(updates)), loss


@partial(jax.jit, static_argnums=(2,))
def _eval_step_cls(state: ClassificationTrainState, batch, smoothing: float = 0.0):
    logits = state.apply_fn({"params": state.params}, batch["input"])
    loss = _cross_entropy(logits, batch["label"], smoothing)
    updates = state.metrics.single_from_model_output(
        loss=loss, logits=logits, labels=batch["label"]
    )
    return state.replace(metrics=state.metrics.merge(updates)), loss


def train_cls(
    state: ClassificationTrainState,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    target_epochs: int,
    *,
    start_epoch: int = 1,
    smoothing: float = 0.0,
    print_output: bool = False,
    prune_callback=None,
    save_best_to: str | None = None,
    log_best_with_mlflow: bool = True,
    initial_best_state_blob: bytes | None = None,
    initial_best_val_loss: float | None = None,
    initial_best_epoch: int | None = None,
):
    """Train classifier, keeping the parameters with lowest val‑loss."""
    import mlflow  # local import keeps top‑of‑file light

    best_val_loss = float("inf") if initial_best_val_loss is None else initial_best_val_loss
    best_state_blob = initial_best_state_blob
    best_epoch = initial_best_epoch

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # ── main epoch loop ────────────────────────────────────────────
    for epoch in range(start_epoch, target_epochs):
        # training
        for batch in data_iterator(X_train, y_train, batch_size, shuffle=True, seed=epoch):
            state, _ = _train_step_cls(state, batch, smoothing)
            state, _ = _eval_step_cls(state, batch, smoothing)
        train_metrics = state.metrics.compute()
        history["train_loss"].append(float(train_metrics["loss"]))
        history["train_accuracy"].append(float(train_metrics["accuracy"]))
        state = state.replace(metrics=state.metrics.empty())

        # validation
        for batch in data_iterator(X_val, y_val, batch_size, shuffle=False):
            state, _ = _eval_step_cls(state, batch, smoothing)
        val_metrics = state.metrics.compute()
        val_loss = float(val_metrics["loss"])
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        state = state.replace(metrics=state.metrics.empty())

        # best‑checkpoint logic
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state_blob = save_state_to_bytes(state.params, state.opt_state)

            if save_best_to is not None:
                with open(save_best_to, "wb") as fh:
                    fh.write(best_state_blob)
            if log_best_with_mlflow and mlflow.active_run():
                with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".pkl") as tmp:
                    tmp.write(best_state_blob)
                    mlflow.log_artifact(tmp.name, artifact_path="best_state")
                os.remove(tmp.name)

        if print_output:
            print(
                f"Epoch {epoch+1}/{target_epochs}  "
                f"train_loss={train_metrics['loss']:.4f}  val_loss={val_loss:.4f}  "
                f"best_val_loss={best_val_loss:.4f}"
            )
        if prune_callback is not None:
            prune_callback(epoch, val_loss)

    return state, history, best_state_blob, best_val_loss, best_epoch

# ================================================================
# ── Regression helpers ───────────────────────────────────────────
# ================================================================

def _mse(pred, target):
    return jnp.mean((pred - target) ** 2)


@jax.jit
def _train_step_reg(state: RegressionTrainState, batch):
    def loss_fn(params):
        preds = state.apply_fn({"params": params}, batch["input"]).squeeze(-1)
        loss = _mse(preds, batch["label"])
        return loss, preds

    (loss, preds), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    mse = _mse(preds, batch["label"])
    updates = state.metrics.single_from_model_output(loss=loss, mse=mse)
    return state.replace(metrics=state.metrics.merge(updates))


@jax.jit
def _eval_step_reg(state: RegressionTrainState, batch):
    preds = state.apply_fn({"params": state.params}, batch["input"]).squeeze(-1)
    loss = _mse(preds, batch["label"])
    mse = _mse(preds, batch["label"])
    updates = state.metrics.single_from_model_output(loss=loss, mse=mse)
    return state.replace(metrics=state.metrics.merge(updates)), loss, mse


def train_reg(
    state: RegressionTrainState,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    target_epochs: int,
    *,
    start_epoch: int = 1,
    print_output: bool = False,
    save_best_to: str | None = None,
    log_best_with_mlflow: bool = True,
    initial_best_state_blob: bytes | None = None,
    initial_best_val_loss: float | None = None,
    initial_best_epoch: int | None = None,
):
    """Train regression model with best‑checkpoint persistence."""
    import mlflow

    best_val_loss = float("inf") if initial_best_val_loss is None else initial_best_val_loss
    best_state_blob = initial_best_state_blob
    best_epoch = initial_best_epoch

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_mse": [],
        "val_loss": [],
        "val_mse": [],
    }

    for epoch in range(start_epoch, target_epochs):
        # training -----------------------------------------------------
        for batch in data_iterator(X_train, y_train, batch_size, shuffle=True, seed=epoch):
            state = _train_step_reg(state, batch)
            state, _, _ = _eval_step_reg(state, batch)
        tr_metrics = state.metrics.compute()
        history["train_loss"].append(float(tr_metrics["loss"]))
        history["train_mse"].append(float(tr_metrics["mse"]))
        state = state.replace(metrics=state.metrics.empty())

        # validation ---------------------------------------------------
        for batch in data_iterator(X_val, y_val, batch_size, shuffle=False):
            state, _, _ = _eval_step_reg(state, batch)
        val_metrics = state.metrics.compute()
        val_loss = float(val_metrics["loss"])
        history["val_loss"].append(val_loss)
        history["val_mse"].append(float(val_metrics["mse"]))
        state = state.replace(metrics=state.metrics.empty())

        # best‑checkpoint ---------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state_blob = save_state_to_bytes(state.params, state.opt_state)
            if save_best_to is not None:
                with open(save_best_to, "wb") as fh:
                    fh.write(best_state_blob)
            if log_best_with_mlflow and mlflow.active_run():
                with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".pkl") as tmp:
                    tmp.write(best_state_blob)
                    mlflow.log_artifact(tmp.name, artifact_path="best_state")
                os.remove(tmp.name)

        if print_output:
            print(
                f"Epoch {epoch+1}/{target_epochs}  train_loss={tr_metrics['loss']:.5f}  "
                f"val_loss={val_loss:.5f}  best_val_loss={best_val_loss:.5f}"
            )

    return state, history, best_state_blob, best_val_loss, best_epoch

# ================================================================
# Misc.
# ================================================================

def count_parameters(params) -> int:
    """Total parameter count (leaf sizes)."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def plot_metrics(metrics_list: List[Dict[str, List[float]]], labels: List[str], *,
                 task: str = "regression", y_scale: str = "log", x_scale: str = "log"):
    """Quick‑and‑dirty matplotlib plots for multiple runs."""
    import matplotlib.pyplot as plt
    import numpy as np

    if task == "regression":
        fig, (ax_tr, ax_val) = plt.subplots(1, 2, figsize=(12, 5))
        for m, lab in zip(metrics_list, labels):
            epochs = np.arange(1, len(m["train_loss"]) + 1)
            (ax_tr.semilogy if y_scale == "log" else ax_tr.plot)(epochs, m["train_loss"], label=lab)
            (ax_val.semilogy if y_scale == "log" else ax_val.plot)(epochs, m["val_loss"], label=lab)
        for ax, title in zip((ax_tr, ax_val), ("Train loss", "Validation loss")):
            ax.set_title(title)
            if x_scale == "log":
                ax.set_xscale("log")
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()
        return

    # classification
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for m, lab in zip(metrics_list, labels):
        e = np.arange(1, len(m["train_accuracy"]) + 1)
        # error = 1 - acc
        (axs[0, 0].semilogy if y_scale == "log" else axs[0, 0].plot)(e, 1 - np.asarray(m["train_accuracy"]), label=lab)
        (axs[0, 1].semilogy if y_scale == "log" else axs[0, 1].plot)(e, 1 - np.asarray(m["val_accuracy"]), label=lab)
        (axs[1, 0].semilogy if y_scale == "log" else axs[1, 0].plot)(e, m["train_loss"], label=lab)
        (axs[1, 1].semilogy if y_scale == "log" else axs[1, 1].plot)(e, m["val_loss"], label=lab)

    for ax in axs.flat:
        ax.set_xlabel("Epoch")
        ax.set_xscale(x_scale)
        ax.grid(True, which="both", ls="--")
        ax.legend()
    axs[0, 0].set_title("Train error (1-acc)")
    axs[0, 1].set_title("Val error (1-acc)")
    axs[1, 0].set_title("Train loss")
    axs[1, 1].set_title("Val loss")
    plt.tight_layout()
    plt.show()
