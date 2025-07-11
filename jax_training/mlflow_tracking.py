"""
MLflow tracking helpers – unchanged except for a tiny utility import tweak.
No functional changes were needed for best‑checkpoint support; the logic now
lives in `training_jax` + `experiments_run`.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from contextlib import contextmanager

import mlflow
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------------------------------------------------------------
# experiment / run utilities
# -----------------------------------------------------------------------------

def initialize_experiment(experiment_name: str, db_path: str = "sqlite:///mlflow.db"):
    """Create experiment if it doesn't exist and set tracking URI."""
    mlflow.set_tracking_uri(db_path)
    client = MlflowClient()
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(experiment_name)
        logging.info("Created experiment '%s'", experiment_name)


def get_or_create_run(experiment_name: str, run_name: str) -> str:
    """Return run_id for (experiment, name), creating if necessary (headless)."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = client.create_experiment(experiment_name)
    else:
        exp_id = exp.experiment_id

    runs = client.search_runs([exp_id], f"tags.mlflow.runName = '{run_name}'")
    if runs:
        run_id = runs[0].info.run_id
        logging.info("Resuming run '%s' (run_id=%s)", run_name, run_id)
        return run_id

    run_id = client.create_run(exp_id, tags={"mlflow.runName": run_name}).info.run_id
    logging.info("Created new run '%s' (run_id=%s)", run_name, run_id)
    return run_id


# -----------------------------------------------------------------------------
# thin wrappers around mlflow.* calls
# -----------------------------------------------------------------------------

def log_params(params: dict[str, any]):
    for k, v in params.items():
        mlflow.log_param(k, v)


def log_metrics(metrics_dict: dict[str, float], step: int | None = None):
    mlflow.log_metrics(metrics_dict, step=step)


def log_all_metrics_batch(run_id: str, metrics_history: dict[str, list[float]], start_epoch: int):
    """Efficient batch logging given the metric lists collected in training."""
    client = MlflowClient()
    metric_objs: list[Metric] = []
    n_epochs = len(metrics_history["train_loss"])
    now = int(round(time.time() * 1000))
    for i in range(n_epochs):
        epoch = (start_epoch + i) if start_epoch else i
        
        if "train_accuracy" in metrics_history.keys():
            metric_objs.extend([
                Metric("train_loss", metrics_history["train_loss"][i], now, epoch),
                Metric("val_loss",   metrics_history["val_loss"][i],   now, epoch),
                Metric("train_accuracy", metrics_history["train_accuracy"][i], now, epoch),
                Metric("val_accuracy",   metrics_history["val_accuracy"][i],   now, epoch),
            ])
        else:
            metric_objs.extend([
                Metric("train_loss", metrics_history["train_loss"][i], now, epoch),
                Metric("val_loss",   metrics_history["val_loss"][i],   now, epoch),
            ])
    client.log_batch(run_id, metrics=metric_objs)


def log_artifact(local_path: str, artifact_name: str):
    """Upload *local_path* to MLflow under *artifact_name*."""
    if os.path.basename(local_path) != artifact_name:
        # keep MLflow directory clean by renaming inside tmp dir
        with tempfile.TemporaryDirectory() as tmpd:
            dst = os.path.join(tmpd, artifact_name)
            shutil.copy2(local_path, dst)
            mlflow.log_artifact(dst)
    else:
        mlflow.log_artifact(local_path)


def start_run(run_id: str):
    return mlflow.start_run(run_id=run_id)


@contextmanager
def dummy_run():
    yield None
