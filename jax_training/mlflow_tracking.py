# mlflow_tracking.py
"""
MLflow Tracking Utilities

Handles MLflow experiment and run management:
 • Initializing/retrieving experiments.
 • Creating/resuming runs.
 • Logging hyperparameters, metrics, and artifacts.

Also provides a dummy context manager (dummy_run) to disable MLflow for testing.
"""

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric
import logging
from contextlib import contextmanager
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def initialize_experiment(experiment_name, db_path="sqlite:///mlflow.db"):
    """
    Ensure the MLflow experiment exists, creating it if necessary.
    """
    mlflow.set_tracking_uri(db_path)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(experiment_name)
        logging.info(f"Created new experiment '{experiment_name}'.")
    else:
        logging.info(f"Experiment '{experiment_name}' already exists.")

def get_or_create_run(experiment_name: str, run_name: str) -> str:
    """
    Return the run_id for (experiment_name, run_name).
    If the run does not exist it is created *via MlflowClient.create_run*,
    which does not occupy the global `mlflow.active_run()` slot.
    """
    client = MlflowClient()

    # ------------------------------------------------------------------
    # 1) Get (or create) the experiment and remember its ID
    # ------------------------------------------------------------------
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    # ------------------------------------------------------------------
    # 2) Look for an existing run with the requested name
    # ------------------------------------------------------------------
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'"
    )
    if runs:
        run_id = runs[0].info.run_id
        logging.info(f"Resuming run '{run_name}' (run_id={run_id})")
        return run_id

    # ------------------------------------------------------------------
    # 3) Otherwise create the run *without starting it locally*
    # ------------------------------------------------------------------
    run_info = client.create_run(
        experiment_id=experiment_id,
        tags={"mlflow.runName": run_name}
    )
    run_id = run_info.info.run_id
    logging.info(f"Created new run '{run_name}' (run_id={run_id})")
    return run_id


def log_params(params_dict):
    """Log a dictionary of hyperparameters to MLflow."""
    for k, v in params_dict.items():
        mlflow.log_param(k, v)

def log_metrics(metrics_dict, step=None):
    """Log a dictionary of metrics to MLflow."""
    mlflow.log_metrics(metrics_dict, step=step)

    
def log_all_metrics_batch(run_id, metrics_history, start_epoch):
    """
    Log all epoch metrics in one batch using MLflow's REST API via log_batch.
    
    Parameters:
      run_id (str): The MLflow run ID.
      metrics_history (dict): A dictionary containing lists for each metric. It must include:
          "train_loss", "train_accuracy", "val_loss", "val_accuracy".
      start_epoch (int): The starting epoch number (for proper step numbering).
      
    This function builds a list of MLflow Metric objects and logs them in a single call.
    """
    client = MlflowClient()
    metric_list = []
    n_epochs = len(metrics_history["train_loss"])
    
    for i in range(n_epochs):
        # If starting from epoch 0 (fresh run), the first element in metrics_history is the baseline (step 0).
        # Otherwise, if resuming, metrics_history starts at start_epoch.
        if start_epoch == 0:
            epoch = (start_epoch - 1) + i  # baseline becomes step 0, training epoch 1 becomes step 1, etc.
        else:
            epoch = start_epoch + i       # resume from the actual training epoch numbers
        timestamp = int(round(time.time() * 1000))
        metric_list.extend([
            Metric("train_loss", metrics_history["train_loss"][i], timestamp, epoch),
#             Metric("train_accuracy", metrics_history["train_accuracy"][i], timestamp, epoch),
            Metric("val_loss", metrics_history["val_loss"][i], timestamp, epoch),
#             Metric("val_accuracy", metrics_history["val_accuracy"][i], timestamp, epoch)
        ])
    
    client.log_batch(run_id, metrics=metric_list)
    
def log_artifact(artifact_path, artifact_filename):
    """
    Log an artifact (e.g. model state) with MLflow.
    """
    import os
    import tempfile
    import shutil

    if os.path.basename(artifact_path) != artifact_filename:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, artifact_filename)
            shutil.copy2(artifact_path, temp_file)
            mlflow.log_artifact(temp_file)
    else:
        mlflow.log_artifact(artifact_path)

def start_run(run_id):
    """
    Start or resume an MLflow run by run_id.
    """
    return mlflow.start_run(run_id=run_id)

@contextmanager
def dummy_run():
    """
    Dummy context manager for when MLflow is disabled.
    """
    yield None
