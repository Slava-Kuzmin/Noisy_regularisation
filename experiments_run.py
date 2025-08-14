"""
Driver script for running (and resuming) JAX training experiments with
Optuna + Celery orchestration.

**What changed**
----------------
1. When resuming a run we now download *best* checkpoint + metric.  
2. The values are forwarded to `train_reg` / `train_cls`.  
3. After training we log the updated best checkpoint + `best_val_loss` metric.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# ––– system thread limitations (CPU‑only baseline) –––
# -----------------------------------------------------------------------------
import os
os.environ.update({
    "JAX_PLATFORMS": "cpu",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
})

# -----------------------------------------------------------------------------
import logging
import tempfile
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import optuna
from celery import Celery

from jax_training.mlflow_tracking import (
    initialize_experiment,
    get_or_create_run,
    start_run,
    log_params,
    log_all_metrics_batch,
    log_artifact,
    dummy_run,
)
from jax_training.training_jax import (
    create_train_state_reg,
    train_reg,
    save_state_to_bytes,
    load_state_from_bytes,
    count_parameters,
)
from jax_training.pca_datasets import generate_dataset_diabetes_pca, generate_dataset_semeion_pca, generate_dataset_wine_pca, generate_dataset_concrete_pca, generate_dataset_energy_pca, generate_dataset_synthetic, generate_dataset_airfoil
from jax_training.models_jax import QNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def generate_run_name(param_dict: Dict[str, Any]) -> str:
    return "_".join(f"{k}{v}" for k, v in sorted(param_dict.items()))

# -----------------------------------------------------------------------------
# core experiment launcher
# -----------------------------------------------------------------------------

def run_experiment(
    param_dict: Dict[str, Any],
    target_epochs: int,
    *,
    experiment_name: str = "default_experiment",
    db_path: str = "sqlite:///mlflow.db",
    prune_callback=None,
    print_output: bool = False,
    use_mlflow: bool = True,
    smoothing: float = 0.0,
    test_size: float = 0.0,
    val_size: float = 0.15,
):
    run_name = generate_run_name(param_dict)
    logging.info("Starting experiment '%s'", run_name)

    # ---- 1) open MLflow context ------------------------------------------------
    if use_mlflow:
        initialize_experiment(experiment_name, db_path)
        run_id = get_or_create_run(experiment_name, run_name)
        run_ctx = start_run(run_id)
    else:
        run_ctx = dummy_run()

    # ---- 2) check for previous run state --------------------------------------
    start_epoch = 0
    params = opt_state = None
    best_state_blob_prev = None
    best_val_loss_prev = None
    best_epoch_prev = None

    if use_mlflow:
        try:
            client = mlflow.tracking.MlflowClient()
            existing_run = client.get_run(run_id)
            if existing_run and existing_run.info.lifecycle_stage == "active":
                # last‑epoch model -------------------------------------------------
                try:
                    art_path = mlflow.artifacts.download_artifacts(
                        artifact_path=f"model_state_{run_name}.pkl", run_id=run_id
                    )
                    with open(art_path, "rb") as fh:
                        params, opt_state = load_state_from_bytes(fh.read())
                except Exception as e:
                    logging.info("No last‑epoch checkpoint yet: %s", e)

                # best checkpoint -------------------------------------------------
                try:
                    best_art_path = mlflow.artifacts.download_artifacts(
                        artifact_path=f"best_model_state_{run_name}.pkl", run_id=run_id
                    )
                    with open(best_art_path, "rb") as fh:
                        best_state_blob_prev = fh.read()
                except Exception:
                    pass  # not logged yet

                # resume epoch number -------------------------------------------
                tl_hist = client.get_metric_history(run_id, "train_loss")
                if tl_hist:
                    start_epoch = max(m.step for m in tl_hist) + 1

                # best metric history ------------------------------------------
                bl_hist = client.get_metric_history(run_id, "best_val_loss")
                if bl_hist:
                    best_val_loss_prev = bl_hist[-1].value
                    best_epoch_prev = bl_hist[-1].step

                if start_epoch >= target_epochs:
                    logging.info("Target epochs reached previously – nothing to do.")
                    return {}
        except Exception as e:
            logging.warning("Resume check failed: %s", e)

    # ---- 3) build model + initial TrainState ----------------------------------
    rng = jax.random.PRNGKey(param_dict["ind_trajectory"])
    num_features = param_dict["num_features"]
    dummy_x = jnp.ones((1, num_features))

    model = QNN(
        num_features=num_features,
        num_frequencies=param_dict["num_frequencies"],
        layer_depth=param_dict["layer_depth"],
        num_output=1,
        init_std=param_dict["init_std"],
        init_std_Q=param_dict["init_std_Q"],
        frequency_min_init=2.0 * np.pi,
        trainable_frequency_min=False,
        ad=param_dict["ad"],
        pd=param_dict["pd"],
        dp=param_dict["dp"],
    )

    state = create_train_state_reg(
        module=model,
        rng=rng,
        learning_rate=param_dict["learning_rate"],
        weight_decay=param_dict["weight_decay"],
        x_item=dummy_x,
    )
    if params is not None:
        state = state.replace(params=params, opt_state=opt_state)

    # ---- 4) dataset ------------------------------------------------------------
#     X_tr, y_tr, X_val, y_val, X_test, y_test = generate_dataset_concrete_pca(
#         n_components=num_features,
#         test_size=test_size,
#         val_size=val_size,
#         random_state=0,
#     )
    
    noise_std = param_dict["noise_std"]
    X_tr, y_tr, X_val, y_val, X_test, y_test = generate_dataset_diabetes_pca(
            n_components=num_features,
            test_size=test_size,
            val_size=val_size,
            random_state=0,
            noise_std_features = noise_std,
            noise_std_targets = noise_std
        )

    # ---- 5) log hyperparams on fresh run --------------------------------------
    if use_mlflow and start_epoch == 0:
        param_dict["num_parameters"] = count_parameters(state.params)
        log_params(param_dict)

    # ---- 6) training -----------------------------------------------------------
    logging.info("Begin training from epoch %d", start_epoch)
    with run_ctx:
        (
            state,
            metrics_hist,
            best_state_blob,
            best_val_loss,
            best_epoch,
        ) = train_reg(
            state=state,
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            y_val=y_val,
            batch_size=param_dict["batch_size"],
            target_epochs=target_epochs,
            start_epoch=start_epoch,
            print_output=print_output,
            initial_best_state_blob=best_state_blob_prev,
            initial_best_val_loss=best_val_loss_prev,
            initial_best_epoch=best_epoch_prev,
        )

        if use_mlflow:
            # metrics (all epochs) ------------------------------------
            log_all_metrics_batch(run_id, metrics_hist, start_epoch)

            # last‑epoch model ---------------------------------------
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
            tmp.write(save_state_to_bytes(state.params, state.opt_state))
            tmp.close()
            log_artifact(tmp.name, f"model_state_{run_name}.pkl")
            os.remove(tmp.name)

            # best checkpoint + metric -------------------------------
            if best_state_blob is not None:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
                tmp.write(best_state_blob)
                tmp.close()
                log_artifact(tmp.name, f"best_model_state_{run_name}.pkl")
                os.remove(tmp.name)
                mlflow.log_metric("best_val_loss", best_val_loss, step=best_epoch)
                mlflow.set_tag("best_epoch", best_epoch)

            logging.info("Run '%s' complete (run_id=%s)", run_name, run_id)

    return metrics_hist

# -----------------------------------------------------------------------------
# Optuna / Celery wrappers (unchanged except import path fixes)
# -----------------------------------------------------------------------------


def worker(args):
    run_experiment(*args)
    return None


def optimize_in_process(objective, storage, study_name, n_trials):
    study = optuna.load_study(study_name=study_name, storage=storage)
    study.optimize(objective, n_trials=n_trials)

# Celery task ------------------------------------------------------

# broker_url = "redis://127.0.0.1:6379/0"
broker_url = "redis://172.21.0.2:6379/0"
app = Celery("experiments_run", broker=broker_url, backend=broker_url)


@app.task
def run_experiment_task(
    param_dict,
    target_epochs,
    db_path,
    prune_callback,
    print_output,
    use_mlflow,
    smoothing,
    test_size,
    val_size,
    experiment_name,
):
    try:
        logging.info("Celery task starting: %s", param_dict)
        return run_experiment(
            param_dict,
            target_epochs=target_epochs,
            experiment_name=experiment_name,
            db_path=db_path,
            prune_callback=prune_callback,
            print_output=print_output,
            use_mlflow=use_mlflow,
            smoothing=smoothing,
            test_size=test_size,
            val_size=val_size,
        )
    except Exception as exc:
        logging.error("Task failed: %s", exc)
        raise
    finally:
        if mlflow.active_run():
            mlflow.end_run()
