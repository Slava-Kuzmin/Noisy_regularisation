# experiments_run.py
import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Use this variable instead of JAX_PLATFORM_NAME.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import tempfile
import logging
import math
import jax
import jax.numpy as jnp
import optuna
import mlflow  # ensure mlflow is imported
import numpy as np

from jax_training.mlflow_tracking import (
    initialize_experiment,
    get_or_create_run,
    start_run,
    log_params,
    log_metrics,
    log_artifact,
    dummy_run,
    log_all_metrics_batch
)

from jax_training.training_jax import (
    create_train_state_cls,
    train_cls,  # Assumes train_cls uses validation data.
    save_state_to_bytes,
    load_state_from_bytes,
    count_parameters,
    train_reg,
    create_train_state_reg
)

from jax_training.pca_datasets import generate_dataset_semeion_pca, generate_dataset_diabetes_pca

# Import models from models_jax.py
from jax_training.models_jax import QNN

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def generate_run_name(param_dict):
    if isinstance(param_dict, dict):
        # Sort the dictionary items and join key and value for each pair.
        return "_".join(f"{key}{value}" for key, value in sorted(param_dict.items()))
    else:
        # If it's not a dictionary, assume it's already a run name string.
        return str(param_dict)

    

def run_experiment(
    param_dict,
    target_epochs,
    experiment_name="default_experiment",
    db_path="sqlite:///experiments.db",
    prune_callback=None,      
    print_output=False,
    use_mlflow=True,
    smoothing=0.0,
    test_size=0.,
    val_size=0.15
):
    print(param_dict)
    # Generate a run name directly from the parameter dictionary.
    run_name = generate_run_name(param_dict)
    logging.info(f"Starting experiment with run name: {run_name}")
    
    # Initialize MLflow run context or dummy context.
    if use_mlflow:
        logging.info("Initializing MLflow experiment tracking...")
        initialize_experiment(experiment_name, db_path=db_path)
        run_id = get_or_create_run(experiment_name, run_name)
        run_context = start_run(run_id)
    else:
        logging.info("Running without MLflow tracking.")
        run_context = dummy_run()
    
    # Attempt to resume previous run if available.
    start_epoch = 0
    state = None
    if use_mlflow:
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            existing_run = client.get_run(run_id)
            if existing_run and existing_run.info.lifecycle_stage == "active":
                logging.info(f"Found existing run: {run_name} with run_id={run_id}")
                artifact_uri = f"model_state_{run_name}.pkl"
                local_path = None
                try:
                    local_path = mlflow.artifacts.download_artifacts(
                        artifact_path=artifact_uri, run_id=run_id
                    )
                except Exception as e:
                    logging.warning(f"No artifact found; starting fresh. {e}")
                if local_path:
                    with open(local_path, "rb") as f:
                        params, opt_state = load_state_from_bytes(f.read())
                        
                    metrics = client.get_metric_history(run_id, "train_loss")
                    if metrics:
                        last_epoch = max(metric.step for metric in metrics)
                        start_epoch = last_epoch + 1
                        logging.info(f"Resuming training from epoch {start_epoch}.")
                        if start_epoch >= target_epochs:
                            logging.info("Target epochs already reached; skipping training.")
                            return {}  # Or return current metrics
                    else:
                        logging.warning("No metrics found; starting from scratch.")
        except Exception as e:
            logging.warning(f"Failed to resume previous run; starting fresh. Error: {e}")
            
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print([start_epoch, target_epochs])
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    if start_epoch >= target_epochs or target_epochs <= 0:
        metrics_history = {
            "train_loss": [],
#             "train_accuracy": [],
            "val_loss": [],
#             "val_accuracy": []
        }
        return metrics_history

    # Create new state.
    rng = jax.random.PRNGKey(0)
    model_type = param_dict["model_type"]
    num_features = param_dict["num_features"]
    # Determine dummy_x based on the model.
    dummy_x = jnp.ones((1, num_features))
    
    # Build model based on model_type.
#     if model_type == "FNN":
#         model = FNN(num_features=num_features, num_frequencies=param_dict["num_frequencies"],
#                     num_output=10, init_std=param_dict["init_std"],
#                     frequency_min_init=1.0, trainable_frequency_min=True)
#     elif model_type == "QNN":
#         model = QNN(num_features=num_features, num_frequencies=param_dict["num_frequencies"],
#                     layer_depth=param_dict["layer_depth"], num_output=10, init_std=param_dict["init_std"],
#                     frequency_min_init=1.0, trainable_frequency_min=True)
#     elif model_type == "AENN":
#         model = AENN(num_features=num_features, layer_depth=param_dict["layer_depth"],
#                      num_output=10, init_std=param_dict["init_std"])
#     elif model_type == "SquareLinearModel":
#         model = SquareLinearModel(num_features=num_features, num_output=10, init_std=param_dict["init_std"])
#     elif model_type == "MPS_FNN":
#         model = MPS_FNN(num_features=num_features, bond_dim=param_dict["bond_dim"],
#                         num_frequencies=param_dict["num_frequencies"], num_output=10,
#                         init_std=param_dict["init_std"], frequency_min_init=1.0,
#                         trainable_frequency_min=True)
#     elif model_type == "LinearModel":
#         model = LinearModel(num_output=10)
    
    model = QNN(
        num_features=num_features, 
        num_frequencies=param_dict["num_frequencies"],
        layer_depth=param_dict["layer_depth"], 
        num_output=1, 
        init_std=param_dict["init_std"],
        init_std_Q=param_dict["init_std_Q"],
        frequency_min_init=2.0*np.pi, 
        trainable_frequency_min=False, 
        ad=param_dict["ad"], 
        pd=param_dict["pd"], 
        dp=param_dict["dp"])
        
    state = create_train_state_reg(module=model, rng=rng, learning_rate=param_dict["learning_rate"],
                                   weight_decay=param_dict["weight_decay"], x_item=dummy_x)
    
    if start_epoch > 0:
        state = state.replace(params=params, opt_state=opt_state)
    
    # Prepare dataset.
#     X_train, y_train, X_val, y_val, X_test, y_test = generate_dataset_semeion_pca(
#         n_components=num_features,
#         test_size=test_size,
#         val_size=val_size,
#         random_state=param_dict["ind_trajectory"]
#     )
    
    X_train, y_train, X_val, y_val, X_test, y_test = generate_dataset_diabetes_pca(
        n_components=num_features,
        test_size=test_size,
        val_size=val_size,
#         random_state=param_dict["ind_trajectory"]
        random_state = 0
    )
    
    # Log hyperparameters. Use param_dict directly plus add "num_parameters".
    if use_mlflow and start_epoch == 0:
        num_params = count_parameters(state.params)
        param_dict["num_parameters"] = num_params
        log_params(param_dict)
    
    logging.info("Starting training...")
    with run_context:
        state, metrics_history = train_reg(
            state=state,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            batch_size=param_dict["batch_size"],
            target_epochs=target_epochs,
            start_epoch=start_epoch,
#             smoothing=smoothing,
            print_output=print_output,
#             prune_callback=prune_callback
        )

        if use_mlflow:
            log_all_metrics_batch(run_id, metrics_history, start_epoch)
            
#         if use_mlflow:
#             for epoch in range(start_epoch, target_epochs + 1):
#                 log_metrics({
#                     "train_loss": metrics_history["train_loss"][epoch - start_epoch],
#                     "train_accuracy": metrics_history["train_accuracy"][epoch - start_epoch],
#                     "val_loss": metrics_history["val_loss"][epoch - start_epoch],
#                     "val_accuracy": metrics_history["val_accuracy"][epoch - start_epoch],
#                 }, step=epoch)
                
        if use_mlflow:
            logging.info("Saving final model state...")
            byte_data = save_state_to_bytes(state.params, state.opt_state)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
                temp_file.write(byte_data)
                artifact_path = temp_file.name
            artifact_filename = f"model_state_{run_name}.pkl"
            log_artifact(artifact_path, artifact_filename)
            os.remove(artifact_path)
            logging.info(f"Experiment completed and logged to MLflow: {run_name}")
        else:
            logging.info("Experiment completed without MLflow logging.")

    return metrics_history


def worker(args):
    run_experiment(*args)
    return None

def optimize_in_process(objective, storage, study_name, n_trials):
    """
    Run optimization using Optuna in-process.
    """
    study = optuna.load_study(study_name=study_name, storage=storage)
    study.optimize(objective, n_trials=n_trials)
    
    
    
import json
from celery import Celery

# broker_url = 'redis://172.17.0.3:6379/0'  # Replace with your Machine 1 IP
broker_url = 'redis://127.0.0.1:6379/0'  # Replace with your Machine 1 IP
app = Celery('experiments_run', broker=broker_url, backend=broker_url)


@app.task
def run_experiment_task(param_dict, target_epochs, db_path, prune_callback,
                        print_output, use_mlflow, smoothing,
                        test_size, val_size, experiment_name):
    """
    Celery task wrapper around `run_experiment`.
    Ensures that *any* MLflow run opened by this worker is closed,
    even if an exception occurs.
    """
    import mlflow  # local import keeps the top of the module lightweight

    try:
        logging.info("Starting run_experiment_task with parameters: %s", param_dict)
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
            val_size=val_size
        )
    except Exception as exc:
        logging.error("Error in run_experiment_task: %s", exc)
        raise
    finally:
        # ------------------------------------------------------------------
        # Good-practice clean-up: close any dangling active run.
        # This prevents the “Run ... is already active” error on the next task.
        # ------------------------------------------------------------------
        if mlflow.active_run() is not None:
            mlflow.end_run()
