import os
from itertools import product
import numpy as np
from experiments_run import run_experiment_task

# ---------------------------
# Main: parallel_experiments
# ---------------------------
def parallel_experiments(
    args,                    # Dictionary of hyperparameters
    target_epochs,           # Scalar: number of training epochs
    n_trajectories,          # Scalar: number of trajectory runs
    db_path,                 # MLflow database URI
    prune_callback,          # Optional: Optuna prune callback or None
    print_output,            # Bool: whether to print output
    use_mlflow,              # Bool: whether to log to MLflow
    smoothing,               # Float: smoothing for classification loss
    test_size,               # Float: test split ratio
    val_size,                # Float: validation split ratio
    experiment_name          # String: name for the MLflow experiment
):
    """
    Runs experiments in parallel using Celery for the Cartesian product of hyperparameters
    and multiple trajectory indices.
    """
    keys = list(args.keys())
    values_list = [
        list(args[k]) if isinstance(args[k], (list, tuple, np.ndarray)) else [args[k]]
        for k in keys
    ]

    tasks = []
    for combination in product(*values_list):
        base_param_dict = {k: v for k, v in zip(keys, combination)}
        for traj in range(n_trajectories):
            task_param_dict = base_param_dict.copy()
            task_param_dict["ind_trajectory"] = traj
            tasks.append(task_param_dict)

    # Submit each task asynchronously using Celery
    futures = []
    for param_dict in tasks:
        future = run_experiment_task.delay(
            param_dict, target_epochs, db_path, prune_callback,
            print_output, use_mlflow, smoothing, test_size, val_size, experiment_name
        )
        futures.append(future)

    return futures


import pandas as pd
import numpy as np
import math
from sqlalchemy import create_engine

def aggregate_metric_histories(metric_histories):
    """
    Given a list of metric history dictionaries, compute the mean and std for each epoch.
    Each history is assumed to have keys: "train_loss", "train_accuracy", "val_loss", "val_accuracy".
    
    Returns a dictionary with keys:
      "train_loss_mean", "train_loss_std",
      "train_accuracy_mean", "train_accuracy_std",
      "val_loss_mean", "val_loss_std",
      "val_accuracy_mean", "val_accuracy_std"
    """
    n_epochs = len(metric_histories[0]["train_loss"])
    agg = {}
#     for key in ["train_loss", "train_accuracy", "val_loss", "val_accuracy"]:
    for key in ["train_loss", "val_loss"]:
        runs = np.array([h[key] for h in metric_histories])
        agg[f"{key}_mean"] = runs.mean(axis=0)
        agg[f"{key}_std"] = runs.std(axis=0)
    return agg

def load_results(experiment_name, args, db_uri="sqlite:///mlflow.db", n_epochs = None):
    """
    Load and aggregate metric histories from the MLflow tracking database based on arbitrary filter arguments,
    and return a list of aggregated results in the order defined by the Cartesian product of the args values.

    Parameters:
      experiment_name (str): The MLflow experiment name.
      args (dict): A dictionary where each key is a parameter name (as logged in MLflow) and each value is either 
                   a scalar or a list/array of acceptable values. For example:
                   {
                       "model_type": ["FNN", "AENN"],
                       "learning_rate": [0.001, 0.0005],
                       "batch_size": [128],
                       "num_features": [6],
                       "init_std": [0.1],
                       "layer_depth": [1],
                       "num_frequencies": [1]
                   }
      db_uri (str): The MLflow tracking database URI.

    Returns:
      results_list (list): A list of aggregated metric history dictionaries (or None) for each combination
          in the Cartesian product of the args values, in order.
          Each aggregated metric dictionary should contain keys:
          "train_loss_mean", "train_loss_std", "train_accuracy_mean", "train_accuracy_std",
          "val_loss_mean", "val_loss_std", "val_accuracy_mean", "val_accuracy_std".
    """
    engine = create_engine(db_uri)
    
    # Query for the experiment ID.
    query_exp = f"SELECT experiment_id FROM experiments WHERE name = '{experiment_name}'"
    exp_df = pd.read_sql(query_exp, engine)
 
    if exp_df.empty:
        raise ValueError(f"Experiment '{experiment_name}' not found in the MLflow database.")
    experiment_id = exp_df.iloc[0]["experiment_id"]
    
    # Query runs and their parameters.
    query_runs = f"""
        SELECT 
            runs.run_uuid AS run_id,
            p.key AS param_key,
            p.value AS param_value
        FROM runs
        LEFT JOIN params p ON runs.run_uuid = p.run_uuid
        WHERE runs.experiment_id = {experiment_id}
    """
    runs_df = pd.read_sql(query_runs, engine)
    if runs_df.empty:
        raise ValueError(f"No runs found for experiment '{experiment_name}'.")
    
    # Pivot the runs so that each row is one run with parameter columns.
    params_pivot = runs_df.pivot_table(index="run_id", columns="param_key", values="param_value", aggfunc="first").reset_index()
    
    # For each key in args, filter the runs.
    for key, val in args.items():
        val_list = val if isinstance(val, (list, tuple, np.ndarray)) else [val]
        # Convert to string since parameters are stored as strings.
        val_list = [str(x) for x in val_list]
        if key in params_pivot.columns:
            params_pivot = params_pivot[params_pivot[key].isin(val_list)]
        else:
            # If a key is missing, no runs match.
            params_pivot = params_pivot[params_pivot.index < 0]
    
    run_ids = params_pivot["run_id"].unique()
    if len(run_ids) == 0:
        raise ValueError("No runs match the given filters.")
    
    # Query metric history for these runs.
    run_ids_str = ",".join([f"'{r}'" for r in run_ids])
    query_metrics = f"""
        SELECT 
            run_uuid AS run_id,
            `key` AS metric_key,
            `value` AS metric_value,
            step AS epoch
        FROM metrics
        WHERE run_uuid IN ({run_ids_str})
          AND `key` IN ('train_loss', 'train_accuracy', 'val_loss', 'val_accuracy')
    """
    metrics_df = pd.read_sql(query_metrics, engine)
    if metrics_df.empty:
        raise ValueError("No metric data found for the filtered runs.")
    
    # Pivot the metrics so that each row is (run_id, epoch) with columns for each metric.
    pivoted = metrics_df.pivot_table(index=["run_id", "epoch"], columns="metric_key", values="metric_value", aggfunc="first").reset_index()
    # Convert metric values to float.
#     for col in ['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']:
    for col in ['train_loss', 'val_loss']:
        if col in pivoted.columns:
            pivoted[col] = pd.to_numeric(pivoted[col], errors='coerce')
    
    # Build a dictionary mapping each run_id to its metric history (list of values per epoch).
    run_histories = {}
    for run_id, group in pivoted.groupby("run_id"):
        group = group.sort_values("epoch")
        history = {
            "train_loss": group["train_loss"].tolist()[:n_epochs],
#             "train_accuracy": group["train_accuracy"].tolist()[:n_epochs],
            "val_loss": group["val_loss"].tolist()[:n_epochs],
#             "val_accuracy": group["val_accuracy"].tolist()[:n_epochs]
        }
        run_histories[run_id] = history
    
    # Group run histories by a label built from the args.
    grouped = {}
    for _, row in params_pivot.iterrows():
        run_id = row["run_id"]
        # Create a label from the keys in args in the order provided by args.
        filter_label = "_".join([f"{k}{row[k]}" for k in args.keys() if k in row])
        if filter_label not in grouped:
            grouped[filter_label] = []
        if run_id in run_histories:
            grouped[filter_label].append(run_histories[run_id])
    
    # Now, build the ordered list.
    # First, ensure each arg value is a list (preserving order) for the keys in the order given by args.
    keys = list(args.keys())
    values_list = [list(args[k]) if isinstance(args[k], (list, tuple, np.ndarray)) else [args[k]] for k in keys]
    
    ordered_results = []
    # Iterate over the Cartesian product of the argument values.
    for combination in product(*values_list):
        # Build the filter label in the same way.
        label = "_".join([f"{k}{str(v)}" for k, v in zip(keys, combination)])
        if label in grouped:
            # Aggregate the trajectories for this group.
            aggregated = aggregate_metric_histories(grouped[label])
            ordered_results.append(aggregated)
        else:
            ordered_results.append(None)
    
    return ordered_results


import matplotlib.pyplot as plt
import numpy as np
from itertools import product

def plot_results(aggregated_results, labels=None, x_scale="log", y_scale="log", title=None):
    """
    Plot aggregated metric histories for classification from a list of aggregated results.

    The plot is arranged in a 2×2 grid:
      - Top-left: Training Loss vs. Epoch.
      - Top-right: Validation Loss vs. Epoch.
      - Bottom-left: Training Error (1 - Accuracy) vs. Epoch.
      - Bottom-right: Validation Error (1 - Accuracy) vs. Epoch.

    Parameters:
      aggregated_results (list): A list of aggregated metric history dictionaries.
      labels (list of str, optional): Legend labels.
      x_scale (str): Scale for the x-axis ("linear" or "log").
      y_scale (str): Scale for the y-axis ("linear" or "log").
      title (str, optional): Global plot title. Defaults to None.
    """
    if not aggregated_results:
        raise ValueError("The list of aggregated results is empty.")
        

    if labels is None:
        labels_to_use = [f"Group {i+1}" for i in range(len(aggregated_results))]
    else:
        if len(labels) != len(aggregated_results):
            raise ValueError("Number of labels must match number of result groups.")
        labels_to_use = labels

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    for agg, label in zip(aggregated_results, labels_to_use):
        
        n_epochs = len(agg["train_loss_mean"])
        epochs = np.arange(0, n_epochs)
        
        # Training Loss
        train_loss_mean = np.array(agg["train_loss_mean"])
        train_loss_std = np.array(agg["train_loss_std"])
        axs[0, 0].plot(epochs, train_loss_mean, label=f"{label} Train Loss")
        axs[0, 0].fill_between(epochs, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.2)

        # Validation Loss
        val_loss_mean = np.array(agg["val_loss_mean"])
        val_loss_std = np.array(agg["val_loss_std"])
        axs[0, 1].plot(epochs, val_loss_mean, label=f"{label} Val Loss")
        axs[0, 1].fill_between(epochs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.2)

        # Training Error
        train_acc_mean = np.array(agg["train_accuracy_mean"])
        train_acc_std = np.array(agg["train_accuracy_std"])
        train_error = 1 - train_acc_mean
        axs[1, 0].plot(epochs, train_error, label=f"{label} Train Error")
        axs[1, 0].fill_between(epochs, train_error - train_acc_std, train_error + train_acc_std, alpha=0.2)

        # Validation Error
        val_acc_mean = np.array(agg["val_accuracy_mean"])
        val_acc_std = np.array(agg["val_accuracy_std"])
        val_error = 1 - val_acc_mean
        axs[1, 1].plot(epochs, val_error, label=f"{label} Val Error")
        axs[1, 1].fill_between(epochs, val_error - val_acc_std, val_error + val_acc_std, alpha=0.2)

    for ax in axs.flat:
        ax.set_xlabel("Epoch")
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        ax.grid(True, which="both", ls="--")
        ax.legend()

    axs[0, 0].set_ylabel("Loss")
    axs[1, 0].set_ylabel("Error (1 - Accuracy)")
    axs[0, 0].set_title("Training Loss")
    axs[0, 1].set_title("Validation Loss")
    axs[1, 0].set_title("Training Error")
    axs[1, 1].set_title("Validation Error")

    if title:
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for title
    else:
        plt.tight_layout()
        
    return axs

    
def load_num_params(experiment_name, args, db_uri="sqlite:///mlflow.db"):
    """
    For a given MLflow experiment, load the number of parameters logged (as the 'num_parameters' parameter)
    for each combination of filter arguments specified in args.

    Parameters:
      experiment_name (str): Name of the MLflow experiment.
      args (dict): A dictionary where each key is a parameter name and each value is a scalar or list/array of acceptable values.
                   For example:
                   {
                       "model_type": ["FNN", "AENN"],
                       "batch_size": [128],
                       "init_std": [0.1]
                   }
      db_uri (str): URI for the MLflow tracking database.
    
    Returns:
      ordered_results (list): A list of values (int or None) corresponding to the "num_parameters" value for each
                              combination of args values in the order defined by the Cartesian product of the args.
                              If no run matches a given combination, None is returned in its place.
    """
    engine = create_engine(db_uri)
    
    # --- 1. Get the experiment_id from the experiments table.
    query_exp = f"SELECT experiment_id FROM experiments WHERE name = '{experiment_name}'"
    exp_df = pd.read_sql(query_exp, engine)
    if exp_df.empty:
        raise ValueError(f"Experiment '{experiment_name}' not found in the MLflow database.")
    experiment_id = exp_df.iloc[0]["experiment_id"]
    
    # --- 2. Query runs and their parameters.
    query_runs = f"""
        SELECT 
            runs.run_uuid AS run_id,
            p.key AS param_key,
            p.value AS param_value
        FROM runs
        LEFT JOIN params p ON runs.run_uuid = p.run_uuid
        WHERE runs.experiment_id = {experiment_id}
    """
    runs_df = pd.read_sql(query_runs, engine)
    if runs_df.empty:
        raise ValueError(f"No runs found for experiment '{experiment_name}'.")
    
    # --- 3. Pivot the runs so that each row corresponds to a run and each column to a parameter.
    params_pivot = runs_df.pivot_table(index="run_id", columns="param_key", values="param_value", aggfunc="first").reset_index()
    
    # --- 4. Filter the runs by the provided args.
    # For each key in args, restrict to rows where the column value is in the allowed list.
    for key, val in args.items():
        # Make sure the value is in list form.
        val_list = val if isinstance(val, (list, tuple, np.ndarray)) else [val]
        # Parameters are stored as strings.
        val_list = [str(x) for x in val_list]
        if key in params_pivot.columns:
            params_pivot = params_pivot[params_pivot[key].isin(val_list)]
        else:
            # If the key is missing in the logged parameters, no runs can match.
            params_pivot = params_pivot[params_pivot.index < 0]
    
    # --- 5. Build the ordered list of num_parameters based on the Cartesian product of the args values.
    keys = list(args.keys())
    # Ensure each value is a list and preserve the order.
    values_list = [list(args[k]) if isinstance(args[k], (list, tuple, np.ndarray)) else [args[k]] for k in keys]
    
    ordered_results = []
    for combination in product(*values_list):
        # For each combination, filter the pivoted dataframe.
        filtered = params_pivot.copy()
        for k, v in zip(keys, combination):
            filtered = filtered[filtered[k] == str(v)]
        
        if not filtered.empty:
            # Get the 'num_parameters' from the first matching run.
            num_params = filtered.iloc[0].get("num_parameters", None)
            # Optionally, convert to int if possible.
            try:
                num_params = int(num_params)
            except (ValueError, TypeError):
                num_params = None
            ordered_results.append(num_params)
        else:
            ordered_results.append(None)
    
    return ordered_results






################################################################
# Plotting functions
################################################################


import pandas as pd
import numpy as np
import math
from sqlalchemy import create_engine
from itertools import product

def aggregate_metric_histories(metric_histories):
    """
    Compute mean and std per epoch for regression losses.

    Each history must have keys:
        "train_loss", "val_loss"
    """
    n_epochs = len(metric_histories[0]["train_loss"])
    agg = {}

    for key in ["train_loss", "val_loss"]:
        runs = np.array([h[key] for h in metric_histories])
        agg[f"{key}_mean"] = runs.mean(axis=0)
        agg[f"{key}_std"]  = runs.std(axis=0)

    return agg


def load_results(experiment_name, args, db_uri="sqlite:///mlflow.db"):
    """
    Fetch and aggregate loss curves (train/val) from MLflow.
    """
    engine = create_engine(db_uri)

    # ---------------------------------------------------------
    # 1. Experiment ID
    # ---------------------------------------------------------
    exp_id = pd.read_sql(
        f"SELECT experiment_id FROM experiments WHERE name='{experiment_name}'",
        engine,
    )
    if exp_id.empty:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    experiment_id = exp_id.iloc[0, 0]

    # ---------------------------------------------------------
    # 2. Runs + parameters
    # ---------------------------------------------------------
    runs_df = pd.read_sql(f"""
        SELECT runs.run_uuid AS run_id,
               p.key        AS param_key,
               p.value      AS param_value
        FROM runs
        LEFT JOIN params p ON runs.run_uuid = p.run_uuid
        WHERE runs.experiment_id = {experiment_id}
    """, engine)

    if runs_df.empty:
        raise ValueError(f"No runs for experiment '{experiment_name}'.")

    params_pivot = runs_df.pivot_table(
        index="run_id", columns="param_key", values="param_value",
        aggfunc="first"
    ).reset_index()

    # ---------------------------------------------------------
    # 3. Filter by args
    # ---------------------------------------------------------
    for k, v in args.items():
        v_list = v if isinstance(v, (list, tuple, np.ndarray)) else [v]
        v_list = [str(i) for i in v_list]
        if k in params_pivot.columns:
            params_pivot = params_pivot[params_pivot[k].isin(v_list)]
        else:
            params_pivot = params_pivot[params_pivot.index < 0]

    run_ids = params_pivot["run_id"].unique()
    if len(run_ids) == 0:
        raise ValueError("No runs match the given filters.")

    # ---------------------------------------------------------
    # 4. Metric history (losses only)
    # ---------------------------------------------------------
    run_ids_str = ",".join(f"'{r}'" for r in run_ids)
    metrics_df = pd.read_sql(f"""
        SELECT run_uuid AS run_id,
               `key`    AS metric_key,
               `value`  AS metric_value,
               step     AS epoch
        FROM metrics
        WHERE run_uuid IN ({run_ids_str})
          AND `key` IN ('train_loss', 'val_loss')
    """, engine)

    if metrics_df.empty:
        raise ValueError("No metric data found for the filtered runs.")

    pivoted = metrics_df.pivot_table(
        index=["run_id", "epoch"], columns="metric_key",
        values="metric_value", aggfunc="first"
    ).reset_index()

    for col in ["train_loss", "val_loss"]:
        pivoted[col] = pd.to_numeric(pivoted[col], errors="coerce")

    # Build run->history dict
    run_histories = {}
    for run_id, g in pivoted.groupby("run_id"):
        g = g.sort_values("epoch")
        run_histories[run_id] = {
            "train_loss": g["train_loss"].tolist(),
            "val_loss":   g["val_loss"].tolist(),
        }

    # ---------------------------------------------------------
    # 5. Aggregate each Cartesian-product combination
    # ---------------------------------------------------------
    keys        = list(args.keys())
    values_list = [
        list(args[k]) if isinstance(args[k], (list, tuple, np.ndarray)) else [args[k]]
        for k in keys
    ]

    ordered_results = []
    for combo in product(*values_list):
        label = "_".join(f"{k}{v}" for k, v in zip(keys, combo))

        histories = [
            run_histories[row["run_id"]]                    # <- fixed
            for _, row in params_pivot.iterrows()
            if (row["run_id"] in run_histories) and
               "_".join(f"{k}{row[k]}" for k in keys) == label
        ]

        ordered_results.append(
            aggregate_metric_histories(histories) if histories else None
        )


    return ordered_results



import matplotlib.pyplot as plt
import numpy as np

def plot_results(aggregated_results, labels=None,
                 x_scale="log", y_scale="log", title=None):
    """
    Plot mean ± std of train/validation loss for regression.
    Creates a 1×2 subplot grid.
    """
    if not aggregated_results:
        raise ValueError("aggregated_results is empty.")

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    if labels is None:
        labels = [f"Group {i+1}" for i in range(len(aggregated_results))]
    if len(labels) != len(aggregated_results):
        raise ValueError("labels length must match aggregated_results length.")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for agg, lab in zip(aggregated_results, labels):
        n_epochs = len(agg["train_loss_mean"])
        epochs   = np.arange(n_epochs)

        # ---- train loss ------------------------------------------------
        tr_mean, tr_std = np.array(agg["train_loss_mean"]), np.array(agg["train_loss_std"])
        axs[0].plot(epochs, tr_mean, label=f"{lab} train")
        axs[0].fill_between(epochs, tr_mean - tr_std, tr_mean + tr_std, alpha=0.25)

        # ---- val loss --------------------------------------------------
        va_mean, va_std = np.array(agg["val_loss_mean"]), np.array(agg["val_loss_std"])
        axs[1].plot(epochs, va_mean, label=f"{lab} val")
        axs[1].fill_between(epochs, va_mean - va_std, va_mean + va_std, alpha=0.25)

    # ------------------------------------------------------------------
    # Cosmetics
    # ------------------------------------------------------------------
    for ax, ttl in zip(axs, ["Training loss", "Validation loss"]):
        ax.set_title(ttl)
        ax.set_xlabel("Epoch")
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        ax.grid(True, which="both", ls="--")
        ax.legend()

    axs[0].set_ylabel("Loss")
    if title:
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()

    plt.show()
    
################################################################