import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch


def save_training_results(cfg: dict, model: torch.nn.Module, results: dict):
    """Saves results from training run to disk.

    Args:
        cfg (dict): config file with paths
        model (torch.nn.Module): trained neural network
        results (dict): logs gathered during training
    """
    save_dir = Path(cfg["training"]["checkpoint_dir"]) / cfg["training"]["run_id"]
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "training_results.pkl", "wb") as f:
        pickle.dump(results, f)
        print(f"saved results to {save_dir}")

    with open(save_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
        print(f"saved model to {save_dir}")


def collect_training_results(
    checkpoint_dir: str = "./checkpoints/", run_name: str = "run_WCWC", n_runs: int = 20
) -> pd.DataFrame:
    """Collects results from individual training runs and returns them as pandas Dataframe.

    Args:
        checkpoint_dir (str, optional): path to checkpoints. Defaults to "./checkpoints/".
        run_name (str, optional): name of experiment to load. Defaults to "run_WCWC".
        n_runs (int, optional): number of training runs. Defaults to 20.

    Returns:
        pd.DataFrame: df with multiindexed rows: 1st index run, 2nd training epoch. columns for loss and r-squared
    """

    for rid in range(1, n_runs + 1):
        results = load_results(checkpoint_dir, run_name + f"_{rid}")
        if rid == 1:
            df = pd.DataFrame(
                np.asarray(
                    [
                        np.ones(len(results["losses"]["training"])),
                        np.arange(1, len(results["losses"]["training"]) + 1),
                        results["losses"]["training"],
                        results["losses"]["validation"],
                        results["r_squared"]["training"],
                        results["r_squared"]["validation"],
                    ]
                ).T,
                columns=[
                    "training run",
                    "epoch",
                    "training loss",
                    "validation loss",
                    "training r_squared",
                    "validation r_squared",
                ],
            )
        else:
            df = pd.concat(
                (
                    df,
                    pd.DataFrame(
                        np.asarray(
                            [
                                np.ones(len(results["losses"]["training"])),
                                np.arange(1, len(results["losses"]["training"]) + 1),
                                results["losses"]["training"],
                                results["losses"]["validation"],
                                results["r_squared"]["training"],
                                results["r_squared"]["validation"],
                            ]
                        ).T,
                        columns=[
                            "training run",
                            "epoch",
                            "training loss",
                            "validation loss",
                            "training r_squared",
                            "validation r_squared",
                        ],
                    ),
                ),
                ignore_index=True,
            )
    return df


def collect_trained_models(
    checkpoint_dir: str = "./checkpoints/", run_name: str = "run_WCWC", n_runs: int = 20
) -> List[torch.nn.Module]:
    """Collects and returns list of trained models.

    Args:
        checkpoint_dir (str, optional): path to checkpoints. Defaults to "./checkpoints/".
        run_name (str, optional): name of experiment to load. Defaults to "run_WCWC".
        n_runs (int, optional): number of training runs. Defaults to 20.

    Returns:
        List[torch.nn.Module]: list of neural network models
    """

    models = []
    for rid in range(1, n_runs + 1):
        models.append(load_model(checkpoint_dir, run_name + f"_{rid}"))
    return models


def load_results(checkpoint_dir: str, run_name: str) -> dict:
    with open(checkpoint_dir + run_name + "/training_results.pkl", "rb") as f:
        results = pickle.load(f)
    return results


def load_model(checkpoint_dir: str, run_name: str) -> torch.nn.Module:
    with open(checkpoint_dir + run_name + "/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model
