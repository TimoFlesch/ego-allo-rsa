import pickle
from pathlib import Path

import torch


def save_training_results(cfg: dict, model: torch.nn.Module, results: dict):
    """saves results from training run to disk

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
