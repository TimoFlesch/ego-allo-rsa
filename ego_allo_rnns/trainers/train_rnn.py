from typing import Callable, Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from ego_allo_rnns.data.EgoVsAllo import make_datasets
from ego_allo_rnns.models.rnns import RNN
from ego_allo_rnns.utils.config import load_config


def run_training(
    config: dict = None, config_path: str = "./configs/"
) -> Tuple[torch.nn.Module, dict]:
    """wrapper function that loads data, trains model and returns results

    Args:
        config (dict): configuration file

    Returns:
        Tuple[torch.nn.Module, dict]: trained model and log file
    """
    if config is None:
        config = load_config(cfg_id="example", cfg_path=config_path)
        print("no config provided, proceeding with default config")

    # import data
    data = make_datasets(**config["data"])

    # instantiate model
    rnn = RNN(**config["architecture"])

    # train and eval model
    optimiser = optim.SGD(rnn.parameters(), config["hyperparams"]["lr"])
    results = train_model(data, rnn, optimiser, **config["training"])

    # dump model and results
    rnn = rnn.to("cpu")

    return rnn, results


def train_model(
    data: Tuple[TensorDataset, TensorDataset],
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    criterion: Callable = torch.nn.MSELoss(),
    log_dir: str = "./logs/",
    run_id: str = "run_1",
    n_epochs: int = 10,
    batch_size: int = 256,
    log_interval: int = 1,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    checkpoint_dir: str = None,
) -> Dict[str, np.ndarray]:
    """loops over training data and performs SGD on provided model

    Args:
        data (Tuple[TensorDataset,TensorDataset]): training and validation data as TensorDatasets
        model (torch.nn.Module): a neural network
        optim (torch.optim.Optimizer): an optimiser (such as SGD or Adam)
        criterion (Callable): loss function. Defaults to MSE loss.
        log_dir (str): directory that contains logs. Defaults to "logs".
        run_id (str): name of run in log_dir. Defaults to "run_1".
        n_epochs (int): Number of training epochs. Defaults to 10.
        batch_size (int): batch size of training steps. Defaults to 256.
        log_interval (int): log to tensorboard every n epochs. Defaults to 1.

    Returns:
        dict: results: loss curves, outputs, patterns
    """
    writer = SummaryWriter(log_dir + run_id)

    training_generator = DataLoader(
        dataset=data[0], batch_size=batch_size, shuffle=True
    )
    test_generator = DataLoader(dataset=data[1], batch_size=50, shuffle=True)
    results: Dict[str, Dict[str, list]] = {
        "losses": {"training": [], "validation": []},
        "r_squared": {"training": [], "validation": []},
    }

    model.to(device)

    for ep in range(n_epochs):
        # Training
        training_loss = train(
            model, optim, criterion, training_generator, device=device
        )

        # validation
        validation_loss = test(model, criterion, test_generator, device=device)

        # logging:
        if (ep + 1) % log_interval == 0:
            r2_training = compute_r2(training_generator, model, device=device)
            r2_validation = compute_r2(test_generator, model, device=device)

            writer.add_scalar(
                "Losses/training",
                np.mean(training_loss),
                (ep + 1) * len(training_generator),
            )
            results["losses"]["training"].append(np.mean(training_loss))
            writer.add_scalar(
                "Losses/validation",
                np.mean(validation_loss),
                (ep + 1) * len(test_generator),
            )
            results["losses"]["validation"].append(np.mean(validation_loss))

            writer.add_scalar(
                "R_squared/training",
                r2_training,
                (ep + 1) * len(training_generator),
            )
            results["r_squared"]["training"].append(np.mean(r2_training))
            writer.add_scalar(
                "R_squared/validation",
                r2_validation,
                (ep + 1) * len(test_generator),
            )
            results["r_squared"]["validation"].append(np.mean(r2_validation))
    print("finished training. yay.")
    return results


def train(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    criterion: Callable,
    training_generator: DataLoader,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> list:
    losses = []
    model.train()
    for i, (x_batch, y_batch) in enumerate(training_generator):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # model training
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        losses.append(loss.item())

        optim.zero_grad()
        loss.backward()
        optim.step()
    return losses


def test(
    model: torch.nn.Module,
    criterion: Callable,
    test_generator: DataLoader,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> list:
    losses = []
    model.eval()
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(test_generator):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # model validation
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            losses.append(loss.item())
    return losses


def compute_r2(dl: DataLoader, model: torch.nn.Module, device: str = "cpu") -> float:
    y_true_all = []
    y_pred_all = []
    model.eval()
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(dl):
            x_batch = x_batch.to(device)
            y_pred_all.append(model(x_batch).detach().cpu().numpy())
            y_true_all.append(y_batch.detach().cpu().numpy())
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)
    return r2_score(y_true_all, y_pred_all)
