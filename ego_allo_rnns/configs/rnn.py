import torch

cfg = {
    "data": {
        "n_train": 5000,
        "n_test": 200,
        "size_ds": 40,
    },
    "architecture": {
        "input_size": 1600,
        "hidden_size": 10,
        "nonlinearity": "tanh",
        "output_size": 2,
    },
    "hyperparams": {
        "lr": 1e-3,
    },
    "training": {
        "log_dir": "./logs/",
        "run_id": "run_1",
        "n_epochs": 10,
        "batch_size": 256,
        "log_interval": 1,
        "criterion": torch.nn.MSELoss(),
    },
}


cfg_integration_test = {
    "data": {
        "n_train": 1000,
        "n_test": 100,
        "size_ds": 40,
    },
    "architecture": {
        "input_size": 1600,
        "hidden_size": 10,
        "nonlinearity": "tanh",
        "output_size": 2,
        "device": "cpu",
    },
    "hyperparams": {
        "lr": 1e-3,
    },
    "training": {
        "log_dir": "./logs/",
        "run_id": "run_integration_test",
        "n_epochs": 10,
        "batch_size": 256,
        "log_interval": 1,
        "criterion": torch.nn.MSELoss(),
        "device": "cpu",
    },
}
