import torch

cfg = {
    "data": {
        "n_train": 10000,
        "n_test": 2000,
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
        "checkpoint_dir": "./checkpoints/",
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
        "checkpoint_dir": "./checkpoints/",
        "run_id": "run_integration_test",
        "n_epochs": 10,
        "batch_size": 256,
        "log_interval": 1,
        "criterion": torch.nn.MSELoss(),
        "device": "cpu",
    },
}


cfg_WCWC = {
    "data": {
        "n_train": 10000,
        "n_test": 2000,
        "size_ds": 40,
        "input_type": "WC",
        "output_type": "WC",
    },
    "architecture": {
        "input_size": 1600,
        "hidden_size": 100,
        "nonlinearity": "tanh",
        "output_size": 2,
    },
    "hyperparams": {
        "lr": 1e-3,
    },
    "training": {
        "log_dir": "./logs/",
        "checkpoint_dir": "./checkpoints/",
        "run_id": "run_WCWC",
        "n_epochs": 200,
        "batch_size": 256,
        "log_interval": 1,
        "criterion": torch.nn.MSELoss(),
    },
}

cfg_WCSC = {
    "data": {
        "n_train": 10000,
        "n_test": 2000,
        "size_ds": 40,
        "input_type": "WC",
        "output_type": "SC",
    },
    "architecture": {
        "input_size": 1600,
        "hidden_size": 100,
        "nonlinearity": "tanh",
        "output_size": 2,
    },
    "hyperparams": {
        "lr": 1e-3,
    },
    "training": {
        "log_dir": "./logs/",
        "checkpoint_dir": "./checkpoints/",
        "run_id": "run_WCSC",
        "n_epochs": 200,
        "batch_size": 256,
        "log_interval": 1,
        "criterion": torch.nn.MSELoss(),
    },
}

cfg_SCWC = {
    "data": {
        "n_train": 10000,
        "n_test": 2000,
        "size_ds": 40,
        "input_type": "SC",
        "output_type": "WC",
    },
    "architecture": {
        "input_size": 1600,
        "hidden_size": 100,
        "nonlinearity": "tanh",
        "output_size": 2,
    },
    "hyperparams": {
        "lr": 1e-3,
    },
    "training": {
        "log_dir": "./logs/",
        "checkpoint_dir": "./checkpoints/",
        "run_id": "run_SCWC",
        "n_epochs": 200,
        "batch_size": 256,
        "log_interval": 1,
        "criterion": torch.nn.MSELoss(),
    },
}


cfg_SCSC = {
    "data": {
        "n_train": 10000,
        "n_test": 2000,
        "size_ds": 40,
        "input_type": "SC",
        "output_type": "SC",
    },
    "architecture": {
        "input_size": 1600,
        "hidden_size": 100,
        "nonlinearity": "tanh",
        "output_size": 2,
    },
    "hyperparams": {
        "lr": 1e-3,
    },
    "training": {
        "log_dir": "./logs/",
        "checkpoint_dir": "./checkpoints/",
        "run_id": "run_SCSC",
        "n_epochs": 200,
        "batch_size": 256,
        "log_interval": 1,
        "criterion": torch.nn.MSELoss(),
    },
}
