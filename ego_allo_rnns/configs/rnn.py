import torch

cfg = {
    "data": {
        "n_training_samples": 5000,
        "n_test_samples": 100,
        "img_resolution": 40,
    },
    "architecture": {
        "n_inputs": 1600,
        "n_hidden": 10,
        "nonlinearity": "tanh",
    },
    "hyperparams": {
        "learning_rate": 1e-3,
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
