import torch
import yaml


def load_config(cfg_id: str = "example", cfg_path: str = "./configs/") -> dict:

    # load corresponding yaml file and convert to dict
    with open(cfg_path + cfg_id + ".yml") as f:
        config = yaml.safe_load(f)
        config = dict(config)

    # update fields
    config = get_callable_lossfunct(config)
    # return dict
    return config


def get_callable_lossfunct(cfg: dict) -> dict:
    """slightly ugly helper function that replaces str in dict created from yaml file
       with corresponding callable. E.g. if "mse" provided, replace with torch.nn.MSELoss()

    Args:
        cfg (dict): config dict

    Returns:
        dict: updated config dict
    """

    if cfg["training"]["criterion"].upper() == "MSE":
        cfg["training"]["criterion"] = torch.nn.MSELoss()
    elif cfg["training"]["criterion"].upper() == "MAE":
        cfg["training"]["criterion"] = torch.nn.L1Loss()
    return cfg
