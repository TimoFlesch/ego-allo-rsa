from argparse import ArgumentParser

import numpy as np

from ego_allo_rnns.trainers.train_rnn import run_training
from ego_allo_rnns.utils.config import load_config
from ego_allo_rnns.utils.io import save_training_results

parser = ArgumentParser()
parser.add_argument(
    "--n_workers",
    default=1,
    help="Number of workers for parallel processing of training runs. Defaults to 1",
)
parser.add_argument(
    "--n_runs", default=1, help="Number of independent training runs. Defaults to 1."
)
parser.add_argument(
    "--configs",
    nargs="+",
    default=["WCWC", "WCSC", "SCWC", "SCSC"],
    help="config files to load.",
)
args = parser.parse_args()
args = vars(args)  # type: ignore


def collect_runs(args: dict):  # type: ignore
    seeds = np.random.randint(1, 99999, size=args["n_runs"])
    for cfg_id in args["configs"]:
        try:
            cfg = load_config(cfg_id=cfg_id, cfg_path="./configs/experiments/")
        except FileNotFoundError:
            cfg = load_config(cfg_id=cfg_id.lower(), cfg_path="./configs/experiments/")

        for seed, r_id in zip(seeds, range(args["n_runs"])):
            cfg["training"]["run_id"] = cfg["training"]["run_id"] + "_" + str(r_id + 1)
            cfg["data"]["random_seed"] = seed
            model, results = run_training(cfg)
            save_training_results(cfg, model, results)


if __name__ == "__main__":

    collect_runs(args)  # type: ignore
