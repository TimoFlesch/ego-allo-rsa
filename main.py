from ego_allo_rnns.trainers.train_rnn import run_training
from ego_allo_rnns.utils.config import load_config
from ego_allo_rnns.utils.io import save_training_results

if __name__ == "__main__":

    configs = ["WCWC", "WCSC", "SCWC", "SCSC"]
    for cfg_id in configs:
        try:
            cfg = load_config(cfg_id=cfg_id, cfg_path="./configs/experiments/")
        except FileNotFoundError:
            cfg = load_config(cfg_id=cfg_id.lower(), cfg_path="./configs/experiments/")

        model, results = run_training(cfg)
        save_training_results(cfg, model, results)
