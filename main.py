from ego_allo_rnns.configs.rnn import cfg_SCSC, cfg_SCWC, cfg_WCSC, cfg_WCWC
from ego_allo_rnns.trainers.train_rnn import run_training
from ego_allo_rnns.utils.io import save_training_results

if __name__ == "__main__":

    configs = [cfg_WCWC, cfg_WCSC, cfg_SCWC, cfg_SCSC]
    for cfg in configs:
        model, results = run_training(cfg)
        save_training_results(cfg, model, results)
