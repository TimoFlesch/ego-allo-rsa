from ego_allo_rnns.configs.rnn import cfg


def load_config(cfg_id: str = "default") -> dict:
    if cfg_id == "default":
        return cfg
    else:  # todo
        return cfg
