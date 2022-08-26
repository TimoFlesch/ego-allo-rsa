import pytest

from ego_allo_rnns.configs import rnn


class TestConfigs:
    def test_default_config(self):
        cfg = rnn.cfg
        keys = ["data", "architecture", "hyperparams", "training"]
        for k in keys:
            assert k in cfg.keys()

    def test_yaml_parser(self):
        from ego_allo_rnns.utils.config import load_config

        cfg = load_config(cfg_id="example")  # noqa F481


if __name__ == "__main__":

    pytest.main([__file__])
