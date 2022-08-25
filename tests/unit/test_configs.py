import pytest

from ego_allo_rnns.configs import rnn


class TestConfigs:
    def test_rnn_config(self):
        cfg = rnn.cfg
        keys = ["data", "architecture", "hyperparams", "training"]
        for k in keys:
            assert k in cfg.keys()


if __name__ == "__main__":

    pytest.main([__file__])
