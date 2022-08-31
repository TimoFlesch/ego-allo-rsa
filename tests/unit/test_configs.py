import pytest


class TestConfigs:
    def test_yaml_parser(self):
        from ego_allo_rnns.utils.config import load_config

        cfg = load_config(cfg_id="example")  # noqa F481


if __name__ == "__main__":

    pytest.main([__file__])
