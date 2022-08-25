import pytest
import torch


class TestNNets:
    def test_rnn_import(self):

        try:
            from ego_allo_rnns.models.rnns import RNN  # noqa F401
        except ImportError as e:
            return pytest.fail(e)

    def test_rnn_init(self):
        from ego_allo_rnns.models.rnns import RNN

        rnn = RNN(input_size=10, hidden_size=256, device="cpu")
        rnn

    def test_rnn_forwardpass(self):
        from ego_allo_rnns.models.rnns import RNN

        rnn = RNN(input_size=5, hidden_size=256, device="cpu")
        x = torch.randn(size=(1, 20, 5))  # 1 sample, 20 frames, 10 dims per frame
        y = rnn(x)
        assert len(y.tolist()[0]) == 2

    def test_rnn_gethidden(self):
        from ego_allo_rnns.models.rnns import RNN

        rnn = RNN(input_size=5, hidden_size=256, device="cpu")
        x = torch.randn(size=(1, 20, 5))  # 1 sample, 20 frames, 10 dims per frame
        _ = rnn(x)
        h = rnn.hidden_states
        assert h.shape == (1, 20, 256)


if __name__ == "__main__":

    pytest.main([__file__])
