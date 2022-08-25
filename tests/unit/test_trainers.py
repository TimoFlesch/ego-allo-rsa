import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ego_allo_rnns.models.rnns import RNN


class TestTrainer:
    def test_import_trainer(self):
        try:
            from ego_allo_rnns.trainers.train_rnn import train_model  # noqa F401
        except ImportError as e:
            pytest.fail(e)

        try:
            from ego_allo_rnns.trainers.train_rnn import train  # noqa F401
        except ImportError as e:
            pytest.fail(e)

        try:
            from ego_allo_rnns.trainers.train_rnn import test  # noqa F401
        except ImportError as e:
            pytest.fail(e)

    def test_train_loop(self):
        from ego_allo_rnns.trainers.train_rnn import train

        rnn = RNN(input_size=5, hidden_size=10, device="cpu")
        optim = torch.optim.SGD(rnn.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        x = torch.randn(size=(100, 20, 5))
        y = torch.randn(size=(100, 2))
        data = TensorDataset(x, y)
        dl = DataLoader(data, batch_size=20, shuffle=True)
        losses = train(rnn, optim, criterion, dl, device="cpu")
        assert len(losses)

    def test_validation_loop(self):
        from ego_allo_rnns.trainers.train_rnn import test

        rnn = RNN(input_size=5, hidden_size=10, device="cpu")
        criterion = torch.nn.MSELoss()
        x = torch.randn(size=(100, 20, 5))
        y = torch.randn(size=(100, 2))
        data = TensorDataset(x, y)
        dl = DataLoader(data, batch_size=20, shuffle=True)
        losses = test(rnn, criterion, dl, device="cpu")
        assert len(losses)


if __name__ == "__main__":
    pytest.main([__file__])
