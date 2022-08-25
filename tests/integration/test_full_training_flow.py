import pytest
from torch import optim

from ego_allo_rnns.configs.rnn import cfg_integration_test
from ego_allo_rnns.data.EgoVsAllo import make_datasets
from ego_allo_rnns.models.rnns import RNN
from ego_allo_rnns.trainers.train_rnn import train_model


class TestFullTrainingFlow:
    """tests the full training run.
    Components:
        - load config
        - dataset generation and import
        - model instantiation
        - training loop + logging
    """

    def test_training_flow(self):
        # import data
        data = make_datasets(**cfg_integration_test["data"])

        # instantiate model
        rnn = RNN(**cfg_integration_test["architecture"])

        # train and eval model
        optimiser = optim.SGD(
            rnn.parameters(), cfg_integration_test["hyperparams"]["lr"]
        )
        train_model(data, rnn, optimiser, **cfg_integration_test["training"])


if __name__ == "__main__":

    pytest.main([__file__])
