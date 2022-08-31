import pytest
from torch import optim

from ego_allo_rnns.data.EgoVsAllo import make_datasets
from ego_allo_rnns.models.rnns import RNN
from ego_allo_rnns.trainers.train_rnn import train_model
from ego_allo_rnns.utils.config import load_config


class TestFullTrainingFlow:
    """tests the full training run.
    Components:
        - load config
        - dataset generation and import
        - model instantiation
        - training loop + logging
    """

    def test_training_flow(self):
        # load configuration file
        cfg = load_config(
            cfg_id="integration_test", cfg_path="./ego_allo_rnns/configs/"
        )

        # import data
        data = make_datasets(**cfg["data"])

        # instantiate model
        rnn = RNN(**cfg["architecture"])

        # init optimiser
        optimiser = optim.SGD(rnn.parameters(), cfg["hyperparams"]["lr"])
        # train model
        train_model(data, rnn, optimiser, **cfg["training"])


if __name__ == "__main__":

    pytest.main([__file__])
