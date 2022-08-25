from torch import optim

from ego_allo_rnns.configs.rnn import cfg_WCWC as cfg
from ego_allo_rnns.data.EgoVsAllo import make_datasets
from ego_allo_rnns.models.rnns import RNN
from ego_allo_rnns.trainers.train_rnn import train_model


def run_training():
    # import data
    data = make_datasets(**cfg["data"])

    # instantiate model
    rnn = RNN(**cfg["architecture"])

    # train and eval model
    optimiser = optim.SGD(rnn.parameters(), cfg["hyperparams"]["lr"])
    train_model(data, rnn, optimiser, **cfg["training"])


if __name__ == "__main__":

    run_training()
