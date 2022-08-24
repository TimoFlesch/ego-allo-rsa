import torch
from torch import nn


class RNN(nn.Module):
    """recurrent neural network with a single recurrent hidden layer and a linear readout layer.
    inputs are n_batch x n_frame x n_input streams of pixel intensities where
    n_input corresponds to the number of pixels in a flattened image and n_frames
    to the number of consecutive frames (hence the need for recurrence)
    Arguments:
            input_size (int, optional): number of features in individual input frames. Defaults to 10.
            hidden_size (int, optional): number of hidden units. Defaults to 100.
            output_size (int, optional): number of output units. Defaults to 2.
            nonlinearity (str, optional): nonlinearity in the hidden layer. Defaults to "tanh".
            device (torch.device, optional): device to run code on (either gpu or cpu). Defaults to
                torch.device("cuda" if torch.cuda.is_available() else "cpu").

    """

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 100,
        output_size: int = 2,
        nonlinearity: str = "tanh",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """

        Args:
            input_size (int, optional): number of features in individual input frames. Defaults to 10.
            hidden_size (int, optional): number of hidden units. Defaults to 100.
            output_size (int, optional): number of output units. Defaults to 2.
            nonlinearity (str, optional): nonlinearity in the hidden layer. Defaults to "tanh".
            device (torch.device, optional): device to run code on (either gpu or cpu).
             Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").
        """
        super().__init__()
        self.n_inputs = input_size
        self.n_hidden = hidden_size
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity=nonlinearity,
            batch_first=True,
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.ro = None
        self.device = device

    def forward(self, x):
        h0 = self._init_hidden(batch_size=x.shape[0])
        self.ro, _ = self.rnn(x, h0)
        y = self.fc(self.ro[:, -1, :])
        return y

    def _init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.n_hidden).to(self.device)

    @property
    def hidden_states(self):
        return self.ro.cpu().detach().numpy()
