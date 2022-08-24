import torch
from torch import nn


class RNN(nn.Module):
    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 100,
        output_size: int = 2,
        nonlinearity: str = "tanh",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(RNN, self).__init__()
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
