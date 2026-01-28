import torch
import torch.nn as nn


class LowdimEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=None, hidden_dims=None):
        super().__init__()
        self.input_dim = int(input_dim)
        hidden_dims = list(hidden_dims or [])
        output_dim = int(output_dim) if output_dim is not None else self.input_dim

        layers = []
        in_dim = self.input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, int(dim)))
            layers.append(nn.ReLU())
            in_dim = int(dim)
        if in_dim != output_dim or not layers:
            layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x):
        return self.net(x)
