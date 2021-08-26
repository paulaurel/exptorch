import torch
from torch import nn

from exptorch.utils.itertools import pairwise


class LinearNet(nn.Module):
    """Linear feed-forward neural network, i.e. linear regression model."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, loss):
        x, y = batch
        x_pred = self.forward(x)
        return loss(x_pred, y)

    def configure_optimizers(self, optimizers: torch.optim, **kwargs):
        optimizer, *_ = optimizers
        return optimizer(self.parameters(), **kwargs)


def get_mlp_layers(channels: list, activation, output_activation=nn.Identity):
    """Define basic multilayered perceptron network architecture."""
    layers = []
    *intermediate_layer_definitions, final_layer_definition = pairwise(channels)

    for in_ch, out_ch in intermediate_layer_definitions:
        intermediate_layer = nn.Linear(in_ch, out_ch)
        layers += [intermediate_layer, activation()]

    layers += [nn.Linear(*final_layer_definition), output_activation()]
    return nn.Sequential(*layers)


class MLP(nn.Module):
    """Multi-layered perceptron network."""

    def __init__(
        self, input_dim: int, output_dim: int, channels: list, activation=nn.ReLU
    ):
        super().__init__()
        mlp_channels = [input_dim] + channels + [output_dim]
        self.net = get_mlp_layers(mlp_channels, activation)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, loss):
        x, y = batch
        x = x.view(x.shape[0], -1)
        y_pred = self.forward(x)
        return loss(y_pred, y)

    def configure_optimizers(self, optimizers: torch.optim, **kwargs):
        optimizer, *_ = optimizers
        return optimizer(self.net.parameters(), **kwargs)
