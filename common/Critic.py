import torch
import torch.nn as nn
import torch.nn.functional as F
from common.utils import device
from common import BaseNetwork


class Critic(BaseNetwork):
    def __init__(self, state_size, action_size, hidden_size, activ):
        super().__init__(activ)

        dims = (state_size,) + hidden_size + (action_size,)

        self.build_layers(dims)
        self.reset_parameters()
        self.to(device)

    def forward(self, state):
        """Maps state to Q-values, Q(s) => Q-values"""
        if type(state) != torch.Tensor:
            state = torch.FloatTensor(state).to(device)

        x = state

        for layer in self.layers[:-1]:
            x = self.activ(layer(x))

        return self.layers[-1](x)