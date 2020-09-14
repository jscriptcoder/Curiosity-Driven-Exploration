import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
from common.utils import device
from common import BaseNetwork


class CategoricalPolicy(BaseNetwork):
    def __init__(self, 
                 state_size, 
                 action_size, 
                 hidden_size, 
                 activ):
        super().__init__(activ)
        
        dims = (state_size,) + hidden_size + (action_size,)
        
        self.build_layers(dims)
        self.reset_parameters()
        self.to(device)
        
    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, state):
        if type(state) != torch.Tensor:
            state = torch.FloatTensor(state).to(device)
        
        x = self.layers[0](state)
        
        for layer in self.layers[1:-1]:
            x = self.activ(layer(x))
        
        return self.layers[-1](x)
    
    def greedy_action(self, state):
        action_logits = self.forward(state)
        action = torch.argmax(action_logits, dim=1, keepdim=True)
        return action

    def sample_action(self, state, eps=1e-6):
        action_logits = self.forward(state)
        action_probs = F.softmax(action_logits, dim=1)
        action_dist = Categorical(action_probs)
        action = action_dist.sample().view(-1, 1)
        
        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * eps
        log_action_probs = torch.log(action_probs + z)
        
        return action, action_probs, log_action_probs