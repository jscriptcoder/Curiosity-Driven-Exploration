import time
import datetime
import torch
import numpy as np
from collections import namedtuple


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Helper to create a experience tuple with named fields
make_experience = namedtuple('Experience',
                             field_names=['state',
                                          'action',
                                          'reward',
                                          'next_state',
                                          'done'])

def hidden_init(layer):
    """Will return a tuple with the range for the initialization
    of hidden layers
    Args:
        layer (torch.nn.Layer)
    Returns:
        Tuple of int
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Args:
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def from_experience(experiences):
    """Returns a tuple with (s, a, r, s', d)
    Args:
        namedtuple: List of tensors
    Returns:
        Tuple of torch.Tensor
    """
    states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)

    actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(device)

    rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)

    next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)

    dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None])\
            .astype(np.uint8)).to(device)
    
    return states, actions, rewards, next_states, dones

def get_time_elapsed(start, end=None):
    """Returns a human readable (HH:mm:ss) time difference between two times
    Args:
        start (float)
        end (float): optional value
            Default: now
    """

    if end is None:
        end = time.time()
    elapsed = round(end-start)
    return str(datetime.timedelta(seconds=elapsed))