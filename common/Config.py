from torch.nn import ReLU
from torch.optim import Adam


class Config():
    seed = 0
    env = None
    log_every = 100

    # When we reach env_solved avarage score (our target score for this environment),
    # we'll run a full evaluation, that means, we're gonna evaluate times_solved #
    # times (this is required to solve the env) and avarage all the rewards:
    env_solved = None
    times_solved = 100

    buffer_size = int(1e6)
    batch_size = 128
    num_episodes = 2000
    num_updates = 1 # how many updates we want to perform in one learning step
    max_steps = 2000 # max steps done per episode if done is never True
    
    # Reward after reaching `max_steps` (punishment, hence negative reward)
    max_steps_reward = None
    
    state_size = None
    action_size = None
    gamma = 0.99 # discount factor
    tau = 1e-3 # interpolation param, used in polyak averaging (soft update)
    lr_actor = 3e-4
    lr_critic = 3e-4
    hidden_actor = (256, 256)
    hidden_critic = (256, 256)
    activ_actor = ReLU()
    activ_critic = ReLU()
    optim_actor = Adam
    optim_critic = Adam
    grad_clip_actor = None # gradient clipping for actor network
    grad_clip_critic = None # gradient clipping for critic network
    use_huber_loss = False # whether to use huber loss (True) or mse loss (False)
    update_every = 1 # how many steps before updating networks

    alpha = 0.01
    alpha_auto_tuning = True # when True, alpha is a learnable
    optim_alpha = Adam # optimizer for alpha
    lr_alpha = 3e-4 # learning rate for alpha