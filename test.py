import gym
import time
import gym.spaces
from common import Config
from agent import SACAgent

env = gym.make('LunarLander-v2')

config = Config()
config.env = env
config.env_solved = 200
config.state_size = env.observation_space.shape[0]
config.action_size = env.action_space.n
config.batch_size = 128
config.update_every = 1
config.max_steps = 500
# config.max_steps_reward = -100

config.tau = 1e-2
config.lr_actor = 3e-4
config.lr_critic = 3e-4
config.hidden_actor = (64, 64)
config.hidden_critic = (256, 256)

# config.grad_clip_actor = 5
# config.grad_clip_critic = 5

config.lr_alpha = 3e-4

agent = SACAgent(config)
agent.summary()

agent.train()