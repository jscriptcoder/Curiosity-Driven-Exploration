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
# config.tau = 1e-2
# config.lr_actor = 1e-3
# config.lr_critic = 1e-3

agent = SACAgent(config)
agent.summary()

agent.train()