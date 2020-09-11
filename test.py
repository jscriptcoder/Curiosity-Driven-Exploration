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
config.action_size = 1

agent = SACAgent(config)
agent.summary()

agent.train()