# With or without value function
# With or without discounted returns
# With or without experience
# With or without exploration
import os
import numpy as np
import torch

from model.environment import Environment
from model.policy import Policy
from utils.settings_reader import SettingsReader

SETTINGS = os.path.join(os.path.dirname(__file__), 'environment_settings.json')
torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    env_settings = SettingsReader.read(SETTINGS)
    env = Environment(**env_settings)
    state = env.reset()

    observation_space = env.observation_space
    action_space = env.action_space

    INPUT_DIM = observation_space[2]
    H1_DIM = 2
    H2_DIM = 20
    OUTPUT_DIM = 1

    actor = Policy(INPUT_DIM, H1_DIM, H2_DIM, OUTPUT_DIM)
    prev_action = np.random.rand(action_space + 1) * 10
    prev_action = prev_action / np.linalg.norm(prev_action)
    action = actor.forward(state, prev_action)
