import os
import logging
import torch.nn as nn
import torch.optim as optim
from rltools.algorithms import PPO
from rltools.azure.utils import parse_hyperparameter_args
from rltools.agents import MLPCritic, PpoGaussianActor
from rltools.utils import Scaler, LoggingConfig

from configs import DirectoryConfig as DIR, TradingConfig as TC, \
    SimulationConfig as SC
from utils import extract_neural_net_dims
from trader.environment import BinanceEnvironment
from trader.evaluation import evaluate
from trader.rewards import RewardGenerator
from trader.states import create_hist_state_processor

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


def train(hp):
    # Client needs a config to set everything up?
    hp = extract_neural_net_dims(hp)
    ACTION_DIM = TC.N_ASSETS + 1
    OBS_DIM = TC.N_VARIABLES

    reward_generator = RewardGenerator(TC)
    hist_state_processor = create_hist_state_processor(TC, SC, DIR.DATA)
    env = BinanceEnvironment(hist_state_processor, reward_generator)

    actor = PpoGaussianActor(OBS_DIM, ACTION_DIM, hp.actor_hidden_dims, nn.ReLU(), 
                             nn.Softmax(dim=-1), hp.action_log_std)
    critic = MLPCritic(OBS_DIM, hp.critic_hidden_dims, nn.ReLU(), nn.Identity())

    buffer_params = {
        "size": hp.buffer_size,
        "state_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
    }
    train_params = {
        "n_epochs": hp.n_epochs,
        "n_episodes": hp.n_episodes, 
        "clip_ratio": hp.clip_ratio,
        "n_actor_updates": hp.n_actor_updates,
        "n_critic_updates": hp.n_critic_updates,
        "update_batch_size": hp.update_batch_size,
        "discount_factor": hp.discount_factor,
        "gae_lambda": hp.gae_lambda,
    }

    ppo = PPO(env, actor, critic, optim.Adam, {'lr': hp.actor_learning_rate}, optim.Adam,
              {'lr': hp.critic_learning_rate}, train_params, buffer_params)
    ppo.train(verbose=True)
    evaluate(env, ppo.actor)
    return


if __name__ == "__main__":
    HYPERPARAMETER_PATH = os.path.join(DIR.ROOT, 'scripts', 'hyperparameter_settings.toml')
    params = parse_hyperparameter_args(HYPERPARAMETER_PATH)
    logger.info(f"{params}")
    train(params)
