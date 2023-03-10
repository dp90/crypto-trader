import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from rltools.algorithms import PPO
from rltools.azure.utils import parse_hyperparameter_args
from rltools.agents import MLPCritic, PpoGaussianActor
from rltools.utils import LoggingConfig

from configs import DirectoryConfig as DIR, TradingConfig as TC
from utils import extract_neural_net_dims
from trader.environment import BinanceEnvironment
from trader.evaluation import (evaluate, evaluate_mvo, evaluate_mcvar,
    evaluate_equal_weights, set_to_evaluation_state, evaluate_hrp)
from trader.rewards import RewardGenerator
from trader.states import create_hist_state_processor,\
    create_limit_order_state_processor

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


def train(hp):
    hp = extract_neural_net_dims(hp)
    torch.manual_seed(hp.seed)
    np.random.seed(hp.seed)
    random.seed(hp.seed)

    # ACTION_DIM = TC.N_ASSETS + 1
    ACTION_DIM = TC.N_ASSETS * 4
    OBS_DIM = (TC.N_INDICATORS + 6) * TC.N_ASSETS + TC.N_ASSETS + 1

    hist_state_processor = create_limit_order_state_processor(TC, DIR.DATA)
    reward_generator = RewardGenerator(TC, hist_state_processor.book_keeper)
    env = BinanceEnvironment(hist_state_processor, reward_generator)

    # actor = PpoGaussianActor(OBS_DIM, ACTION_DIM, hp.actor_hidden_dims, nn.ReLU(), 
    #                          nn.Softmax(dim=-1), hp.action_log_std)    
    actor = PpoGaussianActor(OBS_DIM, ACTION_DIM, hp.actor_hidden_dims, nn.ReLU(), 
                             nn.ReLU(), hp.action_log_std)
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
        "actor_optimizer": optim.Adam,
        "critic_optimizer": optim.Adam,
        "actor_learning_rate": hp.actor_learning_rate,
        "critic_learning_rate": hp.critic_learning_rate,
        "max_trajectory_length": 100,
    }

    ppo = PPO(env, actor, critic, train_params, buffer_params)
    ppo.train(verbose=True)

    eval_state_processor = create_hist_state_processor(TC, DIR.DATA)
    eval_env = BinanceEnvironment(eval_state_processor, reward_generator)
    set_to_evaluation_state(env)
    set_to_evaluation_state(eval_env)
    
    pf_values = evaluate(env, ppo.actor)
    pf_values_equal_weights = evaluate_equal_weights(eval_env)
    pf_values_mvo = evaluate_mvo(eval_env)
    pf_values_mcvar = evaluate_mcvar(eval_env)
    pf_values_hrp = evaluate_hrp(eval_env)
    plt.figure()
    plt.plot(pf_values, label='RL', color='k')
    plt.plot(pf_values_equal_weights, label='equal weights', color='r')
    plt.plot(pf_values_mvo, label='mvo weights', color='b')
    plt.plot(pf_values_mcvar, label='mCVaR weights', color='g')
    plt.plot(pf_values_hrp, label='HRP weights', color='grey')
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    HYPERPARAMETER_PATH = os.path.join(DIR.ROOT, 'scripts', 'hyperparameter_settings.toml')
    params = parse_hyperparameter_args(HYPERPARAMETER_PATH)
    logger.info(f"{params}")
    train(params)
