import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from rltools.algorithms import PPO
from rltools.azure.utils import parse_hyperparameter_args
from rltools.agents import MLPCritic, PpoGaussianActor
from rltools.utils import Scaler, LoggingConfig

from trader.environment import BinanceEnvironment
from trader.states import StateProcessor

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


def train(hp):
    # Client needs a config to set everything up?

    reward_generator = RewardGenerator()
    data_loader = BinanceDataLoader()
    scaler = Scaler({})

    binance_simulator = BinanceSimulator(data_loader)
    book_keeper = BookKeeper()
    action_converter = ActionConverter(book_keeper)
    market_interpreter = MarketInterpreter()

    state_processor = StateProcessor(scaler, binance_simulator, action_converter,
                                     market_interpreter)
    env = BinanceEnvironment(state_processor, reward_generator)

    actor = PpoGaussianActor(OBS_DIM, ACTOR_DIMS, HIDDEN_SIZES, nn.ReLU(), 
                             nn.Softmax(dim=-1), ACTION_LOG_STD)
    critic = MLPCritic(OBS_DIM, CRITIC_DIMS, nn.ReLU(), nn.Identity())

    buffer_params = {
        "size": SIZE,
        "state_dim": OBS_DIM,
        "action_dim": ACT_DIM,
    }
    train_params = {
        "n_epochs": N_EPOCHS,
        "n_episodes": N_EPISODES, 
        "clip_ratio": CLIP_RATIO,
        "n_actor_updates": N_ACTOR_UPDATES,
        "n_critic_updates": N_CRITIC_UPDATES,
        "update_batch_size": UPDATE_BATCH_SIZE,
        "discount_factor": DISCOUNT_FACTOR,
        "gae_lambda": GAE_LAMBDA,
    }

    ppo = PPO(env, actor, critic, optim.Adam, {'lr': ACTOR_LEARNING_RATE}, optim.Adam,
              {'lr': CRITIC_LEARNING_RATE}, train_params, buffer_params)

    evaluate()
    return


if __name__ == "__main__":
    HYPERPARAMETER_PATH = os.path.join(DIR.ROOT, 'opal', 'hyperparameter_settings.toml')
    params = parse_hyperparameter_args(HYPERPARAMETER_PATH)
    logger.info(f"{params}")
    train(params)
