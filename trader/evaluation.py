import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from rltools.agents import IActor
from rltools.utils import LoggingConfig

from trader.benchmarks import MvoAgent
from trader.environment import BinanceEnvironment

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


def evaluate(env: BinanceEnvironment, actor: IActor):    
    state = env.reset()
    pf_values = [env.state_processor.book_keeper.get_portfolio_value()]
    done = False
    actions = []
    while not done:
        dist = actor._distribution(torch.tensor(state))
        action = dist.mean.detach().numpy()
        state, _, done, _ = env.step(action)
        actions.append(action)
        pf_values.append(env.state_processor.book_keeper.get_portfolio_value())
    
    # plot_actions(np.vstack(actions), env.state_processor.binance.data_loader.config)
    return np.array(pf_values)


def evaluate_equal_weights(env: BinanceEnvironment):
    logger.info("Evaluating equal weights strategy ...")
    _ = env.reset()
    pf_values = [env.state_processor.book_keeper.get_portfolio_value()]
    done = False
    n_assets = env.state_processor.binance.c.N_ASSETS + 1
    action = np.ones(n_assets) / n_assets
    while not done:
        _, _, done, _ = env.step(action)
        pf_values.append(env.state_processor.book_keeper.get_portfolio_value())
    logger.info("... Done!")
    return np.array(pf_values)


def evaluate_mvo(env: BinanceEnvironment):
    logger.info("Evaluating MVO strategy ...")
    _ = env.reset()
    pf_values = [env.state_processor.book_keeper.get_portfolio_value()]
    done = False
    mvo_agent = MvoAgent(env.state_processor.binance.data_loader)
    while not done:
        action = mvo_agent.next_mvo_weights()
        _, _, done, _ = env.step(action)
        pf_values.append(env.state_processor.book_keeper.get_portfolio_value())
    logger.info("... Done!")
    return np.array(pf_values)


def evaluate_mcvar(env: BinanceEnvironment):
    logger.info("Evaluating MCVaR strategy ...")
    _ = env.reset()
    pf_values = [env.state_processor.book_keeper.get_portfolio_value()]
    done = False
    mvo_agent = MvoAgent(env.state_processor.binance.data_loader)
    while not done:
        action = mvo_agent.next_mcvar_weights()
        _, _, done, _ = env.step(action)
        pf_values.append(env.state_processor.book_keeper.get_portfolio_value())
    logger.info("... Done!")
    return np.array(pf_values)


def evaluate_hrp(env: BinanceEnvironment):
    logger.info("Evaluating Hierarchical Risk Parity strategy ...")
    _ = env.reset()
    pf_values = [env.state_processor.book_keeper.get_portfolio_value()]
    done = False
    mvo_agent = MvoAgent(env.state_processor.binance.data_loader)
    while not done:
        action = mvo_agent.next_hrp_weights()
        _, _, done, _ = env.step(action)
        pf_values.append(env.state_processor.book_keeper.get_portfolio_value())
    logger.info("... Done!")
    return np.array(pf_values)


def set_to_evaluation_state(env: BinanceEnvironment):
    env.state_processor.binance.data_loader.config.DATA_START_INDEX = \
        env.state_processor.binance.data_loader.config.DATA_FINAL_INDEX + 1
    env.state_processor.binance.data_loader.config.DATA_FINAL_INDEX += 289


def plot_actions(actions: np.ndarray, config):
    fix, ax = plt.subplots()
    time = list(range(len(actions)))
    ax.stackplot(time, actions.T, labels=['CASH'] + config.CURRENCIES, edgecolor='k')
    ax.set_title('RL Strategy')
    plt.legend()
    plt.show()
