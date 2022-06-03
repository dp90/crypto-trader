import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from configs import DirectoryConfig as DIR
from ofrl.agents import MLPCritic, PpoGaussianActor
from ofrl.algorithms import PPO
from ofrl.utils import Scaler
from opal.environments import OpalEnv
from opal.opal_config import OpalConfig as OPAL
from opal.rewards import RewardGeneratorD
from opal.utils import DataLoader, EMode


if __name__ == "__main__":
    SEED = 2
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    SCENARIO_BATCH_SIZE = 50

    mode = EMode.TRAIN
    reward_params = {
        "rwf_dist_1": 0.0,
        "rwf_dist_2": 0.0,
        "rwf_dist_3": 0.3,
        "rwf_dist_4": 0.0,
        "adaptation_rate": 0.04,
    }
    reward_generator = RewardGeneratorD(reward_params=reward_params)
    scaler = Scaler({15: (OPAL.GOAL_WEALTH, OPAL.GOAL_WEALTH), 16: (OPAL.SCENARIO_DURATION/2, OPAL.SCENARIO_DURATION)})
    env = OpalEnv(data_loader=DataLoader(), mode=mode, reward_generator=reward_generator, scaler=scaler,
                  scenario_batch_size=SCENARIO_BATCH_SIZE, n_scenarios=None, seed=SEED)

    # TODO: replace obs_dim with env.state_generator.observation_space.dim
    actor = PpoGaussianActor(obs_dim=17, act_dim=OPAL.N_ASSETS, hidden_sizes=(128, 128), activation=nn.ReLU(), 
                             output_activation=nn.Softmax(dim=-1), action_log_std=-0.7)
    critic = MLPCritic(obs_dim=17, hidden_sizes=(128, 128), activation=nn.ReLU(), output_activation=nn.Identity())

    buffer_params = {
        "size": OPAL.SCENARIO_DURATION * SCENARIO_BATCH_SIZE,
        "state_dim": 17,
        "action_dim": OPAL.N_ASSETS,
    }
    train_params = {
        # If scenarios are run in batches, set n_epochs to n_loops_over_scenario_set * (n_scenarios_in_set / n_scenario_batches),
        # and n_episodes to 1.
        "n_epochs": int(5 * (mode.n_scenarios // env.scenario_batch_size)),
        "n_episodes": 1,
        "clip_ratio": 0.1,
        "n_actor_updates": 4,
        "n_critic_updates": 4,
        "update_batch_size": 32,
    }
    
    ppo = PPO(env, actor, critic, optim.Adam, {'lr': 1e-4}, optim.Adam, {'lr': 1e-4}, train_params, buffer_params)
    ppo.train()
    ppo.save(DIR.MODELS)

    # Evaluate results
    mode = EMode.DEVELOP
    actor = ppo.actor
    eval_env = OpalEnv(data_loader=DataLoader(), mode=mode, reward_generator=reward_generator, scaler=scaler,
                       scenario_batch_size=mode.n_scenarios, n_scenarios=None, seed=123)

    portfolio_wealth = np.zeros((mode.n_scenarios, OPAL.SCENARIO_DURATION + 1))
    state = eval_env.reset()
    done = False
    portfolio_wealth[:, 0] = state[:, 15]
    t = 0
    while not done:
        dist = actor._distribution(torch.tensor(state))
        action = dist.mean.detach().numpy()
        # action, _ = actor.act(torch.tensor(state))
        # if ADJUST:
        #     action = satisfy_constraints(action)
        # actions_taken[:, t] = action
        # n_satisfied = get_satisfied_constraints(action).sum()

        state, _, done, _ = eval_env.step(action)
        portfolio_wealth[:, t+1] = scaler.unscale(state)[:, 15]
        t += 1

    print("\nFraction of scenarios that has wealth > 400K after 40 years:")
    print((portfolio_wealth[:, -1] >= OPAL.GOAL_WEALTH).sum() / len(portfolio_wealth[:, -1]))

    print("\nAverage final portfolio wealth:")
    print(np.mean(portfolio_wealth[:, -1]))
