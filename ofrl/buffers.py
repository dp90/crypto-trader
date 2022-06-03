import numpy as np
import torch

from ofrl.utils import discount_cumsum


class Trajectory(object):
    """Handles parallel collection of trajectories"""

    def __init__(self, max_length, gamma=0.99, gae_lambda=0.9):
        self.max_length = max_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.states, self.actions, self.logprobs, self.rewards, self.values = [], [], [], [], []

    def add(self, state, action, logprob, reward, value):
        """Add a new transition to the trajectory."""
        if self.__len__() == self.max_length:
            raise RuntimeError('Trajectory is full')
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)

    def finish(self, last_value):
        """
        Computes advantages and returns for the trajectory, and resets it.

        Copied from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
        The "last_value" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        returns, advantages = self._compute_returns_and_advantages(last_value)
        states, actions, logprobs = self._states_actions_logprobs_to_numpy()
        transitions = states, actions, logprobs, returns, advantages
        self._reset()
        return transitions

    def _states_actions_logprobs_to_numpy(self):
        actions_dim = self.actions[0].shape[1]
        states_dim = self.states[0].shape[1]
        actions = np.array(self.actions).reshape((-1, actions_dim), order='F')
        states = np.array(self.states).reshape((-1, states_dim), order='F')
        logprobs = np.array(self.logprobs).reshape(-1, order='F')
        return states, actions, logprobs

    def _compute_returns_and_advantages(self, last_value):
        self.rewards.append(last_value)
        self.values.append(last_value)
        rewards = np.array(self.rewards)  # shape: (T, scenario_batch_size)
        values = np.array(self.values)  # shape: (T, scenario_batch_size)
        
        returns = discount_cumsum(rewards, self.gamma)[:-1].reshape(-1, order='F')
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        advantages = discount_cumsum(deltas, self.gamma * self.gae_lambda).reshape(-1, order='F')
        # returns = advantages + values[:-1].reshape(-1, order='F')
        return returns, advantages

    def _reset(self):
        self.states, self.actions, self.logprobs, self.rewards, self.values = [], [], [], [], []

    def __len__(self):
        return len(self.states)


class PpoBuffer(object):
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment.
    """
    def __init__(self, max_size, state_dim, action_dim, batch_size):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.reset()

    def add(self, states, actions, action_log_probs, returns, advantages):
        end_idx = self.idx + len(states)
        if end_idx > self.max_size:
            raise IndexError("Buffer is full. Increase the buffer size, or reduce the number of transitions " +
                             "collected per epoch.")
        self.states[self.idx:end_idx] = states
        self.actions[self.idx:end_idx] = actions
        self.logprobs[self.idx:end_idx] = action_log_probs
        self.returns[self.idx:end_idx] = returns
        self.advantages[self.idx:end_idx] = advantages
        self.idx = end_idx

    def sample(self):
        # normalize advantages - apparently this is a thing.
        self.advantages = (self.advantages - np.mean(self.advantages)) / np.std(self.advantages)
        
        shuffled_indices = torch.randperm(len(self))
        n_batches = len(self) // self.batch_size + 1  # If n_transitions / batch_size is not an integer, final batch is smaller
        for i in range(n_batches):
            end_idx = min(len(self), i * self.batch_size + self.batch_size)  # To prevent index out of bounds of final batch
            batch_indices = shuffled_indices[i * self.batch_size: end_idx]
            yield {"obs": torch.tensor(self.states[batch_indices]),
                   "act": torch.tensor(self.actions[batch_indices]),
                   "logp": torch.tensor(self.logprobs[batch_indices]).flatten(),
                   "ret": torch.tensor(self.returns[batch_indices]).flatten(),
                   "adv": torch.tensor(self.advantages[batch_indices]).flatten()}

    def reset(self):
        self.idx = 0
        self.states = np.zeros((self.max_size, self.state_dim))
        self.actions = np.zeros((self.max_size, self.action_dim))
        self.logprobs = np.zeros((self.max_size))
        self.returns = np.zeros((self.max_size))
        self.advantages = np.zeros((self.max_size))

    def __len__(self):
        return self.idx