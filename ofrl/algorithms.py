import os
import time
from typing import final
import torch
import numpy as np
import torch.optim as optim
from abc import ABC, abstractmethod
from datetime import datetime

from ofrl.agents import IValue, PpoActor
from ofrl.buffers import PpoBuffer, Trajectory
from ofrl.environments import IEnvironment
from ofrl.insights import plot_moving_average

torch.set_default_dtype(torch.float64)


class IAlgorithm(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, env, num_episodes, max_steps, render=False):
        raise NotImplementedError

    @abstractmethod
    def save(self, path):
        raise NotImplementedError


class PPO(IAlgorithm):
    """
    Implementation of the PPO algorithm. 
    """
    
    def __init__(self, env: IEnvironment, actor: PpoActor, critic: IValue, optim_actor: type[optim.Optimizer], 
                 optim_actor_params: dict, optim_critic: type[optim.Optimizer], optim_critic_params: dict,
                 train_params: dict, buffer_params: dict):
        """
        Args:
            env: environment
            actor: actor
            critic: critic
            n_epochs: number of iterations over entire scenario set (set to n_loops_over_scenario_set
                * (n_scenarios_in_set / n_scenario_batches) if using a scenario batch)
            n_episodes: number of episodes per epoch (set to 1 if using a scenario batch)
            batch_size: batch size
            clip_ratio: clip_ratio
            n_actor_updates: number of times actor is updated on all transitions in the buffer
            n_critic_updates: number of times critic is updated on all transitions in the buffer
            buffer_batch_size: number of transitions sampled from buffer per actor/critic update in SGD
        """
        self.env = env
        self.actor = actor
        self.critic = critic
        self.optim_actor = optim_actor(self.actor.parameters(), **optim_actor_params)
        self.optim_critic = optim_critic(self.critic.parameters(), **optim_critic_params)
        self.n_epochs = train_params['n_epochs']
        self.n_episodes = train_params['n_episodes']
        self.update_batch_size = train_params['update_batch_size']
        self.clip_ratio = train_params['clip_ratio']
        self.n_actor_updates = train_params['n_actor_updates']
        self.n_critic_updates = train_params['n_critic_updates']
        self.buffer = PpoBuffer(buffer_params['size'], buffer_params['state_dim'], buffer_params['action_dim'], 
                                train_params['update_batch_size'])
        self.reward_history = []
    
    def train(self, verbose: bool = True):
        start_time = time.time()
        for epoch in range(self.n_epochs):
            self.buffer.reset()
            self.collect_trajectories()
            self.update_actor_critic_networks()

            if verbose:
                self._log_progress(epoch, start_time)
        
        plot_moving_average(np.array(self.reward_history), window=1)

    def collect_trajectories(self):
        trajectory = Trajectory(max_length=100, gamma=1.0, gae_lambda=0.9)
        for _ in range(self.n_episodes):
            state = self.env.reset()
            done = False
            t = 0
            while not done:
                action, action_log_prob = self.actor.act(torch.tensor(state))
                value = self.critic(torch.tensor(state)).detach().numpy()

                new_state, reward, done, _ = self.env.step(action)
                trajectory.add(state, action, action_log_prob, reward, value)
                if len(trajectory) == trajectory.max_length:
                    new_value = self.critic(torch.tensor(state)).detach().numpy()
                    self.buffer.add(*trajectory.finish(new_value))

                state = new_state
                t += 1
            
            self.reward_history.append(np.array(trajectory.rewards).mean())
            self.buffer.add(*trajectory.finish(np.zeros(self.env.scenario_batch_size)))

    def update_actor_critic_networks(self):
        for _ in range(self.n_actor_updates):
            for batch in self.buffer.sample():
                loss_pi = self.compute_loss_pi(batch)  # pi = policy = actor
                self.optim_actor.zero_grad()
                loss_pi.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optim_actor.step()

        for _ in range(self.n_critic_updates):
            for batch in self.buffer.sample():
                loss_v = self.compute_loss_v(batch)
                self.optim_critic.zero_grad()
                loss_v.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optim_critic.step()

    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        _, logp = self.actor(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        return loss_pi

    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.critic(obs) - ret)**2).mean()

    def _log_progress(self, epoch, start_time):
        print(f'Epoch: {epoch + 1} / {self.n_epochs}')
        print(f'Time passed: {time.time() - start_time}s')
        print(f'Average reward: {self.reward_history[-1]}')

    def save(self, path):
        save_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        torch.save(self.actor.state_dict(), os.path.join(path, save_time + '_actor.pt'))
        torch.save(self.critic.state_dict(), os.path.join(path, save_time + '_critic.pt'))
        # TODO: save layer architecture
        # with open(path + save_time + '_architecture.txt', 'w') as f:
        #     f.write(str(self.actor.architecture))
        #     f.write(str(self.critic.architecture))
