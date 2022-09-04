from abc import ABC

from project.infrastructure.replay_buffer import ReplayBuffer, Buffer
from project.policies.MLP_policy import MLPPolicy, MLPSACPolicy, MLPA2CPolicy
from project.values.MLP_value import MLPValue, MLPTD3Value, MLPPPOValue
from project.infrastructure.state import StateReprModuleAve, StateReprModuleP, StateReprModuleU, StateReprModuleN
from project.infrastructure.noise import OUNoise, NormalNoiseStrategy, NormalNoiseDecayStrategy
from project.infrastructure.PMF import PMF
import torch.nn.functional as F
from project.infrastructure.utils import calc_adv_ref, calc_logprob
from .base_agent import BaseAgent
import os

import torch
import numpy as np


class DRRAgent(BaseAgent, ABC):
    def __init__(self, params):
        super(DRRAgent, self).__init__()

        # init vars
        self.params = params

        self.pmf = PMF()
        self.pmf.set_params({"num_feat": self.params['embedding_dim'], "maxepoch": 100})

        state_params = [self.params['user_num'], self.params['item_num'],
                        self.params['embedding_dim'], self.params['N'],
                        self.params['state_repr_lr'],
                        self.params['state_repr_decay']]

        if params['state'] == 'drr_ave':
            self.state_repr = StateReprModuleAve(*state_params)
            ob_dim = self.params['embedding_dim'] * 3

        elif params['state'] == 'drr_p':
            self.state_repr = StateReprModuleP(*state_params)
            ob_dim = int(self.params['embedding_dim'] * (self.params['N'] + self.params['N'] * (
                    self.params['N'] - 1) / 2))

        elif params['state'] == 'drr_u':
            self.state_repr = StateReprModuleU(*state_params)
            ob_dim = int(self.params['embedding_dim'] * (self.params['N'] + self.params['N'] * (
                    self.params['N'] - 1) / 2))

        elif params['state'] == 'drr_n':
            self.state_repr = StateReprModuleN(*state_params)
            ob_dim = self.params['embedding_dim'] * self.params['N']

        self.noise = None
        if params['noise'] == 'ou_noise':
            self.noise = OUNoise(params['embedding_dim'])

        if params['noise'] == 'nor_noise':
            self.noise = NormalNoiseStrategy(params['embedding_dim'])

        if params['noise'] == 'nor_dec_noise':
            self.noise = NormalNoiseDecayStrategy(params['embedding_dim'])

        # actor/policy
        if self.params['method'] == 'ddpg':
            self.actor = MLPPolicy(
                self.params['embedding_dim'],
                ob_dim,
                self.params['n_layers'],
                self.params['hidden_dim'],
                learning_rate=self.params['policy_lr'],
                decay=self.params['policy_decay']
            )
            self.target_actor = MLPPolicy(self.params['embedding_dim'],
                                          ob_dim,
                                          self.params['n_layers'],
                                          self.params['hidden_dim'],
                                          learning_rate=self.params['policy_lr'],
                                          decay=self.params['policy_decay'])

            self.critic = MLPValue(
                self.params['embedding_dim'],
                ob_dim,
                self.params['n_layers'],
                self.params['hidden_dim'],
                learning_rate=self.params['value_lr'],
                decay=self.params['value_decay']
            )

            self.target_critic = MLPValue(self.params['embedding_dim'],
                                          ob_dim,
                                          self.params['n_layers'],
                                          self.params['hidden_dim'],
                                          learning_rate=self.params['value_lr'],
                                          decay=self.params['value_decay'])
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data)

            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data)

        if self.params['method'] == 'td3':
            self.actor = MLPPolicy(
                self.params['embedding_dim'],
                ob_dim,
                self.params['n_layers'],
                self.params['hidden_dim'],
                learning_rate=self.params['policy_lr'],
                decay=self.params['policy_decay']
            )
            self.target_actor = MLPPolicy(self.params['embedding_dim'],
                                          ob_dim,
                                          self.params['n_layers'],
                                          self.params['hidden_dim'],
                                          learning_rate=self.params['policy_lr'],
                                          decay=self.params['policy_decay'])

            self.critic = MLPTD3Value(
                self.params['embedding_dim'],
                ob_dim,
                self.params['n_layers'],
                self.params['hidden_dim'],
                learning_rate=self.params['value_lr'],
                decay=self.params['value_decay']
            )

            self.target_critic = MLPTD3Value(self.params['embedding_dim'],
                                             ob_dim,
                                             self.params['n_layers'],
                                             self.params['hidden_dim'],
                                             learning_rate=self.params['value_lr'],
                                             decay=self.params['value_decay'])

            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data)

            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data)

        if self.params['method'] == 'sac':
            self.actor = MLPSACPolicy(
                self.params['embedding_dim'],
                ob_dim,
                self.params['n_layers'],
                self.params['hidden_dim'],
                learning_rate=self.params['policy_lr'],
                decay=self.params['policy_decay']
            )
            self.critic_a = MLPValue(
                self.params['embedding_dim'],
                ob_dim,
                self.params['n_layers'],
                self.params['hidden_dim'],
                learning_rate=self.params['value_lr'],
                decay=self.params['value_decay']
            )
            self.critic_b = MLPValue(
                self.params['embedding_dim'],
                ob_dim,
                self.params['n_layers'],
                self.params['hidden_dim'],
                learning_rate=self.params['value_lr'],
                decay=self.params['value_decay']
            )

            self.target_critic_a = MLPValue(self.params['embedding_dim'],
                                            ob_dim,
                                            self.params['n_layers'],
                                            self.params['hidden_dim'],
                                            learning_rate=self.params['value_lr'],
                                            decay=self.params['value_decay'])

            self.target_critic_b = MLPValue(self.params['embedding_dim'],
                                            ob_dim,
                                            self.params['n_layers'],
                                            self.params['hidden_dim'],
                                            learning_rate=self.params['value_lr'],
                                            decay=self.params['value_decay'])

            for target_param, param in zip(self.target_critic_a.parameters(), self.critic_a.parameters()):
                target_param.data.copy_(param.data)

            for target_param, param in zip(self.target_critic_b.parameters(), self.critic_b.parameters()):
                target_param.data.copy_(param.data)

        if self.params['method'] == 'ppo':
            self.actor = MLPPolicy(
                self.params['embedding_dim'],
                ob_dim,
                self.params['n_layers'],
                self.params['hidden_dim'],
                learning_rate=self.params['policy_lr'],
                decay=self.params['policy_decay'],
                std=True
            )
            self.action_actor = MLPA2CPolicy(self.actor)
            self.critic = MLPPPOValue(
                ob_dim,
                self.params['n_layers'],
                self.params['hidden_dim'],
                learning_rate=self.params['value_lr'],
                decay=self.params['value_decay']
            )
            self.target_actor = MLPPolicy(
                self.params['embedding_dim'],
                ob_dim,
                self.params['n_layers'],
                self.params['hidden_dim'],
                learning_rate=self.params['policy_lr'],
                decay=self.params['policy_decay'],
                std=True
            )

            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data)

        # replay buffer
        if self.params['method'] != 'ppo':
            self.replay_buffer = ReplayBuffer()
        else:
            self.replay_buffer = Buffer()

    def ddpg_train(self, params):
        batch_size = params['batch_size']
        gamma = params['gamma']
        soft_tau = params['soft_tau']
        idxs, weights, \
        (user, memory, action, reward, next_user, next_memory) = self.replay_buffer.sample(batch_size)
        user = torch.FloatTensor(user)
        memory = torch.FloatTensor(memory)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        next_user = torch.FloatTensor(next_user)
        next_memory = torch.FloatTensor(next_memory)

        state = self.state_repr(user, memory)
        policy_loss = self.critic(state, self.actor(state))
        policy_loss = -policy_loss.mean()

        next_state = self.state_repr(next_user, next_memory)
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())
        expected_value = reward + gamma * target_value

        value = self.critic(state, action)
        td_error = value - expected_value.detach()
        value_loss = (torch.FloatTensor(weights) * td_error).pow(2).mul(0.5).mean()

        self.state_repr.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic.optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self.critic.optimizer.step()
        self.state_repr.optimizer.step()

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        priorities = np.abs(td_error.detach().cpu().numpy())
        self.replay_buffer.update(idxs, priorities)

    def td3_train(self, params, a_ran=5, policy_noise_ratio=0.1):
        batch_size = params['batch_size']
        gamma = params['gamma']
        soft_tau = params['soft_tau']
        idxs, weights, \
        (user, memory, action, reward, next_user, next_memory) = self.replay_buffer.sample(batch_size)
        user = torch.FloatTensor(user)
        memory = torch.FloatTensor(memory)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        next_user = torch.FloatTensor(next_user)
        next_memory = torch.FloatTensor(next_memory)

        state = self.state_repr(user, memory)

        next_state = self.state_repr(next_user, next_memory)

        with torch.no_grad():
            a_noise = torch.randn_like(action) * policy_noise_ratio * a_ran

            argmax_a_q_sp = self.target_actor(next_state)
            noisy_argmax_a_q_sp = argmax_a_q_sp + a_noise
            max_a_q_sp_a, max_a_q_sp_b = self.target_critic(next_state, noisy_argmax_a_q_sp)
            max_a_q_sp = torch.min(max_a_q_sp_a, max_a_q_sp_b)

            target_q_sa = reward + gamma * max_a_q_sp

        q_sa_a, q_sa_b = self.critic(state, action)
        td_error_a = q_sa_a - target_q_sa
        td_error_b = q_sa_b - target_q_sa

        value_loss = td_error_a.pow(2).mul(0.5).mean() + td_error_b.pow(2).mul(0.5).mean()

        argmax_a_q_s = self.actor(state)
        max_a_q_s = self.critic.Qa(state, argmax_a_q_s)

        self.state_repr.optimizer.zero_grad()
        policy_loss = -max_a_q_s.mean()
        self.actor.optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic.optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self.critic.optimizer.step()
        self.state_repr.optimizer.step()

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        priorities = (np.abs(td_error_a.detach().cpu().numpy()) + np.abs(td_error_b.detach().cpu().numpy())) / 2
        self.replay_buffer.update(idxs, priorities)

    def sac_train(self, params):
        batch_size = params['batch_size']
        gamma = params['gamma']
        soft_tau = params['soft_tau']
        idxs, weights, \
        (user, memory, action, reward, next_user, next_memory) = self.replay_buffer.sample(batch_size)
        user = torch.FloatTensor(user)
        memory = torch.FloatTensor(memory)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        next_user = torch.FloatTensor(next_user)
        next_memory = torch.FloatTensor(next_memory)

        state = self.state_repr(user, memory)

        next_state = self.state_repr(next_user, next_memory)

        current_actions, logpi_s = self.actor.full_pass(state)

        target_alpha = (logpi_s + self.actor.target_entropy).detach()
        alpha_loss = -(self.actor.logalpha * target_alpha).mean()

        self.actor.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.actor.alpha_optimizer.step()
        alpha = self.actor.logalpha.exp()

        current_q_sa_a = self.target_critic_a(state, current_actions)
        current_q_sa_b = self.target_critic_b(state, current_actions)
        current_q_sa = torch.min(current_q_sa_a, current_q_sa_b)
        policy_loss = (alpha * logpi_s - current_q_sa).mean()

        # Q loss
        ap, logpi_sp = self.actor.full_pass(next_state)
        q_spap_a = self.target_critic_a(next_state, ap)
        q_spap_b = self.target_critic_b(next_state, ap)
        q_spap = torch.min(q_spap_a, q_spap_b) - alpha * logpi_sp
        target_q_sa = (reward + gamma * q_spap).detach()

        q_sa_a = self.critic_a(state, action)
        q_sa_b = self.critic_b(state, action)
        td_error_a = q_sa_a - target_q_sa
        td_error_b = q_sa_b - target_q_sa
        qa_loss = td_error_a.pow(2).mul(0.5).mean()
        qb_loss = td_error_b.pow(2).mul(0.5).mean()

        self.state_repr.optimizer.zero_grad()

        self.actor.optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_a.optimizer.zero_grad()
        qa_loss.backward(retain_graph=True)
        self.critic_a.optimizer.step()

        self.critic_b.optimizer.zero_grad()
        qb_loss.backward(retain_graph=True)
        self.critic_b.optimizer.step()

        self.state_repr.optimizer.step()

        for target_param, param in zip(self.target_critic_a.parameters(), self.critic_a.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        for target_param, param in zip(self.target_critic_b.parameters(), self.critic_b.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        priorities = (np.abs(td_error_a.detach().cpu().numpy()) + np.abs(td_error_b.detach().cpu().numpy())) / 2
        self.replay_buffer.update(idxs, priorities)

    def ppo_train(self, epoch=10):
        for user in self.replay_buffer.users.keys():
            for _ in range(epoch):
                sample = self.replay_buffer.sample(user)
                if len(sample[:2]) < 2:
                    continue
                traj_states = [self.state_repr(torch.FloatTensor([user]), torch.FloatTensor([memory])) for memory in
                               sample[:, 0]]
                traj_actions = list(sample[:, 1])
                traj_states_v = torch.FloatTensor(torch.cat(traj_states))
                traj_actions_v = torch.FloatTensor(traj_actions)
                traj_adv_v, traj_ref_v = calc_adv_ref(
                    sample[:, 2], self.critic, traj_states_v, device='cpu')
                mu_v = self.target_actor(traj_states_v)
                old_logprob_v = calc_logprob(
                    mu_v, self.target_actor.logstd, traj_actions_v)

                # normalize advantages
                traj_adv_v = traj_adv_v - torch.mean(traj_adv_v)
                traj_adv_v /= torch.std(traj_adv_v)

                # drop last entry from the trajectory, an our adv and ref value calculated without it
                old_logprob_v = old_logprob_v[:-1].detach()
                traj_states_v = traj_states_v[:-1]
                traj_actions_v = traj_actions_v[:-1]
                states_v = traj_states_v
                actions_v = traj_actions_v
                batch_adv_v = traj_adv_v
                batch_adv_v = batch_adv_v.unsqueeze(-1)
                batch_ref_v = traj_ref_v
                batch_old_logprob_v = \
                    old_logprob_v

                # critic training
                self.state_repr.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                value_v = self.critic(states_v)
                loss_value_v = F.mse_loss(
                    value_v.squeeze(-1), batch_ref_v)
                loss_value_v.backward(retain_graph=True)
                self.critic.optimizer.step()

                # actor training
                self.actor.optimizer.zero_grad()
                mu_v = self.actor(states_v)
                logprob_pi_v = calc_logprob(
                    mu_v, self.actor.logstd, actions_v)
                ratio_v = torch.exp(
                    logprob_pi_v - batch_old_logprob_v)
                surr_obj_v = batch_adv_v * ratio_v
                loss_policy_v = -surr_obj_v.mean()
                loss_policy_v.backward(retain_graph=True)
                self.actor.optimizer.step()
                self.state_repr.optimizer.step()

            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data)
        self.replay_buffer.update()

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.store(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample(batch_size)  # HW1: you will modify this

    def save(self, path, names, best=False):
        if best:
            self.actor.save(os.path.join(path, f'best_{names[0]}_final.pth'))
            self.state_repr.save(os.path.join(path, f'best_{names[2]}_final.pth'))
            if self.params['method'] == 'sac':
                self.critic_a.save(os.path.join(path, f'best_{names[1]}_a_final.pth'))
                self.critic_b.save(os.path.join(path, f'best_{names[1]}_b_final.pth'))
            else:
                self.critic.save(os.path.join(path, f'best_{names[1]}_final.pth'))
        else:
            self.actor.save(os.path.join(path, f'{names[0]}_final.pth'))
            self.state_repr.save(os.path.join(path, f'{names[2]}_final.pth'))
            if self.params['method'] == 'sac':
                self.critic_a.save(os.path.join(path, f'{names[1]}_a_final.pth'))
                self.critic_b.save(os.path.join(path, f'{names[1]}_b_final.pth'))
            else:
                self.critic.save(os.path.join(path, f'{names[1]}_final.pth'))
