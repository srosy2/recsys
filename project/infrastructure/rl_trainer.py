import pickle
import numpy as np
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.utils.data as td

from project.infrastructure import pytorch_util as ptu
from project.infrastructure.envionverment import Env
from project.infrastructure.utils import preprocess_data
from project.infrastructure.eval import EvalDataset, run_evaluation
from sklearn.model_selection import train_test_split
from .utils import to_np
import os


class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger, create TF session
        self.params = params

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)

        #############
        ## ENV
        #############

        # Make the gym environment
        self.params['data'], self.params['train_data'], self.params['train_matrix'], self.params['test_data'], \
        self.params['test_matrix'], self.params['user_num'], self.params['item_num'], \
        self.params['appropriate_users'] = preprocess_data(
            self.params['data_dir'], self.params['rating'], self.params['max_pos_item_user'], self.params[
                'min_pos_item_user']
        )

        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.params)

    def run_training_loop(self):

        # init vars at beginning of training
        np.random.seed(self.params['seed'])
        train_env = Env(self.params['train_data'], self.params['item_num'], self.params['user_num'], self.params['N'],
                        self.params['fill_users'])
        valid_dataset = EvalDataset(
            np.array(self.params['test_data'])[np.array(self.params['test_data'])[:, 0] == 3215],
            self.params['item_num'],
            self.params['test_matrix'])
        full_dataset = EvalDataset(np.array(self.params['test_data']), self.params['item_num'], self.params[
            'test_matrix'])
        hits, ndcgs = [], []
        hits_all, ndcgs_all = [], []
        best_hit, best_ndcg = 0, 0
        best_hit_all, best_ndcg_all = 0, 0
        step, best_step, best_step_all = 0, 0, 0
        users = np.random.permutation(self.params['appropriate_users'])
        train, test = train_test_split(np.array(self.params['data']), test_size=0.2, random_state=self.params['seed'])
        if self.params['pretrain_emb']:
            self.agent.pmf.fit(train, test)
            self.agent.state_repr.user_embeddings.weight = nn.Parameter(torch.Tensor(self.agent.pmf.w_User),
                                                                        requires_grad=self.params['train_emb']).float()
            self.agent.state_repr.item_embeddings.weight = nn.Parameter(torch.Tensor(self.agent.pmf.w_Item),
                                                                        requires_grad=self.params['train_emb']).float()

        for u in tqdm(users):
            user, memory = train_env.reset(u)
            if self.agent.noise is not None:
                self.agent.noise.reset()
            for t in range(self.params['episode_length']):
                if self.params['method'] == 'sac':
                    action_emb = self.agent.actor.select_action(self.agent.state_repr(user, memory))
                elif self.params['method'] == 'ppo':
                    action_emb = self.agent.action_actor(self.agent.state_repr(user, memory))
                else:
                    action_emb = self.agent.actor(self.agent.state_repr(user, memory))
                if self.agent.noise is not None:
                    action_emb = self.agent.noise.get_action(action_emb.detach().cpu().numpy()[0], t)
                action = self.agent.actor.get_action(
                    user,
                    torch.tensor(train_env.memory[to_np(user).astype(int), :]),
                    self.agent.state_repr,
                    action_emb,
                    torch.tensor(
                        [item for item in train_env.available_items
                         if item not in train_env.viewed_items]
                    ).long()
                )
                user, memory, reward = train_env.step(
                    action,
                    action_emb,
                    buffer=self.agent.replay_buffer
                )
                if self.params['method'] != 'ppo':
                    if len(self.agent.replay_buffer) > self.params['batch_size']:

                        if self.params['method'] == 'ddpg':

                            self.agent.ddpg_train(self.params)

                        elif self.params['method'] == 'td3':

                            self.agent.td3_train(self.params)

                        elif self.params['method'] == 'sac':

                            self.agent.sac_train(self.params)
                else:
                    if len(self.agent.replay_buffer) > self.agent.replay_buffer.trajectory_size:
                        self.agent.ppo_train()

                if step % 100 == 0 and step > 0:
                    hit, ndcg = run_evaluation(self.agent.actor, self.agent.state_repr, train_env.memory, self.params,
                                               valid_dataset)
                    hits.append(hit)
                    ndcgs.append(ndcg)
                    if np.mean(np.array([hit, ndcg]) - np.array([best_hit, best_ndcg])) > 0:
                        best_hit, best_ndcg = hit, ndcg
                        self.agent.save(self.params['logs_dir'], [self.params['name_actor'], self.params['name_critic'],
                                                                  self.params['name_state']])

                if step % 1000 == 0 and step > 0:
                    hit, ndcg = run_evaluation(self.agent.actor, self.agent.state_repr, train_env.memory, self.params,
                                               full_dataset)
                    print(hit, ndcg)
                    hits_all.append(hit)
                    ndcgs_all.append(ndcg)
                    if np.mean(np.array([hit, ndcg]) - np.array([best_hit_all, best_ndcg_all])) > 0:
                        best_hit_all, best_ndcg_all = hit, ndcg
                        self.agent.save(self.params['logs_dir'], [self.params['name_actor'], self.params['name_critic'],
                                                                  self.params['name_state']], best=True)
                step += 1

        with open(os.path.join(self.params['logs_dir'], 'memory.pickle'), 'wb') as f:
            pickle.dump(train_env.memory, f)

        return hits_all, ndcgs_all
    ####################################
    ####################################
