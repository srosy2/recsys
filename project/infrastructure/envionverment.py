import numpy as np
import torch
from .utils import to_np


class Env():
    def __init__(self, user_item_matrix, item_num, user_num, N, fill_users=True):
        self.matrix = user_item_matrix
        self.item_count = item_num
        self.memory = np.ones([user_num, N]) * item_num
        self.fill_users = fill_users
        # memory is initialized as [item_num] * N for each user
        # it is padding indexes in state_repr and will result in zero embeddings

    def reset(self, user_id):
        self.user_id = user_id
        self.viewed_items = []
        self.related_items = self.matrix[(self.matrix['user'] == user_id) & (self.matrix['rating'] > 0)]['item'
        ].to_list()
        self.num_rele = len(self.related_items)
        self.nonrelated_items = self.matrix[(self.matrix['user'] == user_id) & (self.matrix['rating'] <= 0)]['item'
        ].to_list()
        self.available_items = np.random.permutation(np.array(self.related_items + self.nonrelated_items))
        if self.fill_users:
            self.fill_memory()

        return torch.tensor([self.user_id]), torch.tensor(self.memory[[self.user_id], :])

    def fill_memory(self):
        unviewed_related_items = [item for item in self.related_items if item not in self.viewed_items]
        unviewed_related_items = list(np.random.permutation(unviewed_related_items))
        while self.item_count in self.memory[self.user_id] and unviewed_related_items:
            unviewed_related_item = unviewed_related_items.pop(0)
            if unviewed_related_item not in self.memory[self.user_id]:
                self.memory[self.user_id] = list(self.memory[self.user_id][1:]) + [unviewed_related_item]
                self.viewed_items.append(unviewed_related_item)

    def step(self, action, action_emb=None, buffer=None):
        initial_user = self.user_id
        initial_memory = self.memory[[initial_user], :]

        reward = float(self.matrix[(self.matrix['item'] == to_np(action)[0]) & (self.matrix['user'] == initial_user)][
                           'rating'].values[0])
        self.viewed_items.append(to_np(action)[0])
        if reward:
            if len(action) == 1:
                self.memory[self.user_id] = list(self.memory[self.user_id][1:]) + [action]
            else:
                self.memory[self.user_id] = list(self.memory[self.user_id][1:]) + [action[0]]

        if buffer is not None:
            experience = (np.array([initial_user]), np.array(initial_memory), to_np(action_emb)[0],
                          np.array([reward]), np.array([self.user_id]), self.memory[[self.user_id], :])
            buffer.store(experience)

        return torch.tensor([self.user_id]), torch.tensor(self.memory[[self.user_id], :]), reward