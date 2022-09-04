import numpy as np
import torch.utils.data as td
import torch
from project.infrastructure.envionverment import Env
from project.infrastructure.utils import to_np


class EvalDataset(td.Dataset):
    def __init__(self, positive_data, item_num, positive_mat, negative_samples=99):
        super(EvalDataset, self).__init__()
        self.pos_data, self.neg_data, self.users = None, None, None
        self.positive_data = np.array(positive_data)
        self.item_num = item_num
        self.positive_mat = positive_mat
        self.negative_samples = negative_samples

        self.reset()

    def reset(self):
        print("Resetting dataset")
        self.pos_data, self.neg_data = self.create_valid_data()
        self.users = self.pos_data.keys()

    def create_valid_data(self):
        positive_valid_data = {}
        negative_valid_data = {}
        for user, item in self.positive_data[:, :2]:
            if self.positive_mat[int(user), int(item)] == 1:
                positive_valid_data[user] = positive_valid_data.get(user, []) + [item]
            else:
                negative_valid_data[user] = negative_valid_data.get(user, []) + [item]
        return positive_valid_data, negative_valid_data

    def __getitem__(self, user):
        pos_items, neg_items = self.pos_data.get(user, []), self.neg_data.get(user, [])
        items = pos_items + neg_items
        output = {
            "items": items,
            "pos_items": pos_items,
            "neg_items": neg_items
        }
        return output


def run_evaluation(net, state_representation, training_env_memory, params, loader):
    precisions = []
    ndcgs = []
    test_env = Env(params['test_data'], params['item_num'], params['user_num'], params['N'], params['fill_users'])
    test_env.memory = training_env_memory.copy()
    hit_norm = params['K']
    ndcg_norm = sum([1 / np.log2(num + 1) for num in range(1, params['K'] + 1)])
    for user in loader.users:
        items = loader[user]
        precision = 0
        ndcg = 0
        user, memory = test_env.reset(int(user))
        all_items, pos_items = items['items'], items['pos_items']
        for num in range(1, params['K'] + 1):
            if params['method'] == 'sac':
                action_emb = net.select_action(state_representation(user, memory))
            else:
                action_emb = net(state_representation(user, memory))
            action = net.get_action(
                user,
                torch.tensor(memory),
                state_representation,
                action_emb,
                torch.tensor(
                    [item for item in all_items
                     if item not in test_env.viewed_items]
                ).long()
            )
            user, memory, reward = test_env.step(action)
            precision += int(action in pos_items)
            ndcg += int(action in pos_items) / np.log2(num + 1)

        ndcgs.append(ndcg/ ndcg_norm)
        precisions.append(precision / hit_norm)

    return np.mean(precisions), np.mean(ndcgs)
