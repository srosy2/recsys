import numpy as np
from collections import defaultdict
import os
import pandas as pd
import scipy.sparse as sp
import torch
import math


def prepare_to_numpy(actions):
    if isinstance(actions, np.ndarray):
        return actions
    else:
        return actions.detach().numpy()


def to_np(tensor):
    return tensor.detach().cpu().numpy()


def preprocess_data(data_dir, train_rating, max_user, min_user):
    data = pd.read_csv(os.path.join(data_dir, train_rating),
                       sep='\t', header=None, names=['user', 'item', 'rating'],
                       usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int8})
    user_num = data['user'].max() + 1
    item_num = data['item'].max() + 1
    # adopted_data = data[data['rating'] > 3][['user', 'item']]
    data['rating'] = (data['rating'] - 3) / 2

    train_data = data.sample(frac=0.8, random_state=16)
    test_data = data.drop(train_data.index)
    train_data = train_data

    train_mat = defaultdict(int)
    test_mat = defaultdict(int)
    for user, item, rating in zip(train_data['user'], train_data['item'], train_data['rating']):
        train_mat[user, item] = 1 if rating > 0 else 0
    for user, item, rating in zip(test_data['user'], test_data['item'], test_data['rating']):
        test_mat[user, item] = 1 if rating > 0 else 0
    train_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    dict.update(train_matrix, train_mat)
    test_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    dict.update(test_matrix, test_mat)

    positive_sum_matrix = (train_matrix > 0).sum(1)
    appropriate_users = np.arange(user_num).reshape(-1, 1)[(positive_sum_matrix > min_user) & (
            positive_sum_matrix < max_user)]

    test_positive_sum_matrix = (test_matrix > 0).sum(1)
    test_appropriate_users = np.arange(user_num).reshape(-1, 1)[(test_positive_sum_matrix > min_user) & (
            test_positive_sum_matrix < max_user)]
    test_data = test_data[test_data['user'].isin(test_appropriate_users)]

    return (data, train_data, train_matrix, test_data, test_matrix,
            user_num, item_num, appropriate_users)


def calc_adv_ref(rewards, net_crt, states_v, gamma=0.99, gae_lambda=0.95, device="cpu"):
    """
    By trajectory calculate advantage and 1-step ref value
    :param device:
    :param gae_lambda:
    :param gamma:
    :param rewards: trajectory list
    :param net_crt: critic network
    :param states_v: states tensor
    :return: tuple with advantage numpy array and reference values
    """
    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    # generalized advantage estimator: smoothed version of the advantage
    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, reward in zip(reversed(values[:-1]),
                                     reversed(values[1:]),
                                     reversed(rewards[:-1])):
        delta = reward + gamma * next_val - val
        last_gae = delta + gamma * gae_lambda * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(list(reversed(result_adv)))
    ref_v = torch.FloatTensor(list(reversed(result_ref)))
    return adv_v.to(device), ref_v.to(device)


def calc_logprob(mu_v, logstd_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2
