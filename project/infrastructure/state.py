import numpy as np
import torch.nn as nn
import torch
from torch import optim
import itertools


class StateReprModuleAve(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim, N, learning_rate, decay):
        super().__init__()
        self.user_embeddings = nn.Embedding(user_num, embedding_dim)
        self.item_embeddings = nn.Embedding(item_num + 1, embedding_dim, padding_idx=int(item_num))
        self.drr_ave = torch.nn.Conv1d(in_channels=N, out_channels=1, kernel_size=1)

        self.initialize()
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=decay
        )

    def initialize(self):
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        self.item_embeddings.weight.data[-1].zero_()
        nn.init.uniform_(self.drr_ave.weight)
        self.drr_ave.bias.data.zero_()

    def forward(self, user, memory):
        user_embedding = self.user_embeddings(user.long()).squeeze(1)

        item_embeddings = self.item_embeddings(memory.long()).squeeze(1)
        drr_ave = self.drr_ave(item_embeddings).squeeze(1)

        return torch.cat((user_embedding, user_embedding * drr_ave, drr_ave), 1)

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)


class StateReprModuleN(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim, N, learning_rate, decay):
        super().__init__()
        self.user_embeddings = nn.Embedding(user_num, embedding_dim)
        self.item_embeddings = nn.Embedding(item_num + 1, embedding_dim, padding_idx=int(item_num))
        self.weights = nn.Parameter(
            torch.randn(N, dtype=torch.float32, device='cpu')
        )

        self.initialize()
        self.optimizer = optim.Adam(
            itertools.chain(self.parameters()),
            lr=learning_rate,
            weight_decay=decay
        )

    def initialize(self):
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        self.item_embeddings.weight.data[-1].zero_()

    def forward(self, user, memory):
        item_embeddings = self.item_embeddings(memory.long()).squeeze(1)

        return item_embeddings.reshape(item_embeddings.shape[0], -1)

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)


class StateReprModuleP(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim, N, learning_rate, decay):
        super().__init__()
        self.user_embeddings = nn.Embedding(user_num, embedding_dim)
        self.item_embeddings = nn.Embedding(item_num + 1, embedding_dim, padding_idx=int(item_num))
        self.weights = nn.Parameter(
            torch.randn(N, dtype=torch.float32, device='cpu')
        )

        self.initialize()
        self.optimizer = optim.Adam(
            itertools.chain(self.parameters()),
            lr=learning_rate,
            weight_decay=decay
        )

    def initialize(self):
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        self.item_embeddings.weight.data[-1].zero_()

    def forward(self, user, memory):
        item_embeddings = self.item_embeddings(memory.long()).squeeze(1)
        item_num = item_embeddings.shape[1]
        item_vectors = torch.zeros((item_embeddings.shape[0], int(item_num * (item_num - 1) / 2), item_embeddings.shape[2]))
        for item in range(item_embeddings.shape[0]):
            iter_num = 0
            for item_1 in range(item_embeddings.shape[1]):
                for item_2 in range(item_1 + 1, item_embeddings.shape[1]):
                    item_vectors[item, iter_num, :] = self.weights[item_1] * item_embeddings[item, item_1,
                                                                        :] * self.weights[item_2] * item_embeddings[
                                                                                                    item, item_2, :]
                    iter_num += 1

        return torch.cat((item_embeddings, item_vectors), 1).reshape(item_embeddings.shape[0], -1)

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)


class StateReprModuleU(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim, N, learning_rate, decay):
        super().__init__()
        self.user_embeddings = nn.Embedding(user_num, embedding_dim)
        self.item_embeddings = nn.Embedding(item_num + 1, embedding_dim, padding_idx=int(item_num))
        self.weights = nn.Parameter(
            torch.rand(N, dtype=torch.float32, device='cpu')
        )

        self.initialize()
        self.optimizer = optim.Adam(
            itertools.chain(self.parameters()),
            lr=learning_rate,
            weight_decay=decay
        )

    def initialize(self):
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        self.item_embeddings.weight.data[-1].zero_()

    def forward(self, user, memory):
        user_embedding = self.user_embeddings(user.long()).squeeze(1)

        item_embeddings = self.item_embeddings(memory.long()).squeeze(1)
        item_num = item_embeddings.shape[1]
        user_item_vectors = torch.zeros_like(item_embeddings)

        for user in range(user_embedding.shape[0]):
            for item in range(item_num):
                user_item_vectors[user, item, :] = user_embedding[user, :] * self.weights[item] * item_embeddings[user,
                                                                                                  item, :]

        item_vectors = torch.zeros(
            (item_embeddings.shape[0], int(item_num * (item_num - 1) / 2), item_embeddings.shape[2]))
        for item in range(item_embeddings.shape[0]):
            iter_num = 0
            for item_1 in range(item_num):
                for item_2 in range(item_1 + 1, item_num):
                    item_vectors[item, iter_num, :] = self.weights[item_1] * item_embeddings[item, item_1,
                                                                             :] * self.weights[
                                                          item_2] * item_embeddings[
                                                                    item, item_2, :]
                    iter_num += 1

        return torch.cat((user_item_vectors, item_vectors), 1).reshape(user_embedding.shape[0], -1)

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
