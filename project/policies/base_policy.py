import abc
import numpy as np


class BasePolicy(object, metaclass=abc.ABCMeta):
    def get_action(self, user, memory, state_repr,
                   action_emb,
                   item_num,
                   return_scores) -> np.ndarray:
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError
