import abc


class BaseValue(object, metaclass=abc.ABCMeta):

    def save(self, filepath: str):
        raise NotImplementedError