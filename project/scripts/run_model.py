import os
import time

from project.infrastructure.rl_trainer import RL_Trainer
from project.agents.drr_agent import DRRAgent


class DRR_Trainer(object):

    def __init__(self, params):
        #######################
        ## AGENT PARAMS
        #######################

        self.params = params
        self.params['agent_class'] = DRRAgent  ## HW1: you will modify this

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params)  ## HW1: you will modify this

        #######################
        ## LOAD EXPERT POLICY
        #######################

    def run_training_loop(self):
        return self.rl_trainer.run_training_loop()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rating', type=str, default='ml-1m.train.rating')  # название файла из которого берем данные
    parser.add_argument('--batch_size', type=int, default=512)  # размер батча на котором тренируем данные из реплей
    # баффера
    parser.add_argument('--embedding_dim', type=int, default=8)  # размер эмбедингов для размера стейта и действия
    parser.add_argument('--hidden_dim', type=int, default=16)  # размер скрытых эмбедингов в нейронной сети
    parser.add_argument('--N', type=int, default=5)  # n в множестве Ht={i1, in}, где it - это какой-то айтем
    parser.add_argument('--value_lr', type=float, default=1e-5)  # скорость оптимизации вэлью функции
    parser.add_argument('--value_decay', type=float, default=1e-4)  # L2 штраф вэлью функции
    parser.add_argument('--policy_lr', type=float, default=1e-5)  # скорость оптимизации полиси функции
    parser.add_argument('--policy_decay', type=float, default=1e-6)  # L2 штраф полиси функции
    parser.add_argument('--state_repr_lr', type=float, default=1e-5)  # скорость оптимизации стейта функции
    parser.add_argument('--state_repr_decay', type=float, default=1e-3)  # L2 штраф стейта функции
    parser.add_argument('--gamma', type=float, default=0.8)  # на сколько сильно снижаем будущую награду
    parser.add_argument('--soft_tau', type=float, default=1e-3)  # на сколько сильно оптимизируем таргет функцию
    parser.add_argument('--n_layers', type=int, default=2)  # количество слоев нейронной сети
    parser.add_argument('--seed', type=int, default=1)  # seed
    parser.add_argument('--state', type=str, default='drr_ave')  # какой из методов стейта будем использовать
    parser.add_argument('--train_emb', type=bool, default=True)  # будем ли тренировать эмбединги стейта
    parser.add_argument('--pretrain_emb', type=bool, default=False)  # будут ли эмбединги стейта предтренированные на PMF
    parser.add_argument('--noise', type=str, default='nor_dec_noise')  # какое добавление шума будет использоваться
    parser.add_argument('--method', type=str, default='ddpg')  # метод обучения
    parser.add_argument('--max_pos_item_user', type=int, default=35)  # максимальное позитивное значения
    # для юзера
    parser.add_argument('--min_pos_item_user', type=str, default=25)  # минимальное значения позитивных значений для
    # юзера должен быть на N больше К и длины эпизода если есть заполнение юзеров, иначе должен быть просто больше К и
    # длины эпиода
    parser.add_argument('--fill_users', type=bool, default=True)  # заполнить начальные значения айтемов
    parser.add_argument('--K', type=int, default=10)  # К для precision@K and NDCG@K
    parser.add_argument('--episode_length', type=int, default=20)  # длина взаимодействия с юзером
    parser.add_argument('--name_actor', type=str, default='actor_net')  # название для сохраниения актора
    parser.add_argument('--name_critic', type=str, default='critic_net')  # название для сохраниения критика
    parser.add_argument('--name_state', type=str, default='state_repr')  # название для сохраниения стейта

    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)

    ## directory for loading and saving
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
    logs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../logs')

    if not (os.path.exists(logs_path)):
        os.makedirs(logs_path)

    params['data_dir'] = data_path
    params['logs_dir'] = logs_path

    ###################
    ### RUN TRAINING
    ###################

    trainer = DRR_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
