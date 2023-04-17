import math

from keras.optimizers import Adagrad, SGD, Adam
from keras.optimizers.optimizer_v2.rmsprop import RMSProp

"""
Parameter class to easily change parameters in one common document
"""
class Parameters:
    def __init__(self):
        # BOARD
        self.board_size = 7

        optimizers = {'adagrad': Adagrad, 'sgd': SGD, 'rmsprop': RMSProp, 'adam': Adam}

        # TRAINING
        self.num_games = 1000
        self.sample_size = 512
        self.batch_size = 32
        self.epochs = 1
        self.epsilon = 1.1
        self.sigma = 1.1
        self.epsilon_decay = 0.001
        self.sigma_decay = 0.0005
        self.save_interval = 100
        self.rollouts = 500
        self.exploration = 1
        self.buffer_size = 2048
        self.model_name = "discovery"
        self.plot_dist = False
        self.show_board = False
        self.show_test_acc = True
        self.optimizer = optimizers['adam']
        self.loss_weights = {'actor_output': 1.0, 'critic_output': self.board_size * self.board_size}
        self.lr = 1e-3
        end_lr = 1e-5

        self.lr_scheduler_decay_rate = 0.9

        # Calculating how often we need to decay learning rate
        # start_lr * decay_rate^(x * steps_per_training) = end_lr
        steps_per_training = math.ceil(self.sample_size / self.batch_size)
        x = math.log(end_lr / self.lr) / math.log(self.lr_scheduler_decay_rate) / steps_per_training
        self.lr_scheduler_decay_steps = self.num_games / x

        self.actor_loss = 'kl_divergence'
        self.critic_loss = 'mse'

        # MODEL
        self.conv_layers = [[32, 3, 'relu'],
                            [64, 2, 'relu'],
                            [64, 2, 'relu'],
                            ]
        self.dense_layers = [[32, 'relu']]

        self.actor_activation = 'softmax'
        self.critic_activation = 'sigmoid'

        # TOURNAMENT
        self.visualize_board = False
        self.against_mcts = False
        self.print_winner = True
        self.model_files = ['discovery-300.h5', 'game231.h5']
        self.t_num_games = 30
        self.t_epsilon = 0.
        self.t_sigma = 1
        self.t_rollouts = 100
        self.t_num_tournaments = 1

