import math

from keras.optimizers import Adagrad, SGD, Adam
from keras.optimizers.optimizer_v2.rmsprop import RMSProp

"""
Parameter class to easily change parameters in one common document
"""
class Parameters:
    def __init__(self):
        # BOARD
        self.board_size = 4

        optimizers = {'adagrad': Adagrad, 'sgd': SGD, 'rmsprop': RMSProp, 'adam': Adam}

        # TRAINING
        self.num_games = 100
        self.sample_size = 512
        self.batch_size = 64
        self.epochs = 1
        self.epsilon = 1.0
        self.sigma = 1.0
        self.alpha = 1.2
        self.epsilon_decay = 0.002
        self.sigma_decay = 0.0007
        self.alpha_decay = 0.0007
        self.save_interval = 25
        self.rollouts = 100
        self.exploration = 1
        self.buffer_size = 2048
        self.model_name = ""
        self.plot_dist = False
        self.show_board = False
        self.show_test_acc = True
        self.optimizer = optimizers['adam']
        self.loss_weights = {'actor_output': 1.0, 'critic_output': self.board_size * self.board_size / 2}
        self.lr = 5e-4
        end_lr = 1e-6

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
                            [48, 3, 'relu'],
                            [64, 2, 'relu'],
                            [64, 2, 'relu'],
                            ]
        self.dense_layers = [[128, 'relu'],
                             [64, 'relu']]

        self.actor_activation = 'softmax'
        self.critic_activation = 'sigmoid'

        # TOURNAMENT
        self.visualize_board = False
        self.against_mcts = False
        self.print_winner = True
        self.model_files = ['800.h5', '0.h5', '200.h5', '400.h5']
        self.t_num_games = 30
        self.t_epsilon = 0
        self.t_sigma = 1
        self.t_alpha = 0.
        self.t_rollouts = 50
        self.t_num_tournaments = 1
        self.t_exploration = 1

