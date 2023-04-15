from keras.optimizers import Adagrad, SGD, Adam
from keras.optimizers.optimizer_v2.rmsprop import RMSProp

"""
Parameter class to easily change parameters in one common document
"""
class Parameters:
    def __init__(self):
        # BOARD
        self.board_size = 5

        optimizers = {'adagrad': Adagrad, 'sgd': SGD, 'rmsprop': RMSProp, 'adam': Adam}

        # TRAINING
        self.num_games = 250
        self.batch_size = 128
        self.epochs = 5
        self.epsilon = 1
        self.sigma = 1.2
        self.epsilon_decay = 0.998
        self.sigma_decay = 0.999
        self.save_interval = 25
        self.rollouts = 300
        self.exploration = 1
        self.buffer_size = 512
        self.model_name = "divergence"
        self.plot_dist = False
        self.show_board = False
        self.optimizer = optimizers['adam']
        self.loss_weights = {'actor_output': 1.0, 'critic_output': 1.0}
        self.lr = 1e-2
        self.lr_scheduler_decay_rate = 0.9
        self.lr_scheduler_decay_steps = 200
        self.actor_loss = 'kl_divergence'
        self.critic_loss = 'hinge'

        # MODEL
        self.conv_layers = [[32, 3, 'relu'],
                            [64, 2, 'relu'],
                            [64, 2, 'relu']]
        self.dense_layers = [[256, 'relu'],
                             [128, 'relu']]

        self.actor_activation = 'softmax'
        self.critic_activation = 'sigmoid'

        # TOURNAMENT
        self.visualize_board = True
        self.against_mcts = False
        self.print_winner = True
        self.model_files = ['demo5-200.h5', 'divergence-200.h5']
        self.t_num_games = 30
        self.t_epsilon = 0.6
        self.t_sigma = 0.5
        self.t_rollouts = 300
        self.t_num_tournaments = 1

