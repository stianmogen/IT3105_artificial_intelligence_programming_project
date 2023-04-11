class Parameters:
    def __init__(self):
        # BOARD
        self.board_size = 7

        # TRAINING
        self.num_games = 1000
        self.batch_size = 128
        self.epochs = 5
        self.epsilon = 1.
        self.sigma = 1.
        self.epsilon_decay = 0.998
        self.sigma_decay = 0.999
        self.save_interval = 50
        self.rollouts = 200
        self.exploration = 1
        self.buffer_size = 512

        # MODEL
        self.learning_rate = 1e-3
        self.actor_activation = 'softmax'
        self.actor_loss = 'categorical_crossentropy'
        self.critic_activation = 'sigmoid'
        self.critic_loss = 'mse'

        # TOURNAMENT
        self.visualize_board = False
        self.against_mcts = False
        self.print_winner = False
        self.model_files = ['500rolls-game250.h5', '50rolls-game250.h5', 'game231.h5', '500rolls-game125.h5']
        self.t_num_games = 20
        self.t_epsilon = 0.5
        self.t_sigma = 0.5
        self.t_rollouts = 200

