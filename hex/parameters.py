class Parameters:
    def __init__(self):
        # BOARD
        self.board_size = 5

        # TRAINING
        self.num_games = 200
        self.batch_size = 128
        self.epochs = 5
        self.epsilon = 1.2
        self.sigma = 1.2
        self.epsilon_decay = 0.999
        self.sigma_decay = 0.998
        self.save_interval = 20
        self.rollouts = 200
        self.exploration = 1
        self.buffer_size = 512
        self.model_name = "demo5"

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
        self.model_files = ['demo5-180.h5', 'demo5-1.h5']
        self.t_num_games = 30
        self.t_epsilon = 0.5
        self.t_sigma = 0.5
        self.t_rollouts = 10
        self.t_num_tournaments = 5

