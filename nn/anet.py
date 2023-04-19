import random
import time

import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense, Flatten, Conv2D
from keras.models import load_model
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay

# from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam

from hex.parameters import Parameters

p = Parameters()


class Anet2:
    def __init__(self, board_size, load_path=None):
        """
        Initializing the network with the size of the Hexboard,
        and an optional path for loading pre trained model
        Convolutional network with actor and critic output
        Defines the model and compiles it with the given parameters defined in file
        The model parameters may be tuned in parameters.py
        :param board_size: size of hex board
        :param load_path: load model if not None
        """
        self.load_path = load_path
        self.board_size = board_size
        if self.load_path is not None:
            self.model = self.load_saved_model(load_path)
        else:
            # defining the input layer based on the board_size
            model_input = Input(shape=(self.board_size, self.board_size, 2))

            for i, (filters, kernel_size, activation) in enumerate(p.conv_layers):
                if i == 0:
                    # if this is the first convolutional layer, this is connected to model_input
                    x = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), padding='same',
                               activation=activation)(model_input)
                else:
                    # if not connects to previous layer
                    x = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), padding='same',
                               activation=activation)(x)

            # flatten the output of the conv layers
            x = Flatten()(x)

            # adds dense layer
            for units, activation in p.dense_layers:
                x = Dense(units, activation=activation)(x)

            # actor and critic output as dense layer with different activation functions defined in parameters file
            # the actor will give output of the NxN board size
            # the critic will give a single numerical output
            actor = Dense(self.board_size * self.board_size, activation=p.actor_activation, name='actor_output')(x)
            critic = Dense(1, activation=p.critic_activation, name='critic_output')(x)

            # define model with input, and the actor and critic outputs
            model = Model(inputs=model_input, outputs=[actor, critic], name='anet')

            # loss functions are defined in paremters file
            losses = {
                'actor_output': p.actor_loss,
                'critic_output': p.critic_loss,
            }
            # learning rate scheduler. higher learning rate at start which decays at a step rate
            lr_scheduler = ExponentialDecay(
                initial_learning_rate=p.lr,
                decay_steps=p.lr_scheduler_decay_steps,
                decay_rate=p.lr_scheduler_decay_rate)

            optimizer = p.optimizer(learning_rate=lr_scheduler)

            loss_weights = p.loss_weights

            # model is compiled with the learning rate scheduler, and the loss function and corresponding weights
            model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

            model.summary()
            self.model = model

    def eval_state(self, board, player):
        """
        This method uses the graph execution of the predict method and converts the output back to usable format
        :param board: 1D array representing a board
        :param player: the current player
        :return: float (0-1) representing evaluation of current board state
        """
        return self.predict(board, player).numpy()


    def fit(self, samples):
        """
        This method is used for training the model.
        Uses the samples from the replay buffer to generate input and target values.
        :param samples: index 0 is board, index 1 is a distribution in the same shape as board, index 2 is an int
        """
        x = np.array([self.one_hot_encode(sample[0], 1) for sample in samples])
        # actor and critic target defined
        actor_target = np.array([sample[1] for sample in samples], dtype=np.float32)
        critic_target = np.array([sample[2] for sample in samples], dtype=np.float32)

        # dictionary of actor and critic output targets
        y = {"actor_output": actor_target, "critic_output": critic_target}
        self.model.fit(x, y, verbose=1, batch_size=p.batch_size, shuffle=True, epochs=p.epochs)

    def best_move(self, board, player, alpha):
        """
        This method uses the graph execution of the policy method and converts the output to numpy
        so that it can be used normally.
        :param board: 1D array representing a board
        :param player: the current player
        :param alpha: value used in the policy to decide which value to return
        :return: value representing index of the best move
        """
        return self.policy(board, player, alpha).numpy()

    @tf.function
    def transpose(self, data):
        """
        This method transposes a 1D array representing a board-state. Input is automatically transferred to a tf.Tensor
        and uses graph execution for optimization.
        :param data: 1D board
        :return: transposed 1D board where rows are columns and columns are rows in the 2D version
        """
        data = tf.reshape(data, (self.board_size, self.board_size))
        data = tf.transpose(data)
        return tf.reshape(data, [-1])

    @tf.function
    def predict(self, board, player):
        """
        This method one-hot encodes the board-state and predicts how good the current position is. Input is
        automatically transferred to a tf.Tensor
        and uses graph execution for optimization.
        :param board: 1D array representing board
        :param player: the player to play next
        :return: evaluation of the current board state (range 0-1)
        """
        x = self.one_hot_encode(board, player)
        x = tf.expand_dims(x, axis=0)
        _, critic = self.model(x)
        return critic[0][0]

    @tf.function
    def policy(self, board, player, alpha):
        """
        This method calculates the probability distribution of the current available moves.
        As we are always calculating the board from whites perspective we transpose the final results
        since the input into the model was also transposed. Input is automatically transferred to a tf.Tensor
        and uses graph execution for optimization.
        :param board: 1D array representing board
        :param player: 1 or 2 representing the current player
        :param alpha: variable deciding how likely the model will return best or weighted random move
        :return: returns either the best move or a random move weighted by the distribution
        """
        mask = tf.where(board == 0, 1., 0.)
        x = self.one_hot_encode(board, player)
        x = tf.expand_dims(x, axis=0)
        distribution, _ = self.model(x)
        distribution = tf.reshape(distribution, [-1])

        if player == 2:
            distribution = self.transpose(distribution)

        # Masking invalid board states prevents the actor from returning illegal moves
        distribution = tf.multiply(mask, distribution)

        # Re-normalizing the data
        distribution = tf.divide(distribution, tf.reduce_sum(distribution))

        if random.random() > alpha:
            # Selecting the move with the highest probability
            return tf.math.argmax(distribution)
        else:
            # Selecting move randomly, but weighted by the distribution
            return tf.random.categorical(tf.expand_dims(tf.math.log(distribution, 0), 0), 1)[0][0]

    @tf.function
    def one_hot_encode(self, board, player):
        """
        This method one hot encodes the board state. It also transposes the board and swaps the players
        if the current player is 2. This ensures that input is on the same format regardless of the current player
        and the model only need to learn how to win in one direction.
        :param board: 1D array representing board
        :param player: the current player
        :return: one how encoded preprocessed board
        """
        p1_board = tf.reshape(board == 1, (self.board_size, self.board_size))
        p2_board = tf.reshape(board == 2, (self.board_size, self.board_size))

        if player == 1:
            ohe = tf.concat([tf.expand_dims(p1_board, -1), tf.expand_dims(p2_board, -1)], axis=-1)
        else:
            ohe = tf.concat([tf.expand_dims(tf.transpose(p2_board), -1), tf.expand_dims(tf.transpose(p1_board), -1)],
                            axis=-1)

        return ohe

    def save_model(self, name):
        self.model.save(f"{name}.h5")
        print(f"Model {name}.h5 saved succesfully")

    def load_saved_model(self, filename):
        model = load_model(filename, compile=False)
        print(f"Model {filename} loaded succesfully")
        return model
