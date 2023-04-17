import random
import time

import numpy as np
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

    def predict(self, board, player):
        """
        Predicts the values of the game board based on the state and player
        :param board: hex game at current state
        :param player: player to make next move
        :return: board prediction
        """
        # encodes the board based on the player
        x = self.one_hot_encode(board, player)
        x = np.expand_dims(x, axis=0)
        # gets the critic prediction from the model and returns this
        _, critic = self.model(x)
        return critic.numpy()[0][0]

    def fit(self, samples, epochs):
        """
        trains the model based on the sample data
        :param samples: a sample set of training data
        :param epochs: how many epochs to train for
        :return:
        """
        # encodes the board in samples to be of player 1, to train on similar data
        x = np.array([self.one_hot_encode(sample[0], 1) for sample in samples])
        # actor and critic target defined
        actor_target = np.array([sample[1] for sample in samples], dtype=np.float32)
        critic_target = np.array([sample[2] for sample in samples], dtype=np.float32)

        # dictionary of actor and critic output targets
        y = {"actor_output": actor_target, "critic_output": critic_target}
        # train the model using the input and target tensors
        self.model.fit(x, y, verbose=1, batch_size=p.batch_size, shuffle=True, epochs=epochs)

    def best_move(self, board, player):
        """
        classifies the best move based on the actor output from a given state and player
        :param board: current state
        :param player: to play next move
        :return: best move
        """
        # masks each empty position on the board
        mask = np.where(board == 0, 1, 0)
        # encodes the board based on current player
        x = self.one_hot_encode(board, player)
        x = np.expand_dims(x, axis=0)

        # distribution is the actors output from the model
        distribution, _ = self.model(x)
        distribution = distribution.numpy().flatten()

        if player == 2:
            distribution = distribution.reshape((self.board_size, self.board_size)).T.flatten()

        # distribution is multiplied by mask and normalized
        distribution = np.multiply(mask, distribution)
        distribution = np.divide(distribution, np.sum(distribution))

        # returns the highest distribution
        return np.argmax(distribution)
        #weighted_random = random.choices(range(len(distribution)), weights=distribution, k=1)
        #return weighted_random[0]

    def one_hot_encode(self, board, player):
        """
        Method to encode the board based on the current player
        If player is two, the board needs to be transformed to account for players win conditions
        :param board:
        :param player:
        :return:
        """
        p1_board = (board == 1).astype(int).reshape(self.board_size, self.board_size)
        p2_board = (board == 2).astype(int).reshape(self.board_size, self.board_size)

        ohe = np.zeros(shape=(self.board_size, self.board_size, 2))
        if player == 1:
            ohe[:, :, 0] = p1_board
            ohe[:, :, 1] = p2_board
        else:
            ohe[:, :, 0] = p2_board.T
            ohe[:, :, 1] = p1_board.T

        return ohe

    def save_model(self, name):
        self.model.save(f"{name}.h5")
        print(f"Model {name}.h5 saved succesfully")

    def load_saved_model(self, filename):
        model = load_model(filename, compile=False)
        print(f"Model {filename} loaded succesfully")
        return model
