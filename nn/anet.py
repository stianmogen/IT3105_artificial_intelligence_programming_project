import random

import tensorflow as tf
import numpy as np
import keras
from keras import Input, Model
from keras.layers import Dense, Embedding, Dropout, Flatten, Multiply, Activation, Conv2D, concatenate, Reshape, \
    RepeatVector, MaxPooling2D, Lambda
from keras.models import load_model
from keras.optimizers import Adam


class Anet2:
    def __init__(self, board_size, load_path=None):
        self.load_path = load_path
        if self.load_path is not None:
            self.model = self.load_saved_model(load_path)
        else:
            board_input = Input(shape=(board_size * board_size), name='board_input', dtype=np.uint8)
            board_oh = Lambda(lambda x: tf.one_hot(x, depth=3))(board_input)
            player_input = Input(shape=(1,), name='player_input')

            reshaped_board_input = Reshape((board_size, board_size, 3))(board_oh)
            repeated_player_input = RepeatVector(board_size * board_size)(player_input)
            repeated_player_input = Reshape((board_size, board_size, 1))(repeated_player_input)

            x = concatenate([reshaped_board_input, repeated_player_input])

            x = Reshape((board_size, board_size, 3 + 1))(x)


            # Add convolutional layers here
            x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Flatten()(x)

            # Add dense layers here
            x = Dense(256, activation='relu')(x)
            x = Dense(64, activation='relu')(x)

            actor = Dense(board_size * board_size, activation='softmax', name='actor_output')(x)
            critic = Dense(1, activation='sigmoid', name='critic_output')(x)

            model = Model(inputs=[board_input, player_input], outputs=[actor, critic], name='anet')
            losses = {
                'actor_output': 'kl_divergence',
                'critic_output': 'mse',
            }
            loss_weights = {'actor_output': 1.0, 'critic_output': 1.0}
            model.compile(optimizer=Adam(learning_rate=1e-3), loss=losses, loss_weights=loss_weights)
            model.summary()
            self.model = model

    def predict(self, board, player):
        x = {"board_input": np.array([board], dtype=np.uint8), "player_input": np.array([player == 2], dtype=np.uint8)}
        _, critic = self.model(x)
        return critic.numpy()[0][0]

    def fit(self, samples, epochs):
        boards = np.array([sample[0] for sample in samples], dtype=np.uint8)
        players = np.array([sample[1] == 2 for sample in samples], dtype=np.uint8)
        x = {"board_input": boards, "player_input": players}

        actor_target = np.array([sample[2] for sample in samples], dtype=np.float32)
        critic_target = np.array([sample[3] for sample in samples], dtype=np.float32)
        y = {"actor_output": actor_target, "critic_output": critic_target}

        self.model.fit(x, y, verbose=1, batch_size=64, shuffle=True, epochs=epochs)

    def best_move(self, board, player):
        mask = np.where(board == 0, 1, 0)
        x = {"board_input": np.array([board], dtype=np.uint8), "player_input": np.array([player], np.uint8)}
        distribution, _ = self.model(x)
        distribution = np.multiply(mask, distribution)
        distribution = np.divide(distribution, np.sum(distribution))

        return np.argmax(distribution)

    def save_model(self, name):
        self.model.save(f"{name}.h5")
        print(f"Model {name}.h5 saved succesfully")

    def load_saved_model(self, filename):
        model = load_model(filename, compile=False)
        print(f"Model {filename} loaded succesfully")
        return model
