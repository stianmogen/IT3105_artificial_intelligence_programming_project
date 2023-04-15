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
        self.load_path = load_path
        self.board_size = board_size
        if self.load_path is not None:
            self.model = self.load_saved_model(load_path)
        else:
            model_input = Input(shape=(self.board_size, self.board_size, 2))

            for i, (filters, kernel_size, activation) in enumerate(p.conv_layers):
                if i == 0:
                    x = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), padding='same',
                               activation=activation)(model_input)
                else:
                    x = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), padding='same',
                               activation=activation)(x)

            x = Flatten()(x)

            for units, activation in p.dense_layers:
                x = Dense(units, activation=activation)(x)

            actor = Dense(self.board_size * self.board_size, activation=p.actor_activation, name='actor_output')(x)
            critic = Dense(1, activation=p.critic_activation, name='critic_output')(x)

            model = Model(inputs=model_input, outputs=[actor, critic], name='anet')
            losses = {
                'actor_output': p.actor_loss,
                'critic_output': p.critic_loss,
            }
            lr_scheduler = ExponentialDecay(
                initial_learning_rate=p.learning_rate,
                decay_steps=p.lr_scheduler_decay_steps,
                decay_rate=p.lr_scheduler_decay_rate)

            optimizer = p.optimizer(learning_rate=lr_scheduler)

            loss_weights = p.loss_weights

            model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

            model.summary()
            self.model = model

    def predict(self, board, player):
        x = self.one_hot_encode(board, player)
        x = np.expand_dims(x, axis=0)
        _, critic = self.model(x)
        return critic.numpy()[0][0]

    def fit(self, samples, epochs):
        x = np.array([self.one_hot_encode(sample[0], 1) for sample in samples])
        actor_target = np.array([sample[1] for sample in samples], dtype=np.float32)
        critic_target = np.array([sample[2] for sample in samples], dtype=np.float32)

        y = {"actor_output": actor_target, "critic_output": critic_target}
        self.model.fit(x, y, verbose=1, batch_size=32, shuffle=True, epochs=epochs)

    def best_move(self, board, player):
        mask = np.where(board == 0, 1, 0)
        x = self.one_hot_encode(board, player)
        x = np.expand_dims(x, axis=0)

        distribution, _ = self.model(x)
        distribution = distribution.numpy()

        if player == 2:
            distribution = distribution.reshape((self.board_size, self.board_size)).T.flatten()

        distribution = np.multiply(mask, distribution)
        distribution = np.divide(distribution, np.sum(distribution))

        return np.argmax(distribution)

    def one_hot_encode(self, board, player):
        p1_board = np.where(board == 1, 1, 0).reshape(self.board_size, self.board_size)
        p2_board = np.where(board == 2, 1, 0).reshape(self.board_size, self.board_size)

        ohe = np.zeros(shape=(self.board_size, self.board_size, 2))
        for i in range(self.board_size):
            for j in range(self.board_size):
                if player == 1:
                    ohe[i, j] = [p1_board[i, j], p2_board[i, j]]
                else:
                    ohe[i, j] = [p2_board.T[i, j], p1_board.T[i, j]]

        return ohe

    def save_model(self, name):
        self.model.save(f"{name}.h5")
        print(f"Model {name}.h5 saved succesfully")

    def load_saved_model(self, filename):
        model = load_model(filename, compile=False)
        print(f"Model {filename} loaded succesfully")
        return model
