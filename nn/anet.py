import random

import tensorflow as tf
import numpy as np
import keras
from keras import Input, Model
from keras.layers import Dense, Embedding, Dropout, Flatten, Multiply, Activation
from keras.models import load_model
from keras.optimizers import Adam



class Anet2:
    def __init__(self, input_dim=17, output_dim=16, hidden_dims=(128, 256, 364), dropout_rate=0.1, load_path=None):
        self.load_path = load_path
        if self.load_path is not None:
            self.model = self.load_saved_model(load_path)
        else:
            input_tensor = Input(shape=(input_dim,))

            actor = self.create_actor_branch(input_tensor, output_dim)
            critic = self.create_critic_branch(input_tensor)

            model = keras.Model(
                inputs=input_tensor, outputs=[actor, critic], name="anet"
            )
            losses = {
                "actor_output": "kl_divergence",
                "critic_output": "mse",
            }
            loss_weights = {"actor_output": 1.0, "critic_output": 1.0}
            model.compile(
                optimizer=Adam(learning_rate=1e-2), loss=losses, loss_weights=loss_weights
            )
            model.summary()
            self.model = model
            

    def create_actor_branch(self, x, ouput_dim):
        # create a mask of available tiles
        mask = tf.cast(tf.math.equal(x[:, 1:], 0), tf.float32)

        hidden_dims = [64, 128, 256, 512, 512]
        for hidden_dim in hidden_dims:
            x = Dense(hidden_dim, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros')(x)

        output = Dense(ouput_dim)(x)

        # set the zero values in output to a very large negative number
        large_negative = -1e9
        output_masked = Multiply()([output, mask]) + (1 - mask) * large_negative

        # apply softmax to the modified output
        output_softmax = Activation(activation="softmax", name="actor_output")(output_masked)
        return output_softmax

    def create_critic_branch(self, x):
        hidden_dims = [64, 128, 256, 512, 512]
        for hidden_dim in hidden_dims:
                x = Dense(hidden_dim, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros')(x)
        x = Dense(1)(x)
        x = Activation(activation="sigmoid", name="critic_output")(x)
        return x

    def predict(self, x):
        _, critic = self.model(x)
        return critic.numpy()[0][0]

    def fit(self, samples, epochs):
        x = np.array([sample[0] for sample in samples])
        actor_target = np.array([sample[1] for sample in samples], dtype=np.float32)
        critic_target = np.array([sample[2] for sample in samples], dtype=np.float32)
        y = {"actor_output": actor_target,
             "critic_output": critic_target}

        self.model.fit(x, y, verbose=1, batch_size=32, epochs=epochs)


    def best_move(self, x):
        distribution, _ = self.model(x)
        return np.argmax(distribution)

    def save_model(self, name):
        self.model.save(f"{name}.h5")
        print(f"Model {name}.h5 saved succesfully")

    def load_saved_model(self, filename):
        model = load_model(filename, compile=False)
        print(f"Model {filename} loaded succesfully")
        return model