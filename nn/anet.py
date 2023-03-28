import random

import tensorflow as tf
import numpy as np
import keras
from keras import Input, Model
from keras.layers import Dense, Embedding, Dropout, Flatten, Multiply
from keras.models import load_model
from keras.optimizers import Adam



class Anet2:
    def __init__(self, input_dim=17, output_dim=16, hidden_dims=(128, 256, 364), dropout_rate=0.1, load_path=None):
        self.load_path = load_path
        if self.load_path is not None:
            self.model = self.load_saved_model(load_path)
        else:
            input_tensor = Input(shape=(input_dim,))

            for i in range(len(hidden_dims)):
                if i == 0:
                     x = Dense(hidden_dims[i], activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros')(
                    input_tensor)
                else:
                    x = Dense(hidden_dims[i], activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros')(
                    x)
                x = Dropout(rate=dropout_rate)(x)

            x = Flatten()(x)
            output = Dense(output_dim)(x)

            # create a mask of available tiles
            mask = tf.cast(tf.math.equal(input_tensor[:, 1:], 0), tf.float32)

            # set the zero values in output to a very large negative number
            large_negative = -1e9
            output_masked = Multiply()([output, mask]) + (1 - mask) * large_negative

            # apply softmax to the modified output
            output_softmax = tf.keras.activations.softmax(output_masked)

            self.model = tf.keras.models.Model(inputs=input_tensor, outputs=output_softmax)
            self.model.compile(optimizer=Adam(learning_rate=1e-3), loss="kl_divergence")
            self.model.summary()

    def actor_branch(self, x):
        pass

    def critic_branch(self, x):
        pass

    def predict(self, x):
        return self.model(x)

    def fit(self, samples, epochs):
        x = np.array([sample[0] for sample in samples])
        y = np.array([sample[1] for sample in samples], dtype=np.float32)
        self.model.fit(x, y, verbose=1, epochs=epochs)

    def best_move(self, x):
        predictions = self.predict(x)
        return np.argmax(predictions)

    def eval_state(self, x):
        pass

    def save_model(self, name):
        self.model.save(f"{name}.h5")
        print(f"Model {name}.h5 saved succesfully")

    def load_saved_model(self, filename):
        model = load_model(filename, compile=False)
        print(f"Model {filename} loaded succesfully")
        return model