import random

import numpy as np
import keras
from keras import Input, Model
from keras.layers import Dense, Embedding, Dropout
from keras.models import load_model
from keras.optimizers import Adam


class Anet2:
    def __init__(self, input_dim=17, output_dim=16, hidden_dims=(256, 512, 512, 128), dropout_rate=0.3, load_path=None):
        self.load_path = load_path
        if self.load_path is not None:
            self.model = self.load_saved_model(load_path)
        else:
            input_tensor = Input(shape=(input_dim,))
            x = Embedding(input_dim=3, output_dim=4, input_length=input_dim)(input_tensor)

            for hidden_dim in hidden_dims:
                x = Dense(hidden_dim, activation='relu')(x)
                x = Dropout(dropout_rate)(x)

            output = Dense(output_dim, activation="softmax")(x)
            self.model = Model(input_tensor, output)
            self.model.compile(optimizer=Adam(learning_rate=1e-3), loss="kl_divergence")
            self.model.summary()

    def predict(self, x):
        mask = np.where(x == 0, 1, 0)[:, 1:]
        output = self.model(x)
        masked = np.multiply(output, mask)
        return np.divide(masked, np.sum(masked))

    def fit(self, samples):
        x = np.array([sample[0] for sample in samples])
        y = np.array([sample[1] for sample in samples], dtype=np.float32)
        self.model.fit(x, y, epochs=10)

    def best_move(self, x):
        predictions = self.predict(x)
        return np.argmax(predictions)

    def save_model(self, name):
        self.model.save(f"{name}.h5")
        print(f"Model {name}.h5 saved succesfully")

    def load_saved_model(self, filename):
        model = load_model(filename, compile=False)
        print(f"Model {filename} loaded succesfully")
        return model