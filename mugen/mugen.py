from keras.layers import Conv1D, Dense, Flatten, Input, MaxPool1D
from keras.models import Model
import numpy as np


class Mugen:
    def __init__(self, seq_length: int, channels: int):
        self.model = None
        self.seq_length = seq_length
        self.channels = channels

    def build_model(self):
        input_series = Input(shape=(self.seq_length, self.channels,))

        x = Conv1D(filters=64, kernel_size=6, activation='relu')(input_series)
        x = MaxPool1D(pool_size=2)(x)
        x = Flatten()(x)
        prediction = Dense(self.channels)(x)

        model = Model(input_series, prediction)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['accuracy'])
        self.model = model

    def train(self, input_sequences, next_notes, epochs=1):
        assert input_sequences.shape[1] == self.seq_length
        assert input_sequences.shape[2] == self.channels
        self.model.fit(input_sequences, next_notes, epochs=epochs)

    def predict(self, input_sequences):
        assert input_sequences.shape[1] == self.seq_length
        assert input_sequences.shape[2] == self.channels
        return self.model.predict(input_sequences).round().astype(np.int)
