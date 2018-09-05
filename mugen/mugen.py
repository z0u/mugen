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

        x = input_series
        x = Conv1D(filters=64, kernel_size=6, activation='relu')(x)
        x = MaxPool1D(pool_size=2)(x)
        x = Flatten()(x)
        prediction = Dense(self.channels)(x)

        model = Model(input_series, prediction)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['accuracy'])
        self.model = model

    def train(
            self,
            input_sequences: np.array,
            next_samples: np.array,
            epochs: int=1):
        assert input_sequences.shape[1] == self.seq_length
        assert input_sequences.shape[2] == self.channels
        self.model.fit(input_sequences, next_samples, epochs=epochs)

    def generate_sample_batch(self, input_sequences: np.array) -> np.array:
        assert input_sequences.shape[1] == self.seq_length
        assert input_sequences.shape[2] == self.channels
        return self.model.predict(input_sequences).round().astype(np.int)

    def generate_sample(self, input_sequence: np.array) -> np.array:
        assert input_sequence.shape == (self.seq_length, self.channels)
        batch_shape = (1, self.seq_length, self.channels)
        input_sequences = np.broadcast_to(input_sequence, batch_shape)
        return self.generate_next_batch(input_sequences)
