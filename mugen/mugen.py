from keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D, Reshape
from keras.models import Model
import numpy as np


class Mugen:
    def __init__(self, time_steps: int, pitches: int = 128, tracks: int = 1):
        self.model = None
        self.time_steps = time_steps
        self.tracks = tracks
        self.pitches = pitches

    @property
    def sample_dims(self):
        probability_vars = 1
        voice_vars = 128
        return voice_vars + probability_vars + self.pitches

    def build_model(self):
        input_series = Input(
            shape=(self.time_steps, self.pitches, self.tracks))

        x = input_series
        x = Conv2D(
            filters=64, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPool2D(pool_size=2)(x)
        x = Conv2D(
            filters=64, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPool2D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(self.pitches * self.tracks * 8, activation='relu')(x)
        x = Dense(self.pitches * self.tracks, activation='relu')(x)
        x = Reshape((self.pitches, self.tracks))(x)
        prediction = x

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
            epochs: int = 1):
        assert input_sequences.shape[1:] == (
            self.time_steps, self.pitches, self.tracks)
        self.model.fit(input_sequences, next_samples, epochs=epochs)

    def generate_sample_batch(self, input_sequences: np.array) -> np.array:
        assert input_sequences.shape[1:] == (
            self.time_steps, self.pitches, self.tracks)
        return self.model.predict(input_sequences)

    def generate_sample(self, input_sequence: np.array) -> np.array:
        assert input_sequence.shape == (
            self.time_steps, self.pitches, self.tracks)
        batch_shape = (1, self.time_steps, self.pitches, self.tracks)
        input_sequences = np.broadcast_to(input_sequence, batch_shape)
        return self.generate_sample_batch(input_sequences)
