from keras.layers import (
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    ConvLSTM2D,
    # Dense,
    # Flatten,
    Input,
    Lambda,
    # LSTM,
    # MaxPool2D,
    Reshape,
)
from keras.models import Model
import numpy as np


class Mugen:
    def __init__(self, time_steps: int, pitches: int = 128, tracks: int = 1):
        self.model = None
        self.time_steps = time_steps
        self.tracks = tracks
        self.pitches = pitches

    def build_model(self):
        kernel_size = 2
        strides = 2

        input_series = Input(
            shape=(self.time_steps, self.pitches, self.tracks))

        x = input_series
        x = Reshape((self.time_steps, self.pitches, 1, self.tracks))(x)

        # Contract
        x1 = x
        # x = ConvLSTM2D(
        #     filters=32, kernel_size=(kernel_size, 1), padding='same',
        #     strides=(strides, 1), dropout=0.0, recurrent_dropout=0.0,
        #     return_sequences=True)(x)
        x = ConvLSTM2D(
            filters=64, kernel_size=(kernel_size, 1), padding='same',
            strides=(strides, 1), dropout=0.0, recurrent_dropout=0.0)(x)

        # Expand
        y = x

        y = Conv2DTranspose(
            filters=64, kernel_size=(kernel_size, 1), padding='same',
            strides=(strides, 1), activation='relu')(y)
        x1_last = Lambda(lambda x: x[:, -1, :, :])(x1)
        y = Concatenate()([x1_last, y])
        y = Conv2D(
            filters=64, kernel_size=(kernel_size, 1), padding='same',
            activation='relu')(y)
        y = Conv2D(
            filters=64, kernel_size=(kernel_size, 1), padding='same',
            activation='relu')(y)
        y = Conv2D(
            filters=self.tracks, kernel_size=1, padding='same',
            activation='relu')(y)

        y = Reshape((self.pitches, self.tracks))(y)
        prediction = y

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
            epochs: int = 1,
            callbacks: list = None):
        assert input_sequences.shape[1:] == (
            self.time_steps, self.pitches, self.tracks)
        return self.model.fit(
            input_sequences, next_samples, epochs=epochs,
            verbose=0, callbacks=callbacks)

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
