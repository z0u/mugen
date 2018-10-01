from keras.layers import (
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Conv3D,
    ConvLSTM2D,
    # Dense,
    # Flatten,
    Input,
    Lambda,
    # LSTM,
    MaxPool2D,
    MaxPool3D,
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
        self.kernel_size = 2
        self.pool_size = 2

    def build_model(self):
        input_series = Input(
            shape=(self.time_steps, self.pitches, self.tracks))

        x = input_series
        x = Reshape((self.time_steps, self.pitches, 1, self.tracks))(x)

        x = self.add_unet_layers(x, max_depth=2, base_filters=32)

        y = Conv2D(
            filters=self.tracks, kernel_size=1, padding='same',
            activation='relu')(x)
        y = Reshape((self.pitches, self.tracks))(y)
        prediction = y

        model = Model(input_series, prediction)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['accuracy'])
        self.model = model

    def add_contraction(self, filters, x, is_last_layer):
        high_res = x
        high_res = Conv3D(
            filters, kernel_size=(self.kernel_size, 1, 1), padding='same',
            activation='relu')(high_res)
        high_res = Conv3D(
            filters, kernel_size=(self.kernel_size, 1, 1), padding='same',
            activation='relu')(high_res)
        h = ConvLSTM2D(
            filters, kernel_size=(self.kernel_size, 1), padding='same',
            strides=(1, 1), dropout=0.0, recurrent_dropout=0.0,
            return_sequences=not is_last_layer)(high_res)
        if not is_last_layer:
            h = MaxPool3D(pool_size=(1, self.pool_size, 1), padding='same')(h)
        else:
            h = MaxPool2D(pool_size=(self.pool_size, 1), padding='same')(h)
        return high_res, h

    def add_expansion(self, filters, x_high_res, x):
        h = Conv2DTranspose(
            filters, kernel_size=(self.kernel_size, 1), padding='same',
            strides=(self.pool_size, 1), activation='relu')(x)
        x_high_last = Lambda(lambda x: x[:, -1, :, :])(x_high_res)
        h = Concatenate()([x_high_last, h])
        h = Conv2D(
            filters, kernel_size=(self.kernel_size, 1), padding='same',
            activation='relu')(h)
        h = Conv2D(
            filters, kernel_size=(self.kernel_size, 1), padding='same',
            activation='relu')(h)
        return h

    def add_unet_layers(self, x, max_depth, base_filters, depth=1):
        filters = base_filters * (depth ** 2)
        is_last_layer = depth >= max_depth
        high_res, x = self.add_contraction(filters, x, is_last_layer)

        if not is_last_layer:
            x = self.add_unet_layers(x, max_depth, base_filters, depth + 1)
        else:
            x = Conv2D(
                filters=filters * 2, kernel_size=(self.kernel_size, 1),
                padding='same', activation='relu')(x)
            x = Conv2D(
                filters=filters * 2, kernel_size=(self.kernel_size, 1),
                padding='same', activation='relu')(x)

        x = self.add_expansion(filters, high_res, x)
        return x

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
