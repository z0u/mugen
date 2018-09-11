from math import ceil

from mido import MidiFile
import numpy as np


def load_midi(filepath):
    return MidiFile(filepath)


def rasterize(
        mid: MidiFile, ticks_per_second: float = 12, polyphony: int = 24
        ) -> np.array:
    image = create_empty_image(
        mid.length, ticks_per_second, polyphony)
    quantize(mid, image)
    return image


def create_empty_image(
        length: float, ticks_per_second: float, polyphony: int
        ) -> np.array:
    total_ticks = int(ceil(length * ticks_per_second))

    # Concatenation of three one-hot vectors
    n_voices = 128
    n_pitches = 128
    n_velocities = 128
    data_vector_length = n_voices + n_pitches + n_velocities

    return np.zeros(shape=(total_ticks, polyphony, data_vector_length))


def quantize(mid: MidiFile, image: np.array):
    pass
