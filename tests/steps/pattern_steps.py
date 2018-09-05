from itertools import cycle, islice

import numpy as np
from pytest import fixture
from pytest_bdd import given, when, then
from pytest_bdd.parsers import parse

from mugen import Mugen


@fixture
def prediction():
    return Prediction()


class Prediction:
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def __repr__(self):
        return repr(self._value)


def repeat_for(template, length):
    return islice(cycle(template), length)


def sawtooth(period, length, channels):
    def point(i):
        return (i,) * channels
    template = tuple(point(i) for i in range(period))
    return list(repeat_for(template, length))


@given(parse('a sawtooth wave of {steps:d} steps'))
def sequences(steps):
    batch_size = 100
    seq_length = 100
    channels = 1
    input_sequence = np.array(sawtooth(steps, seq_length, channels))
    input_sequence = np.broadcast_to(
        input_sequence, (batch_size, seq_length, channels))
    return input_sequence


@given(parse('the model has been trained for {epochs:d} epochs'))
def trained_model(sequences, epochs):
    seq_length = sequences.shape[1] - 1
    channels = sequences.shape[2]
    model = Mugen(seq_length, channels)
    model.build_model()
    input_sequences = sequences[:, :seq_length, :]
    next_samples = sequences[:, -1, :]
    for _ in range(epochs):
        model.train(input_sequences, next_samples)
    return model


@when('the next sample is generated')
def predict(sequences, trained_model, prediction):
    seq_length = sequences.shape[1] - 1
    input_sequences = sequences[:, :seq_length, :]
    prediction.value = trained_model.generate_sample_batch(input_sequences)


@then('the extension matches the initial sequence')
def validate(sequences, prediction):
    next_samples = sequences[:, -1, :]
    assert np.all(prediction.value == next_samples)
