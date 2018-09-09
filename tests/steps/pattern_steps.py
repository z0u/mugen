from itertools import cycle, islice

import numpy as np
from pytest import fixture
from pytest_bdd import given, when, then
from pytest_bdd.parsers import parse

from mugen.mugen import Mugen


class Prediction:
    @property
    def value(self) -> np.array:
        return self._value

    @value.setter
    def value(self, value: np.array):
        self._value = value

    def __repr__(self):
        return repr(self._value)


@fixture
def prediction() -> Prediction:
    return Prediction()


def repeat_for(template, offset: int, length: int):
    return islice(cycle(template), offset, offset + length)


def sawtooth(period: int, offset: int, length: int, channels: int):
    def point(i):
        return (i,) * channels
    template = tuple(point(i) for i in range(period))
    return list(repeat_for(template, offset, length))


@given(parse('a sawtooth wave of {steps:d} steps'), target_fixture='sequences')
def sawtooth_sequences(steps: int):
    batch_size = 100
    seq_length = 100
    channels = 1
    sequences = np.stack([
        sawtooth(steps, i, seq_length, channels)
        for i in range(batch_size)])
    return sequences


@given('some random but static sequences', target_fixture='sequences')
def random_sequences():
    batch_size = 100
    seq_length = 100
    channels = 1
    return np.random.random_integers(0, 10, (batch_size, seq_length, channels))


@given(parse('the model has been trained for {epochs:d} epochs'))
def trained_model(sequences, epochs):
    seq_length = sequences.shape[1] - 1
    channels = sequences.shape[2]
    model = Mugen(seq_length, channels)
    model.build_model()
    input_sequences = sequences[:, :seq_length, :]
    next_samples = sequences[:, -1, :]
    model.train(input_sequences, next_samples, epochs)
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
