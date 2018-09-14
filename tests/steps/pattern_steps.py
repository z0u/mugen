from itertools import cycle, islice

import numpy as np
from pytest import fixture
from pytest_bdd import given, when, then
from pytest_bdd.parsers import parse

from mugen.mugen import Mugen

from .mutable_fixture import MutableFixture


@fixture
def prediction() -> MutableFixture:
    return MutableFixture()


def progression_2d(time_steps, pitches):
    base_progression = np.zeros((time_steps, pitches))
    base_progression[
        np.arange(time_steps),
        np.arange(time_steps) // (time_steps // pitches)] = 1
    return base_progression


@given(parse('a progression of 10 pitches'), target_fixture='sequences')
def progression_sequences():
    time_steps = batch_size = 100
    pitches = 10
    tracks = 1

    base_progression = progression_2d(time_steps, pitches)
    sequences = np.zeros((batch_size, time_steps, pitches, tracks))
    for i, variant in enumerate(sequences):
        variant[:, :, 0] = np.roll(base_progression, i, axis=0)

    return sequences


@given('some random but static sequences', target_fixture='sequences')
def random_sequences():
    time_steps = batch_size = 100
    pitches = 10
    tracks = 1
    return np.random.random((batch_size, time_steps, pitches, tracks))


@given(parse('the model has been trained for {epochs:d} epochs'))
def trained_model(sequences: np.array, epochs: int):
    batches, time_steps, pitches, tracks = sequences.shape
    # Use final sample as output
    time_steps -= 1
    model = Mugen(time_steps, pitches, tracks)
    model.build_model()
    input_sequences = sequences[:, :-1, :, :]
    next_samples = sequences[:, -1, :, :]
    model.train(input_sequences, next_samples, epochs)
    return model


@when('the next sample is generated')
def predict(
        sequences: np.array,
        trained_model: Mugen,
        prediction: MutableFixture[np.array]):
    input_sequences = sequences[:, :-1, :, :]
    prediction.value = trained_model.generate_sample_batch(input_sequences)


@then('the extension matches the initial sequence')
def validate(sequences: np.array, prediction: MutableFixture[np.array]):
    next_samples = sequences[:, -1, :, :]
    with np.printoptions(precision=3, floatmode='fixed', suppress=True):
        print(next_samples[:, :, 0])
        print(prediction.value[:, :, 0])
    assert np.all(prediction.value == next_samples)
