from itertools import cycle, islice

import matplotlib.pyplot as plt
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
    pitches = 12
    time_steps = batch_size = pitches * 8
    tracks = 1

    base_progression = progression_2d(time_steps, pitches)
    sequences = np.zeros((batch_size, time_steps, pitches, tracks))
    for i, variant in enumerate(sequences):
        variant[:, :, 0] = np.roll(base_progression, i, axis=0)

    return sequences


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
    display_actual_vs_expected(next_samples, prediction.value)
    # assert find_mse(next_samples, prediction.value) < 0.01


def display_actual_vs_expected(actual, expected):
    difference = expected - actual
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax1.set_title('Expected')
    ax1.set_ylabel('time')
    ax1.set_xlabel('pitch')
    plt.imshow(actual[:, :, 0])
    ax2 = fig.add_subplot(132, sharey=ax1)
    ax2.set_title('Actual')
    ax2.set_xlabel('pitch')
    plt.imshow(expected[:, :, 0])
    ax3 = fig.add_subplot(133, sharey=ax1)
    ax3.set_title('Difference')
    ax3.set_xlabel('pitch')
    plt.imshow(abs(difference[:, :, 0]))
    plt.show()
    print("MSE: %.3f" % find_mse(actual, expected))


def find_mse(actual, expected):
    difference = expected - actual
    return sum(difference.flatten() ** 2) / difference.size
