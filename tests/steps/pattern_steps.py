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


def add_tempo(sequence, ticks_per_beat):
    time_steps, pitches, tracks = sequence.shape
    beats = time_steps // ticks_per_beat
    time_divisions = np.array([2, 4]).reshape((1, 2))
    # time_divisions = np.array([1, 1, 2, 3, 4, 6]).reshape((1, 6))
    n_divisions = time_divisions.shape[1]
    tick_progression = np.repeat(
        np.arange(ticks_per_beat)[:, np.newaxis],
        n_divisions, axis=1)
    tick_progression %= ticks_per_beat // time_divisions
    timestamps = np.tile(tick_progression == 0, (beats, 1)).reshape(
        (time_steps, n_divisions))
    timestamps = np.repeat(timestamps[:, :, np.newaxis], tracks, axis=2)
    return np.concatenate((timestamps, sequence), axis=1)


@given(parse('a progression of 10 pitches'), target_fixture='sequences')
def progression_sequences():
    ticks_per_beat = 12
    pitches = 12
    time_steps = batch_size = pitches * ticks_per_beat
    tracks = 1

    base_progression = progression_2d(time_steps, pitches)
    base_progression = base_progression[:, :, np.newaxis]
    base_progression = add_tempo(base_progression, ticks_per_beat)
    width = base_progression.shape[1]

    sequences = np.zeros((batch_size, time_steps, width, tracks))
    for i, variant in enumerate(sequences):
        variant[:, :, :] = np.roll(base_progression, i, axis=0)

    return sequences


@given(parse('the model has been trained for {epochs:d} epochs'))
def trained_model(sequences: np.array, epochs: int, ui):
    batches, time_steps, pitches, tracks = sequences.shape
    # Use final sample as output
    time_steps -= 1
    model = Mugen(time_steps, pitches, tracks)
    model.build_model()
    ui.plot_model(model.model)
    input_sequences = sequences[:, :-1, :, :]
    next_samples = sequences[:, -1, :, :]
    history = model.train(
        input_sequences, next_samples, epochs,
        callbacks=[ui.progress_callback])
    ui.plot_history(history)
    return model


@when('the next sample is generated')
def predict(
        sequences: np.array,
        trained_model: Mugen,
        prediction: MutableFixture[np.array]):
    input_sequences = sequences[:, :-1, :, :]
    prediction.value = trained_model.generate_sample_batch(input_sequences)


@then('the extension matches the initial sequence')
def validate(sequences: np.array, prediction: MutableFixture[np.array], ui):
    next_samples = sequences[:, -1, :, :]
    ui.plot_actual_vs_expected(next_samples, prediction.value)
    print("MSE: %.3f" % find_mse(next_samples, prediction.value))
    # assert find_mse(next_samples, prediction.value) < 0.01


def find_mse(actual, expected):
    difference = expected - actual
    return sum(difference.flatten() ** 2) / difference.size
