from mido import Message, MidiFile, MidiTrack
import numpy as np
from pytest import fixture

from mugen import input


@fixture
def midifile():
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(Message('program_change', program=12, time=0))
    track.append(Message('note_on', note=32, velocity=64, time=160))
    track.append(Message('note_on', note=33, velocity=32, time=0))
    track.append(Message('note_off', note=32, velocity=127, time=160))
    track.append(Message('note_on', note=33, velocity=0, time=160))

    return mid


@fixture
def rasterizer():
    return input.MIDIRasterizer(pitches=32, pitch_offset=32)


def test_that_input_shape_is_as_expected(rasterizer):
    image = rasterizer.create_empty_image(2.0)
    steps = rasterizer.steps_per_second * 2
    pitches = rasterizer.pitches
    variables = 1
    tracks = 1
    assert steps == 12
    assert pitches == 32
    assert image.shape == (steps, pitches, variables, tracks)


def test_that_an_initialized_image_is_all_nan_except_for_the_first_timestep(
        rasterizer):
    image = np.empty((4, 4, 1, 1))
    rasterizer.initialize_time_series(image)
    assert np.count_nonzero(np.isnan(image)) == 3 * 4
    assert np.count_nonzero(image[0] == 0) == 4


def test_that_note_messages_affect_one_pixel(
        midifile, rasterizer, mocker):
    rasterizer.get_ticks_per_step = mocker.MagicMock(return_value=160.0)
    image = np.full((4, 32, 1, 1), np.nan)
    rasterizer.insert_messages(midifile, image)

    expected = np.full((4, 32, 1, 1), np.nan)
    expected[:, :2, 0, 0] = np.array([
        [np.nan, np.nan],
        [64/127, 32/127],
        [0,      np.nan],
        [np.nan, 0],
    ])
    assert np.allclose(image, expected, equal_nan=True)


def test_get_ticks_per_step(rasterizer):
    assert rasterizer.steps_per_second == 6
    assert rasterizer.get_ticks_per_step(120, 480) == 160


def test_that_events_are_propagated(rasterizer):
    image = np.array([
        [0,      1,      0],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, 1],
        [np.nan, 0,      np.nan],
        [1,      np.nan, np.nan],
    ])
    image.resize((5, 3, 1, 1))
    rasterizer.propagate_events_forward(image)
    expected = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [1, 0, 1],
    ])
    expected.resize((5, 3, 1, 1))
    assert np.all(image == expected)
