import mido
import numpy as np
from pytest import fixture
from pytest_bdd import given, when, then

from mugen import input
from .mutable_fixture import MutableFixture


@fixture
def rasterized_data() -> MutableFixture[np.array]:
    return MutableFixture()


@given('a C-major scale read from a MIDI file')
def midi_data(resolve_data_file) -> mido.MidiFile:
    file_path = resolve_data_file('C-natural_major.mid')
    return input.load_midi(file_path)


@given('a C-major scale read from a raster file')
def raster_data(resolve_data_file) -> np.array:
    return None
    # file_path = resolve_data_file('C-natural_major.h5')


@when('the MIDI data is converted into an array')
def the_midi_data_is_converted_into_an_array(
        midi_data: mido.MidiFile,
        rasterized_data: MutableFixture[np.array]):
    rasterized_data.value = input.rasterize(midi_data, 5)


@then('the array matches the known raster data')
def the_array_matches_the_known_raster_data(
        rasterized_data: MutableFixture[np.array],
        raster_data: np.array):
    assert np.all(rasterized_data.value == raster_data)
