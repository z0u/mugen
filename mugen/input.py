from math import ceil

import mido
import numpy as np


MIDI_MAX_VELOCITY = 127


class MIDIRasterizer:
    def __init__(self, pitches: int = 64, pitch_offset: int = 32):
        '''
        Encodes MIDI data as a piano roll image with dimensions:
        - step: regular absolute time (not variable offset as in MIDI)
        - pitch: MIDI pitch e.g. keys on a keyboard
        - data: various information of a note e.g. velocity
        - track: the instrument
        '''
        self.steps_per_second = 6
        self.pitches = pitches
        self.pitch_offset = pitch_offset

    def load_midi(self, filepath: str) -> mido.MidiFile:
        return mido.MidiFile(filepath)

    def rasterize(self, mid: mido.MidiFile) -> np.array:
        image = self.create_empty_image(mid.length)
        self.initialize_time_series(image)
        self.insert_messages(mid, image)
        self.propagate_events_forward(image)
        return image

    def create_empty_image(self, length_s: float) -> np.array:
        n_steps = int(ceil(length_s * self.steps_per_second))
        width = self.pitches
        n_variables = len(['velocity'])
        n_tracks = 1
        return np.empty((n_steps, width, n_variables, n_tracks))

    def initialize_time_series(self, image):
        # Set first time step to zero, to allow values to be propagated.
        image[:] = np.nan
        image[0] = 0

    def insert_messages(self, mid: mido.MidiFile, image: np.array):
        # Default nominal tempo is 120bpm
        ticks_per_step = self.get_ticks_per_step(120, mid.ticks_per_beat)

        for track_i, track in enumerate(mid.tracks):
            track_img = image[:, :, :, track_i]
            current_tick = 0
            for msg in track:
                current_tick += msg.time
                current_step = int(current_tick // ticks_per_step)
                if msg.type == 'note_on':
                    pitch = msg.note - self.pitch_offset
                    track_img[current_step, pitch, 0] = (
                        msg.velocity / MIDI_MAX_VELOCITY)
                elif msg.type == 'note_off':
                    pitch = msg.note - self.pitch_offset
                    track_img[current_step, pitch, 0] = 0.0

    def get_ticks_per_step(self, bpm, ticks_per_beat):
        tempo = mido.bpm2tempo(bpm) / 1000000
        s_per_step = 1 / self.steps_per_second
        s_per_tick = tempo / ticks_per_beat
        return s_per_step / s_per_tick

    def propagate_events_forward(self, image: np.array):
        for step_a, step_b in zip(image, image[1:]):
            mask = np.isnan(step_b)
            step_b[mask] = step_a[mask]
