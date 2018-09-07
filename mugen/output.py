from time import sleep

import mido


class Progression:
    def __init__(self, output, octave=5):
        self.output = output
        self.last_note = None
        self.offset = octave * 12

    def play(self, note):
        if self.last_note:
            self.output.send(mido.Message(
                'note_off', note=self.last_note, velocity=78))
        note += self.offset
        self.output.send(mido.Message(
            'note_on', note=note, velocity=78))
        self.last_note = note

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        if self.last_note:
            self.output.send(mido.Message(
                'note_off', note=self.last_note, velocity=78))


def play_simple_test():
    output = mido.open_output()
    with Progression(output, 4) as progression:
        for note in (0, 4, 7, 12, 7, 4, 0):
            sleep(1/3)
            progression.play(note)
        sleep(1)
