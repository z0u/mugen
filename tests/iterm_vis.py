import os
import shutil
from sys import stdout
import tempfile

import iterm2_tools
from keras.callbacks import Callback
from keras.utils import plot_model

from file_vis import HeadlessUi, NoninteractiveProgress


class ITermUi(HeadlessUi):
    @property
    def progress_callback(self):
        return InteractiveProgress()

    def plot_model(self, model):
        fname = super().plot_model(model)
        if not fname:
            return
        print()
        iterm2_tools.images.display_image_file(fname)
        print()

    def plot_actual_vs_expected(self, actual, expected):
        fname = super().plot_actual_vs_expected(actual, expected)
        iterm2_tools.images.display_image_file(fname)
        print()


class InteractiveProgress(NoninteractiveProgress):
    def update(self):
        metrics = []
        for metric in self.metrics.items():
            metrics.append("%s: %0.3f" % metric)
        metrics_str = ', '.join(metrics)

        progress = self.epoch / self.epochs
        progress += (self.samples_seen / self.epoch_size) / self.epochs
        progress = min(progress, 1.0)
        prog_str = '%2.1f%%' % (progress * 100)

        bar_cols = self.get_cols() - (len(prog_str) + len(metrics_str) + 2) - 1
        bar_str = self.render_bar(bar_cols, progress)

        message = '%s %s %s' % (prog_str, bar_str, metrics_str)
        stdout.write('\r' + message)
        stdout.flush()

    def render_bar(self, cols, progress):
        n_filled_cols = int((cols - 2) * progress)
        n_unfilled_cols = (cols - 2) - n_filled_cols
        bar = '|'
        bar += '-' * n_filled_cols
        bar += ' ' * n_unfilled_cols
        bar += '|'
        return bar

    def get_cols(self):
        cols, _ = shutil.get_terminal_size((80, 20))
        return cols

    def pad_to_term_width(self, message):
        cols = self.get_cols()
        pad_cols = cols - len(message)
        return message + (' ' * pad_cols)

    def on_train_end(self, logs=None):
        stdout.write('\r' + self.pad_to_term_width('Training complete') + '\n')
        stdout.flush()
