import datetime
import os
import shutil
from sys import stdout
import tempfile
import time

import iterm2_tools
from keras.callbacks import Callback
from keras.utils import plot_model

from file_vis import HeadlessUi, NoninteractiveProgress


class ITermUi(HeadlessUi):
    @property
    def progress_callback(self):
        return InteractiveProgress()

    def plot_model(self, model):
        filename = super().plot_model(model)
        if not filename:
            return
        print()
        iterm2_tools.images.display_image_file(filename)
        print()

    def plot_history(self, history):
        filename = super().plot_history(history)
        iterm2_tools.images.display_image_file(filename)
        print()

    def plot_actual_vs_expected(self, actual, expected):
        filename = super().plot_actual_vs_expected(actual, expected)
        iterm2_tools.images.display_image_file(filename)
        print()


class InteractiveProgress(NoninteractiveProgress):
    def reset(self):
        super().reset()
        self.epoch_duration = 0

    def on_epoch_begin(self, *args, **kwargs):
        super().on_epoch_begin(*args, **kwargs)
        self.epoch_start = time.perf_counter()

    def on_epoch_end(self, *args, **kwargs):
        super().on_epoch_end(*args, **kwargs)
        self.epoch_duration = time.perf_counter() - self.epoch_start

    def update(self):
        metrics = []
        for metric in self.metrics.items():
            metrics.append("%s: %0.3f" % metric)
        metrics_str = ', '.join(metrics)

        progress = self.epoch / self.epochs
        progress += (self.samples_seen / self.epoch_size) / self.epochs
        progress = min(progress, 1.0)
        if self.epoch_duration:
            remaining_duration = int(
                self.epoch_duration * (self.epochs - self.epoch))
            duration_str = str(datetime.timedelta(seconds=remaining_duration))
        else:
            duration_str = '--:--:--'
        prog_str = '%4.1f%% %s' % (progress * 100, duration_str)

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

    def clear_line(self):
        stdout.write('\r' + self.pad_to_term_width('') + '\r')
        stdout.flush()

    def on_train_end(self, logs=None):
        self.clear_line()
