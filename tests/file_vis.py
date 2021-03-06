import os
import shutil
import tempfile

from keras.callbacks import Callback
from keras.utils import plot_model
import matplotlib
from PIL import Image, ImageOps


matplotlib.use('Agg')
import matplotlib.pyplot as plt #noqa


def get_test_data_dir():
    return os.path.join(os.path.dirname(__file__), 'data')


class HeadlessUi:
    def __init__(self, double_resolution: bool, dark_background: bool):
        self.double_resolution = double_resolution
        self.dark_background = dark_background
        self.save_model = bool(shutil.which('dot'))
        self.output_dir = os.path.join(get_test_data_dir(), 'stats')

    @property
    def progress_callback(self):
        return NoninteractiveProgress()

    def ensure_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir

    def invert_image(self, filename):
        image = Image.open(filename)
        if image.mode == 'RGBA':
            r, g, b, a = image.split()
            rgb_image = Image.merge('RGB', (r, g, b))
            inverted_image = ImageOps.invert(rgb_image)
            r, g, b = inverted_image.split()
            inverted_image = Image.merge('RGBA', (r, g, b, a))
        else:
            inverted_image = ImageOps.invert(image)
        inverted_image.save(filename)

    def plot_model(self, model):
        if not self.save_model:
            return None
        output_dir = self.ensure_output_dir()
        filename = os.path.join(output_dir, 'model_graph.png')
        plot_model(model, to_file=filename, show_shapes=True, rankdir='TB')
        if self.dark_background:
            self.invert_image(filename)
        return filename

    @property
    def matplotlib_resolution(self):
        if self.double_resolution:
            return 180
        else:
            return 90

    def plot_history(self, history):
        if self.dark_background:
            plt.style.use('dark_background')
        plt.figure()
        keys = sorted(history.history.keys())
        for k in keys:
            plt.plot(history.history[k])
        plt.legend(keys, loc='upper left')

        output_dir = self.ensure_output_dir()
        filename = os.path.join(output_dir, 'history.png')
        plt.savefig(filename, dpi=self.matplotlib_resolution)
        return filename

    def plot_actual_vs_expected(self, actual, expected):
        difference = expected - actual
        if self.dark_background:
            plt.style.use('dark_background')

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

        output_dir = self.ensure_output_dir()
        filename = os.path.join(output_dir, 'diff.png')
        plt.savefig(filename, dpi=self.matplotlib_resolution)
        return filename


class NoninteractiveProgress(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self):
        self.epoch = 0
        self.epochs = 1
        self.epoch_size = 1
        self.samples_seen = 0
        self.metrics = {}

    def update(self):
        print('\rEpoch %d/%d' % (self.epoch + 1, self.epochs))

    def on_train_begin(self, logs=None):
        self.reset()
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.epoch_size = self.params['samples']
        self.samples_seen = 0

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.samples_seen += logs.get('size', 0)
        self.metrics = {k: logs[k] for k in self.params['metrics']}
        self.update()

    def on_train_end(self, logs=None):
        print('Training complete\n')
