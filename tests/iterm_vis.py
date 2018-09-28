import os
import shutil
import tempfile

import iterm2_tools
from keras.utils import plot_model

from file_vis import PlotToFile


class PlotToITerm(PlotToFile):
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
