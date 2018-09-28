from os.path import dirname
import os.path
import sys

from pytest import fixture


if os.environ.get('TERM_PROGRAM') == 'iTerm.app':
    from iterm_vis import PlotToITerm as Plotter
else:
    from file_vis import PlotToFile as Plotter


project_dir = dirname(dirname(__file__))
sys.path.append(project_dir)


@fixture
def resolve_data_file():
    def resolve(filename):
        return os.path.join(dirname(__file__), 'data', filename)
    return resolve


@fixture
def vis():
    double_resolution = True
    dark_background = True
    return Plotter(double_resolution, dark_background)
