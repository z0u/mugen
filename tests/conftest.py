from os.path import dirname
import os.path
import sys

from pytest import fixture


project_dir = dirname(dirname(__file__))
sys.path.append(project_dir)


@fixture
def resolve_data_file():
    def resolve(filename):
        return os.path.join(dirname(__file__), 'data', filename)
    return resolve
