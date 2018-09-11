from os.path import dirname
import os.path
import random
import sys

import numpy as np
from pytest import fixture
import tensorflow


project_dir = dirname(dirname(__file__))
sys.path.append(project_dir)

random.seed(0)
np.random.seed(0)
tensorflow.set_random_seed(0)


@fixture
def resolve_data_file():
    def resolve(filename):
        return os.path.join(dirname(__file__), 'data', filename)
    return resolve
