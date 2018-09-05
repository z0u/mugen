from os.path import dirname
import random
import sys

import numpy as np
import tensorflow


project_dir = dirname(dirname(__file__))
sys.path.append(project_dir)

random.seed(0)
np.random.seed(0)
tensorflow.set_random_seed(0)
