from functools import partial

import pytest_bdd

from steps import *


scenario = partial(pytest_bdd.scenario, 'pattern_gen.feature')


@scenario('Extend a sawtooth wave')
def test_sawtooth():
    pass


@scenario('Extend random data')
def test_random():
    pass
