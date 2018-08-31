from functools import partial

import pytest_bdd

from steps import *


scenario = partial(pytest_bdd.scenario, 'pattern_gen.feature')


@scenario('Extend a short repeating sequence')
def test_scenario():
    pass
