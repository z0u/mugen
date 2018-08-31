from pytest import fixture
from pytest_bdd import given, when, then
from pytest_bdd.parsers import parse


@fixture
def output_sequence():
    return []


@given(parse('a short sawtooth wave of {steps:d} steps'))
def input_sequence(steps):
    return list(range(steps)) * 20


@given('the model has been trained on that sequence')
def trained_model(input_sequence):
    return None


@when('the next period is generated')
def prediction(input_sequence, trained_model, output_sequence):
    pass


@then('the extension matches the initial sequence')
def validation(input_sequence, output_sequence):
    assert input_sequence == output_sequence
