from mugen import input


def test_that_input_shape_is_as_expected():
    image = input.create_empty_image(10, 12, 3)
    assert image.shape == (10 * 12, 3, 128 * 3)
