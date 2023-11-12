import numpy as np
import pytest
from ..optimizers.sgd import Sgd


@pytest.fixture
def thetas():
    """Fixture for providing initial parameter values."""
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def gradient():
    """Fixture for providing gradient values."""
    return np.array([0.1, 0.2, 0.3])


@pytest.fixture
def expected_updated_thetas():
    """Fixture for providing expected updated parameter values."""
    return np.array([0.99, 1.98, 2.97])


@pytest.mark.parametrize("learning_rate", [0.1])
def test_sgd_optimizer(gradient, thetas, expected_updated_thetas, learning_rate):
    """
    Test the Sgd optimizer.

    Args:
        gradient (numpy.ndarray): Gradients with respect to parameters.
        thetas (numpy.ndarray): Initial parameter values.
        expected_updated_thetas (numpy.ndarray): Expected updated parameter values.
        learning_rate (float): Learning rate for the optimizer.

    Raises:
        AssertionError: If any of the test conditions are not met.
    """
    optimizer = Sgd(learning_rate=learning_rate)
    updated_thetas = optimizer.apply_gradients(gradient, thetas)

    # Check the shape of the updated parameters
    assert updated_thetas.shape == thetas.shape

    # Check if the updated parameters are not equal to the initial parameters
    assert not np.array_equal(updated_thetas, thetas)

    # Check if the updated parameters match the expected values
    assert np.allclose(
        updated_thetas, expected_updated_thetas
    ), f"Expected: {expected_updated_thetas}, Got: {updated_thetas}"
