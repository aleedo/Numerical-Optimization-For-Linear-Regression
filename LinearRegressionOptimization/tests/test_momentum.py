import numpy as np
import pytest
from ..optimizers.momentum import Momentum


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


@pytest.mark.parametrize("learning_rate, gamma", [(0.1, 0.9)])
def test_momentum_optimizer(
    gradient, thetas, expected_updated_thetas, learning_rate, gamma
):
    """
    Test the Momentum optimizer.

    Args:
        gradient (numpy.ndarray): Gradients with respect to parameters.
        thetas (numpy.ndarray): Initial parameter values.
        expected_updated_thetas (numpy.ndarray): Expected updated parameter values.
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Momentum coefficient.

    Raises:
        AssertionError: If any of the test conditions are not met.
    """
    optimizer = Momentum(learning_rate=learning_rate, gamma=gamma)
    updated_thetas = optimizer.apply_gradients(gradient, thetas)

    # Check the shape of the updated parameters
    assert updated_thetas.shape == thetas.shape

    # Check if the updated parameters are not equal to the initial parameters
    assert not np.array_equal(updated_thetas, thetas)

    # Check if the updated parameters match the expected values
    assert np.allclose(
        updated_thetas, expected_updated_thetas
    ), f"Expected: {expected_updated_thetas}, Got: {updated_thetas}"
