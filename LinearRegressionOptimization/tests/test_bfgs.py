import numpy as np
import pytest
from ..optimizers.bfgs import Bfgs


@pytest.fixture
def thetas():
    """Fixture for providing initial parameter values."""
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def gradient():
    """Fixture for providing gradient values."""
    return np.array([0.1, 0.2, 0.3])


@pytest.fixture
def additional_gradient():
    """Fixture for providing additional gradient values."""
    return np.zeros(3)


@pytest.fixture
def expected_updated_thetas():
    """Fixture for providing expected updated parameter values."""
    return np.array([-5.7, -4.8, -3.9])


@pytest.mark.parametrize("learning_rate", [1])
def test_bfgs_optimizer(
    thetas, gradient, additional_gradient, expected_updated_thetas, learning_rate
):
    """
    Test the Bfgs optimizer.

    Args:
        thetas (numpy.ndarray): Initial parameter values.
        gradient (numpy.ndarray): Gradients with respect to parameters.
        additional_gradient (numpy.ndarray): Additional gradients.
        learning_rate (float): Learning rate for the optimizer.

    Raises:
        AssertionError: If any of the test conditions are not met.
    """
    optimizer = Bfgs(learning_rate=learning_rate)
    updated_thetas = optimizer.apply_gradients(gradient, thetas, additional_gradient)

    # Check the shape of the updated parameters
    assert updated_thetas.shape == thetas.shape, updated_thetas

    # Check if the updated parameters are not equal to the initial parameters
    assert not np.array_equal(updated_thetas, thetas)

    # Check if the optimizer has the necessary attributes
    assert hasattr(optimizer, "thetas_n_1")
    assert hasattr(optimizer, "I")
    assert hasattr(optimizer, "B_inv_n_1")

    # Check if the optimizer attributes have the correct shapes
    assert optimizer.thetas_n_1.shape == thetas.shape
    assert optimizer.I.shape == (len(thetas), len(thetas))
    assert optimizer.B_inv_n_1.shape == (len(thetas), len(thetas))

    # Check if the updated parameters match the expected values
    assert np.allclose(
        updated_thetas, expected_updated_thetas
    ), f"Expected: {expected_updated_thetas}, Got: {updated_thetas}"
