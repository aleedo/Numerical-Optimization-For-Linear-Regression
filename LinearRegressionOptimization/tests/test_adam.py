import numpy as np
import pytest
from ..optimizers.adam import Adam


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
    return np.array([0.9, 1.9, 2.9])


@pytest.mark.parametrize("learning_rate, beta_1, beta_2", [(0.1, 0.9, 0.7)])
def test_adam_optimizer(
    gradient, thetas, expected_updated_thetas, learning_rate, beta_1, beta_2
):
    """
    Test the Adam optimizer.

    Args:
        gradient (numpy.ndarray): Gradients with respect to parameters.
        thetas (numpy.ndarray): Initial parameter values.
        expected_updated_thetas (numpy.ndarray): Expected updated parameter values.
        learning_rate (float): Learning rate for the optimizer.
        beta_1 (float): Exponential decay rate for the first moment estimate.
        beta_2 (float): Exponential decay rate for the second moment estimate.

    Raises:
        AssertionError: If any of the test conditions are not met.
    """
    optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    updated_thetas = optimizer.apply_gradients(gradient, thetas)

    # Check if the v_t values are updated correctly
    assert np.allclose(optimizer.v_t, [0.003, 0.012, 0.027])

    # Check if the m_t values are updated correctly
    assert np.allclose(optimizer.m_t, [0.01, 0.02, 0.03])

    # Check the shape of the updated parameters
    assert updated_thetas.shape == thetas.shape

    # Check if the updated parameters are not equal to the initial parameters
    assert not np.array_equal(updated_thetas, thetas)

    # Check if the updated parameters match the expected values
    assert np.allclose(
        updated_thetas, expected_updated_thetas
    ), f"Expected: {expected_updated_thetas}, Got: {updated_thetas}"
