import numpy as np
import pytest
from ..src.linear_regression import LinearRegression
from ..optimizers import Sgd


@pytest.fixture
def X():
    return np.expand_dims(np.arange(1, 6), axis=1)


@pytest.fixture
def y(X):
    return 2 * X + 3


@pytest.fixture
def lr_model():
    return LinearRegression(
        optimizer=Sgd(learning_rate=0.1),
        random_state=42,
        max_epochs=500,
        cost_threshold=1e-6,
    )


@pytest.fixture
def trained_lr_model(lr_model, X, y):
    lr_model.fit(X, y)
    return lr_model


def test_fit_with_invalid_inputs(X, y, lr_model):
    with pytest.raises(ValueError):
        lr_model.fit(np.squeeze(X), y)

    with pytest.raises(ValueError):
        lr_model.fit(X, np.squeeze(y))


def test_fit_simple_linear_weight(trained_lr_model):
    """
    Test the fit function for simple linear regression with known weights.

    Args:
        trained_lr_model (LinearRegression): Trained linear regression model.
    """

    expected_weight = 2.0
    expected_intercept = 3.0

    assert np.allclose(trained_lr_model.coef_, np.array([expected_weight]), rtol=1e-2)
    assert np.allclose(
        trained_lr_model.intercept_, np.array([expected_intercept]), rtol=1e-2
    )


@pytest.mark.parametrize(
    "new_X, expected_predictions",
    [
        (np.array([[10], [20]]), np.array([23, 43])),
        (np.array([[100], [200]]), np.array([203, 403])),
    ],
)
def test_predict(trained_lr_model, new_X, expected_predictions):
    """
    Test the predict function with various input values.

    Args:
        trained_lr_model (LinearRegression): Trained linear regression model.
        new_X (numpy.ndarray): New input features.
        expected_predictions (numpy.ndarray): Expected target value predictions.
    """
    predictions = np.squeeze(trained_lr_model.predict(new_X))
    assert np.allclose(
        predictions, expected_predictions, rtol=1e-2
    ), f"predictions {predictions}"


def test_evaluate():
    """
    Evaluate the model's performance using the R-squared metric.

    Args:
        y_true (numpy.ndarray): True target values.
        y_pred (numpy.ndarray): Predicted target values.

    Returns:
        float: R-squared (coefficient of determination) value.
    """
    y_true = np.array([2.0, 4.0, 5.0, 4.0, 5.0])
    y_pred = np.array([2.8, 3.4, 4, 4.6, 5.2])

    r2 = LinearRegression.evaluate(y_true, y_pred)

    assert np.allclose(r2, 0.6)
