import pytest
import numpy as np
from ..input import get_from_csv, get_from_numpy, train_test_split


@pytest.fixture
def data():
    """
    Generate random data for testing.

    Returns:
        numpy.ndarray: Random data for testing.
    """
    np.random.seed(42)
    return np.random.randint(10, size=(15, 5))


@pytest.fixture
def X(data):
    """
    Extract input features (X) from the data fixture.

    Args:
        data (numpy.ndarray): Random data.

    Returns:
        numpy.ndarray: Input features (X).
    """
    return data[:, :-1]


@pytest.fixture
def y(data):
    """
    Extract target values (y) from the data fixture.

    Args:
        data (numpy.ndarray): Random data.

    Returns:
        numpy.ndarray: Target values (y).
    """
    return data[:, -1, np.newaxis]


def test_get_from_csv():
    """
    Test the 'get_from_csv' function.

    This function tests if the 'get_from_csv' function correctly loads data from a CSV file.
    """
    X, y = get_from_csv(file_name="LinearRegressionOptimization/data/MultiVarLR.csv")
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert y.shape[1] > 0


def test_get_from_numpy_file():
    """
    Test the 'get_from_numpy' function.

    This function tests if the 'get_from_numpy' function correctly loads data from a npy file.
    """
    X, y = get_from_numpy(file_name="LinearRegressionOptimization/data/MultiVarLR.npy")
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert y.shape[1] > 0


def test_get_from_numpy_array(data, X, y):
    """
    Test the 'get_from_numpy' function.

    This function tests if the 'get_from_numpy' function correctly extracts X and y from a numpy array.

    Args:
        data (numpy.ndarray): Random data for testing.
        X (numpy.ndarray): Input features extracted from data fixture.
        y (numpy.ndarray): Target values extracted from data fixture.
    """
    X_, y_ = get_from_numpy(data)

    assert isinstance(X_, np.ndarray)
    assert isinstance(y_, np.ndarray)

    assert np.array_equal(X, X_)
    assert np.array_equal(y, y_)


def test_train_test_split(X, y):
    """
    Test the 'train_test_split' function.

    This function tests if the 'train_test_split' function correctly splits the data into training and testing subsets.

    Args:
        X (numpy.ndarray): Input features.
        y (numpy.ndarray): Target values.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    assert X_train.shape[0] == y_train.shape[0]
    assert y_train.shape[1] == y_test.shape[1]
    assert X.shape[1] == X_train.shape[1] == X_test.shape[1]
