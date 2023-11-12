import numpy as np


def get_from_csv(file_name="LinearRegressionOptimization/data/MultiVarLR.csv"):
    """
    Load data from a CSV file and split it into input features (X) and target values (y).

    Args:
        file_name (str): The path to the CSV file containing the data.

    Returns:
        numpy.ndarray, numpy.ndarray: X and y, the input features and target values.
    """
    data = np.genfromtxt(file_name, delimiter=",")

    X = data[:, :-1]
    y = data[:, -1, np.newaxis]

    return X, y


def get_from_numpy(
    array=None, file_name="LinearRegressionOptimization/data/MultiVarLR.npy"
):
    """
    Load data from a NumPy array or a NumPy binary file and split it into input features (X) and target values (y).

    Args:
        array (numpy.ndarray): An optional NumPy array containing the data. If not provided, the data is loaded from the file.
        file_name (str): The path to the NumPy binary file containing the data.

    Returns:
        numpy.ndarray, numpy.ndarray: X and y, the input features and target values.
    """
    if array is None:
        array = np.load(file_name)

    X = array[:, :-1]
    y = array[:, -1, np.newaxis]
    return X, y


def train_test_split(X, y, test_ratio=0.2, train_ratio=None, random_state=None):
    """
    Split the input data into random train and test subsets.

    Args:
        X (numpy.ndarray): Input features.
        y (numpy.ndarray): Target values.
        test_ratio (float): Proportion of the dataset to include in the test split (default is 0.2).
        train_ratio (float or None): Proportion of the dataset to include in the train split (default is None).
            If None, the value is set to 1 - test_ratio.
        random_state (int or None): Seed for random number generation (default is None).

    Returns:
        Tuple: A tuple of train and test data (X_train, X_test, y_train, y_test).
    """
    if random_state is not None:
        np.random.seed(random_state)

    total_samples = X.shape[0]

    if train_ratio is None:
        train_ratio = 1.0 - test_ratio

    train_size = int(total_samples * train_ratio)

    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, X_test, y_train, y_test
