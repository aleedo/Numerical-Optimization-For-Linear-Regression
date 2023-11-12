import numpy as np
from .base import Optimizer


class Sgd(Optimizer):
    def __init__(self, learning_rate=0.01):
        """
        Initialize the Stochastic Gradient Descent (SGD) optimizer.

        Args:
            learning_rate (float): Learning rate for the optimizer (default is 0.01).
        """
        self.learning_rate = learning_rate

    def apply_gradients(self, gradient, thetas):
        """
        Apply gradients to update the model parameters using Stochastic Gradient Descent (SGD).

        Args:
            gradient (numpy.ndarray): Gradients with respect to model parameters.
            thetas (numpy.ndarray): Current model parameters.

        Returns:
            numpy.ndarray: Updated model parameters after applying gradients.
        """
        thetas = thetas - self.learning_rate * gradient
        return thetas
