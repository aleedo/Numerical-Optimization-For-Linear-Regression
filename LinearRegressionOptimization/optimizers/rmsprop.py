import numpy as np
from .base import Optimizer


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, beta=0.9):
        """
        Initialize the RMSprop optimizer.

        Args:
            learning_rate (float): Learning rate for the optimizer (default is 0.001).
            beta (float): Decay factor for the moving average of squared gradients (default is 0.9).
        """
        self.learning_rate = learning_rate
        self.beta = beta
        self.eps = 1e-8
        self.v_t = None

    def apply_gradients(self, gradient, thetas):
        """
        Apply gradients to update the model parameters using RMSprop.

        Args:
            gradient (numpy.ndarray): Gradients with respect to model parameters.
            thetas (numpy.ndarray): Current model parameters.

        Returns:
            numpy.ndarray: Updated model parameters after applying gradients.
        """
        thetas = thetas.copy()

        if self.v_t is None:
            self.v_t = np.zeros_like(thetas)

        self.v_t = self.beta * self.v_t + (1 - self.beta) * gradient**2

        thetas -= self.learning_rate * gradient / (np.sqrt(self.v_t) + self.eps)

        return thetas
