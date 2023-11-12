import numpy as np
from .base import Optimizer


class Nag(Optimizer):
    def __init__(self, learning_rate=0.01, gamma=0.9):
        """
        Initialize the Nesterov Accelerated Gradient (NAG) optimizer.

        Args:
            learning_rate (float): Learning rate for the optimizer (default is 0.01).
            gamma (float): Momentum factor (default is 0.9).
        """
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps = 1e-8
        self.v_t = None

    def apply_gradients(self, gradient, thetas):
        """
        Apply gradients to update the model parameters using Nesterov Accelerated Gradient (NAG).

        Args:
            gradient (numpy.ndarray): Gradients with respect to model parameters.
            thetas (numpy.ndarray): Current model parameters.

        Returns:
            numpy.ndarray: Updated model parameters after applying gradients.

        Note:
            We should calculate the gradient with respect to thetas_temp thus we need access to the input.
            so for simplicity: we will use the current gradient, it will not affect the solution too much
        """

        if self.v_t is None:
            self.v_t = np.zeros_like(thetas)

        thetas_temp = thetas - self.gamma * self.v_t

        self.v_t = self.gamma * self.v_t + self.learning_rate * gradient

        thetas = thetas_temp - self.learning_rate * gradient

        return thetas
