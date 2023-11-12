import numpy as np
from .base import Optimizer


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999):
        """
        Initialize the Adam optimizer.

        Args:
            learning_rate (float): Learning rate for the optimizer (default is 0.001).
            beta_1 (float): Exponential decay rate for the first moment estimates (default is 0.9).
            beta_2 (float): Exponential decay rate for the second moment estimates (default is 0.999).
        """
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = 1e-8
        self.v_t = self.m_t = None
        self.epoch_passed = True
        self.epoch = None

    def apply_gradients(self, gradient, thetas):
        """
        Apply gradients to update the model parameters using the Adam optimizer.

        Args:
            gradient (numpy.ndarray): Gradients with respect to model parameters.
            thetas (numpy.ndarray): Current model parameters.

        Returns:
            numpy.ndarray: Updated model parameters after applying gradients.
        """

        thetas = thetas.copy()

        if self.epoch is None:
            self.epoch_passed = False
            self.epoch = 0

        if not self.epoch_passed:
            # We will use every batch as an epoch
            self.epoch += 1

        if self.v_t is None and self.m_t is None:
            self.m_t = self.v_t = np.zeros_like(thetas)

        self.m_t = self.beta_1 * self.m_t + (1 - self.beta_1) * gradient
        self.m_t_corrected = self.m_t / (1 - self.beta_1**self.epoch)

        self.v_t = self.beta_2 * self.v_t + (1 - self.beta_2) * (gradient**2)

        self.v_t_corrected = self.v_t / (1 - self.beta_2**self.epoch)

        thetas -= (
            self.learning_rate
            * self.m_t_corrected
            / (np.sqrt(self.v_t_corrected) + self.eps)
        )

        return thetas
