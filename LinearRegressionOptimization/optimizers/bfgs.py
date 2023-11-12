import numpy as np
from .base import Optimizer


class Bfgs(Optimizer):
    def __init__(self, learning_rate=1):
        """
        Initialize the BFGS (Broyden-Fletcher-Goldfarb-Shanno) optimizer.

        Args:
            learning_rate (float): Learning rate for the optimizer (default is 1).
        """
        self.learning_rate = learning_rate
        self.eps = 1e-8
        self.thetas_n_1 = None

    def apply_gradients(self, gradient, thetas, additional_gradient):
        """
        Apply gradients to update the model parameters using the BFGS optimizer.

        Args:
            gradient (numpy.ndarray): Gradients with respect to model parameters.
            thetas (numpy.ndarray): Current model parameters.
            additional_gradient (numpy.ndarray): Additional gradient information for the previous model parameters.

        Returns:
            numpy.ndarray: Updated model parameters after applying gradients.
        """
        if self.thetas_n_1 is None:
            self.thetas_n_1 = np.zeros_like(thetas)
            self.I = self.B_inv_n_1 = np.eye(thetas.shape[0])

        self.thetas = thetas

        d_thetas = self.thetas - self.thetas_n_1
        d_grad = gradient - additional_gradient

        B_inv_first_term = (
            (self.I - ((d_thetas @ d_grad.T) / (d_grad.T @ d_thetas)))
            @ self.B_inv_n_1
            @ (self.I - (d_grad @ d_thetas.T) / (d_grad.T @ d_thetas))
        )

        B_inv_second_term = (d_thetas @ d_thetas.T) / (d_grad.T @ d_thetas)

        self.B_inv = B_inv_first_term + B_inv_second_term

        self.thetas_n_1 = self.thetas
        self.thetas = self.thetas - self.B_inv @ gradient * self.learning_rate

        self.B_inv_n_1 = self.B_inv

        return self.thetas
