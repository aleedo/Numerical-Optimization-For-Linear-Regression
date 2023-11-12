from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply_gradients(self, gradient, thetas, additional_gradient=None):
        pass
