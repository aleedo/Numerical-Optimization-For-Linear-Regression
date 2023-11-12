import numpy as np
import matplotlib.pyplot as plt
from ..optimizers import Adam, Sgd, Momentum, Nag, Adagrad, RMSprop, Bfgs
from ..utils import Colors


class LinearRegression:
    def __init__(
        self,
        batch_size=32,
        max_epochs=1000,
        cost_threshold=0.001,
        gradient_threshold=0.001,
        optimizer="sgd",
        random_state=None,
        normalize=False,
        verbose=True,
    ):
        """
        Initialize the Linear Regression model.

        Args:
            batch_size (int): Batch size for training.
            max_epochs (int): Maximum number of training epochs.
            cost_threshold (float): Threshold for stopping based on cost change.
            gradient_threshold (float): Threshold for stopping based on gradient norm.
            optimizer (str): Name of the optimization algorithm.
            random_state (int or None): Seed for random number generation.
            normalize (bool): Whether to normalize the input features.
            verbose (bool): Whether to print training progress.
        """
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.cost_threshold = cost_threshold
        self.gradient_threshold = gradient_threshold
        self.optimizer = optimizer
        self.random_state = np.random.seed(random_state) if random_state else None
        self.normalize = normalize
        self.verbose = verbose

        self.loss = [np.inf]
        self.thetas_history = []
        self.regression_lines = []

        self.update = {
            "sgd": Sgd(),
            "momentum": Momentum(),
            "nag": Nag(),
            "adagrad": Adagrad(),
            "rmsprop": RMSprop(),
            "adam": Adam(),
            "bfgs": Bfgs(),
        }

        if hasattr(self.optimizer, "learning_rate"):
            name = f"{self.optimizer.__class__.__name__.lower()}"
            self.update[f"{name}"] = self.optimizer
            self.optimizer = f"{name}"

    def __update_thetas(self, gradient, additional_gradient=None):
        """
        Update model parameters (thetas) based on the optimization algorithm.

        Args:
            gradient (numpy.ndarray): Gradient of the cost function.
            additional_gradient (numpy.ndarray or None): Additional gradient information (for BFGS).

        Returns:
            None
        """
        if self.optimizer == "bfgs":
            temp = self.thetas
            self.thetas = self.update[self.optimizer].apply_gradients(
                gradient, self.thetas, additional_gradient
            )
            self.thetas_n_1 = temp
        else:
            self.thetas = self.update[self.optimizer].apply_gradients(
                gradient, self.thetas
            )

    def fit(self, X, y):
        """
        Fit the linear regression model to the provided data.

        Args:
            X (numpy.ndarray): Input features.
            y (numpy.ndarray): Target values.

        Returns:
            LinearRegression: The trained LinearRegression instance.
        """

        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("Input features (X) must be a 2D array.")

        if not isinstance(y, np.ndarray) or y.ndim != 2:
            raise ValueError("Target values (y) must be a 2D array.")

        self.X, self.y = X, y

        if self.normalize:
            self.min_ = np.min(X, axis=0)
            self.range_ = np.ptp(X, axis=0)
            X = (X - self.min_) / self.range_

        X = np.column_stack((np.ones((X.shape[0], 1)), X))
        self.thetas = np.zeros((X.shape[1], 1))

        m = X.shape[0]
        self.batch_size = min(m, self.batch_size)
        self.n_iterations = int(np.ceil(m / self.batch_size))

        data = np.column_stack((X, y))

        self.random_state
        np.random.shuffle(data)

        X_random = data[:, :-1]
        y_random = data[:, -1, np.newaxis]

        if self.optimizer == "bfgs":
            self.thetas_n_1 = self.thetas
            self.thetas = 0.01 * np.ones_like(self.thetas_n_1)

        for epoch in range(self.max_epochs):
            self.epoch = epoch + 1

            if self.optimizer == "adam":
                self.update[self.optimizer].epoch = self.epoch

            for i in range(self.n_iterations):
                self.thetas_history = np.append(self.thetas_history, self.thetas)

                self.X_iter = X_random[
                    i * self.batch_size : i * self.batch_size + self.batch_size, :
                ]
                self.y_iter = y_random[
                    i * self.batch_size : i * self.batch_size + self.batch_size, :
                ]

                h_x = self.X_iter @ self.thetas

                self.error_vec = h_x - self.y_iter
                j = self.error_vec.T @ self.error_vec / (2 * self.batch_size)

                self.loss.append(j[0, 0])

                gradient = self.X_iter.T @ self.error_vec / self.batch_size
                gradient_norm = np.linalg.norm(gradient)

                if self.optimizer == "bfgs":
                    h_x_n_1 = self.X_iter @ self.thetas_n_1
                    error_vec_n_1 = h_x_n_1 - self.y_iter
                    gradient_n_1 = self.X_iter.T @ error_vec_n_1 / self.batch_size
                    self.__update_thetas(gradient, gradient_n_1)
                else:
                    self.__update_thetas(gradient)

            if self.verbose:
                progress_message = (
                    f"{Colors.BOLD}{self.optimizer}:{Colors.GREEN}EPOCH: {epoch:^8}"
                    f"{Colors.RESET}|{Colors.RED}{Colors.BOLD}  COST (BATCH):{j[0, 0]:^14.2f}"
                    f"{Colors.RESET}|{Colors.BLUE}{Colors.BOLD}  R2 (BATCH):{self.evaluate(h_x, self.y_iter):^14.4f}{Colors.RESET}"
                )

                print(progress_message)

            regression_line = X_random @ self.thetas
            self.regression_lines.append(regression_line)

            loss_diff = np.abs(self.loss[-1] - self.loss[-1 * self.n_iterations - 1])
            if loss_diff <= self.cost_threshold:
                break

            if gradient_norm <= self.gradient_threshold:
                break

        self.coef_ = self.thetas[1:]
        self.intercept_ = self.thetas[0]
        self.thetas_history = self.thetas_history.reshape(-1, self.thetas.shape[0])

        return self

    def training_report(self):
        """
        Generate a training report for the linear regression model.

        Returns:
        str: A formatted training report with information on the model's performance.
        """
        report = f"""
******************** Training Report ********************

Linear Regression with {self.optimizer.title()} converged after {self.epoch} epochs

thetas: 
{self.thetas}

Error Vector:
{self.error_vec}

Cost = {self.loss[-1]}

h(x) = y_predict:
{self.predict(self.X)}

y_actual:
{self.y}
    """
        return report

    def predict(self, X):
        """
        Predict target values based on input features.

        Args:
            X (numpy.ndarray): Input features.

        Returns:
            numpy.ndarray: Predicted target values.
        """
        if self.normalize:
            return ((X - self.min_) / self.range_) @ self.coef_ + self.intercept_
        return X @ self.coef_ + self.intercept_

    @staticmethod
    def evaluate(y_true, y_pred):
        """
        Evaluate the model's performance using the R-squared metric.

        Args:
            y_true (numpy.ndarray): True target values.
            y_pred (numpy.ndarray): Predicted target values.

        Returns:
            float: R-squared (coefficient of determination) value.

        R-squared (R2) measures the proportion of the variance in the dependent variable (y_true) that is predictable
        from the independent variable (y_pred).
        An R2 score of 1 indicates a perfect fit, while a score of 0 means the model does not explain the variance at all.
        """
        tss = np.sum((y_true - np.mean(y_true)) ** 2)
        rss = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (rss / (tss + 1e-8))
        return r2

    def plot_loss(self):
        """
        Plot the loss (cost) over training epochs.
        """
        plt.figure(figsize=(6, 4))
        plt.grid()
        plt.title("Loss Over Training Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.plot(self.loss, "mo-")
        plt.show()

    def plot_theta_loss(self):
        """
        Plot the loss for each model parameter (theta) over training iterations.
        """
        num_thetas = self.thetas_history.shape[1]

        num_cols = 2
        num_rows = max(1, (num_thetas + num_cols - 1) // num_cols)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

        for theta_idx in range(num_thetas):
            subplot = axes.flatten()[theta_idx]
            subplot.plot(self.thetas_history[:, theta_idx], self.loss[1:], "-mo")
            subplot.set_title(f"Theta {theta_idx} Loss")
            subplot.set_xlabel("Iterations")
            subplot.set_ylabel("Loss")

        for i in range(num_thetas, num_rows * num_cols):
            fig.delaxes(axes.flatten()[i])

        plt.tight_layout()
        plt.show()

    def plot_regression_lines(self):
        """
        Plot regression lines generated during training.
        """
        plt.figure(figsize=(6, 4))
        plt.grid()
        plt.title("Regression Lines Generated During Training.")
        plt.xlabel("X")
        plt.ylabel("y")
        for regression_line in self.regression_lines:
            plt.plot(regression_line, "r")
        plt.plot(regression_line, "-mo")
        plt.show()

    def plot_best_fit(self):
        """
        Plot the best-fit line alongside the actual data points.
        """
        plt.figure(figsize=(6, 4))
        plt.grid()
        plt.title("Best Fit Line Generated")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.plot(self.y, "-o", self.predict(self.X), "-r")
        plt.legend(("Actual", "Predicted"))
        plt.show()
