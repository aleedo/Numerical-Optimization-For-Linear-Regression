from LinearRegressionOptimization import LinearRegression
from LinearRegressionOptimization.input import get_from_csv, train_test_split
from LinearRegressionOptimization.optimizers import *


def main(plots):
    X, y = get_from_csv()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_ratio=0.1, random_state=42
    )

    full_batch_gd = LinearRegression(
        batch_size=X_train.shape[0],
        optimizer=Sgd(learning_rate=0.1),
        random_state=42,
        normalize=True,
    )

    full_batch_gd.fit(X_train, y_train)
    print(full_batch_gd.training_report())

    train_score = full_batch_gd.evaluate(y_train, full_batch_gd.predict(X_train))
    test_score = full_batch_gd.evaluate(y_test, full_batch_gd.predict(X_test))
    print(f"Train R^2 Score: {train_score}")
    print(f"Test R^2 Score: {test_score}")

    ### Visualize Training Progress
    if plots:
        full_batch_gd.plot_theta_loss()
        full_batch_gd.plot_best_fit()
        full_batch_gd.plot_loss()
        full_batch_gd.plot_regression_lines()


if __name__ == "__main__":
    main(plots=False)
