## Linear Regression
Linear regression is a fundamental statistical method used for modeling the relationship between a dependent variable (target) and one or more independent variables (features). The goal is to find the best-fit linear equation that describes this relationship.


### Numerical Optimization
Optimizing the parameters of a linear regression model involves finding the values that minimize a cost function. We seek the optimal parameters by iteratively adjusting them. These methods are based on derivatives of the cost function with respect to the model parameters. 

There are two common types of derivatives used in optimization:

- **First-Order Derivatives**: These are gradients, which provide information about the slope of the cost function. Optimizers like Stochastic Gradient Descent (SGD), Momentum, Nesterov Accelerated Gradient (NAG), Adagrad, RMSprop, and Adam use first-order derivatives.

- **Second-Order Derivatives**: These include the Hessian matrix, which provides information about the curvature of the cost function. The BFGS (Broyden-Fletcher-Goldfarb-Shanno) optimizer estimates the Hessian and utilizes second-order derivatives.


### Supported Optimizers
The choice of optimizer can impact the model's convergence speed and overall performance.

- **Stochastic Gradient Descent (SGD)**: A classic optimization algorithm that iteratively updates model parameters to minimize the cost function. It offers a customizable learning rate, making it suitable for various datasets and problems.

- **Momentum**: Builds upon SGD by adding a momentum term, which accelerates convergence by damping oscillations.

- **Nesterov Accelerated Gradient (NAG)**: A variant of Momentum that provides better convergence behavior by estimating the gradient at a "lookahead" position.

- **Adagrad**: An adaptive learning rate method that scales the learning rate for each parameter based on past gradients. It tends to perform well with sparse data.

- **RMSprop**: An improvement over Adagrad, RMSprop adapts the learning rate with a moving average of squared gradients, providing better stability and faster convergence.

- **Adam**: Combines Momentum and RMSprop, offering adaptive learning rates and momentum. It's known for fast convergence and robust performance across various datasets.

- **BFGS (Broyden-Fletcher-Goldfarb-Shanno)**: A quasi-Newton optimization algorithm that estimates the Hessian matrix. It provides precise convergence and is effective for smooth, convex cost functions.