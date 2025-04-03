import numpy as np

def cost_function( x, y, theta0, theta1 ):
    """Compute the squared error cost function

    Inputs:
    x        vector of length m containing x values
    y        vector of length m containing y values
    theta_0  (scalar) intercept parameter
    theta_1  (scalar) slope parameter

    Returns:
    cost     (scalar) the cost
    """

    cost = 0.0
    num_of_training_samples = len(x)

    hypothesis = theta0 + theta1 * x
    squared_errors = (hypothesis - y) ** 2
    cost = (1 / 2) * np.sum(squared_errors) 

    return cost


def gradient(x, y, theta_0, theta_1):
    """Compute the partial derivative of the squared error cost function

    Inputs:
    x          vector of length m containing x values
    y          vector of length m containing y values
    theta_0    (scalar) intercept parameter
    theta_1    (scalar) slope parameter

    Returns:
    d_theta_0  (scalar) Partial derivative of cost function wrt theta_0
    d_theta_1  (scalar) Partial derivative of cost function wrt theta_1
    """

    d_theta_0 = 0.0
    d_theta_1 = 0.0

    num_of_training_samples = len(x)
    hypothesis = theta_0 + theta_1 * x
    errors = hypothesis - y

    d_theta_0 = (1 / num_of_training_samples) * np.sum(errors)      # partial derivative with respect to theta_0
    d_theta_1 = (1 / num_of_training_samples) * np.sum(errors * x)  # partial derivative with respect to theta_1

    return d_theta_0, d_theta_1 # return is a tuple

