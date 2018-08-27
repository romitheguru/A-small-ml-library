# Author: Romee
"""
Various optimization algorithms to find best parameters
"""

import numpy as np


def gradient_descent(X, y, alpha, max_iter, stopping, cost_func):
    """
    Computer parameters with minnimum cost

    Implementation of gradient descent algorithm.

    Parameters
    ----------
    X : matrix
        Input sample

    y : array like
        Target variable for supervised learning

    alpha : float
        Learning rate for gradient descent

    max_iter : int
        Maximum iteration of gradient steps

    stopping : float
        Stopping criteria i.e epsilon value

    cost_func, function
        Function to evaluate cost or loss function

    Returns
    -------
    weights : array like
        Best parameters that minimize loss
    """

    # Initialize weights with zeros
    weights = np.zeros(X.shape[1])
    m = len(X)
    # Calculate initial cost
    J = cost_func(X, y, weights)

    for i in range(max_iter):
        # Difference b/w estimate and actual value
        loss = y - X.dot(weights)
        # Update weights
        updated_weights = weights + (alpha / m) * X.T.dot(loss)

        # If converging, update the original weights
        e = cost_func(X, y, updated_weights)
        if J > e:
            weights = updated_weights
            # Increase learning rate by 1% for faster learning
            alpha = alpha * 1.01
        # If diverging, reduce learning rate by 50%
        else:
            alpha = alpha / 2
        # If coverged, stop the loop
        if abs(J - e) < stopping:
            break
        J = e

    return weights
