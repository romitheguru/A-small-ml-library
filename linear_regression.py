""" Linear Regression Model """

import numpy as np
from solvers import gradient_descent


class LinearRegression(object):
    """
    This class provide functionalities to fit a simple linear regression model
    and predict the outcome for an input given.

    Methods
    -------
    fit: Takes input data and target variable.

    predict: Takes test data for which predictions needs to be made.

    Parameters
    ----------
    alpha: float, optional
        Learning rate for gradient descent algorithms. Default is 0.0001

    max_iter: int, optional
        Maximum number of iterations to be performed before gradient
        descent stops (termination criteria). Default is 100000.

    ep: float, optional
        Tolerance parameter for cost of fit. Used as a stopping criteria.
        Default value is 0.0001
    """

    def __init__(self, alpha=0.01, max_iter=100000, ep=0.00001):
        self._weights = None
        self._alpha = alpha
        self._stopping = ep
        self._max_iter = max_iter

    def __h(self, X, weights):
        return X.dot(weights)

    def __cost(self, X, y, weights):
        m = X.shape[0]
        h = self.__h(X, weights)
        return np.sum((h - y) ** 2) / (2 * m)

    def __add_intercept(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape((-1, 1))
        ones = np.ones((X.shape[0], 1))
        return np.concatenate((ones, X), axis=1)

    def fit(self, X, y):
        """
        Given training data and targets, this function will compute
        the parameters for a best fit linear model.

        Arguments:
        X: Data set, can be python list, numpy array.

        y: Labels for the corresponding data.
        """
        X = self.__add_intercept(X)
        y = np.array(y)
        self._weights = gradient_descent(X, y, self._alpha, self._max_iter,
                                         self._stopping, self.__cost, self.__h)

    def predict(self, X):
        """
        Function to predict the values given test data.

        Arguments:
        X_test: Data for which target needs to be predicted.
        """
        assert self._weights is not None, "Model needs to be trained"
        X = self.__add_intercept(X)
        return self.__h(X, self._weights)
