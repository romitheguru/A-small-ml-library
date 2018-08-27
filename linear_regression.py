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

    def __init__(self, alpha=0.0001, max_iter=100000, ep=0.0001):
        self._weights = None
        self._alpha = alpha
        self._stopping = ep
        self._max_iter = max_iter

    def _cost(self, X, y, weights):
        m = len(X)
        return np.sum((X.dot(weights) - y) ** 2) / (2 * m)

    def _transform_data(self, data):
        data = np.array(data)
        if len(data.shape) == 1:
            data = data.reshape((-1, 1))
        ones = np.ones((len(data), 1))
        return np.concatenate((ones, data), axis=1)

    def fit(self, X, y):
        """
        Given training data and targets, this function will compute
        the parameters for a best fit linear model.

        Arguments:
        X: Data set, can be python list, numpy array.

        y: Labels for the corresponding data.
        """
        X = self._transform_data(X)
        y = np.array(y)
        self._weights = gradient_descent(X, y, self._alpha, self._max_iter,
                                         self._stopping, self._cost)

    def predict(self, X_test):
        """
        Function to predict the values given test data.

        Arguments:
        X_test: Data for which target needs to be predicted.
        """
        assert self._weights is not None, "Model needs to be trained"
        X_test = self._transform_data(X_test)
        return X_test.dot(self._weights)

    def model_param(self):
        """
        This will return the model learning parameters.
        """
        assert self._weights is not None, "Model needs to be trained"
        return self._weights

    def residuals(self, X, y):
        """
        This function will calculate the cost of fitting
        linear model on given data. Cost is calculated as
        the average of error-squired.

        Arguments:
        X: Data for model validation.
        y: Correct labels for corresponding data.
        """
        assert self._weights is not None, "Model needs to be trained"
        predicted_y = self.predict(X)
        errors = predicted_y - y
        return np.sum(np.power(errors, 2)) / (2 * len(X))
