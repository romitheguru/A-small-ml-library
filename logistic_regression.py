""" Logistic Regression Model """

import numpy as np
from solvers import gradient_descent


class LogisticRegression(object):
    """
    This class provide functions to fit a logistic regression model
    and predict the value given the input.

    Methods
    -------
    fit: Takes input data and target variable.

    predict: Takes input data for which predictions needs to be made.

    predict_proba: Take input data and returns predicted probabilities

    Parameters
    ----------
    alpha: float, optional
        Learning rate for gradient descent algorithms. Default is 0.0001

    max__iter: int, optional
        Maximum number of iterations to be performed before gradient
        descent stops (termination criteria). Default is 100000.

    ep: float, optional
        Tolerance parameter for cost of fit. Used as a stopping criteria.
        Default value is 0.0001
    """

    def __init__(self, alpha=0.01, max_iter=100000, ep=1e-7):
        self.__weights = None
        self.__alpha = alpha
        self.__stopping = ep
        self.__max_iter = max_iter

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __h(self, X, weights):
        return self.__sigmoid(X.dot(weights))

    def __cost(self, X, y, weights):
        h = self.__h(X, weights)
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

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

        Parameters
        ----------
        X: Data set, can be python list, numpy array.

        y: Labels for the corresponding data.
        """
        X = self.__add_intercept(X)
        y = np.array(y)
        self.__weights = gradient_descent(X, y,
                                          self.__alpha,
                                          self.__max_iter,
                                          self.__stopping,
                                          self.__cost,
                                          self.__h)

    def predict_proba(self, X):
        """
        Function to predict the class probabilities for given test data.

        Parameters
        ----------
        X: array-like
            Data for which target needs to be predicted.

        Returns
        -------
        Predicted class probailities
        """
        assert self.__weights is not None, "Model needs to be trained"
        X = self.__add_intercept(X)
        return self.__h(X, self.__weights)

    def predict(self, X, threshold=0.5):
        """
        Function to predict the classes for given test data.

        Parameters
        ----------
        X: array-like
            Data for which target needs to be predicted.

        Returns
        -------
        Predicted class values
        """
        probs = self.predict_proba(X)
        return np.where(probs >= threshold, 1, 0)
