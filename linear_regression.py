import numpy as np
import matplotlib.pyplot as plt
np.random.seed(35)


class LinearRegression(object):
    """
    This class provide functionalities to fit a simple linear regression model
    and predict the outcome for an input given.

    Methods:
    fit: Takes input data and target variable.

    predict: Takes test data for which predictions needs to be made.

    Parameters:
    alpha: Learning rate for gradient descent algorithms. Default is 0.0001

    max_iter: Maximum number of iterations to be performed before gradient
    descent stops (termination criteria). Default is 10000.

    ep: Tolerance parameter for cost of fit. Used as a stopping criteria.
    Default value is 0.0001
    """
    def __init__(self, alpha=0.0001, max_iter=10000, ep=0.0001):
        self._weights = None
        self._alpha = alpha
        self._stopping = ep
        self._max_iter = max_iter

    def _hypothesis(self, observation, weights):
        return observation.dot(weights)

    def _cost(self, train_data, train_target, wts):
        sq_err = np.power(self._hypothesis(train_data, wts)-train_target, 2)
        return np.sum(sq_err) / (2 * len(train_data))

    def _gradients(self, train_data, train_target, weights):
        err_vec = self._hypothesis(train_data, weights) - train_target
        in_weights = np.zeros(len(weights))
        for j in xrange(len(weights)):
            in_weights[j] = np.sum(err_vec*train_data[:, j]) / len(train_data)
        return in_weights

    def _gradient_solver(self, train_data, train_target):
        weights = np.random.rand(len(train_data[0])) * 10
        alphas = np.full(len(weights), fill_value=self._alpha)
        J = self._cost(train_data, train_target, weights)
        i = 0
        while i < self._max_iter:
            i += 1
            gr = alphas * self._gradients(train_data, train_target, weights)
            weights -= gr
            e = self._cost(train_data, train_target, weights)
            if abs(J - e) < self._stopping:
                break
            J = e

        return weights

    def _transform_data(self, data):
        data = np.array(data)
        if len(data.shape) == 1:
            data = np.reshape(data, (-1, 1))
        row, col = data.shape
        new_data = np.ones((row, col+1))
        new_data[:, 1:] = data
        return new_data

    def fit(self, train_data, train_target):
        """
        Given training data and targets, this function will compute
        the parameters for a best fit linear model.

        Arguments:
        train_data: Data set, can be python list, numpy array.

        train_target: Labels for the corresponding data.
        """
        train_data = self._transform_data(train_data)
        self._weights = self._gradient_solver(train_data, train_target)

    def predict(self, test_data):
        """
        Function to predict the values given test data.

        Arguments:
        test_data: Data for which target needs to be predicted.
        """
        test_data = self._transform_data(test_data)
        return self._hypothesis(test_data, self._weights)

    def model_param(self):
        """
        This will return the model learning parameters.
        """
        return self._weights

    def residuals(self, test_data, test_target):
        """
        This function will calculate the cost of fitting
        linear model on given data. Cost is calculated as
        the average of error-squired.

        Arguments:
        test_data: Data for model validation.
        test_target: Correct labels for corresponding data.
        """
        predicted_targets = self.predict(test_data)
        errors = predicted_targets - test_target
        return np.sum(np.power(errors, 2)) / (2 * len(test_data))


# # Uncomment below lines for testing purpose
# def main():
#     x = np.array([np.random.randint(1, 100) for i in xrange(20)])
#     y = x * 2 + np.random.rand()*93
#     model = LinearRegression(alpha=0.0001, max_iter=100000, ep=0.000001)
#     model.fit(x, y)
#     print model.predict(x)
#     print model.residuals(x, y)
#     print model.model_param()

#     ###################################################
#     # Visual verification
#     ###################################################
#     model.fit(x, y)
#     plt.scatter(x, y)
#     plt.plot(x, model.predict(x), color='red')
#     plt.show()

# main()
