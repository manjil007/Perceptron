import numpy as np


class Perceptron:
    def __init__(self, num_features: int, num_classes: int, lr: float):
        self.weights = np.random.uniform(-1, 1, num_features) * 0.01
        self.bias = np.random.uniform(-1, 1) * 0.01
        self.lr = lr

    def forward(self, X):
        """

        :param X:
        :return:
        """
        z = np.dot(X, self.weights)
        z = z + self.bias
        z1 = np.where(z >= 0, 1, -1)
        return z1

    def fit(self, X, y, num_iterations):
        """

        :param X:
        :param y:
        :param num_iterations:
        :return:
        """
        for i in range(num_iterations):
            predictions = self.forward(X)
            error = y - predictions
            self.weights = self.weights + self.lr * np.dot(error.T, X)
            self.bias = self.bias + self.lr * np.sum(error)

        return self.weights, self.bias

    def fit_gd(self, X, y, num_iterations):
        """

        :param X:
        :param y:
        :param num_iterations:
        :return:
        """
        n = len(X)
        for i in range(num_iterations):
            predictions = self.forward(X)
            error = y - predictions
            mse = (1 / n) * np.sum(error ** 2)
            self.weights = self.weights - (-2 * (np.dot(error.T, X))) / n
            self.bias = self.bias - (-2 * np.sum(error))

        return self.weights, self.bias










