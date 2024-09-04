import numpy as np


class Perceptron:
    def __init__(self, num_features: int, num_classes: int, lr: float):
        self.weights = np.random.uniform(-1, 1, num_features) * 0.01
        self.bias = np.random.uniform(-1, 1) * 0.01
        self.lr = lr

    def forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        z1 = np.where(z >= 0, 1, -1)
        return z1

    def fit(self, X, y, num_iterations):
        for i in range(num_iterations):
            predictions = self.forward(X)
            error = y - predictions
            self.weights = self.weights + self.lr * np.dot(error.T, X)
            self.bias = self.bias + self.lr * np.sum(error)

        return self.weights, self.bias










