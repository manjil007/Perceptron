import numpy as np


class Perceptron:
    def __init__(self, num_features: int, num_classes: int, lr: float):
        self.weights = np.random.uniform(-1, 1, (num_classes, num_features)) * 0.01
        self.bias = np.random.uniform(-1, 1, num_classes) * 0.01
        self.lr = lr

    def forward(self, X):
        z = np.dot(X, self.weights.T) + self.bias
        return np.where(z >= 0, 1, -1)

    def fit(self, X, y, num_iterations):
        for i in range(num_iterations):
            predictions = self.forward(X)
            for j in range(X.shape[0]):
                self.weights = self.weights + self.lr * (y[j] - predictions[j]) * X[j]
                self.bias = self.bias + self.lr * (y[j] - predictions[j])

        return self.weights, self.bias











    # def fit(self, X):







