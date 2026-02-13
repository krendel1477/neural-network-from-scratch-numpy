import numpy as np

class CrossEntropy:
    def forward(self, y_pred, y_true):
        eps = 1e-12
        return -np.sum(y_true * np.log(y_pred + eps)) / y_pred.shape[0]

    def grad(self, y_pred, y_true):
        return (y_pred - y_true) / y_pred.shape[0]
