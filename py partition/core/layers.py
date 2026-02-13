import numpy as np
from .base import Layer

class LinearLayer(Layer):
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)

    def forward(self, x):
        return x @ self.W + self.b

    def grad_x(self, x, grad_out):
        return grad_out @ self.W.T

    def grad_params(self, x, grad_out):
        dW = x.T @ grad_out
        db = grad_out.sum(axis=0)
        return [dW, db]
