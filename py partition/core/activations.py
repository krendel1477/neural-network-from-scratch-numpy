import numpy as np
from .base import Layer

class ReLU(Layer):
    def forward(self, x):
        return np.maximum(0, x)

    def grad_x(self, x, grad_out):
        return grad_out * (x > 0)

class Sigmoid(Layer):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def grad_x(self, x, grad_out):
        s = self.forward(x)
        return grad_out * s * (1 - s)

class Softmax(Layer):
    def forward(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def grad_x(self, x, grad_out):
        return grad_out
