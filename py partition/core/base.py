import numpy as np

def _as_2d(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    return x

class Layer:
    def forward(self, x):
        raise NotImplementedError

    def grad_x(self, x, grad_out):
        raise NotImplementedError

    def grad_params(self, x, grad_out):
        return []

    def num_params(self):
        return 0
