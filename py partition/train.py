from core.layers import LinearLayer
from core.activations import ReLU, Softmax
from core.loss import CrossEntropy
from core.network import NeuralNetwork
from utils.data import load_mnist, preprocess, one_hot

import numpy as np

(x_train, y_train), (x_test, y_test) = load_mnist()

x_train = preprocess(x_train)
y_train = one_hot(y_train, 10)

model = NeuralNetwork(
    layers=[
        LinearLayer(784, 128),
        ReLU(),
        LinearLayer(128, 10),
        Softmax()
    ],
    loss=CrossEntropy()
)

print("Project structure ready. Implement training loop in train.py")
