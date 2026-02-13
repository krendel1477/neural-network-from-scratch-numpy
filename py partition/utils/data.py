import numpy as np

def load_mnist(path="mnist.npz"):
    with np.load(path) as f:
        return (f["x_train"], f["y_train"]), (f["x_test"], f["y_test"])

def preprocess(x):
    return x.reshape((x.shape[0], -1)).astype(float) / 255.0

def one_hot(y, num_classes):
    oh = np.zeros((len(y), num_classes))
    oh[np.arange(len(y)), y] = 1
    return oh
