class NeuralNetwork:
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

    def forward(self, x):
        activations = [x]
        for layer in self.layers:
            x = layer.forward(x)
            activations.append(x)
        return activations

    def backward(self, activations, y_true):
        grads = []
        grad = self.loss.grad(activations[-1], y_true)

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            x = activations[i]
            grads.insert(0, layer.grad_params(x, grad))
            grad = layer.grad_x(x, grad)

        return grads

    def step(self, grads, lr):
        for layer, layer_grads in zip(self.layers, grads):
            if layer_grads:
                layer.W -= lr * layer_grads[0]
                layer.b -= lr * layer_grads[1]
