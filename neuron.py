import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def get_output(self, input):
        sum = np.dot(input, self.weights) + self.bias
        return sigmoid(sum)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_in = [2, 3]
weights = [0, 1]
bias = 4

neuron = Neuron(weights, bias)
print(neuron.get_output(x_in))