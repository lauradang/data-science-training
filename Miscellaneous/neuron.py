import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, input):
        sum = np.dot(input, self.weights) + self.bias
        return sigmoid(sum)

class NeuralNetwork:
    """
    - 2 inputs
    - 1 hidden layer
    - 1 output layer (1 neuron in output layer)

    - weights and bias are the same for each neuron
    """
    def __init__(self):
        self.weights = [0, 1]
        self.bias = 0
        
        self.hidden1 = Neuron(self.weights, self.bias)
        self.hidden2 = Neuron(self.weights, self.bias)
        self.output1 = Neuron(self.weights, self.bias)

    def feedforward(self, input): # remember input is an array
        hidden1_result = self.hidden1.feedforward(input)
        hidden2_result = self.hidden2.feedforward(input)

        output1_result = self.output1.feedforward([hidden1_result, hidden2_result])
        return output1_result

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def mse_loss(pred, actual):
    return np.mean((pred - actual) ** 2)

# Instantiating Neuron class
x_in = [2, 3]
weights = [0, 1]
bias = 4

neuron = Neuron(weights, bias)
print(neuron.feedforward(x_in))

# Instantiating NeuralNetwork class
neural_network = NeuralNetwork()
print(neural_network.feedforward(x_in))

# Testing mse_loss function
print(mse_loss(np.array([1, 0, 0, 1]), np.array([0, 0, 0, 0])))