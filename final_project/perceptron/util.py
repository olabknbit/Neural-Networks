import numpy as np
from numpy import exp


# Activation function - for weights and inputs returns their dot product.
def activate(weights, inputs):
    # Add bias.
    activation = weights[-1]
    for weight, input in zip(weights[:-1], inputs):
        activation += weight * input
    return activation


def linear(activation):
    return activation


def linear_derivative(_):
    return 1


def reLu(activation):
    return max(0, activation)


def reLu_derivative(output):
    return 0 if output < 0 else 1


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2


# Sigmoid transfer function
def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))


# Derivative of transfer function
def sigmoid_derivative(output):
    return output * (1.0 - output)


def write_network_to_file(filename, neural_network):
    with open(filename, 'w') as file:
        if hasattr(neural_network, 'output_classes'):
            file.write("%s\n" % neural_network.output_classes)
        else:
            file.write("\n")
        file.writelines(["%s\n" % l for l in neural_network.get_weights()])


def read_network_layers_from_file(filename):
    with open(filename, 'r') as file:
        rows = file.readlines()
        layers = [eval(row) for row in rows[1:]]
        output_classes = None
        if rows[0].strip() is not None and rows[0] != '\n':
            output_classes = eval(rows[0])

        return layers, output_classes
