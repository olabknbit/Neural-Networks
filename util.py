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


# Sigmoid transfer function
def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))
    # return activation * 0.3
    # return 1.0*(exp(activation)-exp(-activation))/(0.0+exp(activation)+exp(-activation))


# TODO nie tangens hiperboliczny do porownania (inna funkcja)
# Derivative of transfer function
def sigmoid_derivative(output):
    return output * (1.0 - output)
