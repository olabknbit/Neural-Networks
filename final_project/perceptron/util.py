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
    return np.tanh(x) + 1


def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2


# Sigmoid transfer function
def sigmoid(x):
    if -x > np.log(np.finfo(type(x)).max):
        return 0.0
    a = np.exp(-x)
    return 1.0 / (1.0 + a)


# Derivative of transfer function
def sigmoid_derivative(x):
    return exp(-x) / (exp(-x) + 1) ** 2


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


def get_random_neurons(n_inputs, n_neurons):
    from numpy import random
    # random numbers from range [0; 0.3) are proven to be best
    return [{'weights': [random.random() * 0.3 for _ in range(n_inputs)]} for _ in range(n_neurons)]


def initialize_network(neurons, n_inputs, biases, activation_f, activation_f_derivative):
    from neural_network_regression import NeuronLayer, NeuralNetwork
    # Combine the layers to create a neural network
    layers = []
    n_in = n_inputs
    bias = 1 if biases else 0
    for n_neurons in neurons:
        layers.append(NeuronLayer(get_random_neurons(n_in + bias, n_neurons)))
        n_in = n_neurons
    layers.append(NeuronLayer(get_random_neurons(n_in + bias, 1)))

    return NeuralNetwork(layers, activation_f, activation_f_derivative)


def get_activation_f_and_f_d_by_name(activation_f_name):
    if activation_f_name == 'tanh':
        activation_f, activation_f_d = tanh, tanh_derivative
    elif activation_f_name == 'sigmoid':
        activation_f, activation_f_d = sigmoid, sigmoid_derivative
    else:
        print('Error: Do not have %s activation f implemented in util.py' % activation_f_name)
        exit(1)
    return activation_f, activation_f_d


def read_network_from_file(filename, activation_f, activation_f_derivative):
    from neural_network_regression import NeuronLayer, NeuralNetwork
    layers, _ = read_network_layers_from_file(filename)
    neural_network = NeuralNetwork([NeuronLayer(l) for l in layers], activation_f, activation_f_derivative)
    return neural_network


def read_network_from_file_f_name(filename, activation_f_name):
    activation_f, activation_f_d = get_activation_f_and_f_d_by_name(activation_f_name)
    return read_network_from_file(filename, activation_f, activation_f_d)


def initialize_random_network(n_inputs, n_hidden_layers=(1, 5), n_neurons=(5, 100), activation='tanh', biases=True,
                              random_state=1):
    import random
    activation_f, activation_f_d = get_activation_f_and_f_d_by_name(activation)
    n_layers = random.randint(*n_hidden_layers)
    neurons = [random.randint(*n_neurons) for _ in range(n_layers)]
    return initialize_network(neurons, n_inputs, biases, activation_f, activation_f_d)


def scale_data(y_train, y_test):
    min_y = min(y_train + y_test)
    max_y = max(y_train + y_test)

    y_train = [(y - min_y) / (max_y - min_y) for y in y_train]
    y_test = [(y - min_y) / (max_y - min_y) for y in y_test]

    return y_train, y_test


def split_data(data_set):
    X_set = []
    y_set = []
    for row in data_set:
        X_set.append(row[:-1])
        y_set.append(row[-1])
    return X_set, y_set


def read_file(filename):
    import csv
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        rows = []
        first = True
        for row in csv_reader:
            if first:
                first = False
            else:
                row = [float(x) for x in row[:-1]] + [float(row[-1])]
                rows.append(row)
        return rows


def get_split_dataset(train_filename, test_filename):
    train_set_inputs, test_set_inputs = read_file(train_filename), read_file(test_filename)
    X_train, y_train = split_data(train_set_inputs)
    X_test, y_test = split_data(test_set_inputs)
    y_train, y_test = scale_data(y_train, y_test)
    return X_train, y_train, X_test, y_test


def plot_regression_data(X_train, y_train, X_test, y_test, y_predicted, y_sklearn_predicted=None, savefig_filename=None):
    if len(X_train[0]) != 1:
        print('Cannot plot because len(data) = %d > 1' % len(X_train[0]))
        return
    if savefig_filename is None:
        return
    import matplotlib.pyplot as plt
    marker = '.'
    plt.clf()

    plt.plot(X_test, y_test, marker, c='red', label='test_data')
    plt.plot(X_test, y_predicted, marker, c='blue', label='predicted')
    plt.plot(X_train, y_train, marker, c='green', label='training_data')
    if y_sklearn_predicted is not None:
        plt.plot(X_test, y_sklearn_predicted, marker, c='purple', label='sklearn_predicted')

    plt.legend()
    plt.savefig(savefig_filename)
