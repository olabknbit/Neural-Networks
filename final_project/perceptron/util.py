import numpy as np
from numpy import exp


def linear(activation):
    return activation


def linear_derivative(_):
    return 1


def reLu(activation):
    return max(0, activation)


def reLu_derivative(output):
    return 0 if output < 0 else 1


def _tanh(x):
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


def write_network_to_file_classification(filename, neural_network):
    with open(filename, 'w') as file:
        file.write("%s\n" % neural_network.output_classes)
        file.writelines(["%s\n" % l for l in neural_network.get_weights()])


def write_network_to_file_regression(filename, neural_network):
    with open(filename, 'w') as file:
        file.write(neural_network.to_str())


def read_network_layers_from_file_classification(filename):
    with open(filename, 'r') as file:
        rows = file.readlines()
        layers = [eval(row) for row in rows]
        output_classes = None
        if rows[0].strip() is not None and rows[0] != '\n':
            output_classes = eval(rows[0])

        return layers, output_classes


def read_network_layers_from_file_regression(filename):
    """
    Layer is a tuple of network neurons and bias_weights
    :param filename:
    :return:
    """
    with open(filename, 'r') as file:
        rows = file.readlines()
        # layers = [print(neurons, bias_weights) for layer in rows:for neurons, bias_weights in eval(layer)]

        layers = []
        for row in rows:
            layer = eval(row)
            print(layer)
            neurons = eval(layer['neurons'])
            bias_weights = eval(layer['bias_weights'])
            layers.append((neurons, bias_weights))

        return layers


def get_random_biases_weights(n_inputs):
    import random
    return [random.random() * 0.3 for _ in range(n_inputs)]


def get_random_neurons(n_inputs, n_neurons):
    from numpy import random
    # random numbers from range [0; 0.3) are proven to be best
    return [{'weights': [random.random() * 0.3 for _ in range(n_inputs)],
             'active_weights': [True for _ in range(n_inputs)]}
            for _ in range(n_neurons)]


def initialize_network(n_inputs, activation_f, activation_f_derivative):
    from nnr_new import NeuralNetwork, Neuron
    from numpy import random
    id = 0
    neurons = []
    input_neurons = []
    in_ns = {}
    for _ in range(n_inputs):
        neuron = Neuron(id, level=0, in_ns={}, out_ns=[], bias_weight=0, activation_f=activation_f, activation_f_derivative=activation_f_derivative)
        neurons.append(neuron)
        input_neurons.append(neuron)
        in_ns[neuron] = random.random() * 0.3
        id += 1

    output_neuron = Neuron(id, 1, in_ns, [], 0.3, activation_f, activation_f_derivative)
    for in_n in input_neurons:
        in_n.out_ns.append(output_neuron)
    neurons.append(output_neuron)

    return NeuralNetwork(neurons, input_neurons, output_neuron, activation_f, activation_f_derivative)


def get_activation_f_and_f_d_by_name(activation_f_name):
    if activation_f_name == 'tanh':
        activation_f, activation_f_d = _tanh, tanh_derivative
    elif activation_f_name == 'sigmoid':
        activation_f, activation_f_d = sigmoid, sigmoid_derivative
    else:
        print('Error: Do not have %s activation f implemented in util.py' % activation_f_name)
        exit(1)
    return activation_f, activation_f_d


def read_network_from_file(filename, activation_f, activation_f_derivative):
    from neural_network_regression import NeuronLayer, NeuralNetwork
    layers = read_network_layers_from_file_regression(filename)
    neural_network = NeuralNetwork([NeuronLayer(neurons, bias_weights) for neurons, bias_weights in layers], activation_f, activation_f_derivative)
    return neural_network


def read_network_from_file_f_name(filename, activation_f_name):
    activation_f, activation_f_d = get_activation_f_and_f_d_by_name(activation_f_name)
    return read_network_from_file(filename, activation_f, activation_f_d)


def initialize_start_network(n_inputs, activation='tanh'):
    activation_f, activation_f_d = get_activation_f_and_f_d_by_name(activation)
    return initialize_network(n_inputs, activation_f, activation_f_d)


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


def plot_regression_data(X_train, y_train, X_test, y_test, y_predicted, y_sklearn_predicted=None,
                         savefig_filename=None):
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
