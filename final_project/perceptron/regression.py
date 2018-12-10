from neural_network import NeuralNetwork, NeuronLayer


def get_random_neurons(n_inputs, n_neurons):
    from numpy import random
    # random numbers from range [0; 0.3) are proven to be best
    return [{'weights': [random.random() * 0.3 for _ in range(n_inputs)]} for _ in range(n_neurons)]


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


# Return number of features - n_inputs for NN.
def get_n_inputs_outputs(X_data):
    return len(X_data[0])


def plot_data(X_train, y_train, X_test, y_test, predicted_outputs, visualize, savefig_filename):
    import matplotlib.pyplot as plt
    marker = '.'
    plt.clf()

    plt.plot(X_test, y_test, marker, c='red', label='test_data')
    plt.plot(X_test, predicted_outputs, marker, c='blue', label='predicted')
    plt.plot(X_train, y_train, marker, c='green', label='training_data')

    plt.legend()
    if visualize:
        plt.show()
    if savefig_filename:
        plt.savefig(savefig_filename)


def print_data(y_test, predicted_outputs):
    Error = 0.0
    SE = 0.0
    RAEDivider = 0.0
    RRAEDivider = 0.0
    AvgData = sum(predicted_outputs) / len(predicted_outputs)

    for y, pred in zip(y_test, predicted_outputs):
        Error += abs(pred - y)
        SE += (pred - y) ** 2
        RAEDivider += abs(AvgData - y)
        RRAEDivider += (AvgData - y) ** 2

    AvgError = 1.0 * Error / (1.0 * len(y_test))
    MSE = 1.0 * SE / (1.0 * len(y_test))
    RMSE = MSE ** 0.5
    RAE = Error / RAEDivider
    RRAE = SE / RRAEDivider

    print('Error = %.3f' % Error)
    print('AvgError = %.3f' % AvgError)
    print('MSE = %.3f' % MSE)
    print('RMSE = %.3f' % RMSE)
    print('RAE = %.3f' % RAE)
    print('RRAE = %.3f' % RRAE)


def initialize_network(neurons, n_inputs, biases, activation_f, activation_f_derivative):
    # Combine the layers to create a neural network
    layers = []
    n_in = n_inputs
    bias = 1 if biases else 0
    for n_neurons in neurons:
        layers.append(NeuronLayer(get_random_neurons(n_in + bias, n_neurons)))
        n_in = n_neurons
    layers.append(NeuronLayer(get_random_neurons(n_in + bias, 1)))

    return NeuralNetwork(layers, activation_f, activation_f_derivative)


# TODO fix scale data
def scale_data(train_set_inputs, test_set_inputs):
    y_train = [y for x, y in train_set_inputs]
    y_test = [y for x, y in test_set_inputs]
    min_y = min(y_train + y_test)
    max_y = max(y_train + y_test)
    train_set_inputs = [(x, (y - min_y) / (max_y - min_y)) for x,y in train_set_inputs]
    test_set_inputs = [(x,(y - min_y) / (max_y - min_y)) for x,y in test_set_inputs]

    return train_set_inputs, test_set_inputs


def split_data(data_set):
    X_set = []
    y_set = []
    for row in data_set:
        X_set.append(row[:-1])
        y_set.append(row[-1])
    return X_set, y_set


def get_split_dataset(train_filename, test_filename):
    train_set_inputs, test_set_inputs = read_file(train_filename), read_file(test_filename)
    train_set_inputs, test_set_inputs = scale_data(train_set_inputs, test_set_inputs)
    X_train, y_train = split_data(train_set_inputs)
    X_test, y_test = split_data(test_set_inputs)
    return X_train, y_train, X_test, y_test


def main(train_filename, test_filename, create_nn, save_nn, read_nn, number_of_epochs, visualize_every, l_rate, biases,
         savefig_filename, activation_f, activation_f_derivative):
    from util import read_network_layers_from_file, write_network_to_file
    if train_filename is None or test_filename is None:
        print('Both train and test filename has to be provided for scaling')
        exit(1)

    X_train, y_train, X_test, y_test = get_split_dataset(train_filename, test_filename)

    if create_nn is not None:
        # Calculate the number of inputs and outputs from the data.
        n_inputs = get_n_inputs_outputs(X_train)
        neural_network = initialize_network(create_nn, n_inputs, biases, activation_f, activation_f_derivative)
    else:
        layers, _ = read_network_layers_from_file(read_nn)
        neural_network = NeuralNetwork([NeuronLayer(l) for l in layers], activation_f, activation_f_derivative)

    # Train neural network.
    from train import train
    train(neural_network, X_train, y_train, l_rate, number_of_epochs, visualize_every)

    if save_nn is not None:
        write_network_to_file(save_nn, neural_network)

    if neural_network is None:
        layers, _ = read_network_layers_from_file(read_nn)
        neural_network = NeuralNetwork([NeuronLayer(l) for l in layers], activation_f, activation_f_derivative)

    # Test the neural network.
    accuracy, predicted_outputs = neural_network.test(X_test, y_test)

    if len(X_train[0]) == 1 and (visualize_every is not None or savefig_filename is not None):
        plot_data(X_train, y_train, X_test, y_test, predicted_outputs, visualize_every, savefig_filename)
    return accuracy
