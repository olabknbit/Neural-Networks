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
def get_n_inputs_outputs(data):
    n_inputs = len(data[0]) - 1
    return n_inputs


def plot_data(test_data, predicted_outputs, visualize, savefig_filename, training_data=None):
    import matplotlib.pyplot as plt
    marker = '.'
    plt.clf()

    test_data_x = [t[0] for t in test_data]
    test_data_y = [t[1] for t in test_data]
    plt.plot(test_data_x, test_data_y, marker, c='red', label='test_data')
    plt.plot(test_data_x, predicted_outputs, marker, c='blue', label='predicted')

    if training_data is not None:
        training_data_x = [t[0] for t in training_data]
        training_data_y = [t[1] for t in training_data]
        plt.plot(training_data_x, training_data_y, marker, c='green', label='training_data')

    plt.legend()
    if visualize:
        plt.show()
    if savefig_filename:
        plt.savefig(savefig_filename)


def print_data(data, predicted_outputs):
    Error = 0.0
    SE = 0.0
    RAEDivider = 0.0
    RRAEDivider = 0.0
    AvgData = sum(predicted_outputs) / len(predicted_outputs)

    for i, row in enumerate(data):
        Error += abs(predicted_outputs[i] - row[1])
        SE += (predicted_outputs[i] - row[1]) ** 2
        RAEDivider += abs(AvgData - row[1])
        RRAEDivider += (AvgData - row[1]) ** 2

    AvgError = 1.0 * Error / (1.0 * len(data))
    MSE = 1.0 * SE / (1.0 * len(data))
    RMSE = MSE ** 0.5
    RAE = Error / RAEDivider
    RRAE = SE / RRAEDivider

    print('Error = %.3f' % Error)
    print('AvgError = %.3f' % AvgError)
    print('MSE = %.3f' % MSE)
    print('RMSE = %.3f' % RMSE)
    print('RAE = %.3f' % RAE)
    print('RRAE = %.3f' % RRAE)

    err = 0.0
    for i, pred in enumerate(predicted_outputs):
        min = float('inf')

        for j, row in enumerate(data):
            if min > ((row[1] - pred) ** 2 + (row[0] - data[i][0]) ** 2) ** 0.5:
                min = ((row[1] - pred) ** 2 + (row[0] - data[i][0]) ** 2) ** 0.5
        err += min

    print('Sum of distances = %.3f' % err)
    print('Avg distance = %.3f' % (err / len(data)))


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


def scale_data(train_set_inputs, test_set_inputs):
    y_train = [y for x, y in train_set_inputs]
    y_test = [y for x, y in test_set_inputs]
    min_y = min(y_train + y_test)
    max_y = max(y_train + y_test)
    train_set_inputs = [(x, (y - min_y) / (max_y - min_y)) for x,y in train_set_inputs]
    test_set_inputs = [(x,(y - min_y) / (max_y - min_y)) for x,y in test_set_inputs]

    return train_set_inputs, test_set_inputs


def main(train_filename, test_filename, create_nn, save_nn, read_nn, number_of_epochs, visualize_every, l_rate, biases,
         savefig_filename, activation_f, activation_f_derivative):
    from util import read_network_layers_from_file, write_network_to_file
    if train_filename is None or test_filename is None:
        print('Both train and test filename has to be provided for scaling')
        exit(1)

    train_set_inputs = read_file(train_filename)
    test_set_inputs = read_file(test_filename)
    train_set_inputs, test_set_inputs = scale_data(train_set_inputs, test_set_inputs)

    if create_nn is not None:
        # Calculate the number of inputs and outputs from the data.
        n_inputs = get_n_inputs_outputs(train_set_inputs)
        neural_network = initialize_network(create_nn, n_inputs, biases, activation_f, activation_f_derivative)
    else:
        layers, _ = read_network_layers_from_file(read_nn)
        neural_network = NeuralNetwork([NeuronLayer(l) for l in layers], activation_f, activation_f_derivative)

    # Train neural network.
    neural_network.train(train_set_inputs, l_rate, number_of_epochs, visualize_every)

    if save_nn is not None:
        write_network_to_file(save_nn, neural_network)

    if neural_network is None:
        layers, _ = read_network_layers_from_file(read_nn)
        neural_network = NeuralNetwork([NeuronLayer(l) for l in layers], activation_f, activation_f_derivative)

    # Test the neural network.
    accuracy, predicted_outputs = neural_network.test(test_set_inputs)

    print_data(test_set_inputs, predicted_outputs)

    if len(test_set_inputs[0]) == 2 and (visualize_every is not None or savefig_filename is not None):
        plot_data(test_set_inputs, predicted_outputs, visualize_every, savefig_filename, train_set_inputs)
    return accuracy
