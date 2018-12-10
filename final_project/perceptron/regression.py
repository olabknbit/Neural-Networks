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


def plot_data(X_train, y_train, X_test, y_test, y_predicted, y_sklearn_predicted=None, savefig_filename=None):
    if len(X_train[0]) != 1:
        print('Cannot plot because len(data) = %d > 1' % len(X_train[0]))
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
    if savefig_filename is not None:
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


def get_nn(create_nn, read_nn, biases, activation_f, activation_f_derivative, X_train):
    from util import read_network_layers_from_file
    neural_network = None
    if create_nn is not None:
        # Calculate the number of inputs and outputs from the data.
        n_inputs = len(X_train[0])
        neural_network = initialize_network(create_nn, n_inputs, biases, activation_f, activation_f_derivative)
    elif read_nn is not None:
        layers, _ = read_network_layers_from_file(read_nn)
        neural_network = NeuralNetwork([NeuronLayer(l) for l in layers], activation_f, activation_f_derivative)
    else:
        print('Cannot create nor read nn. Exiting')
        exit(1)

    return neural_network


def sklearn_test(nn, X_train, y_train, X_test, y_test):
    from sklearn.neural_network import MLPRegressor
    lr = MLPRegressor(hidden_layer_sizes=nn, activation='tanh').fit(X_train, y_train)
    return lr.predict(X_test)


def get_split_dataset(train_filename, test_filename):
    train_set_inputs, test_set_inputs = read_file(train_filename), read_file(test_filename)
    X_train, y_train = split_data(train_set_inputs)
    X_test, y_test = split_data(test_set_inputs)
    y_train, y_test = scale_data(y_train, y_test)
    return X_train, y_train, X_test, y_test


def main(train_filename, test_filename, create_nn, save_nn, read_nn, number_of_epochs, visualize_every, l_rate, biases,
         savefig_filename, activation_f, activation_f_derivative, compare_to_sklearn=False):
    if train_filename is None or test_filename is None:
        print('Both train and test filename has to be provided for scaling')
        exit(1)

    X_train, y_train, X_test, y_test = get_split_dataset(train_filename, test_filename)
    neural_network = get_nn(create_nn, read_nn, biases, activation_f, activation_f_derivative, X_train)

    # Train neural network.
    from train import train
    train(neural_network, X_train, y_train, l_rate, number_of_epochs, visualize_every)

    if save_nn is not None:
        from util import write_network_to_file
        write_network_to_file(save_nn, neural_network)

    # Test the neural network.
    avg_error, y_predicted = neural_network.test(X_test, y_test)
    print('NN accuracy and errors')
    print_data(y_test, y_predicted)

    y_sklearn_predicted = None
    if compare_to_sklearn:
        y_sklearn_predicted = sklearn_test(create_nn, X_train, y_train, X_test, y_test)
        print('\nsklearn accuracy and errors')
        print_data(y_test, y_sklearn_predicted)

    if savefig_filename is not None:
        plot_data(X_train, y_train, X_test, y_test, y_predicted, y_sklearn_predicted, savefig_filename)
    return avg_error
