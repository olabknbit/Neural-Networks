import numpy as np


def get_random_neurons(n_inputs, n_neurons):
    from numpy import random
    # random numbers from range [0; 0.3) are proven to be best
    return [{'weights': [random.random() * 0.3 for _ in range(n_inputs)]} for _ in range(n_neurons)]


class NeuronLayer:
    def __init__(self, neurons):
        self.neurons = neurons

    def __len__(self):
        return len(self.neurons[0]['weights'])

    # Calculate what are the outputs of the layer for the given inputs.
    def forward_propagate(self, inputs, activation_f):
        from util import activate
        outputs = []
        for neuron in self.neurons:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = activation_f(activation)
            outputs.append(neuron['output'])

        return outputs

    def backward_propagate(self, next_layer, activation_f_derivative):
        for i, neuron_i in enumerate(self.neurons):
            error = 0.0
            for neuron_j in next_layer:
                error += (neuron_j['weights'][i] * neuron_j['delta'])
            neuron_i['delta'] = error * activation_f_derivative(neuron_i['output'])
        return self.neurons

    def update_weights(self, inputs, l_rate):
        for neuron in self.neurons:
            for j, input_j in enumerate(inputs):
                neuron['weights'][j] += l_rate * neuron['delta'] * input_j
            neuron['weights'][-1] += l_rate * neuron['delta']
        return self.neurons

    def linear_propagate(self, inputs):
        from util import activate, linear
        outputs = []
        for neuron in self.neurons:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = linear(activation)
            outputs.append(neuron['output'])

        return outputs


class NeuralNetwork():
    def __init__(self, layers, activation_f, activation_f_derivative):
        self.layers = layers
        self.activation_f = lambda x: activation_f(x)
        self.activation_f_derivative = lambda x: activation_f_derivative(x)

    # Pipe data row through the network and get final outputs.
    def forward_propagate(self, row):
        outputs = row
        for layer in self.layers[:-1]:
            inputs = outputs
            outputs = layer.forward_propagate(inputs, self.activation_f)

        inputs = outputs
        outputs = self.layers[-1].linear_propagate(inputs)

        return outputs

    # Calculate 'delta' for every neuron. This will then be used to update the weights of the neurons.
    def backward_propagate(self, expected_val):
        # Update last layer's (output layer's) 'delta' field.
        # This field is needed to calculate 'delta' fields in previous (closer to input layer) layers,
        # so then we can update the weights.
        layer = self.layers[-1].neurons
        neuron = layer[0]
        error = (expected_val - neuron['output'])
        neuron['delta'] = error * self.activation_f_derivative(neuron['output'])

        # Update other layers' 'delta' field, so we can later update wights based on value of this field.
        next_layer = layer
        for layer in reversed(self.layers[:-1]):
            next_layer = layer.backward_propagate(next_layer, self.activation_f_derivative)

    def update_weights(self, row, l_rate):
        layer = self.layers[0]

        # Crop last cell which is the class that data element/row belongs to.
        inputs = row[:-1]
        previous_layer = layer.update_weights(inputs, l_rate)

        for layer in self.layers[1:]:
            inputs = [neuron['output'] for neuron in previous_layer]
            previous_layer = layer.update_weights(inputs, l_rate)

    def train(self, data_input, l_rate, n_iter, visualize_every):
        last_error = - np.infty
        for epoch in range(n_iter):
            iter_error = 0.0
            for row in data_input:
                # The net should only predict the class based on the features,
                # so the last cell which represents the class is not passed forward.
                output = self.forward_propagate(row[:-1])[0]

                expected = row[-1]
                iter_error += np.sqrt((expected - output) ** 2)
                self.backward_propagate(expected)
                self.update_weights(row, l_rate)
            if visualize_every is not None and epoch % visualize_every == 0:
                import visualize
                from util import read_network_layers_from_file, write_network_to_file
                tmp_filename = "tmp/temp"
                write_network_to_file(tmp_filename, self)
                layers, _ = read_network_layers_from_file(tmp_filename)
                visualize.main(layers, str(epoch))
            if epoch % 100 == 0:
                print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, iter_error))

                # Stop training if iter_error not changing.
                # TODO consider stochastic batches.
                if abs(last_error - iter_error) < 0.001:
                    break
                last_error = iter_error

    def get_weights(self):
        return [str(layer.neurons) for layer in self.layers]

    def predict(self, row):
        return self.forward_propagate(row[:-1])

    def test(self, test_data):
        predicted_outputs = []
        error = 0.0
        for row in test_data:
            predicted_output = self.predict(row)[0]
            correct_output = row[-1]
            predicted_outputs.append(predicted_output)
            error += abs(predicted_output - correct_output)
        return error / len(test_data), predicted_outputs


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

# TODO implement
# def scale_data(y_train, y_test):
#     min_y = min(y_train + y_test)
#     max_y = max(y_train + y_test)
#     y_train = [((y - min_y) / (max_y - min_y)) for y in y_train]
#     y_test = [((y - min_y) / (max_y - min_y)) for y in y_test]
#
#     return y_train, y_test


def main(train_filename, test_filename, create_nn, save_nn, read_nn, number_of_epochs, visualize_every, l_rate, biases,
         savefig_filename, activation_f, activation_f_derivative):
    from util import read_network_layers_from_file, write_network_to_file
    neural_network = None
    training_set_inputs = None
    if train_filename is not None:
        training_set_inputs = read_file(train_filename)

        # TODO prettify scaling data
        xys = training_set_inputs
        ys = [y for x,y in xys]
        min_y = min(ys)
        max_y = max(ys)
        training_set_inputs = [(x, ((y - min_y)/(max_y - min_y))) for x, y in xys]

        if create_nn is not None:
            # Calculate the number of inputs and outputs from the data.
            n_inputs = get_n_inputs_outputs(training_set_inputs)
            neural_network = initialize_network(create_nn, n_inputs, biases, activation_f, activation_f_derivative)
        else:
            layers, _ = read_network_layers_from_file(read_nn)
            neural_network = NeuralNetwork([NeuronLayer(l) for l in layers], activation_f, activation_f_derivative)

        # Train neural network.
        neural_network.train(training_set_inputs, l_rate, number_of_epochs, visualize_every)

        if save_nn is not None:
            write_network_to_file(save_nn, neural_network)

    if test_filename is not None:
        testing_set_inputs = read_file(test_filename)
        # TODO prettify scaling data here as well.
        xys = testing_set_inputs
        testing_set_inputs = [(x, ((y - min_y) / (max_y - min_y))) for x, y in xys]

        if neural_network is None:
            layers, _ = read_network_layers_from_file(read_nn)
            neural_network = NeuralNetwork([NeuronLayer(l) for l in layers], activation_f, activation_f_derivative)

        # Test the neural network.
        accuracy, predicted_outputs = neural_network.test(testing_set_inputs)

        print_data(testing_set_inputs, predicted_outputs)

        if len(testing_set_inputs[0]) == 2 and (visualize_every is not None or savefig_filename is not None):
            plot_data(testing_set_inputs, predicted_outputs, visualize_every, savefig_filename, training_set_inputs)
        return accuracy
