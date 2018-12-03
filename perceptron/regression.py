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
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
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
        from util import linear_derivative
        # Update last layer's (output layer's) 'delta' field.
        # This field is needed to calculate 'delta' fields in previous (closer to input layer) layers,
        # so then we can update the weights.
        layer = self.layers[-1].neurons
        neuron = layer[0]
        error = (expected_val - neuron['output'])
        neuron['delta'] = error * linear_derivative(neuron['output'])

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
        for epoch in range(n_iter):
            iter_error = 0.0
            for row in data_input:
                # The net should only predict the class based on the features,
                # so the last cell which represents the class is not passed forward.
                outputs = self.forward_propagate(row[:-1])

                expected = row[-1]
                iter_error += np.sqrt(expected ** 2)
                self.backward_propagate(expected)
                self.update_weights(row, l_rate)
            if visualize_every is not None and epoch % visualize_every == 0:
                import visualize
                from util import read_network_layers_from_file, write_network_to_file
                write_network_to_file("temp", self)
                layers, _ = read_network_layers_from_file("temp")
                visualize.main(layers, str(epoch))
                print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, iter_error))

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


# Return number of features - n_inputs for NN and outoput_classes which is a map like {3: 0, 2: 1:, 5: 2},
# where each key represents a class (as they occur in original data, eg, here class 3, 2 and 5) and indices they have
#  in NN.
def get_n_inputs_outputs(data):
    n_inputs = len(data[0]) - 1

    outputs_classes = {0: 0}
    return n_inputs, outputs_classes


def plot_data(data, predicted_outputs, training_data=None):
    import matplotlib.pyplot as plt
    colors = ['red', 'blue', 'green']

    for i, row in enumerate(data):
        plt.scatter(row[0], row[1], c=colors[0])
        plt.scatter(row[0], predicted_outputs[i], c=colors[1])

    if training_data is not None:
        for row in training_data:
            plt.scatter(row[0], row[1], c=colors[2])

    plt.show()


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

    print('Sum of distances = %.3f' % (err))
    print('Avg distance = %.3f' % (err / len(data)))


def initialize_network(neurons, n_inputs, outputs_classes, biases):
    # Combine the layers to create a neural network
    layers = []
    n_outputs = len(outputs_classes)
    n_in = n_inputs
    bias = 1 if biases else 0
    for n_neurons in neurons:
        layers.append(NeuronLayer(get_random_neurons(n_in + bias, n_neurons)))
        n_in = n_neurons
    layers.append(NeuronLayer(get_random_neurons(n_in + bias, n_outputs)))

    from util import sigmoid, sigmoid_derivative
    return NeuralNetwork(layers, sigmoid, sigmoid_derivative)


def main(train_filename, test_filename, create_nn, save_nn, read_nn, number_of_epochs, visualize_every, l_rate, biases):
    from util import read_network_layers_from_file, write_network_to_file, sigmoid, sigmoid_derivative
    neural_network = None
    training_set_inputs = None
    if train_filename is not None:
        training_set_inputs = read_file(train_filename)

        if create_nn is not None:
            # Calculate the number of inputs and outputs from the data.
            n_inputs, outputs_classes = get_n_inputs_outputs(training_set_inputs)
            neural_network = initialize_network(create_nn, n_inputs, outputs_classes, biases)
        else:
            layers, _ = read_network_layers_from_file(read_nn)
            neural_network = NeuralNetwork([NeuronLayer(l) for l in layers], sigmoid, sigmoid_derivative)

        # Train neural network.
        neural_network.train(training_set_inputs, l_rate, number_of_epochs, visualize_every)

        if save_nn is not None:
            write_network_to_file(save_nn, neural_network)

    if test_filename is not None:
        testing_set_inputs = read_file(test_filename)

        if neural_network is None:
            layers, _ = read_network_layers_from_file(read_nn)
            neural_network = NeuralNetwork([NeuronLayer(l) for l in layers], sigmoid, sigmoid_derivative)

        # Test the neural network.
        accuracy, predicted_outputs = neural_network.test(testing_set_inputs)
        print("accuracy: %.3f" % accuracy)

        print_data(testing_set_inputs, predicted_outputs)
        if visualize_every is not None:
            plot_data(testing_set_inputs, predicted_outputs, training_set_inputs)
