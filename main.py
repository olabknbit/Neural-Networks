from numpy import exp, random
import numpy as np


# Activation function - for weights and inputs returns their dot product.
def activate(weights, inputs):
    activation = weights[-1]
    for weight, input in zip(weights[:-1], inputs):
        activation += weight * input
    return activation


# TODO: should be implemented so that different (tanh, logistic, etc) transfer functions can be simply interchangeable.
# Sigmoid transfer function
def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))


# TODO: Each transfer function should have it's derivative.
# Derivative of transfer function
def sigmoid_derivative(output):
    return output * (1.0 - output)


class NeuronLayer():
    def __init__(self, n_inputs, n_neurons):
        # Create a layer with n_neurons neurons, each with n_inputs + 1 inputs (the +1 is for the bias).
        # TODO - biases should be settable.
        # random numbers from range [0; 0.3) are proven to be best
        self.neurons = [{'weights': [random.random() * 0.3 for _ in range(n_inputs + 1)]} for _ in range(n_neurons)]

    def __len__(self):
        return len(self.neurons[0]['weights'])

    # Calculate what are the outputs of the layer for the given inputs.
    def forward_propagate(self, inputs):
        outputs = []
        for neuron in self.neurons:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = sigmoid(activation)
            outputs.append(neuron['output'])
        return outputs

    def backward_propagate(self, next_layer):
        for i, neuron_i in enumerate(self.neurons):
            error = 0.0
            for neuron_j in next_layer:
                error += (neuron_j['weights'][i] * neuron_j['delta'])
            neuron_i['delta'] = error * sigmoid_derivative(neuron_i['output'])
        return self.neurons

    def update_weights(self, inputs, l_rate):
        for neuron in self.neurons:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']
        return self.neurons


class NeuralNetwork():
    def __init__(self, layers):
        self.layers = layers

    # Pipe data row through the network and get final outputs.
    def forward_propagate(self, row):
        outputs = row
        for layer in self.layers:
            inputs = outputs
            outputs = layer.forward_propagate(inputs)

        return outputs

    # Calculate 'delta' for every neuron. This will then be used to update the weights of the neurons.
    def backward_propagate(self, expected_vals):
        # Update last layer's (output layer's) 'delta' field.
        # This field is needed to calculate 'delta' fields in previous (closer to input layer) layers,
        # so then we can update the weights.
        layer = self.layers[-1].neurons
        for neuron, expected in zip(layer, expected_vals):
            error = (expected - neuron['output'])
            neuron['delta'] = error * sigmoid_derivative(neuron['output'])

        # Update other layers' 'delta' field, so we can later update wights based on value of this field.
        next_layer = layer
        for layer in reversed(self.layers[:-1]):
            next_layer = layer.backward_propagate(next_layer)

    def update_weights(self, row, l_rate):
        layer = self.layers[0]

        # Crop last cell which is the class that data element/row belongs to.
        inputs = row[:-1]
        previous_layer = layer.update_weights(inputs, l_rate)

        for layer in self.layers[1:]:
            inputs = [neuron['output'] for neuron in previous_layer]
            previous_layer = layer.update_weights(inputs, l_rate)

    def train(self, data_input, l_rate, n_iter, output_classes):
        for epoch in range(n_iter):
            iter_error = 0.0
            for row in data_input:
                # The net should only predict the class based on the features,
                # so the last cell which represents the class is not passed forward.
                outputs = self.forward_propagate(row[:-1])

                # The expected values are 0s for all neurons except for the ith,
                # where i is the class that is the output.
                expected = [0.0 for _ in range(len(output_classes))]
                expected[output_classes[row[-1]]] = 1.0

                iter_error += sum([(expected_i - output_i) ** 2 for expected_i, output_i in zip(expected, outputs)])
                self.backward_propagate(expected)
                self.update_weights(row, l_rate)
                print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, iter_error))

    def print_weights(self):
        for i, layer in enumerate(self.layers):
            n_neurons = len(layer) - 1
            print("    Layer %d (%d neurons): " % (i, n_neurons))
            print(layer.neurons)

    def predict(self, row):
        outputs = self.forward_propagate(row[:-1])
        return outputs.index(max(outputs))

    def test(self, output_classes, test_data):
        predicted_outputs = []
        correct = 0
        for row in test_data:
            predicted_output = self.predict(row)
            correct_output = output_classes[row[-1]]
            predicted_outputs.append(predicted_output)
            if correct_output == predicted_output:
                correct += 1
        return correct / len(test_data), predicted_outputs


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
                row = [float(x) for x in row[:-1]] + [int(row[-1])]
                rows.append(row)
        return rows


def read_train_data():
    return read_file('projekt1/classification/data.simple.train.100.csv')


def read_test_data():
    return read_file('projekt1/classification/data.simple.test.100.csv')


# Return number of features - n_inputs for NN and outoput_classes which is a map like {3: 0, 2: 1:, 5: 2},
# where each key represents a class (as they occur in original data, eg, here class 3, 2 and 5) and indices they have
#  in NN.
def get_n_inputs_outputs(data):
    n_inputs = len(data[0]) - 1
    outputs = set()
    for row in data:
        outputs.add(row[-1])
    outputs_classes = {}
    for i, outpt in enumerate(outputs):
        outputs_classes[outpt] = i
    return n_inputs, outputs_classes


def plot_data(data, outputs_classes, predicted_outputs):
    import matplotlib.pyplot as plt
    colors = ['red', 'blue']

    for row, predicted in zip(data, predicted_outputs) :
        edgecolor = 'none'
        output_class = outputs_classes[row[2]]
        if predicted != output_class:
            edgecolor = 'black'
        plt.scatter(row[0], row[1], c=colors[output_class], edgecolors=edgecolor)
    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-n', '--neurons', nargs='+', type=int, help='<Required> Number of neurons for each layer',
                        required=True)

    parser.add_argument('--regression', dest='regression', action='store_true')
    parser.add_argument('--classification', dest='regression', action='store_false')
    parser.set_defaults(feature=False)

    parser.add_argument('-train_filename')
    parser.add_argument('-test_filename')

    args = parser.parse_args()

    # Seed the random number generator
    random.seed(1)

    training_set_inputs = read_file(args.train_filename)

    # Should calculate the number of inputs and outputs from the data.
    n_inputs, outputs_classes = get_n_inputs_outputs(training_set_inputs)
    n_outputs = len(outputs_classes)

    # Combine the layers to create a neural network
    layers = []
    n_in = n_inputs
    for n_neurons in args.neurons:
        layers.append(NeuronLayer(n_in, n_neurons))
        print(n_in, n_neurons)
        n_in = n_neurons
    layers.append(NeuronLayer(n_in, n_outputs))
    print(n_in, n_outputs)

    neural_network = NeuralNetwork(layers)

    print("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # Train neural network.
    neural_network.train(training_set_inputs, 0.3, 1000, outputs_classes)

    print("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()

    # Test the neural network.
    testing_set_inputs = read_file(args.test_filename)
    accuracy, predicted_outputs = neural_network.test(outputs_classes, testing_set_inputs)
    print("accuracy: %.3f" % accuracy)

    # Plot test data. Dots with black egdes are the ones that didn't get classified correctly.
    plot_data(testing_set_inputs, outputs_classes, predicted_outputs)


if __name__ == "__main__":
    main()