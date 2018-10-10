from numpy import array, exp, random


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

    def train(self, data_input, l_rate, n_iter, n_outputs):
        for epoch in range(n_iter):
            iter_error = 0
            for row in data_input:
                outputs = self.forward_propagate(row)

                # The expected values are 0s for all neurons except for the ith,
                # where i is the class that is the output.
                expected = [0 for _ in range(n_outputs)]
                expected[row[-1]] = 1

                iter_error += sum([(expected_i - output_i) ** 2 for expected_i, output_i in zip(expected, outputs)])
                self.backward_propagate(expected)
                self.update_weights(row, l_rate)
                print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, iter_error))

    def print_weights(self):
        n_inputs = 0
        for i, layer in enumerate(self.layers):
            n_neurons = len(layer) - 1
            print("    Layer %d (%d neurons, each with %d inputs): " % (i, n_neurons, n_inputs))
            n_inputs = n_neurons + 1
            print(layer.neurons)

    def predict(self, row):
        outputs = self.forward_propagate(row)
        return outputs.index(max(outputs))


def main():
    # Seed the random number generator
    random.seed(1)

    # TODO should read layer data from user input.
    # TODO: use argparse or sth to read user input.
    # Create layer (4 neurons, each with 3 inputs)
    input_layer = NeuronLayer(3, 5)

    hidden_layer = NeuronLayer(5, 4)

    # Create layer (2 neurons with 4 inputs)
    output_layer = NeuronLayer(4, 2)

    # Combine the layers to create a neural network
    layers = [input_layer, hidden_layer, output_layer]
    neural_network = NeuralNetwork(layers)

    print("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    # TODO should read data from files.
    training_set_inputs = array([[0, 0, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 1, 0], [0, 0, 0, 0]])
    # Should calculate the number of outputs from the data.
    n_outputs = 2

    # Train neural network.
    neural_network.train(training_set_inputs, 0.5, 6000, n_outputs)

    print("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()

    # Test the neural network with a new situation.
    # TODO: Use test data sets.
    print("Stage 3) Considering a new situation [1, 1, 0] -> ?: ")
    output = neural_network.predict([1, 1, 0])
    print(output)


if __name__ == "__main__":
    main()