from numpy import array, exp, random


class NeuronLayer():
    def __init__(self, n_inputs, n_neurons):
        # Create a layer with n_neurons neurons, each with n_inputs + 1 inputs (the +1 is for the bias).
        # TODO - biases should be settable.
        self.layer = [{'weights': [random.random() for _ in range(n_inputs + 1)]} for _ in range(n_neurons)]

    def __len__(self):
        return len(self.layer[0]['weights'])


class NeuralNetwork():
    def __init__(self, layers):
        self.layers = layers

    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation

    # Sigmoid transfer function
    @staticmethod
    def sigmoid(activation):
        return 1.0 / (1.0 + exp(-activation))

    # Derivative of transfer function
    @staticmethod
    def sigmoid_derivative(output):
        return output * (1.0 - output)

    def forward_propagate(self, row):
        inputs = row
        for layer in self.layers:
            outputs = []
            for neuron in layer.layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.sigmoid(activation)
                outputs.append(neuron['output'])

            inputs = outputs

        return inputs

    def backward_propagate(self, expected):
        pass

    def update_weights(self, row, l_rate):
        pass

    def train(self, data_input, l_rate, n_iter, n_outputs):
        for iteration in range(n_iter):
            iter_error = 0
            for row in data_input:
                outputs = self.forward_propagate(row)
                expected = [0 for i in range(n_outputs)]
                expected[row[-1]] = 1
                iter_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                self.backward_propagate(expected)
                self.update_weights(row, l_rate)
                print('>epoch=%d, lrate=%.3f, error=%.3f' % (iteration, l_rate, iter_error))

    def print_weights(self):
        for i, layer in enumerate(self.layers):
            print("    Layer %d (4 neurons, each with %d inputs): " % (i, (len(layer) - 1)))
            print(layer.layer)

    def guess(self, row):
        return 0, 1


def main():
    # Seed the random number generator
    random.seed(1)

    # TODO should read layer data from user input.
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
    training_set_inputs = array([[0, 0, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 1, 0], [0, 0, 0, 0]])
    # Should calculate the number of outoputs from the data.
    n_outputs = 2

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, 0.5, 60000, n_outputs)

    print("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()

    # Test the neural network with a new situation.
    print("Stage 3) Considering a new situation [1, 1, 0] -> ?: ")
    hidden_state, output = neural_network.guess(array([1, 1, 0]))
    print(output)


if __name__ == "__main__":
    main()