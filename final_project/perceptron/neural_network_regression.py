def activate(weights, inputs, bias):
    activation = bias
    for weight, input in zip(weights, inputs):
        activation += weight * input
    return activation


class NeuronLayer:
    def __init__(self, neurons, bias_weights):
        """
        :param neurons: list, of which an element is a dict with 'weights' : list, 'delta' : float, 'output' : float
        """
        self.neurons = neurons
        self.bias_weights = bias_weights

    def __len__(self):
        return len(self.neurons[0]['weights'])

    # Calculate what are the outputs of the layer for the given inputs.
    def forward_propagate(self, inputs, activation_f):
        outputs = []
        for neuron, bias in zip(self.neurons, self.bias_weights):
            activation = activate(neuron['weights'], inputs, bias)
            neuron['output'] = activation_f(activation)
            outputs.append(neuron['output'])

        return outputs

    def backward_propagate(self, next_layer, activation_f_derivative):
        for i, neuron_i in enumerate(self.neurons):
            error = 0.0
            for neuron_j in next_layer.neurons:
                error += (neuron_j['weights'][i] * neuron_j['delta'])
            neuron_i['delta'] = error * activation_f_derivative(neuron_i['output'])
        return self

    def update_weights(self, inputs, l_rate):
        for i, neuron in enumerate(self.neurons):
            for j in range(len(inputs)):
                if neuron['active_weights'][j]:
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            self.bias_weights[i] += l_rate * neuron['delta']
        return self.neurons

    def linear_propagate(self, inputs):
        from util import linear
        outputs = []
        for neuron, bias in zip(self.neurons, self.bias_weights):
            activation = activate(neuron['weights'], inputs, bias)
            neuron['output'] = linear(activation)
            outputs.append(neuron['output'])
        return outputs

    def get_neurons(self):
        return len(self)


class NeuralNetwork():
    def __init__(self, layers, activation_f, activation_f_derivative):
        """
        :param layers: list of NeuronLayers
        :param activation_f: lambda taking x : float and returning another float
        :param activation_f_derivative: lambda derivative of activation_f, taking x : float and returning another float
        """
        self.layers = layers
        self.activation_f = lambda x: activation_f(x)
        self.activation_f_derivative = lambda x: activation_f_derivative(x)
        self.score = None

    # Pipe data row through the network and get final outputs.
    def forward_propagate(self, row):
        outputs = row
        for layer in self.layers[:-1]:
            inputs = outputs
            outputs = layer.forward_propagate(inputs, self.activation_f)

        inputs = outputs
        outputs = self.layers[-1].linear_propagate(inputs)

        return outputs[0]

    # Calculate 'delta' for every neuron. This will then be used to update the weights of the neurons.
    def backward_propagate(self, expected_val):
        # Update last layer's (output layer's) 'delta' field.
        # This field is needed to calculate 'delta' fields in previous (closer to input layer) layers,
        # so then we can update the weights.
        layer = self.layers[-1]
        neurons = layer.neurons
        neuron = neurons[0]
        error = (expected_val - neuron['output'])
        neuron['delta'] = error * self.activation_f_derivative(neuron['output'])

        # Update other layers' 'delta' field, so we can later update wights based on value of this field.
        next_layer = layer
        for layer in reversed(self.layers[:-1]):
            next_layer = layer.backward_propagate(next_layer, self.activation_f_derivative)

    def update_weights(self, inputs, l_rate):
        layer = self.layers[0]
        previous_layer = layer.update_weights(inputs, l_rate)

        for layer in self.layers[1:]:
            inputs = [neuron['output'] for neuron in previous_layer]
            previous_layer = layer.update_weights(inputs, l_rate)

    def get_params(self):
        """
        :return: list of number of neurons in each deep layer
        """
        return [layer.get_neurons() for layer in self.layers[1:]]

    def get_weights(self):
        return [str({'neurons': str(layer.neurons), 'bias_weights': str(layer.bias_weights)}) for layer in
                self.layers[1:]]

    def predict(self, row):
        return self.forward_propagate(row)

    def test(self, X_test, y_test):
        predicted_outputs = []
        error = 0.0
        for row, expected in zip(X_test, y_test):
            predicted_output = self.predict(row)
            predicted_outputs.append(predicted_output)
            error += abs(predicted_output - expected)
        self.score = error / len(X_test)
        return self.score, predicted_outputs

    def train(self, X_train, y_train, n_iter, l_rate=0.001, visualize_every=None):
        import numpy as np
        last_error = - np.infty
        for epoch in range(n_iter):
            iter_error = 0.0
            for row, expected in zip(X_train, y_train):
                # The net should only predict the class based on the features,
                # so the last cell which represents the class is not passed forward.
                output = self.forward_propagate(row)

                iter_error += np.sqrt((expected - output) ** 2)
                self.backward_propagate(expected)
                self.update_weights(row, l_rate)
            if visualize_every is not None and epoch % visualize_every == 0:
                import visualize
                visualize.visualize_network(self, epoch)

            if epoch % 100 == 0:
                print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, iter_error))
                # Stop training if iter_error not changing.
                # TODO consider stochastic batches.
                if abs(last_error - iter_error) < 0.001:
                    break
                last_error = iter_error

                # TODO use moment


def score(network, X_train, y_train, X_test, y_test, n_iter, savefig_filename=None):
    if network.score is None:
        network.train(X_train, y_train, n_iter)
        score, y_predicted = network.test(X_test, y_test)
        if savefig_filename is not None:
            from util import plot_regression_data
            plot_regression_data(X_test, y_train, X_test, y_test, y_predicted, savefig_filename=savefig_filename)
    return network.score
