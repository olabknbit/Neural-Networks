# Activation function - for weights and inputs returns their dot product.
def activate(weights, inputs):
    # Add bias.
    activation = weights[-1]
    for weight, input in zip(weights[:-1], inputs):
        activation += weight * input
    return activation


class NeuronLayer:
    def __init__(self, neurons, biases=True):
        '''
        :param neurons: list, of which element is a dict with 'weights' : list, 'delta' : float, 'output' : float
        '''
        self.neurons = neurons
        self.biases = biases

    def __len__(self):
        return len(self.neurons[0]['weights'])

    # Calculate what are the outputs of the layer for the given inputs.
    def forward_propagate(self, inputs, activation_f):
        outputs = []
        for neuron in self.neurons:
            activation = activate(neuron['weights'], inputs)
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

    def get_neurons(self):
        biases = -1 if self.biases else 0
        return len(self) + biases


class NeuralNetwork():
    def __init__(self, layers, activation_f, activation_f_derivative):
        '''
        :param layers: list of NeuronLayers
        :param activation_f: lambda taking x : float and returning another float
        :param activation_f_derivative: lambda derivative of activation_f, taking x : float and returning another float
        '''
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
        '''
        :return: list of number of neurons in each deep layer
        '''
        return [layer.get_neurons() for layer in self.layers[1:]]

    def get_weights(self):
        return [str(layer.neurons) for layer in self.layers]

    def predict(self, row):
        return self.forward_propagate(row)

    def test(self, X_test, y_test):
        predicted_outputs = []
        error = 0.0
        for row, correct_output in zip(X_test, y_test):
            predicted_output = self.predict(row)[0]
            predicted_outputs.append(predicted_output)
            error += abs(predicted_output - correct_output)
        return error / len(X_test), predicted_outputs


def train(network, X_train, y_train, n_iter, l_rate=0.001, visualize_every=None):
    import numpy as np
    last_error = - np.infty
    for epoch in range(n_iter):
        iter_error = 0.0
        for row, expected in zip(X_train, y_train):
            # The net should only predict the class based on the features,
            # so the last cell which represents the class is not passed forward.
            output = network.forward_propagate(row)[0]

            iter_error += np.sqrt((expected - output) ** 2)
            network.backward_propagate(expected)
            network.update_weights(row, l_rate)
        if visualize_every is not None and epoch % visualize_every == 0:
            import visualize
            visualize.visualize_network(network, epoch)

        if epoch % 100 == 0:
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, iter_error))
            # Stop training if iter_error not changing.
            # TODO consider stochastic batches.
            if abs(last_error - iter_error) < 0.001:
                break
            last_error = iter_error