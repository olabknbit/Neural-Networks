def activate(weights, inputs, bias):
    activation = bias
    for weight, input in zip(weights, inputs):
        activation += weight * input
    return activation


def n_list_to_str(n_list):
    return str([neuron.to_str() for neuron in n_list])


def n_dict_to_str(n_dict):
    return str({neuron.id: weight for neuron, weight in n_dict.iteritems()})
# class NeuronLayer:
#     def __init__(self, neurons, bias_weights):
#         """
#         :param neurons: list, of which an element is a dict with 'weights' : list, 'delta' : float, 'output' : float
#         """
#         self.neurons = neurons
#
#     def __len__(self):
#         return len(self.neurons[0]['weights'])
#
#     # Calculate what are the outputs of the layer for the given inputs.
#     def forward_propagate(self, inputs, activation_f):
#         outputs = []
#         for neuron, bias in zip(self.neurons, self.bias_weights):
#             activation = activate(neuron['weights'], inputs, bias)
#             neuron['output'] = activation_f(activation)
#             outputs.append(neuron['output'])
#
#         return outputs
#
#     def backward_propagate(self, next_layer, activation_f_derivative):
#         for i, neuron_i in enumerate(self.neurons):
#             error = 0.0
#             for neuron_j in next_layer.neurons:
#                 error += (neuron_j['weights'][i] * neuron_j['delta'])
#             neuron_i['delta'] = error * activation_f_derivative(neuron_i['output'])
#         return self
#
#     def update_weights(self, inputs, l_rate):
#         for i, neuron in enumerate(self.neurons):
#             for j in range(len(inputs)):
#                 if neuron['active_weights'][j]:
#                     neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
#             self.bias_weights[i] += l_rate * neuron['delta']
#         return self.neurons
#
#     def linear_propagate(self, inputs):
#         from util import linear
#         outputs = []
#         for neuron, bias in zip(self.neurons, self.bias_weights):
#             activation = activate(neuron['weights'], inputs, bias)
#             neuron['output'] = linear(activation)
#             outputs.append(neuron['output'])
#         return outputs
#
#     def get_neurons(self):
#         return len(self)


class Neuron:
    def __init__(self, id, level, in_ns, out_ns, bias_weight, activation_f, activation_f_derivative):
        self.id = id
        self.level = level
        self.in_ns = in_ns
        self.out_ns = out_ns
        self.bias_weight = bias_weight
        self.activation_f = activation_f
        self.activation_f_derivative = activation_f_derivative

        self.output = None
        self.delta = None

    def get_output(self):
        if self.output is None:
            weights = []
            inputs = []
            for in_neuron, in_neuron_weight in self.in_ns.iteritems():
                weights.append(in_neuron_weight)
                inputs.append(in_neuron.get_output())
            activation = activate(weights, inputs, self.bias_weight)
            self.output = self.activation_f(activation)

        return self.output

    def set_delta(self):
        error = 0.0
        for neuron in self.out_ns:
            weight = neuron.in_ns[self]
            delta = neuron.get_delta()
            error += (weight * delta)
        self.delta = error * self.activation_f_derivative(self.output)

    def get_delta(self):
        if self.delta is None:
            self.set_delta()
        return self.delta

    def update_weights(self, l_rate):
        for in_neuron, in_neuron_weight in self.in_ns.iteritems():
            self.in_ns[in_neuron] += l_rate * self.delta * in_neuron.get_output()
        self.bias_weight += l_rate * self.delta

    def reset_output_and_delta(self):
        self.output = None
        self.delta = None

    def to_str(self):
        n = {'id': self.id, 'in_ns': n_dict_to_str(self.in_ns), 'out_ns': str([neuron.id for neuron in self.out_ns]), 'bias_weight': self.bias_weight}
        return str(n)


class NeuralNetwork:

    def __init__(self, neurons, input_neurons, output_neuron, activation_f, activation_f_derivative):
        """
        :param layers: list of NeuronLayers
        :param activation_f: lambda taking x : float and returning another float
        :param activation_f_derivative: lambda derivative of activation_f, taking x : float and returning another float
        """
        self.neurons = neurons
        self.input_neurons = input_neurons
        self.output_neuron = output_neuron
        self.activation_f = lambda x: activation_f(x)
        self.activation_f_derivative = lambda x: activation_f_derivative(x)
        self.score = None

    # Pipe data row through the network and get final outputs.
    def forward_propagate(self, row):
        for input_neuron, val in zip(self.input_neurons, row):
            input_neuron.output = val
        return self.output_neuron.get_output()

    # Calculate 'delta' for every neuron. This will then be used to update the weights of the neurons.
    def backward_propagate(self, expected_val):
        # Update last layer's (output layer's) 'delta' field.
        # This field is needed to calculate 'delta' fields in previous (closer to input layer) layers,
        # so then we can update the weights.
        error = (expected_val - self.output_neuron.output)
        self.output_neuron.delta = error * self.activation_f_derivative(self.output_neuron.output)

        for input_neuron in self.input_neurons:
            input_neuron.set_delta()

    def update_weights(self, l_rate):
        for input_neuron in self.neurons:
            input_neuron.update_weights(l_rate)

    def reset(self):
        for neuron in self.neurons:
            neuron.reset_output_and_delta()

    def to_str(self):
        """
        :return: list of number of neurons in each deep layer
        """

        net_s = {'neurons': n_list_to_str(self.neurons),
                 'input_neurons': str([neuron.id for neuron in self.input_neurons]),
                 'output_neuron': self.output_neuron.to_str(),
                 }
        return str(net_s)

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
                self.update_weights(l_rate)
                self.reset()
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
