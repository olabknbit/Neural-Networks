def activate(weights, inputs, bias):
    activation = bias
    for weight, input in zip(weights, inputs):
        activation += weight * input
    return activation


def in_ns_to_str(n_dict):
    return str({neuron_id: weight for neuron_id, weight in n_dict.iteritems()})


def neurons_to_ids_str(n_list):
    return str([neuron_id for neuron_id in n_list])


def neurons_to_str(n_dict):
    return str({neuron_id: neuron.to_str() for neuron_id, neuron in n_dict.iteritems()})


class Innovation:
    def __init__(self, source, end, innovation_number, disabled=False):
        assert type(source) is int
        assert type(end) is int
        assert type(innovation_number) is int
        self.source = source
        self.end = end
        self.innovation_number = innovation_number
        self.disabled = disabled


class Neuron:
    def __init__(self, id, in_ns, out_ns, bias_weight, activation_f, activation_f_derivative):
        assert type(id) is int
        assert type(in_ns) is dict
        for key, value in in_ns.iteritems():
            assert type(key) is int
            assert type(value) is float
        assert type(out_ns) is list
        for neuron_id in out_ns:
            assert type(neuron_id) is int
        self.id = id
        self.in_ns = in_ns
        self.out_ns = out_ns
        self.bias_weight = bias_weight
        self.activation_f = activation_f
        self.activation_f_derivative = activation_f_derivative

        self.output = None
        self.delta = None

    def reset_output_and_delta(self):
        self.output = None
        self.delta = None

    def to_str(self):
        n = {'id': self.id, 'in_ns': in_ns_to_str(self.in_ns), 'out_ns': neurons_to_ids_str(self.out_ns),
             'bias_weight': self.bias_weight}
        return str(n)


class NeuralNetwork:
    def __init__(self, neurons, input_neurons, output_neuron, activation_f, activation_f_derivative,
                 innovations=list(), id=0):
        """
        :param activation_f: lambda taking x : float and returning another float
        :param activation_f_derivative: lambda derivative of activation_f, taking x : float and returning another float
        """
        assert type(neurons) is dict
        for key, value in neurons.iteritems():
            assert type(key) is int
            assert isinstance(value, Neuron)
        for input_neuron in input_neurons:
            assert type(input_neuron) is int
        assert type(output_neuron) is int
        self.neurons = neurons
        self.input_neurons = input_neurons
        self.output_neuron = output_neuron
        self.activation_f = lambda x: activation_f(x)
        self.activation_f_derivative = lambda x: activation_f_derivative(x)

        self.score = None
        self.innovations = innovations
        self.id = id

    def print_input_neurons(self):
        return neurons_to_ids_str(self.input_neurons)

    def get_innovation_weight(self, source, end):
        return self.neurons[end.id].in_ns[source.id]

    def get_output(self, neuron):
        if neuron.output is None:
            weights = []
            inputs = []
            for in_neuron_id, in_neuron_weight in neuron.in_ns.iteritems():
                weights.append(in_neuron_weight)
                in_neuron = self.neurons[in_neuron_id]
                inputs.append(self.get_output(in_neuron))
            activation = activate(weights, inputs, neuron.bias_weight)
            neuron.output = neuron.activation_f(activation)

        return neuron.output

    # Pipe data row through the network and get final outputs.
    def forward_propagate(self, row):
        for input_neuron_id, val in zip(self.input_neurons, row):
            input_neuron = self.neurons[input_neuron_id]
            input_neuron.output = val
        output_neuron = self.neurons[self.output_neuron]
        return self.get_output(output_neuron)

    def set_delta(self, neuron):
        error = 0.0
        for neuron_id in neuron.out_ns:
            out_n = self.neurons[neuron_id]
            weight = out_n.in_ns[neuron.id]
            delta = self.get_delta(out_n)
            error += (weight * delta)
        neuron.delta = error * neuron.activation_f_derivative(neuron.output)

    def get_delta(self, neuron):
        if neuron.delta is None:
            self.set_delta(neuron)
        return neuron.delta

    # Calculate 'delta' for every neuron. This will then be used to update the weights of the neurons.
    def backward_propagate(self, expected_val):
        # Update last layer's (output layer's) 'delta' field.
        # This field is needed to calculate 'delta' fields in previous (closer to input layer) layers,
        # so then we can update the weights.
        output_neuron = self.neurons[self.output_neuron]
        error = (expected_val - output_neuron.output)
        output_neuron.delta = error * self.activation_f_derivative(output_neuron.output)

        for input_neuron_id in self.input_neurons:
            input_neuron = self.neurons[input_neuron_id]
            self.set_delta(input_neuron)

    def update_weights_neuron(self, l_rate, neuron):
        for in_neuron_id, in_neuron_weight in neuron.in_ns.iteritems():
            in_neuron = self.neurons[in_neuron_id]
            neuron.in_ns[in_neuron_id] += l_rate * neuron.delta * self.get_output(in_neuron)
        neuron.bias_weight += l_rate * neuron.delta

    def update_weights(self, l_rate):
        for input_neuron_id in self.neurons:
            input_neuron = self.neurons[input_neuron_id]
            self.update_weights_neuron(l_rate, input_neuron)

    def reset(self):
        for neuron in self.neurons.values():
            neuron.reset_output_and_delta()

    def to_str(self):
        """
        :return: list of number of neurons in each deep layer
        """

        def neurons_to_str(n_dict):
            return str({neuron_id: neuron.to_str() for neuron_id, neuron in n_dict.iteritems()})

        net_s = {'neurons': neurons_to_str(self.neurons),
                 'input_neurons': neurons_to_ids_str(self.input_neurons),
                 'output_neuron': self.output_neuron,
                 }
        return str(net_s)

    def predict(self, row):
        y_predicted = self.forward_propagate(row)
        self.reset()
        return y_predicted

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
    network.train(X_train, y_train, n_iter)
    score, y_predicted = network.test(X_test, y_test)
    if savefig_filename is not None:
        from util import plot_regression_data
        plot_regression_data(X_train, y_train, X_test, y_test, y_predicted, savefig_filename=savefig_filename, title=score)
    return network.score


def clone(network):
    from copy import deepcopy
    return deepcopy(network)
