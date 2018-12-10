import numpy as np


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