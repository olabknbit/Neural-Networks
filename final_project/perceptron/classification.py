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


class NeuralNetwork:
    def __init__(self, layers, activation_f, activation_f_derivative, output_classes):
        self.layers = layers
        self.activation_f = lambda x: activation_f(x)
        self.activation_f_derivative = lambda x: activation_f_derivative(x)
        self.output_classes = output_classes

    # Pipe data row through the network and get final outputs.
    def forward_propagate(self, row):
        outputs = row
        for layer in self.layers:
            inputs = outputs
            outputs = layer.forward_propagate(inputs, self.activation_f)

        return outputs

    # Calculate 'delta' for every neuron. This will then be used to update the weights of the neurons.
    def backward_propagate(self, expected_vals):
        # Update last layer's (output layer's) 'delta' field.
        # This field is needed to calculate 'delta' fields in previous (closer to input layer) layers,
        # so then we can update the weights.
        layer = self.layers[-1].neurons
        for neuron, expected in zip(layer, expected_vals):
            error = (expected - neuron['output'])
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
    def genetic(self)
        import random
    #Mutation disconnect neuron
        for layer in self.layers[:-1]
            for neuron in layer
                if random.random()<0.02:
                    neuron['output'] = 0
    #Mutation 
        for layer in self.layers[:-1]
            for neuron in layer
                if random.random()<0.1:
                    neuron['output'] = random.uniform(-1,1)
    

    def train(self, data_input, l_rate, n_iter, visualize_every):
        for epoch in range(n_iter):
            SE = 0.0
            correct = 0.0
            for row in data_input:
                # The net should only predict the class based on the features,
                # so the last cell which represents the class is not passed forward.
                outputs = self.forward_propagate(row[:-1])
                max_index = outputs.index(max(outputs))
                correct_index = self.output_classes[row[-1]]
                outputs_up = [0 if i is not max_index else 1 for i in range(len(outputs))]
                if max_index == correct_index:
                    correct += 1

                # The expected values are 0s for all neurons except for the ith,
                # where i is the class that is the output.
                expected = [0.0 for _ in range(len(self.output_classes))]
                expected[self.output_classes[row[-1]]] = 1.0

                SE += sum([(expected_i - output_i) ** 2 for expected_i, output_i in zip(expected, outputs)])

                self.backward_propagate(expected)
                # albo stochastic albo batchowe update TODO
                self.update_weights(row, l_rate)

            self.genetic()

            MSE = 1.0 * SE / (1.0 * len(data_input))
            accuracy = correct / len(data_input)
            if visualize_every is not None and epoch % visualize_every == 0:
                print('>epoch=%d, lrate=%.3f, mse=%.3f, accuracy=%.3f' % (epoch, l_rate, MSE, accuracy))
                # print(self.get_weights())



    def get_weights(self):
        return [str(layer.neurons) for layer in self.layers]

    def predict(self, row):
        outputs = self.forward_propagate(row[:-1])
        return outputs.index(max(outputs))

    def test(self, test_data):
        predicted_outputs = []
        correct = 0
        for row in test_data:
            predicted_output = self.predict(row)
            correct_output = self.output_classes[row[-1]]
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


def plot_data(data, outputs_classes, predicted_outputs, accuracy, filename):
    import matplotlib.pyplot as plt
    colors = ['red', 'blue', 'green', 'yellow', 'pink']
    plt.clf()
    datas = []
    for _ in range(len(outputs_classes)):
        datas.append(([[], []], [[], []]))

    for row, predicted in zip(data, predicted_outputs):
        i = row[2]
        output_class = outputs_classes[i]
        correct, incorrect = datas[output_class]
        if predicted != output_class:
            x, y = incorrect
            x.append(row[0])
            y.append(row[1])
        else:
            x, y = correct
            x.append(row[0])
            y.append(row[1])

    for i, c in enumerate(datas):
        correct, incorrect = c
        xc, yc = correct
        xi, yi = incorrect
        len_c = len(xc)
        len_i = len(xi)
        len_all = len_c + len_i
        label_c = 'correct + %d/%d + (%f)' % (len_c, len_all, (len_c / len_all))
        label_i = 'incorrect + %d/%d + (%f)' % (len_i, len_all, (len_i / len_all))
        plt.plot(xc, yc, linestyle='none', marker='o', markerfacecolor=colors[i], markeredgecolor='none', label=label_c)
        plt.plot(xi, yi, linestyle='none', marker='o', markerfacecolor=colors[i], markeredgecolor='black',
                 label=label_i)

    plt.legend()
    plt.title(filename + ' ' + str(accuracy))
    plt.savefig(filename + '.png')


def initialize_network(neurons, n_inputs, outputs_classes, biases, activation_f, activation_f_derivative):
    # Combine the layers to create a neural network
    layers = []
    n_outputs = len(outputs_classes)
    n_in = n_inputs
    bias = 1 if biases else 0
    # Create a layer with n_neurons neurons, each with n_inputs.
    for n_neurons in neurons:
        layers.append(NeuronLayer(get_random_neurons(n_in + bias, n_neurons)))
        n_in = n_neurons
    layers.append(NeuronLayer(get_random_neurons(n_in + bias, n_outputs)))

    return NeuralNetwork(layers, activation_f, activation_f_derivative, outputs_classes)


def main(train_filename, test_filename, create_nn, save_nn, read_nn, number_of_epochs, visualize_every, l_rate, biases,
         activation_f, activation_f_derivative):
    from util import write_network_to_file, read_network_layers_from_file

    neural_network = None
    outputs_classes = None
    if train_filename is not None:
        training_set_inputs = read_file(train_filename)

        if create_nn is not None:
            # Calculate the number of inputs and outputs from the data.
            n_inputs, outputs_classes = get_n_inputs_outputs(training_set_inputs)
            neural_network = initialize_network(create_nn, n_inputs, outputs_classes, biases, activation_f,
                                                activation_f_derivative)
        else:
            layers, outputs_classes = read_network_layers_from_file(read_nn)
            neural_network = NeuralNetwork([NeuronLayer(l) for l in layers], activation_f, activation_f_derivative,
                                           outputs_classes)

        # Train neural network.
        neural_network.train(training_set_inputs, l_rate, number_of_epochs, visualize_every)

        if save_nn is not None:
            write_network_to_file(save_nn, neural_network)

    if test_filename is not None:
        testing_set_inputs = read_file(test_filename)

        if neural_network is None:
            layers, outputs_classes = read_network_layers_from_file(read_nn)
            neural_network = NeuralNetwork([NeuronLayer(l) for l in layers], activation_f, activation_f_derivative,
                                           outputs_classes)

        # Test the neural network.
        accuracy, predicted_outputs = neural_network.test(testing_set_inputs)
        print("accuracy: %.3f" % accuracy)

        if visualize_every is not None:
            # Plot test data. Dots with black egdes are the ones that didn't get classified correctly.
            plot_data(testing_set_inputs, outputs_classes, predicted_outputs, accuracy, save_nn)
