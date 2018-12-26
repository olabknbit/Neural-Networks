import random

from nnr_new import Innovation


def validate(network):
    neurons = network.neurons

    for neuron_id, neuron in neurons.iteritems():
        for neuron_in_id in neuron.in_ns:
            neuron_in = neurons[neuron_in_id]
            if neuron_id not in neuron_in.out_ns:
                print(network.to_str())
                print("BIG BAD ERROR type 1")
                exit(1)
        for neuron_out_id in neuron.out_ns:
            neuron_out = neurons[neuron_out_id]
            if neuron_id not in neuron_out.in_ns:
                print(neuron_id, 'not in', neuron_out.in_ns)
                print(network.to_str())
                print("BIG BAD ERROR type 2")

                exit(1)
    return True


def add_link(neuron1, neuron2, weight):
    neuron2.in_ns[neuron1.id] = weight
    neuron1.out_ns.append(neuron2.id)


def remove_link(neuron1, neuron2):
    neuron2.in_ns.pop(neuron1.id)
    neuron1.out_ns.remove(neuron2.id)


def contains_cycle_util(neurons, node, visited, rec_stack):
    visited.add(node.id)
    rec_stack.add(node.id)
    for neighbour_id in node.out_ns:

        if neighbour_id not in visited:
            neighbour = neurons[neighbour_id]
            if contains_cycle_util(neurons, neighbour, visited, rec_stack):
                return True
        elif neighbour_id in rec_stack:
            return True
    rec_stack.remove(node.id)
    return False


def contains_cycle(network):
    rec_stack = set()
    visited = set()
    for node_id in network.input_neurons:
        if node_id not in visited:
            node = network.neurons[node_id]
            if contains_cycle_util(network.neurons, node, visited, rec_stack):
                return True
    return False


def randomly_add_link(network, neuron_source_id, neuron_end_id, innovation_number):
    prob_add_link = 0.5
    succeeded = False
    if random.random() < prob_add_link:
        neurons = network.neurons

        neuron_source = neurons.get(neuron_source_id)
        neuron_end = neurons.get(neuron_end_id)

        def prem_approve(neuron1, neuron2):

            def already_have_link(neuron1, neuron2):
                return neuron1.id in neuron2.in_ns or neuron2.id in neuron1.in_ns

            return neuron1 is not None and neuron2 is not None \
                   and not already_have_link(neuron1, neuron2) \
                   and not neuron1.id == neuron2.id

        if prem_approve(neuron_source, neuron_end):
            # try_adding_link
            add_link(neuron_source, neuron_end, weight=random.random() * 0.3)
            if not contains_cycle(network):
                network.innovations.append(
                    Innovation(neuron_source_id, neuron_end_id, innovation_number=innovation_number))
                return network, True
            else:
                remove_link(neuron_source, neuron_end)
                return network, False
    return network, succeeded


def randomly_add_neuron(network, neuron_id, neuron_source_id, neuron_end_id, innovation_number):
    prob_add_neuron = 0.3
    succeeded = False

    if random.random() < prob_add_neuron:
        neurons = network.neurons
        neuron1 = neurons.get(neuron_source_id)
        neuron2 = neurons.get(neuron_end_id)
        if neuron1 is None or neuron2 is None or neuron2.in_ns.get(neuron1.id) is None:
            return network, succeeded

        from nnr_new import Neuron
        new_in_ns = {neuron1.id: 1.0}
        new_out_ns = [neuron2.id]
        neuron1.out_ns.remove(neuron2.id)

        weight = neuron2.in_ns.pop(neuron1.id)
        new_neuron = Neuron(neuron_id, new_in_ns, new_out_ns, 0.3)
        neuron1.out_ns.append(new_neuron.id)
        neuron2.in_ns[new_neuron.id] = weight
        neurons[new_neuron.id] = new_neuron

        def get_innovation_index():
            for index, inn in enumerate(network.innovations):
                source = neurons[inn.source]
                end = neurons[inn.end]
                if source.id == neuron1.id and end.id == neuron2.id:
                    return index

        innovation_index = get_innovation_index()
        network.innovations[innovation_index].disabled = True
        network.innovations.append(Innovation(new_neuron.id, neuron2.id, innovation_number))
        network.innovations.append(Innovation(neuron1.id, new_neuron.id, innovation_number + 1))
        validate(network)
        succeeded = True
    return network, succeeded


def calculate_compatibility(net1, net2):
    # excess genes E
    # disjoint genes D
    # average wight diff W of matching genes
    # N = 1 for small networks (fewer genomes than 20)
    N = 1
    E = 0
    W = 0
    D = 0
    c1 = 1.0
    c2 = 1.0
    c3 = 0.4
    for neuron in net1.neurons:
        if neuron not in net2.neurons:
            E += 1
    for neuron in net2.neurons:
        if neuron not in net1.neurons:
            E += 1

    for innovation in net1.innovations:
        s = innovation.source
        e = innovation.end
        weight2 = net2.get_innovation_weight(s, e)
        if weight2 is not None:
            D += 1
            weight1 = net1.get_innovation_weight(s, e)
            if weight1 is not None:
                W += abs(weight1 - weight2)
    if D != 0:
        W /= D
    return c1 * E / N + c2 * D / N + c3 * W


def main(train_filename, test_filename, n_networks=10, n_generations=30):
    activation = 'tanh'
    from util import get_activation_f_and_f_d_by_name, initialize_network, get_split_dataset
    from nnr_new import score, clone
    X_train, y_train, X_test, y_test = get_split_dataset(train_filename, test_filename)
    n_inputs = len(X_train[0])
    innovation_number = n_inputs
    number_of_neurons = n_inputs + 1
    activation_f, activation_f_d = get_activation_f_and_f_d_by_name(activation)
    networks = [initialize_network(n_inputs, activation_f=activation_f, activation_f_derivative=activation_f_d, _id=id)
                for id in range(n_networks)]

    network_id = len(networks)

    print("Initialized %d networks:" % n_networks)

    def cmp_nn(n1, n2):
        if n1.score is None or n2.score is None:
            return 0
        else:
            return int((n1.score - n2.score) * 100)

    for gen in range(n_generations):
        print('Genration', gen)

        print('Mutate networks:\n\tKeep 20% of the best unchanged,\n\tLose 20% of the worst,\n\tMutate rest.')
        n_keep = int(len(networks) * 0.2)
        best_nets = networks[:-n_keep]
        nets_to_mutate1 = clone(networks[:-n_keep])
        nets_to_mutate2 = clone(networks[:-n_keep])

        def get_iterator():
            import itertools
            combs = list(itertools.combinations(range(number_of_neurons), 2))
            perm = range(len(combs))
            random.shuffle(perm)

            return [combs[comb_index] for comb_index in perm]

        def add_connection(networks, innovation_number, network_id):
            mutated_nets = []
            my_it = get_iterator()
            for neuron_source_id, neuron_end_id in my_it:
                for network in networks:
                    network, succeeded = randomly_add_link(network, neuron_source_id, neuron_end_id, innovation_number)
                    if succeeded:
                        network.id = network_id
                        network_id += 1
                        mutated_nets.append(network)

                if len(mutated_nets) is not 0:
                    innovation_number += 1
                    break
            return mutated_nets, innovation_number, network_id

        mutated_nets1, innovation_number, network_id = \
            add_connection(nets_to_mutate1, innovation_number, network_id)

        def add_neuron(networks, number_of_neurons, innovation_number, network_id):
            mutated_nets = []
            my_it = get_iterator()
            for neuron_source_id, neuron_end_id in my_it:
                for network in networks:
                    network, succeeded = randomly_add_neuron(network, number_of_neurons, neuron_source_id,
                                                             neuron_end_id, innovation_number)
                    if succeeded:
                        network.id = network_id
                        network_id += 1
                        mutated_nets.append(network)
                if len(mutated_nets) != 0:
                    innovation_number += 1
                    number_of_neurons += 1
                    break
            return mutated_nets, innovation_number, number_of_neurons

        mutated_nets2, innovation_number, number_of_neurons = \
            add_neuron(nets_to_mutate2, number_of_neurons, innovation_number, network_id)

        networks = best_nets + mutated_nets1 + mutated_nets2

        for network1 in networks:
            for network2 in networks:
                # print('breeding ', network1.id, network2.id, calculate_compatibility(network1, network2))
                # TODO finish breeding
                pass

        print('Score networks')
        for i, n in enumerate(networks):
            dir = 'tmp/'
            base_name = str(n.id) + '-' + str(gen)
            savefig_filename = dir + base_name + '.png'
            print('gen', gen, 'scoring network id', n.id)
            score(n, X_train, y_train, X_test, y_test, n_iter=101, savefig_filename=savefig_filename)
            save_nn_filename = dir + base_name + '-' + str(n.score)
            from util import write_network_to_file_regression
            write_network_to_file_regression(save_nn_filename, n)

        networks.sort(cmp=cmp_nn)
