import random

from nnr_new import Innovation

INNOVATION_NUMBER = 0


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


def randomly_mutate(network, activation_f, activation_f_d):
    global INNOVATION_NUMBER
    prob_add_link = 0.5
    prob_add_neuron = 0.3
    neurons = network.neurons
    n_neurons = len(neurons)

    if random.random() < prob_add_link:
        def already_have_link(neuron1, neuron2):
            return neuron1.id in neuron2.in_ns or neuron2.id in neuron1.in_ns

        import itertools
        combs = list(itertools.combinations(neurons, 2))
        perm = range(len(combs))
        random.shuffle(perm)

        def prem_approve(neuron1, neuron2):
            return neuron1 is not None and neuron2 is not None and not already_have_link(neuron1, neuron2)

        for comb_index in perm:
            neuron1_id, neuron2_id = combs[comb_index]
            neuron1, neuron2 = network.neurons[neuron1_id], network.neurons[neuron2_id]
            if prem_approve(neuron1, neuron2):

                def try_adding_link(neuron1, neuron2):
                    global INNOVATION_NUMBER
                    add_link(neuron1, neuron2, weight=random.random() * 0.3)
                    if not contains_cycle(network):
                        network.innovations.append(
                            Innovation(neuron1.id, neuron2.id, innovation_number=INNOVATION_NUMBER))
                        INNOVATION_NUMBER += 1
                        return True
                    else:
                        remove_link(neuron1, neuron2)
                        return False

                if try_adding_link(neuron1, neuron2):
                    break
                elif try_adding_link(neuron2, neuron1):
                    break

    if random.random() < prob_add_neuron:
        neuron1_id = neurons.keys()[random.randint(1, n_neurons - 1)]
        neuron1 = neurons[neuron1_id]
        select_neuron_ids = neuron1.out_ns + neuron1.in_ns.keys()
        if len(select_neuron_ids) == 1:
            neuron2_id = select_neuron_ids[0]
        else:
            neuron2_id = select_neuron_ids[random.randint(1, len(select_neuron_ids) - 1)]

        neuron2 = neurons[neuron2_id]
        if neuron2.id not in neuron1.out_ns:
            tmp = neuron1
            neuron1 = neuron2
            neuron2 = tmp
        from nnr_new import Neuron
        new_in_ns = {neuron1.id: 1.0}
        new_out_ns = [neuron2.id]
        neuron1.out_ns.remove(neuron2.id)

        if neuron2.level - neuron1.level > 1:
            level = int((neuron2.level - neuron1.level) / 2)
        else:
            level = neuron2.level
            for neuron_id, neuron in neurons.iteritems():
                if neuron.level >= level:
                    neuron.level += 1

        weight = neuron2.in_ns.pop(neuron1.id)
        new_neuron = Neuron(len(neurons) + 1, level, new_in_ns, new_out_ns, 0.3, activation_f, activation_f_d)
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
        network.innovations.append(Innovation(new_neuron.id, neuron2.id, INNOVATION_NUMBER))
        INNOVATION_NUMBER += 1
        network.innovations.append(Innovation(neuron1.id, new_neuron.id, INNOVATION_NUMBER + 1))
        INNOVATION_NUMBER += 1
        validate(network)
    return network


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
    global INNOVATION_NUMBER
    INNOVATION_NUMBER = n_inputs
    activation_f, activation_f_d = get_activation_f_and_f_d_by_name(activation)
    networks = [initialize_network(n_inputs, activation_f=activation_f, activation_f_derivative=activation_f_d, _id=id)
                for id in range(n_networks)]
    networks = [randomly_mutate(network, activation_f, activation_f_d) for network in networks]
    network_id = n_networks

    print("Initialized %d networks:" % n_networks)

    def cmp_nn(n1, n2):
        if n1.score is None or n2.score is None:
            return 0
        else:
            return int((n1.score - n2.score) * 100)

    for gen in range(n_generations):
        print('Genration', gen)

        print('Score networks')
        for i, n in enumerate(networks):
            dir = 'tmp/'
            base_name = str(n.id) + '-' + str(gen)
            savefig_filename = dir + base_name + '.png'
            score(n, X_train, y_train, X_test, y_test, n_iter=101, savefig_filename=savefig_filename)
            save_nn_filename = dir + base_name + '-' + str(n.score)
            from util import write_network_to_file_regression
            write_network_to_file_regression(save_nn_filename, n)

        networks.sort(cmp=cmp_nn)

        print('Mutate networks:\n\tKeep 20% of the best unchanged,\n\tLose 20% of the worst,\n\tMutate rest.')
        n_keep = int(len(networks) * 0.2)
        best_nets = clone(networks[:n_keep])
        mutated_nets = [randomly_mutate(network, activation_f, activation_f_d) for network in networks[:-n_keep]]
        for network in mutated_nets:
            network.id = network_id
            network_id += 1
        networks = best_nets + mutated_nets
