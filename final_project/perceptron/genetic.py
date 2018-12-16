INNOVATION_NUMBER = 0


def validate(net):
    neurons = net.neurons

    for neuron in neurons:
        for neuron_in in neuron.in_ns:
            if neuron not in neuron_in.out_ns:
                print(net.to_str())
                print("BIG BAD ERROR type 1")
                exit(1)
        for neuron_out in neuron.out_ns:
            if neuron not in neuron_out.in_ns:
                print(net.to_str())
                print("BIG BAD ERROR type 2")

                exit(1)
    return True


def randomly_mutate(net, activation_f, activation_f_d):
    global INNOVATION_NUMBER
    import random
    from nnr_new import Innovation
    prob_add_link = 0.5
    prob_add_neuron = 0.3
    neurons = net.neurons
    n_neurons = len(neurons)

    if random.random() < prob_add_link:

        def already_have_link(neuron1, neuron2):
            return neuron1 in neuron2.in_ns or neuron2 in neuron1.in_ns

        def get_two_neurons():
            import itertools
            combs = list(itertools.combinations(neurons, 2))
            perm = range(len(combs))
            random.shuffle(perm)

            def approve(neuron1, neuron2):
                return neuron1 is not None and neuron2 is not None and neuron1.level != neuron2.level \
                       and not already_have_link(neuron1, neuron2)

            for comb_index in perm:
                neuron1, neuron2 = combs[comb_index]
                if approve(neuron1, neuron2):
                    break

            if approve(neuron1, neuron2):
                if neuron1.level > neuron2.level:
                    return neuron2, neuron1
                else:
                    return neuron1, neuron2
            return None

        nns = get_two_neurons()
        if nns is not None:
            neuron1, neuron2 = nns
            neuron2.in_ns[neuron1] = random.random() * 0.3
            neuron1.out_ns.append(neuron2)
            net.innovations.append(Innovation(neuron1, neuron2, innovation_number=INNOVATION_NUMBER))
            INNOVATION_NUMBER += 1

    if random.random() < prob_add_neuron:
        neuron1 = neurons[random.randint(1, n_neurons - 1)]
        select_neurons = neuron1.out_ns + neuron1.in_ns.keys()
        if len(select_neurons) == 1:
            neuron2 = select_neurons[0]
        else:
            neuron2 = select_neurons[random.randint(1, len(select_neurons) - 1)]

        if neuron2 not in neuron1.out_ns:
            tmp = neuron1
            neuron1 = neuron2
            neuron2 = tmp
        from nnr_new import Neuron
        new_in_ns = {neuron1: 1}
        new_out_ns = [neuron2]
        neuron1.out_ns.remove(neuron2)

        if neuron2.level - neuron1.level > 1:
            level = int((neuron2.level - neuron1.level) / 2)
        else:
            level = neuron2.level
            for neuron in neurons:
                if neuron.level >= level:
                    neuron.level += 1

        weight = neuron2.in_ns.pop(neuron1)
        new_neuron = Neuron(len(neurons) + 1, level, new_in_ns, new_out_ns, 0.3, activation_f, activation_f_d)
        neuron1.out_ns.append(new_neuron)
        neuron2.in_ns[new_neuron] = weight
        neurons.append(new_neuron)

        def get_innovation_index():
            for index, inn in enumerate(net.innovations):
                if inn.source.id == neuron1.id and inn.end.id == neuron2.id:
                    return index

        innovation_index = get_innovation_index()
        net.innovations[innovation_index].disabled = True
        net.innovations.append(Innovation(new_neuron, neuron2, INNOVATION_NUMBER))
        INNOVATION_NUMBER += 1
        net.innovations.append(Innovation(neuron1, new_neuron, INNOVATION_NUMBER + 1))
        INNOVATION_NUMBER += 1
        validate(net)
    return net


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


def main(train_filename, test_filename, get_nn_filenames_f, n_networks=10, n_generations=10):
    activation = 'tanh'
    from util import get_activation_f_and_f_d_by_name, initialize_network, get_split_dataset
    from nnr_new import score, clone
    X_train, y_train, X_test, y_test = get_split_dataset(train_filename, test_filename)
    n_inputs = len(X_train[0])
    global INNOVATION_NUMBER
    INNOVATION_NUMBER = n_inputs
    activation_f, activation_f_d = get_activation_f_and_f_d_by_name(activation)
    networks = [initialize_network(n_inputs, activation_f=activation_f, activation_f_derivative=activation_f_d, _id=id) for id in range(n_networks)]

    networks = [randomly_mutate(net, activation_f, activation_f_d) for net in networks]

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
            savefig_filename = str(n.id) + '-' + str(gen) + '.png'
            score(n, X_train, y_train, X_test, y_test, n_iter=100, savefig_filename=savefig_filename)

        networks.sort(cmp=cmp_nn)

        print('Mutate networks:\n\tKeep 20% of the best unchanged,\n\tLose 20% of the worst,\n\tMutate rest.')
        n_keep = int(len(networks) * 0.2)
        best_nets = clone(networks[:n_keep])
        mutated_nets = [randomly_mutate(net, activation_f, activation_f_d) for net in networks[:-n_keep]]
        networks = best_nets + mutated_nets
