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
    import random
    prob_add_link = 0.05
    prob_add_neuron = 0.03
    neurons = net.neurons
    n_neurons = len(neurons)

    def already_have_link(neuron1, neuron2):
        print(neuron1.id, neuron2.id, neuron1 in neuron2.in_ns or neuron2 in neuron1.in_ns)
        return neuron1 in neuron2.in_ns or neuron2 in neuron1.in_ns

    def get_two_neurons():
        import itertools
        combs = list(itertools.combinations(neurons, 2))
        perms = list(itertools.permutations(range(len(combs))))
        perm = perms[random.randint(0, len(perms) - 1)]

        def approve(neuron1, neuron2):
            return neuron1 is not None and neuron2 is not None and neuron1.level != neuron2.level and not already_have_link(
                neuron1, neuron2)

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

    if random.random() < prob_add_link:
        nns = get_two_neurons()
        if nns is not None:
            neuron1, neuron2 = nns
            print('adding link', neuron1.id, neuron2.id)
            neuron2.in_ns[neuron1] = random.random() * 0.3
            neuron1.out_ns.append(neuron2)

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
        print(len(neuron1.out_ns), [n.id for n in neuron1.out_ns])
        neuron1.out_ns.remove(neuron2)
        print(len(neuron1.out_ns), [n.id for n in neuron1.out_ns])

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

        validate(net)
    return net


def main(train_filename, test_filename, get_nn_filenames_f, n_networks=10, n_generations=10):
    activation = 'tanh'
    from util import get_activation_f_and_f_d_by_name, initialize_start_network, get_split_dataset
    from neural_network_regression import score
    X_train, y_train, X_test, y_test = get_split_dataset(train_filename, test_filename)
    networks = [initialize_start_network(len(X_train[0]), activation=activation) for _ in range(n_networks)]
    activation_f, activation_f_d = get_activation_f_and_f_d_by_name(activation)
    networks = [randomly_mutate(net, activation_f, activation_f_d) for net in networks]

    print("Initialized %d networks:" % n_networks)

    print('Score networks')
    for i, n in enumerate(networks):
        score(n, X_train, y_train, X_test, y_test, n_iter=2)

    def cmp_nn(n1, n2):
        if n1.score is None or n2.score is None:
            return 0
        else:
            return int((n1.score - n2.score) * 100)

    networks.sort(cmp=cmp_nn)

    print("Network scores:")
    for n in networks:
        print(n.score)

    from visualize import simple_vis
    simple_vis(networks[0])
