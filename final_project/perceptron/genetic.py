import random

from perceptron.nnr_new import Innovation


def validate(network, change=""):
    neurons = network.neurons

    def print_error_and_exit(error):
        print(error)
        print("\n")
        print(change)
        print("\n")
        print(network.to_str())
        print("\n")
        exit(1)

    if len(network.innovations) == 0:
        print_error_and_exit("WTF 0 innovations")

    if len(neurons) == 0:
        print_error_and_exit("WTF 0 neurons")

    for innovation in network.innovations:
        if innovation.source not in neurons or innovation.end not in neurons:
            print_error_and_exit("innovation neuron not in neurons")

    for neuron_id, neuron in neurons.items():
        for neuron_in_id in neuron.in_ns:
            neuron_in = neurons[neuron_in_id]
            if neuron_id not in neuron_in.out_ns:
                print_error_and_exit("BIG BAD ERROR type 1")

        for neuron_out_id in neuron.out_ns:
            neuron_out = neurons[neuron_out_id]
            if neuron_id not in neuron_out.in_ns:
                print("\n")
                print(""+str(neuron_id)+"\n")
                print_error_and_exit("BIG BAD ERROR type 2:\n\t" + str(neuron_id) + ' not in' + str(neuron_out.in_ns))
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
    return network, False


def randomly_add_neuron(network, neuron_id, neuron_source_id, neuron_end_id, innovation_number):
    neurons = network.neurons
    neuron_source = neurons.get(neuron_source_id)
    neuron_end = neurons.get(neuron_end_id)
    if neuron_source is None or neuron_end is None or neuron_end.in_ns.get(neuron_source.id) is None:
        return network, False

    from perceptron.nnr_new import Neuron
    new_in_ns = {neuron_source.id: 1.0}
    new_out_ns = [neuron_end.id]
    neuron_source.out_ns.remove(neuron_end.id)

    weight = neuron_end.in_ns.pop(neuron_source.id)
    new_neuron = Neuron(neuron_id, new_in_ns, new_out_ns, 0.3)
    neuron_source.out_ns.append(new_neuron.id)
    neuron_end.in_ns[new_neuron.id] = weight
    neurons[new_neuron.id] = new_neuron

    def get_innovation_index():
        print(network.to_str())
        for index, innovation in enumerate(network.innovations):
            source = neurons[innovation.source]
            end = neurons[innovation.end]
            if source.id == neuron_source_id and end.id == neuron_end_id and not innovation.disabled:
                return index

    innovation_index = get_innovation_index()
    network.innovations[innovation_index].disabled = True
    network.innovations.append(Innovation(new_neuron.id, neuron_end.id, innovation_number))
    network.innovations.append(Innovation(neuron_source.id, new_neuron.id, innovation_number + 1))
    change = "Adding neuron %d between %d and %d. Innovation number %d. Network %s" \
             % (neuron_id, neuron_source_id, neuron_end_id, innovation_number, network.to_str())
    print(change)
    print("\n")
    validate(network, change=change)
    return network, True


def calculate_compatibility(network1, network2):
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
    for neuron in network1.neurons:
        if neuron not in network2.neurons:
            E += 1
    for neuron in network2.neurons:
        if neuron not in network1.neurons:
            D += 1

    for innovation in network1.innovations:
        s = innovation.source
        e = innovation.end
        weight2 = network2.get_innovation_weight(s, e)
        if weight2 is not None:
            weight1 = network1.get_innovation_weight(s, e)
            if weight1 is not None:
                W += abs(weight1 - weight2)
    if D != 0:
        W /= D
    res = c1 * E / N + c2 * D / N + c3 * W
    if network1.id == network2.id and res != 0.0:
        print("WTF two same networks should have compability 0.0!")
        exit(1)
    return res


def breed_children(network1, network2, innovation_number):
    if network1.equals(network2):
        return network1

    from perceptron.nnr_new import Neuron, NeuralNetwork
    i1 = 0
    i2 = 0

    neurons, input_neurons, innovations = {}, [], []

    def add_neurons_from_innovation(innovation, original_network):
        from perceptron.nnr_new import innovation_of
        if innovation.disabled:
            innovations.append(innovation_of(innovation))
            return
        neuron_source_id = innovation.source
        neuron_end_id = innovation.end
        weight = original_network.neurons[neuron_end_id].in_ns[neuron_source_id]

        def get_if_exists_create_otherwise(neuron_id):
            if neuron_id in neurons.keys():
                neuron = neurons[neuron_id]
            else:
                in_ns = {}
                out_ns = []
                neuron = Neuron(neuron_id, in_ns, out_ns, bias_weight=0.3)
                neurons[neuron_id] = neuron
                if neuron_id in original_network.input_neurons:
                    input_neurons.append(neuron_id)
            return neuron

        neuron_source = get_if_exists_create_otherwise(neuron_source_id)
        neuron_end = get_if_exists_create_otherwise(neuron_end_id)
        neuron_end.in_ns[neuron_source.id] = weight
        neuron_source.out_ns.append(neuron_end.id)
        innovations.append(innovation_of(innovation))
        return

    for i in range(innovation_number + 1):
        innovation1 = network1.get_innovation_or_none(i1)
        innovation2 = network2.get_innovation_or_none(i2)
        if innovation1 is not None and innovation2 is not None and innovation1.innovation_number == innovation2.innovation_number:
            # pick which parent child should inherit from
            # copy neurons and connection weight to the child
            # proceed
            r = random.random()
            if r < 0.5:
                add_neurons_from_innovation(innovation1, network1)
            else:
                add_neurons_from_innovation(innovation2, network2)

            i1 += 1
            i2 += 1
        elif (
                innovation1 is not None and innovation2 is not None and innovation1.innovation_number < innovation2.innovation_number) \
                or (innovation1 is not None and innovation2 is None):
            # add connection (and neuron) from innovation1 to the child
            add_neurons_from_innovation(innovation1, network1)
            i1 += 1
        elif (
                innovation1 is not None and innovation2 is not None and innovation1.innovation_number > innovation2.innovation_number) \
                or (innovation2 is not None and innovation1 is None):
            # add connection (and neuron) from innovation2 to the child
            add_neurons_from_innovation(innovation2, network2)
            i2 += 1
        else:
            break
    network = NeuralNetwork(neurons, input_neurons, network1.output_neuron, innovations=innovations)
    validate(network, change="breeding a child between \n\t%s\n\t%s\nchild\n\t%s" % (network1.to_str(), network2.to_str(), network.to_str()))
    return network


def get_iterator(number_of_elements):
    import itertools
    combs = list(itertools.combinations(range(number_of_elements), 2))
    perm = list(range(len(combs)))
    random.shuffle(perm)

    return [combs[comb_index] for comb_index in perm]


class Species:
    """
    Has a Representative
    and Fitness
    """

    def __init__(self, network):
        self.networks = [network]

        # pick random network as species' representative
        self.representative = network
        self.sum_adjusted_fitness = 0
        self.population_size = 1

    def remove_all_networks_but_representative(self):
        self.networks.remove(self.representative)
        all_networks_but_representative = [network for network in self.networks]
        self.networks = [self.representative]
        return all_networks_but_representative

    def add(self, network):
        self.networks.append(network)

    def save_n_networks(self, n):
        self.networks = self.networks[:n]
        change_repr = True
        for network in self.networks:
            if network.id == self.representative.id:
                change_repr = False
        if change_repr:
            self.representative = self.networks[random.randint(0, len(self.networks) - 1)]

    def show_off(self, id):
        for network in self.networks:
            print('species:', id, 'network id:', network.id, 'network_score:', network.score)


class NEAT:
    def __init__(self, network, number_of_neurons, innovation_number, data, neat_params, train_params, verbose=True):
        self.networks = {network.id: network}
        self.number_of_neurons = number_of_neurons
        self.innovation_number = innovation_number
        self.network_id = 0
        self.X_train, self.y_train, self.X_test, self.y_test = data
        self.species = [Species(network)]
        self.fitness = {}
        self.adjusted_fitness = {}
        self.COMPATIBILITY_THRESHOLD, self.FITTEST_PERCENTAGE, self.BABIES_PER_GENERATION, \
        self.ASEXUAL_REPRODUCTION_CHANCE, self.ADD_SYNAPSE_MUTATION_CHANCE, self.ADD_NEURON_MUTATION_CHANCE = neat_params
        self.total_adjusted_fitness = 0
        self.verbose = verbose
        self.n_iter, self.l_rate = train_params

    def calculate_fitness(self):
        print("---CALCULATE FITNESS---")
        for spec in self.species:
            from perceptron.nnr_new import score
            for network in spec.networks:
                print("\n")
                print(network.to_str())
                fitness = score(network, self.X_train, self.y_train, self.X_test, self.y_test, n_iter=self.n_iter)
                self.fitness[network.id] = fitness

    def calculate_adjusted_fitness(self):
        print("---CALCULATE ADJUSTED FITNESS---")
        self.total_adjusted_fitness = 0
        for spec in self.species:
            spec.sum_adjusted_fitness = 0
            for network in spec.networks:
                self.adjusted_fitness[network.id] = self._network_adjusted_fitness(network)
                spec.sum_adjusted_fitness += self.adjusted_fitness[network.id]
            self.total_adjusted_fitness += spec.sum_adjusted_fitness

    def _network_adjusted_fitness(self, network):
        fitness = self.fitness[network.id]
        fitness_sum = 0
        for spec in self.species:
            for comp_network in spec.networks:
                compatibility_distance = calculate_compatibility(network, comp_network)
                sh = 1
                if compatibility_distance > self.COMPATIBILITY_THRESHOLD:
                    sh = 0
                fitness_sum = fitness_sum + sh
        return fitness / fitness_sum

    def survival_of_the_fittest(self):
        print("---SURVIVAL OF THE FITTEST---")

        def cmp_nn(n1, n2):
            if self.fitness.get(n1.id) is None or self.fitness.get(n2.id) is None:
                return 0
            else:
                return int((self.fitness.get(n1.id) - self.fitness.get(n2.id)) * 100)

        for spec_id, spec in enumerate(self.species):
            # Sort networks by fitness scores.
            from functools import cmp_to_key
            spec.networks.sort(key = cmp_to_key(cmp_nn))
            #spec.networks.sort(cmp=cmp_nn)
            if self.verbose:
                spec.show_off(spec_id)

            # Only kill networks in species larger than 2 - the small species will extinct anyways/
            the_weak_count = 0
            if len(spec.networks) > 2:
                the_weak_count = len(spec.networks) - int(self.FITTEST_PERCENTAGE * len(spec.networks))
            those_who_survived = len(spec.networks) - the_weak_count
            spec.population_size = len(spec.networks)

            # Allow the population to breed children based on performance
            spec.population_size += int(
                (spec.sum_adjusted_fitness / self.total_adjusted_fitness) * self.BABIES_PER_GENERATION)

            # Remove worst performing networks from the species
            spec.save_n_networks(those_who_survived)

    def mate(self):
        print("---MATING SEASON---")
        for spec in self.species:
            baby_count = spec.population_size - len(spec.networks)

            for _ in range(baby_count):
                if random.random() < self.ASEXUAL_REPRODUCTION_CHANCE or len(spec.networks) == 1:
                    # Reproduce asexually:
                    # Pick parent
                    if len(spec.networks) == 1:
                        parent = spec.networks[0]
                    else:
                        parent = spec.networks[random.randint(0, len(spec.networks) - 1)]
                    from perceptron.nnr_new import clone
                    child = clone(parent)
                    self.mutate(child)
                    child.id = self.get_new_network_id()
                else:
                    # Mate
                    parent_id, second_parent_id = get_iterator(len(spec.networks))[0]
                    parent, second_parent = spec.networks[parent_id], spec.networks[second_parent_id]
                    child = breed_children(parent, second_parent, self.innovation_number)
                    child.id = self.get_new_network_id()

                spec.networks.append(child)

    def mutate(self, network):
        print("----MUTATE----")
        if random.random() < self.ADD_SYNAPSE_MUTATION_CHANCE:
            self.add_connection(network)
        if random.random() < self.ADD_NEURON_MUTATION_CHANCE:
            self.add_neuron(network)

    def get_new_network_id(self):
        self.network_id += 1
        return self.network_id

    def add_connection(self, network):
        neurons = list(network.neurons.values())
        my_it = get_iterator(len(neurons))
        for id1, id2 in my_it:
            neuron_source_id, neuron_end_id = neurons[id1].id, neurons[id2].id
            network, succeeded = randomly_add_link(network, neuron_source_id, neuron_end_id, self.innovation_number)
            if succeeded:
                network.id = self.get_new_network_id()
                self.innovation_number += 1
                break
        return network

    def add_neuron(self, network):
        neurons = list(network.neurons.values())
        my_it = get_iterator(len(neurons))
        for id1, id2 in my_it:
            neuron_source_id, neuron_end_id = neurons[id1].id, neurons[id2].id
            network, succeeded = randomly_add_neuron(network, self.number_of_neurons, neuron_source_id,
                                                     neuron_end_id, self.innovation_number)
            if succeeded:
                network.id = self.get_new_network_id()
                self.innovation_number += 1
                self.number_of_neurons += 1
                break
        return network

    def re_speciate(self):
        print("---RE-SPECIATE---")
        unorganized_genomes = []
        for spec in self.species:
            unorganized_genomes += spec.remove_all_networks_but_representative()

        for unorganized_genome in unorganized_genomes:
            allotted = False
            for spec in self.species:
                compability_distance = calculate_compatibility(unorganized_genome, spec.representative)
                if compability_distance <= self.COMPATIBILITY_THRESHOLD:
                    spec.add(unorganized_genome)
                    allotted = True
                    break

            if not allotted:
                spec = Species(unorganized_genome)
                self.species.append(spec)

    def sanity_check(self):
        print("---SANITY CHECK---")
        for spec in self.species:
            for network1_id, network1 in enumerate(spec.networks):
                for network2_id, network2 in enumerate(spec.networks):
                    if network1_id < network2_id:
                        if network1.id == network2.id:
                            print("WTF a species has the same network twice", network1.id)
                            print(network1.to_str())
                            print(network2.to_str())
                            exit(1)

        for spec1_id, spec1 in enumerate(self.species):
            for spec2_id, spec2 in enumerate(self.species):
                if spec1_id < spec2_id:
                    for network1 in spec1.networks:
                        for network2 in spec2.networks:
                            if network1.id == network2.id:
                                print("WTF two species have the same network", network1.id)
                                print(network1.to_str())
                                print(network2.to_str())
                                exit(1)

    def final(self):
        self.calculate_fitness()
        self.calculate_adjusted_fitness()
        self.survival_of_the_fittest()

    def save_networks(self):
        for spec in self.species:
            for network in spec.networks:
                dir = ''
                base_name = str(network.id)
                savefig_filename = dir + base_name + '.png'
                from perceptron.nnr_new import score
                fitness = score(network, self.X_train, self.y_train, self.X_test, self.y_test, n_iter=self.n_iter,
                                savefig_filename=savefig_filename)
                print(savefig_filename, network.score)
                self.fitness[network.id] = fitness
                save_nn_filename = dir + base_name + '-' + str(network.score)
                from perceptron.util import write_network_to_file_regression
                write_network_to_file_regression(save_nn_filename, network)

    def show_off(self):
        for spec_id, spec in enumerate(self.species):
            spec.show_off(spec_id)
    


def create_random_network(train_filename, test_filename, neat_params, train_params, activation='tanh', verbose=True):
    from perceptron.util import get_activation_f_and_f_d_by_name, initialize_network, get_split_dataset
    X_train, y_train, X_test, y_test = get_split_dataset(train_filename, test_filename)
    n_inputs = len(X_train[0])
    innovation_number = n_inputs
    number_of_neurons = n_inputs + 1
    activation_f, activation_f_d = get_activation_f_and_f_d_by_name(activation)
    network = initialize_network(n_inputs, activation_f=activation_f, activation_f_derivative=activation_f_d, _id=0)

    return NEAT(network, number_of_neurons, innovation_number, data=(X_train, y_train, X_test, y_test),
                neat_params=neat_params, train_params=train_params, verbose=verbose)


def main(train_filename, test_filename, neat_params, n_generations, train_params, save=False, verbose=True):
    neat = create_random_network(train_filename, test_filename, neat_params, train_params, verbose=verbose)
    results = list()
    for generation in range(0, n_generations):
        print("-GENERATION %d-" % generation)
        print("\n" + str(random.randint(0,10)))

        neat.calculate_fitness()
        neat.calculate_adjusted_fitness()
        neat.survival_of_the_fittest()
        neat.mate()
        neat.re_speciate()
        neat.sanity_check()


        gen_result= list()
        for spec in neat.species:
            for ii in spec.networks:
                print(ii.score)
                if(ii.score is None):
                    continue
                gen_result.append(ii.score)
        results.append(gen_result)

    generation = n_generations + 1
    print("-FINAL GENERATION %d-" % generation)
    neat.final()
    if save:
        neat.save_networks()
    else:
        neat.show_off()


    from perceptron.util import visualize_result
    visualize_result(results, "save_fig.png")
