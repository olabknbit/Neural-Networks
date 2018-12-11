def main(train_filename, test_filename, get_nn_filenames_f, n_networks=20, n_generations=10, use_cached=True):
    '''
    Train 20 networks for 10 generation, after every generation kill 12 networks, and mutate the rest.
    Optimize for average error.
    Mutate
    - number of hidden layers
    - width of hidden layer
    (for now - in future TODO mutate more params - like delete connections between certain neurons etc.)
    :return: NeuralNetwork - the best trained network
    '''

    activation='tanh'
    from util import initialize_random_network, get_split_dataset, write_network_to_file
    from neural_network_regression import score
    X_train, y_train, X_test, y_test = get_split_dataset(train_filename, test_filename)
    networks = [initialize_random_network(len(X_train[0]), n_hidden_layers=(1,2), n_neurons=(5,40), activation=activation) for _ in range(n_networks)]

    print("Initialized %d networks:" % n_networks)
    for n in networks:
        print(' ', n.get_params())

    print('Score networks')
    for n in networks:
        savefig_filename, save_nn_filename = get_nn_filenames_f(n.get_params())
        if use_cached:
            from util import read_network_from_file_f_name
            n = read_network_from_file_f_name(save_nn_filename, activation)
            n.test(X_test, y_test)
            print(n.get_params(), n.score)
        score(n, X_train, y_train, X_test, y_test, n_iter=1, savefig_filename=savefig_filename)
        write_network_to_file(save_nn_filename, n)

    # def cmp_nn(n1, n2):
    #     if n1.score is None or n2.score is None:
    #         return 0
    #     else:
    #         return int((n1.score - n2.score) * 100)
    # networks.sort(cmp=cmp_nn)
    # print("Network scores:")
    # for n in networks:
    #     print(n.score, n.get_params())
