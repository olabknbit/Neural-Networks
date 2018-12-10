def move_data(filename, d):
    m = int(d / 2)
    new_f = filename + '1'
    with open(filename, 'r') as in_f, open(new_f, 'w') as out_f:
        for line in in_f.readlines():
            prts = line.split(',')
            x, y = int(prts[0]), prts[1]
            line = str(x - m) + ',' + y
            out_f.write(line)
    return new_f


def generate_data(routes_lines, routes_stops, hours_n, hours_buses, trips_m, trips_transfers):
    import generate_hours, generate_trips, generate_train_test_data
    import util
    generate_hours.generate_hours(hours_buses, hours_n)
    generate_trips.generate_trips(routes_lines, routes_stops, trips_m, trips_transfers)
    generate_train_test_data.generate_test_train_data_without_filenames(routes_lines, routes_stops, trips_m,
                                                                        trips_transfers, hours_n, hours_buses)

    train_data_filename = util.get_train_data_filename(routes_lines, routes_stops, trips_m, trips_transfers, hours_n,
                                                       hours_buses)
    test_data_filename = util.get_test_data_filename(routes_lines, routes_stops, trips_m, trips_transfers, hours_n,
                                                     hours_buses)
    # TODO move scaling data somewhere else.
    from util import DAY_LENGTH
    train_data_filename = move_data(train_data_filename, DAY_LENGTH)
    test_data_filename = move_data(test_data_filename, DAY_LENGTH)

    return train_data_filename, test_data_filename


def get_nn_filenames(routes_lines, routes_stops, hours_n, hours_buses, trips_m, trips_transfers, nn):
    import util
    savefig_filename = util.get_savefig_filename(routes_lines, routes_stops, trips_m, trips_transfers, hours_n,
                                                 hours_buses, nn)
    save_nn_filename = util.get_save_nn_filename(routes_lines, routes_stops, trips_m, trips_transfers, hours_n,
                                                 hours_buses, nn)
    return savefig_filename, save_nn_filename


def main():
    number_of_epochs = 1500
    for routes_lines in [1]:
        for routes_stops in [6]:
            for hours_n in [1000]:  # how many rows of train and test data to generate (split 60:40)
                for hours_buses in [[1]]:  # how many buses of each line should run
                    for trips_m in [1000]:  # how many passenger's trips there are
                        for trips_transfers in [0]:  # max how many transfers each passenger can have
                            train_data_filename, test_data_filename = generate_data(
                                routes_lines, routes_stops, hours_n, hours_buses, trips_m, trips_transfers)
                            # [5], [10], [20],
                            for nn in [[40], [60], [80]]:
                                savefig_filename, save_nn_filename = get_nn_filenames(routes_lines, routes_stops,
                                                                                      hours_n, hours_buses, trips_m,
                                                                                      trips_transfers, nn)
                                from perceptron.util import tanh, tanh_derivative
                                activation_f, activation_f_derivative = tanh, tanh_derivative
                                from perceptron import regression
                                accuracy = regression.main(train_data_filename, test_data_filename,
                                                           activation_f=activation_f,
                                                           activation_f_derivative=activation_f_derivative,
                                                           visualize_every=None,
                                                           create_nn=nn, save_nn=save_nn_filename,
                                                           read_nn=None, number_of_epochs=number_of_epochs,
                                                           l_rate=0.001, biases=True, savefig_filename=savefig_filename)
                                print(nn, 'accuracy', accuracy)
                                print('_________________________________________')


if __name__ == "__main__":
    main()
