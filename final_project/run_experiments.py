def generate_data(routes_lines, routes_stops, hours_n, day_length, hours_buses, trips_m, trips_transfers):
    import generate_hours, generate_trips, generate_train_test_data, util
    from perceptron import regression
    generate_hours.generate_hours(hours_buses, hours_n)
    generate_trips.generate_trips(routes_lines, routes_stops, trips_m, trips_transfers)
    generate_train_test_data.generate_test_train_data_without_filenames(routes_lines, routes_stops, trips_m,
                                                                        trips_transfers, hours_n, hours_buses)

    train_data_filename = util.get_train_data_filename(routes_lines, routes_stops, trips_m, trips_transfers, hours_n,
                                                       hours_buses)
    test_data_filename = util.get_test_data_filename(routes_lines, routes_stops, trips_m, trips_transfers, hours_n,
                                                     hours_buses)
    from perceptron.util import tanh, tanh_derivative
    accuracy = regression.main(train_data_filename, test_data_filename, activation_f=tanh,
                               activation_f_derivative=tanh_derivative,
                               create_nn=[3], save_nn=None, read_nn=None, number_of_epochs=10000, visualize_every=None,
                               l_rate=0.001, biases=True, savefig_filename=None)
    print('accuracy', accuracy)


def main():
    for routes_lines in [1]:
        for routes_stops in [6]:
            for hours_n in [2000]:
                for day_length in [50]:
                    for hours_buses in [[1]]:
                        for trips_m in [10]:
                            for trips_transfers in [0]:
                                generate_data(routes_lines, routes_stops, hours_n, day_length, hours_buses, trips_m,
                                              trips_transfers)


if __name__ == "__main__":
    main()
